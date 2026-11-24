package com.lotobj.pc.engine

import com.lotobj.pc.model.BacktestResult
import com.lotobj.pc.model.LotoDraw
import com.lotobj.pc.model.MonteCarloBaseline
import com.lotobj.pc.util.LotoDateUtils
import java.util.concurrent.Callable
import java.util.concurrent.ForkJoinPool
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

object ScientificMath {

    private const val EPS = 1e-12

    fun computeBrierScore(marginalProbs: DoubleArray, actualSet: Set<Int>): Double {
        var sumSq = 0.0
        for (n in 1..90) {
            val y = if (n in actualSet) 1.0 else 0.0
            val diff = marginalProbs[n] - y
            sumSq += diff * diff
        }
        return sumSq / 90.0
    }

    fun computeBrierScoreMask(marginalProbs: DoubleArray, actualMask: BooleanArray): Double {
        var sumSq = 0.0
        for (n in 1..90) {
            val y = if (actualMask[n]) 1.0 else 0.0
            val diff = marginalProbs[n] - y
            sumSq += diff * diff
        }
        return sumSq / 90.0
    }

    fun computeBernoulliLogLoss(marginalProbs: DoubleArray, actualSet: Set<Int>): Double {
        var sumLoss = 0.0
        for (n in 1..90) {
            val p = marginalProbs[n].coerceIn(1e-6, 1.0 - 1e-6)
            val y = if (n in actualSet) 1.0 else 0.0
            sumLoss -= y * ln(p) + (1.0 - y) * ln(1.0 - p)
        }
        return sumLoss / 90.0
    }

    fun calibratedProbAtLeastOne(predicted: List<Int>, marginalProbs: DoubleArray): Float {
        var prodNone = 1.0
        for (n in predicted) {
            if (n in 1..90) {
                prodNone *= (1.0 - marginalProbs[n].coerceIn(0.0, 0.999))
            }
        }
        return (1.0 - prodNone).toFloat().coerceIn(0f, 1f)
    }

    fun computeCalibrationEce(probHistory: List<Pair<DoubleArray, Set<Int>>>): Double {
        if (probHistory.isEmpty()) return 0.0
        val binCount = 10
        val binPredSum = DoubleArray(binCount)
        val binTrueSum = DoubleArray(binCount)
        val binCounts = IntArray(binCount)

        for ((probs, actual) in probHistory) {
            for (n in 1..90) {
                val p = probs[n].coerceIn(0.0, 1.0)
                val bin = ((p / 0.25) * binCount).toInt().coerceIn(0, binCount - 1)
                binPredSum[bin] += p
                binTrueSum[bin] += if (n in actual) 1.0 else 0.0
                binCounts[bin]++
            }
        }

        val total = probHistory.size * 90.0
        var ece = 0.0
        for (b in 0 until binCount) {
            if (binCounts[b] > 0) {
                val avgP = binPredSum[b] / binCounts[b]
                val avgY = binTrueSum[b] / binCounts[b]
                ece += (binCounts[b] / total) * abs(avgP - avgY)
            }
        }
        return ece
    }

    fun wilson95Interval(successes: Int, total: Int): Pair<Float, Float> {
        if (total <= 0) return 0f to 0f
        val z = 1.959964
        val p = successes.toDouble() / total
        val denom = 1.0 + z * z / total
        val center = (p + z * z / (2.0 * total)) / denom
        val half = (z * sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)) / denom
        return (center - half).coerceIn(0.0, 1.0).toFloat() to
            (center + half).coerceIn(0.0, 1.0).toFloat()
    }

    fun mean95Interval(values: List<Int>): Pair<Float, Float> {
        if (values.isEmpty()) return 0f to 0f
        val n = values.size.toDouble()
        val mean = values.sum() / n
        if (values.size == 1) return mean.toFloat() to mean.toFloat()
        var sq = 0.0
        for (v in values) {
            val d = v - mean
            sq += d * d
        }
        val sd = sqrt(sq / (n - 1.0))
        val se = sd / sqrt(n)
        val half = 1.959964 * se
        return (mean - half).coerceAtLeast(0.0).toFloat() to (mean + half).toFloat()
    }

    fun twoSidedPValueVsRandom(matches: List<Int>): Double {
        if (matches.isEmpty() || matches.sum() == 0) return 1.0
        val n = matches.size.toDouble()
        val mean = matches.sum() / n
        val expected = BacktestResult.RANDOM_EXPECTED_MATCHES.toDouble()
        val hyperVar = 5.0 * (5.0 / 90.0) * (85.0 / 90.0) * (85.0 / 89.0)
        val se = sqrt(hyperVar / n)
        val z = abs(mean - expected) / se.coerceAtLeast(EPS)
        return erfcNormal(z)
    }

    private fun erfcNormal(z: Double): Double {
        val x = z / sqrt(2.0)
        val t = 1.0 / (1.0 + 0.3275911 * x)
        val poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        return (poly * exp(-x * x)).coerceIn(0.0, 1.0)
    }

    /**
     * Exécute les simulations Monte Carlo en parallèle sur tous les cœurs CPU via ForkJoinPool.
     */
    fun runMonteCarloParallel(
        drawsSubset: List<LotoDraw>,
        simulations: Int = 5_000,
        seed: Long = 4242L,
        pool: ForkJoinPool = ForkJoinPool.commonPool()
    ): MonteCarloBaseline {
        if (drawsSubset.isEmpty() || simulations <= 0) {
            return MonteCarloBaseline(0, seed, 0, 0, 0f, 0f, 0f, 0f, 0f, 0, 0, 0f, 0, 0)
        }

        val nDraws = drawsSubset.size
        val targetMasks = Array(nDraws) { idx ->
            val mask = BooleanArray(91)
            val d = drawsSubset[idx]
            if (d.num1 in 1..90) mask[d.num1] = true
            if (d.num2 in 1..90) mask[d.num2] = true
            if (d.num3 in 1..90) mask[d.num3] = true
            if (d.num4 in 1..90) mask[d.num4] = true
            if (d.num5 in 1..90) mask[d.num5] = true
            mask
        }
        val dayIndices = HashMap<String, Int>()
        val drawDayIdx = IntArray(nDraws) { idx ->
            val key = LotoDateUtils.formatDayKey(drawsSubset[idx].timestamp)
            dayIndices.getOrPut(key) { dayIndices.size }
        }
        val daysCount = dayIndices.size
        val dayPositions = Array(daysCount) { mutableListOf<Int>() }
        for (i in 0 until nDraws) {
            dayPositions[drawDayIdx[i]].add(i)
        }
        val dayPenultimateIdx = IntArray(daysCount) { dIdx ->
            val list = dayPositions[dIdx]
            if (list.size >= 2) list[list.size - 2] else -1
        }

        val avgList = FloatArray(simulations)
        val onePlusList = FloatArray(simulations)
        val twoPlusList = IntArray(simulations)
        val qualDaysList = IntArray(simulations)

        val workers = pool.parallelism.coerceIn(1, simulations)
        val chunkSize = (simulations + workers - 1) / workers

        val tasks = (0 until workers).map { workerIdx ->
            val startSim = workerIdx * chunkSize
            val endSim = minOf(simulations, startSim + chunkSize)
            Callable {
                if (startSim >= endSim) return@Callable
                val rng = Random(seed + workerIdx * 1_000_003L)
                val localPool = IntArray(90) { it + 1 }
                val dayTwoPlusCounts = IntArray(daysCount)
                val hitCounts = IntArray(nDraws)

                for (s in startSim until endSim) {
                    dayTwoPlusCounts.fill(0)
                    var totalHits = 0
                    var onePlus = 0
                    var twoPlus = 0

                    for (d in 0 until nDraws) {
                        var hits = 0
                        val tMask = targetMasks[d]
                        for (k in 0 until 5) {
                            val swapIdx = k + rng.nextInt(90 - k)
                            val picked = localPool[swapIdx]
                            localPool[swapIdx] = localPool[k]
                            localPool[k] = picked
                            if (tMask[picked]) hits++
                        }
                        hitCounts[d] = hits
                        totalHits += hits
                        if (hits >= 1) onePlus++
                        if (hits >= 2) {
                            twoPlus++
                            dayTwoPlusCounts[drawDayIdx[d]]++
                        }
                    }

                    var qDays = 0
                    for (dayIdx in 0 until daysCount) {
                        val penIdx = dayPenultimateIdx[dayIdx]
                        if (dayTwoPlusCounts[dayIdx] >= 3 && penIdx >= 0 && hitCounts[penIdx] >= 2) qDays++
                    }

                    avgList[s] = totalHits.toFloat() / nDraws
                    onePlusList[s] = onePlus.toFloat() / nDraws
                    twoPlusList[s] = twoPlus
                    qualDaysList[s] = qDays
                }
            }
        }

        pool.invokeAll(tasks).forEach { it.get() }

        avgList.sort()
        twoPlusList.sort()
        qualDaysList.sort()
        val lo = (0.025 * simulations).toInt().coerceIn(0, simulations - 1)
        val hi = (0.975 * simulations).toInt().coerceIn(0, simulations - 1)

        return MonteCarloBaseline(
            simulationsCount = simulations,
            seed = seed,
            drawsCount = nDraws,
            daysCount = daysCount,
            meanAvgMatches = avgList.average().toFloat(),
            avgMatchesCiLow = avgList[lo],
            avgMatchesCiHigh = avgList[hi],
            meanOnePlusRate = onePlusList.average().toFloat(),
            meanTwoPlusCount = twoPlusList.average().toFloat(),
            twoPlusCiLow = twoPlusList[lo],
            twoPlusCiHigh = twoPlusList[hi],
            meanQualifyingDays = qualDaysList.average().toFloat(),
            qualifyingDaysCiLow = qualDaysList[lo],
            qualifyingDaysCiHigh = qualDaysList[hi]
        )
    }

    fun emptyBacktestResult(periodLabel: String): BacktestResult {
        return BacktestResult(
            successRate = 0f,
            totalTested = 0,
            exactMatches = 0,
            partialMatches = 0,
            drawsWithTwoPlusMatches = 0,
            drawsWithThreePlusMatches = 0,
            drawsWithFourPlusMatches = 0,
            qualifyingDaysCount = 0,
            totalDaysTested = 0,
            averageMatches = 0f,
            avgMatchesCiLow = 0f,
            avgMatchesCiHigh = 0f,
            rateOnePlusCiLow = 0f,
            rateOnePlusCiHigh = 0f,
            averageBrierScore = BacktestResult.RANDOM_BRIER_SCORE,
            averageBernoulliLogLoss = BacktestResult.RANDOM_BERNOULLI_LOG_LOSS,
            calibrationErrorEce = 0.0,
            matchDistribution = listOf(0, 0, 0, 0, 0, 0),
            allStepsCausalVerified = false,
            periodLabel = periodLabel,
            dailySummaries = emptyList(),
            horaireBreakdown = emptyList(),
            steps = emptyList()
        )
    }
}
