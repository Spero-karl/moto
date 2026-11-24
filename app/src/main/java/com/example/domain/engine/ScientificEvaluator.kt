package com.example.domain.engine

import com.example.data.local.LotoResultEntity
import com.example.domain.util.LotoDateUtils
import java.util.Locale
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

data class LabSavedModelInput(
    val id: String,
    val displayName: String,
    val savedStreak: Int,
    val slotSplitLabel: String,
    val genome: SymbolicGenome
)

class AdaptiveMetaLearner {

    private val expertMethods = PredictionMethod.entries.filter {
        it != PredictionMethod.ADAPTIVE_ML && it != PredictionMethod.AUTO_SEARCH
    }
    private val expertCount = expertMethods.size

    private val globalWeights = DoubleArray(expertCount) { 1.0 / expertCount }
    private val horaireWeights = mutableMapOf<String, DoubleArray>()
    private val horaireCounts = mutableMapOf<String, Int>()
    private val recentBrierSums = DoubleArray(expertCount) { 0.0 }
    private val recentCounts = IntArray(expertCount) { 0 }
    private val expertZeroStreaks = IntArray(expertCount) { 0 }

    fun predict(
        strictPast: List<LotoResultEntity>,
        context: DrawContext
    ): Triple<List<Int>, DoubleArray, List<Pair<PredictionMethod, DoubleArray>>> {
        return predictWithPrecomputed(strictPast, context, emptyMap())
    }

    fun predictWithPrecomputed(
        strictPast: List<LotoResultEntity>,
        context: DrawContext,
        precomputed: Map<PredictionMethod, Pair<List<Int>, DoubleArray>>
    ): Triple<List<Int>, DoubleArray, List<Pair<PredictionMethod, DoubleArray>>> {
        val expertProbs = expertMethods.map { method ->
            val probs = precomputed[method]?.second
                ?: StatisticalStrategies.predictIndividual(method, strictPast, context).second
            method to probs
        }

        val activeWeights = effectiveWeights(context.targetHoraire)
        val combinedProbs = DoubleArray(91) { 0.0 }
        for (k in expertProbs.indices) {
            val w = activeWeights[k]
            val p = expertProbs[k].second
            for (n in 1..90) {
                combinedProbs[n] += w * p[n]
            }
        }

        // Anti-repetition intra-jour douce + exploration sur serie nulle
        val (top5, finalProbs) = StatisticalStrategies.applyIntraDayDiversity(
            baseProbs = combinedProbs,
            context = context,
            diversityLambda = 0.35
        )

        return Triple(top5, finalProbs, expertProbs)
    }

    /**
     * Mise a jour bayesienne / Hedge par score propre (Brier somme sur 90 numeros)
     * STRICTEMENT APRES revelation de S(t), avec penalite de serie nulle.
     */
    fun updateAfterOutcome(
        expertProbs: List<Pair<PredictionMethod, DoubleArray>>,
        actualSet: Set<Int>,
        horaire: String
    ) {
        val brierSums = DoubleArray(expertProbs.size)
        val droughtMultipliers = DoubleArray(expertProbs.size) { 1.0 }
        for (k in expertProbs.indices) {
            val probs = expertProbs[k].second
            val brier = ScientificEvaluator.computeBrierScore(probs, actualSet)
            // Echelle sum(90 * brier) ~ 4.7 au lieu de ~0.052 afin que les gradients Hedge adaptent vraiment les poids
            brierSums[k] = brier * 90.0
            recentBrierSums[k] += brier
            recentCounts[k] += 1

            val top5 = (1..90)
                .sortedWith(compareByDescending<Int> { probs[it] }.thenBy { it })
                .take(5)
            val hit = top5.any { it in actualSet }
            if (hit) {
                expertZeroStreaks[k] = 0
            } else {
                expertZeroStreaks[k]++
                if (expertZeroStreaks[k] >= 4) {
                    droughtMultipliers[k] = 0.88
                }
            }
        }

        applyHedgeStep(globalWeights, brierSums, droughtMultipliers, eta = 0.45, shrinkage = 0.015)

        val hWeights = horaireWeights.getOrPut(horaire) { globalWeights.copyOf() }
        applyHedgeStep(hWeights, brierSums, droughtMultipliers, eta = 0.40, shrinkage = 0.025)
        horaireCounts[horaire] = (horaireCounts[horaire] ?: 0) + 1
    }

    fun snapshotWeights(horaire: String? = null, initialWeights: DoubleArray? = null): List<ExpertWeightInfo> {
        val w = if (horaire != null) effectiveWeights(horaire) else globalWeights
        return expertMethods.mapIndexed { idx, method ->
            val avgBrier = if (recentCounts[idx] > 0) {
                recentBrierSums[idx] / recentCounts[idx]
            } else {
                BacktestResult.RANDOM_BRIER_SCORE
            }
            val base = initialWeights?.getOrNull(idx) ?: (1.0 / expertCount)
            ExpertWeightInfo(
                method = method,
                name = method.shortLabel,
                weightPercent = (w[idx] * 100.0).toFloat(),
                recentBrier = avgBrier,
                weightDelta = ((w[idx] - base) * 100.0).toFloat()
            )
        }
    }

    private fun effectiveWeights(horaire: String): DoubleArray {
        val hCount = horaireCounts[horaire] ?: 0
        val hWeights = horaireWeights[horaire] ?: return globalWeights
        if (hCount < 10) return globalWeights
        val alpha = (hCount / (hCount + 25.0)).coerceIn(0.0, 0.7)
        return DoubleArray(expertCount) { k ->
            (1.0 - alpha) * globalWeights[k] + alpha * hWeights[k]
        }
    }

    private fun applyHedgeStep(
        weights: DoubleArray,
        brierSums: DoubleArray,
        droughtMultipliers: DoubleArray,
        eta: Double,
        shrinkage: Double
    ) {
        val minBrier = brierSums.minOrNull() ?: 0.0
        var sum = 0.0
        val updated = DoubleArray(weights.size)
        for (k in weights.indices) {
            val loss = (brierSums[k] - minBrier).coerceAtLeast(0.0)
            updated[k] = weights[k] * exp(-eta * loss) * droughtMultipliers[k]
            sum += updated[k]
        }
        val uniform = 1.0 / weights.size
        if (sum > 0.0) {
            for (k in weights.indices) {
                weights[k] = (1.0 - shrinkage) * (updated[k] / sum) + shrinkage * uniform
            }
        }
    }
}

object ScientificEvaluator {

    private const val MONTE_CARLO_SEED = 42L
    private const val MONTE_CARLO_SIMS = 10_000
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

    /**
     * Calibration Error (ECE) sur 10 intervalles de probabilites marginales.
     */
    fun computeCalibrationEce(probHistory: List<Pair<DoubleArray, Set<Int>>>): Double {
        if (probHistory.isEmpty()) return 0.0
        val binCount = 10
        val binPredSum = DoubleArray(binCount)
        val binTrueSum = DoubleArray(binCount)
        val binCounts = IntArray(binCount)

        for ((probs, actual) in probHistory) {
            for (n in 1..90) {
                val p = probs[n].coerceIn(0.0, 1.0)
                // La plupart des probabilites sont entre 0 et 0.25 -> echelle fine sur [0, 0.25]
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

    /**
     * Test z bilatere comparant la moyenne de correspondances observee a l'esperance theorique 25/90.
     */
    fun twoSidedPValueVsRandom(matches: List<Int>): Double {
        if (matches.isEmpty() || matches.sum() == 0) return 1.0
        val n = matches.size.toDouble()
        val mean = matches.sum() / n
        val expected = BacktestResult.RANDOM_EXPECTED_MATCHES.toDouble()
        // Variance exacte d'une loi hypergeometrique H(90, 5, 5) :
        // Var(X) = 5 * (5/90) * (85/90) * ((90-5)/(90-1)) = 0.25042
        val hyperVar = 5.0 * (5.0 / 90.0) * (85.0 / 90.0) * (85.0 / 89.0)
        val se = sqrt(hyperVar / n)
        val z = abs(mean - expected) / se.coerceAtLeast(EPS)
        // Approximation erf complementaire precise (Abramowitz & Stegun)
        return erfcNormal(z)
    }

    private fun erfcNormal(z: Double): Double {
        val x = z / sqrt(2.0)
        val t = 1.0 / (1.0 + 0.3275911 * x)
        val poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
        val erfc = poly * exp(-x * x)
        return erfc.coerceIn(0.0, 1.0)
    }

    fun runMonteCarloBaseline(
        drawsSubset: List<LotoResultEntity>,
        simulations: Int = MONTE_CARLO_SIMS,
        seed: Long = MONTE_CARLO_SEED
    ): MonteCarloBaseline {
        if (drawsSubset.isEmpty()) {
            return MonteCarloBaseline(0, seed, 0, 0, 0f, 0f, 0f, 0f, 0f, 0, 0, 0f, 0, 0)
        }

        val rng = Random(seed)
        val nDraws = drawsSubset.size
        val targetMasks = Array(nDraws) { d ->
            BooleanArray(91).also { mask ->
                for (num in StatisticalStrategies.extractDistinctBalls(drawsSubset[d])) {
                    if (num in 1..90) mask[num] = true
                }
            }
        }
        val dayIndices = HashMap<String, Int>()
        val drawDayIdx = IntArray(nDraws) { idx ->
            val key = LotoDateUtils.formatDayKey(drawsSubset[idx].timestamp)
            dayIndices.getOrPut(key) { dayIndices.size }
        }
        val daysCount = dayIndices.size
        val dayLastDrawIdx = IntArray(daysCount) { -1 }
        val dayPenultimateDrawIdx = IntArray(daysCount) { -1 }
        for (idx in 0 until nDraws) {
            val dIdx = drawDayIdx[idx]
            dayPenultimateDrawIdx[dIdx] = dayLastDrawIdx[dIdx]
            dayLastDrawIdx[dIdx] = idx
        }

        val avgList = FloatArray(simulations)
        val onePlusList = FloatArray(simulations)
        val twoPlusList = IntArray(simulations)
        val qualDaysList = IntArray(simulations)

        val pool = IntArray(90) { it + 1 }
        val dayTwoPlusCounts = IntArray(daysCount)
        val hitCounts = IntArray(nDraws)

        for (s in 0 until simulations) {
            dayTwoPlusCounts.fill(0)
            var totalHits = 0
            var onePlus = 0
            var twoPlus = 0

            for (d in 0 until nDraws) {
                var hits = 0
                val tMask = targetMasks[d]
                for (k in 0 until 5) {
                    val swapIdx = k + rng.nextInt(90 - k)
                    val picked = pool[swapIdx]
                    pool[swapIdx] = pool[k]
                    pool[k] = picked
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
                val penIdx = dayPenultimateDrawIdx[dayIdx]
                val penOk = penIdx >= 0 && hitCounts[penIdx] >= 2
                if (dayTwoPlusCounts[dayIdx] >= 3 && penOk) qDays++
            }

            avgList[s] = totalHits.toFloat() / nDraws
            onePlusList[s] = onePlus.toFloat() / nDraws
            twoPlusList[s] = twoPlus
            qualDaysList[s] = qDays
        }

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

    /**
     * Backtest Walk-Forward strict sur une plage d'indices [startIndex, endIndexExclusive)
     * dans la liste chronologique complete.
     */
    fun evaluateStrategyOnRange(
        chronological: List<LotoResultEntity>,
        method: PredictionMethod,
        startIndex: Int,
        endIndexExclusive: Int,
        periodLabel: String,
        preTrainedLearner: AdaptiveMetaLearner? = null,
        maxSearchRounds: Int = 4,
        useSliding6DrawBlocks: Boolean = (periodLabel.contains("Jour") && !periodLabel.contains("Tout")) ||
            periodLabel.startsWith("1 Jour")
    ): BacktestResult {
        if (method == PredictionMethod.AUTO_SEARCH) {
            return AutoSearchEngine.evaluateAutoSearch(
                chronological = chronological,
                startIndex = startIndex,
                endIndexExclusive = endIndexExclusive,
                periodLabel = periodLabel,
                maxRounds = maxSearchRounds,
                useSliding6DrawBlocks = useSliding6DrawBlocks
            )
        }

        val safeStart = startIndex.coerceAtLeast(1)
        val safeEnd = endIndexExclusive.coerceAtMost(chronological.size)
        if (safeStart >= safeEnd) {
            return emptyBacktestResult(method, periodLabel)
        }

        val learner = preTrainedLearner ?: AdaptiveMetaLearner()
        val expertCount = PredictionMethod.entries.count {
            it != PredictionMethod.ADAPTIVE_ML && it != PredictionMethod.AUTO_SEARCH
        }
        val initialWeights = if (method == PredictionMethod.ADAPTIVE_ML) {
            DoubleArray(expertCount) { 1.0 / expertCount }
        } else {
            null
        }

        var exactMatches = 0
        var partialMatches = 0
        var twoPlusMatches = 0
        var threePlusMatches = 0
        var fourPlusMatches = 0
        var brierSum = 0.0
        var logLossSum = 0.0
        val distribution = IntArray(6)
        val matchList = ArrayList<Int>(safeEnd - safeStart)
        val stepResults = ArrayList<BacktestStepResult>(safeEnd - safeStart)
        val calibrationHistory = ArrayList<Pair<DoubleArray, Set<Int>>>(safeEnd - safeStart)
        val weightSnapshots = mutableListOf<List<ExpertWeightInfo>>()

        var activeDayKey = ""
        val sameDayPredCounts = HashMap<Int, Int>()
        var currentZeroStreak = 0
        var maxZeroStreak = 0
        var strictEnd = 0

        for (t in safeStart until safeEnd) {
            val targetDraw = chronological[t]
            val actualNumbers = StatisticalStrategies.extractDistinctBalls(targetDraw).sorted()
            if (actualNumbers.size < 5) continue

            while (strictEnd < t && chronological[strictEnd].timestamp < targetDraw.timestamp) {
                strictEnd++
            }
            if (strictEnd <= 0) continue

            val strictPast = chronological.subList(0, strictEnd)
            val maxPastTs = strictPast[strictEnd - 1].timestamp
            check(maxPastTs < targetDraw.timestamp) {
                "Violation causale : maxPastTimestamp=$maxPastTs >= target=${targetDraw.timestamp}"
            }

            val dayKey = LotoDateUtils.formatDayKey(targetDraw.timestamp)
            if (dayKey != activeDayKey) {
                activeDayKey = dayKey
                sameDayPredCounts.clear()
            }

            val horaire = LotoDateUtils.normalizeHoraire(targetDraw)
            val game = LotoDateUtils.normalizeGame(targetDraw)
            val blindContext = DrawContext(
                targetTimestamp = targetDraw.timestamp,
                targetHoraire = horaire,
                targetGame = game,
                targetDayOfWeek = LotoDateUtils.dayOfWeek(targetDraw.timestamp),
                sameDayPredictedCounts = sameDayPredCounts.toMap(),
                currentZeroMatchStreak = currentZeroStreak
            )

            var expertProbsForUpdate: List<Pair<PredictionMethod, DoubleArray>>? = null
            val (predicted, marginalProbs) = if (method == PredictionMethod.ADAPTIVE_ML) {
                val (top5, probs, experts) = learner.predict(strictPast, blindContext)
                expertProbsForUpdate = experts
                top5 to probs
            } else {
                StatisticalStrategies.predictIndividual(method, strictPast, blindContext)
            }

            check(predicted.size == 5 && predicted.distinct().size == 5 && predicted.all { it in 1..90 }) {
                "Prediction invalide: $predicted"
            }

            val actualSet = actualNumbers.toSet()
            val matched = predicted.intersect(actualSet).sorted()
            val matchCount = matched.size.coerceIn(0, 5)
            val stepBrier = computeBrierScore(marginalProbs, actualSet)
            val stepLogLoss = computeBernoulliLogLoss(marginalProbs, actualSet)
            val expEstimate = predicted.sumOf { marginalProbs[it] }.toFloat()

            if (matchCount == 5) exactMatches++
            if (matchCount >= 1) partialMatches++
            if (matchCount >= 2) twoPlusMatches++
            if (matchCount >= 3) threePlusMatches++
            if (matchCount >= 4) fourPlusMatches++

            if (matchCount == 0) {
                currentZeroStreak++
                if (currentZeroStreak > maxZeroStreak) maxZeroStreak = currentZeroStreak
            } else {
                currentZeroStreak = 0
            }

            for (n in predicted) {
                sameDayPredCounts[n] = (sameDayPredCounts[n] ?: 0) + 1
            }

            distribution[matchCount]++
            matchList.add(matchCount)
            brierSum += stepBrier
            logLossSum += stepLogLoss
            calibrationHistory.add(marginalProbs to actualSet)

            val formattedDate = LotoDateUtils.formatFrenchDayAndDate(targetDraw.timestamp)
            val label = "${targetDraw.game} ${targetDraw.time}".trim()

            stepResults.add(
                BacktestStepResult(
                    stepIndex = stepResults.size + 1,
                    drawId = targetDraw.uniqueId,
                    drawLabel = label,
                    drawDate = formattedDate,
                    dayKey = dayKey,
                    horaire = horaire,
                    game = game,
                    trainSampleCount = strictPast.size,
                    maxTrainTimestamp = maxPastTs,
                    targetTimestamp = targetDraw.timestamp,
                    causalVerified = maxPastTs < targetDraw.timestamp,
                    predictedNumbers = predicted,
                    actualNumbers = actualNumbers,
                    matchedNumbers = matched,
                    matchCount = matchCount,
                    expectedMatchesEstimate = expEstimate,
                    stepBrierScore = stepBrier,
                    stepBernoulliLogLoss = stepLogLoss
                )
            )

            // Mise a jour adaptative STRICTEMENT APRES revelation du tirage
            if (method == PredictionMethod.ADAPTIVE_ML && expertProbsForUpdate != null) {
                learner.updateAfterOutcome(expertProbsForUpdate, actualSet, horaire)
                if (stepResults.size % 20 == 0 || t == safeEnd - 1) {
                    weightSnapshots.add(learner.snapshotWeights(initialWeights = initialWeights))
                }
            }
        }

        val totalTested = stepResults.size
        if (totalTested == 0) return emptyBacktestResult(method, periodLabel)

        val dailySummaries = buildDailySummaries(stepResults, useSliding6DrawBlocks)
        val maxConsecDays = if (useSliding6DrawBlocks) {
            var consec = 0
            var maxC = 0
            var lastIdx = -2
            dailySummaries.reversed().forEachIndexed { pos, d ->
                if (d.isQualifyingDay) {
                    consec = if (pos == lastIdx + 1) consec + 1 else 1
                    if (consec > maxC) maxC = consec
                    lastIdx = pos
                } else consec = 0
            }
            maxC
        } else {
            computeMaxConsecutiveQualifyingDays(dailySummaries)
        }
        val horaireBreakdown = buildHoraireBreakdown(stepResults)

        val (avgLo, avgHi) = mean95Interval(matchList)
        val (rateLo, rateHi) = wilson95Interval(partialMatches, totalTested)

        return BacktestResult(
            method = method,
            successRate = partialMatches.toFloat() / totalTested,
            totalTested = totalTested,
            exactMatches = exactMatches,
            partialMatches = partialMatches,
            drawsWithTwoPlusMatches = twoPlusMatches,
            drawsWithThreePlusMatches = threePlusMatches,
            drawsWithFourPlusMatches = fourPlusMatches,
            qualifyingDaysCount = dailySummaries.count { it.isQualifyingDay },
            totalDaysTested = dailySummaries.size,
            averageMatches = matchList.sum().toFloat() / totalTested,
            avgMatchesCiLow = avgLo,
            avgMatchesCiHigh = avgHi,
            rateOnePlusCiLow = rateLo,
            rateOnePlusCiHigh = rateHi,
            averageBrierScore = brierSum / totalTested,
            averageBernoulliLogLoss = logLossSum / totalTested,
            calibrationErrorEce = computeCalibrationEce(calibrationHistory),
            matchDistribution = distribution.toList(),
            allStepsCausalVerified = stepResults.all { it.causalVerified },
            periodLabel = periodLabel,
            learnedExpertWeights = if (method == PredictionMethod.ADAPTIVE_ML) {
                learner.snapshotWeights(initialWeights = initialWeights)
            } else {
                emptyList()
            },
            weightSnapshots = weightSnapshots,
            dailySummaries = dailySummaries,
            horaireBreakdown = horaireBreakdown,
            steps = stepResults.reversed(),
            maxConsecutiveQualifyingDays = maxConsecDays,
            maxZeroMatchStreak = maxZeroStreak,
            currentZeroMatchStreak = currentZeroStreak
        )
    }

    internal fun buildDailySummaries(
        stepResults: List<BacktestStepResult>,
        useSliding6DrawBlocks: Boolean = false
    ): List<DailyBacktestSummary> {
        if (stepResults.isEmpty()) return emptyList()
        if (useSliding6DrawBlocks) {
            val orderedSteps = stepResults.sortedBy { it.targetTimestamp }
            val total = orderedSteps.size
            val totalBlocks = (total + 5) / 6
            return orderedSteps.withIndex().groupBy { (idx, _) ->
                val blockFromEnd = (total - 1 - idx) / 6
                totalBlocks - 1 - blockFromEnd
            }.entries.sortedByDescending { it.key }.map { (bIdx, indexedSteps) ->
                val list = indexedSteps.map { it.value }
                val twoPlusInDay = list.count { it.matchCount >= 2 }
                val penHit = if (list.size >= 2) list[list.size - 2].matchCount else 0
                val penOk = penHit >= 2
                val firstDay = list.first().dayKey
                val lastDay = list.last().dayKey
                val label = if (firstDay == lastDay) {
                    "${list.first().drawDate} (${list.size} tir.)"
                } else {
                    "$firstDay → $lastDay (${list.size} tir.)"
                }
                DailyBacktestSummary(
                    dayKey = "B${bIdx + 1}_$lastDay",
                    dayLabel = label,
                    drawsTested = list.size,
                    drawsWithTwoPlus = twoPlusInDay,
                    bestMatchCount = list.maxOfOrNull { it.matchCount } ?: 0,
                    isQualifyingDay = twoPlusInDay == list.size && list.isNotEmpty() && penOk,
                    penultimateHitCount = penHit,
                    penultimateTwoPlus = penOk
                )
            }
        }
        return stepResults
            .groupBy { it.dayKey }
            .entries
            .sortedByDescending { it.key }
            .map { (dayKey, daySteps) ->
                val ordered = daySteps.sortedBy { it.targetTimestamp }
                val twoPlusInDay = ordered.count { it.matchCount >= 2 }
                val penHit = if (ordered.size >= 2) ordered[ordered.size - 2].matchCount else 0
                val penOk = penHit >= 2
                val bestInDay = ordered.maxOfOrNull { it.matchCount } ?: 0
                DailyBacktestSummary(
                    dayKey = dayKey,
                    dayLabel = ordered.first().drawDate,
                    drawsTested = ordered.size,
                    drawsWithTwoPlus = twoPlusInDay,
                    bestMatchCount = bestInDay,
                    isQualifyingDay = twoPlusInDay == ordered.size && ordered.isNotEmpty() && penOk,
                    penultimateHitCount = penHit,
                    penultimateTwoPlus = penOk
                )
            }
    }

    internal fun computeMaxConsecutiveQualifyingDays(dailySummaries: List<DailyBacktestSummary>): Int {
        if (dailySummaries.isEmpty()) return 0
        val chronologicalDays = dailySummaries.sortedBy { it.dayKey }
        var currentStreak = 0
        var maxStreak = 0
        var lastQualDayKey = ""
        for (day in chronologicalDays) {
            if (day.isQualifyingDay) {
                currentStreak = if (
                    lastQualDayKey.isNotEmpty() &&
                    LotoDateUtils.areConsecutiveCalendarDays(lastQualDayKey, day.dayKey)
                ) {
                    currentStreak + 1
                } else {
                    1
                }
                if (currentStreak > maxStreak) maxStreak = currentStreak
                lastQualDayKey = day.dayKey
            } else {
                currentStreak = 0
            }
        }
        return maxStreak
    }

    internal fun buildHoraireBreakdown(stepResults: List<BacktestStepResult>): List<HorairePerformance> {
        return stepResults
            .groupBy { it.horaire }
            .entries
            .sortedBy { it.key }
            .map { (h, hSteps) ->
                HorairePerformance(
                    horaire = h,
                    testedCount = hSteps.size,
                    averageMatches = hSteps.sumOf { it.matchCount }.toFloat() / hSteps.size,
                    onePlusCount = hSteps.count { it.matchCount >= 1 },
                    twoPlusCount = hSteps.count { it.matchCount >= 2 }
                )
            }
    }

    /**
     * Construit le decoupage chronologique rigoureux :
     * - Si les donnees couvrent Nov 2026+ et Oct 2026 : Oct 18-31 reste dans Validation (jamais Test Final!).
     * - Sinon, decoupage chronologique 20% warm-up / 35% Dev / 25% Validation / 20% Test Final Independant.
     */
    fun computeChronologicalSplit(chronological: List<LotoResultEntity>): Triple<IntRange, IntRange, IntRange> {
        val n = chronological.size
        if (n < 20) {
            val w = (n / 3).coerceAtLeast(1)
            val dEnd = (w + (n - w) / 2).coerceAtMost(n - 1)
            return Triple(w until dEnd, dEnd until n, n until n)
        }

        val sepStart = chronological.indexOfFirst { LotoDateUtils.formatDayKey(it.timestamp) >= "2026-09-01" }
        val novStart = chronological.indexOfFirst { LotoDateUtils.formatDayKey(it.timestamp) >= "2026-11-01" }
        val octStart = chronological.indexOfFirst { LotoDateUtils.formatDayKey(it.timestamp) >= "2026-10-01" }

        if (sepStart >= 20 && octStart > sepStart && novStart > octStart && (n - novStart) >= 20) {
            // Dev: jusqu'a fin septembre (ou Warmup->Oct), Validation: Octobre (incl. 18-31 oct), Test Final: Nov+
            val warmup = maxOf(20, sepStart)
            return Triple(warmup until octStart, octStart until novStart, novStart until n)
        }

        val warmup = maxOf(20, (n * 0.25).toInt()).coerceAtMost(n - 9)
        val remaining = n - warmup
        val devCount = maxOf(3, (remaining * 0.45).toInt())
        val valCount = maxOf(3, (remaining * 0.30).toInt())
        val devEnd = (warmup + devCount).coerceAtMost(n - 4)
        val valEnd = (devEnd + valCount).coerceAtMost(n - 2)
        return Triple(warmup until devEnd, devEnd until valEnd, valEnd until n)
    }

    fun computePeriodSplit(
        chronological: List<LotoResultEntity>,
        period: BacktestPeriod = BacktestPeriod.THREE_WEEKS,
        referenceNowMs: Long = System.currentTimeMillis(),
        customDays: Int? = period.days
    ): Triple<IntRange, IntRange, IntRange> {
        val range = AutoSearchEngine.computePeriodRange(
            chronological = chronological,
            period = period,
            referenceNowMs = referenceNowMs,
            customDays = customDays
        )
        val startIdx = range.first
        val endExclusive = (range.last + 1).coerceAtMost(chronological.size)
        val span = endExclusive - startIdx
        if (span < 3) {
            return computeChronologicalSplit(chronological)
        }
        if (span < 6) {
            val devEnd = (startIdx + maxOf(1, span / 2)).coerceAtMost(endExclusive - 2)
            val valEnd = (devEnd + 1).coerceAtMost(endExclusive - 1)
            return Triple(startIdx until devEnd, devEnd until valEnd, valEnd until endExclusive)
        }
        val devCount = maxOf(2, (span * 45) / 100)
        val valCount = maxOf(2, (span * 30) / 100)
        val devEnd = (startIdx + devCount).coerceAtMost((endExclusive - 2).coerceAtLeast(startIdx + 1))
        val valEnd = (devEnd + valCount).coerceAtMost((endExclusive - 1).coerceAtLeast(devEnd + 1))
        return Triple(startIdx until devEnd, devEnd until valEnd, valEnd until endExclusive)
    }

    fun computeThreeWeekSplit(
        chronological: List<LotoResultEntity>,
        referenceNowMs: Long = System.currentTimeMillis()
    ): Triple<IntRange, IntRange, IntRange> =
        computePeriodSplit(chronological, BacktestPeriod.THREE_WEEKS, referenceNowMs, 21)

    private data class MethodTripleResult(
        val method: PredictionMethod,
        val rDev: BacktestResult,
        val rVal: BacktestResult,
        val rTest: BacktestResult
    )

    fun runFullScientificComparison(
        data: List<LotoResultEntity>,
        period: BacktestPeriod = BacktestPeriod.THREE_WEEKS,
        referenceNowMs: Long = System.currentTimeMillis(),
        customDays: Int? = period.days,
        savedModels: List<LabSavedModelInput> = emptyList()
    ): ScientificReport {
        val rawChronological = data
            .distinctBy { it.uniqueId }
            .sortedWith(compareBy<LotoResultEntity> { it.timestamp }.thenBy { it.uniqueId })

        val resolvedDays = if (period == BacktestPeriod.ALL_HISTORY && customDays == null) null else (customDays ?: period.days)?.coerceAtLeast(1)
        val periodLabel = when {
            resolvedDays == null -> period.label
            period.days == resolvedDays -> period.label
            else -> "$resolvedDays Jours (Perso)"
        }
        val useSliding = resolvedDays != null

        val chronological = if (
            resolvedDays != null &&
            rawChronological.any { it.timestamp <= referenceNowMs }
        ) {
            rawChronological.filter { it.timestamp <= referenceNowMs }
        } else {
            rawChronological
        }

        val (devRange, valRange, testRange) = if (resolvedDays != null) {
            computePeriodSplit(chronological, period, referenceNowMs, resolvedDays)
        } else {
            computeChronologicalSplit(chronological)
        }

        val devDraws = if (!devRange.isEmpty()) chronological.subList(devRange.first, devRange.last + 1) else emptyList()
        val valDraws = if (!valRange.isEmpty()) chronological.subList(valRange.first, valRange.last + 1) else emptyList()
        val testDraws = if (!testRange.isEmpty()) chronological.subList(testRange.first, testRange.last + 1) else emptyList()

        val activeWindowRange = if (resolvedDays != null) {
            AutoSearchEngine.computePeriodRange(chronological, period, referenceNowMs, resolvedDays)
        } else {
            devRange.first..testRange.last
        }
        val useActiveWindowForAudit = testDraws.size < 6 && !activeWindowRange.isEmpty()
        val auditRange = if (useActiveWindowForAudit) activeWindowRange else testRange
        val auditDraws = if (!auditRange.isEmpty()) chronological.subList(auditRange.first, auditRange.last + 1) else testDraws
        val auditWindowName = if (useActiveWindowForAudit) {
            "Fenêtre active $periodLabel (${auditDraws.size} tir.)"
        } else {
            "Test Indépendant (${testDraws.size} tir.)"
        }

        val mcDev = runMonteCarloBaseline(devDraws, simulations = 10_000, seed = 42L)
        val mcVal = runMonteCarloBaseline(valDraws, simulations = 10_000, seed = 43L)
        val mcTest = runMonteCarloBaseline(testDraws, simulations = 10_000, seed = 44L)
        val mcAudit = if (useActiveWindowForAudit) {
            runMonteCarloBaseline(auditDraws, simulations = 10_000, seed = 45L)
        } else {
            mcTest
        }

        val allMethods = PredictionMethod.entries
        val bonferroniThreshold = 0.05 / allMethods.size

        val parallelResults = DiscoveryEngine.mapIndexedParallel(allMethods.size) { idx ->
            val method = allMethods[idx]
            val sharedLearner = if (method == PredictionMethod.ADAPTIVE_ML) AdaptiveMetaLearner() else null
            val rDev = evaluateStrategyOnRange(chronological, method, devRange.first, devRange.last + 1, "Développement", sharedLearner, useSliding6DrawBlocks = useSliding)
            val rVal = evaluateStrategyOnRange(chronological, method, valRange.first, valRange.last + 1, "Validation", sharedLearner, useSliding6DrawBlocks = useSliding)
            val rTest = evaluateStrategyOnRange(chronological, method, testRange.first, testRange.last + 1, "Test Indépendant", sharedLearner, useSliding6DrawBlocks = useSliding)
            MethodTripleResult(method, rDev, rVal, rTest)
        }

        val valResults = parallelResults.associate { it.method to it.rVal }
        val testResults = parallelResults.associate { it.method to it.rTest }

        val devEntries = parallelResults.map { toComparisonEntry(it.rDev, "Développement", bonferroniThreshold) }
        val valEntries = parallelResults.map { toComparisonEntry(it.rVal, "Validation", bonferroniThreshold) }
        val testEntries = parallelResults.map { toComparisonEntry(it.rTest, "Test Indépendant", bonferroniThreshold) }

        val uniqueSaved = savedModels.distinctBy { it.id }
        val savedModelAudits = if (uniqueSaved.isNotEmpty()) {
            val auditLabel = if (useSliding) periodLabel else "Test Indépendant"
            DiscoveryEngine.mapIndexedParallel(uniqueSaved.size) { idx ->
                val saved = uniqueSaved[idx]
                val bt = if (!auditRange.isEmpty()) {
                    DiscoveryEngine.evaluateGenomeOnRange(
                        chronological = chronological,
                        genome = saved.genome,
                        startIndex = auditRange.first,
                        endIndexExclusive = auditRange.last + 1,
                        periodLabel = auditLabel
                    )
                } else {
                    emptyBacktestResult(PredictionMethod.AUTO_SEARCH, auditLabel)
                }
                val matches = bt.steps.map { it.matchCount }
                val pRaw = twoSidedPValueVsRandom(matches)
                val penOkDays = bt.dailySummaries.count { it.penultimateTwoPlus }
                val zeroCount = bt.matchDistribution.getOrElse(0) { 0 }
                val zeroPct = if (bt.totalTested > 0) ((zeroCount * 100.0) / bt.totalTested).toInt() else 100
                val beatsRandom = bt.totalTested > 0 &&
                    bt.averageMatches > maxOf(BacktestResult.RANDOM_EXPECTED_MATCHES, mcAudit.meanAvgMatches) &&
                    (pRaw < 0.05 || bt.drawsWithTwoPlusMatches >= 2)
                val objectiveValidated = bt.qualifyingDaysCount >= 1 && bt.allStepsCausalVerified
                val verdictReason = if (objectiveValidated) {
                    "OUI : ${bt.qualifyingDaysCount} jour(s) validé(s) sur ${bt.totalDaysTested} (≥3 tir. à ≥2 boules dont avant-dernier ≥2)"
                } else {
                    "NON : ${bt.qualifyingDaysCount}/${bt.totalDaysTested} j. validés (${bt.drawsWithTwoPlusMatches} tir. ≥2 • avant-dernier ≥2 : $penOkDays/${bt.totalDaysTested})"
                }
                SavedModelLabAudit(
                    id = saved.id,
                    displayName = saved.displayName,
                    formulaCompact = saved.genome.describeCompact(),
                    slotSplitLabel = saved.slotSplitLabel,
                    savedStreak = saved.savedStreak,
                    windowName = auditWindowName,
                    testedDraws = bt.totalTested,
                    twoPlusDraws = bt.drawsWithTwoPlusMatches,
                    qualifyingDaysCount = bt.qualifyingDaysCount,
                    penultimateOkDaysCount = penOkDays,
                    totalDays = bt.totalDaysTested,
                    averageMatches = bt.averageMatches,
                    zeroMatchRatePct = zeroPct,
                    pValueVsRandom = pRaw,
                    causalVerified = bt.allStepsCausalVerified,
                    beatsRandom = beatsRandom,
                    objectiveValidated = objectiveValidated,
                    verdictReason = verdictReason
                )
            }.sortedWith(
                compareByDescending<SavedModelLabAudit> { it.objectiveValidated }
                    .thenByDescending { it.qualifyingDaysCount }
                    .thenByDescending { it.penultimateOkDaysCount }
                    .thenByDescending { it.twoPlusDraws }
                    .thenByDescending { it.averageMatches }
            )
        } else {
            emptyList()
        }

        val recommended = selectParsimoniousWinner(valEntries)
        val anyObjectiveValidated = savedModelAudits.any { it.objectiveValidated } ||
            testEntries.any { it.objectiveValidated }
        val recReason = buildRecommendationReason(
            method = recommended,
            valRes = valResults[recommended],
            testRes = testResults[recommended],
            savedAudits = savedModelAudits,
            periodLabel = periodLabel,
            anyObjectiveValidated = anyObjectiveValidated
        )

        val flaggedTest = testEntries.map {
            it.copy(isRecommendedParsimonious = it.method == recommended)
        }.sortedWith(compareByDescending<StrategyComparisonEntry> { it.objectiveValidated }.thenByDescending { it.qualifyingDaysCount }.thenByDescending { it.averageMatches })

        val flaggedVal = valEntries.map {
            it.copy(isRecommendedParsimonious = it.method == recommended)
        }.sortedWith(compareByDescending<StrategyComparisonEntry> { it.objectiveValidated }.thenByDescending { it.qualifyingDaysCount }.thenByDescending { it.averageMatches })

        val flaggedDev = devEntries.map {
            it.copy(isRecommendedParsimonious = it.method == recommended)
        }.sortedWith(compareByDescending<StrategyComparisonEntry> { it.objectiveValidated }.thenByDescending { it.qualifyingDaysCount }.thenByDescending { it.averageMatches })

        val complementary = computeComplementarity(testResults.ifEmpty { valResults })

        val splitInfo = WindowSplitInfo(
            totalDrawsAvailable = chronological.size,
            warmupDrawsCount = devRange.first,
            devDateRange = formatRangeLabel(devDraws),
            devDrawsCount = devDraws.size,
            valDateRange = formatRangeLabel(valDraws),
            valDrawsCount = valDraws.size,
            indepTestDateRange = formatRangeLabel(testDraws),
            indepTestDrawsCount = testDraws.size
        )

        val auditNote = "Audit Cube90 : T(n)=1+((n³+9) mod 90) n'est pas bijective (gcd(3, φ(9))=3). " +
            "Elle réduit 90 numéros à 30 images (collisions 3-pour-1 : {n, n+30, n+60}) et exclut 60 numéros."

        return ScientificReport(
            splitInfo = splitInfo,
            monteCarloDev = mcDev,
            monteCarloVal = mcVal,
            monteCarloIndepTest = mcTest,
            devComparisons = flaggedDev,
            valComparisons = flaggedVal,
            indepTestComparisons = flaggedTest,
            recommendedMethod = recommended,
            recommendationReason = recReason,
            complementaryPairs = complementary,
            cube90AuditSummary = auditNote,
            periodLabel = periodLabel,
            savedModelAudits = savedModelAudits,
            anyObjectiveValidated = anyObjectiveValidated
        )
    }

    private fun selectParsimoniousWinner(
        valEntries: List<StrategyComparisonEntry>
    ): PredictionMethod {
        val candidates = valEntries.filter { it.testedCount > 0 }
        if (candidates.isEmpty()) return PredictionMethod.DIRICHLET_MULTINOMIAL

        val bestScore = candidates.maxOf { entry ->
            entry.averageMatches.toDouble() - 2.0 * entry.brierScore
        }
        val simpleCandidate = candidates
            .filter { !it.method.isComplex }
            .maxByOrNull { it.averageMatches.toDouble() - 2.0 * it.brierScore }

        val overallBest = candidates.maxBy { it.averageMatches.toDouble() - 2.0 * it.brierScore }

        if (simpleCandidate != null) {
            val simpleScore = simpleCandidate.averageMatches.toDouble() - 2.0 * simpleCandidate.brierScore
            if (bestScore - simpleScore <= 0.015) {
                return simpleCandidate.method
            }
        }
        return overallBest.method
    }

    private fun buildRecommendationReason(
        method: PredictionMethod,
        valRes: BacktestResult?,
        testRes: BacktestResult?,
        savedAudits: List<SavedModelLabAudit>,
        periodLabel: String,
        anyObjectiveValidated: Boolean
    ): String {
        val validSaved = savedAudits.filter { it.objectiveValidated }
        if (validSaved.isNotEmpty()) {
            val top = validSaved.first()
            return "OUI — Modèle IA conservé « ${top.displayName} » validé sur ${top.windowName} : " +
                "${top.qualifyingDaysCount}/${top.totalDays} j. validés (≥3 tir. à ≥2 dont avant-dernier ≥2), moy. ${String.format(Locale.FRANCE, "%.2f", top.averageMatches)}/tir."
        }
        val testAvg = testRes?.let { String.format(Locale.FRANCE, "%.3f", it.averageMatches) } ?: "N/A"
        val valAvg = valRes?.let { String.format(Locale.FRANCE, "%.3f", it.averageMatches) } ?: "N/A"
        return if (anyObjectiveValidated) {
            "OUI — ${method.displayName} valide l'objectif sur Test Indépendant (Validation=$valAvg, Test=$testAvg)."
        } else {
            "NON — Sur $periodLabel, ni les ${savedAudits.size} modèle(s) IA conservé(s) ni les 13 méthodes classiques " +
                "n'atteignent l'objectif (≥3 tir./jour à ≥2 dont avant-dernier ≥2). Meilleur repère classique : ${method.shortLabel} (Val=$valAvg, Test=$testAvg)."
        }
    }

    private fun computeComplementarity(resultsMap: Map<PredictionMethod, BacktestResult>): List<ComplementarityPair> {
        val methods = resultsMap.keys.filter { it != PredictionMethod.ADAPTIVE_ML && it != PredictionMethod.AUTO_SEARCH }
        val pairs = mutableListOf<ComplementarityPair>()

        for (i in methods.indices) {
            for (j in i + 1 until methods.size) {
                val mA = methods[i]
                val mB = methods[j]
                val stepsA = resultsMap[mA]?.steps.orEmpty().associateBy { it.drawId }
                val stepsB = resultsMap[mB]?.steps.orEmpty().associateBy { it.drawId }
                val commonIds = stepsA.keys.intersect(stepsB.keys)
                if (commonIds.size < 5) continue

                var jaccardSum = 0.0
                var compHits = 0
                val hitsA = ArrayList<Double>(commonIds.size)
                val hitsB = ArrayList<Double>(commonIds.size)

                for (id in commonIds) {
                    val sA = stepsA.getValue(id)
                    val sB = stepsB.getValue(id)
                    val setA = sA.predictedNumbers.toSet()
                    val setB = sB.predictedNumbers.toSet()
                    val unionSize = setA.union(setB).size.coerceAtLeast(1)
                    jaccardSum += setA.intersect(setB).size.toDouble() / unionSize

                    if ((sA.matchCount >= 2 && sB.matchCount == 0) || (sB.matchCount >= 2 && sA.matchCount == 0)) {
                        compHits++
                    }
                    hitsA.add(sA.matchCount.toDouble())
                    hitsB.add(sB.matchCount.toDouble())
                }

                val corr = pearsonCorrelation(hitsA, hitsB)
                pairs.add(
                    ComplementarityPair(
                        methodA = mA,
                        methodB = mB,
                        jaccardOverlap = (jaccardSum / commonIds.size).toFloat(),
                        errorCorrelation = corr.toFloat(),
                        complementaryHitsCount = compHits
                    )
                )
            }
        }

        // Trie par faible chevauchement + faible correlation + nombre de succes complementaires
        return pairs
            .sortedWith(compareByDescending<ComplementarityPair> { it.complementaryHitsCount }.thenBy { it.errorCorrelation })
            .take(6)
    }

    private fun pearsonCorrelation(x: List<Double>, y: List<Double>): Double {
        if (x.size != y.size || x.size < 2) return 0.0
        val meanX = x.average()
        val meanY = y.average()
        var num = 0.0
        var denX = 0.0
        var denY = 0.0
        for (i in x.indices) {
            val dx = x[i] - meanX
            val dy = y[i] - meanY
            num += dx * dy
            denX += dx * dx
            denY += dy * dy
        }
        val den = sqrt(denX * denY)
        return if (den > EPS) (num / den).coerceIn(-1.0, 1.0) else 0.0
    }

    private fun toComparisonEntry(
        res: BacktestResult,
        windowName: String,
        bonferroniThreshold: Double
    ): StrategyComparisonEntry {
        val matches = res.steps.map { it.matchCount }
        val pVal = twoSidedPValueVsRandom(matches)
        val penOkDays = res.dailySummaries.count { it.penultimateTwoPlus }
        return StrategyComparisonEntry(
            method = res.method,
            windowName = windowName,
            testedCount = res.totalTested,
            averageMatches = res.averageMatches,
            onePlusRate = res.successRate,
            twoPlusCount = res.drawsWithTwoPlusMatches,
            threePlusCount = res.drawsWithThreePlusMatches,
            qualifyingDaysCount = res.qualifyingDaysCount,
            totalDays = res.totalDaysTested,
            brierScore = res.averageBrierScore,
            bernoulliLogLoss = res.averageBernoulliLogLoss,
            calibrationError = res.calibrationErrorEce,
            pValueVsRandom = pVal,
            adjustedSignificant = pVal < bonferroniThreshold && res.averageMatches > BacktestResult.RANDOM_EXPECTED_MATCHES,
            isRecommendedParsimonious = false,
            penultimateOkDaysCount = penOkDays,
            objectiveValidated = res.qualifyingDaysCount >= 1 && res.allStepsCausalVerified
        )
    }

    private fun formatRangeLabel(draws: List<LotoResultEntity>): String {
        if (draws.isEmpty()) return "Aucune donnée"
        return "${LotoDateUtils.formatShortDate(draws.first().timestamp)} → ${LotoDateUtils.formatShortDate(draws.last().timestamp)}"
    }

    internal fun emptyBacktestResult(method: PredictionMethod, periodLabel: String): BacktestResult {
        return BacktestResult(
            method = method,
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
            learnedExpertWeights = emptyList(),
            weightSnapshots = emptyList(),
            dailySummaries = emptyList(),
            horaireBreakdown = emptyList(),
            steps = emptyList()
        )
    }
}
