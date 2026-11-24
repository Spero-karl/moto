package com.lotobj.pc.engine

import com.lotobj.pc.model.BacktestPeriod
import com.lotobj.pc.model.BacktestResult
import com.lotobj.pc.model.BacktestStepResult
import com.lotobj.pc.model.ComplementarityPair
import com.lotobj.pc.model.DailyBacktestSummary
import com.lotobj.pc.model.DrawContext
import com.lotobj.pc.model.ExpertWeightInfo
import com.lotobj.pc.model.HorairePerformance
import com.lotobj.pc.model.LotoDraw
import com.lotobj.pc.model.NextDrawTarget
import com.lotobj.pc.model.PredictedNumberConfidence
import com.lotobj.pc.model.PredictionMethod
import com.lotobj.pc.model.PredictionResult
import com.lotobj.pc.model.SavedModelLabAudit
import com.lotobj.pc.model.ScientificReport
import com.lotobj.pc.model.StrategyComparisonEntry
import com.lotobj.pc.model.SymbolicGenome
import com.lotobj.pc.model.WindowSplitInfo
import com.lotobj.pc.util.LotoDateUtils
import java.util.Locale
import java.util.concurrent.Callable
import java.util.concurrent.ForkJoinPool
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.random.Random

object StatisticalStrategies {

    private val KNIGHT_MOVES = arrayOf(
        -2 to -1, -2 to 1, -1 to -2, -1 to 2,
        1 to -2, 1 to 2, 2 to -1, 2 to 1
    )

    fun extractOrderedBalls(draw: LotoDraw): List<Int> {
        return listOf(draw.num1, draw.num2, draw.num3, draw.num4, draw.num5)
            .filter { it in 1..90 }
    }

    fun extractDistinctBalls(draw: LotoDraw): List<Int> {
        return extractOrderedBalls(draw).distinct()
    }

    fun normalizeToMarginalFive(rawScores: DoubleArray, priorStrengthDraws: Double = 25.0): DoubleArray {
        val cleaned = DoubleArray(91) { 0.0 }
        var sum = 0.0
        for (i in 1..90) {
            val v = rawScores[i]
            val safe = if (v.isFinite() && v > 0.0) v else 0.0
            cleaned[i] = safe
            sum += safe
        }

        val uniformMarginal = 5.0 / 90.0
        val probs = DoubleArray(91) { 0.0 }

        if (sum <= 0.0) {
            for (i in 1..90) probs[i] = uniformMarginal
            return probs
        }

        val priorTotalBalls = priorStrengthDraws * 5.0
        val scale = sum.coerceAtLeast(1.0)
        for (i in 1..90) {
            val empiricalShare = cleaned[i] / scale
            val p = (empiricalShare * scale + priorTotalBalls * (1.0 / 90.0)) / (scale + priorTotalBalls)
            probs[i] = 5.0 * p
        }

        clampSumToFive(probs)
        return probs
    }

    private fun clampSumToFive(probs: DoubleArray) {
        val fixed = BooleanArray(91)
        var remainingTarget = 5.0
        var activeCount = 90

        for (pass in 0 until 5) {
            var activeSum = 0.0
            for (i in 1..90) {
                if (!fixed[i]) activeSum += probs[i]
            }
            if (activeSum <= 0.0 || activeCount <= 0) break

            val factor = remainingTarget / activeSum
            var clampedAny = false
            for (i in 1..90) {
                if (!fixed[i]) {
                    val scaled = probs[i] * factor
                    if (scaled >= 0.999) {
                        probs[i] = 0.999
                        fixed[i] = true
                        remainingTarget -= 0.999
                        activeCount--
                        clampedAny = true
                    } else if (scaled <= 1e-4) {
                        probs[i] = 1e-4
                        fixed[i] = true
                        remainingTarget -= 1e-4
                        activeCount--
                        clampedAny = true
                    } else {
                        probs[i] = scaled
                    }
                }
            }
            if (!clampedAny) break
        }
    }

    fun predictIndividual(
        method: PredictionMethod,
        strictPast: List<LotoDraw>,
        context: DrawContext
    ): Pair<List<Int>, DoubleArray> {
        require(method != PredictionMethod.ADAPTIVE_ML && method != PredictionMethod.AUTO_SEARCH) {
            "Utiliser AdaptiveMetaLearner ou AutoSearchEngine pour $method"
        }

        if (strictPast.isEmpty()) {
            val uniform = DoubleArray(91) { if (it == 0) 0.0 else 5.0 / 90.0 }
            return listOf(1, 2, 3, 4, 5) to uniform
        }

        return when (method) {
            PredictionMethod.COLD_HOT_BALANCE -> predictColdHotVariant(strictPast, context, coldRatio = 0.5f, diversityLambda = 0.65)
            PredictionMethod.DIRICHLET_MULTINOMIAL -> predictDirichlet(strictPast, context)
            PredictionMethod.EMA_RECENCY -> predictEma(strictPast)
            PredictionMethod.MARKOV_STRATIFIED -> predictMarkovStratified(strictPast, context)
            PredictionMethod.COOCCURRENCE_GRAPH -> predictCooccurrenceGraph(strictPast)
            PredictionMethod.SPECTRAL_FOURIER -> predictSpectralFourier(strictPast)
            PredictionMethod.MATRIX_9X10 -> predictMatrix9x10(strictPast)
            PredictionMethod.CHANCE_CROIX_PYRAMID -> predictChanceCroixPyramid(strictPast)
            PredictionMethod.LUNAR_ASTRONOMICAL -> predictLunar(strictPast, context)
            PredictionMethod.CUBE90 -> predictCube90(strictPast)
            PredictionMethod.HORAIRE_OPTIM -> predictHoraireHungarian(strictPast, context)
            PredictionMethod.ADAPTIVE_ML, PredictionMethod.AUTO_SEARCH -> error("Unreachable")
        }
    }

    fun countColdNumbers(numbers: List<Int>, past: List<LotoDraw>): Int {
        if (past.isEmpty()) return 0
        val gaps = computeCurrentGaps(past)
        return numbers.count { it in 1..90 && gaps[it] >= 18 }
    }

    fun applyIntraDayDiversity(
        baseProbs: DoubleArray,
        context: DrawContext,
        diversityLambda: Double
    ): Pair<List<Int>, DoubleArray> {
        if (diversityLambda <= 0.0 || (context.sameDayPredictedCounts.isEmpty() && context.currentZeroMatchStreak < 3)) {
            return selectTop5(baseProbs) to baseProbs
        }
        val adjusted = DoubleArray(91)
        val streakBoost = if (context.currentZeroMatchStreak >= 3) 0.25 else 0.0
        for (n in 1..90) {
            val repCount = context.sameDayPredictedCounts[n] ?: 0
            val penalty = exp(-diversityLambda * repCount)
            val freshBonus = if (repCount == 0 && streakBoost > 0.0) (1.0 + streakBoost) else 1.0
            adjusted[n] = baseProbs[n] * penalty * freshBonus
        }
        val probs = normalizeToMarginalFive(adjusted, priorStrengthDraws = 20.0)
        return selectTop5(probs) to probs
    }

    fun computeCurrentGaps(past: List<LotoDraw>): IntArray {
        val lastSeen = IntArray(91) { -1 }
        for (i in past.indices) {
            for (n in extractDistinctBalls(past[i])) {
                lastSeen[n] = i
            }
        }
        val total = past.size
        return IntArray(91) { n ->
            if (n == 0) 0 else if (lastSeen[n] < 0) total else (total - 1 - lastSeen[n])
        }
    }

    fun predictColdHotVariant(
        past: List<LotoDraw>,
        context: DrawContext,
        coldRatio: Float = 0.5f,
        diversityLambda: Double = 0.65,
        emaDecay: Double = 0.94
    ): Pair<List<Int>, DoubleArray> {
        val total = past.size.coerceAtLeast(1)
        val gaps = computeCurrentGaps(past)
        val appearances = IntArray(91)
        val horaireCounts = DoubleArray(91)
        val emaHot = DoubleArray(91)

        var w = 1.0
        for (i in past.indices.reversed()) {
            val d = past[i]
            val sameH = LotoDateUtils.normalizeHoraire(d) == context.targetHoraire
            for (n in extractDistinctBalls(d)) {
                appearances[n]++
                if (i >= past.size - 140) emaHot[n] += w
                if (sameH) horaireCounts[n] += 1.0
            }
            if (i >= past.size - 140) w *= emaDecay
        }

        val lastDrawSet = extractDistinctBalls(past.last()).toSet()
        val coWithLast = DoubleArray(91)
        for (d in past.takeLast(120)) {
            val balls = extractDistinctBalls(d)
            if (balls.any { it in lastDrawSet }) {
                for (n in balls) coWithLast[n] += 1.0
            }
        }

        val coldScores = DoubleArray(91)
        val hotScores = DoubleArray(91)
        val combinedRaw = DoubleArray(91)

        val dynamicColdWeight = (coldRatio.toDouble() + if (context.currentZeroMatchStreak >= 3) 0.15 else 0.0)
            .coerceIn(0.2, 0.8)

        val maxEma = (1..90).maxOf { emaHot[it] }.coerceAtLeast(1.0)
        val maxHoraire = (1..90).maxOf { horaireCounts[it] }.coerceAtLeast(1.0)
        val maxCo = (1..90).maxOf { coWithLast[it] }.coerceAtLeast(1.0)

        for (n in 1..90) {
            val expectedGap = (total + 1.0) / (appearances[n] + 1.0)
            val overdueRatio = (gaps[n] + 0.5) / expectedGap.coerceAtLeast(1.0)
            val coldSignal = (overdueRatio.coerceIn(0.2, 3.2) * 0.7) + if (gaps[n] >= 18) 0.45 else 0.0
            val hotSignal = 0.50 * (emaHot[n] / maxEma) +
                0.25 * (horaireCounts[n] / maxHoraire) +
                0.25 * (coWithLast[n] / maxCo)

            val repCount = context.sameDayPredictedCounts[n] ?: 0
            val diversityMult = exp(-diversityLambda * repCount)

            coldScores[n] = coldSignal * diversityMult
            hotScores[n] = hotSignal * diversityMult
            combinedRaw[n] = dynamicColdWeight * coldScores[n] + (1.0 - dynamicColdWeight) * hotScores[n]
        }

        val probs = normalizeToMarginalFive(combinedRaw, priorStrengthDraws = 22.0)

        val targetColdSlots = when {
            dynamicColdWeight >= 0.65 -> 3
            dynamicColdWeight >= 0.40 -> 2
            else -> 1
        }
        val selected = LinkedHashSet<Int>(5)

        val coldRanked = (1..90)
            .filter { gaps[it] >= 14 }
            .sortedWith(compareByDescending<Int> { coldScores[it] }.thenBy { it })
        for (n in coldRanked) {
            if (selected.size >= targetColdSlots) break
            selected.add(n)
        }
        val hotRanked = (1..90).sortedWith(compareByDescending<Int> { hotScores[it] }.thenBy { it })
        for (n in hotRanked) {
            if (selected.size >= 5) break
            selected.add(n)
        }
        if (selected.size < 5) {
            for (n in (1..90).sortedWith(compareByDescending<Int> { probs[it] }.thenBy { it })) {
                if (selected.size >= 5) break
                selected.add(n)
            }
        }
        return selected.sorted() to probs
    }

    fun computeColdSignalVector(past: List<LotoDraw>): Pair<DoubleArray, IntArray> {
        val total = past.size.coerceAtLeast(1)
        val gaps = computeCurrentGaps(past)
        val appearances = IntArray(91)
        for (d in past) {
            for (n in extractDistinctBalls(d)) appearances[n]++
        }
        val coldSignals = DoubleArray(91)
        for (n in 1..90) {
            val expectedGap = (total + 1.0) / (appearances[n] + 1.0)
            val overdueRatio = (gaps[n] + 0.5) / expectedGap.coerceAtLeast(1.0)
            coldSignals[n] = (overdueRatio.coerceIn(0.2, 3.2) * 0.7) + if (gaps[n] >= 18) 0.45 else 0.0
        }
        return coldSignals to gaps
    }

    fun predictHybridWithColdSlots(
        coldSignals: DoubleArray,
        gaps: IntArray,
        hotBaseProbs: DoubleArray,
        context: DrawContext,
        coldRatio: Float,
        diversityLambda: Double,
        streakThreshold: Int = 3
    ): Pair<List<Int>, DoubleArray> {
        val dynamicCold = (coldRatio.toDouble() + if (context.currentZeroMatchStreak >= streakThreshold) 0.18 else 0.0)
            .coerceIn(0.0, 0.85)
        val maxHot = (1..90).maxOf { hotBaseProbs[it] }.coerceAtLeast(1e-6)
        val maxCold = (1..90).maxOf { coldSignals[it] }.coerceAtLeast(1e-6)

        val coldScores = DoubleArray(91)
        val hotScores = DoubleArray(91)
        val combined = DoubleArray(91)
        for (n in 1..90) {
            val rep = context.sameDayPredictedCounts[n] ?: 0
            val mult = if (diversityLambda > 0.0) exp(-diversityLambda * rep) else 1.0
            coldScores[n] = (coldSignals[n] / maxCold) * mult
            hotScores[n] = (hotBaseProbs[n] / maxHot) * mult
            combined[n] = dynamicCold * coldScores[n] + (1.0 - dynamicCold) * hotScores[n]
        }
        val probs = normalizeToMarginalFive(combined, priorStrengthDraws = 22.0)

        val targetColdSlots = when {
            dynamicCold >= 0.65 -> 3
            dynamicCold >= 0.40 -> 2
            dynamicCold >= 0.20 -> 1
            else -> 0
        }
        if (targetColdSlots == 0) {
            return selectTop5(probs) to probs
        }

        val selected = LinkedHashSet<Int>(5)
        val coldRanked = (1..90)
            .filter { gaps[it] >= 14 }
            .sortedWith(compareByDescending<Int> { coldScores[it] }.thenBy { it })
        for (n in coldRanked) {
            if (selected.size >= targetColdSlots) break
            selected.add(n)
        }
        val rankedCombined = (1..90).sortedWith(compareByDescending<Int> { probs[it] }.thenBy { it })
        for (n in rankedCombined) {
            if (selected.size >= 5) break
            selected.add(n)
        }
        return selected.sorted() to probs
    }

    private fun selectTop5(scores: DoubleArray): List<Int> {
        return (1..90)
            .sortedWith(compareByDescending<Int> { scores[it] }.thenBy { it })
            .take(5)
            .sorted()
    }

    private fun predictDirichlet(past: List<LotoDraw>, context: DrawContext): Pair<List<Int>, DoubleArray> {
        val globalCounts = DoubleArray(91)
        val horaireCounts = DoubleArray(91)
        for (d in past) {
            val sameHoraire = LotoDateUtils.normalizeHoraire(d) == context.targetHoraire
            for (n in extractDistinctBalls(d)) {
                globalCounts[n] += 1.0
                if (sameHoraire) horaireCounts[n] += 1.0
            }
        }
        val scores = DoubleArray(91) { n -> if (n == 0) 0.0 else horaireCounts[n] + 0.5 * globalCounts[n] }
        val probs = normalizeToMarginalFive(scores, priorStrengthDraws = 40.0)
        return selectTop5(probs) to probs
    }

    private fun predictEma(past: List<LotoDraw>): Pair<List<Int>, DoubleArray> {
        val scores = DoubleArray(91)
        val recent = past.takeLast(150)
        var weight = 1.0
        for (i in recent.indices.reversed()) {
            for (n in extractDistinctBalls(recent[i])) scores[n] += weight
            weight *= 0.95
        }
        val probs = normalizeToMarginalFive(scores, priorStrengthDraws = 25.0)
        return selectTop5(probs) to probs
    }

    private fun predictMarkovStratified(past: List<LotoDraw>, context: DrawContext): Pair<List<Int>, DoubleArray> {
        val sameGame = past.filter { LotoDateUtils.normalizeGame(it) == context.targetGame }
        val sameHoraire = past.filter { LotoDateUtils.normalizeHoraire(it) == context.targetHoraire }
        val stratum = when {
            sameGame.size >= 25 -> sameGame
            sameHoraire.size >= 25 -> sameHoraire
            else -> past
        }

        val trans1 = Array(91) { DoubleArray(91) }
        val trans2 = Array(91) { DoubleArray(91) }
        val prior = DoubleArray(91)
        for (d in past) {
            for (n in extractDistinctBalls(d)) prior[n] += 1.0
        }
        val priorTotal = (1..90).sumOf { prior[it] }.coerceAtLeast(1.0)

        for (i in 0 until stratum.size - 1) {
            val c1 = extractDistinctBalls(stratum[i])
            val nxt = extractDistinctBalls(stratum[i + 1])
            for (a in c1) for (b in nxt) trans1[a][b] += 1.0
            if (i + 2 < stratum.size) {
                val nxt2 = extractDistinctBalls(stratum[i + 2])
                for (a in c1) for (b in nxt2) trans2[a][b] += 1.0
            }
        }

        val last1 = extractDistinctBalls(stratum.last())
        val last2 = if (stratum.size >= 2) extractDistinctBalls(stratum[stratum.size - 2]) else emptyList()

        val scores = DoubleArray(91)
        val alphaPrior = 35.0
        for (n in 1..90) {
            val baseP = (prior[n] + 1.0) / (priorTotal + 90.0)
            var s1 = 0.0
            for (c in last1) {
                val rowSum = trans1[c].sum()
                s1 += (trans1[c][n] + alphaPrior * baseP) / (rowSum + alphaPrior)
            }
            if (last1.isNotEmpty()) s1 /= last1.size

            var s2 = baseP
            if (last2.isNotEmpty()) {
                var sum2 = 0.0
                for (c in last2) {
                    val rowSum = trans2[c].sum()
                    sum2 += (trans2[c][n] + alphaPrior * baseP) / (rowSum + alphaPrior)
                }
                s2 = sum2 / last2.size
            }
            scores[n] = 0.7 * s1 + 0.3 * s2
        }

        val probs = normalizeToMarginalFive(scores, priorStrengthDraws = 30.0)
        return selectTop5(probs) to probs
    }

    private fun predictCooccurrenceGraph(past: List<LotoDraw>): Pair<List<Int>, DoubleArray> {
        val window = past.takeLast(160)
        val adj = Array(91) { DoubleArray(91) }
        for (d in window) {
            val nums = extractDistinctBalls(d)
            for (a in nums) for (b in nums) if (a != b) adj[a][b] += 1.0
        }
        val seeds = past.takeLast(3)
        val scores = DoubleArray(91)
        var decay = 1.0
        for (idx in seeds.indices.reversed()) {
            for (s in extractDistinctBalls(seeds[idx])) {
                for (n in 1..90) scores[n] += decay * adj[s][n]
            }
            decay *= 0.65
        }
        val probs = normalizeToMarginalFive(scores, priorStrengthDraws = 30.0)
        return selectTop5(probs) to probs
    }

    private fun predictSpectralFourier(past: List<LotoDraw>): Pair<List<Int>, DoubleArray> {
        val window = past.takeLast(128)
        val w = window.size
        if (w < 16) return predictDirichlet(past, DrawContext(0L, "", "", 1))
        val maxK = (w / 4).coerceAtLeast(1)
        val scores = DoubleArray(91)
        for (n in 1..90) {
            var count = 0
            val sig = DoubleArray(w) { t ->
                val hit = n in extractDistinctBalls(window[t])
                if (hit) {
                    count++
                    1.0
                } else {
                    0.0
                }
            }
            val mean = count.toDouble() / w
            var bestAmp = 0.0
            var projected = mean
            for (k in 1..maxK) {
                var re = 0.0
                var im = 0.0
                val freq = 2.0 * PI * k / w
                for (t in 0 until w) {
                    val centered = sig[t] - mean
                    re += centered * cos(freq * t)
                    im -= centered * sin(freq * t)
                }
                val amp = re * re + im * im
                if (amp > bestAmp) {
                    bestAmp = amp
                    projected = (mean + (re / w)).coerceAtLeast(0.0)
                }
            }
            scores[n] = projected
        }
        val probs = normalizeToMarginalFive(scores, priorStrengthDraws = 35.0)
        return selectTop5(probs) to probs
    }

    private fun predictMatrix9x10(past: List<LotoDraw>): Pair<List<Int>, DoubleArray> {
        val scores = DoubleArray(91)
        val lastNums = extractDistinctBalls(past.last())
        for (n in lastNums) {
            val r = (n - 1) / 10
            val c = (n - 1) % 10
            for (dr in -1..1) {
                for (dc in -1..1) {
                    if (dr == 0 && dc == 0) continue
                    val nr = r + dr
                    val nc = c + dc
                    if (nr in 0..8 && nc in 0..9) {
                        val idx = nr * 10 + nc + 1
                        scores[idx] += if (dr == 0 || dc == 0) 1.0 else 0.7
                    }
                }
            }
            for ((dr, dc) in KNIGHT_MOVES) {
                val nr = r + dr
                val nc = c + dc
                if (nr in 0..8 && nc in 0..9) {
                    scores[nr * 10 + nc + 1] += 0.6
                }
            }
        }
        var w = 1.0
        for (d in past.takeLast(40).reversed()) {
            for (n in extractDistinctBalls(d)) scores[n] += 0.04 * w
            w *= 0.95
        }
        val probs = normalizeToMarginalFive(scores, priorStrengthDraws = 25.0)
        return selectTop5(probs) to probs
    }

    private fun predictChanceCroixPyramid(past: List<LotoDraw>): Pair<List<Int>, DoubleArray> {
        val scores = DoubleArray(91)
        val lastNums = extractDistinctBalls(past.last())
        for (n in lastNums) {
            val r9 = (n - 1) / 10
            val c10 = (n - 1) % 10
            for (cc in 0..9) scores[r9 * 10 + cc + 1] += 0.35
            for (rr in 0..8) scores[rr * 10 + c10 + 1] += 0.35
            val r10 = (n - 1) / 9
            val c9 = (n - 1) % 9
            for (cc in 0..8) scores[r10 * 9 + cc + 1] += 0.2
            for (rr in 0..9) scores[rr * 9 + c9 + 1] += 0.2
        }
        for (d in past.takeLast(10)) {
            var row = extractDistinctBalls(d).sorted()
            while (row.size > 1) {
                val next = ArrayList<Int>(row.size - 1)
                for (i in 0 until row.size - 1) {
                    val diff = abs(row[i + 1] - row[i])
                    if (diff in 1..90) scores[diff] += 0.8
                    next.add(diff)
                }
                row = next
            }
            val sum = extractDistinctBalls(d).sum()
            if (sum > 0) {
                val root = 1 + ((sum - 1) % 9)
                var m = root
                while (m <= 90) {
                    scores[m] += 0.15
                    m += 9
                }
            }
        }
        val probs = normalizeToMarginalFive(scores, priorStrengthDraws = 25.0)
        return selectTop5(probs) to probs
    }

    private fun predictLunar(past: List<LotoDraw>, context: DrawContext): Pair<List<Int>, DoubleArray> {
        val targetOctant = LotoDateUtils.lunarPhaseOctant(context.targetTimestamp)
        val octantCounts = DoubleArray(91)
        val globalCounts = DoubleArray(91)
        var octantDraws = 0
        for (d in past) {
            val sameOctant = LotoDateUtils.lunarPhaseOctant(d.timestamp) == targetOctant
            if (sameOctant) octantDraws++
            for (n in extractDistinctBalls(d)) {
                globalCounts[n] += 1.0
                if (sameOctant) octantCounts[n] += 1.0
            }
        }
        val scores = DoubleArray(91)
        val globalWeight = if (octantDraws < 15) 0.5 else 0.15
        for (n in 1..90) scores[n] = octantCounts[n] + globalWeight * globalCounts[n]
        val probs = normalizeToMarginalFive(scores, priorStrengthDraws = 35.0)
        return selectTop5(probs) to probs
    }

    private fun predictCube90(past: List<LotoDraw>): Pair<List<Int>, DoubleArray> {
        val counts = DoubleArray(91)
        for (d in past.takeLast(20)) {
            for (n in extractOrderedBalls(d)) {
                val t = HungarianSolver.cube90Transform(n)
                counts[t] += 1.0
            }
        }
        val top5 = (1..90)
            .sortedWith(compareByDescending<Int> { counts[it] }.thenBy { it })
            .take(5)
            .sorted()
        val smoothed = DoubleArray(91) { n -> if (n == 0) 0.0 else counts[n] + 0.05 }
        val probs = normalizeToMarginalFive(smoothed, priorStrengthDraws = 5.0)
        return top5 to probs
    }

    private fun predictHoraireHungarian(past: List<LotoDraw>, context: DrawContext): Pair<List<Int>, DoubleArray> {
        val sameHoraire = past.filter { LotoDateUtils.normalizeHoraire(it) == context.targetHoraire }
        val stratum = if (sameHoraire.size >= 10) sameHoraire.takeLast(60) else past.takeLast(60)
        val cMatrix = Array(5) { DoubleArray(91) }
        val globalCounts = DoubleArray(91)
        for (d in stratum) {
            val balls = extractOrderedBalls(d)
            for (p in balls.indices) {
                if (p < 5) {
                    val v = balls[p]
                    cMatrix[p][v] += 1.0
                    globalCounts[v] += 1.0
                }
            }
        }
        for (p in 0 until 5) {
            for (n in 1..90) cMatrix[p][n] = cMatrix[p][n] + 0.25 * globalCounts[n]
        }
        val assigned = HungarianSolver.solveMaxWeightAssignment(cMatrix).sorted()
        val marginalRaw = DoubleArray(91) { n ->
            if (n == 0) 0.0 else (0 until 5).sumOf { p -> cMatrix[p][n] }
        }
        val probs = normalizeToMarginalFive(marginalRaw, priorStrengthDraws = 30.0)
        return assigned to probs
    }
}

class AdaptiveMetaLearner {
    private val expertMethods = PredictionMethod.entries.filter {
        it != PredictionMethod.ADAPTIVE_ML && it != PredictionMethod.AUTO_SEARCH
    }
    private val expertCount = expertMethods.size
    private val globalWeights = DoubleArray(expertCount) { 1.0 / expertCount }
    private val horaireWeights = mutableMapOf<String, DoubleArray>()
    private val horaireCounts = mutableMapOf<String, Int>()
    private val recentBrierSums = DoubleArray(expertCount)
    private val recentCounts = IntArray(expertCount)
    private val expertZeroStreaks = IntArray(expertCount)

    fun predict(
        strictPast: List<LotoDraw>,
        context: DrawContext
    ): Triple<List<Int>, DoubleArray, List<Pair<PredictionMethod, DoubleArray>>> {
        return predictWithPrecomputed(strictPast, context, emptyMap())
    }

    fun predictWithPrecomputed(
        strictPast: List<LotoDraw>,
        context: DrawContext,
        precomputed: Map<PredictionMethod, Pair<List<Int>, DoubleArray>>
    ): Triple<List<Int>, DoubleArray, List<Pair<PredictionMethod, DoubleArray>>> {
        val expertProbs = expertMethods.map { method ->
            val probs = precomputed[method]?.second
                ?: StatisticalStrategies.predictIndividual(method, strictPast, context).second
            method to probs
        }
        val activeWeights = effectiveWeights(context.targetHoraire)
        val combinedProbs = DoubleArray(91)
        for (k in expertProbs.indices) {
            val w = activeWeights[k]
            val p = expertProbs[k].second
            for (n in 1..90) combinedProbs[n] += w * p[n]
        }
        val (top5, finalProbs) = StatisticalStrategies.applyIntraDayDiversity(
            baseProbs = combinedProbs,
            context = context,
            diversityLambda = 0.35
        )
        return Triple(top5, finalProbs, expertProbs)
    }

    fun updateAfterOutcome(
        expertProbs: List<Pair<PredictionMethod, DoubleArray>>,
        actualSet: Set<Int>,
        horaire: String
    ) {
        val brierSums = DoubleArray(expertProbs.size)
        val droughtMultipliers = DoubleArray(expertProbs.size) { 1.0 }
        for (k in expertProbs.indices) {
            val probs = expertProbs[k].second
            val brier = ScientificMath.computeBrierScore(probs, actualSet)
            brierSums[k] = brier * 90.0
            recentBrierSums[k] += brier
            recentCounts[k] += 1
            val top5 = (1..90).sortedWith(compareByDescending<Int> { probs[it] }.thenBy { it }).take(5)
            if (top5.any { it in actualSet }) {
                expertZeroStreaks[k] = 0
            } else {
                expertZeroStreaks[k]++
                if (expertZeroStreaks[k] >= 4) droughtMultipliers[k] = 0.88
            }
        }
        applyHedgeStep(globalWeights, brierSums, droughtMultipliers, eta = 0.45, shrinkage = 0.015)
        val hWeights = horaireWeights.getOrPut(horaire) { globalWeights.copyOf() }
        applyHedgeStep(hWeights, brierSums, droughtMultipliers, eta = 0.40, shrinkage = 0.025)
        horaireCounts[horaire] = (horaireCounts[horaire] ?: 0) + 1
    }

    private fun effectiveWeights(horaire: String): DoubleArray {
        val hCount = horaireCounts[horaire] ?: 0
        val hWeights = horaireWeights[horaire] ?: return globalWeights
        if (hCount < 10) return globalWeights
        val alpha = (hCount / (hCount + 25.0)).coerceIn(0.0, 0.7)
        return DoubleArray(expertCount) { k -> (1.0 - alpha) * globalWeights[k] + alpha * hWeights[k] }
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

data class LabSavedModelInput(
    val id: String,
    val displayName: String,
    val savedStreak: Int,
    val slotSplitLabel: String,
    val genome: SymbolicGenome
)

object ClassicalAndScientificLab {

    private const val EPS = 1e-12

    fun computePeriodRange(
        chronological: List<LotoDraw>,
        period: BacktestPeriod = BacktestPeriod.THREE_WEEKS,
        referenceNowMs: Long = System.currentTimeMillis(),
        customDays: Int? = period.days
    ): IntRange {
        val n = chronological.size
        if (n < 4) return 1 until n
        return SymbolicFormula.computePeriodRange(
            chronological = chronological,
            period = period,
            referenceNowMs = referenceNowMs,
            customDays = customDays
        )
    }

    fun computeThreeWeekRange(
        chronological: List<LotoDraw>,
        referenceNowMs: Long = System.currentTimeMillis()
    ): IntRange = computePeriodRange(chronological, BacktestPeriod.THREE_WEEKS, referenceNowMs, 21)

    fun computePeriodSplit(
        chronological: List<LotoDraw>,
        period: BacktestPeriod = BacktestPeriod.THREE_WEEKS,
        referenceNowMs: Long = System.currentTimeMillis(),
        customDays: Int? = period.days
    ): Triple<IntRange, IntRange, IntRange> {
        val range = computePeriodRange(chronological, period, referenceNowMs, customDays)
        val startIdx = range.first
        val endExclusive = (range.last + 1).coerceAtMost(chronological.size)
        val span = endExclusive - startIdx
        if (span < 3) return computeChronologicalSplit(chronological)
        val devCount = if (span < 6) maxOf(1, span - 2) else maxOf(2, (span * 45) / 100)
        val valCount = if (span < 6) 1 else maxOf(2, (span * 30) / 100)
        val devEnd = (startIdx + devCount).coerceAtMost((endExclusive - 2).coerceAtLeast(startIdx + 1))
        val valEnd = (devEnd + valCount).coerceAtMost((endExclusive - 1).coerceAtLeast(devEnd + 1))
        return Triple(startIdx until devEnd, devEnd until valEnd, valEnd until endExclusive)
    }

    fun computeThreeWeekSplit(
        chronological: List<LotoDraw>,
        referenceNowMs: Long = System.currentTimeMillis()
    ): Triple<IntRange, IntRange, IntRange> =
        computePeriodSplit(chronological, BacktestPeriod.THREE_WEEKS, referenceNowMs, 21)

    fun computeChronologicalSplit(chronological: List<LotoDraw>): Triple<IntRange, IntRange, IntRange> {
        val n = chronological.size
        if (n < 20) {
            val w = (n / 4).coerceAtLeast(1)
            val remaining = (n - w).coerceAtLeast(2)
            val devCount = (remaining * 45 / 100).coerceAtLeast(1)
            val valCount = (remaining * 30 / 100).coerceAtLeast(1)
            val devEnd = (w + devCount).coerceAtMost((n - 2).coerceAtLeast(w + 1))
            val valEnd = (devEnd + valCount).coerceAtMost((n - 1).coerceAtLeast(devEnd + 1))
            return Triple(w until devEnd, devEnd until valEnd, valEnd until n)
        }
        val warmup = maxOf(20, (n * 0.25).toInt()).coerceAtMost(n - 9)
        val remaining = n - warmup
        val devCount = maxOf(3, (remaining * 0.45).toInt())
        val valCount = maxOf(3, (remaining * 0.30).toInt())
        val devEnd = (warmup + devCount).coerceAtMost(n - 4)
        val valEnd = (devEnd + valCount).coerceAtMost(n - 2)
        return Triple(warmup until devEnd, devEnd until valEnd, valEnd until n)
    }

    fun evaluateStrategyOnRange(
        chronological: List<LotoDraw>,
        method: PredictionMethod,
        startIndex: Int,
        endIndexExclusive: Int,
        periodLabel: String,
        preTrainedLearner: AdaptiveMetaLearner? = null,
        useSliding6DrawBlocks: Boolean = (periodLabel.contains("Jour") && !periodLabel.contains("Tout")) ||
            periodLabel.startsWith("1 Jour")
    ): BacktestResult {
        val safeStart = startIndex.coerceAtLeast(1)
        val safeEnd = endIndexExclusive.coerceAtMost(chronological.size)
        if (safeStart >= safeEnd) {
            return ScientificMath.emptyBacktestResult(periodLabel).copy(methodName = method.displayName)
        }

        val effectiveMethod = if (method == PredictionMethod.AUTO_SEARCH) PredictionMethod.COLD_HOT_BALANCE else method
        val learner = preTrainedLearner ?: AdaptiveMetaLearner()

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
            val (predicted, marginalProbs) = if (effectiveMethod == PredictionMethod.ADAPTIVE_ML) {
                val (top5, probs, experts) = learner.predict(strictPast, blindContext)
                expertProbsForUpdate = experts
                top5 to probs
            } else {
                StatisticalStrategies.predictIndividual(effectiveMethod, strictPast, blindContext)
            }

            val actualMask = BooleanArray(91)
            for (n in actualNumbers) if (n in 1..90) actualMask[n] = true
            val actualSet = actualNumbers.toSet()
            val matched = predicted.filter { it in 1..90 && actualMask[it] }.sorted()
            val matchCount = matched.size.coerceIn(0, 5)
            val stepBrier = ScientificMath.computeBrierScoreMask(marginalProbs, actualMask)
            val stepLogLoss = ScientificMath.computeBernoulliLogLoss(marginalProbs, actualSet)
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

            for (n in predicted) sameDayPredCounts[n] = (sameDayPredCounts[n] ?: 0) + 1

            distribution[matchCount]++
            matchList.add(matchCount)
            brierSum += stepBrier
            logLossSum += stepLogLoss
            calibrationHistory.add(marginalProbs to actualSet)

            stepResults.add(
                BacktestStepResult(
                    stepIndex = stepResults.size + 1,
                    drawId = targetDraw.uniqueId,
                    drawLabel = "${targetDraw.game} ${targetDraw.time}".trim(),
                    drawDate = LotoDateUtils.formatFrenchDayAndDate(targetDraw.timestamp),
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

            if (effectiveMethod == PredictionMethod.ADAPTIVE_ML && expertProbsForUpdate != null) {
                learner.updateAfterOutcome(expertProbsForUpdate, actualSet, horaire)
            }
        }

        val totalTested = stepResults.size
        if (totalTested == 0) {
            return ScientificMath.emptyBacktestResult(periodLabel).copy(methodName = method.displayName)
        }

        val dailySummaries: List<DailyBacktestSummary>
        val maxConsecDays: Int
        if (useSliding6DrawBlocks) {
            val totalBlocks = (totalTested + 5) / 6
            val groupedBlocks = stepResults.withIndex().groupBy { (idx, _) ->
                val blockFromEnd = (totalTested - 1 - idx) / 6
                totalBlocks - 1 - blockFromEnd
            }.entries.sortedBy { it.key }
            val summariesChrono = groupedBlocks.map { (bIdx, indexedSteps) ->
                val list = indexedSteps.map { it.value }.sortedBy { it.targetTimestamp }
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
            var consec = 0
            var maxC = 0
            var lastIdx = -2
            summariesChrono.forEachIndexed { pos, d ->
                if (d.isQualifyingDay) {
                    consec = if (pos == lastIdx + 1) consec + 1 else 1
                    if (consec > maxC) maxC = consec
                    lastIdx = pos
                } else {
                    consec = 0
                }
            }
            maxConsecDays = maxC
            dailySummaries = summariesChrono.reversed()
        } else {
            dailySummaries = stepResults.groupBy { it.dayKey }.entries
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
            maxConsecDays = computeMaxConsecutiveDays(dailySummaries)
        }
        val horaireBreakdown = stepResults.groupBy { it.horaire }.entries
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
        val (avgLo, avgHi) = ScientificMath.mean95Interval(matchList)
        val (rateLo, rateHi) = ScientificMath.wilson95Interval(partialMatches, totalTested)

        return BacktestResult(
            methodName = method.displayName,
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
            calibrationErrorEce = ScientificMath.computeCalibrationEce(calibrationHistory),
            matchDistribution = distribution.toList(),
            allStepsCausalVerified = stepResults.all { it.causalVerified },
            periodLabel = periodLabel,
            dailySummaries = dailySummaries,
            horaireBreakdown = horaireBreakdown,
            steps = stepResults.reversed(),
            maxConsecutiveQualifyingDays = maxConsecDays,
            maxZeroMatchStreak = maxZeroStreak,
            currentZeroMatchStreak = currentZeroStreak
        )
    }

    /**
     * Comparaison Scientifique complète Android-identique, parallélisée sur tous les cœurs CPU PC,
     * incluant l'audit sans fuite des modèles IA/Chercheur conservés sur la fenêtre active ou Test Indépendant.
     */
    fun runFullScientificComparison(
        data: List<LotoDraw>,
        period: BacktestPeriod = BacktestPeriod.THREE_WEEKS,
        monteCarloSims: Int = 10_000,
        referenceNowMs: Long = System.currentTimeMillis(),
        pool: ForkJoinPool = ForkJoinPool.commonPool(),
        customDays: Int? = period.days,
        savedModels: List<LabSavedModelInput> = emptyList()
    ): ScientificReport {
        val rawChronological = data
            .distinctBy { it.uniqueId }
            .sortedWith(compareBy<LotoDraw> { it.timestamp }.thenBy { it.uniqueId })

        val resolvedDays = if (period == BacktestPeriod.ALL_HISTORY && customDays == null) null else (customDays ?: period.days)?.coerceAtLeast(1)
        val periodLabel = when {
            resolvedDays == null -> period.label
            period.days == resolvedDays -> period.label
            else -> "$resolvedDays Jours (Perso)"
        }
        val useSliding = resolvedDays != null

        val chronological = if (resolvedDays != null && rawChronological.any { it.timestamp <= referenceNowMs }) {
            rawChronological.filter { it.timestamp <= referenceNowMs }
        } else {
            rawChronological
        }

        val (devRange, valRange, testRange) = computePeriodSplit(
            chronological = chronological,
            period = period,
            referenceNowMs = referenceNowMs,
            customDays = resolvedDays
        )

        val devDraws = if (!devRange.isEmpty()) chronological.subList(devRange.first, devRange.last + 1) else emptyList()
        val valDraws = if (!valRange.isEmpty()) chronological.subList(valRange.first, valRange.last + 1) else emptyList()
        val testDraws = if (!testRange.isEmpty()) chronological.subList(testRange.first, testRange.last + 1) else emptyList()

        val activeWindowRange = if (resolvedDays != null) {
            computePeriodRange(chronological, period, referenceNowMs, resolvedDays)
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

        val mcDev = ScientificMath.runMonteCarloParallel(devDraws, simulations = monteCarloSims, seed = 42L, pool = pool)
        val mcVal = ScientificMath.runMonteCarloParallel(valDraws, simulations = monteCarloSims, seed = 43L, pool = pool)
        val mcTest = ScientificMath.runMonteCarloParallel(testDraws, simulations = monteCarloSims, seed = 44L, pool = pool)
        val mcAudit = if (useActiveWindowForAudit) {
            ScientificMath.runMonteCarloParallel(auditDraws, simulations = monteCarloSims, seed = 45L, pool = pool)
        } else {
            mcTest
        }

        val allMethods = PredictionMethod.entries
        val bonferroniThreshold = 0.05 / allMethods.size

        val tasks = allMethods.map { method ->
            Callable {
                val sharedLearner = if (method == PredictionMethod.ADAPTIVE_ML) AdaptiveMetaLearner() else null
                val rDev = evaluateStrategyOnRange(chronological, method, devRange.first, devRange.last + 1, "Développement", sharedLearner, useSliding)
                val rVal = evaluateStrategyOnRange(chronological, method, valRange.first, valRange.last + 1, "Validation", sharedLearner, useSliding)
                val rTest = evaluateStrategyOnRange(chronological, method, testRange.first, testRange.last + 1, "Test Indépendant", sharedLearner, useSliding)
                QuadrupleMethodEval(method, rDev, rVal, rTest)
            }
        }
        val resultsList = pool.invokeAll(tasks).map { it.get() }

        val valResults = resultsList.associate { it.method to it.rVal }
        val testResults = resultsList.associate { it.method to it.rTest }

        val devEntries = resultsList.map { toComparisonEntry(it.method, it.rDev, "Développement", bonferroniThreshold) }
        val valEntries = resultsList.map { toComparisonEntry(it.method, it.rVal, "Validation", bonferroniThreshold) }
        val testEntries = resultsList.map { toComparisonEntry(it.method, it.rTest, "Test Indépendant", bonferroniThreshold) }

        val savedAuditTasks = savedModels.distinctBy { it.id }.map { saved ->
            Callable {
                val auditLabel = if (useSliding) periodLabel else "Test Indépendant"
                val bt = if (!auditRange.isEmpty()) {
                    PcDiscoveryEngine.evaluateGenomeOnRange(
                        chronological = chronological,
                        genome = saved.genome,
                        startIndex = auditRange.first,
                        endIndexExclusive = auditRange.last + 1,
                        periodLabel = auditLabel
                    )
                } else {
                    ScientificMath.emptyBacktestResult(auditLabel)
                }
                val matches = bt.steps.map { it.matchCount }
                val pRaw = ScientificMath.twoSidedPValueVsRandom(matches)
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
            }
        }
        val savedModelAudits = if (savedAuditTasks.isNotEmpty()) {
            pool.invokeAll(savedAuditTasks).map { it.get() }
                .sortedWith(
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

        val flaggedTest = testEntries.map { it.copy(isRecommendedParsimonious = it.method == recommended) }
            .sortedWith(compareByDescending<StrategyComparisonEntry> { it.objectiveValidated }.thenByDescending { it.qualifyingDaysCount }.thenByDescending { it.averageMatches })
        val flaggedVal = valEntries.map { it.copy(isRecommendedParsimonious = it.method == recommended) }
            .sortedWith(compareByDescending<StrategyComparisonEntry> { it.objectiveValidated }.thenByDescending { it.qualifyingDaysCount }.thenByDescending { it.averageMatches })
        val flaggedDev = devEntries.map { it.copy(isRecommendedParsimonious = it.method == recommended) }
            .sortedWith(compareByDescending<StrategyComparisonEntry> { it.objectiveValidated }.thenByDescending { it.qualifyingDaysCount }.thenByDescending { it.averageMatches })

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

    fun evaluateAndPredictClassicalMethod(
        method: PredictionMethod,
        draws: List<LotoDraw>,
        period: BacktestPeriod,
        nextTarget: NextDrawTarget,
        referenceNowMs: Long = System.currentTimeMillis(),
        customDays: Int? = period.days
    ): Pair<BacktestResult, PredictionResult> {
        val chronological = draws.distinctBy { it.uniqueId }
            .sortedWith(compareBy<LotoDraw> { it.timestamp }.thenBy { it.uniqueId })
        val resolvedDays = if (period == BacktestPeriod.ALL_HISTORY && customDays == null) null else (customDays ?: period.days)?.coerceAtLeast(1)
        val range = computePeriodRange(chronological, period, referenceNowMs = referenceNowMs, customDays = resolvedDays)
        val periodLabel = when {
            resolvedDays == null -> period.label
            period.days == resolvedDays -> period.label
            else -> "$resolvedDays Jours (Perso)"
        }
        val bt = evaluateStrategyOnRange(chronological, method, range.first, range.last + 1, periodLabel)

        val effectiveMethod = if (method == PredictionMethod.AUTO_SEARCH) PredictionMethod.COLD_HOT_BALANCE else method
        val gaps = StatisticalStrategies.computeCurrentGaps(chronological)
        val ctx = DrawContext(
            targetTimestamp = nextTarget.targetTimestamp,
            targetHoraire = nextTarget.horaire,
            targetGame = LotoDateUtils.normalizeGame(chronological.lastOrNull() ?: LotoDraw("", "", "", "Fortune", 1, 2, 3, 4, 5, 0L)),
            targetDayOfWeek = LotoDateUtils.dayOfWeek(nextTarget.targetTimestamp)
        )
        val (pred, probs) = if (effectiveMethod == PredictionMethod.ADAPTIVE_ML) {
            val learner = AdaptiveMetaLearner()
            val warm = maxOf(10, chronological.size - 80)
            var warmStrictEnd = 0
            for (t in warm until chronological.size) {
                val cur = chronological[t]
                while (warmStrictEnd < t && chronological[warmStrictEnd].timestamp < cur.timestamp) {
                    warmStrictEnd++
                }
                if (warmStrictEnd > 0) {
                    val past = chronological.subList(0, warmStrictEnd)
                    val c = DrawContext(cur.timestamp, LotoDateUtils.normalizeHoraire(cur), LotoDateUtils.normalizeGame(cur), LotoDateUtils.dayOfWeek(cur.timestamp))
                    val (_, _, experts) = learner.predict(past, c)
                    learner.updateAfterOutcome(experts, StatisticalStrategies.extractDistinctBalls(cur).toSet(), c.targetHoraire)
                }
            }
            learner.predict(chronological, ctx).let { it.first to it.second }
        } else {
            StatisticalStrategies.predictIndividual(effectiveMethod, chronological, ctx)
        }

        val ranked = (1..90).sortedWith(compareByDescending<Int> { probs[it] }.thenBy { it })
        val rankMap = HashMap<Int, Int>(90).apply { ranked.forEachIndexed { idx, num -> put(num, idx + 1) } }
        val maxP = (1..90).maxOf { probs[it] }.coerceAtLeast(1e-6)
        val baseline = 5.0 / 90.0
        val toConf: (Int) -> PredictedNumberConfidence = { n ->
            val p = probs[n]
            PredictedNumberConfidence(
                number = n,
                marginalProbabilityPercent = (p * 100.0).toFloat(),
                relativeConfidencePercent = ((p / maxP) * 100.0).toFloat().coerceIn(0f, 100f),
                liftVsRandom = (p / baseline).toFloat(),
                rank = rankMap[n] ?: 90,
                sourceBranch = method.shortLabel,
                isGapCandidate = gaps[n] >= 18
            )
        }
        val predRes = PredictionResult(
            predictedDraw = pred.sorted(),
            expectedMatchesEstimate = pred.sumOf { probs[it] }.toFloat(),
            calibratedProbAtLeastOne = ScientificMath.calibratedProbAtLeastOne(pred, probs),
            trainedOnCount = chronological.size,
            methodName = method.displayName,
            horaire = nextTarget.horaire,
            marginalSumCheck = (1..90).sumOf { probs[it] }.toFloat(),
            auditNote = "Méthode Labo Scientifique (${method.family})",
            discoveredStrategyName = method.displayName,
            coldNumbersCount = StatisticalStrategies.countColdNumbers(pred, chronological),
            searchRoundsExecuted = 1,
            totalStrategiesSearched = 1,
            objectiveFourDaysAchieved = bt.maxConsecutiveQualifyingDays >= 4,
            bestCausalStreakFound = bt.maxConsecutiveQualifyingDays,
            methodANumbers = pred.take(3),
            methodBNumbers = pred.drop(3),
            methodALabel = method.displayName,
            methodBLabel = method.family,
            numberConfidences = pred.map(toConf),
            candidateConfidences = ranked.take(10).map(toConf)
        )
        return bt to predRes
    }

    private data class QuadrupleMethodEval(
        val method: PredictionMethod,
        val rDev: BacktestResult,
        val rVal: BacktestResult,
        val rTest: BacktestResult
    )

    private fun computeMaxConsecutiveDays(dailySummaries: List<DailyBacktestSummary>): Int {
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
                ) currentStreak + 1 else 1
                if (currentStreak > maxStreak) maxStreak = currentStreak
                lastQualDayKey = day.dayKey
            } else {
                currentStreak = 0
            }
        }
        return maxStreak
    }

    private fun selectParsimoniousWinner(valEntries: List<StrategyComparisonEntry>): PredictionMethod {
        val candidates = valEntries.filter { it.testedCount > 0 }
        if (candidates.isEmpty()) return PredictionMethod.DIRICHLET_MULTINOMIAL
        val bestScore = candidates.maxOf { it.averageMatches.toDouble() - 2.0 * it.brierScore }
        val simpleCandidate = candidates.filter { !it.method.isComplex }
            .maxByOrNull { it.averageMatches.toDouble() - 2.0 * it.brierScore }
        val overallBest = candidates.maxBy { it.averageMatches.toDouble() - 2.0 * it.brierScore }
        if (simpleCandidate != null) {
            val simpleScore = simpleCandidate.averageMatches.toDouble() - 2.0 * simpleCandidate.brierScore
            if (bestScore - simpleScore <= 0.015) return simpleCandidate.method
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
        method: PredictionMethod,
        res: BacktestResult,
        windowName: String,
        bonferroniThreshold: Double
    ): StrategyComparisonEntry {
        val matches = res.steps.map { it.matchCount }
        val pVal = ScientificMath.twoSidedPValueVsRandom(matches)
        val penOkDays = res.dailySummaries.count { it.penultimateTwoPlus }
        return StrategyComparisonEntry(
            method = method,
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

    private fun formatRangeLabel(draws: List<LotoDraw>): String {
        if (draws.isEmpty()) return "Aucune donnée"
        return "${LotoDateUtils.formatShortDate(draws.first().timestamp)} → ${LotoDateUtils.formatShortDate(draws.last().timestamp)}"
    }
}
