package com.example.domain.engine

import com.example.data.local.LotoResultEntity
import com.example.domain.util.LotoDateUtils
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.sin

object StatisticalStrategies {

    private val KNIGHT_MOVES = arrayOf(
        -2 to -1, -2 to 1, -1 to -2, -1 to 2,
        1 to -2, 1 to 2, 2 to -1, 2 to 1
    )

    fun extractOrderedBalls(draw: LotoResultEntity): List<Int> {
        return listOf(draw.num1, draw.num2, draw.num3, draw.num4, draw.num5)
            .filter { it in 1..90 }
    }

    fun extractDistinctBalls(draw: LotoResultEntity): List<Int> {
        return extractOrderedBalls(draw).distinct()
    }

    /**
     * Normalise un vecteur de scores positifs en probabilites marginales d'inclusion p_i = P(i in S),
     * avec lissage bayesien vers 5/90 et contrainte exacte sum_{i=1..90} p_i = 5.0.
     */
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
        // Met a l'echelle les scores pour correspondre a un nombre effectif d'observations
        val scale = (sum.coerceAtLeast(1.0))
        for (i in 1..90) {
            val empiricalShare = cleaned[i] / scale
            val p = (empiricalShare * scale + priorTotalBalls * (1.0 / 90.0)) / (scale + priorTotalBalls)
            probs[i] = 5.0 * p
        }

        // Projection exacte dans [1e-4, 0.999] preservant sum = 5.0
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

    /**
     * Evalue une strategie individuelle uniquement sur le passe strict.
     * Retourne : (top5 prediction, probabilites marginales p_1..90 sommant a 5.0).
     */
    fun predictIndividual(
        method: PredictionMethod,
        strictPast: List<LotoResultEntity>,
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

    /**
     * Compte combien de numeros dans `numbers` sont consideres froids/en retard (ecart >= 18 tirages).
     */
    fun countColdNumbers(numbers: List<Int>, past: List<LotoResultEntity>): Int {
        if (past.isEmpty()) return 0
        val gaps = computeCurrentGaps(past)
        return numbers.count { it in 1..90 && gaps[it] >= 18 }
    }

    /**
     * Applique une penalite causale anti-repetition intra-jour sur un vecteur de probabilites marginales
     * afin d'eviter de repeter les memes 5 numeros sur tous les tirages d'une meme journee.
     */
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

    fun computeCurrentGaps(past: List<LotoResultEntity>): IntArray {
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

    /**
     * Strategie Equilibre Chaud/Froid + Analyse des Ecarts + Anti-repetition intra-jour.
     * Combine explicitement des numeros froids/en retard avec des signaux chauds/transitionnels
     * et diversifie automatiquement par rapport aux predictions deja emises dans la meme journee.
     */
    fun predictColdHotVariant(
        past: List<LotoResultEntity>,
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
                if (i >= past.size - 140) {
                    emaHot[n] += w
                }
                if (sameH) horaireCounts[n] += 1.0
            }
            if (i >= past.size - 140) w *= emaDecay
        }

        // Co-occurrence avec le tirage immediatement precedent
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

        // Si serie a 0 correspondance >= 3, renforce l'exploration vers les numeros froids
        val dynamicColdWeight = (coldRatio.toDouble() + if (context.currentZeroMatchStreak >= 3) 0.15 else 0.0)
            .coerceIn(0.2, 0.8)

        val maxEma = (1..90).maxOf { emaHot[it] }.coerceAtLeast(1.0)
        val maxHoraire = (1..90).maxOf { horaireCounts[it] }.coerceAtLeast(1.0)
        val maxCo = (1..90).maxOf { coWithLast[it] }.coerceAtLeast(1.0)

        for (n in 1..90) {
            val expectedGap = (total + 1.0) / (appearances[n] + 1.0)
            val overdueRatio = (gaps[n] + 0.5) / expectedGap.coerceAtLeast(1.0)
            // Pic Weibull doux pour les numeros froids ayant depasse leur cycle moyen (ratio >= 1.0)
            val coldSignal = (overdueRatio.coerceIn(0.2, 3.2) * 0.7) + if (gaps[n] >= 18) 0.45 else 0.0

            val hotSignal = 0.50 * (emaHot[n] / maxEma) +
                0.25 * (horaireCounts[n] / maxHoraire) +
                0.25 * (coWithLast[n] / maxCo)

            val repCount = context.sameDayPredictedCounts[n] ?: 0
            val diversityMult = exp(-diversityLambda * repCount)

            coldScores[n] = coldSignal * diversityMult
            hotScores[n] = hotSignal * diversityMult
            combinedRaw[n] = (dynamicColdWeight * coldScores[n] + (1.0 - dynamicColdWeight) * hotScores[n])
        }

        val probs = normalizeToMarginalFive(combinedRaw, priorStrengthDraws = 22.0)

        // Selection structuree garantissant un melange reel de numeros froids et chauds non repetes
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

    /**
     * Calcule une seule fois les signaux de retard/froid (gaps et ecarts normalises) pour un pas donne.
     */
    fun computeColdSignalVector(past: List<LotoResultEntity>): Pair<DoubleArray, IntArray> {
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

    /**
     * Melange rapide O(90) entre une distribution chaude/structurelle precalculee et le vecteur froid,
     * avec anti-repetition intra-jour et injection adaptative de numeros froids en cas de serie nulle.
     */
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

    // 1. Modele Bayesien Dirichlet-Multinomial hierarchique (global + strate horaire)
    private fun predictDirichlet(
        past: List<LotoResultEntity>,
        context: DrawContext
    ): Pair<List<Int>, DoubleArray> {
        val globalCounts = DoubleArray(91)
        val horaireCounts = DoubleArray(91)
        for (d in past) {
            val sameHoraire = LotoDateUtils.normalizeHoraire(d) == context.targetHoraire
            for (n in extractDistinctBalls(d)) {
                globalCounts[n] += 1.0
                if (sameHoraire) horaireCounts[n] += 1.0
            }
        }
        val scores = DoubleArray(91)
        for (n in 1..90) {
            // Shrinkage hierarchique : horaire renforce par l'a priori global
            scores[n] = horaireCounts[n] + 0.5 * globalCounts[n]
        }
        val probs = normalizeToMarginalFive(scores, priorStrengthDraws = 40.0)
        return selectTop5(probs) to probs
    }

    // 2. Moyenne Mobile Exponentielle (EMA)
    private fun predictEma(past: List<LotoResultEntity>): Pair<List<Int>, DoubleArray> {
        val scores = DoubleArray(91)
        val recent = past.takeLast(150)
        var weight = 1.0
        val decay = 0.95
        for (i in recent.indices.reversed()) {
            for (n in extractDistinctBalls(recent[i])) {
                scores[n] += weight
            }
            weight *= decay
        }
        val probs = normalizeToMarginalFive(scores, priorStrengthDraws = 25.0)
        return selectTop5(probs) to probs
    }

    // 3. Chaine de Markov hierarchique stratifiee par Jeu + Horaire (ordre 1 et 2)
    private fun predictMarkovStratified(
        past: List<LotoResultEntity>,
        context: DrawContext
    ): Pair<List<Int>, DoubleArray> {
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
            for (a in c1) {
                for (b in nxt) trans1[a][b] += 1.0
            }
            if (i + 2 < stratum.size) {
                val nxt2 = extractDistinctBalls(stratum[i + 2])
                for (a in c1) {
                    for (b in nxt2) trans2[a][b] += 1.0
                }
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

    // 4. Graphe de co-occurrence pondere
    private fun predictCooccurrenceGraph(past: List<LotoResultEntity>): Pair<List<Int>, DoubleArray> {
        val window = past.takeLast(160)
        val adj = Array(91) { DoubleArray(91) }
        for (d in window) {
            val nums = extractDistinctBalls(d)
            for (a in nums) {
                for (b in nums) {
                    if (a != b) adj[a][b] += 1.0
                }
            }
        }
        val seeds = past.takeLast(3)
        val scores = DoubleArray(91)
        var decay = 1.0
        for (idx in seeds.indices.reversed()) {
            for (s in extractDistinctBalls(seeds[idx])) {
                for (n in 1..90) {
                    scores[n] += decay * adj[s][n]
                }
            }
            decay *= 0.65
        }
        val probs = normalizeToMarginalFive(scores, priorStrengthDraws = 30.0)
        return selectTop5(probs) to probs
    }

    // 5. Decomposition spectrale de Fourier (DFT)
    private fun predictSpectralFourier(past: List<LotoResultEntity>): Pair<List<Int>, DoubleArray> {
        val window = past.takeLast(128)
        val w = window.size
        if (w < 16) {
            return predictDirichlet(past, DrawContext(0L, "", "", 1))
        }
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
                    // Extrapolation harmonique dominante au pas t = w (regularisee)
                    projected = (mean + (re / w)).coerceAtLeast(0.0)
                }
            }
            scores[n] = projected
        }
        val probs = normalizeToMarginalFive(scores, priorStrengthDraws = 35.0)
        return selectTop5(probs) to probs
    }

    // 6. Geometrie matricielle 9x10 (voisins H/V/diagonaux + deplacements en cavalier)
    private fun predictMatrix9x10(past: List<LotoResultEntity>): Pair<List<Int>, DoubleArray> {
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
        // Tie-break doux par recence EMA
        var w = 1.0
        for (d in past.takeLast(40).reversed()) {
            for (n in extractDistinctBalls(d)) scores[n] += 0.04 * w
            w *= 0.95
        }
        val probs = normalizeToMarginalFive(scores, priorStrengthDraws = 25.0)
        return selectTop5(probs) to probs
    }

    // 7. Chance-Croix (9x10 & 10x9) + Pyramides de differences + Racines numeriques
    private fun predictChanceCroixPyramid(past: List<LotoResultEntity>): Pair<List<Int>, DoubleArray> {
        val scores = DoubleArray(91)
        val lastNums = extractDistinctBalls(past.last())
        for (n in lastNums) {
            // Croix sur grille 9x10
            val r9 = (n - 1) / 10
            val c10 = (n - 1) % 10
            for (cc in 0..9) scores[r9 * 10 + cc + 1] += 0.35
            for (rr in 0..8) scores[rr * 10 + c10 + 1] += 0.35
            // Croix sur grille 10x9
            val r10 = (n - 1) / 9
            val c9 = (n - 1) % 9
            for (cc in 0..8) scores[r10 * 9 + cc + 1] += 0.2
            for (rr in 0..9) scores[rr * 9 + c9 + 1] += 0.2
        }
        // Pyramide des differences successives + racine numerique sur 10 derniers tirages
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

    // 8. Phases lunaires astronomiques exactes (hypothese experimentale)
    private fun predictLunar(
        past: List<LotoResultEntity>,
        context: DrawContext
    ): Pair<List<Int>, DoubleArray> {
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
        for (n in 1..90) {
            scores[n] = octantCounts[n] + globalWeight * globalCounts[n]
        }
        val probs = normalizeToMarginalFive(scores, priorStrengthDraws = 35.0)
        return selectTop5(probs) to probs
    }

    // 9. Cube90 (Formule exacte auditee : 30 images sur 90, collisions 3-pour-1)
    private fun predictCube90(past: List<LotoResultEntity>): Pair<List<Int>, DoubleArray> {
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

        // Lissage minimal (plancher 0.05) pour garder un Log-Loss defini malgre 60 numeros a zero
        val smoothed = DoubleArray(91) { n -> if (n == 0) 0.0 else counts[n] + 0.05 }
        val probs = normalizeToMarginalFive(smoothed, priorStrengthDraws = 5.0)
        return top5 to probs
    }

    // 10. Optimisation positionnelle par Horaire C_{p,n} (Algorithme Hongrois)
    private fun predictHoraireHungarian(
        past: List<LotoResultEntity>,
        context: DrawContext
    ): Pair<List<Int>, DoubleArray> {
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
        // Regularisation douce de chaque position vers la frequence globale de la strate
        for (p in 0 until 5) {
            for (n in 1..90) {
                cMatrix[p][n] = cMatrix[p][n] + 0.25 * globalCounts[n]
            }
        }

        val assigned = HungarianSolver.solveMaxWeightAssignment(cMatrix).sorted()
        val marginalRaw = DoubleArray(91) { n ->
            if (n == 0) 0.0 else (0 until 5).sumOf { p -> cMatrix[p][n] }
        }
        val probs = normalizeToMarginalFive(marginalRaw, priorStrengthDraws = 30.0)
        return assigned to probs
    }
}
