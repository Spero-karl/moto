package com.example.domain.engine

import com.example.data.local.LotoResultEntity
import com.example.domain.util.LotoDateUtils
import kotlin.random.Random

data class AutoSearchNextPrediction(
    val predictedDraw: List<Int>,
    val marginalProbs: DoubleArray,
    val discoveredLabel: String,
    val coldNumbersCount: Int,
    val diversityApplied: Boolean,
    val expertWeights: List<ExpertWeightInfo>,
    val roundsExecuted: Int,
    val totalStrategiesSearched: Int,
    val objectiveFourDaysAchieved: Boolean,
    val bestConsecutiveStreak: Int
)

private data class SearchCandidateSpec(
    val id: String,
    val label: String,
    val roundNumber: Int,
    val expertMix: List<Pair<PredictionMethod, Double>>,
    val coldRatio: Float,
    val diversityLambda: Double,
    val streakThreshold: Int = 3,
    val usesAdaptiveMl: Boolean = false
)

private class StepPrecomputedData(
    val draw: LotoResultEntity,
    val actualNumbers: List<Int>,
    val actualSet: Set<Int>,
    val strictPastSize: Int,
    val maxPastTs: Long,
    val dayKey: String,
    val horaire: String,
    val game: String,
    val baseProbs: Map<PredictionMethod, DoubleArray>,
    val coldSignals: DoubleArray,
    val gaps: IntArray,
    val mlProbs: DoubleArray
)

private class CandidateCausalTracker(val spec: SearchCandidateSpec) {
    var testedCount = 0
    var hitsSum = 0
    var twoPlusCount = 0
    var brierSum = 0.0
    var currentZeroStreak = 0
    var maxZeroStreak = 0

    private var currentDayKey: String = ""
    private var currentDayTwoPlus = 0
    private var currentDayPrevMatchCount = 0
    private var currentDayLastMatchCount = 0
    var qualifyingDaysCount = 0
        private set
    private var consecutiveQualDays = 0
    var maxConsecutiveQualDays = 0
        private set
    private var lastQualDayKey: String = ""

    private var predDayKey: String = ""
    private val predDayCounts = HashMap<Int, Int>()
    private val recentHits = ArrayDeque<Int>()
    private var recentHitsSum = 0

    fun sameDayCountsFor(dayKey: String): Map<Int, Int> {
        return if (dayKey == predDayKey) predDayCounts else emptyMap()
    }

    fun recordPredictionAndOutcome(
        dayKey: String,
        predicted: List<Int>,
        matchCount: Int,
        stepBrier: Double
    ) {
        if (dayKey != predDayKey) {
            predDayKey = dayKey
            predDayCounts.clear()
        }
        for (n in predicted) {
            predDayCounts[n] = (predDayCounts[n] ?: 0) + 1
        }

        testedCount++
        hitsSum += matchCount
        if (matchCount >= 2) twoPlusCount++
        brierSum += stepBrier

        recentHits.addLast(matchCount)
        recentHitsSum += matchCount
        if (recentHits.size > 24) {
            recentHitsSum -= recentHits.removeFirst()
        }

        if (matchCount == 0) {
            currentZeroStreak++
            if (currentZeroStreak > maxZeroStreak) maxZeroStreak = currentZeroStreak
        } else {
            currentZeroStreak = 0
        }

        if (dayKey != currentDayKey) {
            finalizePendingDay()
            currentDayKey = dayKey
            currentDayTwoPlus = 0
            currentDayPrevMatchCount = 0
            currentDayLastMatchCount = 0
        }
        currentDayPrevMatchCount = currentDayLastMatchCount
        currentDayLastMatchCount = matchCount
        if (matchCount >= 2) {
            currentDayTwoPlus++
        }
    }

    private fun finalizePendingDay() {
        if (currentDayKey.isEmpty()) return
        val qualified = currentDayTwoPlus >= 3 && currentDayPrevMatchCount >= 2
        if (qualified) {
            qualifyingDaysCount++
            consecutiveQualDays = if (
                lastQualDayKey.isNotEmpty() &&
                LotoDateUtils.areConsecutiveCalendarDays(lastQualDayKey, currentDayKey)
            ) {
                consecutiveQualDays + 1
            } else {
                1
            }
            if (consecutiveQualDays > maxConsecutiveQualDays) {
                maxConsecutiveQualDays = consecutiveQualDays
            }
            lastQualDayKey = currentDayKey
        } else {
            consecutiveQualDays = 0
        }
    }

    fun snapshotMaxStreakIncludingPendingDay(): Int {
        if (currentDayKey.isEmpty() || currentDayTwoPlus < 3 || currentDayPrevMatchCount < 2) return maxConsecutiveQualDays
        val pendingStreak = if (
            lastQualDayKey.isNotEmpty() &&
            LotoDateUtils.areConsecutiveCalendarDays(lastQualDayKey, currentDayKey)
        ) {
            consecutiveQualDays + 1
        } else {
            1
        }
        return maxOf(maxConsecutiveQualDays, pendingStreak)
    }

    fun snapshotQualifyingDaysIncludingPending(): Int {
        return qualifyingDaysCount + if (currentDayKey.isNotEmpty() && currentDayTwoPlus >= 3 && currentDayPrevMatchCount >= 2) 1 else 0
    }

    fun causalUtilityScore(): Double {
        if (testedCount == 0) return 0.0
        val recentAvg = recentHitsSum.toDouble() / recentHits.size.coerceAtLeast(1)
        val avgBrier = brierSum / testedCount
        val twoPlusRate = twoPlusCount.toDouble() / testedCount
        val streak = snapshotMaxStreakIncludingPendingDay()
        val qDays = snapshotQualifyingDaysIncludingPending()
        return (18.0 * twoPlusRate) +
            (9.0 * streak) +
            (4.5 * qDays) +
            (2.8 * twoPlusCount) +
            (0.9 * recentAvg) -
            (0.75 * currentZeroStreak) -
            (0.25 * maxZeroStreak) -
            (5.0 * avgBrier)
    }

    fun toSummary(isWinner: Boolean): AutoSearchCandidateSummary {
        finalizePendingDay()
        currentDayKey = ""
        currentDayTwoPlus = 0
        val avg = if (testedCount > 0) hitsSum.toFloat() / testedCount else 0f
        val avgBrier = if (testedCount > 0) brierSum / testedCount else BacktestResult.RANDOM_BRIER_SCORE
        return AutoSearchCandidateSummary(
            id = spec.id,
            label = spec.label,
            coldRatio = spec.coldRatio,
            diversityStrength = spec.diversityLambda.toFloat(),
            testedCount = testedCount,
            averageMatches = avg,
            twoPlusCount = twoPlusCount,
            qualifyingDaysCount = qualifyingDaysCount,
            maxConsecutiveQualifyingDays = maxConsecutiveQualDays,
            maxZeroMatchStreak = maxZeroStreak,
            brierScore = avgBrier,
            isWinner = isWinner
        )
    }
}

object AutoSearchEngine {

    private val baseExpertMethods = PredictionMethod.entries.filter {
        it != PredictionMethod.ADAPTIVE_ML && it != PredictionMethod.AUTO_SEARCH
    }

    private val coreMixableMethods = listOf(
        PredictionMethod.DIRICHLET_MULTINOMIAL,
        PredictionMethod.EMA_RECENCY,
        PredictionMethod.MARKOV_STRATIFIED,
        PredictionMethod.COOCCURRENCE_GRAPH,
        PredictionMethod.SPECTRAL_FOURIER,
        PredictionMethod.MATRIX_9X10,
        PredictionMethod.CHANCE_CROIX_PYRAMID,
        PredictionMethod.LUNAR_ASTRONOMICAL,
        PredictionMethod.HORAIRE_OPTIM
    )

    private fun buildRoundCandidates(round: Int): List<SearchCandidateSpec> {
        return when (round) {
            1 -> listOf(
                SearchCandidateSpec("r1_cold_50", "Chaud/Froid 50-50 + Anti-Répétition", 1, listOf(PredictionMethod.COLD_HOT_BALANCE to 1.0), 0.50f, 0.65, 3),
                SearchCandidateSpec("r1_cold_70", "Froid Dominant 70-30 + Écarts", 1, listOf(PredictionMethod.COLD_HOT_BALANCE to 1.0), 0.70f, 0.90, 2),
                SearchCandidateSpec("r1_cold_35", "Chaud 65 / Froid 35 + Anti-Répétition", 1, listOf(PredictionMethod.COLD_HOT_BALANCE to 1.0), 0.35f, 0.50, 3),
                SearchCandidateSpec("r1_ml_div", "ML Bayésien Réactif + Diversité", 1, emptyList(), 0.25f, 0.45, 3, usesAdaptiveMl = true),
                SearchCandidateSpec("r1_dir_div", "Dirichlet + Anti-Répétition Jour", 1, listOf(PredictionMethod.DIRICHLET_MULTINOMIAL to 1.0), 0.20f, 0.65, 3),
                SearchCandidateSpec("r1_ema_div", "EMA + Anti-Répétition Jour", 1, listOf(PredictionMethod.EMA_RECENCY to 1.0), 0.20f, 0.70, 3),
                SearchCandidateSpec("r1_markov_div", "Markov Stratifié + Anti-Répétition", 1, listOf(PredictionMethod.MARKOV_STRATIFIED to 1.0), 0.20f, 0.55, 3),
                SearchCandidateSpec("r1_coocc_div", "Graphe Co-occ. + Anti-Répétition", 1, listOf(PredictionMethod.COOCCURRENCE_GRAPH to 1.0), 0.20f, 0.60, 3),
                SearchCandidateSpec("r1_horaire_div", "Horaire Hongrois + Anti-Répétition", 1, listOf(PredictionMethod.HORAIRE_OPTIM to 1.0), 0.20f, 0.65, 3),
                SearchCandidateSpec("r1_matrix_div", "Grille 9×10 + Anti-Répétition", 1, listOf(PredictionMethod.MATRIX_9X10 to 1.0), 0.25f, 0.55, 3),
                SearchCandidateSpec("r1_croix_div", "Croix/Pyramide + Anti-Répétition", 1, listOf(PredictionMethod.CHANCE_CROIX_PYRAMID to 1.0), 0.25f, 0.55, 3),
                SearchCandidateSpec("r1_fourier_div", "Fourier DFT + Anti-Répétition", 1, listOf(PredictionMethod.SPECTRAL_FOURIER to 1.0), 0.20f, 0.60, 3),
                SearchCandidateSpec("r1_lunar_div", "Lunaire (Exp.) + Anti-Répétition", 1, listOf(PredictionMethod.LUNAR_ASTRONOMICAL to 1.0), 0.20f, 0.65, 3),
                SearchCandidateSpec("r1_cube90", "Cube90 (30 images)", 1, listOf(PredictionMethod.CUBE90 to 1.0), 0.0f, 0.0, 99),
                SearchCandidateSpec("r1_dir_pure", "Dirichlet Pur", 1, listOf(PredictionMethod.DIRICHLET_MULTINOMIAL to 1.0), 0.0f, 0.0, 99),
                SearchCandidateSpec("r1_markov_pure", "Markov Stratifié Pur", 1, listOf(PredictionMethod.MARKOV_STRATIFIED to 1.0), 0.0f, 0.0, 99)
            )
            2 -> listOf(
                SearchCandidateSpec("r2_cold_markov_45", "Hybride Markov + Froids 45%", 2, listOf(PredictionMethod.MARKOV_STRATIFIED to 1.0), 0.45f, 0.70, 2),
                SearchCandidateSpec("r2_cold_markov_65", "Hybride Markov + Froids 65%", 2, listOf(PredictionMethod.MARKOV_STRATIFIED to 1.0), 0.65f, 0.85, 2),
                SearchCandidateSpec("r2_cold_horaire_45", "Hybride Horaire + Froids 45%", 2, listOf(PredictionMethod.HORAIRE_OPTIM to 1.0), 0.45f, 0.75, 2),
                SearchCandidateSpec("r2_cold_horaire_65", "Hybride Horaire + Froids 65%", 2, listOf(PredictionMethod.HORAIRE_OPTIM to 1.0), 0.65f, 0.90, 2),
                SearchCandidateSpec("r2_cold_matrix_50", "Hybride Grille 9×10 + Froids 50%", 2, listOf(PredictionMethod.MATRIX_9X10 to 1.0), 0.50f, 0.70, 2),
                SearchCandidateSpec("r2_cold_coocc_50", "Hybride Co-occ. + Froids 50%", 2, listOf(PredictionMethod.COOCCURRENCE_GRAPH to 1.0), 0.50f, 0.75, 2),
                SearchCandidateSpec("r2_cold_croix_50", "Hybride Croix/Pyr. + Froids 50%", 2, listOf(PredictionMethod.CHANCE_CROIX_PYRAMID to 1.0), 0.50f, 0.70, 2),
                SearchCandidateSpec("r2_cold_fourier_45", "Hybride Fourier + Froids 45%", 2, listOf(PredictionMethod.SPECTRAL_FOURIER to 1.0), 0.45f, 0.75, 2),
                SearchCandidateSpec("r2_cold_lunar_45", "Hybride Lunaire + Froids 45%", 2, listOf(PredictionMethod.LUNAR_ASTRONOMICAL to 1.0), 0.45f, 0.75, 2),
                SearchCandidateSpec("r2_cold_ema_55", "Hybride EMA + Froids 55%", 2, listOf(PredictionMethod.EMA_RECENCY to 1.0), 0.55f, 0.85, 2),
                SearchCandidateSpec("r2_mix_mk_hr", "Duo Markov + Horaire + Froids 40%", 2, listOf(PredictionMethod.MARKOV_STRATIFIED to 0.55, PredictionMethod.HORAIRE_OPTIM to 0.45), 0.40f, 0.65, 2),
                SearchCandidateSpec("r2_mix_co_mx", "Duo Co-occ. + Grille + Froids 45%", 2, listOf(PredictionMethod.COOCCURRENCE_GRAPH to 0.5, PredictionMethod.MATRIX_9X10 to 0.5), 0.45f, 0.75, 2),
                SearchCandidateSpec("r2_mix_cx_ft", "Duo Croix + Fourier + Froids 45%", 2, listOf(PredictionMethod.CHANCE_CROIX_PYRAMID to 0.5, PredictionMethod.SPECTRAL_FOURIER to 0.5), 0.45f, 0.70, 2),
                SearchCandidateSpec("r2_mix_dir_ema", "Duo Dirichlet + EMA + Froids 60%", 2, listOf(PredictionMethod.DIRICHLET_MULTINOMIAL to 0.45, PredictionMethod.EMA_RECENCY to 0.55), 0.60f, 0.90, 2),
                SearchCandidateSpec("r2_ml_cold_50", "ML Bayésien + Froids 50%", 2, emptyList(), 0.50f, 0.70, 2, usesAdaptiveMl = true),
                SearchCandidateSpec("r2_cold_80_div", "Froid Extrême 80% + Coupe-Série", 2, listOf(PredictionMethod.COLD_HOT_BALANCE to 1.0), 0.80f, 1.15, 2)
            )
            3 -> listOf(
                SearchCandidateSpec("r3_tri_mk_hr_co", "Trio Markov/Horaire/Co-occ. + Froids", 3, listOf(PredictionMethod.MARKOV_STRATIFIED to 0.4, PredictionMethod.HORAIRE_OPTIM to 0.35, PredictionMethod.COOCCURRENCE_GRAPH to 0.25), 0.45f, 0.85, 2),
                SearchCandidateSpec("r3_tri_mx_cx_ft", "Trio Grille/Croix/Fourier + Froids", 3, listOf(PredictionMethod.MATRIX_9X10 to 0.35, PredictionMethod.CHANCE_CROIX_PYRAMID to 0.35, PredictionMethod.SPECTRAL_FOURIER to 0.30), 0.50f, 0.85, 2),
                SearchCandidateSpec("r3_tri_ema_mk_ln", "Trio EMA/Markov/Lunaire + Froids", 3, listOf(PredictionMethod.EMA_RECENCY to 0.4, PredictionMethod.MARKOV_STRATIFIED to 0.35, PredictionMethod.LUNAR_ASTRONOMICAL to 0.25), 0.50f, 0.95, 2),
                SearchCandidateSpec("r3_tri_dir_hr_mx", "Trio Dirichlet/Horaire/Grille + Froids", 3, listOf(PredictionMethod.DIRICHLET_MULTINOMIAL to 0.35, PredictionMethod.HORAIRE_OPTIM to 0.35, PredictionMethod.MATRIX_9X10 to 0.30), 0.55f, 1.00, 2),
                SearchCandidateSpec("r3_ml_high_div", "ML Bayésien Anti-Répétition Forte", 3, emptyList(), 0.35f, 1.05, 2, usesAdaptiveMl = true),
                SearchCandidateSpec("r3_cold_25_div12", "Chaud 75% / Froid 25% + Div. 1.2", 3, listOf(PredictionMethod.COLD_HOT_BALANCE to 1.0), 0.25f, 1.20, 2),
                SearchCandidateSpec("r3_cold_42_div11", "Équilibre 42% Froid + Div. 1.1", 3, listOf(PredictionMethod.COLD_HOT_BALANCE to 1.0), 0.42f, 1.10, 2),
                SearchCandidateSpec("r3_cold_58_div12", "Froid 58% + Div. 1.2 + Coupe-0", 3, listOf(PredictionMethod.COLD_HOT_BALANCE to 1.0), 0.58f, 1.20, 2),
                SearchCandidateSpec("r3_cold_75_div13", "Froid 75% + Div. 1.3 + Coupe-0", 3, listOf(PredictionMethod.COLD_HOT_BALANCE to 1.0), 0.75f, 1.30, 2),
                SearchCandidateSpec("r3_pure_cold_85", "Cycle Écarts Pur 85% Froid", 3, listOf(PredictionMethod.DIRICHLET_MULTINOMIAL to 1.0), 0.85f, 1.10, 2),
                SearchCandidateSpec("r3_mk_cold30_div", "Markov + Froids 30% + Div. 1.0", 3, listOf(PredictionMethod.MARKOV_STRATIFIED to 1.0), 0.30f, 1.00, 2),
                SearchCandidateSpec("r3_hr_cold30_div", "Horaire + Froids 30% + Div. 1.0", 3, listOf(PredictionMethod.HORAIRE_OPTIM to 1.0), 0.30f, 1.00, 2),
                SearchCandidateSpec("r3_co_cold60_div", "Co-occ. + Froids 60% + Div. 1.05", 3, listOf(PredictionMethod.COOCCURRENCE_GRAPH to 1.0), 0.60f, 1.05, 2),
                SearchCandidateSpec("r3_ft_cold60_div", "Fourier + Froids 60% + Div. 1.05", 3, listOf(PredictionMethod.SPECTRAL_FOURIER to 1.0), 0.60f, 1.05, 2),
                SearchCandidateSpec("r3_cube_cold50", "Cube90 + Injection Froids 50%", 3, listOf(PredictionMethod.CUBE90 to 1.0), 0.50f, 0.85, 2),
                SearchCandidateSpec("r3_quad_all", "Quatuor Équilibré + Froids 48%", 3, listOf(PredictionMethod.EMA_RECENCY to 0.25, PredictionMethod.MARKOV_STRATIFIED to 0.25, PredictionMethod.COOCCURRENCE_GRAPH to 0.25, PredictionMethod.HORAIRE_OPTIM to 0.25), 0.48f, 0.90, 2)
            )
            else -> {
                val rng = Random(round * 10_007L + 7919L)
                (0 until 16).map { idx ->
                    val kExperts = 2 + rng.nextInt(3)
                    val chosenExperts = coreMixableMethods.shuffled(rng).take(kExperts)
                    val rawWeights = DoubleArray(kExperts) { 0.2 + rng.nextDouble() }
                    val wSum = rawWeights.sum()
                    val mix = chosenExperts.mapIndexed { i, m -> m to (rawWeights[i] / wSum) }
                    val cold = (0.20 + rng.nextDouble() * 0.60).toFloat()
                    val div = 0.35 + rng.nextDouble() * 0.95
                    val streakThr = 2 + rng.nextInt(3)
                    val shortNames = chosenExperts.joinToString("+") { it.shortLabel.take(6) }
                    SearchCandidateSpec(
                        id = "r${round}_exp_$idx",
                        label = "R$round [$shortNames] F${(cold * 100).toInt()}%",
                        roundNumber = round,
                        expertMix = mix,
                        coldRatio = cold,
                        diversityLambda = div,
                        streakThreshold = streakThr
                    )
                }
            }
        }
    }

    fun evaluateAutoSearch(
        chronological: List<LotoResultEntity>,
        startIndex: Int,
        endIndexExclusive: Int,
        periodLabel: String,
        maxRounds: Int = 4,
        useSliding6DrawBlocks: Boolean = (periodLabel.contains("Jour") && !periodLabel.contains("Tout")) ||
            periodLabel.startsWith("1 Jour")
    ): BacktestResult {
        val safeStart = startIndex.coerceAtLeast(1)
        val safeEnd = endIndexExclusive.coerceAtMost(chronological.size)
        if (safeStart >= safeEnd) {
            return ScientificEvaluator.emptyBacktestResult(PredictionMethod.AUTO_SEARCH, periodLabel)
        }

        val safeMaxRounds = maxRounds.coerceIn(1, 10)
        val metaLearner = AdaptiveMetaLearner()
        val initialWeights = DoubleArray(baseExpertMethods.size) { 1.0 / baseExpertMethods.size }
        val weightSnapshots = mutableListOf<List<ExpertWeightInfo>>()

        // 1. Precalcul causal unique des experts de base + signaux froids a chaque pas t (O(T))
        val cachedSteps = ArrayList<StepPrecomputedData>(safeEnd - safeStart)
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
                "Violation causale AutoSearch : maxPast=$maxPastTs >= target=${targetDraw.timestamp}"
            }

            val dayKey = LotoDateUtils.formatDayKey(targetDraw.timestamp)
            val horaire = LotoDateUtils.normalizeHoraire(targetDraw)
            val game = LotoDateUtils.normalizeGame(targetDraw)
            val actualSet = actualNumbers.toSet()

            val baseCtx = DrawContext(
                targetTimestamp = targetDraw.timestamp,
                targetHoraire = horaire,
                targetGame = game,
                targetDayOfWeek = LotoDateUtils.dayOfWeek(targetDraw.timestamp)
            )

            val baseOutputs = HashMap<PredictionMethod, Pair<List<Int>, DoubleArray>>(baseExpertMethods.size)
            for (m in baseExpertMethods) {
                baseOutputs[m] = StatisticalStrategies.predictIndividual(m, strictPast, baseCtx)
            }
            val (coldSignals, gaps) = StatisticalStrategies.computeColdSignalVector(strictPast)
            val (_, mlProbs, expertProbsForUpdate) = metaLearner.predictWithPrecomputed(
                strictPast = strictPast,
                context = baseCtx,
                precomputed = baseOutputs
            )

            cachedSteps.add(
                StepPrecomputedData(
                    draw = targetDraw,
                    actualNumbers = actualNumbers,
                    actualSet = actualSet,
                    strictPastSize = strictPast.size,
                    maxPastTs = maxPastTs,
                    dayKey = dayKey,
                    horaire = horaire,
                    game = game,
                    baseProbs = baseOutputs.mapValues { it.value.second },
                    coldSignals = coldSignals,
                    gaps = gaps,
                    mlProbs = mlProbs
                )
            )

            metaLearner.updateAfterOutcome(expertProbsForUpdate, actualSet, horaire)
            if (cachedSteps.size % 20 == 0 || t == safeEnd - 1) {
                weightSnapshots.add(metaLearner.snapshotWeights(initialWeights = initialWeights))
            }
        }

        if (cachedSteps.isEmpty()) {
            return ScientificEvaluator.emptyBacktestResult(PredictionMethod.AUTO_SEARCH, periodLabel)
        }

        // 2. Generation et evaluation causale des rondes 1..maxRounds (avec arret de recherche au 1er succes 4j)
        val allSpecs = (1..safeMaxRounds).flatMap { r -> buildRoundCandidates(r) }
        val trackers = allSpecs.map { CandidateCausalTracker(it) }

        var exactMatches = 0
        var partialMatches = 0
        var twoPlusMatches = 0
        var threePlusMatches = 0
        var fourPlusMatches = 0
        var brierSum = 0.0
        var logLossSum = 0.0
        val distribution = IntArray(6)
        val matchList = ArrayList<Int>(cachedSteps.size)
        val stepResults = ArrayList<BacktestStepResult>(cachedSteps.size)
        val calibrationHistory = ArrayList<Pair<DoubleArray, Set<Int>>>(cachedSteps.size)

        var activeDayKey = ""
        val activeDayPredCounts = HashMap<Int, Int>()
        var activeZeroStreak = 0
        var activeMaxZeroStreak = 0

        val warmupSteps = minOf(10, maxOf(3, cachedSteps.size / 6))

        for ((idx, step) in cachedSteps.withIndex()) {
            if (step.dayKey != activeDayKey) {
                activeDayKey = step.dayKey
                activeDayPredCounts.clear()
            }

            val activeContext = DrawContext(
                targetTimestamp = step.draw.timestamp,
                targetHoraire = step.horaire,
                targetGame = step.game,
                targetDayOfWeek = LotoDateUtils.dayOfWeek(step.draw.timestamp),
                sameDayPredictedCounts = activeDayPredCounts,
                currentZeroMatchStreak = activeZeroStreak
            )

            // Choix causal du meilleur candidat base STRICTEMENT sur les pas < t
            val chosenTracker = if (idx < warmupSteps) {
                trackers.first()
            } else {
                val sortedByScore = trackers.sortedByDescending { it.causalUtilityScore() }
                val best = sortedByScore.first()
                if (activeZeroStreak >= 3 || best.currentZeroStreak >= 3) {
                    sortedByScore.firstOrNull { it.currentZeroStreak < 3 } ?: best
                } else {
                    best
                }
            }

            val (chosenPred, chosenProbs) = predictCandidateFromCache(
                spec = chosenTracker.spec,
                step = step,
                context = activeContext
            )

            check(chosenPred.size == 5 && chosenPred.distinct().size == 5 && chosenPred.all { it in 1..90 }) {
                "Prediction AutoSearch invalide: $chosenPred"
            }

            val matched = chosenPred.intersect(step.actualSet).sorted()
            val matchCount = matched.size.coerceIn(0, 5)
            val stepBrier = ScientificEvaluator.computeBrierScore(chosenProbs, step.actualSet)
            val stepLogLoss = ScientificEvaluator.computeBernoulliLogLoss(chosenProbs, step.actualSet)
            val expEstimate = chosenPred.sumOf { chosenProbs[it] }.toFloat()

            if (matchCount == 5) exactMatches++
            if (matchCount >= 1) partialMatches++
            if (matchCount >= 2) twoPlusMatches++
            if (matchCount >= 3) threePlusMatches++
            if (matchCount >= 4) fourPlusMatches++

            if (matchCount == 0) {
                activeZeroStreak++
                if (activeZeroStreak > activeMaxZeroStreak) activeMaxZeroStreak = activeZeroStreak
            } else {
                activeZeroStreak = 0
            }

            for (n in chosenPred) {
                activeDayPredCounts[n] = (activeDayPredCounts[n] ?: 0) + 1
            }

            distribution[matchCount]++
            matchList.add(matchCount)
            brierSum += stepBrier
            logLossSum += stepLogLoss
            calibrationHistory.add(chosenProbs to step.actualSet)

            val formattedDate = LotoDateUtils.formatFrenchDayAndDate(step.draw.timestamp)
            val label = "${step.draw.game} ${step.draw.time} • [${chosenTracker.spec.label.substringBefore(" +")}]".trim()

            stepResults.add(
                BacktestStepResult(
                    stepIndex = stepResults.size + 1,
                    drawId = step.draw.uniqueId,
                    drawLabel = label,
                    drawDate = formattedDate,
                    dayKey = step.dayKey,
                    horaire = step.horaire,
                    game = step.game,
                    trainSampleCount = step.strictPastSize,
                    maxTrainTimestamp = step.maxPastTs,
                    targetTimestamp = step.draw.timestamp,
                    causalVerified = step.maxPastTs < step.draw.timestamp,
                    predictedNumbers = chosenPred,
                    actualNumbers = step.actualNumbers,
                    matchedNumbers = matched,
                    matchCount = matchCount,
                    expectedMatchesEstimate = expEstimate,
                    stepBrierScore = stepBrier,
                    stepBernoulliLogLoss = stepLogLoss
                )
            )

            // Mise a jour causale de tous les candidats STRICTEMENT APRES revelation de S(t)
            for (tracker in trackers) {
                val cCtx = DrawContext(
                    targetTimestamp = step.draw.timestamp,
                    targetHoraire = step.horaire,
                    targetGame = step.game,
                    targetDayOfWeek = LotoDateUtils.dayOfWeek(step.draw.timestamp),
                    sameDayPredictedCounts = tracker.sameDayCountsFor(step.dayKey),
                    currentZeroMatchStreak = tracker.currentZeroStreak
                )
                val (cPred, cProbs) = predictCandidateFromCache(tracker.spec, step, cCtx)
                val cMatches = cPred.count { it in step.actualSet }
                val cBrier = ScientificEvaluator.computeBrierScore(cProbs, step.actualSet)
                tracker.recordPredictionAndOutcome(step.dayKey, cPred, cMatches, cBrier)
            }
        }

        val dailySummaries = ScientificEvaluator.buildDailySummaries(stepResults, useSliding6DrawBlocks)
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
            ScientificEvaluator.computeMaxConsecutiveQualifyingDays(dailySummaries)
        }
        val horaireBreakdown = ScientificEvaluator.buildHoraireBreakdown(stepResults)
        val (avgLo, avgHi) = ScientificEvaluator.mean95Interval(matchList)
        val (rateLo, rateHi) = ScientificEvaluator.wilson95Interval(partialMatches, stepResults.size)

        val winnerId = trackers.maxByOrNull { it.causalUtilityScore() }?.spec?.id
        val candidateSummaries = trackers
            .map { it.toSummary(isWinner = it.spec.id == winnerId) }
            .sortedWith(
                compareByDescending<AutoSearchCandidateSummary> { it.maxConsecutiveQualifyingDays }
                    .thenByDescending { it.qualifyingDaysCount }
                    .thenByDescending { it.twoPlusCount }
                    .thenBy { it.maxZeroMatchStreak }
                    .thenByDescending { it.averageMatches }
            )

        // Determine a quelle ronde l'objectif 4 jours d'affilee a ete atteint (sinon safeMaxRounds)
        val firstSuccessRound = (1..safeMaxRounds).firstOrNull { r ->
            trackers.any { it.spec.roundNumber <= r && it.maxConsecutiveQualDays >= 4 }
        }
        val roundsReported = firstSuccessRound ?: safeMaxRounds
        val objectiveMet = firstSuccessRound != null || maxConsecDays >= 4
        val strategiesReported = trackers.count { it.spec.roundNumber <= roundsReported }

        return BacktestResult(
            method = PredictionMethod.AUTO_SEARCH,
            successRate = partialMatches.toFloat() / stepResults.size,
            totalTested = stepResults.size,
            exactMatches = exactMatches,
            partialMatches = partialMatches,
            drawsWithTwoPlusMatches = twoPlusMatches,
            drawsWithThreePlusMatches = threePlusMatches,
            drawsWithFourPlusMatches = fourPlusMatches,
            qualifyingDaysCount = dailySummaries.count { it.isQualifyingDay },
            totalDaysTested = dailySummaries.size,
            averageMatches = matchList.sum().toFloat() / stepResults.size,
            avgMatchesCiLow = avgLo,
            avgMatchesCiHigh = avgHi,
            rateOnePlusCiLow = rateLo,
            rateOnePlusCiHigh = rateHi,
            averageBrierScore = brierSum / stepResults.size,
            averageBernoulliLogLoss = logLossSum / stepResults.size,
            calibrationErrorEce = ScientificEvaluator.computeCalibrationEce(calibrationHistory),
            matchDistribution = distribution.toList(),
            allStepsCausalVerified = stepResults.all { it.causalVerified },
            periodLabel = periodLabel,
            learnedExpertWeights = metaLearner.snapshotWeights(initialWeights = initialWeights),
            weightSnapshots = weightSnapshots,
            dailySummaries = dailySummaries,
            horaireBreakdown = horaireBreakdown,
            steps = stepResults.reversed(),
            maxConsecutiveQualifyingDays = maxConsecDays,
            maxZeroMatchStreak = activeMaxZeroStreak,
            currentZeroMatchStreak = activeZeroStreak,
            discoveredStrategiesSummary = candidateSummaries,
            searchRoundsExecuted = roundsReported,
            totalStrategiesSearched = strategiesReported,
            objectiveFourDaysAchieved = objectiveMet
        )
    }

    fun predictNextAutoSearch(
        cleanHistory: List<LotoResultEntity>,
        context: DrawContext,
        period: BacktestPeriod = BacktestPeriod.THREE_WEEKS,
        maxRounds: Int = 4
    ): AutoSearchNextPrediction {
        if (cleanHistory.size < 12) {
            val (pred, probs) = StatisticalStrategies.predictColdHotVariant(cleanHistory, context)
            return AutoSearchNextPrediction(
                predictedDraw = pred,
                marginalProbs = probs,
                discoveredLabel = "Chaud/Froid 50-50 (démarrage)",
                coldNumbersCount = StatisticalStrategies.countColdNumbers(pred, cleanHistory),
                diversityApplied = context.sameDayPredictedCounts.isNotEmpty(),
                expertWeights = emptyList(),
                roundsExecuted = 1,
                totalStrategiesSearched = 1,
                objectiveFourDaysAchieved = false,
                bestConsecutiveStreak = 0
            )
        }

        val searchStart = computePeriodStartIndex(cleanHistory, period)
        val searchBt = evaluateAutoSearch(
            chronological = cleanHistory,
            startIndex = searchStart,
            endIndexExclusive = cleanHistory.size,
            periodLabel = period.label,
            maxRounds = maxRounds
        )

        val winnerSummary = searchBt.discoveredStrategiesSummary.firstOrNull { it.isWinner }
            ?: searchBt.discoveredStrategiesSummary.firstOrNull()
        val allSpecs = (1..maxRounds.coerceIn(1, 10)).flatMap { buildRoundCandidates(it) }
        val winnerSpec = allSpecs.firstOrNull { it.id == winnerSummary?.id } ?: allSpecs.first()

        // Diversifie par rapport aux tirages deja observes aujourd'hui
        val todayKey = LotoDateUtils.formatDayKey(context.targetTimestamp)
        val todayCounts = HashMap<Int, Int>(context.sameDayPredictedCounts)
        for (d in cleanHistory.takeLast(12)) {
            if (LotoDateUtils.formatDayKey(d.timestamp) == todayKey) {
                for (n in StatisticalStrategies.extractDistinctBalls(d)) {
                    todayCounts[n] = (todayCounts[n] ?: 0) + 1
                }
            }
        }
        val enrichedContext = context.copy(
            sameDayPredictedCounts = todayCounts,
            currentZeroMatchStreak = searchBt.currentZeroMatchStreak
        )

        val (pred, probs) = predictWinnerOnLiveHistory(winnerSpec, cleanHistory, enrichedContext)
        val bestStreak = maxOf(
            searchBt.maxConsecutiveQualifyingDays,
            winnerSummary?.maxConsecutiveQualifyingDays ?: 0
        )

        return AutoSearchNextPrediction(
            predictedDraw = pred,
            marginalProbs = probs,
            discoveredLabel = winnerSpec.label,
            coldNumbersCount = StatisticalStrategies.countColdNumbers(pred, cleanHistory),
            diversityApplied = enrichedContext.sameDayPredictedCounts.isNotEmpty() && winnerSpec.diversityLambda > 0.0,
            expertWeights = searchBt.learnedExpertWeights,
            roundsExecuted = searchBt.searchRoundsExecuted,
            totalStrategiesSearched = searchBt.totalStrategiesSearched,
            objectiveFourDaysAchieved = searchBt.objectiveFourDaysAchieved || bestStreak >= 4,
            bestConsecutiveStreak = bestStreak
        )
    }

    fun computePeriodRange(
        chronological: List<LotoResultEntity>,
        period: BacktestPeriod,
        referenceNowMs: Long = System.currentTimeMillis(),
        customDays: Int? = period.days
    ): IntRange {
        val n = chronological.size
        if (n < 4) return 1 until n
        val effectiveDays = if (period == BacktestPeriod.ALL_HISTORY && customDays == null) null else (customDays ?: period.days)?.coerceAtLeast(1)
        val minWarmup = when {
            effectiveDays != null && effectiveDays < 7 -> minOf(5, maxOf(1, (n * 0.15).toInt()))
            n < 20 -> minOf(8, maxOf(2, (n * 0.25).toInt()))
            else -> minOf(20, maxOf(4, (n * 0.4).toInt()))
        }
        if (effectiveDays != null) {
            val endExclusive = (chronological.indexOfLast { it.timestamp <= referenceNowMs } + 1).takeIf { it >= 2 } ?: n
            val start = if (effectiveDays == 1) {
                (endExclusive - 6).coerceAtLeast(1).coerceAtMost(endExclusive - 2)
            } else {
                val anchorTs = chronological[endExclusive - 1].timestamp
                val cutoffMs = anchorTs - effectiveDays.toLong() * 86_400_000L
                val timeStart = chronological.subList(0, endExclusive)
                    .indexOfFirst { it.timestamp > cutoffMs }
                    .takeIf { it >= 1 } ?: 1
                val countStart = endExclusive - (effectiveDays * 6)
                maxOf(timeStart, countStart).coerceAtLeast(1).coerceAtMost(endExclusive - 2)
            }
            return start until endExclusive.coerceAtLeast(start + 1).coerceAtMost(n)
        }
        return minWarmup.coerceAtMost((n - 2).coerceAtLeast(1)) until n
    }

    fun computePeriodStartIndex(
        chronological: List<LotoResultEntity>,
        period: BacktestPeriod,
        referenceNowMs: Long = System.currentTimeMillis(),
        customDays: Int? = period.days
    ): Int {
        return computePeriodRange(chronological, period, referenceNowMs, customDays).first
    }

    private fun predictCandidateFromCache(
        spec: SearchCandidateSpec,
        step: StepPrecomputedData,
        context: DrawContext
    ): Pair<List<Int>, DoubleArray> {
        if (spec.usesAdaptiveMl) {
            return StatisticalStrategies.predictHybridWithColdSlots(
                coldSignals = step.coldSignals,
                gaps = step.gaps,
                hotBaseProbs = step.mlProbs,
                context = context,
                coldRatio = spec.coldRatio,
                diversityLambda = spec.diversityLambda,
                streakThreshold = spec.streakThreshold
            )
        }

        val blended = DoubleArray(91)
        for ((method, weight) in spec.expertMix) {
            val probs = step.baseProbs[method] ?: continue
            for (n in 1..90) {
                blended[n] += weight * probs[n]
            }
        }
        return StatisticalStrategies.predictHybridWithColdSlots(
            coldSignals = step.coldSignals,
            gaps = step.gaps,
            hotBaseProbs = blended,
            context = context,
            coldRatio = spec.coldRatio,
            diversityLambda = spec.diversityLambda,
            streakThreshold = spec.streakThreshold
        )
    }

    private fun predictWinnerOnLiveHistory(
        spec: SearchCandidateSpec,
        cleanHistory: List<LotoResultEntity>,
        context: DrawContext
    ): Pair<List<Int>, DoubleArray> {
        val (coldSignals, gaps) = StatisticalStrategies.computeColdSignalVector(cleanHistory)
        val hotProbs = if (spec.usesAdaptiveMl) {
            val learner = AdaptiveMetaLearner()
            val warm = minOf(30, maxOf(10, cleanHistory.size / 3))
            var warmStrictEnd = 0
            for (t in maxOf(warm, cleanHistory.size - 100) until cleanHistory.size) {
                val target = cleanHistory[t]
                val actual = StatisticalStrategies.extractDistinctBalls(target)
                if (actual.size < 5) continue
                while (warmStrictEnd < t && cleanHistory[warmStrictEnd].timestamp < target.timestamp) {
                    warmStrictEnd++
                }
                if (warmStrictEnd <= 0) continue
                val past = cleanHistory.subList(0, warmStrictEnd)
                val ctx = DrawContext(
                    targetTimestamp = target.timestamp,
                    targetHoraire = LotoDateUtils.normalizeHoraire(target),
                    targetGame = LotoDateUtils.normalizeGame(target),
                    targetDayOfWeek = LotoDateUtils.dayOfWeek(target.timestamp)
                )
                val (_, _, experts) = learner.predict(past, ctx)
                learner.updateAfterOutcome(experts, actual.toSet(), ctx.targetHoraire)
            }
            learner.predict(cleanHistory, context).second
        } else {
            val blended = DoubleArray(91)
            for ((method, weight) in spec.expertMix) {
                val probs = StatisticalStrategies.predictIndividual(method, cleanHistory, context).second
                for (n in 1..90) {
                    blended[n] += weight * probs[n]
                }
            }
            blended
        }

        return StatisticalStrategies.predictHybridWithColdSlots(
            coldSignals = coldSignals,
            gaps = gaps,
            hotBaseProbs = hotProbs,
            context = context,
            coldRatio = spec.coldRatio,
            diversityLambda = spec.diversityLambda,
            streakThreshold = spec.streakThreshold
        )
    }
}
