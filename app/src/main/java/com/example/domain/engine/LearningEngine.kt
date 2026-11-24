package com.example.domain.engine

import com.example.data.local.LotoResultEntity
import com.example.domain.util.LotoDateUtils

class LearningEngine {

    fun availableHoraires(history: List<LotoResultEntity>): List<String> {
        val standardOrder = listOf("00H", "8H", "11H", "14H", "18H", "20H", "21H")
        val found = history
            .map { LotoDateUtils.normalizeHoraire(it) }
            .filter { it.isNotBlank() }
            .distinct()
        return found.sortedWith(
            compareBy<String> {
                val idx = standardOrder.indexOf(it)
                if (idx >= 0) idx else Int.MAX_VALUE
            }.thenBy { it }
        )
    }

    fun predictNextFromHistory(
        history: List<LotoResultEntity>,
        method: PredictionMethod = PredictionMethod.AUTO_SEARCH,
        targetHoraire: String? = null,
        period: BacktestPeriod = BacktestPeriod.THREE_WEEKS,
        maxSearchRounds: Int = 4,
        referenceNowMs: Long = System.currentTimeMillis()
    ): PredictionResult {
        val cleanHistory = history
            .distinctBy { it.uniqueId }
            .sortedWith(compareBy<LotoResultEntity> { it.timestamp }.thenBy { it.uniqueId })

        val nextTarget = LotoDateUtils.resolveNextUpcomingDraw(cleanHistory, referenceNowMs)
        val resolvedHoraire = targetHoraire ?: nextTarget.horaire

        if (cleanHistory.isEmpty()) {
            return PredictionResult(
                predictedDraw = listOf(1, 2, 3, 4, 5),
                expectedMatchesEstimate = BacktestResult.RANDOM_EXPECTED_MATCHES,
                calibratedProbAtLeastOne = BacktestResult.RANDOM_AT_LEAST_ONE_RATE,
                trainedOnCount = 0,
                method = method,
                horaire = resolvedHoraire,
                marginalSumCheck = 5.0f
            )
        }

        val lastDraw = cleanHistory.last()
        val nextTs = if (targetHoraire == null) nextTarget.targetTimestamp else (lastDraw.timestamp + 3_600_000L)
        val inferredGame = cleanHistory
            .lastOrNull { LotoDateUtils.normalizeHoraire(it) == resolvedHoraire }
            ?.let { LotoDateUtils.normalizeGame(it) }
            ?: LotoDateUtils.normalizeGame(lastDraw)

        val todayKey = LotoDateUtils.formatDayKey(nextTs)
        val todayCounts = HashMap<Int, Int>()
        for (d in cleanHistory.takeLast(12)) {
            if (LotoDateUtils.formatDayKey(d.timestamp) == todayKey) {
                for (n in StatisticalStrategies.extractDistinctBalls(d)) {
                    todayCounts[n] = (todayCounts[n] ?: 0) + 1
                }
            }
        }

        val context = DrawContext(
            targetTimestamp = nextTs,
            targetHoraire = resolvedHoraire,
            targetGame = inferredGame,
            targetDayOfWeek = LotoDateUtils.dayOfWeek(nextTs),
            sameDayPredictedCounts = todayCounts
        )

        var learnedWeights = emptyList<ExpertWeightInfo>()
        var discoveredName = ""
        var diversityApplied = false
        var roundsExecuted = 0
        var totalSearched = 0
        var objectiveAchieved = false
        var bestStreak = 0

        val (predicted, marginalProbs) = when (method) {
            PredictionMethod.AUTO_SEARCH -> {
                val autoRes = AutoSearchEngine.predictNextAutoSearch(
                    cleanHistory = cleanHistory,
                    context = context,
                    period = period,
                    maxRounds = maxSearchRounds
                )
                learnedWeights = autoRes.expertWeights
                discoveredName = autoRes.discoveredLabel
                diversityApplied = autoRes.diversityApplied
                roundsExecuted = autoRes.roundsExecuted
                totalSearched = autoRes.totalStrategiesSearched
                objectiveAchieved = autoRes.objectiveFourDaysAchieved
                bestStreak = autoRes.bestConsecutiveStreak
                autoRes.predictedDraw to autoRes.marginalProbs
            }
            PredictionMethod.ADAPTIVE_ML -> {
                val learner = AdaptiveMetaLearner()
                val warmup = minOf(30, maxOf(10, cleanHistory.size / 3))
                val trainStart = maxOf(warmup, cleanHistory.size - 120)
                var warmStrictEnd = 0
                for (t in trainStart until cleanHistory.size) {
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
                val (top5, probs, _) = learner.predict(cleanHistory, context)
                learnedWeights = learner.snapshotWeights(horaire = resolvedHoraire)
                diversityApplied = todayCounts.isNotEmpty()
                top5 to probs
            }
            else -> {
                diversityApplied = method == PredictionMethod.COLD_HOT_BALANCE && todayCounts.isNotEmpty()
                StatisticalStrategies.predictIndividual(method, cleanHistory, context)
            }
        }

        val marginalSum = (1..90).sumOf { marginalProbs[it] }.toFloat()
        val expectedMatches = predicted.sumOf { marginalProbs[it] }.toFloat()
        val probAtLeastOne = ScientificEvaluator.calibratedProbAtLeastOne(predicted, marginalProbs)
        val coldCount = StatisticalStrategies.countColdNumbers(predicted, cleanHistory)

        val auditNote = when (method) {
            PredictionMethod.CUBE90 ->
                "Attention : Cube90 n'est pas bijective (30 images possibles, 60 numéros exclus)."
            PredictionMethod.AUTO_SEARCH ->
                "Méthode retenue (${period.label}) : $discoveredName • $totalSearched stratégies ($roundsExecuted r.)"
            else ->
                "Estimation calibrée ($coldCount/5 froids • somme marginales = ${String.format(java.util.Locale.US, "%.2f", marginalSum)})."
        }

        return PredictionResult(
            predictedDraw = predicted,
            expectedMatchesEstimate = expectedMatches,
            calibratedProbAtLeastOne = probAtLeastOne,
            trainedOnCount = cleanHistory.size,
            method = method,
            horaire = resolvedHoraire,
            marginalSumCheck = marginalSum,
            expertWeights = learnedWeights,
            auditNote = auditNote,
            discoveredStrategyName = discoveredName,
            coldNumbersCount = coldCount,
            diversityPenaltyApplied = diversityApplied,
            searchRoundsExecuted = roundsExecuted,
            totalStrategiesSearched = totalSearched,
            objectiveFourDaysAchieved = objectiveAchieved,
            bestCausalStreakFound = bestStreak
        )
    }

    fun backtest(
        data: List<LotoResultEntity>,
        method: PredictionMethod = PredictionMethod.AUTO_SEARCH,
        period: BacktestPeriod = BacktestPeriod.THREE_WEEKS,
        maxSearchRounds: Int = 4,
        referenceNowMs: Long = System.currentTimeMillis(),
        customDays: Int? = period.days
    ): BacktestResult {
        val chronological = data
            .distinctBy { it.uniqueId }
            .sortedWith(compareBy<LotoResultEntity> { it.timestamp }.thenBy { it.uniqueId })

        val range = AutoSearchEngine.computePeriodRange(
            chronological = chronological,
            period = period,
            referenceNowMs = referenceNowMs,
            customDays = customDays
        )
        val splitIndex = range.first
        val endIndexExclusive = (range.last + 1).coerceAtMost(chronological.size)
        val activeLabel = customDays?.takeIf { it != period.days }?.let { "$it jours" } ?: period.label

        val label = if (chronological.size >= 2 && splitIndex < endIndexExclusive) {
            "$activeLabel : ${LotoDateUtils.formatShortDate(chronological[splitIndex].timestamp)} → ${LotoDateUtils.formatShortDate(chronological[endIndexExclusive - 1].timestamp)}"
        } else {
            activeLabel
        }

        return ScientificEvaluator.evaluateStrategyOnRange(
            chronological = chronological,
            method = method,
            startIndex = splitIndex,
            endIndexExclusive = endIndexExclusive,
            periodLabel = label,
            maxSearchRounds = maxSearchRounds
        )
    }

    fun runScientificComparison(
        data: List<LotoResultEntity>,
        period: BacktestPeriod = BacktestPeriod.THREE_WEEKS,
        customDays: Int? = period.days,
        referenceNowMs: Long = System.currentTimeMillis(),
        savedModels: List<LabSavedModelInput> = emptyList()
    ): ScientificReport {
        return ScientificEvaluator.runFullScientificComparison(
            data = data,
            period = period,
            referenceNowMs = referenceNowMs,
            customDays = customDays,
            savedModels = savedModels
        )
    }
}
