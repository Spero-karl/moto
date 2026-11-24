package com.example.domain.engine

import com.example.data.local.LotoDao
import com.example.data.local.LotoResultEntity
import com.example.data.repository.ArchiveImporter
import com.example.data.repository.LotoRepository
import com.example.domain.util.LotoDateUtils
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOf
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.text.SimpleDateFormat
import java.util.Locale

class ScientificAuditTest {

    private class FakeDao : LotoDao {
        override fun getAllResults(): Flow<List<LotoResultEntity>> = flowOf(emptyList())
        override suspend fun getAllResultsOnce(): List<LotoResultEntity> = emptyList()
        override suspend fun insertAll(results: List<LotoResultEntity>) {}
        override suspend fun getCount(): Int = 0
    }

    private fun buildDatedDraws(startDateStr: String, count: Int): List<LotoResultEntity> {
        val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).apply {
            timeZone = LotoDateUtils.beninTz
        }
        val startMs = sdf.parse("$startDateStr 08:00")!!.time
        val horaires = listOf("8H", "11H", "14H", "18H")
        return (0 until count).map { i ->
            val ts = startMs + i * 6L * 3_600_000L
            val b = (i * 11 % 80) + 1
            LotoResultEntity(
                uniqueId = "d_$i",
                date = LotoDateUtils.formatFrenchDayAndDate(ts),
                time = horaires[i % horaires.size],
                game = if (i % 3 == 0) "Digital" else "Fortune",
                num1 = b,
                num2 = ((b + 7) % 90) + 1,
                num3 = ((b + 13) % 90) + 1,
                num4 = ((b + 23) % 90) + 1,
                num5 = ((b + 37) % 90) + 1,
                timestamp = ts
            )
        }
    }

    @Test
    fun marginalProbabilities_alwaysSumToFiveAcrossAllStrategies() {
        val draws = buildDatedDraws("2026-08-01", 60)
        val ctx = DrawContext(
            targetTimestamp = draws.last().timestamp + 3_600_000L,
            targetHoraire = "18H",
            targetGame = "Fortune",
            targetDayOfWeek = LotoDateUtils.dayOfWeek(draws.last().timestamp + 3_600_000L)
        )

        for (method in PredictionMethod.entries) {
            if (method == PredictionMethod.ADAPTIVE_ML || method == PredictionMethod.AUTO_SEARCH) continue
            val (pred, probs) = StatisticalStrategies.predictIndividual(method, draws, ctx)
            assertEquals(5, pred.distinct().size)
            assertTrue(pred.all { it in 1..90 })

            val sum = (1..90).sumOf { probs[it] }
            assertEquals("Somme des probabilites marginales != 5.0 pour $method", 5.0, sum, 1e-4)
            assertTrue((1..90).all { probs[it] > 0.0 && probs[it] < 1.0 })

            val pAtLeastOne = ScientificEvaluator.calibratedProbAtLeastOne(pred, probs)
            assertTrue("P(>=1) doit etre dans (0,1) pour $method", pAtLeastOne in 0.01f..0.99f)
        }

        val engine = LearningEngine()
        for (metaMethod in listOf(PredictionMethod.ADAPTIVE_ML, PredictionMethod.AUTO_SEARCH)) {
            val res = engine.predictNextFromHistory(draws, method = metaMethod, targetHoraire = "18H")
            assertEquals(5, res.predictedDraw.distinct().size)
            assertEquals(5.0f, res.marginalSumCheck, 1e-3f)
            assertTrue(res.calibratedProbAtLeastOne in 0.01f..0.99f)
        }
    }

    @Test
    fun intraDayDiversification_preventsRepeatingExactSameNumbersAllDay() {
        val draws = buildDatedDraws("2026-10-01", 80)
        val bt = LearningEngine().backtest(draws, method = PredictionMethod.COLD_HOT_BALANCE)
        assertTrue(bt.totalTested > 8)

        val byDay = bt.steps.groupBy { it.dayKey }.values.filter { it.size >= 3 }
        assertTrue("Doit avoir des journees avec >= 3 tirages", byDay.isNotEmpty())
        val diversifiedDays = byDay.count { daySteps ->
            daySteps.map { it.predictedNumbers }.distinct().size > 1
        }
        assertTrue("Les predictions d'une meme journee doivent se diversifier", diversifiedDays == byDay.size)

        val pred = LearningEngine().predictNextFromHistory(draws, method = PredictionMethod.COLD_HOT_BALANCE)
        assertTrue("Chaud/Froid doit inclure au moins 1 numero froid", pred.coldNumbersCount >= 1)
    }

    @Test
    fun autoSearch_discoversCandidatesAndTracksConsecutiveQualifyingDays() {
        val draws = buildDatedDraws("2026-09-01", 90)
        val bt = LearningEngine().backtest(draws, method = PredictionMethod.AUTO_SEARCH, maxSearchRounds = 4)
        assertTrue(bt.allStepsCausalVerified)
        assertEquals(64, bt.discoveredStrategiesSummary.size)
        assertEquals(1, bt.discoveredStrategiesSummary.count { it.isWinner })
        assertTrue(bt.searchRoundsExecuted in 1..4)
        assertTrue(bt.maxConsecutiveQualifyingDays >= 0)
        assertTrue(bt.maxZeroMatchStreak >= 0)

        // Verification directe du calcul de serie consecutive de jours qualifiants
        val fakeDays = listOf(
            DailyBacktestSummary("2026-10-01", "Jeudi 01/10/2026", 4, 2, 2, true),
            DailyBacktestSummary("2026-10-02", "Vendredi 02/10/2026", 4, 2, 3, true),
            DailyBacktestSummary("2026-10-03", "Samedi 03/10/2026", 4, 3, 2, true),
            DailyBacktestSummary("2026-10-04", "Dimanche 04/10/2026", 4, 2, 2, true),
            DailyBacktestSummary("2026-10-05", "Lundi 05/10/2026", 4, 0, 1, false)
        )
        assertEquals(4, ScientificEvaluator.computeMaxConsecutiveQualifyingDays(fakeDays))
    }

    @Test
    fun oneMonthPeriod_restrictsBacktestToLastThirtyDaysUpToLatestDraw() {
        // 4 tirages/jour sur ~65 jours = 260 tirages
        val draws = buildDatedDraws("2026-09-10", 260)
        val btMonth = LearningEngine().backtest(
            data = draws,
            method = PredictionMethod.AUTO_SEARCH,
            period = BacktestPeriod.ONE_MONTH
        )
        val btAll = LearningEngine().backtest(
            data = draws,
            method = PredictionMethod.AUTO_SEARCH,
            period = BacktestPeriod.ALL_HISTORY
        )
        assertTrue("1 mois doit tester moins de tirages que tout l'historique", btMonth.totalTested < btAll.totalTested)
        val latestTs = draws.last().timestamp
        val earliestTestedTs = btMonth.steps.minOf { it.targetTimestamp }
        val spanDays = (latestTs - earliestTestedTs) / 86_400_000.0
        assertTrue("Fenetre 1 mois doit rester <= 31 jours (observe=$spanDays)", spanDays <= 31.0)
    }

    @Test
    fun monteCarloBaseline_isReproducibleAndMatchesTheoreticalExpectation() {
        val draws = buildDatedDraws("2026-10-01", 100)
        val mc1 = ScientificEvaluator.runMonteCarloBaseline(draws, simulations = 10_000, seed = 42L)
        val mc2 = ScientificEvaluator.runMonteCarloBaseline(draws, simulations = 10_000, seed = 42L)

        assertEquals(mc1.meanAvgMatches, mc2.meanAvgMatches, 1e-6f)
        assertEquals(mc1.meanTwoPlusCount, mc2.meanTwoPlusCount, 1e-6f)
        assertEquals(mc1.qualifyingDaysCiLow, mc2.qualifyingDaysCiLow)
        assertEquals(mc1.qualifyingDaysCiHigh, mc2.qualifyingDaysCiHigh)

        // Theorie 5/90 : esperance = 25/90 = 0.2778
        assertEquals(25f / 90f, mc1.meanAvgMatches, 0.01f)
        assertEquals(0.2518f, mc1.meanOnePlusRate, 0.01f)
    }

    @Test
    fun chronologicalSplit_neverTreatsOctober18To31AsIndependentFinalTest() {
        // 4 tirages par jour du 2026-08-01 au 2026-11-15 (~420 tirages)
        val draws = buildDatedDraws("2026-08-01", 430)
        val (devRange, valRange, testRange) = ScientificEvaluator.computeChronologicalSplit(draws)

        assertTrue(!devRange.isEmpty() && !valRange.isEmpty() && !testRange.isEmpty())
        assertTrue(devRange.last < valRange.first)
        assertTrue(valRange.last < testRange.first)

        // Aucun tirage d'octobre ne doit apparaitre dans le Test Final Independant
        for (idx in testRange) {
            val dayKey = LotoDateUtils.formatDayKey(draws[idx].timestamp)
            assertTrue("Tirage $dayKey ne doit pas etre dans le test independant", dayKey >= "2026-11-01")
        }
    }

    @Test
    fun repositoryValidation_rejectsMissingDateAndDuplicateBalls() {
        val repo = LotoRepository(FakeDao())

        // 1. Date manquante -> DOIT etre rejete (jamais de repli sur System.currentTimeMillis)
        val missingDateJson = Json.parseToJsonElement(
            """{"id":100,"name":"Fortune 18H","type":"Fortune","drawResults":"1,2,3,4,5"}"""
        ).jsonObject
        assertNull(repo.parseV3Draw(missingDateJson))

        // 2. Doublon dans un meme tirage -> DOIT etre rejete
        val duplicateBallJson = Json.parseToJsonElement(
            """{"id":101,"closureDate":1794330000000,"name":"Fortune 18H","type":"Fortune","drawResults":"12,12,34,56,78"}"""
        ).jsonObject
        assertNull(repo.parseV3Draw(duplicateBallJson))

        // 3. Numero hors bornes (91) -> DOIT etre rejete
        val outOfRangeJson = Json.parseToJsonElement(
            """{"id":102,"closureDate":1794330000000,"name":"Fortune 18H","type":"Fortune","drawResults":"12,15,34,56,91"}"""
        ).jsonObject
        assertNull(repo.parseV3Draw(outOfRangeJson))

        // 4. Tirage valide -> accepte avec date formatee avec nom du jour
        val validJson = Json.parseToJsonElement(
            """{"id":103,"closureDate":1794330000000,"name":"Fortune 18H","type":"Fortune","drawResults":"57,83,65,89,13"}"""
        ).jsonObject
        val parsed = repo.parseV3Draw(validJson)
        assertNotNull(parsed)
        assertEquals("lnb_103", parsed!!.uniqueId)
    }

    @Test
    fun archiveImporter_validatesCsvAndJsonAndDeduplicates() {
        val csv = """
            date;game;time;n1;n2;n3;n4;n5
            10/11/2026;Fortune;18H;5;12;33;44;89
            10/11/2026;Fortune;18H;5;12;33;44;89
            bad-date;Fortune;14H;1;2;3;4;5
            11/11/2026;Star;11H;1;1;3;4;5
        """.trimIndent()

        val res = ArchiveImporter.parseAndValidate(csv, "historique.csv")
        assertEquals(4, res.totalRows)
        assertEquals(1, res.acceptedCount) // 1 valide unique (1 doublon dedup + 2 invalides rejetes)
        assertEquals(3, res.rejectedCount)

        val jsonArchive = """
            [
              {"closureDate": 1794330000000, "name": "Fortune 18H", "type": "Fortune", "drawResults": "10,20,30,40,50"},
              {"closureDate": 1794416400000, "name": "Star 14H", "type": "Star", "drawResults": "1,2,3,4,99"}
            ]
        """.trimIndent()
        val jsonRes = ArchiveImporter.parseAndValidate(jsonArchive, "backup.json")
        assertEquals(2, jsonRes.totalRows)
        assertEquals(1, jsonRes.acceptedCount)
    }

    @Test
    fun dailyQualificationStrictlyRequiresThreeDraws_andChercheurEmitsThreeVariants() {
        val steps = listOf(
            BacktestStepResult(1, "d1_1", "F 8H", "Lundi 01/10", "2026-10-01", "8H", "Fortune", 10, 1L, 2L, true, listOf(1,2,3,4,5), listOf(1,2,10,11,12), listOf(1,2), 2, 0.3f, 0.05, 0.2),
            BacktestStepResult(2, "d1_2", "F 11H", "Lundi 01/10", "2026-10-01", "11H", "Fortune", 11, 2L, 3L, true, listOf(1,2,3,4,5), listOf(1,2,20,21,22), listOf(1,2), 2, 0.3f, 0.05, 0.2),
            BacktestStepResult(3, "d1_3", "F 14H", "Lundi 01/10", "2026-10-01", "14H", "Fortune", 12, 3L, 4L, true, listOf(1,2,3,4,5), listOf(1,30,31,32,33), listOf(1), 1, 0.3f, 0.05, 0.2),
            BacktestStepResult(4, "d2_1", "F 8H", "Mardi 02/10", "2026-10-02", "8H", "Fortune", 13, 4L, 5L, true, listOf(1,2,3,4,5), listOf(1,2,50,51,52), listOf(1,2), 2, 0.3f, 0.05, 0.2),
            BacktestStepResult(5, "d2_2", "F 11H", "Mardi 02/10", "2026-10-02", "11H", "Fortune", 14, 5L, 6L, true, listOf(1,2,3,4,5), listOf(2,3,60,61,62), listOf(2,3), 2, 0.3f, 0.05, 0.2),
            BacktestStepResult(6, "d2_3", "F 14H", "Mardi 02/10", "2026-10-02", "14H", "Fortune", 15, 6L, 7L, true, listOf(1,2,3,4,5), listOf(3,4,70,71,72), listOf(3,4), 2, 0.3f, 0.05, 0.2)
        )
        val summaries = ScientificEvaluator.buildDailySummaries(steps)
        val day1 = summaries.first { it.dayKey == "2026-10-01" }
        val day2 = summaries.first { it.dayKey == "2026-10-02" }
        assertEquals(2, day1.drawsWithTwoPlus)
        org.junit.Assert.assertFalse("2 tirages >=2 ne doit pas qualifier", day1.isQualifyingDay)
        assertEquals(3, day2.drawsWithTwoPlus)
        assertTrue("3 tirages >=2 doit qualifier", day2.isQualifyingDay)

        val draws = buildDatedDraws("2026-10-01", 40)
        val cache = DiscoveryEngine.buildCausalCache(draws, BacktestPeriod.THREE_WEEKS, referenceNowMs = draws.last().timestamp + 1_000L)
        val cycle = ChercheurEngine.runResearchAndDistill(cache, cycleIndex = 1, generation = 1, seenSignatures = emptySet())
        assertTrue(cycle.teachings.isNotEmpty())
        assertTrue(cycle.synthesizedGenomes.size >= 3)
    }

    @Test
    fun oneDayPeriod_andIntraDayTenThirtyToElevenThirtyChainWorkCausallyWithoutCheat() {
        val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).apply {
            timeZone = LotoDateUtils.beninTz
        }
        val slots = listOf("8H" to "08:00", "11H" to "11:00", "14H" to "14:00", "18H" to "18:00", "20H" to "20:00")
        val draws = ArrayList<LotoResultEntity>()
        var seedVal = 7
        for (day in 1..4) {
            val dayStr = String.format(Locale.US, "2026-10-%02d", day)
            for ((hLabel, hhmm) in slots) {
                val ts = sdf.parse("$dayStr $hhmm")!!.time
                val nums = (0 until 5).map { idx -> ((seedVal + idx * 17) % 90) + 1 }.distinct().let { list ->
                    if (list.size == 5) list else (1..5).toList()
                }.sorted()
                seedVal += 13
                draws.add(
                    LotoResultEntity(
                        uniqueId = "d_${dayStr}_$hLabel",
                        date = LotoDateUtils.formatFrenchDayAndDate(ts),
                        time = hLabel,
                        game = "Fortune",
                        num1 = nums[0], num2 = nums[1], num3 = nums[2], num4 = nums[3], num5 = nums[4],
                        timestamp = ts
                    )
                )
            }
        }
        val day5 = "2026-10-05"
        val ts8H = sdf.parse("$day5 08:00")!!.time
        val draw8H = LotoResultEntity(
            uniqueId = "d_${day5}_8H",
            date = LotoDateUtils.formatFrenchDayAndDate(ts8H),
            time = "8H",
            game = "Fortune",
            num1 = 3, num2 = 19, num3 = 35, num4 = 51, num5 = 78,
            timestamp = ts8H
        )
        val ts11H = sdf.parse("$day5 11:00")!!.time
        val draw11H = LotoResultEntity(
            uniqueId = "d_${day5}_11H",
            date = LotoDateUtils.formatFrenchDayAndDate(ts11H),
            time = "11H",
            game = "Fortune",
            num1 = 8, num2 = 24, num3 = 44, num4 = 61, num5 = 88,
            timestamp = ts11H
        )

        val at1030 = sdf.parse("$day5 10:30")!!.time
        val at1130 = sdf.parse("$day5 11:30")!!.time
        val historyAt1030 = draws + draw8H
        val historyAt1130 = historyAt1030 + draw11H

        // 1. A 10h30 : cible 11H, fenetre 1 Jour causale, backtest + prediction 11H
        val target1030 = LotoDateUtils.resolveNextUpcomingDraw(historyAt1030, at1030)
        assertEquals("11H", target1030.horaire)
        val cache1030 = DiscoveryEngine.buildCausalCache(
            history = historyAt1030,
            period = BacktestPeriod.ONE_DAY,
            referenceNowMs = at1030,
            customDays = 1
        )
        assertEquals("1 Jour", cache1030.periodLabel)
        assertTrue(cache1030.steps.isNotEmpty())
        assertTrue(cache1030.steps.all { it.maxPastTs < it.draw.timestamp })

        val genome = SymbolicGenome(
            id = "g_1d_chain",
            generation = 2,
            parentIds = emptyList(),
            primaryExpr = ExprNode.Binary(
                BinaryOp.ADD,
                ExprNode.Var(RawFeature.SAME_DAY_REP),
                ExprNode.Var(RawFeature.GAP_NORM)
            ),
            secondaryExpr = ExprNode.Var(RawFeature.FREQ_W15),
            primarySlots = 3,
            samplingTemp = 0.65,
            exploreNoise = 0.0,
            seed = 1130L
        )

        val (bt1030, ledger1030) = DiscoveryEngine.mergeAppendOnlyApprovedBacktest(
            genome = genome,
            cache = cache1030,
            existingLedger = null,
            generationsExecuted = 2,
            experimentsCount = 48
        )
        val pred11H = DiscoveryEngine.predictNextWithGenome(
            genome = genome,
            cleanHistory = historyAt1030,
            targetHoraire = target1030.horaire,
            validatedBacktest = bt1030,
            targetTimestamp = target1030.targetTimestamp
        )
        val ledgerWith11HPending = DiscoveryEngine.recordLivePredictionInLedger(
            ledger = ledger1030,
            prediction = pred11H,
            targetTimestamp = target1030.targetTimestamp
        )
        assertEquals(5, pred11H.predictedDraw.distinct().size)

        // 2. A 11h30 : le tirage 11H est arrive -> cible 14H, etape 8H verrouillee, 11H reconciliee exactement, prediction 14H
        val target1130 = LotoDateUtils.resolveNextUpcomingDraw(historyAt1130, at1130)
        assertEquals("14H", target1130.horaire)
        val cache1130 = DiscoveryEngine.buildCausalCache(
            history = historyAt1130,
            period = BacktestPeriod.ONE_DAY,
            referenceNowMs = at1130,
            customDays = 1
        )
        val (bt1130, _) = DiscoveryEngine.mergeAppendOnlyApprovedBacktest(
            genome = genome,
            cache = cache1130,
            existingLedger = ledgerWith11HPending,
            generationsExecuted = 2,
            experimentsCount = 48
        )
        assertEquals(bt1030.steps.size, bt1130.approvedStepsPreservedCount)
        assertEquals(1, bt1130.newlyAppendedStepsCount)
        assertEquals(1, bt1130.livePredictionsMatchedCount)
        assertEquals(pred11H.predictedDraw.sorted(), bt1130.steps.first().predictedNumbers.sorted())
        assertEquals(bt1030.steps, bt1130.steps.takeLast(bt1030.steps.size))

        val pred14H = DiscoveryEngine.predictNextWithGenome(
            genome = genome,
            cleanHistory = historyAt1130,
            targetHoraire = target1130.horaire,
            validatedBacktest = bt1130,
            targetTimestamp = target1130.targetTimestamp
        )
        assertEquals("14H", pred14H.horaire)
        assertEquals(5, pred14H.predictedDraw.distinct().size)

        // 3. Evaluation scientifique 1 Jour sans crash
        val sci1d = ScientificEvaluator.runFullScientificComparison(
            data = historyAt1130,
            period = BacktestPeriod.ONE_DAY,
            referenceNowMs = at1130,
            customDays = 1
        )
        assertTrue(sci1d.indepTestComparisons.isNotEmpty())
    }
}
