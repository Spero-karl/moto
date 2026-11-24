package com.example.domain.engine

import com.example.data.local.LotoResultEntity
import com.example.domain.util.LotoDateUtils
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.text.SimpleDateFormat
import java.util.Locale

class PrecisionAndOneDayAuditTest {

    @Test
    fun oneDayWindowCoversSixDrawsZeroToTwentyHAcrossMidnight_andIsDeterministic_android() {
        val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).apply {
            timeZone = LotoDateUtils.beninTz
        }
        val slots = listOf(
            "00H" to "00:05",
            "8H" to "08:00",
            "11H" to "11:00",
            "14H" to "14:00",
            "18H" to "18:00",
            "20H" to "20:00"
        )
        val draws = ArrayList<LotoResultEntity>()
        var idx = 0
        for (day in 1..4) {
            val dayStr = String.format(Locale.US, "2026-10-%02d", day)
            for ((hLabel, hhmm) in slots) {
                val ts = sdf.parse("$dayStr $hhmm")!!.time
                val nums = listOf(3, 8, 15 + (idx % 10), 37 + (idx % 14), 68 + (idx % 20)).sorted()
                draws.add(
                    LotoResultEntity(
                        uniqueId = "a_six_${dayStr}_$hLabel",
                        date = LotoDateUtils.formatFrenchDayAndDate(ts),
                        time = hLabel,
                        game = "Fortune",
                        num1 = nums[0], num2 = nums[1], num3 = nums[2], num4 = nums[3], num5 = nums[4],
                        timestamp = ts
                    )
                )
                idx++
            }
        }
        val ref1130 = sdf.parse("2026-10-04 11:30")!!.time
        val historyBefore1130 = draws.filter { it.timestamp < ref1130 }

        val cache1d = DiscoveryEngine.buildCausalCache(
            history = historyBefore1130,
            period = BacktestPeriod.ONE_DAY,
            referenceNowMs = ref1130,
            customDays = 1
        )
        assertEquals(6, cache1d.steps.size)
        assertEquals(listOf("14H", "18H", "20H", "00H", "8H", "11H"), cache1d.steps.map { it.horaire })
        assertEquals(1, cache1d.steps.map { it.dayEpoch }.distinct().size)
        assertEquals(2, cache1d.discoverySteps.size)
        assertEquals(1, cache1d.rollingSteps.size)
        assertEquals(2, cache1d.confirmationSteps.size)
        assertEquals(1, cache1d.sealedSteps.size)
        assertTrue(cache1d.steps.all { it.maxPastTs < it.draw.timestamp })

        val initState = DiscoveryEngine.createInitialState(sessionSalt = 404L)
        assertTrue(initState.population.all { it.exploreNoise == 0.0 })

        val gen1 = DiscoveryEngine.executeEvolutionGeneration(
            state = initState,
            cache = cache1d,
            periodName = BacktestPeriod.ONE_DAY.name,
            minObjectiveStreak = 1
        )
        assertTrue(gen1.nextState.population.all { it.exploreNoise == 0.0 })

        val best = (gen1.winningEval ?: gen1.bestTrainEval)?.genome ?: initState.population.first()
        val (bt1d, _) = DiscoveryEngine.mergeAppendOnlyApprovedBacktest(best, cache1d, null, 1, 24)
        val nextTarget = LotoDateUtils.resolveNextUpcomingDraw(historyBefore1130, ref1130)
        assertEquals("14H", nextTarget.horaire)

        val predA = DiscoveryEngine.predictNextWithGenome(best, historyBefore1130, nextTarget.horaire, bt1d, nextTarget.targetTimestamp)
        val predB = DiscoveryEngine.predictNextWithGenome(best, historyBefore1130, nextTarget.horaire, bt1d, nextTarget.targetTimestamp)
        assertEquals(predA.predictedDraw, predB.predictedDraw)
    }

    @Test
    fun precisionTargetEngine_android_runsIndependentlyAndPersistsLedger() {
        val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).apply {
            timeZone = LotoDateUtils.beninTz
        }
        val slots = listOf("00H" to "00:05", "8H" to "08:00", "11H" to "11:00", "14H" to "14:00", "18H" to "18:00", "20H" to "20:00")
        val draws = ArrayList<LotoResultEntity>()
        for (day in 1..6) {
            val dayStr = String.format(Locale.US, "2026-10-%02d", day)
            for ((sIdx, pair) in slots.withIndex()) {
                val (hLabel, hhmm) = pair
                val ts = sdf.parse("$dayStr $hhmm")!!.time
                val p1 = if ((day + sIdx) % 2 == 0) 9 else 16
                val p2 = if (p1 == 9) 24 else 35
                draws.add(
                    LotoResultEntity(
                        uniqueId = "aprec_${dayStr}_$hLabel",
                        date = LotoDateUtils.formatFrenchDayAndDate(ts),
                        time = hLabel,
                        game = "Fortune",
                        num1 = p1, num2 = p2, num3 = 48, num4 = 66, num5 = 82,
                        timestamp = ts
                    )
                )
            }
        }
        val tempDir = java.nio.file.Files.createTempDirectory("android_precision_test").toFile()
        try {
            val initialHistory = draws.dropLast(1)
            val lastDraw = draws.last()
            for (mode in PrecisionTargetMode.entries) {
                val cache = PrecisionTargetEngine.buildCausalCache(
                    history = initialHistory,
                    period = BacktestPeriod.THREE_DAYS,
                    referenceNowMs = lastDraw.timestamp - 60_000L,
                    customDays = 3
                )
                assertTrue(cache.steps.isNotEmpty())
                assertTrue(cache.steps.all { it.maxPastTs < it.draw.timestamp })

                val initState = PrecisionTargetEngine.createInitialState(
                    mode = mode,
                    populationSize = 32,
                    sessionSalt = 55L + mode.ordinal
                )
                assertTrue(initState.population.all { it.exploreNoise == 0.0 })

                val gen1 = PrecisionTargetEngine.executeEvolutionGeneration(
                    state = initState,
                    cache = cache,
                    minObjectiveStreak = 1,
                    populationSize = 32
                )
                val best = (gen1.winningEval ?: gen1.bestTrainEval)?.genome ?: initState.population.first()
                assertEquals(mode, best.mode)

                PrecisionTargetEngine.saveBestGenome(tempDir, mode, best)
                val reloadedBest = PrecisionTargetEngine.loadSavedBestGenome(tempDir, mode)
                assertNotNull(reloadedBest)
                assertEquals(best.signatureKey(), reloadedBest?.signatureKey())

                val (bt1, ledger1) = PrecisionTargetEngine.mergeAppendOnlyApprovedBacktest(
                    genome = best,
                    cache = cache,
                    existingLedger = null,
                    confirmationTestsCount = gen1.nextState.confirmationTestsCount
                )
                assertTrue(bt1.allStepsCausalVerified)

                val target = LotoDateUtils.resolveNextUpcomingDraw(initialHistory, lastDraw.timestamp - 60_000L)
                val pred1 = PrecisionTargetEngine.predictNextUpcoming(initialHistory, best, target)
                val pred2 = PrecisionTargetEngine.predictNextUpcoming(initialHistory, best, target)
                assertEquals(mode.outputCount, pred1.predictedNumbers.distinct().size)
                assertEquals(pred1.predictedNumbers, pred2.predictedNumbers)

                val ledgerWithLive = PrecisionTargetEngine.recordLivePredictionInLedger(ledger1, pred1)
                PrecisionTargetEngine.saveLedger(tempDir, ledgerWithLive)
                val loadedLedger = PrecisionTargetEngine.loadLedger(tempDir, ledgerWithLive.ledgerKey)
                assertNotNull(loadedLedger)

                val cache2 = PrecisionTargetEngine.buildCausalCache(
                    history = draws,
                    period = BacktestPeriod.THREE_DAYS,
                    referenceNowMs = lastDraw.timestamp + 60_000L,
                    customDays = 3
                )
                val (bt2, _) = PrecisionTargetEngine.mergeAppendOnlyApprovedBacktest(
                    genome = best,
                    cache = cache2,
                    existingLedger = loadedLedger,
                    confirmationTestsCount = gen1.nextState.confirmationTestsCount
                )
                assertEquals(bt1.steps.size, bt2.preservedStepsCount)
                assertEquals(1, bt2.appendedStepsCount)
                assertEquals(1, bt2.liveMatchedCount)
                assertEquals(pred1.predictedNumbers.sorted(), bt2.steps.last().predictedNumbers.sorted())
            }
        } finally {
            tempDir.deleteRecursively()
        }
    }

    @Test
    fun incrementalExtractorsMatchBatchExtractionExactly_includingTieBarrier_android() {
        val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).apply {
            timeZone = LotoDateUtils.beninTz
        }
        val slots = listOf("00H" to "00:05", "8H" to "08:00", "11H" to "11:00", "14H" to "14:00", "18H" to "18:00", "20H" to "20:00")
        val raw = ArrayList<LotoResultEntity>()
        var seedIdx = 1
        for (day in 1..6) {
            val dayStr = String.format(Locale.US, "2026-10-%02d", day)
            for ((hLabel, hhmm) in slots) {
                val ts = sdf.parse("$dayStr $hhmm")!!.time
                val a = (seedIdx % 15) + 1
                val b = ((seedIdx * 3) % 20) + 20
                val c = ((seedIdx * 5) % 20) + 42
                val d = ((seedIdx * 7) % 15) + 63
                val e = 80 + (seedIdx % 10)
                raw.add(
                    LotoResultEntity(
                        uniqueId = "par_${dayStr}_$hLabel",
                        date = LotoDateUtils.formatFrenchDayAndDate(ts),
                        time = hLabel,
                        game = "Fortune",
                        num1 = a, num2 = b, num3 = c, num4 = d, num5 = e,
                        timestamp = ts
                    )
                )
                seedIdx++
            }
        }
        val tieTarget = raw[10]
        raw.add(tieTarget.copy(uniqueId = "par_tie_equal_ts", time = "20H", num1 = 5, num2 = 25, num3 = 45, num4 = 65, num5 = 85))
        val chronological = raw
            .distinctBy { it.uniqueId }
            .sortedWith(compareBy<LotoResultEntity> { it.timestamp }.thenBy { it.uniqueId })

        val testIndices = intArrayOf(1, 5, 9, 10, 11, 12, 22, chronological.lastIndex)
        val seqSym = SymbolicFormula.extractRawMatrixSequence(chronological, testIndices)
        val seqPrec = PrecisionTargetEngine.extractPrecisionMatrixSequence(chronological, testIndices)

        for (t in testIndices) {
            val target = chronological[t]
            val strictPast = chronological.subList(0, t).filter { it.timestamp < target.timestamp && it.uniqueId != target.uniqueId }
            val horaire = LotoDateUtils.normalizeHoraire(target)

            val expSym = SymbolicFormula.extractRawMatrix(strictPast, horaire)
            val actSym = seqSym[t]!!
            for (feat in RawFeature.entries) {
                for (n in 1..90) {
                    val diff = kotlin.math.abs(expSym.get(feat.ordinal, n) - actSym.get(feat.ordinal, n))
                    assertTrue(
                        "Android Symbolic O(T) mismatch at t=$t feat=$feat n=$n",
                        diff < 1e-12
                    )
                }
            }

            val expPrec = PrecisionTargetEngine.extractFeatures(strictPast, horaire)
            val actPrec = seqPrec[t]!!
            for (feat in PrecisionFeature.entries) {
                for (n in 1..90) {
                    val diff = kotlin.math.abs(expPrec.get(feat.ordinal, n) - actPrec.get(feat.ordinal, n))
                    assertTrue(
                        "Android Precision O(T) mismatch at t=$t feat=$feat n=$n",
                        diff < 1e-12
                    )
                }
            }
        }
    }

    @Test
    fun causalAiLearnerAndChercheurZeroLeakage_andLogarithmicCorrection_android() {
        val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).apply {
            timeZone = LotoDateUtils.beninTz
        }
        val slots = listOf("00H" to "00:05", "8H" to "08:00", "11H" to "11:00", "14H" to "14:00", "18H" to "18:00", "20H" to "20:00")
        val drawsA = ArrayList<LotoResultEntity>()
        var k = 1
        for (day in 1..8) {
            val dayStr = String.format(Locale.US, "2026-10-%02d", day)
            for ((hLabel, hhmm) in slots) {
                val ts = sdf.parse("$dayStr $hhmm")!!.time
                drawsA.add(
                    LotoResultEntity(
                        uniqueId = "leak_${dayStr}_$hLabel",
                        date = LotoDateUtils.formatFrenchDayAndDate(ts),
                        time = hLabel,
                        game = "Fortune",
                        num1 = (k % 12) + 1,
                        num2 = ((k * 2) % 18) + 18,
                        num3 = ((k * 3) % 18) + 38,
                        num4 = ((k * 5) % 18) + 58,
                        num5 = 78 + (k % 12),
                        timestamp = ts
                    )
                )
                k++
            }
        }
        val cacheA = DiscoveryEngine.buildCausalCache(drawsA, BacktestPeriod.THREE_WEEKS, drawsA.last().timestamp + 1_000L, null)
        assertTrue("Android discoverySteps must not be empty", cacheA.discoverySteps.isNotEmpty())
        val lastDiscoveryTs = cacheA.discoverySteps.last().draw.timestamp
        val drawsB = drawsA.mapIndexed { idx, d ->
            if (d.timestamp > lastDiscoveryTs) d.copy(num1 = 89 - (idx % 9), num2 = 77, num3 = 66, num4 = 55, num5 = 44) else d
        }
        val cacheB = DiscoveryEngine.buildCausalCache(drawsB, BacktestPeriod.THREE_WEEKS, drawsB.last().timestamp + 1_000L, null)
        assertEquals(cacheA.discoverySteps.size, cacheB.discoverySteps.size)
        val ridgeA = CausalAiLearner.fitSymbolicRidgeGenomes(cacheA.discoverySteps, 1, 77L, maxCount = 8)
        val ridgeB = CausalAiLearner.fitSymbolicRidgeGenomes(cacheB.discoverySteps, 1, 77L, maxCount = 8)
        assertTrue("Android CausalAiLearner must generate symbolic Ridge genomes", ridgeA.isNotEmpty())
        assertEquals(ridgeA.map { it.describeCompact() }, ridgeB.map { it.describeCompact() })

        val chA = ChercheurEngine.runResearchAndDistill(cacheA, 1, 1, emptySet())
        val chB = ChercheurEngine.runResearchAndDistill(cacheB, 1, 1, emptySet())
        assertTrue("Android Chercheur must synthesize genomes", chA.synthesizedGenomes.isNotEmpty())
        assertEquals(
            chA.synthesizedGenomes.map { it.describeCompact() },
            chB.synthesizedGenomes.map { it.describeCompact() }
        )

        val f500 = DiscoveryEngine.effectiveMultipleTestFactor(500)
        assertTrue("Android logarithmic cap expected << 500, got $f500", f500 < 20.0)
        assertEquals(f500, PrecisionTargetEngine.effectiveMultipleTestFactor(500), 1e-9)
    }

    @Test
    fun penultimateDrawMandatoryRule_enforcedAcrossDiscoveryLabAndPrecision_android() {
        val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).apply {
            timeZone = LotoDateUtils.beninTz
        }
        val fixedLowGenome = SymbolicGenome(
            id = "android_pen_fixed_1_5",
            generation = 1,
            parentIds = emptyList(),
            primaryExpr = ExprNode.Unary(UnaryOp.NEG, ExprNode.Var(RawFeature.X_NORM)),
            secondaryExpr = ExprNode.Unary(UnaryOp.NEG, ExprNode.Var(RawFeature.X_NORM)),
            primarySlots = 3,
            samplingTemp = 0.25,
            exploreNoise = 0.0,
            seed = 1L
        )
        val slots = listOf("00H" to "00:05", "8H" to "08:00", "11H" to "11:00", "14H" to "14:00", "18H" to "18:00", "20H" to "20:00")

        fun buildSteps(dayStr: String, numsBySlot: List<List<Int>>): List<BacktestStepResult> {
            return slots.mapIndexed { sIdx, (hLabel, hhmm) ->
                val ts = sdf.parse("$dayStr $hhmm")!!.time
                val nums = numsBySlot[sIdx]
                val matches = nums.filter { it in 1..5 }
                BacktestStepResult(
                    stepIndex = sIdx + 1,
                    drawId = "a_${dayStr}_$hLabel",
                    drawLabel = "Fortune $hLabel",
                    drawDate = LotoDateUtils.formatFrenchDayAndDate(ts),
                    dayKey = dayStr,
                    horaire = hLabel,
                    game = "Fortune",
                    trainSampleCount = 5,
                    maxTrainTimestamp = ts - 1_000L,
                    targetTimestamp = ts,
                    causalVerified = true,
                    predictedNumbers = listOf(1, 2, 3, 4, 5),
                    actualNumbers = nums,
                    matchedNumbers = matches,
                    matchCount = matches.size,
                    expectedMatchesEstimate = 0.3f,
                    stepBrierScore = 0.05,
                    stepBernoulliLogLoss = 0.2
                )
            }
        }

        // Sans avant-dernier tirage (18H) à >=2 -> refusé
        val missSteps = buildSteps(
            "2026-10-05",
            listOf(
                listOf(1, 2, 20, 21, 22),
                listOf(1, 2, 30, 31, 32),
                listOf(1, 2, 40, 41, 42),
                listOf(50, 51, 52, 53, 54),
                listOf(60, 61, 62, 63, 64),
                listOf(70, 71, 72, 73, 74)
            )
        )
        val missSummary = ScientificEvaluator.buildDailySummaries(missSteps, useSliding6DrawBlocks = false).first()
        assertEquals(3, missSummary.drawsWithTwoPlus)
        org.junit.Assert.assertFalse(missSummary.penultimateTwoPlus)
        org.junit.Assert.assertFalse("3 tirages >=2 sans avant-dernier >=2 ne doit pas qualifier", missSummary.isQualifyingDay)

        // Avec tous les tirages (6/6) à >=2 dont l'avant-dernier (18H) à >=2 -> qualifié
        val hitSteps = buildSteps(
            "2026-10-06",
            listOf(
                listOf(1, 2, 20, 21, 22),
                listOf(1, 2, 30, 31, 32),
                listOf(1, 2, 40, 41, 42),
                listOf(1, 2, 50, 51, 52),
                listOf(2, 3, 70, 71, 72),
                listOf(1, 2, 80, 81, 82)
            )
        )
        val hitSummary = ScientificEvaluator.buildDailySummaries(hitSteps, useSliding6DrawBlocks = true).first()
        assertEquals(6, hitSummary.drawsWithTwoPlus)
        assertTrue(hitSummary.penultimateTwoPlus)
        assertTrue("Tous les tirages >=2 avec avant-dernier >=2 doit qualifier", hitSummary.isQualifyingDay)

        // Labo Scientifique Android avec modèles conservés
        val draws = ArrayList<LotoResultEntity>()
        var seq = 1
        for (day in 1..10) {
            val dayStr = String.format(Locale.US, "2026-10-%02d", day)
            for ((hLabel, hhmm) in slots) {
                val ts = sdf.parse("$dayStr $hhmm")!!.time
                val n1 = (seq % 10) + 1
                val n2 = n1 + 12
                val n3 = n2 + 15
                val n4 = n3 + 18
                val n5 = (n4 + 19).coerceAtMost(90)
                draws.add(LotoResultEntity("ad_${dayStr}_$hLabel", LotoDateUtils.formatFrenchDayAndDate(ts), hLabel, "Fortune", n1, n2, n3, n4, n5, ts))
                seq++
            }
        }
        val report = ScientificEvaluator.runFullScientificComparison(
            data = draws,
            period = BacktestPeriod.ONE_DAY,
            referenceNowMs = draws.last().timestamp + 1_000L,
            customDays = 1,
            savedModels = listOf(
                LabSavedModelInput(
                    id = "saved_android_1",
                    displayName = "Modèle Android 1J",
                    savedStreak = 1,
                    slotSplitLabel = "3+2",
                    genome = fixedLowGenome
                )
            )
        )
        assertEquals("1 Jour", report.periodLabel)
        assertEquals(1, report.savedModelAudits.size)
        val audit = report.savedModelAudits.first()
        assertTrue(audit.causalVerified)
        assertEquals(audit.qualifyingDaysCount >= 1 && audit.causalVerified, audit.objectiveValidated)

        // Moteur 2 TWO_NUMBERS : avant-dernier exact obligatoire dans le bloc
        val precMiss = missSteps.map { st ->
            PrecisionBacktestStep(
                stepIndex = st.stepIndex,
                drawId = st.drawId,
                drawDate = st.drawDate,
                dayKey = st.dayKey,
                horaire = st.horaire,
                game = st.game,
                strictPastCount = 5,
                maxPastTs = st.maxTrainTimestamp,
                targetTimestamp = st.targetTimestamp,
                causalVerified = true,
                predictedNumbers = listOf(1, 2),
                actualPos1 = st.actualNumbers[0],
                actualNumbers = st.actualNumbers,
                matchedNumbers = st.matchedNumbers,
                matchCount = st.matchCount,
                exactTargetHit = st.matchCount >= 2,
                stepBrierScore = 0.05
            )
        }
        val precMissBlock = PrecisionTargetEngine.buildDailyBlocks(precMiss, PrecisionTargetMode.TWO_NUMBERS).first()
        org.junit.Assert.assertFalse(precMissBlock.penultimateExact)
        org.junit.Assert.assertFalse(precMissBlock.isQualifyingBlock)

        val precHit = hitSteps.map { st ->
            PrecisionBacktestStep(
                stepIndex = st.stepIndex,
                drawId = st.drawId,
                drawDate = st.drawDate,
                dayKey = st.dayKey,
                horaire = st.horaire,
                game = st.game,
                strictPastCount = 5,
                maxPastTs = st.maxTrainTimestamp,
                targetTimestamp = st.targetTimestamp,
                causalVerified = true,
                predictedNumbers = listOf(2, 3),
                actualPos1 = st.actualNumbers[0],
                actualNumbers = st.actualNumbers,
                matchedNumbers = st.matchedNumbers,
                matchCount = st.matchCount,
                exactTargetHit = st.matchCount >= 2,
                stepBrierScore = 0.05
            )
        }
        val precHitBlock = PrecisionTargetEngine.buildDailyBlocks(precHit, PrecisionTargetMode.TWO_NUMBERS).first()
        assertTrue(precHitBlock.penultimateExact)
        assertTrue(precHitBlock.isQualifyingBlock)
    }

    @Test
    fun androidEvolutionAvoidsCloneDeadlockAndGuidesTowardPenultimateHits() {
        val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).apply {
            timeZone = LotoDateUtils.beninTz
        }
        val slots = listOf("00H" to "00:05", "8H" to "08:00", "11H" to "11:00", "14H" to "14:00", "18H" to "18:00", "20H" to "20:00")
        val draws = ArrayList<LotoResultEntity>()
        val drawRng = kotlin.random.Random(909L)
        val ballPool = (1..90).toMutableList()
        for (day in 1..14) {
            val dayStr = String.format(Locale.US, "2026-10-%02d", day)
            for ((hLabel, hhmm) in slots) {
                val ts = sdf.parse("$dayStr $hhmm")!!.time
                ballPool.shuffle(drawRng)
                val nums = ballPool.take(5).sorted()
                draws.add(
                    LotoResultEntity(
                        uniqueId = "ad_anti_${dayStr}_$hLabel",
                        date = LotoDateUtils.formatFrenchDayAndDate(ts),
                        time = hLabel,
                        game = "Fortune",
                        num1 = nums[0], num2 = nums[1], num3 = nums[2], num4 = nums[3], num5 = nums[4],
                        timestamp = ts
                    )
                )
            }
        }
        val cache = DiscoveryEngine.buildCausalCache(
            history = draws,
            period = BacktestPeriod.THREE_DAYS,
            referenceNowMs = draws.last().timestamp + 1_000L
        )
        val steps = cache.steps.take(6)
        val (pen0, _, score0) = DiscoveryEngine.computeBlockProgressGradient(steps, listOf(2, 2, 0, 0, 0, 0))
        val (pen1, _, score1) = DiscoveryEngine.computeBlockProgressGradient(steps, listOf(2, 0, 0, 0, 2, 0))
        assertEquals(0, pen0)
        assertEquals(1, pen1)
        assertTrue(score1 > score0)

        var state = DiscoveryEngine.createInitialState(sessionSalt = 909L)
        var prevSeen = 0
        repeat(4) {
            val res = DiscoveryEngine.executeEvolutionGeneration(
                state = state,
                cache = cache,
                periodName = BacktestPeriod.THREE_DAYS.name,
                minObjectiveStreak = 1
            )
            state = res.nextState
            assertTrue("seenSignatures must keep growing across generations", state.seenSignatures.size > prevSeen)
            prevSeen = state.seenSignatures.size
            // Aucun clonage exact de masse dans la population suivante
            assertTrue(state.population.map { it.describeCompact() }.distinct().size >= state.population.size / 2)
        }
        assertTrue("No false positive on uniform random 5/90 draws", state.validatedCandidate == null)

        val oneDayCache = DiscoveryEngine.buildCausalCache(
            history = draws,
            period = BacktestPeriod.ONE_DAY,
            referenceNowMs = draws.last().timestamp + 1_000L,
            customDays = 1
        )
        val precCache = PrecisionTargetEngine.buildCausalCache(
            history = draws,
            period = BacktestPeriod.ONE_DAY,
            referenceNowMs = draws.last().timestamp + 1_000L,
            customDays = 1
        )
        assertEquals(6, oneDayCache.steps.size)
        val precState = PrecisionTargetEngine.createInitialState(
            mode = PrecisionTargetMode.TWO_NUMBERS,
            sessionSalt = 919L
        )
        val precRes = PrecisionTargetEngine.executeEvolutionGeneration(
            state = precState,
            cache = precCache,
            minObjectiveStreak = 1,
            populationSize = precState.population.size
        )
        assertTrue(precRes.nextState.seenSignatures.size >= precState.population.size)
    }
}

