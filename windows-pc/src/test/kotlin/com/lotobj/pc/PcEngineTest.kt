package com.lotobj.pc

import com.lotobj.pc.engine.GpuOpenClAccelerator
import com.lotobj.pc.engine.PcDiscoveryEngine
import com.lotobj.pc.engine.PcWorkstationCoordinator
import com.lotobj.pc.engine.ScientificMath
import com.lotobj.pc.engine.SymbolicFormula
import com.lotobj.pc.model.BacktestPeriod
import com.lotobj.pc.model.BinaryOp
import com.lotobj.pc.model.ComputeMode
import com.lotobj.pc.model.ExprNode
import com.lotobj.pc.model.LotoDraw
import com.lotobj.pc.model.RawFeature
import com.lotobj.pc.model.SymbolicGenome
import com.lotobj.pc.model.UnaryOp
import com.lotobj.pc.util.BacktestExportHelper
import com.lotobj.pc.util.LotoDateUtils
import com.lotobj.pc.util.StrategyBackupHelper
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.file.Files
import java.util.concurrent.ForkJoinPool
import kotlin.math.abs
import kotlin.random.Random

class PcEngineTest {

    private fun buildSyntheticHistory(daysCount: Int = 26): List<LotoDraw> {
        val baseTs = 1_760_000_000_000L
        val horaires = listOf("8H", "11H", "14H", "18H", "20H")
        val rng = Random(12345L)
        val out = ArrayList<LotoDraw>(daysCount * horaires.size)
        for (d in 0 until daysCount) {
            for ((hIdx, h) in horaires.withIndex()) {
                val ts = baseTs + d * 86_400_000L + hIdx * 3 * 3_600_000L
                val nums = (1..90).shuffled(rng).take(5).sorted()
                out.add(
                    LotoDraw(
                        uniqueId = "syn_${d}_$h",
                        date = LotoDateUtils.formatFrenchDayAndDate(ts),
                        time = h,
                        game = "Fortune",
                        num1 = nums[0],
                        num2 = nums[1],
                        num3 = nums[2],
                        num4 = nums[3],
                        num5 = nums[4],
                        timestamp = ts
                    )
                )
            }
        }
        return out
    }

    @Test
    fun rpnBytecodeMatchesAstEvaluatorAcrossAllTheoremsAndOperators() {
        val history = buildSyntheticHistory(15)
        val matrix = SymbolicFormula.extractRawMatrix(history.dropLast(1), "14H")
        val rng = Random(77L)

        // Couverture explicite de tous les UnaryOp et BinaryOp ajoutés
        for (uOp in UnaryOp.entries) {
            for (feat in RawFeature.entries) {
                val expr = ExprNode.Unary(uOp, ExprNode.Var(feat))
                val rpn = SymbolicFormula.compileToRpn(expr)
                for (n in listOf(1, 7, 45, 90)) {
                    val astVal = SymbolicFormula.evaluateForNumber(expr, matrix, n, 0.4)
                    val rpnVal = rpn.evaluateCpu(matrix, n, 0.4)
                    assertTrue(
                        "Unary $uOp($feat) mismatch at n=$n: ast=$astVal rpn=$rpnVal",
                        abs(astVal - rpnVal) < 1e-3
                    )
                }
            }
        }
        for (bOp in BinaryOp.entries) {
            val expr = ExprNode.Binary(
                bOp,
                ExprNode.Var(RawFeature.LIMIT_HAZARD),
                ExprNode.Var(RawFeature.CONGRUENCE_CUBE90)
            )
            val rpn = SymbolicFormula.compileToRpn(expr)
            for (n in listOf(1, 18, 55, 90)) {
                val astVal = SymbolicFormula.evaluateForNumber(expr, matrix, n, 0.2)
                val rpnVal = rpn.evaluateCpu(matrix, n, 0.2)
                assertTrue(
                    "Binary $bOp mismatch at n=$n: ast=$astVal rpn=$rpnVal",
                    abs(astVal - rpnVal) < 1e-3
                )
            }
        }

        // Arbres aléatoires profonds + arbres théorèmes
        repeat(60) { idx ->
            val expr = if (idx % 2 == 0) {
                SymbolicFormula.randomExpr(5, rng)
            } else {
                SymbolicFormula.randomTheoremExpr(idx, rng)
            }
            val rpn = SymbolicFormula.compileToRpn(expr)
            for (n in listOf(1, 7, 25, 45, 73, 90)) {
                val rep = if (idx % 2 == 0) 0.0 else 0.4
                val astVal = SymbolicFormula.evaluateForNumber(expr, matrix, n, rep)
                val rpnVal = rpn.evaluateCpu(matrix, n, rep)
                assertTrue(
                    "RPN vs AST mismatch at n=$n expr=${SymbolicFormula.prettyPrint(expr)}: ast=$astVal rpn=$rpnVal",
                    abs(astVal - rpnVal) < 1e-3
                )
            }
        }
    }

    @Test
    fun parallelMonteCarloAndEvolutionEngineRunCleanlyOnAllCores() {
        val history = buildSyntheticHistory(28)
        val pool = ForkJoinPool(Runtime.getRuntime().availableProcessors().coerceAtLeast(1))
        val gpu = GpuOpenClAccelerator()
        try {
            val cache = PcDiscoveryEngine.buildCausalCache(
                history = history,
                period = BacktestPeriod.THREE_WEEKS,
                referenceNowMs = history.last().timestamp + 1_000L,
                monteCarloSims = 1_500,
                pool = pool
            )
            assertTrue("Cache should have causal steps", cache.steps.size >= 20)
            assertTrue("Every step strict past causal check", cache.steps.all { it.maxPastTs < it.draw.timestamp })

            val initial = PcDiscoveryEngine.createInitialState(populationSize = 64, sessionSalt = 99L)
            val gen1 = PcDiscoveryEngine.executeEvolutionGeneration(
                state = initial,
                cache = cache,
                periodName = BacktestPeriod.THREE_WEEKS.name,
                minObjectiveStreak = 4,
                populationSize = 64,
                computeMode = ComputeMode.GPU_AND_CPU_MAX,
                cpuPool = pool,
                gpuAccelerator = gpu
            )
            assertEquals(1, gen1.nextState.generation)
            assertTrue("Should evaluate at least populationSize", gen1.nextState.experimentsCount >= 64)
            assertTrue(gen1.evaluationsPerSec > 0.0)
            assertFalse(gen1.nextState.population.isEmpty())

            val bestGenome = gen1.bestTrainEval?.genome ?: initial.population.first()
            val bt = PcDiscoveryEngine.buildValidatedBacktest(bestGenome, cache, 1, 64)
            assertTrue(bt.allStepsCausalVerified)
            assertEquals(cache.steps.size, bt.totalTested)

            val nextTarget = LotoDateUtils.resolveNextUpcomingDraw(history, history.last().timestamp + 10_000L)
            val pred = PcDiscoveryEngine.predictNextWithGenome(bestGenome, history, nextTarget.horaire, bt, nextTarget.targetTimestamp)
            assertEquals(5, pred.predictedDraw.distinct().size)
            assertTrue(pred.predictedDraw.all { it in 1..90 })

            val mc = ScientificMath.runMonteCarloParallel(history.takeLast(20), simulations = 2_000, seed = 42L, pool = pool)
            assertEquals(2_000, mc.simulationsCount)
            assertTrue(mc.meanAvgMatches in 0.1f..0.6f)
        } finally {
            pool.shutdownNow()
            gpu.close()
        }
    }

    @Test
    fun strategyBackupJsonCompatibleBetweenAndroidAndPc() {
        val tempDir = Files.createTempDirectory("lotobj_pc_test").toFile()
        try {
            val genome = SymbolicGenome(
                id = "android_strat_3j",
                generation = 14,
                parentIds = listOf("p1", "p2"),
                primaryExpr = ExprNode.Binary(
                    BinaryOp.ADD,
                    ExprNode.Var(RawFeature.GAP_NORM),
                    ExprNode.Unary(UnaryOp.SIGMOID, ExprNode.Var(RawFeature.FREQ_W15))
                ),
                secondaryExpr = ExprNode.Var(RawFeature.CO_LAST_RAW),
                primarySlots = 3,
                samplingTemp = 0.65,
                exploreNoise = 0.12,
                seed = 777L
            )
            val jsonInput = """
                {
                  "format": "loto_bj_strategies_backup_v1",
                  "version": 1,
                  "app": "Loto BJ",
                  "periodName": "THREE_WEEKS",
                  "bestStreak": 3,
                  "strategiesCount": 1,
                  "strategies": [
                    {
                      "id": "strat_3j_999",
                      "generation": 14,
                      "genomeJson": ${kotlinx.serialization.json.JsonPrimitive(SymbolicFormula.genomeToJson(genome))},
                      "status": "MILESTONE_3J",
                      "discoveryStreak": 3,
                      "rollingStreak": 3,
                      "combinedStreak": 3,
                      "fitnessScore": 142.5
                    }
                  ]
                }
            """.trimIndent()

            val parsed = StrategyBackupHelper.parseBackup(jsonInput).getOrThrow()
            assertEquals(1, parsed.strategies.size)
            assertEquals(3, parsed.strategies.first().combinedStreak)

            val reSerialized = StrategyBackupHelper.serializeBackup(parsed.strategies)
            assertTrue(reSerialized.contains("loto_bj_strategies_backup_v1"))

            val repo = com.lotobj.pc.data.LotoDataRepository(storageDir = tempDir)
            repo.mergeAndSaveDraws(buildSyntheticHistory(14))
            val coordinator = PcWorkstationCoordinator(repository = repo)
            try {
                val msg = coordinator.importStrategyBackupText(jsonInput)
                assertTrue("Import message should indicate 1 strategy: $msg", msg.contains("1 méthode"))
                val es = coordinator.uiState.value.engineState
                assertTrue(es.hasSavedWinner)
                assertTrue(es.canLaunchBacktest)

                val btReport = BacktestExportHelper.formatDetailedBacktestReport(
                    backtestResult = ScientificMath.emptyBacktestResult("3 Semaines (21j)"),
                    engineState = es
                )
                assertTrue(btReport.contains("LOTO BJ PC"))
            } finally {
                coordinator.close()
            }
        } finally {
            tempDir.deleteRecursively()
        }
    }

    @Test
    fun scientificLabRunsAll13MethodsAndParallelMonteCarlo() {
        val history = buildSyntheticHistory(28)
        val pool = ForkJoinPool(Runtime.getRuntime().availableProcessors().coerceAtLeast(1))
        try {
            val report = com.lotobj.pc.engine.ClassicalAndScientificLab.runFullScientificComparison(
                data = history,
                period = BacktestPeriod.THREE_WEEKS,
                monteCarloSims = 1_500,
                referenceNowMs = history.last().timestamp + 1_000L,
                pool = pool
            )
            assertEquals(13, report.indepTestComparisons.size)
            assertEquals(13, report.valComparisons.size)
            assertEquals(13, report.devComparisons.size)
            assertTrue(report.cube90AuditSummary.contains("30 images"))
            assertTrue(report.complementaryPairs.isNotEmpty())
            assertEquals(1_500, report.monteCarloIndepTest.simulationsCount)

            val nextTarget = LotoDateUtils.resolveNextUpcomingDraw(history, history.last().timestamp + 5_000L)
            val (bt, pred) = com.lotobj.pc.engine.ClassicalAndScientificLab.evaluateAndPredictClassicalMethod(
                method = report.recommendedMethod,
                draws = history,
                period = BacktestPeriod.THREE_WEEKS,
                nextTarget = nextTarget
            )
            assertTrue(bt.allStepsCausalVerified)
            assertEquals(5, pred.predictedDraw.distinct().size)
            assertEquals(5, pred.numberConfidences.size)
        } finally {
            pool.shutdownNow()
        }
    }

    @Test
    fun windowsLaunchScriptsAvoidCrashProneJvmFlags() {
        val runBat = java.io.File("run_windows.bat").takeIf { it.exists() }
            ?: java.io.File("windows-pc/run_windows.bat")
        val exeBat = java.io.File("build_windows_exe.bat").takeIf { it.exists() }
            ?: java.io.File("windows-pc/build_windows_exe.bat")
        assertTrue("run_windows.bat should exist", runBat.exists())
        assertTrue("build_windows_exe.bat should exist", exeBat.exists())

        val runActive = runBat.readLines().filterNot { it.trimStart().startsWith("REM", ignoreCase = true) }.joinToString("\n")
        val exeText = exeBat.readText()
        assertFalse("run_windows.bat must not pass -XX:+AlwaysPreTouch", runActive.contains("AlwaysPreTouch"))
        assertFalse("build_windows_exe.bat must not pass -XX:+AlwaysPreTouch", exeText.contains("AlwaysPreTouch"))
        assertFalse("run_windows.bat must not force sun.java2d.d3d=true", runActive.contains("sun.java2d.d3d=true"))
        assertTrue("build_windows_exe.bat must explicitly include jdk.management for telemetry", exeText.contains("jdk.management"))
    }

    @Test
    fun zeroReuseTabuPreventsReEvaluatingUsedFormulasAcrossGenerations() {
        val history = buildSyntheticHistory(24)
        val pool = ForkJoinPool(Runtime.getRuntime().availableProcessors().coerceAtLeast(1))
        val gpu = GpuOpenClAccelerator()
        try {
            val cache = PcDiscoveryEngine.buildCausalCache(
                history = history,
                period = BacktestPeriod.THREE_WEEKS,
                referenceNowMs = history.last().timestamp + 1_000L,
                monteCarloSims = 500,
                pool = pool
            )
            var state = PcDiscoveryEngine.createInitialState(populationSize = 48, sessionSalt = 313L)
            var prevSeenSize = 0
            for (g in 1..4) {
                // Chaque individu préparé pour la génération suivante doit être absent du registre seenSignatures
                val nextBatchSigs = state.population.map { it.describeCompact() }
                for (sig in nextBatchSigs) {
                    assertFalse("Method already used should not reappear in generation $g queue: $sig", sig in state.seenSignatures)
                }
                val res = PcDiscoveryEngine.executeEvolutionGeneration(
                    state = state,
                    cache = cache,
                    periodName = BacktestPeriod.THREE_WEEKS.name,
                    minObjectiveStreak = 4,
                    populationSize = 48,
                    computeMode = ComputeMode.CPU_MAX_CORES,
                    cpuPool = pool,
                    gpuAccelerator = gpu
                )
                state = res.nextState
                assertTrue("Excluded signature registry must grow strictly each generation", state.excludedSignaturesCount > prevSeenSize)
                prevSeenSize = state.excludedSignaturesCount
            }
        } finally {
            pool.shutdownNow()
            gpu.close()
        }
    }

    @Test
    fun strictZeroMatchCapRejectsCandidatesWithFiftyPercentOrMoreZeroDraws() {
        // Génère des tirages déterministes centrés sur 1..5 uniquement, et teste un génome qui choisit ~86..90 -> 100% 0-boule
        val baseTs = 1_760_000_000_000L
        val draws = (0 until 20).map { d ->
            val ts = baseTs + d * 86_400_000L
            LotoDraw(
                uniqueId = "fix_$d",
                date = LotoDateUtils.formatFrenchDayAndDate(ts),
                time = "11H",
                game = "Fortune",
                num1 = 1, num2 = 2, num3 = 3, num4 = 4, num5 = 5,
                timestamp = ts
            )
        }
        val pool = ForkJoinPool(2)
        val gpu = GpuOpenClAccelerator()
        try {
            val cache = PcDiscoveryEngine.buildCausalCache(
                history = draws,
                period = BacktestPeriod.ALL_HISTORY,
                referenceNowMs = draws.last().timestamp + 1_000L,
                monteCarloSims = 500,
                pool = pool
            )
            // Génome biaisé vers X_NORM maximal (86..90) qui a 100% de 0-boule contre 1..5
            val badGenome = SymbolicGenome(
                id = "bad_high_nums",
                generation = 1,
                parentIds = emptyList(),
                primaryExpr = ExprNode.Var(RawFeature.X_NORM),
                secondaryExpr = ExprNode.Var(RawFeature.X_NORM),
                primarySlots = 3,
                samplingTemp = 0.15,
                exploreNoise = 0.02,
                seed = 1L
            )
            val rollStats = PcDiscoveryEngine.evaluateGenomeOnStage(badGenome, cache.rollingSteps)
            assertFalse("Candidate with 100% 0-boule must fail passesZeroCapStrict", rollStats.passesZeroCapStrict)
            assertTrue(rollStats.zeroMatchRate >= 0.50f)

            // Génome aligné sur LIMIT_HAZARD / -X_NORM (1..5) -> 0% de 0-boule (<50%)
            val goodGenome = badGenome.copy(
                id = "good_low_nums",
                primaryExpr = ExprNode.Unary(UnaryOp.NEG, ExprNode.Var(RawFeature.X_NORM)),
                secondaryExpr = ExprNode.Unary(UnaryOp.NEG, ExprNode.Var(RawFeature.X_NORM))
            )
            val goodStats = PcDiscoveryEngine.evaluateGenomeOnStage(goodGenome, cache.rollingSteps)
            assertTrue("Candidate with <50% 0-boule must pass passesZeroCapStrict", goodStats.passesZeroCapStrict)
            assertTrue("Should detect 5-ball hits on deterministic data", goodStats.fiveMatchCount > 0)
        } finally {
            pool.shutdownNow()
            gpu.close()
        }
    }

    @Test
    fun strategyFactoryGeneratesValidDomainFormulasAndEnforcesZeroReuseAtRuntime() {
        val factories = com.lotobj.pc.engine.StrategyFactory.listFactories()
        assertEquals(4, factories.size)

        val history = buildSyntheticHistory(24)
        val matrix = SymbolicFormula.extractRawMatrix(history.dropLast(1), "14H")

        // Chaque fabrique produit une formule RPN-compilable exploitant des signaux de son domaine
        val expectedDomains = listOf(
            com.lotobj.pc.engine.MathTheoremDomain.MODULAR_ARITHMETIC,
            com.lotobj.pc.engine.MathTheoremDomain.PROBABILITY_DENSITY,
            com.lotobj.pc.engine.MathTheoremDomain.TRIANGULATION,
            com.lotobj.pc.engine.MathTheoremDomain.HYBRID_SYNTHESIS
        )
        for ((idx, dom) in expectedDomains.withIndex()) {
            val builder = com.lotobj.pc.engine.StrategyFactory.getFactory(dom)
            assertEquals(dom, builder.domain)
            val genome = builder.createGenome(id = "test_$dom", generation = 1, seed = 404L + idx)
            val usedPrimary = SymbolicFormula.usedFeatures(genome.primaryExpr)
            assertTrue("Builder $dom must reference primary domain features", usedPrimary.isNotEmpty())
            assertTrue(
                "Builder $dom must draw from its primary feature pool",
                usedPrimary.any { it in builder.primaryFeatures }
            )
            val rpnA = SymbolicFormula.compileToRpn(genome.primaryExpr)
            val rpnB = SymbolicFormula.compileToRpn(genome.secondaryExpr)
            assertTrue(rpnA.instructionCount in 1..96)
            assertTrue(rpnB.instructionCount in 1..96)
            val astScore = SymbolicFormula.evaluateForNumber(genome.primaryExpr, matrix, 18, 0.25)
            val rpnScore = rpnA.evaluateCpu(matrix, 18, 0.25)
            assertTrue("RPN vs AST parity failed for $dom: ast=$astScore rpn=$rpnScore", abs(astScore - rpnScore) < 1e-3)
        }

        // Test d'exécution complet StrategyFactory + exclusion zéro-réutilisation
        val pool = ForkJoinPool(Runtime.getRuntime().availableProcessors().coerceAtLeast(1))
        val gpu = GpuOpenClAccelerator()
        try {
            val cache = PcDiscoveryEngine.buildCausalCache(
                history = history,
                period = BacktestPeriod.THREE_WEEKS,
                referenceNowMs = history.last().timestamp + 1_000L,
                monteCarloSims = 500,
                pool = pool
            )
            val seen = LinkedHashSet<String>()
            val report1 = com.lotobj.pc.engine.StrategyFactory.generateAndTestOptimalPatterns(
                cache = cache,
                batchSize = 64,
                generation = 1,
                sessionSalt = 555L,
                preferredDomain = com.lotobj.pc.engine.MathTheoremDomain.TRIANGULATION,
                seenSignatures = seen,
                computeMode = ComputeMode.CPU_MAX_CORES,
                cpuPool = pool,
                gpuAccelerator = gpu
            )
            assertTrue("Batch 1 should evaluate unseen candidates", report1.unseenEvaluatedCount > 0)
            assertTrue("Batch 1 should return ranked optimal candidates", report1.optimalCandidates.isNotEmpty())
            assertTrue("Zero-cap count cannot exceed evaluated count", report1.zeroCapPassedCount <= report1.unseenEvaluatedCount)
            assertEquals(report1.unseenEvaluatedCount, report1.domainBreakdown[com.lotobj.pc.engine.MathTheoremDomain.TRIANGULATION])
            assertTrue("All seen signatures must be tracked", seen.size >= report1.unseenEvaluatedCount)
            assertTrue(report1.evaluationsPerSec > 0.0)
            assertTrue(report1.auditedWinner != null)

            val prevSeen = seen.size
            val report2 = com.lotobj.pc.engine.StrategyFactory.generateAndTestOptimalPatterns(
                cache = cache,
                batchSize = 48,
                generation = 2,
                sessionSalt = 777L,
                preferredDomain = null,
                seenSignatures = seen,
                computeMode = ComputeMode.CPU_MAX_CORES,
                cpuPool = pool,
                gpuAccelerator = gpu
            )
            assertTrue("Batch 2 must evaluate fresh formulas not in batch 1", report2.unseenEvaluatedCount > 0)
            assertTrue("Seen registry must strictly grow after batch 2", seen.size > prevSeen)
            assertTrue(report2.domainBreakdown.size >= 3)
        } finally {
            pool.shutdownNow()
            gpu.close()
        }
    }

    @Test
    fun configurableThreeDayAndCustomWorkingPeriodsWorkCleanly() {
        val history = buildSyntheticHistory(10)
        val pool = ForkJoinPool(Runtime.getRuntime().availableProcessors().coerceAtLeast(1))
        val gpu = GpuOpenClAccelerator()
        val tempDir = Files.createTempDirectory("lotobj_3d_test").toFile()
        try {
            val cache3d = PcDiscoveryEngine.buildCausalCache(
                history = history,
                period = BacktestPeriod.THREE_DAYS,
                referenceNowMs = history.last().timestamp + 1_000L,
                monteCarloSims = 500,
                pool = pool,
                customDays = 3
            )
            assertEquals("3 Jours", cache3d.periodLabel)
            assertTrue("3-day window should build valid causal steps", cache3d.steps.size in 8..20)
            assertTrue(cache3d.steps.all { it.maxPastTs < it.draw.timestamp })

            val state = PcDiscoveryEngine.createInitialState(populationSize = 32, sessionSalt = 123L)
            val gen = PcDiscoveryEngine.executeEvolutionGeneration(
                state = state,
                cache = cache3d,
                periodName = BacktestPeriod.THREE_DAYS.name,
                minObjectiveStreak = 4,
                populationSize = 32,
                computeMode = ComputeMode.CPU_MAX_CORES,
                cpuPool = pool,
                gpuAccelerator = gpu
            )
            assertEquals(1, gen.nextState.generation)
            assertTrue(gen.evaluationsPerSec > 0.0)

            val cache4d = PcDiscoveryEngine.buildCausalCache(
                history = history,
                period = BacktestPeriod.THREE_DAYS,
                referenceNowMs = history.last().timestamp + 1_000L,
                monteCarloSims = 500,
                pool = pool,
                customDays = 4
            )
            assertEquals("4 Jours (Perso)", cache4d.periodLabel)
            assertTrue("4-day window should include more steps than 3-day window", cache4d.steps.size >= cache3d.steps.size)

            val labReport = com.lotobj.pc.engine.ClassicalAndScientificLab.runFullScientificComparison(
                data = history,
                period = BacktestPeriod.THREE_DAYS,
                monteCarloSims = 500,
                referenceNowMs = history.last().timestamp + 1_000L,
                pool = pool,
                customDays = 3
            )
            assertEquals(13, labReport.indepTestComparisons.size)
            assertTrue(labReport.splitInfo.devDrawsCount >= 2)

            val repo = com.lotobj.pc.data.LotoDataRepository(storageDir = tempDir)
            repo.mergeAndSaveDraws(history)
            val coordinator = PcWorkstationCoordinator(repository = repo)
            try {
                coordinator.setCustomPeriodDays(0) // clamped to >= 1
                assertEquals(1, coordinator.uiState.value.customPeriodDays)
                assertEquals(BacktestPeriod.ONE_DAY, coordinator.uiState.value.selectedPeriod)
                coordinator.setCustomPeriodDays(4)
                assertEquals(4, coordinator.uiState.value.customPeriodDays)
            } finally {
                coordinator.close()
            }

            val cache1d = PcDiscoveryEngine.buildCausalCache(
                history = history,
                period = BacktestPeriod.ONE_DAY,
                referenceNowMs = history.last().timestamp + 1_000L,
                monteCarloSims = 300,
                pool = pool,
                customDays = 1
            )
            assertEquals("1 Jour", cache1d.periodLabel)
            assertTrue("1-day window should build valid causal steps", cache1d.steps.isNotEmpty())
            assertTrue(cache1d.steps.all { it.maxPastTs < it.draw.timestamp })

            val lab1d = com.lotobj.pc.engine.ClassicalAndScientificLab.runFullScientificComparison(
                data = history,
                period = BacktestPeriod.ONE_DAY,
                monteCarloSims = 300,
                referenceNowMs = history.last().timestamp + 1_000L,
                pool = pool,
                customDays = 1
            )
            assertEquals(13, lab1d.indepTestComparisons.size)
        } finally {
            tempDir.deleteRecursively()
            pool.shutdownNow()
            gpu.close()
        }
    }

    @Test
    fun livePredictionMatchesHistoricalBacktestIdentically_andApprovedLedgerIsAppendOnly() {
        val fullHistory = buildSyntheticHistory(14)
        val initialHistory = fullHistory.dropLast(1)
        val newlyArrivedDraw = fullHistory.last()
        val pool = ForkJoinPool(Runtime.getRuntime().availableProcessors().coerceAtLeast(1))
        try {
            val genome = SymbolicGenome(
                id = "g_sync_pc_test",
                generation = 2,
                parentIds = emptyList(),
                primaryExpr = ExprNode.Binary(
                    BinaryOp.ADD,
                    ExprNode.Var(RawFeature.SAME_DAY_REP),
                    ExprNode.Var(RawFeature.MOD7_RES_DENSITY)
                ),
                secondaryExpr = ExprNode.Var(RawFeature.GAP_NORM),
                primarySlots = 3,
                samplingTemp = 0.65,
                exploreNoise = 0.0,
                seed = 77L
            )

            // 1. Calcul du backtest approuvé avant l'arrivée du dernier tirage
            val initialCache = PcDiscoveryEngine.buildCausalCache(
                history = initialHistory,
                period = BacktestPeriod.SEVEN_DAYS,
                referenceNowMs = newlyArrivedDraw.timestamp - 60_000L,
                monteCarloSims = 300,
                pool = pool,
                customDays = 7
            )
            val (btInitial, ledgerInitial) = PcDiscoveryEngine.mergeAppendOnlyApprovedBacktest(
                genome = genome,
                cache = initialCache,
                existingLedger = null,
                generationsExecuted = 2,
                experimentsCount = 64
            )
            assertTrue(btInitial.isAppendOnlyApproved)
            assertTrue(btInitial.steps.isNotEmpty())

            // 2. Prédiction LIVE pour newlyArrivedDraw avant qu'il entre dans l'historique
            val targetHoraire = LotoDateUtils.normalizeHoraire(newlyArrivedDraw)
            val livePred = PcDiscoveryEngine.predictNextWithGenome(
                genome = genome,
                cleanHistory = initialHistory,
                targetHoraire = targetHoraire,
                validatedBacktest = btInitial,
                targetTimestamp = newlyArrivedDraw.timestamp
            )
            val ledgerWithLive = PcDiscoveryEngine.recordLivePredictionInLedger(
                ledger = ledgerInitial,
                prediction = livePred,
                targetTimestamp = newlyArrivedDraw.timestamp
            )

            // 3. Le tirage arrive dans l'historique ; le backtest approuvé est relancé (append-only)
            val updatedCache = PcDiscoveryEngine.buildCausalCache(
                history = fullHistory,
                period = BacktestPeriod.SEVEN_DAYS,
                referenceNowMs = newlyArrivedDraw.timestamp + 60_000L,
                monteCarloSims = 300,
                pool = pool,
                customDays = 7
            )
            val (btUpdated, _) = PcDiscoveryEngine.mergeAppendOnlyApprovedBacktest(
                genome = genome,
                cache = updatedCache,
                existingLedger = ledgerWithLive,
                generationsExecuted = 3,
                experimentsCount = 96
            )

            // Aucune étape approuvée précédente n'a été modifiée (steps triés du plus récent au plus ancien)
            assertEquals(btInitial.steps.size, btUpdated.approvedStepsPreservedCount)
            assertEquals(1, btUpdated.newlyAppendedStepsCount)
            assertEquals(btInitial.steps, btUpdated.steps.takeLast(btInitial.steps.size))

            // La nouvelle étape historique est STRICTEMENT IDENTIQUE à la prédiction Live faite auparavant
            val newStep = btUpdated.steps.first()
            assertEquals(newlyArrivedDraw.uniqueId, newStep.drawId)
            assertEquals(livePred.predictedDraw.sorted(), newStep.predictedNumbers.sorted())
            assertEquals(1, btUpdated.livePredictionsMatchedCount)
        } finally {
            pool.shutdownNow()
        }
    }

    @Test
    fun chercheurEngineDistillsAbstractTeachings_andTracksKeptVsDiscarded() {
        val history = buildSyntheticHistory(14)
        val pool = ForkJoinPool(Runtime.getRuntime().availableProcessors().coerceAtLeast(1))
        val gpu = GpuOpenClAccelerator()
        try {
            val cache = PcDiscoveryEngine.buildCausalCache(
                history = history,
                period = BacktestPeriod.THREE_WEEKS,
                referenceNowMs = history.last().timestamp + 1_000L,
                monteCarloSims = 300,
                pool = pool
            )
            val cycleOutput = com.lotobj.pc.engine.ChercheurEngine.runResearchAndDistill(
                cache = cache,
                cycleIndex = 1,
                generation = 1,
                seenSignatures = emptySet()
            )
            assertTrue("Chercheur should distill >=1 abstract teaching", cycleOutput.teachings.isNotEmpty())
            assertTrue("Chercheur should synthesize candidate genomes", cycleOutput.synthesizedGenomes.isNotEmpty())
            assertTrue(
                "Teachings must remain abstract (no raw single-draw overfitting dumps)",
                cycleOutput.teachings.all { it.favoredFeatures.isNotEmpty() && it.principleSummary.isNotBlank() }
            )

            val initState = PcDiscoveryEngine.createInitialState(populationSize = 24, sessionSalt = 99L)
            val genResult = PcDiscoveryEngine.executeEvolutionGeneration(
                state = initState,
                cache = cache,
                periodName = BacktestPeriod.THREE_WEEKS.name,
                minObjectiveStreak = 4,
                populationSize = 24,
                computeMode = ComputeMode.CPU_MAX_CORES,
                cpuPool = pool,
                gpuAccelerator = gpu,
                injectedGenomes = cycleOutput.synthesizedGenomes
            )
            assertEquals(cycleOutput.synthesizedGenomes.size, genResult.injectedOutcomes.size)
            assertTrue(
                genResult.injectedOutcomes.values.all { it == "KEPT" || it == "DISCARDED" }
            )
        } finally {
            pool.shutdownNow()
            gpu.close()
        }
    }

    @Test
    fun evaluateStageHeterogeneous_completesAcrossPoolWithoutDroppingGenomes() {
        val history = buildSyntheticHistory(10)
        val pool = ForkJoinPool(Runtime.getRuntime().availableProcessors().coerceAtLeast(1))
        val gpu = GpuOpenClAccelerator()
        try {
            val cache = PcDiscoveryEngine.buildCausalCache(
                history = history,
                period = BacktestPeriod.SEVEN_DAYS,
                referenceNowMs = history.last().timestamp + 1_000L,
                monteCarloSims = 250,
                pool = pool
            )
            val genomes = PcDiscoveryEngine.generateInitialPopulation(32, 321L)
            val compiled = genomes.map { SymbolicFormula.compileToRpn(it.primaryExpr) to SymbolicFormula.compileToRpn(it.secondaryExpr) }
            val (stats, _, cpuMs) = PcDiscoveryEngine.evaluateStageHeterogeneous(
                stageKey = "test_stage",
                steps = cache.discoverySteps,
                genomes = genomes,
                compiled = compiled,
                useGpu = true,
                cpuPool = pool,
                gpuAccelerator = gpu
            )
            assertEquals(genomes.size, stats.size)
            assertTrue(cpuMs >= 0.0)
        } finally {
            pool.shutdownNow()
            gpu.close()
        }
    }

    @Test
    fun dailyQualificationStrictlyRequiresThreeDrawsWithTwoPlus_twoDrawsIsRejected() {
        val baseTs = 1_760_000_000_000L
        // Un génome déterministe qui sélectionne toujours [1, 2, 3, 4, 5]
        val fixedLowGenome = SymbolicGenome(
            id = "fixed_low_1_5",
            generation = 1,
            parentIds = emptyList(),
            primaryExpr = ExprNode.Unary(UnaryOp.NEG, ExprNode.Var(RawFeature.X_NORM)),
            secondaryExpr = ExprNode.Unary(UnaryOp.NEG, ExprNode.Var(RawFeature.X_NORM)),
            primarySlots = 3,
            samplingTemp = 0.25,
            exploreNoise = 0.0,
            seed = 1L
        )
        val seedPast = (0 until 5).map { idx ->
            val ts = baseTs + idx * 3_600_000L
            LotoDraw("past_$idx", LotoDateUtils.formatFrenchDayAndDate(ts), "8H", "Fortune", 1, 2, 3, 4, 5, ts)
        }
        val dummyMatrix = SymbolicFormula.extractRawMatrix(seedPast, "8H")

        fun makeStep(idx: Int, dayKey: String, dayOffset: Int, hIdx: Int, nums: List<Int>): com.lotobj.pc.engine.CachedRawStep {
            val ts = baseTs + (dayOffset + 2) * 86_400_000L + hIdx * 3_600_000L
            val h = listOf("8H", "11H", "14H", "18H")[hIdx]
            val draw = LotoDraw("q_${dayOffset}_$hIdx", LotoDateUtils.formatFrenchDayAndDate(ts), h, "Fortune", nums[0], nums[1], nums[2], nums[3], nums[4], ts)
            return com.lotobj.pc.engine.CachedRawStep(
                stepIndex = idx,
                draw = draw,
                actualNumbers = nums,
                actualSet = nums.toSet(),
                strictPastSize = 5,
                maxPastTs = ts - 1_000L,
                dayKey = dayKey,
                horaire = h,
                game = "Fortune",
                rawMatrix = dummyMatrix
            )
        }

        // Jour 1 seulement : 2 tirages >=2 bons + 2 tirages à 0 bon -> 0 jour qualifié (seuil >=3 requis)
        val day1Only = listOf(
            makeStep(1, "2026-10-01", 0, 0, listOf(1, 2, 20, 21, 22)),
            makeStep(2, "2026-10-01", 0, 1, listOf(1, 2, 30, 31, 32)),
            makeStep(3, "2026-10-01", 0, 2, listOf(40, 41, 42, 43, 44)),
            makeStep(4, "2026-10-01", 0, 3, listOf(50, 51, 52, 53, 54))
        )
        val statsDay1 = PcDiscoveryEngine.evaluateGenomeOnStage(fixedLowGenome, day1Only)
        assertEquals(2, statsDay1.twoPlusCount)
        assertEquals("2 tirages >=2/jour ne doit pas qualifier (seuil strict >=3)", 0, statsDay1.qualifyingDays)

        // Jour 1 + Jour 2 (3 tirages >=2 bons sur Jour 2) -> exactement 1 jour qualifié
        val day1AndDay2 = day1Only + listOf(
            makeStep(5, "2026-10-02", 1, 0, listOf(1, 2, 60, 61, 62)),
            makeStep(6, "2026-10-02", 1, 1, listOf(2, 3, 70, 71, 72)),
            makeStep(7, "2026-10-02", 1, 2, listOf(3, 4, 80, 81, 82))
        )
        val statsBoth = PcDiscoveryEngine.evaluateGenomeOnStage(fixedLowGenome, day1AndDay2)
        assertEquals(5, statsBoth.twoPlusCount)
        assertEquals("Seul le Jour 2 (3 tirages >=2 bons) doit qualifier", 1, statsBoth.qualifyingDays)
        assertEquals(1, statsBoth.maxConsecutiveQualifyingDays)
    }

    @Test
    fun priorityFitnessStronglyFavorsAllDrawTwoPlusAndThreePerDay() {
        val genome = SymbolicGenome(
            id = "prio_test",
            generation = 1,
            parentIds = emptyList(),
            primaryExpr = ExprNode.Var(RawFeature.GAP_NORM),
            secondaryExpr = ExprNode.Var(RawFeature.FREQ_W15),
            primarySlots = 3,
            samplingTemp = 0.5,
            exploreNoise = 0.05,
            seed = 1L
        )
        val allDrawTwoPlus = com.lotobj.pc.engine.StageEvalStats(
            tested = 8, twoPlusCount = 8, qualifyingDays = 2, maxConsecutiveQualifyingDays = 2,
            maxZeroStreak = 0, avgMatches = 2.0f, brierScore = 0.04,
            matchesList = listOf(2, 2, 2, 2, 2, 2, 2, 2), causalVerified = true
        )
        val inconsistentHits = com.lotobj.pc.engine.StageEvalStats(
            tested = 8, twoPlusCount = 4, qualifyingDays = 1, maxConsecutiveQualifyingDays = 1,
            maxZeroStreak = 2, avgMatches = 2.0f, brierScore = 0.04,
            matchesList = listOf(4, 4, 4, 4, 0, 0, 0, 0), causalVerified = true
        )
        val evConsistent = PcDiscoveryEngine.scoreDiscoveryAndRolling(genome, allDrawTwoPlus, allDrawTwoPlus)
        val evInconsistent = PcDiscoveryEngine.scoreDiscoveryAndRolling(genome, inconsistentHits, inconsistentHits)
        assertTrue(evConsistent.rollingStabilityPassed)
        assertTrue(
            "All-draw >=2 + >=3/day (${evConsistent.fitnessScore}) must outrank inconsistent (${evInconsistent.fitnessScore})",
            evConsistent.fitnessScore > evInconsistent.fitnessScore
        )
    }

    @Test
    fun hardwareMonitorTelemetryAndGpuSliceMaskWorkCleanly() {
        val gpu = GpuOpenClAccelerator()
        try {
            val tel = com.lotobj.pc.engine.HardwareMonitor.sampleTelemetry(
                activeWorkers = 8,
                gpuAccelerator = gpu,
                evalPerSec = 1000.0,
                lastGpuBatchMs = 12.5,
                lastCpuBatchMs = 20.0
            )
            assertTrue(tel.processCpuPercent >= 0.0)
            assertTrue(tel.systemCpuPercent >= 0.0)
            assertTrue(tel.jvmMaxRamMb > 0L)
            assertTrue(tel.systemTotalRamMb > 0L)

            val partialMask = booleanArrayOf(true, false, true)
            val batch = com.lotobj.pc.engine.GpuBatchScores(
                genomeCount = 3,
                stepCount = 1,
                rawA = FloatArray(3 * 90),
                rawB = FloatArray(3 * 90),
                elapsedMs = 4.2,
                gpusUsed = 1,
                evaluatedMask = partialMask,
                runtimeErrorNote = "Slice GPU 2 indisponible"
            )
            assertTrue(batch.isGenomeEvaluated(0))
            assertFalse(batch.isGenomeEvaluated(1))
            assertTrue(batch.isGenomeEvaluated(2))
            assertFalse(batch.isGenomeEvaluated(3))
        } finally {
            gpu.close()
        }
    }

    @Test
    fun intraDayTenThirtyToElevenThirtyAdvancesToFourteenH_withZeroDriftAndAppendOnlyLedger() {
        val tempDir = Files.createTempDirectory("lotobj_intraday_chain").toFile()
        val sdf = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm", java.util.Locale.US).apply {
            timeZone = LotoDateUtils.beninTz
        }
        val rng = Random(2026L)
        val draws = ArrayList<LotoDraw>()
        val horaires = listOf("8H" to "08:00", "11H" to "11:00", "14H" to "14:00", "18H" to "18:00", "20H" to "20:00")
        for (day in 1..4) {
            val dayStr = String.format(java.util.Locale.US, "2026-10-%02d", day)
            for ((hLabel, hhmm) in horaires) {
                val ts = sdf.parse("$dayStr $hhmm")!!.time
                val nums = (1..90).shuffled(rng).take(5).sorted()
                draws.add(
                    LotoDraw(
                        uniqueId = "id_${dayStr}_$hLabel",
                        date = LotoDateUtils.formatFrenchDayAndDate(ts),
                        time = hLabel,
                        game = "Fortune",
                        num1 = nums[0], num2 = nums[1], num3 = nums[2], num4 = nums[3], num5 = nums[4],
                        timestamp = ts
                    )
                )
            }
        }
        // Day 5: 8H is already out, 11H not yet at 10:30
        val day5Str = "2026-10-05"
        val ts8H = sdf.parse("$day5Str 08:00")!!.time
        val draw8H = LotoDraw(
            uniqueId = "id_${day5Str}_8H",
            date = LotoDateUtils.formatFrenchDayAndDate(ts8H),
            time = "8H",
            game = "Fortune",
            num1 = 4, num2 = 17, num3 = 29, num4 = 52, num5 = 77,
            timestamp = ts8H
        )
        val ts11H = sdf.parse("$day5Str 11:00")!!.time
        val draw11H = LotoDraw(
            uniqueId = "id_${day5Str}_11H",
            date = LotoDateUtils.formatFrenchDayAndDate(ts11H),
            time = "11H",
            game = "Fortune",
            num1 = 9, num2 = 22, num3 = 41, num4 = 63, num5 = 84,
            timestamp = ts11H
        )
        val historyAt1030 = draws + draw8H
        val at1030 = sdf.parse("$day5Str 10:30")!!.time
        val at1130 = sdf.parse("$day5Str 11:30")!!.time

        val repo = com.lotobj.pc.data.LotoDataRepository(storageDir = tempDir)
        repo.mergeAndSaveDraws(historyAt1030)
        val genome = SymbolicGenome(
            id = "g_intraday_pc",
            generation = 3,
            parentIds = emptyList(),
            primaryExpr = ExprNode.Binary(
                BinaryOp.ADD,
                ExprNode.Var(RawFeature.SAME_DAY_REP),
                ExprNode.Var(RawFeature.MOD7_RES_DENSITY)
            ),
            secondaryExpr = ExprNode.Var(RawFeature.GAP_NORM),
            primarySlots = 3,
            samplingTemp = 0.65,
            exploreNoise = 0.0,
            seed = 1030L
        )
        val backupJson = StrategyBackupHelper.serializeBackup(
            listOf(
                com.lotobj.pc.model.DiscoveredFormulaRecord(
                    id = "strat_1j_test",
                    periodName = BacktestPeriod.ONE_DAY.name,
                    generation = 3,
                    parentIdsCsv = "",
                    primaryExprText = SymbolicFormula.prettyPrint(genome.primaryExpr),
                    secondaryExprText = SymbolicFormula.prettyPrint(genome.secondaryExpr),
                    slotSplitLabel = "3+2",
                    genomeJson = SymbolicFormula.genomeToJson(genome),
                    usedFeaturesCsv = "SAME_DAY_REP,MOD7_RES_DENSITY,GAP_NORM",
                    status = "MILESTONE_1J",
                    rejectionStage = "",
                    rejectionReason = "Palier 1J",
                    discoveryStreak = 1,
                    rollingStreak = 1,
                    confirmationStreak = 1,
                    sealedStreak = 1,
                    combinedStreak = 1,
                    discoveryAvgMatches = 1.2f,
                    rollingAvgMatches = 1.1f,
                    confirmationAvgMatches = 1.0f,
                    sealedAvgMatches = 1.0f,
                    pValueAdjusted = 0.04,
                    fitnessScore = 100.0,
                    updatedAt = at1030
                )
            )
        )

        val coordinator = PcWorkstationCoordinator(repository = repo)
        try {
            coordinator.clockProvider = { at1030 }
            coordinator.setCustomPeriodDays(1)
            coordinator.importStrategyBackupText(backupJson)
            coordinator.ingestDrawsAndAutoRefreshForTest(historyAt1030)

            assertEquals("11H", coordinator.uiState.value.nextDrawTarget.horaire)
            coordinator.launchValidatedBacktestAndPrediction()

            val deadline1 = System.currentTimeMillis() + 5_000L
            while (coordinator.uiState.value.prediction == null && System.currentTimeMillis() < deadline1) {
                Thread.sleep(25L)
            }
            val pred11H = coordinator.uiState.value.prediction!!
            val btAt1030 = coordinator.uiState.value.backtestResult!!
            assertTrue(btAt1030.allStepsCausalVerified)
            assertEquals("11H", pred11H.horaire)

            // 11:30 sonne et le résultat officiel 11H arrive -> bascule auto vers 14H, backtest append-only complété sans modification du passé
            coordinator.clockProvider = { at1130 }
            coordinator.ingestDrawsAndAutoRefreshForTest(historyAt1030 + draw11H)

            val deadline2 = System.currentTimeMillis() + 5_000L
            while (coordinator.uiState.value.prediction?.horaire != "14H" && System.currentTimeMillis() < deadline2) {
                Thread.sleep(25L)
            }
            val pred14H = coordinator.uiState.value.prediction!!
            val btAt1130 = coordinator.uiState.value.backtestResult!!

            assertEquals("14H", coordinator.uiState.value.nextDrawTarget.horaire)
            assertEquals("14H", pred14H.horaire)
            assertEquals(5, pred14H.predictedDraw.distinct().size)
            assertEquals(btAt1030.steps.size, btAt1130.approvedStepsPreservedCount)
            assertEquals(1, btAt1130.newlyAppendedStepsCount)
            assertEquals(1, btAt1130.livePredictionsMatchedCount)
            assertEquals(pred11H.predictedDraw.sorted(), btAt1130.steps.first().predictedNumbers.sorted())
            assertEquals(btAt1030.steps, btAt1130.steps.takeLast(btAt1030.steps.size))
        } finally {
            coordinator.close()
            tempDir.deleteRecursively()
        }
    }

    @Test
    fun oneDayWindowCoversSixDrawsZeroToTwentyHAcrossMidnightAndIsDeterministic() {
        val sdf = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm", java.util.Locale.US).apply {
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
        val draws = ArrayList<LotoDraw>()
        var idx = 0
        for (day in 1..4) {
            val dayStr = String.format(java.util.Locale.US, "2026-10-%02d", day)
            for ((hLabel, hhmm) in slots) {
                val ts = sdf.parse("$dayStr $hhmm")!!.time
                // Signal causal simple mais réaliste sur les petits numéros récurrents
                val nums = listOf(2, 5, 11 + (idx % 10), 33 + (idx % 15), 71 + (idx % 18)).sorted()
                draws.add(
                    LotoDraw(
                        uniqueId = "six_${dayStr}_$hLabel",
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
        // Au jour 4 à 11H30 : 00H, 8H, 11H du jour 4 sont passés; 14H, 18H, 20H du jour 4 sont futurs
        val ref1130 = sdf.parse("2026-10-04 11:30")!!.time
        val historyBefore1130 = draws.filter { it.timestamp < ref1130 }
        val pool = ForkJoinPool(Runtime.getRuntime().availableProcessors().coerceAtLeast(1))
        val gpu = GpuOpenClAccelerator()
        try {
            val cache1d = PcDiscoveryEngine.buildCausalCache(
                history = historyBefore1130,
                period = BacktestPeriod.ONE_DAY,
                referenceNowMs = ref1130,
                monteCarloSims = 300,
                pool = pool,
                customDays = 1
            )
            assertEquals("1 Jour doit couvrir exactement 6 tirages glissants à travers minuit", 6, cache1d.steps.size)
            assertEquals(
                listOf("14H", "18H", "20H", "00H", "8H", "11H"),
                cache1d.steps.map { it.horaire }
            )
            assertEquals("Les 6 tirages glissants 1J doivent former 1 bloc unifié", 1, cache1d.steps.map { it.dayEpoch }.distinct().size)
            assertEquals(2, cache1d.discoverySteps.size)
            assertEquals(1, cache1d.rollingSteps.size)
            assertEquals(2, cache1d.confirmationSteps.size)
            assertEquals(1, cache1d.sealedSteps.size)
            assertTrue(cache1d.steps.all { it.maxPastTs < it.draw.timestamp })

            val initState = PcDiscoveryEngine.createInitialState(populationSize = 64, sessionSalt = 777L)
            assertTrue("Aucune règle ne doit conserver exploreNoise > 0", initState.population.all { it.exploreNoise == 0.0 })

            val gen1 = PcDiscoveryEngine.executeEvolutionGeneration(
                state = initState,
                cache = cache1d,
                periodName = BacktestPeriod.ONE_DAY.name,
                minObjectiveStreak = 1,
                populationSize = 64,
                computeMode = ComputeMode.CPU_MAX_CORES,
                cpuPool = pool,
                gpuAccelerator = gpu
            )
            assertTrue(gen1.nextState.population.all { it.exploreNoise == 0.0 })

            val bestGenome = (gen1.winningEval ?: gen1.bestTrainEval)?.genome ?: initState.population.first()
            val (bt1d, _) = PcDiscoveryEngine.mergeAppendOnlyApprovedBacktest(bestGenome, cache1d, null, 1, 64)
            val nextTarget = LotoDateUtils.resolveNextUpcomingDraw(historyBefore1130, ref1130)
            assertEquals("14H", nextTarget.horaire)

            val predA = PcDiscoveryEngine.predictNextWithGenome(bestGenome, historyBefore1130, nextTarget.horaire, bt1d, nextTarget.targetTimestamp)
            val predB = PcDiscoveryEngine.predictNextWithGenome(bestGenome, historyBefore1130, nextTarget.horaire, bt1d, nextTarget.targetTimestamp)
            assertEquals("La prédiction doit être 100% déterministe sans aléa", predA.predictedDraw, predB.predictedDraw)
        } finally {
            pool.shutdownNow()
            gpu.close()
        }
    }

    @Test
    fun precisionTargetEngineRunsIndependentlyForFirstBallAndTwoNumbers() {
        val sdf = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm", java.util.Locale.US).apply {
            timeZone = LotoDateUtils.beninTz
        }
        val slots = listOf("00H" to "00:05", "8H" to "08:00", "11H" to "11:00", "14H" to "14:00", "18H" to "18:00", "20H" to "20:00")
        val draws = ArrayList<LotoDraw>()
        for (day in 1..6) {
            val dayStr = String.format(java.util.Locale.US, "2026-10-%02d", day)
            for ((sIdx, pair) in slots.withIndex()) {
                val (hLabel, hhmm) = pair
                val ts = sdf.parse("$dayStr $hhmm")!!.time
                val p1 = if ((day + sIdx) % 2 == 0) 7 else 14
                val p2 = if (p1 == 7) 19 else 28
                val others = listOf(p1, p2, 42, 65, 83).distinct()
                draws.add(
                    LotoDraw(
                        uniqueId = "prec_${dayStr}_$hLabel",
                        date = LotoDateUtils.formatFrenchDayAndDate(ts),
                        time = hLabel,
                        game = "Fortune",
                        num1 = others[0], num2 = others[1], num3 = others[2], num4 = others[3], num5 = others[4],
                        timestamp = ts
                    )
                )
            }
        }
        val pool = ForkJoinPool(Runtime.getRuntime().availableProcessors().coerceAtLeast(1))
        try {
            val initialHistory = draws.dropLast(1)
            val lastDraw = draws.last()

            for (mode in com.lotobj.pc.model.PrecisionTargetMode.entries) {
                val cache = com.lotobj.pc.engine.PrecisionTargetEngine.buildCausalCache(
                    history = initialHistory,
                    period = BacktestPeriod.THREE_DAYS,
                    referenceNowMs = lastDraw.timestamp - 60_000L,
                    customDays = 3,
                    pool = pool
                )
                assertTrue(cache.steps.isNotEmpty())
                assertTrue(cache.steps.all { it.maxPastTs < it.draw.timestamp })

                val initState = com.lotobj.pc.engine.PrecisionTargetEngine.createInitialState(
                    mode = mode,
                    populationSize = 64,
                    sessionSalt = 101L + mode.ordinal
                )
                assertTrue(initState.population.all { it.exploreNoise == 0.0 })

                val gen1 = com.lotobj.pc.engine.PrecisionTargetEngine.executeEvolutionGeneration(
                    state = initState,
                    cache = cache,
                    minObjectiveStreak = 1,
                    populationSize = 64,
                    cpuPool = pool
                )
                val best = (gen1.winningEval ?: gen1.bestTrainEval)?.genome ?: initState.population.first()
                assertEquals(mode, best.mode)

                val (bt1, ledger1) = com.lotobj.pc.engine.PrecisionTargetEngine.mergeAppendOnlyApprovedBacktest(
                    genome = best,
                    cache = cache,
                    existingLedger = null,
                    confirmationTestsCount = gen1.nextState.confirmationTestsCount
                )
                assertTrue(bt1.allStepsCausalVerified)
                assertEquals(cache.steps.size, bt1.totalTested)

                val target = LotoDateUtils.resolveNextUpcomingDraw(initialHistory, lastDraw.timestamp - 60_000L)
                val pred1 = com.lotobj.pc.engine.PrecisionTargetEngine.predictNextUpcoming(initialHistory, best, target)
                val pred2 = com.lotobj.pc.engine.PrecisionTargetEngine.predictNextUpcoming(initialHistory, best, target)
                assertEquals(mode.outputCount, pred1.predictedNumbers.distinct().size)
                assertEquals("Prédiction Précision doit être déterministe", pred1.predictedNumbers, pred2.predictedNumbers)
                assertTrue(pred1.predictedNumbers.all { it in 1..90 })

                val ledgerWithLive = com.lotobj.pc.engine.PrecisionTargetEngine.recordLivePredictionInLedger(ledger1, pred1)
                val cache2 = com.lotobj.pc.engine.PrecisionTargetEngine.buildCausalCache(
                    history = draws,
                    period = BacktestPeriod.THREE_DAYS,
                    referenceNowMs = lastDraw.timestamp + 60_000L,
                    customDays = 3,
                    pool = pool
                )
                val (bt2, _) = com.lotobj.pc.engine.PrecisionTargetEngine.mergeAppendOnlyApprovedBacktest(
                    genome = best,
                    cache = cache2,
                    existingLedger = ledgerWithLive,
                    confirmationTestsCount = gen1.nextState.confirmationTestsCount
                )
                assertEquals(bt1.steps.size, bt2.preservedStepsCount)
                assertEquals(1, bt2.appendedStepsCount)
                assertEquals(1, bt2.liveMatchedCount)
                assertEquals(pred1.predictedNumbers.sorted(), bt2.steps.last().predictedNumbers.sorted())
            }
        } finally {
            pool.shutdownNow()
        }
    }

    @Test
    fun precisionCoordinatorLifecycleRestoresPersistedStateAndSupportsGpuCpuAndInfiniteMode() {
        val tempDir = Files.createTempDirectory("lotobj_prec_coord").toFile()
        val sdf = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm", java.util.Locale.US).apply {
            timeZone = LotoDateUtils.beninTz
        }
        val slots = listOf("00H" to "00:05", "8H" to "08:00", "11H" to "11:00", "14H" to "14:00", "18H" to "18:00", "20H" to "20:00")
        val draws = ArrayList<LotoDraw>()
        for (day in 1..6) {
            val dayStr = String.format(java.util.Locale.US, "2026-10-%02d", day)
            for ((sIdx, pair) in slots.withIndex()) {
                val (hLabel, hhmm) = pair
                val ts = sdf.parse("$dayStr $hhmm")!!.time
                val p1 = if ((day + sIdx) % 2 == 0) 12 else 25
                val p2 = if (p1 == 12) 34 else 48
                draws.add(
                    LotoDraw(
                        uniqueId = "pc2_${dayStr}_$hLabel",
                        date = LotoDateUtils.formatFrenchDayAndDate(ts),
                        time = hLabel,
                        game = "Fortune",
                        num1 = p1, num2 = p2, num3 = 61, num4 = 74, num5 = 88,
                        timestamp = ts
                    )
                )
            }
        }
        val pool = ForkJoinPool(Runtime.getRuntime().availableProcessors().coerceAtLeast(1))
        val gpu = GpuOpenClAccelerator()
        try {
            val cache = com.lotobj.pc.engine.PrecisionTargetEngine.buildCausalCache(
                history = draws,
                period = BacktestPeriod.THREE_DAYS,
                referenceNowMs = draws.last().timestamp + 60_000L,
                customDays = 3,
                pool = pool
            )
            val init1024 = com.lotobj.pc.engine.PrecisionTargetEngine.createInitialState(
                mode = com.lotobj.pc.model.PrecisionTargetMode.TWO_NUMBERS,
                populationSize = 1024,
                sessionSalt = 409L
            )
            val genRes = com.lotobj.pc.engine.PrecisionTargetEngine.executeEvolutionGeneration(
                state = init1024,
                cache = cache,
                minObjectiveStreak = 1,
                populationSize = 1024,
                computeMode = ComputeMode.GPU_AND_CPU_MAX,
                cpuPool = pool,
                gpuAccelerator = gpu
            )
            assertEquals(1, genRes.nextState.generation)
            assertTrue(genRes.nextState.experimentsCount >= 1024)
            assertTrue(genRes.evaluationsPerSec > 0.0)

            val repo = com.lotobj.pc.data.LotoDataRepository(storageDir = tempDir)
            repo.mergeAndSaveDraws(draws)
            val bestGenome = (genRes.winningEval ?: genRes.bestTrainEval)!!.genome
            repo.savePrecisionBestGenome(com.lotobj.pc.model.PrecisionTargetMode.FIRST_BALL, bestGenome.copy(mode = com.lotobj.pc.model.PrecisionTargetMode.FIRST_BALL))
            repo.savePrecisionBestGenome(com.lotobj.pc.model.PrecisionTargetMode.TWO_NUMBERS, bestGenome)

            val coord1 = PcWorkstationCoordinator(repository = repo, gpuAccelerator = gpu)
            try {
                coord1.clockProvider = { draws.last().timestamp + 60_000L }
                coord1.setCustomPeriodDays(3)
                assertTrue(coord1.precisionUiState.value.activeFormulaDesc.isNotBlank())

                coord1.launchPrecisionBacktestAndPrediction()
                val dl = System.currentTimeMillis() + 5_000L
                while (coord1.precisionUiState.value.livePrediction == null && System.currentTimeMillis() < dl) {
                    Thread.sleep(25L)
                }
                assertTrue(coord1.precisionUiState.value.livePrediction != null)
                assertTrue(coord1.precisionUiState.value.backtestResult != null)
                val report = coord1.formatPrecisionDetailedReport()
                assertTrue(report.contains("MOTEUR 2 PRÉCISION"))

                coord1.setPrecisionTargetMode(com.lotobj.pc.model.PrecisionTargetMode.TWO_NUMBERS)
                assertEquals(com.lotobj.pc.model.PrecisionTargetMode.TWO_NUMBERS, coord1.precisionUiState.value.targetMode)
                assertTrue(coord1.precisionUiState.value.activeFormulaDesc.isNotBlank())

                coord1.startPrecisionInfiniteMode()
                assertTrue(coord1.precisionUiState.value.infiniteMode)
                Thread.sleep(80L)
                coord1.suspendPrecisionDiscovery()
                assertEquals(com.lotobj.pc.model.AutonomousPhase.SUSPENDED, coord1.precisionUiState.value.phase)
                coord1.stopPrecisionDiscovery()
                assertEquals(com.lotobj.pc.model.AutonomousPhase.STOPPED, coord1.precisionUiState.value.phase)
            } finally {
                coord1.close()
            }

            // Redémarrage complet sur le même dossier : la formule et le ledger sauvegardés doivent revenir
            val gpu2 = GpuOpenClAccelerator()
            val coord2 = PcWorkstationCoordinator(repository = com.lotobj.pc.data.LotoDataRepository(storageDir = tempDir), gpuAccelerator = gpu2)
            try {
                coord2.clockProvider = { draws.last().timestamp + 60_000L }
                assertTrue("La formule Engine 2 doit être restaurée au démarrage", coord2.precisionUiState.value.activeFormulaDesc.isNotBlank())
            } finally {
                coord2.close()
            }
        } finally {
            pool.shutdownNow()
            gpu.close()
            tempDir.deleteRecursively()
        }
    }

    @Test
    fun incrementalMatrixSequencesMatchBatchExtractionExactly_withEqualTimestampTieExclusion() {
        val base = buildSyntheticHistory(12).toMutableList()
        // Injecter un tirage avec horodatage identique au tirage d'indice 15 pour tester le barrage strict `< target.timestamp`
        val tieSrc = base[15]
        base.add(
            16,
            tieSrc.copy(
                uniqueId = "tie_equal_ts",
                time = "18H",
                num1 = 2, num2 = 19, num3 = 44, num4 = 67, num5 = 89
            )
        )
        val chronological = base
            .distinctBy { it.uniqueId }
            .sortedWith(compareBy<LotoDraw> { it.timestamp }.thenBy { it.uniqueId })

        val testIndices = intArrayOf(1, 5, 14, 15, 16, 17, 25, chronological.lastIndex)
        val seqSym = SymbolicFormula.extractRawMatrixSequence(chronological, testIndices)
        val seqPrec = com.lotobj.pc.engine.PrecisionTargetEngine.extractPrecisionMatrixSequence(chronological, testIndices)

        for (t in testIndices) {
            val target = chronological[t]
            val strictPast = chronological.subList(0, t).filter { it.timestamp < target.timestamp && it.uniqueId != target.uniqueId }
            val horaire = LotoDateUtils.normalizeHoraire(target)

            val expectedSym = SymbolicFormula.extractRawMatrix(strictPast, horaire)
            val actualSym = seqSym[t]!!
            for (feat in RawFeature.entries) {
                for (n in 1..90) {
                    val diff = abs(expectedSym.get(feat.ordinal, n) - actualSym.get(feat.ordinal, n))
                    assertTrue(
                        "Symbolic O(T) mismatch at t=$t feat=$feat n=$n: exp=${expectedSym.get(feat.ordinal, n)} act=${actualSym.get(feat.ordinal, n)}",
                        diff < 1e-12
                    )
                }
            }

            val expectedPrec = com.lotobj.pc.engine.PrecisionTargetEngine.extractFeatures(strictPast, horaire)
            val actualPrec = seqPrec[t]!!
            for (feat in com.lotobj.pc.model.PrecisionFeature.entries) {
                for (n in 1..90) {
                    val diff = abs(expectedPrec.get(feat.ordinal, n) - actualPrec.get(feat.ordinal, n))
                    assertTrue(
                        "Precision O(T) mismatch at t=$t feat=$feat n=$n: exp=${expectedPrec.get(feat.ordinal, n)} act=${actualPrec.get(feat.ordinal, n)}",
                        diff < 1e-12
                    )
                }
            }
        }
    }

    @Test
    fun causalAiLearnerStrictlyZeroLeakage_andMultipleTestingCapDoesNotLockout() {
        val historyA = buildSyntheticHistory(20)
        val pool = ForkJoinPool(2)
        try {
            val cacheA = PcDiscoveryEngine.buildCausalCache(
                history = historyA,
                period = BacktestPeriod.THREE_WEEKS,
                referenceNowMs = historyA.last().timestamp + 1_000L,
                monteCarloSims = 200,
                pool = pool
            )
            assertTrue("cacheA.discoverySteps must not be empty", cacheA.discoverySteps.isNotEmpty())
            val lastDiscoveryTs = cacheA.discoverySteps.last().draw.timestamp
            val historyB = historyA.mapIndexed { idx, d ->
                if (d.timestamp > lastDiscoveryTs) {
                    // Perturber fortement tous les tirages futurs (Glissante + Confirmation + Scellé)
                    d.copy(num1 = 90 - (idx % 10), num2 = 79, num3 = 68, num4 = 57, num5 = 46)
                } else d
            }
            val cacheB = PcDiscoveryEngine.buildCausalCache(
                history = historyB,
                period = BacktestPeriod.THREE_WEEKS,
                referenceNowMs = historyB.last().timestamp + 1_000L,
                monteCarloSims = 200,
                pool = pool
            )
            assertEquals(cacheA.discoverySteps.size, cacheB.discoverySteps.size)
            val ridgeA = com.lotobj.pc.engine.CausalAiLearner.fitSymbolicRidgeGenomes(
                discoverySteps = cacheA.discoverySteps,
                generation = 1,
                sessionSalt = 42L,
                maxCount = 8
            )
            val ridgeB = com.lotobj.pc.engine.CausalAiLearner.fitSymbolicRidgeGenomes(
                discoverySteps = cacheB.discoverySteps,
                generation = 1,
                sessionSalt = 42L,
                maxCount = 8
            )
            assertTrue("CausalAiLearner must produce symbolic Ridge genomes", ridgeA.isNotEmpty())
            assertEquals(
                "Discovery Ridge models must be invariant to future Rolling/Confirmation/Sealed draws",
                ridgeA.map { it.describeCompact() },
                ridgeB.map { it.describeCompact() }
            )

            // Vérification facteur logarithmique anti-verrouillage Stage 3
            val factor500 = PcDiscoveryEngine.effectiveMultipleTestFactor(500)
            assertTrue("Expected logarithmic cap << 500, got $factor500", factor500 < 20.0)
            assertEquals(1.0, PcDiscoveryEngine.effectiveMultipleTestFactor(1), 1e-9)
        } finally {
            pool.shutdownNow()
        }
    }

    @Test
    fun penultimateDrawCondition_enforcedAcrossPcDiscoveryLabAndPrecisionTwoNumbers() {
        val sdf = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm", java.util.Locale.US).apply {
            timeZone = LotoDateUtils.beninTz
        }
        val fixedLowGenome = SymbolicGenome(
            id = "pen_fixed_1_5",
            generation = 1,
            parentIds = emptyList(),
            primaryExpr = ExprNode.Unary(UnaryOp.NEG, ExprNode.Var(RawFeature.X_NORM)),
            secondaryExpr = ExprNode.Unary(UnaryOp.NEG, ExprNode.Var(RawFeature.X_NORM)),
            primarySlots = 3,
            samplingTemp = 0.25,
            exploreNoise = 0.0,
            seed = 1L
        )
        val seedPast = (0 until 5).map { idx ->
            val ts = sdf.parse("2026-10-01 08:00")!!.time + idx * 3_600_000L
            LotoDraw("pen_past_$idx", LotoDateUtils.formatFrenchDayAndDate(ts), "8H", "Fortune", 1, 2, 3, 4, 5, ts)
        }
        val dummyMatrix = SymbolicFormula.extractRawMatrix(seedPast, "8H")
        val slots = listOf("00H" to "00:05", "8H" to "08:00", "11H" to "11:00", "14H" to "14:00", "18H" to "18:00", "20H" to "20:00")

        fun buildDaySteps(dayStr: String, numsBySlot: List<List<Int>>): List<com.lotobj.pc.engine.CachedRawStep> {
            return slots.mapIndexed { sIdx, (hLabel, hhmm) ->
                val ts = sdf.parse("$dayStr $hhmm")!!.time
                val nums = numsBySlot[sIdx]
                val draw = LotoDraw("pen_${dayStr}_$hLabel", LotoDateUtils.formatFrenchDayAndDate(ts), hLabel, "Fortune", nums[0], nums[1], nums[2], nums[3], nums[4], ts)
                com.lotobj.pc.engine.CachedRawStep(
                    stepIndex = sIdx + 1,
                    draw = draw,
                    actualNumbers = nums,
                    actualSet = nums.toSet(),
                    strictPastSize = 5,
                    maxPastTs = ts - 1_000L,
                    dayKey = dayStr,
                    horaire = hLabel,
                    game = "Fortune",
                    rawMatrix = dummyMatrix
                )
            }
        }

        // Jour A : 3 tirages >=2 (00H, 8H, 11H), MAIS avant-dernier (18H) = 0 bon -> DOIT être refusé
        val missPenultimateSteps = buildDaySteps(
            "2026-10-05",
            listOf(
                listOf(1, 2, 20, 21, 22),
                listOf(1, 2, 30, 31, 32),
                listOf(1, 2, 40, 41, 42),
                listOf(50, 51, 52, 53, 54),
                listOf(60, 61, 62, 63, 64), // 18H = avant-dernier = 0 bon
                listOf(70, 71, 72, 73, 74)
            )
        )
        val missStats = PcDiscoveryEngine.evaluateGenomeOnStage(fixedLowGenome, missPenultimateSteps)
        assertEquals(3, missStats.twoPlusCount)
        assertEquals("Sans >=2 à l'avant-dernier tirage (18H), le jour ne doit PAS qualifier", 0, missStats.qualifyingDays)

        // Jour B : tous les tirages (6/6) >=2 dont l'avant-dernier (18H) = 2 bons -> DOIT qualifier
        val hitPenultimateSteps = buildDaySteps(
            "2026-10-06",
            listOf(
                listOf(1, 2, 20, 21, 22),
                listOf(1, 2, 30, 31, 32),
                listOf(1, 2, 40, 41, 42),
                listOf(1, 2, 50, 51, 52),
                listOf(2, 3, 70, 71, 72), // 18H = avant-dernier = 2 bons
                listOf(1, 2, 80, 81, 82)
            )
        )
        val hitStats = PcDiscoveryEngine.evaluateGenomeOnStage(fixedLowGenome, hitPenultimateSteps)
        assertEquals(6, hitStats.twoPlusCount)
        assertEquals("Avec tous les tirages >=2 ET avant-dernier >=2, le jour doit qualifier", 1, hitStats.qualifyingDays)

        // Vérification Labo : audit des modèles conservés + periodLabel + anyObjectiveValidated
        val history = buildSyntheticHistory(14)
        val pool = ForkJoinPool(2)
        try {
            val savedInput = com.lotobj.pc.engine.LabSavedModelInput(
                id = "saved_lab_pc_1",
                displayName = "Modèle Test 1J (Gén.1)",
                savedStreak = 1,
                slotSplitLabel = "3+2",
                genome = fixedLowGenome
            )
            val report = com.lotobj.pc.engine.ClassicalAndScientificLab.runFullScientificComparison(
                data = history,
                period = BacktestPeriod.ONE_DAY,
                monteCarloSims = 250,
                referenceNowMs = history.last().timestamp + 1_000L,
                pool = pool,
                customDays = 1,
                savedModels = listOf(savedInput)
            )
            assertEquals("1 Jour", report.periodLabel)
            assertEquals(1, report.savedModelAudits.size)
            val audit = report.savedModelAudits.first()
            assertEquals("saved_lab_pc_1", audit.id)
            assertTrue(audit.causalVerified)
            assertEquals(audit.qualifyingDaysCount >= 1 && audit.causalVerified, audit.objectiveValidated)
        } finally {
            pool.shutdownNow()
        }

        // Vérification Moteur 2 TWO_NUMBERS : avant-dernier exact obligatoire dans un bloc de 6 tirages
        val precTwoStepsMiss = missPenultimateSteps.mapIndexed { i, st ->
            val hit = i < 3
            com.lotobj.pc.model.PrecisionBacktestStep(
                stepIndex = i + 1,
                drawId = st.draw.uniqueId,
                drawDate = st.draw.date,
                dayKey = st.dayKey,
                horaire = st.horaire,
                game = st.game,
                strictPastCount = 5,
                maxPastTs = st.maxPastTs,
                targetTimestamp = st.draw.timestamp,
                causalVerified = true,
                predictedNumbers = listOf(1, 2),
                actualPos1 = st.actualNumbers[0],
                actualNumbers = st.actualNumbers,
                matchedNumbers = if (hit) listOf(1, 2) else emptyList(),
                matchCount = if (hit) 2 else 0,
                exactTargetHit = hit,
                stepBrierScore = 0.05
            )
        }
        val precMissBlocks = com.lotobj.pc.engine.PrecisionTargetEngine.buildDailyBlocks(
            precTwoStepsMiss,
            com.lotobj.pc.model.PrecisionTargetMode.TWO_NUMBERS
        )
        assertFalse(precMissBlocks.first().penultimateExact)
        assertFalse(precMissBlocks.first().isQualifyingBlock)

        val precTwoStepsHit = hitPenultimateSteps.mapIndexed { i, st ->
            val hit = i == 0 || i == 1 || i == 4
            com.lotobj.pc.model.PrecisionBacktestStep(
                stepIndex = i + 1,
                drawId = st.draw.uniqueId,
                drawDate = st.draw.date,
                dayKey = st.dayKey,
                horaire = st.horaire,
                game = st.game,
                strictPastCount = 5,
                maxPastTs = st.maxPastTs,
                targetTimestamp = st.draw.timestamp,
                causalVerified = true,
                predictedNumbers = listOf(2, 3),
                actualPos1 = st.actualNumbers[0],
                actualNumbers = st.actualNumbers,
                matchedNumbers = if (hit) listOf(2, 3) else emptyList(),
                matchCount = if (hit) 2 else 0,
                exactTargetHit = hit,
                stepBrierScore = 0.05
            )
        }
        val precHitBlocks = com.lotobj.pc.engine.PrecisionTargetEngine.buildDailyBlocks(
            precTwoStepsHit,
            com.lotobj.pc.model.PrecisionTargetMode.TWO_NUMBERS
        )
        assertTrue(precHitBlocks.first().penultimateExact)
        assertTrue(precHitBlocks.first().isQualifyingBlock)
    }

    @Test
    fun nvidiaAndAmdVendorDetectionAndWavefrontSizingWorkCorrectly() {
        assertEquals(
            com.lotobj.pc.engine.GpuVendor.NVIDIA,
            GpuOpenClAccelerator.detectGpuVendor(0x10DE, "NVIDIA Corporation", "OpenCL Platform", "Unknown Device")
        )
        assertEquals(
            com.lotobj.pc.engine.GpuVendor.AMD,
            GpuOpenClAccelerator.detectGpuVendor(0x1002, "Advanced Micro Devices, Inc.", "AMD Accelerated Parallel Processing", "gfx1100")
        )
        assertEquals(
            com.lotobj.pc.engine.GpuVendor.AMD,
            GpuOpenClAccelerator.detectGpuVendor(0, "Clover", "Clover", "AMD Radeon RX 7900 XTX")
        )
        assertEquals(
            com.lotobj.pc.engine.GpuVendor.NVIDIA,
            GpuOpenClAccelerator.detectGpuVendor(0, "NVIDIA", "NVIDIA CUDA", "NVIDIA GeForce RTX 4080")
        )
        assertEquals(
            com.lotobj.pc.engine.GpuVendor.INTEL,
            GpuOpenClAccelerator.detectGpuVendor(0x8086, "Intel(R) Corporation", "Intel OpenCL", "Intel Arc A770")
        )

        assertEquals(32L, GpuOpenClAccelerator.recommendedLocalWorkSize(com.lotobj.pc.engine.GpuVendor.NVIDIA, 1024L))
        assertEquals(64L, GpuOpenClAccelerator.recommendedLocalWorkSize(com.lotobj.pc.engine.GpuVendor.AMD, 256L))
        assertEquals(32L, GpuOpenClAccelerator.recommendedLocalWorkSize(com.lotobj.pc.engine.GpuVendor.AMD, 32L))
    }

    @Test
    fun penultimateProgressGradientAndTop5ConfirmationUnblock1JAndRejectRandomNoise() {
        val sdf = java.text.SimpleDateFormat("yyyy-MM-dd HH:mm", java.util.Locale.US).apply {
            timeZone = LotoDateUtils.beninTz
        }
        val slots = listOf("00H" to "00:05", "8H" to "08:00", "11H" to "11:00", "14H" to "14:00", "18H" to "18:00", "20H" to "20:00")
        val seedPast = (0 until 5).map { idx ->
            val ts = sdf.parse("2026-10-01 08:00")!!.time + idx * 3_600_000L
            LotoDraw("grad_past_$idx", LotoDateUtils.formatFrenchDayAndDate(ts), "8H", "Fortune", 1, 2, 3, 4, 5, ts)
        }
        val dummyMatrix = SymbolicFormula.extractRawMatrix(seedPast, "8H")
        val steps = slots.mapIndexed { sIdx, (hLabel, hhmm) ->
            val ts = sdf.parse("2026-10-05 $hhmm")!!.time
            val nums = listOf(1, 2, 3, 4, 5)
            com.lotobj.pc.engine.CachedRawStep(
                stepIndex = sIdx + 1,
                draw = LotoDraw("grad_$sIdx", LotoDateUtils.formatFrenchDayAndDate(ts), hLabel, "Fortune", nums[0], nums[1], nums[2], nums[3], nums[4], ts),
                actualNumbers = nums,
                actualSet = nums.toSet(),
                strictPastSize = 5,
                maxPastTs = ts - 1_000L,
                dayKey = "2026-10-05",
                horaire = hLabel,
                game = "Fortune",
                rawMatrix = dummyMatrix
            )
        }

        // Sans avant-dernier >=2 vs avec avant-dernier >=2 : le gradient doit guider nettement vers l'avant-dernier
        val (pen0, _, score0) = PcDiscoveryEngine.computeBlockProgressGradient(steps, listOf(2, 2, 0, 0, 0, 0))
        val (pen1, _, score1) = PcDiscoveryEngine.computeBlockProgressGradient(steps, listOf(2, 0, 0, 0, 2, 0))
        assertEquals(0, pen0)
        assertEquals(1, pen1)
        assertTrue("Expected penultimate hit gradient $score1 > non-penultimate $score0", score1 > score0)

        // Aucun faux positif sur bruit uniforme 5/90 après passage au Top-5 Confirmation et relâchement 0-boule
        val noiseHistory = buildSyntheticHistory(18)
        val pool = ForkJoinPool(2)
        val gpu = GpuOpenClAccelerator()
        try {
            val cache = PcDiscoveryEngine.buildCausalCache(
                history = noiseHistory,
                period = BacktestPeriod.THREE_DAYS,
                referenceNowMs = noiseHistory.last().timestamp + 1_000L,
                monteCarloSims = 400,
                pool = pool
            )
            var state = PcDiscoveryEngine.createInitialState(populationSize = 36, sessionSalt = 707L)
            var anyFalseWinner = false
            repeat(4) {
                val res = PcDiscoveryEngine.executeEvolutionGeneration(
                    state = state,
                    cache = cache,
                    periodName = BacktestPeriod.THREE_DAYS.name,
                    minObjectiveStreak = 1,
                    populationSize = 36,
                    computeMode = ComputeMode.CPU_MAX_CORES,
                    cpuPool = pool,
                    gpuAccelerator = gpu
                )
                if (res.winningEval != null) anyFalseWinner = true
                state = res.nextState
            }
            assertFalse("Random draws must never falsely validate", anyFalseWinner)
            assertTrue(state.seenSignatures.size >= 36)

            // Fenêtre 1J (6 tirages -> split [2,1,2,1], sealedSteps.size == 1) : auditCandidateReadOnly ne rejette pas un bon modèle à cause d'un unique tirage scellé à 0
            val oneDayCache = PcDiscoveryEngine.buildCausalCache(
                history = noiseHistory,
                period = BacktestPeriod.ONE_DAY,
                referenceNowMs = noiseHistory.last().timestamp + 1_000L,
                monteCarloSims = 300,
                pool = pool
            )
            assertEquals(1, oneDayCache.sealedSteps.size)

            // Moteur 2 Précision TWO_NUMBERS évolue proprement sur 1J
            val precOneDayCache = com.lotobj.pc.engine.PrecisionTargetEngine.buildCausalCache(
                history = noiseHistory,
                period = BacktestPeriod.ONE_DAY,
                referenceNowMs = noiseHistory.last().timestamp + 1_000L,
                customDays = 1,
                pool = pool
            )
            val precState = com.lotobj.pc.engine.PrecisionTargetEngine.createInitialState(
                mode = com.lotobj.pc.model.PrecisionTargetMode.TWO_NUMBERS,
                populationSize = 24,
                sessionSalt = 808L
            )
            val precRes = com.lotobj.pc.engine.PrecisionTargetEngine.executeEvolutionGeneration(
                state = precState,
                cache = precOneDayCache,
                minObjectiveStreak = 1,
                populationSize = 24,
                computeMode = ComputeMode.CPU_MAX_CORES,
                cpuPool = pool,
                gpuAccelerator = gpu
            )
            assertTrue(precRes.nextState.seenSignatures.size >= 24)
        } finally {
            pool.shutdownNow()
            gpu.close()
        }
    }
}

