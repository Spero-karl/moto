package com.example.domain.engine

import android.app.Application
import android.app.Service
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.assertTextContains
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo
import androidx.compose.ui.test.performScrollToNode
import androidx.compose.ui.test.performTextReplacement
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import com.example.data.local.AppDatabase
import com.example.data.local.DiscoveredFormulaEntity
import com.example.data.local.DiscoverySessionEntity
import com.example.data.local.LotoResultEntity
import com.example.domain.util.BacktestExportHelper
import com.example.domain.util.LotoDateUtils
import com.example.domain.util.StrategyBackupExport
import com.example.domain.util.StrategyBackupHelper
import com.example.domain.util.StrategyBackupImportResult
import androidx.compose.runtime.mutableStateOf
import com.example.service.AutonomousDiscoveryCoordinator
import com.example.service.DiscoveryForegroundService
import com.example.ui.components.BacktestStrategySelectorCard
import com.example.ui.components.StrategyBackupActionsCard
import com.example.ui.screens.AutonomousEngineOutputScreen
import com.example.ui.screens.PredictionsScreen
import com.example.ui.theme.MyApplicationTheme
import com.example.ui.viewmodel.DiscoveryViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.Robolectric
import org.robolectric.RobolectricTestRunner
import org.robolectric.Shadows
import org.robolectric.annotation.Config
import java.text.SimpleDateFormat
import java.util.Locale
import kotlin.random.Random

@RunWith(RobolectricTestRunner::class)
@Config(sdk = [36])
class DiscoveryEngineTest {

    @get:Rule
    val composeRule = createComposeRule()

    private fun buildDatedDraws(
        startDateStr: String,
        count: Int,
        tailSalt: Int = 0,
        mutateFromIndex: Int = (count * 75) / 100
    ): List<LotoResultEntity> {
        val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).apply {
            timeZone = LotoDateUtils.beninTz
        }
        val startMs = sdf.parse("$startDateStr 08:00")!!.time
        val horaires = listOf("8H", "11H", "14H", "18H")
        val pool = IntArray(90) { it + 1 }
        return (0 until count).map { i ->
            val ts = startMs + i * 6L * 3_600_000L
            val rng = Random(if (i < mutateFromIndex) i * 31L + 7L else i * 31L + 7L + tailSalt * 999L)
            val chosen = LinkedHashSet<Int>(5)
            while (chosen.size < 5) {
                chosen.add(pool[rng.nextInt(90)])
            }
            val nums = chosen.toList()
            LotoResultEntity(
                uniqueId = "sym_d_$i",
                date = LotoDateUtils.formatFrenchDayAndDate(ts),
                time = horaires[i % horaires.size],
                game = if (i % 3 == 0) "Digital" else "Fortune",
                num1 = nums[0],
                num2 = nums[1],
                num3 = nums[2],
                num4 = nums[3],
                num5 = nums[4],
                timestamp = ts
            )
        }
    }

    @Test
    fun symbolicFormula_simplifiesCrossesMutatesAndRoundTripsJson() {
        val rng = Random(77L)
        val exprA = SymbolicFormula.randomExpr(4, rng)
        val exprB = SymbolicFormula.randomExpr(4, rng)
        val crossed = SymbolicFormula.crossover(exprA, exprB, rng)
        val mutated = SymbolicFormula.mutate(crossed, rng)

        assertTrue(SymbolicFormula.depth(crossed) <= 4)
        assertTrue(SymbolicFormula.nodeCount(crossed) <= 21)
        assertTrue(SymbolicFormula.usedFeatures(mutated).isNotEmpty())
        assertTrue(SymbolicFormula.prettyPrint(mutated).isNotBlank())

        val genome = SymbolicGenome(
            id = "g_test",
            generation = 2,
            parentIds = listOf("p1", "p2"),
            primaryExpr = exprA,
            secondaryExpr = mutated,
            primarySlots = 3,
            samplingTemp = 0.65,
            exploreNoise = 0.12,
            seed = 12345L
        )
        val json = SymbolicFormula.genomeToJson(genome)
        val restored = SymbolicFormula.genomeFromJson(json)
        assertEquals(genome.id, restored.id)
        assertEquals(genome.primarySlots, restored.primarySlots)
        assertEquals(genome.describeCompact(), restored.describeCompact())
    }

    @Test
    fun discoveryEngine_fourStageSplitAndZeroLeakageIsolation() {
        val drawsA = buildDatedDraws("2026-10-01", 100, tailSalt = 0, mutateFromIndex = 75)
        val drawsB = buildDatedDraws("2026-10-01", 100, tailSalt = 91, mutateFromIndex = 75)

        val cacheA = DiscoveryEngine.buildCausalCache(drawsA, BacktestPeriod.ONE_MONTH, referenceNowMs = drawsA.last().timestamp)
        val cacheB = DiscoveryEngine.buildCausalCache(drawsB, BacktestPeriod.ONE_MONTH, referenceNowMs = drawsB.last().timestamp)

        assertTrue(cacheA.discoverySteps.isNotEmpty())
        assertTrue(cacheA.rollingSteps.isNotEmpty())
        assertTrue(cacheA.confirmationSteps.isNotEmpty())
        assertTrue(cacheA.sealedSteps.isNotEmpty())
        assertTrue(cacheA.steps.all { it.maxPastTs < it.draw.timestamp })

        var stateA = DiscoveryEngine.createInitialState(sessionSalt = 15L)
        var stateB = DiscoveryEngine.createInitialState(sessionSalt = 15L)

        // Executer 3 generations : la population GP et le meilleur candidat d'entrainement
        // doivent rester 100% identiques malgre la modification des tirages de Confirmation & Scelle !
        repeat(3) {
            val resA = DiscoveryEngine.executeEvolutionGeneration(stateA, cacheA, BacktestPeriod.ONE_MONTH.name)
            val resB = DiscoveryEngine.executeEvolutionGeneration(stateB, cacheB, BacktestPeriod.ONE_MONTH.name)
            stateA = resA.nextState
            stateB = resB.nextState
            assertEquals(resA.bestTrainEval?.genome?.describeCompact(), resB.bestTrainEval?.genome?.describeCompact())
            assertEquals(
                stateA.population.map { it.describeCompact() },
                stateB.population.map { it.describeCompact() }
            )
        }
    }

    @Test
    fun discoveryEngine_doesNotFakeValidationOnUniformRandomDraws() {
        val randomDraws = buildDatedDraws("2026-10-01", 110, tailSalt = 5)
        val cache = DiscoveryEngine.buildCausalCache(randomDraws, BacktestPeriod.ONE_MONTH)
        var state = DiscoveryEngine.createInitialState(sessionSalt = 21L)

        var anyFalsePositive = false
        repeat(5) {
            val res = DiscoveryEngine.executeEvolutionGeneration(state, cache, BacktestPeriod.ONE_MONTH.name)
            if (res.winningEval != null) anyFalsePositive = true
            state = res.nextState
        }
        assertFalse("Un bruit uniforme 5/90 ne doit pas etre valide facilement par hasard", anyFalsePositive)
        assertTrue("Les formules rejetees doivent etre tracees honnetement", state.rejectedHistory.isNotEmpty())
    }

    @Test
    fun roomDao_persistsDiscoveredFormulasAndSessionsAcrossRestarts() = runBlocking {
        val context = ApplicationProvider.getApplicationContext<Application>()
        val db = Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java)
            .allowMainThreadQueries()
            .build()
        try {
            val dao = db.discoveryDao()
            val genome = DiscoveryEngine.generateInitialPopulation(2, 9L).first()
            val entityTop = DiscoveredFormulaEntity(
                id = "f_top",
                periodName = BacktestPeriod.ONE_MONTH.name,
                generation = 4,
                parentIdsCsv = "",
                primaryExprText = genome.describePrimary(),
                secondaryExprText = genome.describeSecondary(),
                slotSplitLabel = "3+2",
                genomeJson = SymbolicFormula.genomeToJson(genome),
                usedFeaturesCsv = "gap,f15",
                status = "VALIDATED",
                rejectionStage = "",
                rejectionReason = "Validé 4 étapes",
                discoveryStreak = 4,
                rollingStreak = 4,
                confirmationStreak = 4,
                sealedStreak = 2,
                combinedStreak = 4,
                discoveryAvgMatches = 0.65f,
                rollingAvgMatches = 0.60f,
                confirmationAvgMatches = 0.58f,
                sealedAvgMatches = 0.50f,
                pValueAdjusted = 0.008,
                fitnessScore = 120.0,
                updatedAt = 2000L
            )
            val entityRej = entityTop.copy(
                id = "f_rej",
                status = "REJECTED",
                rejectionStage = "Confirmation Indépendante",
                rejectionReason = "Ne bat pas Monte Carlo",
                fitnessScore = 40.0,
                updatedAt = 1500L
            )
            dao.upsertFormulas(listOf(entityTop, entityRej))

            val session = DiscoverySessionEntity(
                periodName = BacktestPeriod.ONE_MONTH.name,
                phaseName = AutonomousPhase.SUSPENDED.name,
                generation = 4,
                experimentsCount = 72,
                confirmationTestsCount = 3,
                sessionSalt = 9L,
                populationJson = SymbolicFormula.genomeListToJson(listOf(genome)),
                activeGenomeJson = SymbolicFormula.genomeToJson(genome),
                validatedGenomeJson = SymbolicFormula.genomeToJson(genome),
                bestDiscoveryStreak = 4,
                bestRollingStreak = 4,
                bestConfirmationStreak = 4,
                bestSealedStreak = 2,
                bestCombinedStreak = 4,
                honestStatusNote = "Sauvegardé",
                logsJson = "[\"log1\"]",
                updatedAt = 2100L
            )
            dao.upsertSession(session)

            val loadedSession = dao.getSession(BacktestPeriod.ONE_MONTH.name)
            assertNotNull(loadedSession)
            assertEquals(72, loadedSession!!.experimentsCount)
            assertEquals(1, dao.getTopFormulas(BacktestPeriod.ONE_MONTH.name, 5).size)
            assertEquals(1, dao.getRejectedFormulas(BacktestPeriod.ONE_MONTH.name, 5).size)
            assertEquals("f_top", dao.getValidatedFormula(BacktestPeriod.ONE_MONTH.name)?.id)

            dao.deleteSession(BacktestPeriod.ONE_MONTH.name)
            assertNull(dao.getSession(BacktestPeriod.ONE_MONTH.name))
        } finally {
            db.close()
        }
    }

    @Test
    fun discoveryViewModel_startsSuspendsResumesAndStopsWithoutResetting() {
        val app = ApplicationProvider.getApplicationContext<Application>()
        val vm = DiscoveryViewModel(app)
        val draws = buildDatedDraws("2026-10-01", 80)
        vm.injectHistoryForTest(draws)

        vm.startContinuousDiscovery(reset = true)

        val deadline = System.currentTimeMillis() + 5_000L
        while (System.currentTimeMillis() < deadline &&
            vm.discoveryState.value.generation < 1 &&
            vm.discoveryState.value.phase != AutonomousPhase.VALIDATED
        ) {
            Thread.sleep(25L)
        }

        val stateBeforeSuspend = vm.discoveryState.value
        assertTrue(stateBeforeSuspend.generation >= 1 || stateBeforeSuspend.phase == AutonomousPhase.VALIDATED)

        if (stateBeforeSuspend.phase == AutonomousPhase.ANALYZING) {
            val expAtSuspend = stateBeforeSuspend.experimentsCount
            vm.suspendDiscovery()
            assertEquals(AutonomousPhase.SUSPENDED, vm.discoveryState.value.phase)
            assertTrue(vm.discoveryState.value.canResume)

            vm.resumeDiscovery()
            val deadline2 = System.currentTimeMillis() + 4_000L
            while (System.currentTimeMillis() < deadline2 &&
                vm.discoveryState.value.experimentsCount <= expAtSuspend &&
                vm.discoveryState.value.phase == AutonomousPhase.ANALYZING
            ) {
                Thread.sleep(25L)
            }
            assertTrue(vm.discoveryState.value.experimentsCount >= expAtSuspend)

            vm.stopDiscovery()
            assertEquals(AutonomousPhase.STOPPED, vm.discoveryState.value.phase)
        }
    }

    @Test
    fun predictionsScreen_enforcesLifecycleAndLocksBacktestUntilValidated() {
        var started = false
        var launchedBacktest = false
        composeRule.setContent {
            MyApplicationTheme {
                PredictionsScreen(
                    engineState = EngineActionState(
                        phase = AutonomousPhase.IDLE,
                        rejectedFormulas = listOf(
                            RejectedFormulaSummary(
                                id = "rej1",
                                generation = 2,
                                formulaPreview = "[3+2] A=(gap) | B=(f15)",
                                rejectedAtStage = "Confirmation Indépendante",
                                reason = "Ne bat pas le hasard Monte Carlo",
                                trainAvgMatches = 0.45f,
                                validationAvgMatches = 0.25f
                            )
                        )
                    ),
                    prediction = null,
                    backtestResult = null,
                    selectedMethod = PredictionMethod.AUTO_SEARCH,
                    selectedPeriod = BacktestPeriod.ONE_MONTH,
                    selectedHoraire = "18H",
                    availableHoraires = listOf("8H", "14H", "18H"),
                    resultsCount = 80,
                    isLoading = false,
                    onSelectPeriod = {},
                    onSelectHoraire = {},
                    onStartAnalysis = { started = true },
                    onSuspendAnalysis = {},
                    onResumeAnalysis = {},
                    onStopAnalysis = {},
                    onLaunchBacktestAndPrediction = { launchedBacktest = true }
                )
            }
        }

        composeRule.onNodeWithTag("start_analysis_button").assertIsEnabled().performClick()
        assertTrue(started)

        composeRule.onNodeWithTag("launch_backtest_button").assertIsNotEnabled()
        assertFalse(launchedBacktest)

        composeRule.onNodeWithTag("predictions_list")
            .performScrollToNode(androidx.compose.ui.test.hasTestTag("rejected_formulas_card"))
        composeRule.onNodeWithText("Formules Abandonnées / Rejetées (1)", substring = true).assertIsDisplayed()
        composeRule.onNodeWithText("Ne bat pas le hasard Monte Carlo", substring = true).assertIsDisplayed()
    }

    @Test
    fun predictNextWithGenome_populatesCalibratedNumberConfidences() {
        val draws = buildDatedDraws("2026-10-01", 80)
        val cache = DiscoveryEngine.buildCausalCache(draws, BacktestPeriod.ONE_MONTH)
        val genome = DiscoveryEngine.generateInitialPopulation(2, 42L).first()
        val bt = DiscoveryEngine.buildValidatedBacktest(genome, cache, generationsExecuted = 3, experimentsCount = 54)
        val pred = DiscoveryEngine.predictNextWithGenome(genome, draws, "18H", bt)

        assertEquals(5, pred.numberConfidences.size)
        assertEquals(pred.predictedDraw.toSet(), pred.numberConfidences.map { it.number }.toSet())
        assertTrue(pred.numberConfidences.all { it.marginalProbabilityPercent > 0f && it.marginalProbabilityPercent < 100f })
        assertTrue(pred.numberConfidences.all { it.relativeConfidencePercent in 0f..100f })
        assertTrue(pred.numberConfidences.all { it.rank in 1..90 })
        assertTrue(pred.numberConfidences.all { it.sourceBranch == "Formule A" || it.sourceBranch == "Formule B" })
        assertEquals(8, pred.candidateConfidences.size)
    }

    @Test
    fun autonomousEngineOutputScreen_displaysPredictedNumbersAndConfidencePercentages() {
        val draws = buildDatedDraws("2026-10-01", 80)
        val cache = DiscoveryEngine.buildCausalCache(draws, BacktestPeriod.ONE_MONTH)
        val genome = DiscoveryEngine.generateInitialPopulation(2, 7L).first()
        val bt = DiscoveryEngine.buildValidatedBacktest(genome, cache, generationsExecuted = 6, experimentsCount = 108)
        val pred = DiscoveryEngine.predictNextWithGenome(genome, draws, "18H", bt)

        val engineState = EngineActionState(
            phase = AutonomousPhase.VALIDATED,
            generation = 6,
            experimentsCount = 108,
            fourStageMetrics = FourStageValidationMetrics(
                discoveryDraws = cache.discoverySteps.size,
                discoveryStreak = 4,
                discoveryQualifyingDays = 5,
                discoveryTwoPlusCount = 12,
                discoveryAvgMatches = 0.62f,
                rollingDraws = cache.rollingSteps.size,
                rollingStreak = 4,
                rollingQualifyingDays = 4,
                rollingTwoPlusCount = 9,
                rollingAvgMatches = 0.58f,
                rollingStabilityPassed = true,
                confirmationDraws = cache.confirmationSteps.size,
                confirmationStreak = 4,
                confirmationQualifyingDays = 4,
                confirmationTwoPlusCount = 8,
                confirmationAvgMatches = 0.56f,
                sealedDraws = cache.sealedSteps.size,
                sealedStreak = 2,
                sealedQualifyingDays = 2,
                sealedTwoPlusCount = 4,
                sealedAvgMatches = 0.51f,
                sealedCausalVerified = true,
                combinedStreak = 4,
                combinedQualifyingDays = 8,
                combinedMaxZeroStreak = 2,
                combinedBrierScore = 0.049,
                isFourStageValidated = true
            ),
            backtestTriggered = true
        )

        composeRule.setContent {
            MyApplicationTheme {
                AutonomousEngineOutputScreen(
                    engineState = engineState,
                    prediction = pred,
                    selectedHoraire = "18H",
                    availableHoraires = listOf("8H", "14H", "18H"),
                    isBusy = false,
                    onSelectHoraire = {},
                    onStartOrResumeAnalysis = {},
                    onLaunchBacktestAndPrediction = {},
                    onOpenDiscoveryConsole = {}
                )
            }
        }

        composeRule.onNodeWithTag("engine_output_screen").assertIsDisplayed()
        composeRule.onNodeWithText("Sortie du Moteur Autonome").assertIsDisplayed()

        // Verifier que les 5 numeros predits et leurs pourcentages de confiance s'affichent
        pred.numberConfidences.forEach { conf ->
            composeRule.onNodeWithTag("engine_output_list")
                .performScrollToNode(androidx.compose.ui.test.hasTestTag("predicted_confidence_tile_${conf.number}"))
            composeRule.onNodeWithTag("confidence_pct_${conf.number}", useUnmergedTree = true).assertIsDisplayed()
        }

        // Cliquer sur le deuxieme numero predit et verifier la mise a jour du panneau detail
        val secondBall = pred.numberConfidences[1]
        composeRule.onNodeWithTag("engine_output_list")
            .performScrollToNode(androidx.compose.ui.test.hasTestTag("predicted_confidence_tile_${secondBall.number}"))
        composeRule.onNodeWithTag("predicted_confidence_tile_${secondBall.number}").performClick()
        composeRule.onNodeWithTag("engine_output_list")
            .performScrollToNode(androidx.compose.ui.test.hasTestTag("selected_number_detail_card"))
        composeRule.onNodeWithText("Détail Numéro", substring = true).assertIsDisplayed()

        composeRule.onNodeWithTag("engine_output_list")
            .performScrollToNode(androidx.compose.ui.test.hasTestTag("confidence_distribution_card"))
        composeRule.onNodeWithTag("confidence_distribution_card").assertIsDisplayed()
    }

    @Test
    fun autonomousEngineOutputScreen_displaysLockedStateWhenUnvalidated() {
        composeRule.setContent {
            MyApplicationTheme {
                AutonomousEngineOutputScreen(
                    engineState = EngineActionState(phase = AutonomousPhase.IDLE),
                    prediction = null,
                    selectedHoraire = "18H",
                    availableHoraires = listOf("8H", "18H"),
                    isBusy = false,
                    onSelectHoraire = {},
                    onStartOrResumeAnalysis = {},
                    onLaunchBacktestAndPrediction = {},
                    onOpenDiscoveryConsole = {}
                )
            }
        }

        composeRule.onNodeWithTag("engine_output_empty_state").assertIsDisplayed()
        composeRule.onNodeWithText("Sortie Verrouillée Avant Validation").assertIsDisplayed()
        composeRule.onNodeWithTag("output_start_or_console_button").assertIsEnabled()
    }

    @Test
    fun discoveryCoordinator_survivesViewModelRecreationWithoutStoppingAndStartsForegroundService() {
        val app = ApplicationProvider.getApplicationContext<Application>()
        val vm1 = DiscoveryViewModel(app)
        val draws = buildDatedDraws("2026-10-01", 80)
        vm1.injectHistoryForTest(draws)

        vm1.startContinuousDiscovery(reset = true)

        val deadline1 = System.currentTimeMillis() + 5_000L
        while (System.currentTimeMillis() < deadline1 &&
            vm1.discoveryState.value.generation < 1 &&
            vm1.discoveryState.value.phase != AutonomousPhase.VALIDATED
        ) {
            Thread.sleep(25L)
        }

        val startedService = Shadows.shadowOf(app).nextStartedService
        assertNotNull("Le service foreground doit etre demarre quand la recherche se lance", startedService)
        assertEquals(DiscoveryForegroundService::class.java.name, startedService?.component?.className)

        val genBeforeLeave = vm1.discoveryState.value.generation
        val expBeforeLeave = vm1.discoveryState.value.experimentsCount

        // Simuler la fermeture / recreation de l'UI (Activity detruite -> ViewModel cleared)
        val onClearedMethod = androidx.lifecycle.ViewModel::class.java.getDeclaredMethod("onCleared")
        onClearedMethod.isAccessible = true
        onClearedMethod.invoke(vm1)

        // Un nouveau ViewModel cree au retour de l'utilisateur doit retrouver la session toujours active
        val vm2 = DiscoveryViewModel(app)
        if (vm2.discoveryState.value.phase == AutonomousPhase.ANALYZING) {
            val deadline2 = System.currentTimeMillis() + 4_000L
            while (System.currentTimeMillis() < deadline2 &&
                vm2.discoveryState.value.experimentsCount <= expBeforeLeave &&
                vm2.discoveryState.value.phase == AutonomousPhase.ANALYZING
            ) {
                Thread.sleep(25L)
            }
            assertTrue(
                "Le moteur doit continuer a progresser apres sortie/retour de l'UI",
                vm2.discoveryState.value.generation >= genBeforeLeave &&
                    vm2.discoveryState.value.experimentsCount >= expBeforeLeave
            )
            vm2.stopDiscovery()
        }
    }

    @Test
    fun discoveryForegroundService_runsStickyAndStopsOnlyOnExplicitCommand() {
        val controller = Robolectric.buildService(DiscoveryForegroundService::class.java).create()
        val service = controller.get()

        val startIntent = Intent(ApplicationProvider.getApplicationContext(), DiscoveryForegroundService::class.java).apply {
            action = DiscoveryForegroundService.ACTION_START_OR_UPDATE
            putExtra(DiscoveryForegroundService.EXTRA_GENERATION, 12)
            putExtra(DiscoveryForegroundService.EXTRA_EXPERIMENTS, 216)
            putExtra(DiscoveryForegroundService.EXTRA_BEST_STREAK, 3)
        }
        val stickyResult = service.onStartCommand(startIntent, 0, 1)
        assertEquals(Service.START_STICKY, stickyResult)

        // Verifier que null intent (redemarrage OS) reste START_STICKY
        val restartResult = service.onStartCommand(null, 0, 2)
        assertEquals(Service.START_STICKY, restartResult)

        val stopIntent = Intent(ApplicationProvider.getApplicationContext(), DiscoveryForegroundService::class.java).apply {
            action = DiscoveryForegroundService.ACTION_STOP_SERVICE
        }
        val stopResult = service.onStartCommand(stopIntent, 0, 3)
        assertEquals(Service.START_NOT_STICKY, stopResult)
        controller.destroy()
    }

    @Test
    fun discoveryCoordinator_savesBestMilestoneAndRecoversFromErrorContinuingWithBestGenome() {
        val app = ApplicationProvider.getApplicationContext<Application>()
        val coordinator = AutonomousDiscoveryCoordinator.getInstance(app)
        val draws = buildDatedDraws("2026-10-01", 80)
        coordinator.injectHistoryForTest(draws)

        coordinator.startContinuousDiscovery(reset = true)

        val deadline1 = System.currentTimeMillis() + 5_000L
        while (System.currentTimeMillis() < deadline1 &&
            coordinator.discoveryState.value.generation < 1 &&
            coordinator.discoveryState.value.phase != AutonomousPhase.VALIDATED
        ) {
            Thread.sleep(25L)
        }

        val stateBeforeErr = coordinator.discoveryState.value
        assertTrue(stateBeforeErr.generation >= 1 || stateBeforeErr.phase == AutonomousPhase.VALIDATED)

        if (stateBeforeErr.phase == AutonomousPhase.ANALYZING) {
            val savedBestExpr = stateBeforeErr.currentPrimaryExpr
            val savedGen = stateBeforeErr.generation
            val savedStreak = stateBeforeErr.bestCombinedStreak
            assertTrue("Un meilleur genome doit etre enregistre des l'avancement", savedBestExpr.isNotBlank())

            // Verifier que le checkpoint existe deja en base Room
            runBlocking {
                val db = AppDatabase.getDatabase(app)
                val session = db.discoveryDao().getSession(BacktestPeriod.THREE_WEEKS.name)
                val checkpoint = db.discoveryDao().getCheckpointFormula("checkpoint_best_${BacktestPeriod.THREE_WEEKS.name}")
                assertNotNull(session)
                assertTrue(session!!.activeGenomeJson.isNotBlank())
                assertNotNull("Le meilleur palier doit etre sauvegarde immediatement", checkpoint)
            }

            // Injecter une erreur en plein calcul et verifier la reprise avec le meilleur genome
            coordinator.testInjectedErrorOnce = { throw IllegalStateException("Erreur simulee") }

            val deadline2 = System.currentTimeMillis() + 5_000L
            while (System.currentTimeMillis() < deadline2 &&
                coordinator.discoveryState.value.generation <= savedGen &&
                coordinator.discoveryState.value.phase == AutonomousPhase.ANALYZING
            ) {
                Thread.sleep(25L)
            }

            val stateAfterRecovery = coordinator.discoveryState.value
            assertTrue(
                "Le moteur doit continuer apres une erreur sans repasser a IDLE/STOPPED",
                stateAfterRecovery.phase == AutonomousPhase.ANALYZING ||
                    stateAfterRecovery.phase == AutonomousPhase.VALIDATED
            )
            assertTrue(
                "Le compteur de generations doit continuer a avancer apres reprise",
                stateAfterRecovery.generation > savedGen
            )
            assertTrue(
                "La meilleure serie atteinte doit etre preservee",
                stateAfterRecovery.bestCombinedStreak >= savedStreak
            )
            assertTrue(
                "Le journal doit tracer la reprise avec le meilleur genome",
                stateAfterRecovery.liveLogs.any { it.contains("Erreur récupérée") || it.contains("Palier") }
            )
            coordinator.stopDiscovery()
        }
    }

    @Test
    fun discoveryDao_protectsCheckpointBestFormulaDuringPruning() = runBlocking {
        val context = ApplicationProvider.getApplicationContext<Application>()
        val db = Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java)
            .allowMainThreadQueries()
            .build()
        try {
            val dao = db.discoveryDao()
            val genome = DiscoveryEngine.generateInitialPopulation(2, 13L).first()
            val checkpoint = DiscoveredFormulaEntity(
                id = "checkpoint_best_${BacktestPeriod.ONE_MONTH.name}",
                periodName = BacktestPeriod.ONE_MONTH.name,
                generation = 3,
                parentIdsCsv = "",
                primaryExprText = genome.describePrimary(),
                secondaryExprText = genome.describeSecondary(),
                slotSplitLabel = "3+2",
                genomeJson = SymbolicFormula.genomeToJson(genome),
                usedFeaturesCsv = "gap",
                status = "ROLLING_PASSED",
                rejectionStage = "",
                rejectionReason = "",
                discoveryStreak = 3,
                rollingStreak = 3,
                confirmationStreak = 0,
                sealedStreak = 0,
                combinedStreak = 3,
                discoveryAvgMatches = 0.6f,
                rollingAvgMatches = 0.55f,
                confirmationAvgMatches = 0f,
                sealedAvgMatches = 0f,
                pValueAdjusted = 1.0,
                fitnessScore = 10.0,
                updatedAt = 100L
            )
            val noise = (1..40).map { idx ->
                checkpoint.copy(
                    id = "noise_$idx",
                    combinedStreak = 1,
                    fitnessScore = 50.0 + idx,
                    updatedAt = 200L + idx
                )
            }
            dao.upsertFormulas(listOf(checkpoint) + noise)
            dao.pruneOldFormulas(BacktestPeriod.ONE_MONTH.name)

            val preserved = dao.getCheckpointFormula(checkpoint.id)
            assertNotNull("Le checkpoint du meilleur genome ne doit jamais etre purge", preserved)
            assertEquals(3, preserved!!.combinedStreak)
        } finally {
            db.close()
        }
    }

    @Test
    fun roomDao_preservesSavedWinnersAcrossClearAndListsByHighestStreak() = runBlocking {
        val context = ApplicationProvider.getApplicationContext<Application>()
        val db = Room.inMemoryDatabaseBuilder(context, AppDatabase::class.java)
            .allowMainThreadQueries()
            .build()
        try {
            val dao = db.discoveryDao()
            val genome = DiscoveryEngine.generateInitialPopulation(2, 33L).first()
            val winner3j = DiscoveredFormulaEntity(
                id = "val_3j_test",
                periodName = BacktestPeriod.ONE_MONTH.name,
                generation = 5,
                parentIdsCsv = "",
                primaryExprText = genome.describePrimary(),
                secondaryExprText = genome.describeSecondary(),
                slotSplitLabel = "3+2",
                genomeJson = SymbolicFormula.genomeToJson(genome),
                usedFeaturesCsv = "gap",
                status = "MILESTONE_3J",
                rejectionStage = "",
                rejectionReason = "Palier 3J causal",
                discoveryStreak = 3,
                rollingStreak = 3,
                confirmationStreak = 1,
                sealedStreak = 1,
                combinedStreak = 3,
                discoveryAvgMatches = 0.6f,
                rollingAvgMatches = 0.58f,
                confirmationAvgMatches = 0.42f,
                sealedAvgMatches = 0.39f,
                pValueAdjusted = 0.08,
                fitnessScore = 95.0,
                updatedAt = 1000L
            )
            val winner4j = winner3j.copy(
                id = "val_4j_test",
                status = "VALIDATED",
                rejectionReason = "Validé 4 étapes (4j)",
                combinedStreak = 4,
                confirmationStreak = 4,
                fitnessScore = 135.0,
                updatedAt = 2000L
            )
            val nonWinner = winner3j.copy(
                id = "non_winner",
                status = "ROLLING_PASSED",
                combinedStreak = 2
            )

            dao.upsertFormulas(listOf(winner3j, winner4j, nonWinner))
            dao.clearFormulasForPeriod(BacktestPeriod.ONE_MONTH.name)

            val savedWinners = dao.getValidatedFormulas(BacktestPeriod.ONE_MONTH.name, 10)
            assertEquals(2, savedWinners.size)
            assertEquals("val_4j_test", savedWinners[0].id)
            assertEquals("val_3j_test", savedWinners[1].id)
            assertNull("Les formules non validées doivent être nettoyées", dao.getFormulaById("non_winner"))
            assertNotNull("Le palier 3J doit rester réutilisable plus tard", dao.getFormulaById("val_3j_test"))
        } finally {
            db.close()
        }
    }

    @Test
    fun predictionsScreen_displaysThreeOutOfFourAndUnlocksStrongestStrategyWithSwitching() {
        var backtestClicked = false
        var selectedWinnerId: String? = null
        val savedWinner4j = SymbolicFormulaSummary(
            id = "strat_4j_b",
            generation = 14,
            primaryExprText = "f15 + recency",
            secondaryExprText = "gap",
            slotSplitLabel = "2+3",
            usedVariables = listOf("f15", "recency", "gap"),
            status = "VALIDATED",
            statusReason = "Validé 4 étapes (4J)",
            discoveryStreak = 4,
            rollingStreak = 4,
            confirmationStreak = 4,
            sealedStreak = 2,
            combinedStreak = 4,
            discoveryAvgMatches = 0.66f,
            rollingAvgMatches = 0.62f,
            confirmationAvgMatches = 0.59f,
            pValueAdjusted = 0.004,
            fitnessScore = 140.0,
            isValidated = true
        )
        val savedWinner3j = SymbolicFormulaSummary(
            id = "strat_3j_a",
            generation = 7,
            primaryExprText = "gap + f15",
            secondaryExprText = "recency",
            slotSplitLabel = "3+2",
            usedVariables = listOf("gap", "f15"),
            status = "MILESTONE_3J",
            statusReason = "Palier 3J causal",
            discoveryStreak = 3,
            rollingStreak = 2,
            confirmationStreak = 1,
            sealedStreak = 1,
            combinedStreak = 3,
            discoveryAvgMatches = 0.61f,
            rollingAvgMatches = 0.58f,
            confirmationAvgMatches = 0.44f,
            pValueAdjusted = 0.09,
            fitnessScore = 99.0,
            isValidated = false
        )

        composeRule.setContent {
            MyApplicationTheme {
                PredictionsScreen(
                    engineState = EngineActionState(
                        phase = AutonomousPhase.ANALYZING,
                        generation = 14,
                        experimentsCount = 252,
                        bestCombinedStreak = 3,
                        targetObjectiveStreak = 4,
                        bestValidatedStreak = 4,
                        activeWinnerId = "strat_4j_b",
                        savedWinners = listOf(savedWinner4j, savedWinner3j)
                    ),
                    prediction = null,
                    backtestResult = null,
                    selectedMethod = PredictionMethod.AUTO_SEARCH,
                    selectedPeriod = BacktestPeriod.ONE_MONTH,
                    selectedHoraire = "18H",
                    availableHoraires = listOf("8H", "14H", "18H"),
                    resultsCount = 80,
                    isLoading = false,
                    onSelectPeriod = {},
                    onSelectHoraire = {},
                    onStartAnalysis = {},
                    onSuspendAnalysis = {},
                    onResumeAnalysis = {},
                    onStopAnalysis = {},
                    onLaunchBacktestAndPrediction = { backtestClicked = true },
                    onSelectSavedWinner = { selectedWinnerId = it }
                )
            }
        }

        // Le score 3 doit s'afficher sur l'objectif 4 (3 / 4 j)
        composeRule.onNodeWithText("3 / 4 j", substring = true).assertIsDisplayed()
        // Le bouton Backtest & Prediction doit afficher la strategie la plus forte active (4J)
        composeRule.onNodeWithTag("launch_backtest_button")
            .performScrollTo()
            .assertIsEnabled()
            .performClick()
        assertTrue("Le backtest doit être débloqué avec la stratégie la plus forte", backtestClicked)
        composeRule.onNodeWithText("LANCER BACKTEST & PRÉDICTION (4J)", substring = true).assertIsDisplayed()

        // Verifier le choix d'une strategie enregistree (3J ou 4J)
        composeRule.onNodeWithTag("predictions_list")
            .performScrollToNode(androidx.compose.ui.test.hasTestTag("saved_winners_card"))
        composeRule.onNodeWithText("Gagnants Enregistrés (2)", substring = true).assertIsDisplayed()
        composeRule.onNodeWithTag("use_saved_winner_strat_3j_a")
            .performScrollTo()
            .assertIsEnabled()
            .performClick()
        assertEquals("strat_3j_a", selectedWinnerId)
    }

    @Test
    fun predictionsScreen_displaysContinueBeyondButtonWhenValidated() {
        var continuedBeyond = false
        composeRule.setContent {
            MyApplicationTheme {
                PredictionsScreen(
                    engineState = EngineActionState(
                        phase = AutonomousPhase.VALIDATED,
                        generation = 12,
                        experimentsCount = 216,
                        bestCombinedStreak = 4,
                        targetObjectiveStreak = 4,
                        bestValidatedStreak = 4,
                        activeWinnerId = "val_4j_x"
                    ),
                    prediction = null,
                    backtestResult = null,
                    selectedMethod = PredictionMethod.AUTO_SEARCH,
                    selectedPeriod = BacktestPeriod.ONE_MONTH,
                    selectedHoraire = "18H",
                    availableHoraires = listOf("8H", "18H"),
                    resultsCount = 80,
                    isLoading = false,
                    onSelectPeriod = {},
                    onSelectHoraire = {},
                    onStartAnalysis = {},
                    onSuspendAnalysis = {},
                    onResumeAnalysis = {},
                    onStopAnalysis = {},
                    onLaunchBacktestAndPrediction = {},
                    onContinueBeyondObjective = { continuedBeyond = true }
                )
            }
        }

        composeRule.onNodeWithTag("continue_beyond_button")
            .performScrollTo()
            .assertIsEnabled()
            .performClick()
        assertTrue(continuedBeyond)
    }

    @Test
    fun discoveryEngine_auditCandidateReadOnly_populatesStage3And4WithoutMutatingState() {
        val draws = buildDatedDraws("2026-10-01", 110, tailSalt = 7)
        val cache = DiscoveryEngine.buildCausalCache(draws, BacktestPeriod.ONE_MONTH)
        val state = DiscoveryEngine.createInitialState(sessionSalt = 19L)
        val res = DiscoveryEngine.executeEvolutionGeneration(
            state = state,
            cache = cache,
            periodName = BacktestPeriod.ONE_MONTH.name,
            minObjectiveStreak = 4
        )
        val candidate = res.bestTrainEval!!
        val audited = DiscoveryEngine.auditCandidateReadOnly(
            candidate = candidate,
            cache = cache,
            confirmationTestsCount = res.nextState.confirmationTestsCount.coerceAtLeast(1)
        )
        assertNotNull(audited.confStats)
        assertNotNull(audited.sealedStats)
        assertTrue(audited.sealedStats!!.causalVerified)
        assertTrue(audited.status == "VALIDATED" || audited.status.startsWith("MILESTONE_"))
    }

    @Test
    fun discoveryEngine_uniformRandomNoiseStillFailsStrict4jObjective() {
        val randomDraws = buildDatedDraws("2026-10-01", 110, tailSalt = 19)
        val cache = DiscoveryEngine.buildCausalCache(randomDraws, BacktestPeriod.ONE_MONTH)
        var state = DiscoveryEngine.createInitialState(sessionSalt = 41L)

        var anyFalsePositive = false
        repeat(5) {
            val res = DiscoveryEngine.executeEvolutionGeneration(
                state = state,
                cache = cache,
                periodName = BacktestPeriod.ONE_MONTH.name,
                minObjectiveStreak = 4
            )
            if (res.winningEval != null) anyFalsePositive = true
            state = res.nextState
        }
        assertFalse("L'objectif 4j sans triche ne doit pas valider du bruit aléatoire 5/90", anyFalsePositive)
    }

    @Test
    fun discoveryEngine_optimizedTopKMatchesFiveDistinctNumbersAndExactMarginalSum() {
        val draws = buildDatedDraws("2026-10-01", 80)
        val cache = DiscoveryEngine.buildCausalCache(draws, BacktestPeriod.ONE_MONTH)
        val genome = DiscoveryEngine.generateInitialPopulation(3, 99L).first()
        val bt = DiscoveryEngine.buildValidatedBacktest(genome, cache, generationsExecuted = 2, experimentsCount = 36)
        val pred = DiscoveryEngine.predictNextWithGenome(genome, draws, "18H", bt)

        assertEquals(5, pred.predictedDraw.distinct().size)
        assertTrue(pred.predictedDraw.all { it in 1..90 })
        assertTrue(pred.marginalSumCheck in 4.9f..5.1f)
        assertTrue(bt.steps.all { it.predictedNumbers.distinct().size == 5 })
    }

    @Test
    fun backtestStrategySelectorCard_selects3Day4DayAndCustomNDayTiersToUnlockModeling() {
        val winner4j = SymbolicFormulaSummary(
            id = "strat_4j_best",
            generation = 12,
            primaryExprText = "f15 + recency",
            secondaryExprText = "gap",
            slotSplitLabel = "3+2",
            usedVariables = listOf("f15", "recency", "gap"),
            status = "VALIDATED",
            statusReason = "Validé 4 étapes",
            discoveryStreak = 4,
            rollingStreak = 4,
            confirmationStreak = 4,
            sealedStreak = 2,
            combinedStreak = 4,
            discoveryAvgMatches = 0.65f,
            rollingAvgMatches = 0.61f,
            confirmationAvgMatches = 0.58f,
            pValueAdjusted = 0.005,
            fitnessScore = 138.0,
            isValidated = true
        )
        val winner3j = winner4j.copy(
            id = "strat_3j_alt",
            generation = 6,
            primaryExprText = "gap * co1",
            status = "MILESTONE_3J",
            statusReason = "Palier 3J",
            discoveryStreak = 3,
            rollingStreak = 3,
            confirmationStreak = 1,
            combinedStreak = 3,
            fitnessScore = 95.0,
            isValidated = false
        )

        val preferredState = mutableStateOf<Int?>(null)
        var unlockClicked = false

        composeRule.setContent {
            val engineState = EngineActionState(
                phase = AutonomousPhase.ANALYZING,
                generation = 12,
                experimentsCount = 216,
                bestCombinedStreak = 4,
                targetObjectiveStreak = 5,
                bestValidatedStreak = 4,
                activeWinnerId = if (preferredState.value == 3) "strat_3j_alt" else "strat_4j_best",
                savedWinners = listOf(winner4j, winner3j),
                preferredStrategyDays = preferredState.value
            )
            MyApplicationTheme {
                BacktestStrategySelectorCard(
                    engineState = engineState,
                    isBusy = false,
                    onSelectPreferredDays = { preferredState.value = it },
                    onSelectWinner = {},
                    onUnlockPredictiveModeling = { unlockClicked = true }
                )
            }
        }

        // Par defaut (Auto), la strategie 4J est debloquee et prete
        composeRule.onNodeWithTag("strategy_selector_card").assertIsDisplayed()
        composeRule.onNodeWithTag("strategy_unlock_cta").assertIsEnabled().performClick()
        assertTrue(unlockClicked)

        // Selectionner le palier 3J et verifier l'affichage de la formule 3J
        composeRule.onNodeWithTag("strategy_tier_list")
            .performScrollToNode(androidx.compose.ui.test.hasTestTag("strategy_tier_3"))
        composeRule.onNodeWithTag("strategy_tier_3").performClick()
        composeRule.waitForIdle()
        assertEquals(3, preferredState.value)
        composeRule.onNodeWithText("MODÉLISER AVEC LA STRATÉGIE 3J", substring = true).assertIsDisplayed()

        // Selectionner le palier 5J (non encore decouvert) -> verrouille avec progression + repli Auto
        composeRule.onNodeWithTag("strategy_tier_list")
            .performScrollToNode(androidx.compose.ui.test.hasTestTag("strategy_tier_5"))
        composeRule.onNodeWithTag("strategy_tier_5").performClick()
        composeRule.waitForIdle()
        assertEquals(5, preferredState.value)
        composeRule.onNodeWithTag("strategy_unlock_cta").assertDoesNotExist()
        composeRule.onNodeWithTag("strategy_fallback_auto_button").assertIsEnabled()

        // Incrementer N jours personnalise (5 -> 6J) et appliquer
        composeRule.onNodeWithTag("strategy_n_stepper_plus").performClick()
        composeRule.onNodeWithTag("strategy_n_apply").performClick()
        composeRule.waitForIdle()
        assertEquals(6, preferredState.value)
        composeRule.onNodeWithText("CIBLE 6J", substring = true).assertIsDisplayed()

        // Revenir en mode Auto via le bouton de repli -> re-debloque immediatement la strategie 4J
        composeRule.onNodeWithTag("strategy_fallback_auto_button").performClick()
        composeRule.waitForIdle()
        assertNull(preferredState.value)
        composeRule.onNodeWithTag("strategy_unlock_cta").assertIsEnabled()
    }

    @Test
    fun nextUpcomingDraw_selectsOnlyUpcomingBeninHoraireAndSkipsDrawnToday() {
        val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).apply {
            timeZone = LotoDateUtils.beninTz
        }
        val baseDraws = buildDatedDraws("2026-10-01", 40)

        // 1) A 09:30 heure Benin (aucun tirage aujourd'hui) -> doit cibler 11H le meme jour
        val nowMorningMs = sdf.parse("2026-10-12 09:30")!!.time
        val morningTarget = LotoDateUtils.resolveNextUpcomingDraw(baseDraws, nowMorningMs)
        assertEquals("11H", morningTarget.horaire)
        assertFalse(morningTarget.isTomorrow)
        assertEquals("09:30", morningTarget.currentBeninTimeLabel)

        // 2) A 19:00 heure Benin apres le dernier horaire (18H) -> doit cibler 8H le lendemain
        val nowEveningMs = sdf.parse("2026-10-12 19:00")!!.time
        val eveningTarget = LotoDateUtils.resolveNextUpcomingDraw(baseDraws, nowEveningMs)
        assertEquals("8H", eveningTarget.horaire)
        assertTrue(eveningTarget.isTomorrow)

        // 3) A 10:45 heure Benin si 11H est deja publie aujourd'hui -> doit sauter 11H et cibler 14H
        val today11hMs = sdf.parse("2026-10-12 11:00")!!.time
        val withToday11hDrawn = baseDraws + LotoResultEntity(
            uniqueId = "today_11h",
            date = LotoDateUtils.formatFrenchDayAndDate(today11hMs),
            time = "11H",
            game = "Fortune",
            num1 = 5,
            num2 = 12,
            num3 = 27,
            num4 = 44,
            num5 = 78,
            timestamp = today11hMs
        )
        val nowLateMorningMs = sdf.parse("2026-10-12 10:45")!!.time
        val skippedTarget = LotoDateUtils.resolveNextUpcomingDraw(withToday11hDrawn, nowLateMorningMs)
        assertEquals("14H", skippedTarget.horaire)
        assertFalse(skippedTarget.isTomorrow)
    }

    @Test
    fun threeWeekWindow_filtersStrict21DaysAnchoredAtDateXOfDayJ() {
        val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).apply {
            timeZone = LotoDateUtils.beninTz
        }
        // 120 tirages espacés de 6h (~30 jours), puis referenceNowMs situe 20 jours apres debut
        val draws = buildDatedDraws("2026-10-01", 120)
        val referenceNowMs = sdf.parse("2026-10-25 12:00")!!.time
        val cutoffMs = referenceNowMs - 21L * 86_400_000L

        val window = LotoDateUtils.resolveThreeWeekWindow(draws, referenceNowMs)
        assertTrue(window.draws.isNotEmpty())
        assertTrue("Aucun tirage avant J-21j", window.draws.all { it.timestamp >= cutoffMs })
        assertTrue("Aucun tirage futur apres referenceNowMs", window.draws.all { it.timestamp <= referenceNowMs })

        val backtest = LearningEngine().backtest(
            data = draws,
            method = PredictionMethod.AUTO_SEARCH,
            period = BacktestPeriod.THREE_WEEKS,
            referenceNowMs = referenceNowMs
        )
        assertTrue(backtest.steps.isNotEmpty())
        assertTrue("Le backtest 3 semaines ne doit evaluer aucun tirage hors [J-21j..J]", backtest.steps.all {
            it.targetTimestamp in cutoffMs..referenceNowMs
        })

        val report = ScientificEvaluator.runFullScientificComparison(
            data = draws,
            period = BacktestPeriod.THREE_WEEKS,
            referenceNowMs = referenceNowMs
        )
        val totalWindowSteps = report.splitInfo.devDrawsCount +
            report.splitInfo.valDrawsCount +
            report.splitInfo.indepTestDrawsCount
        assertTrue(totalWindowSteps <= window.draws.size)
        assertTrue(report.splitInfo.totalDrawsAvailable <= draws.count { it.timestamp <= referenceNowMs })
    }

    @Test
    fun uiDisplaysMatchedDaysCounterAndLocksOtherHoraires() {
        val draws = buildDatedDraws("2026-10-01", 80)
        val cache = DiscoveryEngine.buildCausalCache(draws, BacktestPeriod.THREE_WEEKS)
        val genome = DiscoveryEngine.generateInitialPopulation(1, 55L).first()
        val winner = SymbolicFormulaSummary(
            id = "strat_4j_ui",
            generation = 10,
            primaryExprText = genome.describePrimary(),
            secondaryExprText = genome.describeSecondary(),
            slotSplitLabel = "3+2",
            usedVariables = listOf("f15", "gap"),
            status = "VALIDATED",
            statusReason = "Validé 4 étapes",
            discoveryStreak = 4,
            rollingStreak = 4,
            confirmationStreak = 4,
            sealedStreak = 2,
            combinedStreak = 4,
            discoveryAvgMatches = 0.65f,
            rollingAvgMatches = 0.62f,
            confirmationAvgMatches = 0.58f,
            pValueAdjusted = 0.005,
            fitnessScore = 135.0,
            isValidated = true
        )
        val bt = DiscoveryEngine.buildValidatedBacktest(
            genome = genome,
            cache = cache,
            generationsExecuted = 10,
            experimentsCount = 180
        ).copy(
            qualifyingDaysCount = 3,
            maxConsecutiveQualifyingDays = 3
        )
        val pred = DiscoveryEngine.predictNextWithGenome(
            genome = genome,
            cleanHistory = draws,
            targetHoraire = "14H",
            validatedBacktest = bt
        )
        val target = NextDrawTarget(
            horaire = "14H",
            targetTimestamp = 1_760_000_000_000L,
            dayLabel = "Lundi 12/10/2026",
            currentBeninTimeLabel = "11:35",
            isTomorrow = false
        )
        val state = EngineActionState(
            phase = AutonomousPhase.VALIDATED,
            generation = 10,
            experimentsCount = 180,
            bestCombinedStreak = 3,
            targetObjectiveStreak = 4,
            bestValidatedStreak = 4,
            preferredStrategyDays = 4,
            backtestTriggered = true,
            activeWinnerId = "strat_4j_ui",
            savedWinners = listOf(winner)
        )

        composeRule.setContent {
            MyApplicationTheme {
                PredictionsScreen(
                    engineState = state,
                    prediction = pred,
                    backtestResult = bt,
                    selectedMethod = PredictionMethod.AUTO_SEARCH,
                    selectedPeriod = BacktestPeriod.THREE_WEEKS,
                    selectedHoraire = "14H",
                    availableHoraires = listOf("8H", "11H", "14H", "18H"),
                    nextDrawTarget = target,
                    resultsCount = 80,
                    isLoading = false,
                    onSelectPeriod = {},
                    onSelectHoraire = {},
                    onStartAnalysis = {},
                    onSuspendAnalysis = {},
                    onResumeAnalysis = {},
                    onStopAnalysis = {},
                    onLaunchBacktestAndPrediction = {}
                )
            }
        }

        // Verification du compteur X/N = 3/4 sur le selecteur de strategie, la prediction et le backtest
        composeRule.onNodeWithTag("predictions_list")
            .performScrollToNode(androidx.compose.ui.test.hasTestTag("strategy_matched_days_counter"))
        composeRule.onNodeWithTag("strategy_matched_days_counter")
            .assertIsDisplayed()
            .assertTextContains("Jours réussis : 3/4", substring = true)

        composeRule.onNodeWithTag("predictions_list")
            .performScrollToNode(androidx.compose.ui.test.hasTestTag("prediction_matched_days_counter"))
        composeRule.onNodeWithTag("prediction_matched_days_counter")
            .assertIsDisplayed()
            .assertTextContains("Jours réussis : 3/4", substring = true)

        composeRule.onNodeWithTag("predictions_list")
            .performScrollToNode(androidx.compose.ui.test.hasTestTag("backtest_matched_days_counter"))
        composeRule.onNodeWithTag("backtest_matched_days_counter")
            .assertIsDisplayed()
            .assertTextContains("3/4", substring = true)

        composeRule.onNodeWithTag("predictions_list")
            .performScrollToNode(androidx.compose.ui.test.hasTestTag("next_draw_horaire_badge"))
        composeRule.onNodeWithTag("next_draw_horaire_badge")
            .assertIsDisplayed()
            .assertTextContains("Prochain tirage ciblé uniquement : 14H", substring = true)

        // Les anciens selecteurs multi-horaires / multi-periodes ne doivent plus exister
        composeRule.onNodeWithTag("horaire_8H").assertDoesNotExist()
        composeRule.onNodeWithTag("period_ONE_MONTH").assertDoesNotExist()
    }

    @Test
    fun twoDayMilestone_unlocksBacktestAndPredictionAndAllows2JSelection() {
        var backtestClicked = false
        var selectedDays: Int? = null
        val winner2j = SymbolicFormulaSummary(
            id = "strat_2j_demo",
            generation = 4,
            primaryExprText = "f15 + gap",
            secondaryExprText = "recency",
            slotSplitLabel = "3+2",
            usedVariables = listOf("f15", "gap", "recency"),
            status = "MILESTONE_2J",
            statusReason = "Palier 2J causal",
            discoveryStreak = 2,
            rollingStreak = 2,
            confirmationStreak = 1,
            sealedStreak = 1,
            combinedStreak = 2,
            discoveryAvgMatches = 0.52f,
            rollingAvgMatches = 0.49f,
            confirmationAvgMatches = 0.38f,
            pValueAdjusted = 0.12,
            fitnessScore = 74.0,
            isValidated = false
        )

        val stateStoppedWith2J = EngineActionState(
            phase = AutonomousPhase.STOPPED,
            generation = 4,
            experimentsCount = 72,
            bestCombinedStreak = 2,
            targetObjectiveStreak = 4,
            bestValidatedStreak = 2,
            preferredStrategyDays = 2,
            activeWinnerId = "strat_2j_demo",
            savedWinners = listOf(winner2j)
        )
        assertTrue(stateStoppedWith2J.canLaunchBacktest)
        assertEquals(2, stateStoppedWith2J.selectedPatternLength)

        composeRule.setContent {
            MyApplicationTheme {
                PredictionsScreen(
                    engineState = stateStoppedWith2J,
                    prediction = null,
                    backtestResult = null,
                    selectedMethod = PredictionMethod.AUTO_SEARCH,
                    selectedPeriod = BacktestPeriod.THREE_WEEKS,
                    selectedHoraire = "14H",
                    availableHoraires = listOf("8H", "14H", "18H"),
                    resultsCount = 80,
                    isLoading = false,
                    onSelectPeriod = {},
                    onSelectHoraire = {},
                    onStartAnalysis = {},
                    onSuspendAnalysis = {},
                    onResumeAnalysis = {},
                    onStopAnalysis = {},
                    onLaunchBacktestAndPrediction = { backtestClicked = true },
                    onSelectPreferredStrategyDays = { selectedDays = it }
                )
            }
        }

        composeRule.onNodeWithTag("predictions_list")
            .performScrollToNode(androidx.compose.ui.test.hasTestTag("launch_backtest_button"))
        composeRule.onNodeWithTag("launch_backtest_button")
            .assertIsEnabled()
            .assertTextContains("LANCER BACKTEST & PRÉDICTION (2J)", substring = true)
            .performClick()
        assertTrue("Le palier 2J doit débloquer le backtest même après arrêt", backtestClicked)

        composeRule.onNodeWithTag("predictions_list")
            .performScrollToNode(androidx.compose.ui.test.hasTestTag("strategy_tier_2"))
        composeRule.onNodeWithTag("strategy_tier_2").assertIsDisplayed().performClick()
        assertEquals(2, selectedDays)
    }

    @Test
    fun backtestExportHelper_formatsDetailedReportAndCreatesTxtShareIntentAndClipboardCopy() {
        val context = ApplicationProvider.getApplicationContext<Application>()
        val draws = buildDatedDraws("2026-10-01", 80)
        val cache = DiscoveryEngine.buildCausalCache(draws, BacktestPeriod.THREE_WEEKS)
        val genome = DiscoveryEngine.generateInitialPopulation(1, 77L).first()
        val bt = DiscoveryEngine.buildValidatedBacktest(
            genome = genome,
            cache = cache,
            generationsExecuted = 8,
            experimentsCount = 144
        ).copy(
            qualifyingDaysCount = 3,
            maxConsecutiveQualifyingDays = 3
        )
        val pred = DiscoveryEngine.predictNextWithGenome(
            genome = genome,
            cleanHistory = draws,
            targetHoraire = "14H",
            validatedBacktest = bt
        )
        val target = NextDrawTarget(
            horaire = "14H",
            targetTimestamp = 1_760_000_000_000L,
            dayLabel = "Lundi 12/10/2026",
            currentBeninTimeLabel = "11:35",
            isTomorrow = false
        )
        val state = EngineActionState(
            phase = AutonomousPhase.VALIDATED,
            generation = 8,
            experimentsCount = 144,
            bestCombinedStreak = 3,
            targetObjectiveStreak = 4,
            bestValidatedStreak = 4,
            preferredStrategyDays = 4,
            currentPrimaryExpr = genome.describePrimary(),
            currentSecondaryExpr = genome.describeSecondary(),
            currentSlotSplit = "${genome.primarySlots}+${5 - genome.primarySlots}",
            backtestTriggered = true
        )

        val report = BacktestExportHelper.formatDetailedBacktestReport(
            backtestResult = bt,
            engineState = state,
            prediction = pred,
            nextDrawTarget = target
        )
        assertTrue(report.contains("RAPPORT DÉTAILLÉ DE BACKTEST"))
        assertTrue(report.contains("Jours réussis      : 3/4 j"))
        assertTrue(report.contains("DÉTAIL TIRAGE PAR TIRAGE (${bt.steps.size} étapes)"))
        assertTrue(report.contains("Prédit :"))
        assertTrue(report.contains("Tirage :"))

        val fileName = BacktestExportHelper.buildExportFileName(pred, bt)
        assertTrue(fileName.startsWith("backtest_loto_bj_14H_3j_"))
        assertTrue(fileName.endsWith(".txt"))

        assertTrue(BacktestExportHelper.copyToClipboard(context, report))
        val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        val copiedText = clipboard.primaryClip?.getItemAt(0)?.text?.toString()
        assertEquals(report, copiedText)

        clearFileProviderCacheForTest()
        val chooserIntent = BacktestExportHelper.createShareTextFileIntent(context, report, fileName)
        assertEquals(Intent.ACTION_CHOOSER, chooserIntent.action)
        @Suppress("DEPRECATION")
        val wrappedSend = chooserIntent.getParcelableExtra<Intent>(Intent.EXTRA_INTENT)
        assertNotNull(wrappedSend)
        assertEquals(Intent.ACTION_SEND, wrappedSend!!.action)
        assertEquals("text/plain", wrappedSend.type)
        @Suppress("DEPRECATION")
        val streamUri = wrappedSend.getParcelableExtra<Uri>(Intent.EXTRA_STREAM)
        assertNotNull("Le fichier .txt partage doit fournir un URI FileProvider", streamUri)
        val exportedFile = java.io.File(context.cacheDir, "exports/$fileName")
        assertTrue(exportedFile.exists())
        assertEquals(report, exportedFile.readText(Charsets.UTF_8))
    }

    @Test
    fun predictionsScreen_copyAndTransferButtonsWorkWhenBacktestIsDisplayed() {
        val context = ApplicationProvider.getApplicationContext<Application>()
        val draws = buildDatedDraws("2026-10-01", 80)
        val cache = DiscoveryEngine.buildCausalCache(draws, BacktestPeriod.THREE_WEEKS)
        val genome = DiscoveryEngine.generateInitialPopulation(1, 55L).first()
        val winner = SymbolicFormulaSummary(
            id = "strat_2j_export",
            generation = 5,
            primaryExprText = genome.describePrimary(),
            secondaryExprText = genome.describeSecondary(),
            slotSplitLabel = "3+2",
            usedVariables = listOf("f15", "gap"),
            status = "MILESTONE_2J",
            statusReason = "Palier 2J",
            discoveryStreak = 2,
            rollingStreak = 2,
            confirmationStreak = 1,
            sealedStreak = 1,
            combinedStreak = 2,
            discoveryAvgMatches = 0.52f,
            rollingAvgMatches = 0.49f,
            confirmationAvgMatches = 0.39f,
            pValueAdjusted = 0.12,
            fitnessScore = 72.0,
            isValidated = false
        )
        val bt = DiscoveryEngine.buildValidatedBacktest(
            genome = genome,
            cache = cache,
            generationsExecuted = 5,
            experimentsCount = 90
        ).copy(qualifyingDaysCount = 2, maxConsecutiveQualifyingDays = 2)
        val pred = DiscoveryEngine.predictNextWithGenome(
            genome = genome,
            cleanHistory = draws,
            targetHoraire = "14H",
            validatedBacktest = bt
        )
        val state = EngineActionState(
            phase = AutonomousPhase.STOPPED,
            generation = 5,
            experimentsCount = 90,
            bestCombinedStreak = 2,
            targetObjectiveStreak = 4,
            bestValidatedStreak = 2,
            preferredStrategyDays = 2,
            backtestTriggered = true,
            activeWinnerId = "strat_2j_export",
            savedWinners = listOf(winner)
        )

        composeRule.setContent {
            MyApplicationTheme {
                PredictionsScreen(
                    engineState = state,
                    prediction = pred,
                    backtestResult = bt,
                    selectedMethod = PredictionMethod.AUTO_SEARCH,
                    selectedPeriod = BacktestPeriod.THREE_WEEKS,
                    selectedHoraire = "14H",
                    availableHoraires = listOf("8H", "14H", "18H"),
                    resultsCount = 80,
                    isLoading = false,
                    onSelectPeriod = {},
                    onSelectHoraire = {},
                    onStartAnalysis = {},
                    onSuspendAnalysis = {},
                    onResumeAnalysis = {},
                    onStopAnalysis = {},
                    onLaunchBacktestAndPrediction = {}
                )
            }
        }

        composeRule.onNodeWithTag("predictions_list")
            .performScrollToNode(androidx.compose.ui.test.hasTestTag("copy_backtest_button"))
        composeRule.onNodeWithTag("copy_backtest_button").assertIsEnabled().performClick()
        composeRule.waitForIdle()
        composeRule.onNodeWithTag("backtest_export_feedback")
            .performScrollTo()
            .assertIsDisplayed()
            .assertTextContains("copié dans le presse-papiers", substring = true)

        val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        val clipContent = clipboard.primaryClip?.getItemAt(0)?.text?.toString().orEmpty()
        assertTrue(clipContent.contains("DÉTAIL TIRAGE PAR TIRAGE"))

        composeRule.onNodeWithTag("predictions_list")
            .performScrollToNode(androidx.compose.ui.test.hasTestTag("transfer_backtest_txt_button"))
        composeRule.onNodeWithTag("transfer_backtest_txt_button").assertIsEnabled()

        composeRule.onNodeWithTag("predictions_list")
            .performScrollToNode(androidx.compose.ui.test.hasTestTag("save_backtest_txt_button"))
        composeRule.onNodeWithTag("save_backtest_txt_button").assertIsEnabled()
    }

    @Test
    fun strategyBackupHelper_roundTripsBestMethodsAndExtractsJsonFromChatText() {
        val context = ApplicationProvider.getApplicationContext<Application>()
        val genomes = DiscoveryEngine.generateInitialPopulation(3, 99L)
        val winner4J = DiscoveredFormulaEntity(
            id = "strat_4j_demo",
            periodName = BacktestPeriod.THREE_WEEKS.name,
            generation = 42,
            parentIdsCsv = "",
            primaryExprText = genomes[0].describePrimary(),
            secondaryExprText = genomes[0].describeSecondary(),
            slotSplitLabel = "${genomes[0].primarySlots}+${5 - genomes[0].primarySlots}",
            genomeJson = SymbolicFormula.genomeToJson(genomes[0]),
            usedFeaturesCsv = "f15,gap",
            status = "VALIDATED",
            rejectionStage = "",
            rejectionReason = "Validé 4 niveaux (4J)",
            discoveryStreak = 4,
            rollingStreak = 4,
            confirmationStreak = 3,
            sealedStreak = 2,
            combinedStreak = 4,
            discoveryAvgMatches = 0.65f,
            rollingAvgMatches = 0.58f,
            confirmationAvgMatches = 0.51f,
            sealedAvgMatches = 0.44f,
            pValueAdjusted = 0.012,
            fitnessScore = 112.5,
            updatedAt = 1_760_000_000_000L
        )
        val winner2J = DiscoveredFormulaEntity(
            id = "strat_2j_demo",
            periodName = BacktestPeriod.THREE_WEEKS.name,
            generation = 18,
            parentIdsCsv = "",
            primaryExprText = genomes[1].describePrimary(),
            secondaryExprText = genomes[1].describeSecondary(),
            slotSplitLabel = "${genomes[1].primarySlots}+${5 - genomes[1].primarySlots}",
            genomeJson = SymbolicFormula.genomeToJson(genomes[1]),
            usedFeaturesCsv = "f40,co1",
            status = "MILESTONE_2J",
            rejectionStage = "",
            rejectionReason = "Palier 2J",
            discoveryStreak = 2,
            rollingStreak = 2,
            confirmationStreak = 1,
            sealedStreak = 1,
            combinedStreak = 2,
            discoveryAvgMatches = 0.51f,
            rollingAvgMatches = 0.46f,
            confirmationAvgMatches = 0.35f,
            sealedAvgMatches = 0.31f,
            pValueAdjusted = 0.14,
            fitnessScore = 64.0,
            updatedAt = 1_760_000_000_500L
        )
        val checkpoint = DiscoveredFormulaEntity(
            id = "checkpoint_best_THREE_WEEKS",
            periodName = BacktestPeriod.THREE_WEEKS.name,
            generation = 45,
            parentIdsCsv = "",
            primaryExprText = genomes[2].describePrimary(),
            secondaryExprText = genomes[2].describeSecondary(),
            slotSplitLabel = "${genomes[2].primarySlots}+${5 - genomes[2].primarySlots}",
            genomeJson = SymbolicFormula.genomeToJson(genomes[2]),
            usedFeaturesCsv = "hit1,fg",
            status = "DISCOVERED",
            rejectionStage = "",
            rejectionReason = "",
            discoveryStreak = 1,
            rollingStreak = 1,
            confirmationStreak = 0,
            sealedStreak = 0,
            combinedStreak = 1,
            discoveryAvgMatches = 0.42f,
            rollingAvgMatches = 0.39f,
            confirmationAvgMatches = 0f,
            sealedAvgMatches = 0f,
            pValueAdjusted = 1.0,
            fitnessScore = 31.0,
            updatedAt = 1_760_000_001_000L
        )

        val jsonBackup = StrategyBackupHelper.serializeBackup(
            strategies = listOf(winner2J, winner4J),
            checkpointBest = checkpoint,
            periodName = BacktestPeriod.THREE_WEEKS.name
        )
        assertTrue(jsonBackup.contains(StrategyBackupHelper.BACKUP_FORMAT))

        // Simuler un message recu via WhatsApp / SMS avec du texte autour du bloc JSON
        val wrappedMessage = "Salut voici mes meilleures methodes Loto BJ :\n\n$jsonBackup\n\nBon jeu!"
        val parsed = StrategyBackupHelper.parseBackup(wrappedMessage, BacktestPeriod.THREE_WEEKS.name).getOrThrow()
        assertEquals(2, parsed.strategies.size)
        assertEquals(4, parsed.strategies.first().combinedStreak)
        assertEquals("VALIDATED", parsed.strategies.first().status)
        assertEquals(genomes[0].describePrimary(), parsed.strategies.first().primaryExprText)
        assertNotNull(parsed.checkpointBest)
        assertEquals("checkpoint_best_THREE_WEEKS", parsed.checkpointBest!!.id)

        assertTrue(StrategyBackupHelper.copyBackupToClipboard(context, jsonBackup))
        assertEquals(jsonBackup, StrategyBackupHelper.readTextFromClipboard(context))

        val fileName = StrategyBackupHelper.buildBackupFileName(bestStreak = 4)
        assertTrue(fileName.startsWith("loto_bj_methodes_4j_"))
        assertTrue(fileName.endsWith(".json"))

        clearFileProviderCacheForTest()
        val shareIntent = StrategyBackupHelper.createShareBackupIntent(
            context = context,
            payload = jsonBackup,
            fileName = fileName,
            strategiesCount = 2,
            bestStreak = 4
        )
        assertEquals(Intent.ACTION_CHOOSER, shareIntent.action)
        @Suppress("DEPRECATION")
        val innerIntent = shareIntent.getParcelableExtra<Intent>(Intent.EXTRA_INTENT)
        assertNotNull(innerIntent)
        assertEquals(Intent.ACTION_SEND, innerIntent!!.action)
        @Suppress("DEPRECATION")
        val uri = innerIntent.getParcelableExtra<Uri>(Intent.EXTRA_STREAM)
        assertNotNull(uri)

        assertTrue(StrategyBackupHelper.parseBackup("pas du tout un json", BacktestPeriod.THREE_WEEKS.name).isFailure)
    }

    @Test
    fun coordinator_exportsBestMethodsAndImportsIntoCleanAppInstanceForBacktest() = runBlocking {
        val context = ApplicationProvider.getApplicationContext<Application>()
        val db = AppDatabase.getDatabase(context)
        val dao = db.discoveryDao()
        val coordinator = AutonomousDiscoveryCoordinator.getInstance(context)

        val draws = buildDatedDraws("2026-10-01", 80)
        coordinator.injectHistoryForTest(draws)

        val genome = DiscoveryEngine.generateInitialPopulation(1, 777L).first()
        val sig = genome.describeCompact()
        val entity = DiscoveredFormulaEntity(
            id = "strat_3j_${sig.hashCode().toUInt()}",
            periodName = BacktestPeriod.THREE_WEEKS.name,
            generation = 25,
            parentIdsCsv = "",
            primaryExprText = genome.describePrimary(),
            secondaryExprText = genome.describeSecondary(),
            slotSplitLabel = "${genome.primarySlots}+${5 - genome.primarySlots}",
            genomeJson = SymbolicFormula.genomeToJson(genome),
            usedFeaturesCsv = "f15,gap",
            status = "VALIDATED",
            rejectionStage = "",
            rejectionReason = "Palier 3J exportable",
            discoveryStreak = 3,
            rollingStreak = 3,
            confirmationStreak = 2,
            sealedStreak = 1,
            combinedStreak = 3,
            discoveryAvgMatches = 0.58f,
            rollingAvgMatches = 0.51f,
            confirmationAvgMatches = 0.44f,
            sealedAvgMatches = 0.36f,
            pValueAdjusted = 0.04,
            fitnessScore = 95.0,
            updatedAt = System.currentTimeMillis()
        )
        dao.upsertFormulas(listOf(entity))

        val exported = coordinator.buildSavedStrategiesBackup().getOrThrow()
        assertEquals(1, exported.strategiesCount)
        assertEquals(3, exported.bestStreak)

        // Simuler une seconde application vierge en effacant les strategies locales
        withContext(Dispatchers.IO) {
            db.clearAllTables()
            db.lotoDao().insertAll(draws)
        }
        coordinator.injectHistoryForTest(draws)

        val imported = coordinator.importSavedStrategiesBackup(exported.jsonPayload).getOrThrow()
        assertEquals(1, imported.importedCount)
        assertEquals(3, imported.bestStreakImported)

        val stateAfterImport = coordinator.discoveryState.value
        assertTrue(stateAfterImport.canLaunchBacktest)
        assertEquals(3, stateAfterImport.bestValidatedStreak)
        assertEquals(1, stateAfterImport.savedWinners.size)
        assertEquals(genome.describePrimary(), stateAfterImport.savedWinners.first().primaryExprText)
    }

    @Test
    fun strategyBackupCard_copiesAndPastesBackupBetweenAppsInComposeUi() {
        val context = ApplicationProvider.getApplicationContext<Application>()
        val genome = DiscoveryEngine.generateInitialPopulation(1, 321L).first()
        val validEntity = DiscoveredFormulaEntity(
            id = "strat_4j_ui",
            periodName = BacktestPeriod.THREE_WEEKS.name,
            generation = 19,
            parentIdsCsv = "",
            primaryExprText = genome.describePrimary(),
            secondaryExprText = genome.describeSecondary(),
            slotSplitLabel = "${genome.primarySlots}+${5 - genome.primarySlots}",
            genomeJson = SymbolicFormula.genomeToJson(genome),
            usedFeaturesCsv = "f15",
            status = "VALIDATED",
            rejectionStage = "",
            rejectionReason = "Validé 4J",
            discoveryStreak = 4,
            rollingStreak = 4,
            confirmationStreak = 3,
            sealedStreak = 2,
            combinedStreak = 4,
            discoveryAvgMatches = 0.6f,
            rollingAvgMatches = 0.55f,
            confirmationAvgMatches = 0.5f,
            sealedAvgMatches = 0.45f,
            pValueAdjusted = 0.01,
            fitnessScore = 105.0,
            updatedAt = 1_760_000_000_000L
        )
        val payload = StrategyBackupHelper.serializeBackup(
            strategies = listOf(validEntity),
            periodName = BacktestPeriod.THREE_WEEKS.name
        )
        var importedPayloadSeen: String? = null
        val state = EngineActionState(
            phase = AutonomousPhase.VALIDATED,
            generation = 19,
            bestCombinedStreak = 4,
            bestValidatedStreak = 4,
            activeWinnerId = validEntity.id,
            savedWinners = listOf(
                SymbolicFormulaSummary(
                    id = validEntity.id,
                    generation = 19,
                    primaryExprText = validEntity.primaryExprText,
                    secondaryExprText = validEntity.secondaryExprText,
                    slotSplitLabel = validEntity.slotSplitLabel,
                    usedVariables = listOf("f15"),
                    status = "VALIDATED",
                    statusReason = "Validé 4J",
                    discoveryStreak = 4,
                    rollingStreak = 4,
                    confirmationStreak = 3,
                    sealedStreak = 2,
                    combinedStreak = 4,
                    discoveryAvgMatches = 0.6f,
                    rollingAvgMatches = 0.55f,
                    confirmationAvgMatches = 0.5f,
                    pValueAdjusted = 0.01,
                    fitnessScore = 105.0,
                    isValidated = true
                )
            )
        )

        composeRule.setContent {
            MyApplicationTheme {
                StrategyBackupActionsCard(
                    engineState = state,
                    isBusy = false,
                    onExportStrategiesBackup = {
                        Result.success(
                            StrategyBackupExport(
                                jsonPayload = payload,
                                strategiesCount = 1,
                                hasCheckpoint = false,
                                bestStreak = 4,
                                fileName = "loto_bj_methodes_4j_test.json"
                            )
                        )
                    },
                    onImportStrategiesBackup = { raw ->
                        importedPayloadSeen = raw
                        val parsed = StrategyBackupHelper.parseBackup(raw).getOrThrow()
                        Result.success(
                            StrategyBackupImportResult(
                                importedCount = parsed.strategies.size,
                                updatedCount = 0,
                                checkpointRestored = false,
                                bestStreakImported = parsed.strategies.maxOf { it.combinedStreak },
                                primaryWinnerId = parsed.strategies.first().id,
                                message = "${parsed.strategies.size} meilleure(s) méthode(s) importée(s) (max 4J)"
                            )
                        )
                    }
                )
            }
        }

        // 1. Copier dans le presse-papiers
        composeRule.onNodeWithTag("copy_strategies_backup_button").assertIsEnabled().performClick()
        composeRule.waitForIdle()
        composeRule.onNodeWithTag("strategies_backup_feedback")
            .assertIsDisplayed()
            .assertTextContains("copiée", substring = true)
        assertEquals(payload, StrategyBackupHelper.readTextFromClipboard(context))

        // 2. Coller directement depuis le presse-papiers (transfert vers autre appli)
        composeRule.onNodeWithTag("paste_strategies_backup_button").assertIsEnabled().performClick()
        composeRule.waitForIdle()
        assertEquals(payload, importedPayloadSeen)
        composeRule.onNodeWithTag("strategies_backup_feedback")
            .assertIsDisplayed()
            .assertTextContains("importée(s)", substring = true)

        // 3. Si le presse-papiers est vide, ouverture du dialogue de collage manuel
        val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        clipboard.setPrimaryClip(android.content.ClipData.newPlainText("empty", ""))
        importedPayloadSeen = null

        composeRule.onNodeWithTag("paste_strategies_backup_button").performClick()
        composeRule.waitForIdle()
        composeRule.onNodeWithTag("strategies_paste_dialog").assertIsDisplayed()
        composeRule.onNodeWithTag("strategies_paste_input").performTextReplacement(payload)
        composeRule.onNodeWithTag("strategies_paste_confirm").assertIsEnabled().performClick()
        composeRule.waitForIdle()
        assertEquals(payload, importedPayloadSeen)
    }

    @Test
    fun configurableThreeDayAndCustomPeriods_workAndRpnMatchesAst() {
        val sdf = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.US).apply {
            timeZone = LotoDateUtils.beninTz
        }
        val draws = buildDatedDraws("2026-10-01", 90)
        val refNowMs = sdf.parse("2026-10-23 12:00")!!.time

        // 1. DateUtils fastDayNumber & sliding window a partir de 3 jours
        assertTrue(LotoDateUtils.areConsecutiveCalendarDays("2026-10-20", "2026-10-21"))
        assertFalse(LotoDateUtils.areConsecutiveCalendarDays("2026-10-20", "2026-10-22"))
        val win3 = LotoDateUtils.resolveSlidingWindow(draws, days = 3, referenceNowMs = refNowMs)
        val win4 = LotoDateUtils.resolveSlidingWindow(draws, days = 4, referenceNowMs = refNowMs)
        assertTrue(win3.draws.isNotEmpty())
        assertTrue(win4.draws.size >= win3.draws.size)

        // 2. Build causal cache & run 1 generation on 3J and custom 4J
        val cache3 = DiscoveryEngine.buildCausalCache(
            history = draws,
            period = BacktestPeriod.THREE_DAYS,
            referenceNowMs = refNowMs
        )
        val cache4 = DiscoveryEngine.buildCausalCache(
            history = draws,
            period = BacktestPeriod.THREE_DAYS,
            referenceNowMs = refNowMs,
            customDays = 4
        )
        assertTrue(cache3.steps.isNotEmpty())
        assertTrue(cache4.steps.size >= cache3.steps.size)

        val initState = DiscoveryEngine.createInitialState(sessionSalt = 42L)
        val gen3 = DiscoveryEngine.executeEvolutionGeneration(initState, cache3, BacktestPeriod.THREE_DAYS.name)
        assertEquals(1, gen3.nextState.generation)
        assertTrue(gen3.nextState.experimentsCount > 0)

        // 3. Parity check between compiled RPN bytecode and AST SymbolicFormula.evaluateForNumber
        val population = DiscoveryEngine.generateInitialPopulation(12, 99L)
        val sampleMatrix = cache4.steps.first().rawMatrix
        for (genome in population) {
            val rpnA = SymbolicFormula.compileToRpn(genome.primaryExpr)
            val rpnB = SymbolicFormula.compileToRpn(genome.secondaryExpr)
            for (ball in 1..90) {
                val astA = SymbolicFormula.evaluateForNumber(genome.primaryExpr, sampleMatrix, ball, 0.25)
                val vmA = rpnA.evaluateCpu(sampleMatrix, ball, 0.25)
                assertEquals(astA, vmA, 1e-6)

                val astB = SymbolicFormula.evaluateForNumber(genome.secondaryExpr, sampleMatrix, ball, 0.25)
                val vmB = rpnB.evaluateCpu(sampleMatrix, ball, 0.25)
                assertEquals(astB, vmB, 1e-6)
            }
        }

        // 4. Backtest & Scientific comparison on 3J and custom 4J
        val le = LearningEngine()
        val bt3 = le.backtest(draws, PredictionMethod.AUTO_SEARCH, BacktestPeriod.THREE_DAYS, referenceNowMs = refNowMs, customDays = 3)
        val bt4 = le.backtest(draws, PredictionMethod.AUTO_SEARCH, BacktestPeriod.THREE_DAYS, referenceNowMs = refNowMs, customDays = 4)
        assertTrue(bt3.totalTested > 0)
        assertTrue(bt4.totalTested >= bt3.totalTested)
        val sci = le.runScientificComparison(draws, period = BacktestPeriod.THREE_DAYS, customDays = 4, referenceNowMs = refNowMs)
        assertTrue(sci.indepTestComparisons.isNotEmpty())
    }

    @Test
    fun predictionsScreen_configurableWorkingPeriodFromThreeDays_updatesSelection() {
        val selectedPeriodState = mutableStateOf(BacktestPeriod.THREE_WEEKS)
        val customDaysState = mutableStateOf<Int?>(21)

        composeRule.setContent {
            MyApplicationTheme {
                PredictionsScreen(
                    engineState = EngineActionState(),
                    prediction = null,
                    backtestResult = null,
                    selectedMethod = PredictionMethod.AUTO_SEARCH,
                    selectedPeriod = selectedPeriodState.value,
                    customPeriodDays = customDaysState.value,
                    selectedHoraire = "14H",
                    availableHoraires = listOf("8H", "14H", "18H"),
                    resultsCount = 40,
                    isLoading = false,
                    onSelectPeriod = { p ->
                        selectedPeriodState.value = p
                        customDaysState.value = p.days
                    },
                    onSelectCustomPeriodDays = { days ->
                        customDaysState.value = days
                        selectedPeriodState.value = BacktestPeriod.THREE_DAYS
                    },
                    onSelectHoraire = {},
                    onStartAnalysis = {},
                    onSuspendAnalysis = {},
                    onResumeAnalysis = {},
                    onStopAnalysis = {},
                    onLaunchBacktestAndPrediction = {}
                )
            }
        }

        // Cliquer sur le preset 1J : bouton '-' desactive a 1J (plancher strict a 1 jour)
        composeRule.onNodeWithTag("working_period_chip_ONE_DAY").assertIsDisplayed().performClick()
        composeRule.waitForIdle()
        assertEquals(BacktestPeriod.ONE_DAY, selectedPeriodState.value)
        assertEquals(1, customDaysState.value)
        composeRule.onNodeWithTag("working_period_minus").assertIsNotEnabled()
        composeRule.onNodeWithTag("working_period_plus").assertIsEnabled().performClick()
        composeRule.onNodeWithTag("working_period_value").assertTextContains("2J", substring = true)

        // Cliquer sur le preset 3J puis ajuster a 4J
        composeRule.onNodeWithTag("working_period_chip_THREE_DAYS").assertIsDisplayed().performClick()
        composeRule.waitForIdle()
        assertEquals(BacktestPeriod.THREE_DAYS, selectedPeriodState.value)
        assertEquals(3, customDaysState.value)
        composeRule.onNodeWithTag("working_period_minus").assertIsEnabled()
        composeRule.onNodeWithTag("working_period_plus").assertIsEnabled().performClick()
        composeRule.onNodeWithTag("working_period_value").assertTextContains("4J", substring = true)
        composeRule.onNodeWithTag("working_period_apply").assertIsEnabled().performClick()
        composeRule.waitForIdle()
        assertEquals(4, customDaysState.value)
    }

    @Test
    fun livePredictionMatchesHistoricalBacktestIdentically_andApprovedLedgerIsAppendOnly() {
        val fullHistory = buildDatedDraws("2026-10-01", 40)
        val initialHistory = fullHistory.dropLast(1)
        val newlyArrivedDraw = fullHistory.last()
        val genome = SymbolicGenome(
            id = "g_sync_android_test",
            generation = 1,
            parentIds = emptyList(),
            primaryExpr = ExprNode.Binary(
                BinaryOp.ADD,
                ExprNode.Var(RawFeature.SAME_DAY_REP),
                ExprNode.Var(RawFeature.FREQ_W15)
            ),
            secondaryExpr = ExprNode.Var(RawFeature.GAP_NORM),
            primarySlots = 3,
            samplingTemp = 0.7,
            exploreNoise = 0.0,
            seed = 88L
        )

        val initialCache = DiscoveryEngine.buildCausalCache(
            history = initialHistory,
            period = BacktestPeriod.SEVEN_DAYS,
            referenceNowMs = newlyArrivedDraw.timestamp - 60_000L,
            customDays = 7
        )
        val (btInitial, ledgerInitial) = DiscoveryEngine.mergeAppendOnlyApprovedBacktest(
            genome = genome,
            cache = initialCache,
            existingLedger = null,
            generationsExecuted = 1,
            experimentsCount = 24
        )
        assertTrue(btInitial.isAppendOnlyApproved)
        assertTrue(btInitial.steps.isNotEmpty())

        val targetHoraire = LotoDateUtils.normalizeHoraire(newlyArrivedDraw)
        val livePred = DiscoveryEngine.predictNextWithGenome(
            genome = genome,
            cleanHistory = initialHistory,
            targetHoraire = targetHoraire,
            validatedBacktest = btInitial,
            targetTimestamp = newlyArrivedDraw.timestamp
        )
        val ledgerWithLive = DiscoveryEngine.recordLivePredictionInLedger(
            ledger = ledgerInitial,
            prediction = livePred,
            targetTimestamp = newlyArrivedDraw.timestamp
        )

        val updatedCache = DiscoveryEngine.buildCausalCache(
            history = fullHistory,
            period = BacktestPeriod.SEVEN_DAYS,
            referenceNowMs = newlyArrivedDraw.timestamp + 60_000L,
            customDays = 7
        )
        val (btUpdated, _) = DiscoveryEngine.mergeAppendOnlyApprovedBacktest(
            genome = genome,
            cache = updatedCache,
            existingLedger = ledgerWithLive,
            generationsExecuted = 2,
            experimentsCount = 48
        )

        assertEquals(btInitial.steps.size, btUpdated.approvedStepsPreservedCount)
        assertEquals(1, btUpdated.newlyAppendedStepsCount)
        assertEquals(btInitial.steps, btUpdated.steps.takeLast(btInitial.steps.size))

        val appendedStep = btUpdated.steps.first()
        assertEquals(newlyArrivedDraw.uniqueId, appendedStep.drawId)
        assertEquals(livePred.predictedDraw.sorted(), appendedStep.predictedNumbers.sorted())
        assertEquals(1, btUpdated.livePredictionsMatchedCount)
    }

    @Test
    fun chercheurEngineDistillsAbstractPrinciples_andInjectsIntoDiscoveryEngine() {
        val history = buildDatedDraws("2026-10-01", 40)
        val cache = DiscoveryEngine.buildCausalCache(
            history = history,
            period = BacktestPeriod.THREE_WEEKS,
            referenceNowMs = history.last().timestamp + 1_000L
        )
        val cycle = ChercheurEngine.runResearchAndDistill(
            cache = cache,
            cycleIndex = 1,
            generation = 1,
            seenSignatures = emptySet()
        )
        assertTrue(cycle.teachings.isNotEmpty())
        assertTrue(cycle.synthesizedGenomes.isNotEmpty())
        assertTrue(cycle.teachings.all { it.favoredFeatures.isNotEmpty() })

        val initState = DiscoveryEngine.createInitialState(sessionSalt = 77L)
        val gen = DiscoveryEngine.executeEvolutionGeneration(
            state = initState,
            cache = cache,
            periodName = BacktestPeriod.THREE_WEEKS.name,
            injectedGenomes = cycle.synthesizedGenomes
        )
        assertEquals(cycle.synthesizedGenomes.size, gen.injectedOutcomes.size)
        assertTrue(gen.injectedOutcomes.values.all { it == "KEPT" || it == "DISCARDED" })
    }

    @Test
    fun dailyQualificationStrictlyRequiresThreeDrawsWithTwoPlus_android() {
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
        assertFalse(day1.isQualifyingDay)
        assertEquals(3, day2.drawsWithTwoPlus)
        assertTrue(day2.isQualifyingDay)
    }

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

    private fun clearFileProviderCacheForTest() {
        runCatching {
            val field = androidx.core.content.FileProvider::class.java.getDeclaredField("sCache")
            field.isAccessible = true
            @Suppress("UNCHECKED_CAST")
            val cache = field.get(null) as? MutableMap<*, *>
            cache?.clear()
        }
    }
}
