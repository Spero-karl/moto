package com.lotobj.pc.engine

import com.lotobj.pc.data.ArchiveImportResult
import com.lotobj.pc.data.FetchSummary
import com.lotobj.pc.data.LotoDataRepository
import com.lotobj.pc.model.AutonomousPhase
import com.lotobj.pc.model.BacktestPeriod
import com.lotobj.pc.model.BacktestResult
import com.lotobj.pc.model.ChercheurStatusSummary
import com.lotobj.pc.model.ChercheurTeaching
import com.lotobj.pc.model.ComputeMode
import com.lotobj.pc.model.DiscoveredFormulaRecord
import com.lotobj.pc.model.EngineActionState
import com.lotobj.pc.model.FourStageValidationMetrics
import com.lotobj.pc.model.HardwareCapabilities
import com.lotobj.pc.model.HardwareTelemetry
import com.lotobj.pc.model.LotoDraw
import com.lotobj.pc.model.NextDrawTarget
import com.lotobj.pc.model.PredictionMethod
import com.lotobj.pc.model.PredictionResult
import com.lotobj.pc.model.PrecisionGenome
import com.lotobj.pc.model.PrecisionTargetMode
import com.lotobj.pc.model.PrecisionUiState
import com.lotobj.pc.model.ScientificReport
import com.lotobj.pc.model.SymbolicFormulaSummary
import com.lotobj.pc.model.SymbolicGenome
import com.lotobj.pc.util.BacktestExportHelper
import com.lotobj.pc.util.DesktopClipboard
import com.lotobj.pc.util.LotoDateUtils
import com.lotobj.pc.util.StrategyBackupHelper
import com.lotobj.pc.util.ThreeWeekWindow
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.yield
import java.io.Closeable
import java.io.File
import java.util.Locale
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.ForkJoinPool

data class PcWorkstationUiState(
    val allDraws: List<LotoDraw> = emptyList(),
    val threeWeekWindow: ThreeWeekWindow = LotoDateUtils.resolveThreeWeekWindow(emptyList()),
    val nextDrawTarget: NextDrawTarget = LotoDateUtils.resolveNextUpcomingDraw(emptyList()),
    val engineState: EngineActionState = EngineActionState(),
    val backtestResult: BacktestResult? = null,
    val prediction: PredictionResult? = null,
    val hardwareCaps: HardwareCapabilities,
    val hardwareTelemetry: HardwareTelemetry = HardwareTelemetry(),
    val computeMode: ComputeMode = ComputeMode.GPU_AND_CPU_MAX,
    val populationSize: Int = 256,
    val monteCarloSims: Int = 5_000,
    val selectedPeriod: BacktestPeriod = BacktestPeriod.THREE_WEEKS,
    val customPeriodDays: Int? = BacktestPeriod.THREE_WEEKS.days,
    val isFetchingOnline: Boolean = false,
    val isComparing: Boolean = false,
    val scientificReport: ScientificReport? = null,
    val selectedComparisonMethod: PredictionMethod = PredictionMethod.AUTO_SEARCH,
    val selectedFactoryDomain: MathTheoremDomain? = null,
    val isFactoryRunning: Boolean = false,
    val lastFactoryReport: StrategyFactorySearchReport? = null,
    val statusBanner: String = "Prêt — Station de calcul PC Loto BJ",
    val detailedReportText: String = ""
)

class PcWorkstationCoordinator(
    private val repository: LotoDataRepository = LotoDataRepository(),
    val gpuAccelerator: GpuOpenClAccelerator = GpuOpenClAccelerator()
) : Closeable {

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var evolutionJob: Job? = null
    private var backtestJob: Job? = null
    private var telemetryJob: Job? = null
    private var precisionJob: Job? = null
    private var precisionBacktestJob: Job? = null

    @Volatile
    internal var clockProvider: () -> Long = { System.currentTimeMillis() }

    private var cpuPoolMax: ForkJoinPool = createPool(Runtime.getRuntime().availableProcessors().coerceAtLeast(1))
    private var cpuPoolBalanced: ForkJoinPool = createPool((Runtime.getRuntime().availableProcessors() / 2).coerceAtLeast(1))

    private var cachedCausalKey: String = ""
    private var cachedCausal: FourStageCausalCache? = null
    private val savedFormulasCache = ConcurrentHashMap<String, List<DiscoveredFormulaRecord>>()
    private var activeEvolutionState: SymbolicEvolutionState? = null
    private var winningCandidateEval: GenomeCandidateEvaluation? = null
    private var bestCandidateEvalOverall: GenomeCandidateEvaluation? = null
    private var activeWinnerGenome: SymbolicGenome? = null
    private var activeWinnerRecord: DiscoveredFormulaRecord? = null
    private val unlockedMilestones = mutableMapOf<Int, GenomeCandidateEvaluation>()
    private val sessionExcludedSignatures = LinkedHashSet<String>()
    private val liveLogsBuffer = ArrayDeque<String>(64)

    private fun getCachedFormulas(periodName: String): List<DiscoveredFormulaRecord> {
        return savedFormulasCache.computeIfAbsent(periodName) { key ->
            repository.loadFormulas(key)
        }
    }

    private fun upsertFormulasCached(records: List<DiscoveredFormulaRecord>) {
        if (records.isEmpty()) return
        repository.upsertFormulas(records)
        val affectedPeriods = records.map { it.periodName }.toSet()
        for (p in affectedPeriods) {
            savedFormulasCache[p] = repository.loadFormulas(p)
        }
    }

    // État du Moteur Précision 100 % Indépendant (1er N° & 2 N° exacts)
    private var precisionCacheKey: String = ""
    private var precisionCache: PrecisionCausalCache? = null
    private var precisionEvoState: PrecisionEvolutionState? = null
    private var precisionBestGenome: PrecisionGenome? = null
    private var precisionWinnerEval: PrecisionCandidateEval? = null
    private val precisionLogsBuffer = ArrayDeque<String>(48)
    @Volatile private var lastLoggedGpuError: String = ""

    // État du moteur "Le Chercheur"
    private val chercheurLock = Any()
    private val pendingChercheurGenomes = mutableListOf<SymbolicGenome>()
    private val genomeToTeachingId = mutableMapOf<String, String>()
    private val activeTeachings = mutableListOf<ChercheurTeaching>()
    private var chercheurCycles = 0
    private var totalProducedTeachings = 0
    private var totalInjectedStrategies = 0
    private var totalKeptStrategies = 0
    private var totalDiscardedStrategies = 0

    @Volatile private var lastEvalPerSec: Double = 0.0
    @Volatile private var lastGpuMs: Double = 0.0
    @Volatile private var lastCpuMs: Double = 0.0

    private val initialCaps = HardwareMonitor.detectCapabilities(gpuAccelerator)
    private val _uiState = MutableStateFlow(
        PcWorkstationUiState(
            hardwareCaps = initialCaps,
            computeMode = if (initialCaps.gpuAvailable) ComputeMode.GPU_AND_CPU_MAX else ComputeMode.CPU_MAX_CORES,
            populationSize = recommendedPopulationSize(initialCaps),
            monteCarloSims = if (initialCaps.logicalCores >= 8) 10_000 else 5_000
        )
    )
    val uiState: StateFlow<PcWorkstationUiState> = _uiState.asStateFlow()

    private val _precisionUiState = MutableStateFlow(PrecisionUiState())
    val precisionUiState: StateFlow<PrecisionUiState> = _precisionUiState.asStateFlow()

    init {
        loadInitialLocalData()
        startTelemetryMonitor()
    }

    private fun activePool(mode: ComputeMode): ForkJoinPool {
        return when (mode) {
            ComputeMode.GPU_AND_CPU_MAX, ComputeMode.CPU_MAX_CORES -> cpuPoolMax
            ComputeMode.BALANCED -> cpuPoolBalanced
        }
    }

    private fun recommendedPopulationSize(caps: HardwareCapabilities): Int {
        return when {
            caps.gpuAvailable && caps.gpuCount >= 2 -> 4096
            caps.gpuAvailable && caps.maxJvmHeapMb >= 2048 -> 2048
            caps.gpuAvailable -> 1024
            caps.logicalCores >= 8 -> 1024
            else -> 512
        }
    }

    private fun createPool(threads: Int): ForkJoinPool {
        return ForkJoinPool(threads.coerceAtLeast(1))
    }

    private fun startTelemetryMonitor() {
        telemetryJob?.cancel()
        telemetryJob = scope.launch {
            while (isActive) {
                val st = _uiState.value
                val pool = activePool(st.computeMode)
                val sample = HardwareMonitor.sampleTelemetry(
                    activeWorkers = pool.parallelism,
                    gpuAccelerator = gpuAccelerator,
                    evalPerSec = lastEvalPerSec,
                    lastGpuBatchMs = lastGpuMs,
                    lastCpuBatchMs = lastCpuMs
                )
                _uiState.update { it.copy(hardwareTelemetry = sample) }
                delay(1000L)
            }
        }
    }

    private fun loadInitialLocalData() {
        val draws = repository.loadSavedDraws()
        val window = LotoDateUtils.resolveThreeWeekWindow(draws)
        val nextDraw = LotoDateUtils.resolveNextUpcomingDraw(draws)
        val persistedFormulas = getCachedFormulas(BacktestPeriod.THREE_WEEKS.name)
        val winners = extractSavedWinnerSummaries(persistedFormulas)
        val topRecord = persistedFormulas
            .filter { it.status == "VALIDATED" || it.status.startsWith("MILESTONE_") || it.combinedStreak >= 1 }
            .maxWithOrNull(
                compareBy<DiscoveredFormulaRecord> { it.combinedStreak }
                    .thenBy { it.fitnessScore }
                    .thenBy { it.updatedAt }
            )
        if (topRecord != null) {
            runCatching { SymbolicFormula.genomeFromJson(topRecord.genomeJson) }.getOrNull()?.let { g ->
                activeWinnerGenome = g
                activeWinnerRecord = topRecord
            }
        }
        val bestValidatedStreak = winners.maxOfOrNull { it.combinedStreak } ?: 0
        appendLog("Station prête (${draws.size} tirages locaux • ${initialCaps.logicalCores} cœurs CPU • ${initialCaps.gpuBackendNote}).")

        val restoredPrecMode = _precisionUiState.value.targetMode
        val restoredPrecGenome = repository.loadPrecisionBestGenome(restoredPrecMode)
        precisionBestGenome = restoredPrecGenome
        appendPrecisionLog("Moteur 2 Précision (${restoredPrecMode.label}) prêt (${initialCaps.gpuBackendNote}).")

        _uiState.update { cur ->
            cur.copy(
                allDraws = draws,
                threeWeekWindow = window,
                nextDrawTarget = nextDraw,
                engineState = cur.engineState.copy(
                    savedWinners = winners,
                    bestValidatedStreak = bestValidatedStreak,
                    activeWinnerId = topRecord?.id ?: winners.firstOrNull()?.id.orEmpty(),
                    currentPrimaryExpr = topRecord?.primaryExprText.orEmpty(),
                    currentSecondaryExpr = topRecord?.secondaryExprText.orEmpty(),
                    currentSlotSplit = topRecord?.slotSplitLabel.orEmpty(),
                    liveLogs = liveLogsBuffer.toList(),
                    honestStatusNote = if (draws.size < 8) {
                        "Cliquez sur « Télécharger Résultats LNB » ou importez un fichier CSV/JSON pour démarrer (min. 8 tirages)."
                    } else {
                        "Base prête (${draws.size} tirages). Cliquez sur « Démarrer Recherche GPU/CPU »."
                    }
                )
            )
        }
        _precisionUiState.update { cur ->
            cur.copy(
                activeFormulaDesc = restoredPrecGenome?.describeCompact().orEmpty(),
                honestStatusNote = when {
                    draws.size < 8 -> "Téléchargez ou importez au moins 8 tirages pour lancer le Moteur 2 (Précision 1er N° / 2 N°)."
                    restoredPrecGenome != null -> "Formule ${restoredPrecMode.label} restaurée — cliquez « Backtest & Prédiction » ou « Démarrer »."
                    else -> "Moteur 2 (${restoredPrecMode.label}) prêt — GPU+CPU disponibles."
                },
                liveLogs = precisionLogsBuffer.toList()
            )
        }
        if (restoredPrecGenome != null && draws.size >= 4) {
            restorePrecisionLedgerIfPresentAsync(restoredPrecMode, restoredPrecGenome, draws)
        }
    }

    fun setComputeMode(mode: ComputeMode) {
        _uiState.update { it.copy(computeMode = mode, statusBanner = "Mode matériel : ${mode.label}") }
    }

    fun setPopulationSize(size: Int) {
        _uiState.update { it.copy(populationSize = size.coerceIn(32, 4096)) }
    }

    fun setMonteCarloSims(sims: Int) {
        if (_uiState.value.monteCarloSims != sims) {
            cachedCausalKey = ""
            cachedCausal = null
            _uiState.update { it.copy(monteCarloSims = sims.coerceIn(500, 50_000)) }
        }
    }

    fun setSelectedPeriod(period: BacktestPeriod) {
        val cur = _uiState.value
        if (cur.selectedPeriod != period || cur.customPeriodDays != period.days) {
            cachedCausalKey = ""
            cachedCausal = null
            precisionCacheKey = ""
            precisionCache = null
            val periodFormulas = getCachedFormulas(period.name)
            val winners = extractSavedWinnerSummaries(periodFormulas)
            val topRecord = periodFormulas
                .filter { it.status == "VALIDATED" || it.status.startsWith("MILESTONE_") || it.combinedStreak >= 1 }
                .maxWithOrNull(compareBy<DiscoveredFormulaRecord> { it.combinedStreak }.thenBy { it.fitnessScore })
            val loadedGenome = topRecord?.let { runCatching { SymbolicFormula.genomeFromJson(it.genomeJson) }.getOrNull() }
            if (loadedGenome != null) {
                activeWinnerGenome = loadedGenome
                activeWinnerRecord = topRecord
            }
            _uiState.update {
                it.copy(
                    selectedPeriod = period,
                    customPeriodDays = period.days,
                    statusBanner = "Période de travail : ${period.label}",
                    engineState = it.engineState.copy(
                        savedWinners = winners.ifEmpty { it.engineState.savedWinners },
                        bestValidatedStreak = maxOf(winners.maxOfOrNull { w -> w.combinedStreak } ?: 0, it.engineState.bestValidatedStreak),
                        activeWinnerId = topRecord?.id ?: winners.firstOrNull()?.id ?: it.engineState.activeWinnerId
                    )
                )
            }
        }
    }

    fun setCustomPeriodDays(days: Int) {
        val safeDays = days.coerceIn(1, 3650)
        val matchedPreset = BacktestPeriod.entries.firstOrNull { it.days == safeDays } ?: _uiState.value.selectedPeriod.let {
            if (it == BacktestPeriod.ALL_HISTORY) BacktestPeriod.ONE_DAY else it
        }
        val cur = _uiState.value
        if (cur.customPeriodDays != safeDays || cur.selectedPeriod != matchedPreset) {
            cachedCausalKey = ""
            cachedCausal = null
            precisionCacheKey = ""
            precisionCache = null
            val periodFormulas = getCachedFormulas(matchedPreset.name)
            val winners = extractSavedWinnerSummaries(periodFormulas)
            val topRecord = periodFormulas
                .filter { it.status == "VALIDATED" || it.status.startsWith("MILESTONE_") || it.combinedStreak >= 1 }
                .maxWithOrNull(compareBy<DiscoveredFormulaRecord> { it.combinedStreak }.thenBy { it.fitnessScore })
            val loadedGenome = topRecord?.let { runCatching { SymbolicFormula.genomeFromJson(it.genomeJson) }.getOrNull() }
            if (loadedGenome != null) {
                activeWinnerGenome = loadedGenome
                activeWinnerRecord = topRecord
            }
            _uiState.update {
                it.copy(
                    selectedPeriod = matchedPreset,
                    customPeriodDays = safeDays,
                    statusBanner = "Période de travail configurée : $safeDays jours",
                    engineState = it.engineState.copy(
                        savedWinners = winners.ifEmpty { it.engineState.savedWinners },
                        bestValidatedStreak = maxOf(winners.maxOfOrNull { w -> w.combinedStreak } ?: 0, it.engineState.bestValidatedStreak),
                        activeWinnerId = topRecord?.id ?: winners.firstOrNull()?.id ?: it.engineState.activeWinnerId
                    )
                )
            }
        }
    }

    fun selectPreferredStrategyDays(days: Int?) {
        val curPeriodName = _uiState.value.selectedPeriod.name
        val cachedRecords = getCachedFormulas(curPeriodName)
        _uiState.update { cur ->
            val updated = cur.engineState.copy(preferredStrategyDays = days)
            val chosen = updated.activeStrategySummary
            if (chosen != null) {
                val rec = cachedRecords.firstOrNull { it.id == chosen.id }
                if (rec != null) {
                    runCatching { SymbolicFormula.genomeFromJson(rec.genomeJson) }.getOrNull()?.let { g ->
                        activeWinnerGenome = g
                        activeWinnerRecord = rec
                    }
                }
            }
            cur.copy(engineState = updated)
        }
    }

    fun setFactoryDomain(domain: MathTheoremDomain?) {
        val label = domain?.label ?: "Toutes Familles (Modulaire + Proba + Triangulation + Hybride)"
        _uiState.update {
            it.copy(
                selectedFactoryDomain = domain,
                statusBanner = "Domaine Strategy Factory : $label"
            )
        }
    }

    fun runStrategyFactoryBurst(batchSize: Int = _uiState.value.populationSize.coerceIn(64, 1024)) {
        val st = _uiState.value
        if (st.isFactoryRunning || st.engineState.isAnalyzing) return
        if (st.allDraws.size < 8) {
            _uiState.update {
                it.copy(statusBanner = "Au moins 8 tirages sont nécessaires pour exécuter Strategy Factory (actuellement ${st.allDraws.size}).")
            }
            return
        }

        scope.launch {
            val pool = activePool(st.computeMode)
            val domainDesc = st.selectedFactoryDomain?.shortLabel ?: "Toutes Familles"
            val customDays = st.customPeriodDays
            _uiState.update {
                it.copy(
                    isFactoryRunning = true,
                    statusBanner = "Strategy Factory [$domainDesc] : génération & test GPU/CPU de $batchSize formules inédites..."
                )
            }
            appendLog("Strategy Factory [$domainDesc] : synthèse & banc d'essai ($batchSize formules inédites)...")
            publishLogs()

            val draws = st.allDraws
            val period = st.selectedPeriod
            val cache = withContext(Dispatchers.Default) {
                val expectedKey = "${period.name}|${customDays ?: "ALL"}|${draws.size}|${draws.lastOrNull()?.uniqueId}|${st.monteCarloSims}"
                if (cachedCausal != null && cachedCausalKey == expectedKey) {
                    cachedCausal!!
                } else {
                    PcDiscoveryEngine.buildCausalCache(
                        history = draws,
                        period = period,
                        monteCarloSims = st.monteCarloSims,
                        pool = pool,
                        customDays = customDays
                    ).also {
                        cachedCausal = it
                        cachedCausalKey = expectedKey
                    }
                }
            }

            val nextGen = (st.engineState.generation + 1).coerceAtLeast(1)
            val confCount = (st.engineState.confirmationTestsCount + 1).coerceAtLeast(1)
            val report = withContext(Dispatchers.Default) {
                StrategyFactory.generateAndTestOptimalPatterns(
                    cache = cache,
                    batchSize = batchSize,
                    generation = nextGen,
                    sessionSalt = System.currentTimeMillis() + nextGen * 313L,
                    preferredDomain = st.selectedFactoryDomain,
                    seenSignatures = sessionExcludedSignatures,
                    minObjectiveStreak = st.engineState.preferredStrategyDays ?: 4,
                    confirmationTestsCount = confCount,
                    computeMode = st.computeMode,
                    cpuPool = pool,
                    gpuAccelerator = gpuAccelerator
                )
            }

            lastEvalPerSec = report.evaluationsPerSec
            if (report.gpuBatchMs > 0.0) lastGpuMs = report.gpuBatchMs

            val now = System.currentTimeMillis()
            val winner = report.auditedWinner
            val toPersist = ArrayList<DiscoveredFormulaRecord>()

            if (winner != null) {
                val sum = winner.toSummary()
                val winnerZeroOk = winner.rollingStabilityPassed
                if (winnerZeroOk && (sum.combinedStreak >= 1 || winner.status == "VALIDATED" || winner.status.startsWith("MILESTONE_"))) {
                    val rec = winner.toRecord(period.name, now).copy(
                        id = if (winner.status == "VALIDATED") winner.genome.id else "fac_${sum.combinedStreak}j_${winner.genome.describeCompact().hashCode().toUInt()}"
                    )
                    toPersist.add(rec)
                    if (activeWinnerGenome == null || sum.combinedStreak >= (activeWinnerRecord?.combinedStreak ?: 0)) {
                        activeWinnerGenome = winner.genome
                        activeWinnerRecord = rec
                    }
                    if (winner.status == "VALIDATED") {
                        winningCandidateEval = winner
                    }
                    unlockedMilestones.putIfAbsent(sum.combinedStreak, winner)
                }
                if (bestCandidateEvalOverall == null || winner.fitnessScore > bestCandidateEvalOverall!!.fitnessScore) {
                    bestCandidateEvalOverall = winner
                    toPersist.add(winner.toRecord(period.name, now).copy(id = "checkpoint_best_${period.name}"))
                }
            }

            for (cand in report.optimalCandidates.take(6)) {
                val sum = cand.toSummary()
                if (cand.rollingStabilityPassed && sum.combinedStreak >= 1 && cand.genome.id != winner?.genome?.id) {
                    val rec = cand.toRecord(period.name, now)
                        .copy(id = "fac_${sum.combinedStreak}j_${cand.genome.describeCompact().hashCode().toUInt()}")
                    toPersist.add(rec)
                }
            }

            if (toPersist.isNotEmpty()) {
                upsertFormulasCached(toPersist)
            }

            activeEvolutionState = activeEvolutionState?.let { curEvo ->
                val mergedSeen = LinkedHashSet(curEvo.seenSignatures).apply {
                    addAll(sessionExcludedSignatures)
                }
                curEvo.copy(seenSignatures = mergedSeen)
            }

            val allSaved = getCachedFormulas(period.name)
            val winners = extractSavedWinnerSummaries(allSaved)
            val bestValStreak = winners.maxOfOrNull { it.combinedStreak } ?: 0
            val fourMetrics = winner?.let {
                FourStageValidationMetrics(
                    discoveryDraws = it.discStats.tested,
                    discoveryStreak = it.discStats.maxConsecutiveQualifyingDays,
                    discoveryQualifyingDays = it.discStats.qualifyingDays,
                    discoveryTwoPlusCount = it.discStats.twoPlusCount,
                    discoveryAvgMatches = it.discStats.avgMatches,
                    rollingDraws = it.rollStats.tested,
                    rollingStreak = it.rollStats.maxConsecutiveQualifyingDays,
                    rollingQualifyingDays = it.rollStats.qualifyingDays,
                    rollingTwoPlusCount = it.rollStats.twoPlusCount,
                    rollingAvgMatches = it.rollStats.avgMatches,
                    rollingStabilityPassed = it.rollingStabilityPassed,
                    confirmationDraws = it.confStats?.tested ?: cache.confirmationSteps.size,
                    confirmationStreak = it.confStats?.maxConsecutiveQualifyingDays ?: 0,
                    confirmationQualifyingDays = it.confStats?.qualifyingDays ?: 0,
                    confirmationTwoPlusCount = it.confStats?.twoPlusCount ?: 0,
                    confirmationAvgMatches = it.confStats?.avgMatches ?: 0f,
                    sealedDraws = it.sealedStats?.tested ?: cache.sealedSteps.size,
                    sealedStreak = it.sealedStats?.maxConsecutiveQualifyingDays ?: 0,
                    sealedQualifyingDays = it.sealedStats?.qualifyingDays ?: 0,
                    sealedTwoPlusCount = it.sealedStats?.twoPlusCount ?: 0,
                    sealedAvgMatches = it.sealedStats?.avgMatches ?: 0f,
                    sealedCausalVerified = it.sealedStats?.causalVerified ?: false,
                    combinedStreak = maxOf(
                        it.discStats.maxConsecutiveQualifyingDays,
                        it.rollStats.maxConsecutiveQualifyingDays,
                        it.confStats?.maxConsecutiveQualifyingDays ?: 0
                    ),
                    combinedQualifyingDays = maxOf(it.discStats.qualifyingDays, it.rollStats.qualifyingDays),
                    combinedMaxZeroStreak = maxOf(it.discStats.maxZeroStreak, it.rollStats.maxZeroStreak),
                    combinedBrierScore = it.discStats.brierScore,
                    isFourStageValidated = it.status == "VALIDATED",
                    rollingZeroMatchRate = it.rollStats.zeroMatchRate,
                    rollingThreePlusCount = it.rollStats.threePlusCount,
                    rollingFourPlusCount = it.rollStats.fourPlusCount,
                    rollingFiveMatchCount = it.rollStats.fiveMatchCount
                )
            }

            appendLog(
                "${report.summaryNote} | ${String.format(Locale.FRANCE, "%.0f", report.evaluationsPerSec)} éval/s" +
                    if (report.gpuBatchMs > 0.0) " (GPU ${String.format(Locale.FRANCE, "%.1f", report.gpuBatchMs)}ms)" else ""
            )

            _uiState.update { cur ->
                val winnerSum = winner?.toSummary()
                cur.copy(
                    isFactoryRunning = false,
                    lastFactoryReport = report,
                    statusBanner = report.summaryNote,
                    engineState = cur.engineState.copy(
                        generation = maxOf(cur.engineState.generation, nextGen),
                        experimentsCount = cur.engineState.experimentsCount + report.unseenEvaluatedCount,
                        confirmationTestsCount = maxOf(cur.engineState.confirmationTestsCount, if (winner != null) confCount else 0),
                        excludedMethodsCount = maxOf(cur.engineState.excludedMethodsCount, sessionExcludedSignatures.size),
                        currentPrimaryExpr = winner?.genome?.describePrimary() ?: cur.engineState.currentPrimaryExpr,
                        currentSecondaryExpr = winner?.genome?.describeSecondary() ?: cur.engineState.currentSecondaryExpr,
                        currentSlotSplit = winner?.genome?.let { "${it.primarySlots}+${5 - it.primarySlots}" } ?: cur.engineState.currentSlotSplit,
                        bestFormulas = report.optimalCandidates.map { it.toSummary() },
                        rejectedFormulas = report.rejectedCandidates,
                        fourStageMetrics = fourMetrics ?: cur.engineState.fourStageMetrics,
                        savedWinners = winners,
                        bestValidatedStreak = maxOf(cur.engineState.bestValidatedStreak, bestValStreak),
                        activeWinnerId = activeWinnerRecord?.id ?: winners.firstOrNull()?.id.orEmpty(),
                        bestZeroMatchRate = winner?.rollStats?.zeroMatchRate ?: cur.engineState.bestZeroMatchRate,
                        bestThreePlusCount = maxOf(cur.engineState.bestThreePlusCount, winnerSum?.rollingThreePlusCount ?: 0),
                        bestFourPlusCount = maxOf(cur.engineState.bestFourPlusCount, winnerSum?.rollingFourPlusCount ?: 0),
                        bestFiveMatchCount = maxOf(cur.engineState.bestFiveMatchCount, winnerSum?.rollingFiveMatchCount ?: 0),
                        activeParadigmName = "Factory : ${st.selectedFactoryDomain?.shortLabel ?: "Multi-Domaines"}",
                        honestStatusNote = report.summaryNote,
                        liveLogs = liveLogsBuffer.toList()
                    )
                )
            }
        }
    }

    fun selectSavedWinner(winnerId: String) {
        val periodName = _uiState.value.selectedPeriod.name
        val record = getCachedFormulas(periodName).firstOrNull { it.id == winnerId }
            ?: getCachedFormulas(BacktestPeriod.THREE_WEEKS.name).firstOrNull { it.id == winnerId }
            ?: BacktestPeriod.entries.asSequence()
                .flatMap { getCachedFormulas(it.name).asSequence() }
                .firstOrNull { it.id == winnerId }
        if (record != null) {
            runCatching { SymbolicFormula.genomeFromJson(record.genomeJson) }.getOrNull()?.let { g ->
                activeWinnerGenome = g
                activeWinnerRecord = record
                _uiState.update { cur ->
                    cur.copy(
                        engineState = cur.engineState.copy(
                            activeWinnerId = winnerId,
                            currentPrimaryExpr = g.describePrimary(),
                            currentSecondaryExpr = g.describeSecondary(),
                            currentSlotSplit = "${g.primarySlots}+${5 - g.primarySlots}"
                        ),
                        statusBanner = "Stratégie ${record.combinedStreak}J sélectionnée : ${g.describeCompact()}"
                    )
                }
            }
        }
    }

    fun fetchLatestOnlineResults() {
        if (_uiState.value.isFetchingOnline) return
        val prevSize = _uiState.value.allDraws.size
        val prevLastId = _uiState.value.allDraws.lastOrNull()?.uniqueId
        scope.launch {
            _uiState.update { it.copy(isFetchingOnline = true, statusBanner = "Téléchargement des derniers tirages LNB Bénin...") }
            val result = repository.fetchLatestResults()
            result.onSuccess { (summary: FetchSummary, allDraws: List<LotoDraw>) ->
                cachedCausalKey = ""
                cachedCausal = null
                val nowMs = clockProvider()
                val window = LotoDateUtils.resolveThreeWeekWindow(allDraws, nowMs)
                val nextDraw = LotoDateUtils.resolveNextUpcomingDraw(allDraws, nowMs)
                val drawsChanged = allDraws.size != prevSize || allDraws.lastOrNull()?.uniqueId != prevLastId
                appendLog("Téléchargement LNB OK : ${summary.fetchedUniqueCount} tirages récupérés (${summary.pagesLoaded} pages, total local ${allDraws.size}).")
                _uiState.update { cur ->
                    cur.copy(
                        allDraws = allDraws,
                        threeWeekWindow = window,
                        nextDrawTarget = nextDraw,
                        isFetchingOnline = false,
                        statusBanner = "${summary.fetchedUniqueCount} tirages LNB synchronisés (total ${allDraws.size})",
                        engineState = cur.engineState.copy(liveLogs = liveLogsBuffer.toList())
                    )
                }
                if (drawsChanged) {
                    onNewDrawsArrivedAutoRefresh()
                }
            }.onFailure { err ->
                appendLog("Erreur réseau LNB : ${err.message}")
                _uiState.update { cur ->
                    cur.copy(
                        isFetchingOnline = false,
                        statusBanner = "Erreur téléchargement LNB : ${err.message}",
                        engineState = cur.engineState.copy(liveLogs = liveLogsBuffer.toList())
                    )
                }
            }
        }
    }

    fun importArchiveFile(file: File) {
        val prevSize = _uiState.value.allDraws.size
        val prevLastId = _uiState.value.allDraws.lastOrNull()?.uniqueId
        scope.launch {
            val res = repository.importArchiveFile(file)
            res.onSuccess { (parsed: ArchiveImportResult, allDraws: List<LotoDraw>) ->
                cachedCausalKey = ""
                cachedCausal = null
                val nowMs = clockProvider()
                val window = LotoDateUtils.resolveThreeWeekWindow(allDraws, nowMs)
                val nextDraw = LotoDateUtils.resolveNextUpcomingDraw(allDraws, nowMs)
                val drawsChanged = allDraws.size != prevSize || allDraws.lastOrNull()?.uniqueId != prevLastId
                appendLog("Import archive '${file.name}' : ${parsed.acceptedCount} acceptés, ${parsed.rejectedCount} rejetés (total ${allDraws.size}).")
                _uiState.update { cur ->
                    cur.copy(
                        allDraws = allDraws,
                        threeWeekWindow = window,
                        nextDrawTarget = nextDraw,
                        statusBanner = "Import réussi : +${parsed.acceptedCount} tirages (${allDraws.size} au total)",
                        engineState = cur.engineState.copy(liveLogs = liveLogsBuffer.toList())
                    )
                }
                if (drawsChanged) {
                    onNewDrawsArrivedAutoRefresh()
                }
            }.onFailure { err ->
                appendLog("Échec import '${file.name}' : ${err.message}")
                _uiState.update { cur ->
                    cur.copy(
                        statusBanner = "Échec import : ${err.message}",
                        engineState = cur.engineState.copy(liveLogs = liveLogsBuffer.toList())
                    )
                }
            }
        }
    }

    internal fun ingestDrawsAndAutoRefreshForTest(allDraws: List<LotoDraw>) {
        val prevSize = _uiState.value.allDraws.size
        val prevLastId = _uiState.value.allDraws.lastOrNull()?.uniqueId
        val sorted = allDraws.distinctBy { it.uniqueId }.sortedWith(compareBy<LotoDraw> { it.timestamp }.thenBy { it.uniqueId })
        repository.mergeAndSaveDraws(sorted)
        cachedCausalKey = ""
        cachedCausal = null
        val nowMs = clockProvider()
        val window = LotoDateUtils.resolveThreeWeekWindow(sorted, nowMs)
        val nextDraw = LotoDateUtils.resolveNextUpcomingDraw(sorted, nowMs)
        val drawsChanged = sorted.size != prevSize || sorted.lastOrNull()?.uniqueId != prevLastId
        _uiState.update { cur ->
            cur.copy(
                allDraws = sorted,
                threeWeekWindow = window,
                nextDrawTarget = nextDraw
            )
        }
        if (drawsChanged) {
            onNewDrawsArrivedAutoRefresh()
        }
    }

    private fun onNewDrawsArrivedAutoRefresh() {
        precisionCacheKey = ""
        precisionCache = null
        val cur = _uiState.value
        if (cur.engineState.backtestTriggered) {
            launchValidatedBacktestAndPrediction()
        } else if (cur.prediction != null && cur.selectedComparisonMethod != PredictionMethod.AUTO_SEARCH) {
            runClassicalMethodBacktestAndPrediction(cur.selectedComparisonMethod)
        }
        val pState = _precisionUiState.value
        if (pState.backtestResult != null || pState.livePrediction != null) {
            launchPrecisionBacktestAndPrediction()
        }
    }

    fun setInfiniteMode(enabled: Boolean) {
        _uiState.update { cur ->
            cur.copy(
                statusBanner = if (enabled) {
                    "Mode infini activé — le moteur cherche jusqu'à Pause ou Stop"
                } else {
                    "Mode infini désactivé — arrêt automatique dès validation"
                },
                engineState = cur.engineState.copy(infiniteMode = enabled)
            )
        }
    }

    fun startInfiniteMode() {
        setInfiniteMode(true)
        val curPhase = _uiState.value.engineState.phase
        when {
            curPhase == AutonomousPhase.ANALYZING -> Unit
            curPhase == AutonomousPhase.SUSPENDED -> resumeDiscovery()
            else -> startAutonomousDiscovery(continueBeyond = activeEvolutionState != null)
        }
    }

    fun restartDiscoveryFromZero() {
        evolutionJob?.cancel()
        evolutionJob = null
        activeEvolutionState = null
        winningCandidateEval = null
        bestCandidateEvalOverall = null
        unlockedMilestones.clear()
        sessionExcludedSignatures.clear()
        synchronized(chercheurLock) {
            pendingChercheurGenomes.clear()
            genomeToTeachingId.clear()
            activeTeachings.clear()
            chercheurCycles = 0
            totalProducedTeachings = 0
            totalInjectedStrategies = 0
            totalKeptStrategies = 0
            totalDiscardedStrategies = 0
        }
        appendLog("Remise à zéro complète du moteur de recherche (nouvelle session vierge).")
        _uiState.update { cur ->
            cur.copy(
                engineState = cur.engineState.copy(
                    phase = AutonomousPhase.IDLE,
                    generation = 0,
                    experimentsCount = 0,
                    confirmationTestsCount = 0,
                    bestCombinedStreak = 0,
                    excludedMethodsCount = 0,
                    stagnationGenerations = 0,
                    chercheurStatus = ChercheurStatusSummary(),
                    liveLogs = liveLogsBuffer.toList()
                )
            )
        }
        startAutonomousDiscovery(continueBeyond = false)
    }

    fun runChercheurAndInject() {
        val st = _uiState.value
        if (st.allDraws.size < 8) {
            _uiState.update {
                it.copy(statusBanner = "Le Chercheur exige au moins 8 tirages historiques (actuellement ${st.allDraws.size}).")
            }
            return
        }
        scope.launch {
            val pool = activePool(st.computeMode)
            val cache = getOrBuildCausalCache(st.allDraws, st.selectedPeriod, st.customPeriodDays, st.monteCarloSims, pool)
            if (cache.steps.size < 4) {
                _uiState.update { it.copy(statusBanner = "Historique causal insuffisant pour Le Chercheur.") }
                return@launch
            }
            val nextCycle = synchronized(chercheurLock) { ++chercheurCycles }
            _uiState.update { cur ->
                cur.copy(
                    engineState = cur.engineState.copy(
                        chercheurStatus = cur.engineState.chercheurStatus.copy(
                            isRunning = true,
                            lastLessonNote = "Le Chercheur étudie chaque numéro, position (Pos1..5) et tirage complet..."
                        )
                    )
                )
            }

            val cycleOutput = withContext(Dispatchers.Default) {
                ChercheurEngine.runResearchAndDistill(
                    cache = cache,
                    cycleIndex = nextCycle,
                    generation = st.engineState.generation.coerceAtLeast(1),
                    seenSignatures = sessionExcludedSignatures
                )
            }

            // Si l'évolution tourne déjà, on met en file d'attente pour la prochaine génération ; sinon on audite immédiatement via PcDiscoveryEngine !
            if (_uiState.value.engineState.phase == AutonomousPhase.ANALYZING) {
                synchronized(chercheurLock) {
                    pendingChercheurGenomes.addAll(cycleOutput.synthesizedGenomes)
                    genomeToTeachingId.putAll(cycleOutput.teachingByGenomeId)
                    totalProducedTeachings += cycleOutput.teachings.size
                    totalInjectedStrategies += cycleOutput.synthesizedGenomes.size
                    activeTeachings.addAll(0, cycleOutput.teachings)
                    while (activeTeachings.size > 16) activeTeachings.removeLast()
                }
                appendLog("${cycleOutput.cycleSummaryNote} (injecté dans le flux GPU/CPU)")
                publishChercheurStatus(isRunning = false, note = cycleOutput.cycleSummaryNote)
            } else {
                val baseEvo = activeEvolutionState ?: PcDiscoveryEngine.createInitialState(
                    populationSize = st.populationSize.coerceIn(64, 4096),
                    sessionSalt = System.currentTimeMillis() % 100_000L,
                    preExcludedSignatures = sessionExcludedSignatures
                )
                val evalRes = withContext(Dispatchers.Default) {
                    PcDiscoveryEngine.executeEvolutionGeneration(
                        state = baseEvo,
                        cache = cache,
                        periodName = st.selectedPeriod.name,
                        minObjectiveStreak = st.engineState.preferredStrategyDays ?: 4,
                        populationSize = st.populationSize.coerceIn(64, 4096),
                        computeMode = st.computeMode,
                        cpuPool = pool,
                        gpuAccelerator = gpuAccelerator,
                        injectedGenomes = cycleOutput.synthesizedGenomes
                    )
                }
                activeEvolutionState = evalRes.nextState
                if (evalRes.persistedRecords.isNotEmpty()) {
                    upsertFormulasCached(evalRes.persistedRecords)
                }
                evalRes.bestTrainEval?.let { top ->
                    if (top.rollingStabilityPassed && top.toSummary().combinedStreak >= 2) {
                        val rec = top.toRecord(st.selectedPeriod.name, System.currentTimeMillis())
                            .copy(id = "ch_${top.toSummary().combinedStreak}j_${top.genome.describeCompact().hashCode().toUInt()}")
                        upsertFormulasCached(listOf(rec))
                        if (activeWinnerGenome == null || top.toSummary().combinedStreak > (activeWinnerRecord?.combinedStreak ?: 0)) {
                            activeWinnerGenome = top.genome
                            activeWinnerRecord = rec
                        }
                    }
                }
                recordChercheurOutcomes(cycleOutput, evalRes.injectedOutcomes)
                val allSaved = getCachedFormulas(st.selectedPeriod.name)
                val winners = extractSavedWinnerSummaries(allSaved)
                appendLog(cycleOutput.cycleSummaryNote)
                _uiState.update { cur ->
                    cur.copy(
                        statusBanner = cycleOutput.cycleSummaryNote,
                        engineState = cur.engineState.copy(
                            savedWinners = winners,
                            bestValidatedStreak = maxOf(cur.engineState.bestValidatedStreak, winners.maxOfOrNull { it.combinedStreak } ?: 0),
                            excludedMethodsCount = evalRes.nextState.excludedSignaturesCount,
                            liveLogs = liveLogsBuffer.toList()
                        )
                    )
                }
            }
        }
    }

    private fun recordChercheurOutcomes(
        cycleOutput: ChercheurCycleOutput?,
        injectedOutcomes: Map<String, String>
    ) {
        synchronized(chercheurLock) {
            if (cycleOutput != null) {
                genomeToTeachingId.putAll(cycleOutput.teachingByGenomeId)
                totalProducedTeachings += cycleOutput.teachings.size
                totalInjectedStrategies += cycleOutput.synthesizedGenomes.size
                activeTeachings.addAll(0, cycleOutput.teachings)
                while (activeTeachings.size > 16) activeTeachings.removeLast()
            }
            for ((gId, verdict) in injectedOutcomes) {
                if (verdict == "KEPT") totalKeptStrategies++ else totalDiscardedStrategies++
                val tId = genomeToTeachingId[gId]
                if (tId != null) {
                    val idx = activeTeachings.indexOfFirst { it.id == tId }
                    if (idx >= 0) {
                        val existing = activeTeachings[idx]
                        val resolved = if (existing.status == "KEPT" || verdict == "KEPT") "KEPT" else "DISCARDED"
                        activeTeachings[idx] = existing.copy(status = resolved)
                    }
                }
            }
        }
        val summaryNote = cycleOutput?.cycleSummaryNote
            ?: "Audit Chercheur : $totalKeptStrategies gardées / $totalDiscardedStrategies jetées sur $totalInjectedStrategies injectées."
        publishChercheurStatus(isRunning = false, note = summaryNote)
    }

    private fun publishChercheurStatus(isRunning: Boolean, note: String) {
        val snap = synchronized(chercheurLock) {
            ChercheurStatusSummary(
                isRunning = isRunning,
                cyclesCompleted = chercheurCycles,
                teachingsProduced = totalProducedTeachings,
                teachingsInjected = totalInjectedStrategies,
                strategiesKept = totalKeptStrategies,
                strategiesDiscarded = totalDiscardedStrategies,
                lastLessonNote = note,
                teachings = activeTeachings.toList()
            )
        }
        _uiState.update { cur ->
            cur.copy(engineState = cur.engineState.copy(chercheurStatus = snap))
        }
    }

    private suspend fun getOrBuildCausalCache(
        draws: List<LotoDraw>,
        period: BacktestPeriod,
        customDays: Int?,
        monteCarloSims: Int,
        pool: ForkJoinPool
    ): FourStageCausalCache = withContext(Dispatchers.Default) {
        val nowMs = clockProvider()
        val dayBucket = LotoDateUtils.formatDayKey(nowMs)
        val expectedKey = "${period.name}|${customDays ?: "ALL"}|${draws.size}|${draws.firstOrNull()?.uniqueId}|${draws.lastOrNull()?.uniqueId}|$monteCarloSims|$dayBucket"
        if (cachedCausal != null && cachedCausalKey == expectedKey) {
            cachedCausal!!
        } else {
            PcDiscoveryEngine.buildCausalCache(
                history = draws,
                period = period,
                referenceNowMs = nowMs,
                monteCarloSims = monteCarloSims,
                pool = pool,
                customDays = customDays
            ).also {
                cachedCausal = it
                cachedCausalKey = expectedKey
            }
        }
    }

    fun startAutonomousDiscovery(continueBeyond: Boolean = false) {
        val stateSnap = _uiState.value
        val draws = stateSnap.allDraws
        if (draws.size < 8) {
            _uiState.update { it.copy(statusBanner = "Au moins 8 tirages sont nécessaires (actuellement ${draws.size}).") }
            return
        }

        evolutionJob?.cancel()
        evolutionJob = scope.launch {
            val period = _uiState.value.selectedPeriod
            val customDays = _uiState.value.customPeriodDays
            val pool = activePool(_uiState.value.computeMode)

            if (!continueBeyond || cachedCausal == null) {
                val periodDesc = customDays?.let { "$it jours" } ?: period.label
                appendLog("Préparation cache causal 4 étapes ($periodDesc, ${_uiState.value.monteCarloSims} sim. Monte Carlo)...")
                publishLogs()
                val cache = getOrBuildCausalCache(draws, period, customDays, _uiState.value.monteCarloSims, pool)

                if (cache.steps.size < 3) {
                    appendLog("Fenêtre trop courte (${cache.steps.size} tirages). Téléchargez plus d'historique ou augmentez le nombre de jours.")
                    _uiState.update {
                        it.copy(
                            statusBanner = "Fenêtre trop courte (${cache.steps.size} tirages).",
                            engineState = it.engineState.copy(phase = AutonomousPhase.IDLE, liveLogs = liveLogsBuffer.toList())
                        )
                    }
                    return@launch
                }

                val cachedPeriodFormulas = getCachedFormulas(period.name)
                val savedCheckpoint = cachedPeriodFormulas
                    .firstOrNull { it.id == "checkpoint_best_${period.name}" }
                    ?.let { rec -> runCatching { SymbolicFormula.genomeFromJson(rec.genomeJson) }.getOrNull() }

                val popSize = _uiState.value.populationSize
                val persistedSigs = cachedPeriodFormulas
                    .mapNotNull { rec -> runCatching { SymbolicFormula.genomeFromJson(rec.genomeJson).describeCompact() }.getOrNull() }
                sessionExcludedSignatures.addAll(persistedSigs)
                val baseState = PcDiscoveryEngine.createInitialState(
                    populationSize = popSize,
                    sessionSalt = System.currentTimeMillis() % 100_000L,
                    preExcludedSignatures = sessionExcludedSignatures
                )
                // Utilise l'élite précédente comme parent d'évolution sans jamais ré-évaluer sa formule identique
                val seededBest = listOfNotNull(savedCheckpoint, activeWinnerGenome).mapNotNull { g ->
                    if (cache.discoverySteps.isNotEmpty() && cache.rollingSteps.isNotEmpty()) {
                        val ds = PcDiscoveryEngine.evaluateGenomeOnStage(g, cache.discoverySteps)
                        val rs = PcDiscoveryEngine.evaluateGenomeOnStage(g, cache.rollingSteps)
                        GenomeCandidateEvaluation(
                            genome = g,
                            discStats = ds,
                            rollStats = rs,
                            rollingStabilityPassed = !(rs.tested >= 4 && rs.avgMatches < BacktestResult.RANDOM_EXPECTED_MATCHES * 0.78f),
                            fitnessScore = rs.avgMatches * 140.0,
                            status = "SEED"
                        )
                    } else null
                }
                activeEvolutionState = baseState.copy(
                    bestGenomes = seededBest,
                    seenSignatures = sessionExcludedSignatures
                )
                winningCandidateEval = null
                bestCandidateEvalOverall = null
                unlockedMilestones.clear()
            } else {
                val cur = activeEvolutionState ?: PcDiscoveryEngine.createInitialState(
                    populationSize = _uiState.value.populationSize,
                    preExcludedSignatures = sessionExcludedSignatures
                )
                activeEvolutionState = cur.copy(seenSignatures = sessionExcludedSignatures)
            }

            val cache = cachedCausal ?: return@launch
            val cacheDaysCount = cache.steps.map { it.dayEpoch }.distinct().size.coerceAtLeast(1)
            val effectiveTarget = (_uiState.value.engineState.preferredStrategyDays ?: 4).coerceIn(1, cacheDaysCount)
            _uiState.update { cur ->
                cur.copy(
                    statusBanner = "Recherche GPU/CPU en cours (${cur.populationSize} génomes/gén., ${pool.parallelism} cœurs)...",
                    engineState = cur.engineState.copy(
                        phase = AutonomousPhase.ANALYZING,
                        discoveryDrawsCount = cache.discoverySteps.size,
                        rollingDrawsCount = cache.rollingSteps.size,
                        confirmationDrawsCount = cache.confirmationSteps.size,
                        sealedDrawsCount = cache.sealedSteps.size,
                        periodDateRangeLabel = cache.dateRangeLabel,
                        targetObjectiveStreak = effectiveTarget
                    )
                )
            }

            runEvolutionLoop(cache, period)
        }
    }

    fun suspendDiscovery() {
        evolutionJob?.cancel()
        evolutionJob = null
        val curStreak = _uiState.value.engineState.bestCombinedStreak
        appendLog("Recherche mise en pause à la génération ${_uiState.value.engineState.generation} (meilleur palier : ${curStreak}J).")
        _uiState.update { cur ->
            cur.copy(
                statusBanner = "En pause (checkpoint conservé — cliquez Continuer pour reprendre)",
                engineState = cur.engineState.copy(
                    phase = AutonomousPhase.SUSPENDED,
                    honestStatusNote = "En pause : la population GPU/CPU est conservée intacte. Cliquez sur « Continuer » pour reprendre ou « Stop » pour fermer l'analyse.",
                    liveLogs = liveLogsBuffer.toList()
                )
            )
        }
    }

    fun resumeDiscovery() {
        val st = _uiState.value
        if (!st.engineState.canResume) return
        val cache = cachedCausal
        if (cache == null || activeEvolutionState == null) {
            startAutonomousDiscovery(continueBeyond = false)
            return
        }
        evolutionJob?.cancel()
        evolutionJob = scope.launch {
            appendLog("Reprise (Continuer) de la recherche à la génération ${activeEvolutionState?.generation ?: 0}...")
            _uiState.update { cur ->
                cur.copy(
                    statusBanner = "Continuer : reprise du moteur GPU/CPU...",
                    engineState = cur.engineState.copy(phase = AutonomousPhase.ANALYZING, liveLogs = liveLogsBuffer.toList())
                )
            }
            runEvolutionLoop(cache, st.selectedPeriod)
        }
    }

    fun stopDiscovery() {
        evolutionJob?.cancel()
        evolutionJob = null
        val bestEval = winningCandidateEval ?: bestCandidateEvalOverall
        var savedMsg = "Analyse fermée (Stop)."
        if (bestEval != null) {
            val audited = cachedCausal?.let {
                PcDiscoveryEngine.auditCandidateReadOnly(bestEval, it, activeEvolutionState?.confirmationTestsCount ?: 1)
            } ?: bestEval
            if (audited.discStats.maxConsecutiveQualifyingDays >= 1 || audited.rollStats.maxConsecutiveQualifyingDays >= 1) {
                val rec = audited.toRecord(_uiState.value.selectedPeriod.name, System.currentTimeMillis())
                upsertFormulasCached(listOf(rec))
                activeWinnerGenome = audited.genome
                activeWinnerRecord = rec
                savedMsg = "Analyse fermée (Stop) — Meilleur palier ${audited.toSummary().combinedStreak}J sauvegardé !"
            }
        }
        appendLog(savedMsg)
        refreshSavedWinnersAndState(AutonomousPhase.STOPPED, savedMsg)
    }

    private suspend fun runEvolutionLoop(cache: FourStageCausalCache, period: BacktestPeriod) {
        var lastUiEmitMs = 0L
        while (currentCoroutineContext().isActive) {
            val st = _uiState.value
            if (st.engineState.phase != AutonomousPhase.ANALYZING) break
            val stateBefore = activeEvolutionState ?: break
            val cacheDaysCount = cache.steps.map { it.dayEpoch }.distinct().size.coerceAtLeast(1)
            val minObjective = (st.engineState.preferredStrategyDays ?: 4).coerceIn(1, cacheDaysCount)
            val pool = activePool(st.computeMode)

            // Cycle automatique "Le Chercheur" toutes les 3 générations pour injecter de nouveaux enseignements abstraits
            var autoCycleOutput: ChercheurCycleOutput? = null
            val injectedBatch = ArrayList<SymbolicGenome>()
            synchronized(chercheurLock) {
                if (pendingChercheurGenomes.isNotEmpty()) {
                    injectedBatch.addAll(pendingChercheurGenomes)
                    pendingChercheurGenomes.clear()
                }
            }
            if (injectedBatch.isEmpty() && stateBefore.generation % 3 == 0 && cache.steps.size >= 4) {
                val nextCycle = synchronized(chercheurLock) { ++chercheurCycles }
                autoCycleOutput = withContext(Dispatchers.Default) {
                    ChercheurEngine.runResearchAndDistill(
                        cache = cache,
                        cycleIndex = nextCycle,
                        generation = (stateBefore.generation + 1).coerceAtLeast(1),
                        seenSignatures = sessionExcludedSignatures
                    )
                }
                injectedBatch.addAll(autoCycleOutput.synthesizedGenomes)
            }

            val genResult = withContext(Dispatchers.Default) {
                PcDiscoveryEngine.executeEvolutionGeneration(
                    state = stateBefore,
                    cache = cache,
                    periodName = period.name,
                    minObjectiveStreak = minObjective,
                    populationSize = st.populationSize,
                    computeMode = st.computeMode,
                    cpuPool = pool,
                    gpuAccelerator = gpuAccelerator,
                    injectedGenomes = injectedBatch
                )
            }
            if (autoCycleOutput != null || genResult.injectedOutcomes.isNotEmpty()) {
                recordChercheurOutcomes(autoCycleOutput, genResult.injectedOutcomes)
            }

            activeEvolutionState = genResult.nextState
            lastEvalPerSec = genResult.evaluationsPerSec
            lastGpuMs = genResult.gpuBatchMs
            lastCpuMs = genResult.cpuBatchMs
            val gpuErr = gpuAccelerator.lastRuntimeError
            if (gpuErr.isNotBlank() && gpuErr != lastLoggedGpuError) {
                lastLoggedGpuError = gpuErr
                appendLog("Info GPU OpenCL : $gpuErr (calcul assuré sur cœurs CPU actifs)")
            }

            if (genResult.persistedRecords.isNotEmpty() &&
                (genResult.winningEval != null || genResult.nextState.generation % 10 == 0)
            ) {
                upsertFormulasCached(genResult.persistedRecords)
            }

            val topTrain = genResult.bestTrainEval
            if (topTrain != null && (bestCandidateEvalOverall == null || topTrain.fitnessScore > bestCandidateEvalOverall!!.fitnessScore)) {
                bestCandidateEvalOverall = topTrain
                if (genResult.winningEval != null || genResult.nextState.generation % 10 == 0) {
                    val cpRec = topTrain.toRecord(period.name, System.currentTimeMillis()).copy(id = "checkpoint_best_${period.name}")
                    upsertFormulasCached(listOf(cpRec))
                }
            }

            // Enregistrer & améliorer paliers 1J, 2J, 3J, 4J+ dès qu'ils sont atteints hors-fuite
            val bestGenStreak = maxOf(
                topTrain?.discStats?.maxConsecutiveQualifyingDays ?: 0,
                topTrain?.rollStats?.maxConsecutiveQualifyingDays ?: 0,
                topTrain?.consolidatedTrainStreak ?: 0,
                genResult.auditedEval?.confStats?.maxConsecutiveQualifyingDays ?: 0,
                genResult.auditedEval?.consolidatedFullStreak ?: 0
            )
            val milestonePool = buildList {
                genResult.winningEval?.let { add(it) }
                topTrain?.let { add(it) }
                addAll(genResult.nextState.bestGenomes.take(6))
            }.distinctBy { it.genome.describeCompact() }
            var milestoneUnlockedNow = false
            for (cand in milestonePool) {
                if (!cand.rollingStabilityPassed) continue
                if (cand.rollStats.avgMatches < BacktestResult.RANDOM_EXPECTED_MATCHES) continue
                val candStreak = maxOf(
                    cand.discStats.maxConsecutiveQualifyingDays,
                    cand.rollStats.maxConsecutiveQualifyingDays,
                    cand.consolidatedTrainStreak,
                    cand.confStats?.maxConsecutiveQualifyingDays ?: 0,
                    cand.consolidatedFullStreak
                )
                val trainTwoPlusTotal = cand.discStats.twoPlusCount + cand.rollStats.twoPlusCount
                val oneDayMilestoneReady = cacheDaysCount == 1 &&
                    (cand.consolidatedTrainStreak >= 1 || trainTwoPlusTotal >= 2)
                val tier = when {
                    candStreak >= 2 -> candStreak
                    candStreak == 1 || oneDayMilestoneReady -> 1
                    else -> 0
                }
                if (tier < 1) continue
                val existingTier = unlockedMilestones[tier]
                val shouldSaveTier = existingTier == null || cand.fitnessScore > existingTier.fitnessScore + 0.25
                if (!shouldSaveTier) continue

                val audited = PcDiscoveryEngine.auditCandidateReadOnly(cand, cache, genResult.nextState.confirmationTestsCount)
                val sig = audited.genome.describeCompact()
                val rec = audited.toRecord(period.name, System.currentTimeMillis())
                    .copy(
                        combinedStreak = maxOf(audited.toSummary().combinedStreak, tier),
                        id = "strat_${tier}j_${sig.hashCode().toUInt()}"
                    )
                upsertFormulasCached(listOf(rec))
                unlockedMilestones[tier] = audited
                milestoneUnlockedNow = true
                if (activeWinnerGenome == null || tier > (activeWinnerRecord?.combinedStreak ?: 0) ||
                    (tier == (activeWinnerRecord?.combinedStreak ?: 0) && audited.fitnessScore > (activeWinnerRecord?.fitnessScore ?: Double.NEGATIVE_INFINITY) + 0.25)
                ) {
                    activeWinnerGenome = audited.genome
                    activeWinnerRecord = rec
                }
                appendLog("★ Palier ${tier}J atteint & sauvegardé (0-boule ${(audited.rollStats.zeroMatchRate * 100).toInt()}%, ≥3:${audited.rollStats.threePlusCount}) : $sig")
            }

            if (genResult.winningEval != null) {
                val winner = genResult.winningEval
                winningCandidateEval = winner
                val rec = winner.toRecord(period.name, System.currentTimeMillis())
                upsertFormulasCached(listOf(rec))
                activeWinnerGenome = winner.genome
                activeWinnerRecord = rec
                appendLog("✔ GAGNANT VALIDÉ 4 ÉTAPES (${winner.toSummary().combinedStreak}J) : ${winner.genome.describeCompact()}")
            } else {
                appendLog(
                    "${genResult.generationNote} | ${String.format(Locale.FRANCE, "%.0f", genResult.evaluationsPerSec)} éval/s" +
                        if (genResult.gpuBatchMs > 0.0) " (GPU ${String.format(Locale.FRANCE, "%.1f", genResult.gpuBatchMs)}ms)" else ""
                )
            }

            val refEval = genResult.winningEval ?: genResult.auditedEval ?: topTrain ?: bestCandidateEvalOverall
            val fourMetrics = refEval?.let {
                FourStageValidationMetrics(
                    discoveryDraws = it.discStats.tested,
                    discoveryStreak = it.discStats.maxConsecutiveQualifyingDays,
                    discoveryQualifyingDays = it.discStats.qualifyingDays,
                    discoveryTwoPlusCount = it.discStats.twoPlusCount,
                    discoveryAvgMatches = it.discStats.avgMatches,
                    rollingDraws = it.rollStats.tested,
                    rollingStreak = it.rollStats.maxConsecutiveQualifyingDays,
                    rollingQualifyingDays = it.rollStats.qualifyingDays,
                    rollingTwoPlusCount = it.rollStats.twoPlusCount,
                    rollingAvgMatches = it.rollStats.avgMatches,
                    rollingStabilityPassed = it.rollingStabilityPassed,
                    confirmationDraws = it.confStats?.tested ?: cache.confirmationSteps.size,
                    confirmationStreak = it.confStats?.maxConsecutiveQualifyingDays ?: 0,
                    confirmationQualifyingDays = it.confStats?.qualifyingDays ?: 0,
                    confirmationTwoPlusCount = it.confStats?.twoPlusCount ?: 0,
                    confirmationAvgMatches = it.confStats?.avgMatches ?: 0f,
                    sealedDraws = it.sealedStats?.tested ?: cache.sealedSteps.size,
                    sealedStreak = it.sealedStats?.maxConsecutiveQualifyingDays ?: 0,
                    sealedQualifyingDays = it.sealedStats?.qualifyingDays ?: 0,
                    sealedTwoPlusCount = it.sealedStats?.twoPlusCount ?: 0,
                    sealedAvgMatches = it.sealedStats?.avgMatches ?: 0f,
                    sealedCausalVerified = it.sealedStats?.causalVerified ?: false,
                        combinedStreak = maxOf(
                            it.discStats.maxConsecutiveQualifyingDays,
                            it.rollStats.maxConsecutiveQualifyingDays,
                            it.confStats?.maxConsecutiveQualifyingDays ?: 0,
                            it.consolidatedTrainStreak,
                            it.consolidatedFullStreak
                        ),
                        combinedQualifyingDays = maxOf(it.discStats.qualifyingDays, it.rollStats.qualifyingDays, it.consolidatedTrainStreak, it.consolidatedFullStreak),
                        combinedMaxZeroStreak = maxOf(it.discStats.maxZeroStreak, it.rollStats.maxZeroStreak),
                        combinedBrierScore = it.discStats.brierScore,
                        isFourStageValidated = it.status == "VALIDATED",
                        rollingZeroMatchRate = it.rollStats.zeroMatchRate,
                        rollingThreePlusCount = it.rollStats.threePlusCount,
                        rollingFourPlusCount = it.rollStats.fourPlusCount,
                        rollingFiveMatchCount = it.rollStats.fiveMatchCount,
                        rollingTwoPlusRate = it.rollStats.twoPlusRate
                    )
                }

            val nowEmitMs = System.currentTimeMillis()
            val infiniteActive = _uiState.value.engineState.infiniteMode
            val shouldStopAfterWin = genResult.winningEval != null && !infiniteActive
            val mustEmitUi = genResult.winningEval != null ||
                milestoneUnlockedNow ||
                genResult.nextState.activeParadigmIndex != stateBefore.activeParadigmIndex ||
                genResult.nextState.generation <= 2 ||
                (nowEmitMs - lastUiEmitMs) >= 140L

            if (mustEmitUi) {
                lastUiEmitMs = nowEmitMs
                val allSaved = getCachedFormulas(period.name)
                val winners = extractSavedWinnerSummaries(allSaved)
                val bestValStreak = winners.maxOfOrNull { it.combinedStreak } ?: 0
                val finalPhase = if (shouldStopAfterWin) AutonomousPhase.VALIDATED else AutonomousPhase.ANALYZING
                val activeParadigmName = SymbolicFormula.THEOREM_PARADIGMS[
                    genResult.nextState.activeParadigmIndex.coerceIn(0, SymbolicFormula.THEOREM_PARADIGMS.lastIndex)
                ]

                _uiState.update { cur ->
                    cur.copy(
                        statusBanner = genResult.generationNote,
                        engineState = cur.engineState.copy(
                            phase = finalPhase,
                            generation = genResult.nextState.generation,
                            experimentsCount = genResult.nextState.experimentsCount,
                            confirmationTestsCount = genResult.nextState.confirmationTestsCount,
                            targetObjectiveStreak = minObjective,
                            currentPrimaryExpr = refEval?.genome?.describePrimary().orEmpty(),
                            currentSecondaryExpr = refEval?.genome?.describeSecondary().orEmpty(),
                            currentSlotSplit = refEval?.genome?.let { "${it.primarySlots}+${5 - it.primarySlots}" }.orEmpty(),
                            bestDiscoveryStreak = maxOf(cur.engineState.bestDiscoveryStreak, refEval?.discStats?.maxConsecutiveQualifyingDays ?: 0),
                            bestRollingStreak = maxOf(cur.engineState.bestRollingStreak, refEval?.rollStats?.maxConsecutiveQualifyingDays ?: 0),
                            bestConfirmationStreak = maxOf(cur.engineState.bestConfirmationStreak, refEval?.confStats?.maxConsecutiveQualifyingDays ?: 0),
                            bestSealedStreak = maxOf(cur.engineState.bestSealedStreak, refEval?.sealedStats?.maxConsecutiveQualifyingDays ?: 0),
                            bestCombinedStreak = maxOf(cur.engineState.bestCombinedStreak, bestGenStreak, bestValStreak),
                            bestQualifyingDays = maxOf(cur.engineState.bestQualifyingDays, refEval?.rollStats?.qualifyingDays ?: 0, bestGenStreak),
                            bestTwoPlusCount = maxOf(cur.engineState.bestTwoPlusCount, refEval?.rollStats?.twoPlusCount ?: 0),
                            bestMaxZeroStreak = refEval?.rollStats?.maxZeroStreak ?: cur.engineState.bestMaxZeroStreak,
                            discoveredFeaturesHistogram = genResult.featureHistogram,
                            bestFormulas = genResult.nextState.bestGenomes.map { it.toSummary() },
                            rejectedFormulas = genResult.nextState.rejectedHistory,
                            liveLogs = liveLogsBuffer.toList(),
                            fourStageMetrics = fourMetrics,
                            randomReference = genResult.randomComparison,
                            honestStatusNote = genResult.generationNote,
                            bestValidatedStreak = bestValStreak,
                            activeWinnerId = activeWinnerRecord?.id ?: winners.firstOrNull()?.id.orEmpty(),
                            savedWinners = winners,
                            bestZeroMatchRate = refEval?.rollStats?.zeroMatchRate ?: cur.engineState.bestZeroMatchRate,
                            bestThreePlusCount = maxOf(cur.engineState.bestThreePlusCount, refEval?.rollStats?.threePlusCount ?: 0),
                            bestFourPlusCount = maxOf(cur.engineState.bestFourPlusCount, refEval?.rollStats?.fourPlusCount ?: 0),
                            bestFiveMatchCount = maxOf(cur.engineState.bestFiveMatchCount, refEval?.rollStats?.fiveMatchCount ?: 0),
                            excludedMethodsCount = genResult.nextState.excludedSignaturesCount,
                            stagnationGenerations = genResult.nextState.stagnationGenerations,
                            activeParadigmName = activeParadigmName
                        )
                    )
                }
            }

            if (genResult.winningEval != null) {
                launchValidatedBacktestAndPrediction()
                if (!infiniteActive) break
            }
            yield()
        }
    }

    fun launchValidatedBacktestAndPrediction() {
        val st = _uiState.value
        val draws = st.allDraws
        if (draws.size < 4) {
            _uiState.update { it.copy(statusBanner = "Pas assez de tirages pour lancer le backtest (min. 4).") }
            return
        }

        val targetGenome = activeWinnerGenome
            ?: st.engineState.activeStrategySummary?.let { summary ->
                getCachedFormulas(st.selectedPeriod.name).firstOrNull { it.id == summary.id }
                    ?.let { runCatching { SymbolicFormula.genomeFromJson(it.genomeJson) }.getOrNull() }
            }
            ?: winningCandidateEval?.genome
            ?: bestCandidateEvalOverall?.genome

        if (targetGenome == null) {
            _uiState.update { it.copy(statusBanner = "Aucune stratégie débloquée pour le backtest (atteignez un palier ou importez une sauvegarde).") }
            return
        }

        backtestJob?.cancel()
        backtestJob = scope.launch {
            val pool = activePool(_uiState.value.computeMode)
            val cache = getOrBuildCausalCache(draws, st.selectedPeriod, st.customPeriodDays, st.monteCarloSims, pool)
            val strategySig = targetGenome.describeCompact()
            val ledgerKey = PcDiscoveryEngine.buildLedgerKey(strategySig, cache.periodLabel)
            val existingLedger = repository.loadApprovedBacktestLedger(ledgerKey)

            val (backtest, updatedLedger) = withContext(Dispatchers.Default) {
                PcDiscoveryEngine.mergeAppendOnlyApprovedBacktest(
                    genome = targetGenome,
                    cache = cache,
                    existingLedger = existingLedger,
                    generationsExecuted = maxOf(st.engineState.generation, targetGenome.generation),
                    experimentsCount = maxOf(st.engineState.experimentsCount, 1)
                )
            }
            val nowMs = clockProvider()
            val nextDraw = LotoDateUtils.resolveNextUpcomingDraw(draws, nowMs)
            val pred = withContext(Dispatchers.Default) {
                PcDiscoveryEngine.predictNextWithGenome(
                    genome = targetGenome,
                    cleanHistory = draws,
                    targetHoraire = nextDraw.horaire,
                    validatedBacktest = backtest,
                    targetTimestamp = nextDraw.targetTimestamp
                )
            }

            val finalLedger = PcDiscoveryEngine.recordLivePredictionInLedger(updatedLedger, pred, nextDraw.targetTimestamp)
            repository.saveApprovedBacktestLedger(finalLedger)

            val approvedNote = if (backtest.approvedStepsPreservedCount > 0) {
                "Backtest approuvé verrouillé : ${backtest.approvedStepsPreservedCount} étapes intactes + ${backtest.newlyAppendedStepsCount} nouveau(x) tirage(s) complété(s) (${backtest.livePredictionsMatchedCount} Live vérifié(s))."
            } else {
                "Backtest approuvé verrouillé (${backtest.totalTested} étapes en mode append-only ; les anciens résultats restent intacts)."
            }

            val updatedEngine = _uiState.value.engineState.copy(
                backtestTriggered = true,
                currentPrimaryExpr = targetGenome.describePrimary(),
                currentSecondaryExpr = targetGenome.describeSecondary(),
                currentSlotSplit = "${targetGenome.primarySlots}+${5 - targetGenome.primarySlots}",
                approvedBacktestNote = approvedNote
            )
            val reportText = BacktestExportHelper.formatDetailedBacktestReport(backtest, updatedEngine, pred, nextDraw)
            appendLog("$approvedNote | Prochain tirage ${nextDraw.horaire} : ${pred.predictedDraw.joinToString("-")}")

            _uiState.update { cur ->
                cur.copy(
                    nextDrawTarget = nextDraw,
                    backtestResult = backtest,
                    prediction = pred,
                    detailedReportText = reportText,
                    statusBanner = "Prédiction ${nextDraw.horaire} prête : ${pred.predictedDraw.joinToString(" - ") { it.toString().padStart(2, '0') }}",
                    engineState = updatedEngine.copy(liveLogs = liveLogsBuffer.toList())
                )
            }
        }
    }

    fun runScientificComparison() {
        val st = _uiState.value
        if (st.isComparing) return
        if (st.allDraws.size < 8) {
            _uiState.update { it.copy(statusBanner = "Au moins 8 tirages sont requis pour l'évaluation scientifique (actuellement ${st.allDraws.size}).") }
            return
        }
        scope.launch {
            val pool = activePool(_uiState.value.computeMode)
            val periodDesc = st.customPeriodDays?.let { "$it jours" } ?: st.selectedPeriod.label
            _uiState.update {
                it.copy(
                    isComparing = true,
                    statusBanner = "Audit modèles IA & 13 méthodes classiques ($periodDesc, ${pool.parallelism} cœurs CPU)..."
                )
            }
            appendLog("Lancement Labo Scientifique ($periodDesc, ${st.monteCarloSims} sim. MC, ${pool.parallelism} cœurs)...")
            publishLogs()

            val periodsToCheck = (listOf(st.selectedPeriod) + BacktestPeriod.entries).distinct()
            val seenSigs = HashSet<String>()
            val savedModels = ArrayList<LabSavedModelInput>()
            for (p in periodsToCheck) {
                for (rec in getCachedFormulas(p.name)) {
                    if (rec.id.startsWith("checkpoint_best_")) continue
                    if (rec.status != "VALIDATED" && !rec.status.startsWith("MILESTONE_") && rec.combinedStreak < 1) continue
                    val g = runCatching { SymbolicFormula.genomeFromJson(rec.genomeJson) }.getOrNull() ?: continue
                    val sig = g.describeCompact()
                    if (!seenSigs.add(sig)) continue
                    savedModels.add(
                        LabSavedModelInput(
                            id = rec.id,
                            displayName = "Modèle ${rec.combinedStreak}J (Gén.${rec.generation})",
                            savedStreak = rec.combinedStreak,
                            slotSplitLabel = rec.slotSplitLabel,
                            genome = g
                        )
                    )
                }
            }
            activeWinnerGenome?.let { g ->
                val sig = g.describeCompact()
                if (seenSigs.add(sig)) {
                    val rec = activeWinnerRecord
                    savedModels.add(
                        LabSavedModelInput(
                            id = rec?.id ?: g.id,
                            displayName = "Modèle actif ${rec?.combinedStreak ?: 1}J (Gén.${g.generation})",
                            savedStreak = rec?.combinedStreak ?: 1,
                            slotSplitLabel = "${g.primarySlots}+${5 - g.primarySlots}",
                            genome = g
                        )
                    )
                }
            }
            val topSaved = savedModels.sortedByDescending { it.savedStreak }.take(12)

            val report = withContext(Dispatchers.Default) {
                ClassicalAndScientificLab.runFullScientificComparison(
                    data = st.allDraws,
                    period = st.selectedPeriod,
                    monteCarloSims = st.monteCarloSims,
                    pool = pool,
                    customDays = st.customPeriodDays,
                    savedModels = topSaved
                )
            }

            val verdictBanner = if (report.anyObjectiveValidated) {
                "Labo OK : Objectif VALIDÉ (${report.savedModelAudits.count { it.objectiveValidated }} modèle(s) IA)"
            } else {
                "Labo : Objectif NON validé sur $periodDesc (${topSaved.size} modèle(s) IA vérifiés)"
            }
            appendLog("Labo Scientifique terminé — $verdictBanner")
            _uiState.update { cur ->
                cur.copy(
                    isComparing = false,
                    scientificReport = report,
                    selectedComparisonMethod = report.recommendedMethod,
                    statusBanner = verdictBanner,
                    engineState = cur.engineState.copy(liveLogs = liveLogsBuffer.toList())
                )
            }
        }
    }

    fun runClassicalMethodBacktestAndPrediction(method: PredictionMethod) {
        val st = _uiState.value
        if (st.allDraws.size < 8) {
            _uiState.update { it.copy(statusBanner = "Au moins 8 tirages sont requis (actuellement ${st.allDraws.size}).") }
            return
        }
        scope.launch {
            val nowMs = clockProvider()
            val nextDraw = LotoDateUtils.resolveNextUpcomingDraw(st.allDraws, nowMs)
            val (backtest, pred) = withContext(Dispatchers.Default) {
                ClassicalAndScientificLab.evaluateAndPredictClassicalMethod(
                    method = method,
                    draws = st.allDraws,
                    period = st.selectedPeriod,
                    nextTarget = nextDraw,
                    referenceNowMs = nowMs,
                    customDays = st.customPeriodDays
                )
            }
            val reportText = BacktestExportHelper.formatDetailedBacktestReport(backtest, st.engineState, pred, nextDraw)
            appendLog("Backtest & Prédiction [${method.shortLabel}] → ${nextDraw.horaire} : ${pred.predictedDraw.joinToString("-")}")
            _uiState.update { cur ->
                cur.copy(
                    nextDrawTarget = nextDraw,
                    selectedComparisonMethod = method,
                    backtestResult = backtest,
                    prediction = pred,
                    detailedReportText = reportText,
                    statusBanner = "${method.shortLabel} (${nextDraw.horaire}) : ${pred.predictedDraw.joinToString(" - ") { it.toString().padStart(2, '0') }}",
                    engineState = cur.engineState.copy(liveLogs = liveLogsBuffer.toList())
                )
            }
        }
    }

    fun exportStrategyBackupJson(): Pair<String, String> {
        val periodName = _uiState.value.selectedPeriod.name
        val all = getCachedFormulas(periodName)
        val cp = all.firstOrNull { it.id == "checkpoint_best_$periodName" }
        val strategies = all.filter {
            !it.id.startsWith("checkpoint_best_") &&
                (it.status == "VALIDATED" || it.status.startsWith("MILESTONE_") || it.combinedStreak >= 1)
        }.ifEmpty {
            activeWinnerRecord?.let { listOf(it) }.orEmpty()
        }
        val json = StrategyBackupHelper.serializeBackup(strategies, cp, periodName)
        val bestStreak = strategies.maxOfOrNull { it.combinedStreak } ?: cp?.combinedStreak ?: 0
        val fileName = StrategyBackupHelper.buildBackupFileName(bestStreak)
        return json to fileName
    }

    fun copyStrategyBackupToClipboard(): String {
        val (json, _) = exportStrategyBackupJson()
        val parsed = StrategyBackupHelper.parseBackup(json).getOrNull()
        val count = parsed?.strategies?.size ?: 0
        return if (count == 0 && parsed?.checkpointBest == null) {
            "Aucune méthode (palier 2J+) disponible à sauvegarder."
        } else if (DesktopClipboard.copyText(json)) {
            "Sauvegarde JSON ($count méthode(s)) copiée dans le presse-papiers — prête à coller sur Android ou un autre PC !"
        } else {
            "Impossible d'accéder au presse-papiers système."
        }
    }

    fun importStrategyBackupText(rawText: String): String {
        val periodName = _uiState.value.selectedPeriod.name
        val parsed = StrategyBackupHelper.parseBackup(rawText, periodName)
            .getOrElse { return "Import impossible : ${it.message}" }

        val toSave = ArrayList<DiscoveredFormulaRecord>(parsed.strategies.size + 1)
        toSave.addAll(parsed.strategies)
        parsed.checkpointBest?.let { toSave.add(it) }
        upsertFormulasCached(toSave)

        val best = parsed.strategies.maxByOrNull { it.combinedStreak } ?: parsed.checkpointBest
        if (best != null) {
            runCatching { SymbolicFormula.genomeFromJson(best.genomeJson) }.getOrNull()?.let { g ->
                activeWinnerGenome = g
                activeWinnerRecord = best
            }
        }

        val allSaved = getCachedFormulas(periodName)
        val winners = extractSavedWinnerSummaries(allSaved)
        val maxStreak = winners.maxOfOrNull { it.combinedStreak } ?: 0
        val msg = "Import réussi : ${parsed.strategies.size} méthode(s) restaurée(s) (max ${maxStreak}J) !"
        appendLog(msg)
        _uiState.update { cur ->
            cur.copy(
                statusBanner = msg,
                engineState = cur.engineState.copy(
                    savedWinners = winners,
                    bestValidatedStreak = maxOf(cur.engineState.bestValidatedStreak, maxStreak),
                    activeWinnerId = best?.id ?: winners.firstOrNull()?.id.orEmpty(),
                    currentPrimaryExpr = best?.primaryExprText ?: cur.engineState.currentPrimaryExpr,
                    currentSecondaryExpr = best?.secondaryExprText ?: cur.engineState.currentSecondaryExpr,
                    currentSlotSplit = best?.slotSplitLabel ?: cur.engineState.currentSlotSplit,
                    liveLogs = liveLogsBuffer.toList()
                )
            )
        }
        return msg
    }

    fun copyBacktestReportToClipboard(): String {
        val report = _uiState.value.detailedReportText
        if (report.isBlank()) return "Lancez d'abord un backtest pour générer le rapport."
        return if (DesktopClipboard.copyText(report)) {
            "Rapport détaillé de backtest copié dans le presse-papiers."
        } else {
            "Impossible de copier dans le presse-papiers."
        }
    }

    private fun refreshSavedWinnersAndState(phase: AutonomousPhase, banner: String) {
        val periodName = _uiState.value.selectedPeriod.name
        val allSaved = getCachedFormulas(periodName)
        val winners = extractSavedWinnerSummaries(allSaved)
        val maxStreak = winners.maxOfOrNull { it.combinedStreak } ?: 0
        _uiState.update { cur ->
            cur.copy(
                statusBanner = banner,
                engineState = cur.engineState.copy(
                    phase = phase,
                    savedWinners = winners,
                    bestValidatedStreak = maxOf(cur.engineState.bestValidatedStreak, maxStreak),
                    activeWinnerId = activeWinnerRecord?.id ?: winners.firstOrNull()?.id.orEmpty(),
                    honestStatusNote = banner,
                    liveLogs = liveLogsBuffer.toList()
                )
            )
        }
    }

    private fun extractSavedWinnerSummaries(records: List<DiscoveredFormulaRecord>): List<SymbolicFormulaSummary> {
        return records
            .filter {
                !it.id.startsWith("checkpoint_best_") &&
                    (it.status == "VALIDATED" || it.status.startsWith("MILESTONE_") || it.combinedStreak >= 1)
            }
            .distinctBy { it.primaryExprText + "|" + it.secondaryExprText + "|" + it.slotSplitLabel }
            .sortedWith(compareByDescending<DiscoveredFormulaRecord> { it.combinedStreak }.thenByDescending { it.fitnessScore })
            .map { e ->
                SymbolicFormulaSummary(
                    id = e.id,
                    generation = e.generation,
                    primaryExprText = e.primaryExprText,
                    secondaryExprText = e.secondaryExprText,
                    slotSplitLabel = e.slotSplitLabel,
                    usedVariables = e.usedFeaturesCsv.split(",").filter { it.isNotBlank() },
                    status = e.status,
                    statusReason = e.rejectionReason,
                    discoveryStreak = e.discoveryStreak,
                    rollingStreak = e.rollingStreak,
                    confirmationStreak = e.confirmationStreak,
                    sealedStreak = e.sealedStreak,
                    combinedStreak = e.combinedStreak,
                    discoveryAvgMatches = e.discoveryAvgMatches,
                    rollingAvgMatches = e.rollingAvgMatches,
                    confirmationAvgMatches = e.confirmationAvgMatches,
                    pValueAdjusted = e.pValueAdjusted,
                    fitnessScore = e.fitnessScore,
                    isValidated = e.status == "VALIDATED"
                )
            }
    }

    private fun appendLog(line: String) {
        synchronized(liveLogsBuffer) {
            if (liveLogsBuffer.size >= 60) liveLogsBuffer.removeFirst()
            liveLogsBuffer.addLast("[${LotoDateUtils.formatBeninTime(System.currentTimeMillis())}] $line")
        }
    }

    private fun publishLogs() {
        _uiState.update { it.copy(engineState = it.engineState.copy(liveLogs = liveLogsBuffer.toList())) }
    }

    // =========================================================================
    // MOTEUR PRÉCISION 100 % INDÉPENDANT (Mode 1: 1er N° Pos1 • Mode 2: 2 N° Exacts)
    // =========================================================================

    private fun restorePrecisionLedgerIfPresentAsync(
        mode: PrecisionTargetMode,
        genome: PrecisionGenome,
        draws: List<LotoDraw>
    ) {
        val st = _uiState.value
        val periodLabel = st.customPeriodDays?.let { d ->
            BacktestPeriod.entries.firstOrNull { it.days == d }?.label ?: "$d Jours (Perso)"
        } ?: st.selectedPeriod.label
        val ledgerKey = PrecisionTargetEngine.buildLedgerKey(mode, genome.signatureKey(), periodLabel)
        val existing = repository.loadPrecisionLedger(ledgerKey) ?: return
        scope.launch {
            val pool = activePool(st.computeMode)
            val cache = getOrBuildPrecisionCache(draws, st.selectedPeriod, st.customPeriodDays, pool)
            val (bt, updatedLedger) = withContext(Dispatchers.Default) {
                PrecisionTargetEngine.mergeAppendOnlyApprovedBacktest(
                    genome = genome,
                    cache = cache,
                    existingLedger = existing,
                    confirmationTestsCount = maxOf(1, _precisionUiState.value.confirmationTestsCount)
                )
            }
            val nowMs = clockProvider()
            val nextTarget = LotoDateUtils.resolveNextUpcomingDraw(draws, nowMs)
            val livePred = withContext(Dispatchers.Default) {
                PrecisionTargetEngine.predictNextUpcoming(draws, genome, nextTarget)
            }
            val finalLedger = PrecisionTargetEngine.recordLivePredictionInLedger(updatedLedger, livePred)
            repository.savePrecisionLedger(finalLedger)
            val note = "Registre restauré : ${bt.preservedStepsCount} pas intacts + ${bt.appendedStepsCount} complétés (${bt.liveMatchedCount} Live vérifié(s))."
            appendPrecisionLog("$note | Prochain ${livePred.horaire} : ${livePred.predictedNumbers.joinToString("-") { it.toString().padStart(2, '0') }}")
            _precisionUiState.update { cur ->
                if (cur.targetMode != mode) cur else cur.copy(
                    backtestResult = bt,
                    livePrediction = livePred,
                    approvedNote = note,
                    activeFormulaDesc = genome.describeCompact(),
                    honestStatusNote = "Prédiction ${mode.label} (${livePred.horaire}) : ${livePred.predictedNumbers.joinToString("-") { it.toString().padStart(2, '0') }}",
                    liveLogs = precisionLogsBuffer.toList()
                )
            }
        }
    }

    fun setPrecisionTargetMode(mode: PrecisionTargetMode) {
        val cur = _precisionUiState.value
        if (cur.targetMode == mode) return
        precisionJob?.cancel()
        precisionJob = null
        precisionBacktestJob?.cancel()
        precisionBacktestJob = null
        precisionEvoState = null
        precisionWinnerEval = null
        val restored = repository.loadPrecisionBestGenome(mode)
        precisionBestGenome = restored
        appendPrecisionLog("Cible Précision changée : ${mode.label} (${mode.description}).")
        _precisionUiState.value = PrecisionUiState(
            targetMode = mode,
            phase = AutonomousPhase.IDLE,
            infiniteMode = cur.infiniteMode,
            activeFormulaDesc = restored?.describeCompact().orEmpty(),
            honestStatusNote = if (restored != null) {
                "Mode ${mode.label} restauré — cliquez « Backtest & Prédiction » ou « Démarrer »."
            } else {
                "Mode ${mode.label} prêt — indépendant du moteur 5 boules."
            },
            liveLogs = precisionLogsBuffer.toList()
        )
        val draws = _uiState.value.allDraws
        if (restored != null && draws.size >= 4) {
            restorePrecisionLedgerIfPresentAsync(mode, restored, draws)
        }
    }

    fun setPrecisionInfiniteMode(enabled: Boolean) {
        _precisionUiState.update { cur ->
            cur.copy(
                infiniteMode = enabled,
                honestStatusNote = if (enabled) {
                    "Mode infini Précision actif : recherche continue jusqu'à Pause ou Stop."
                } else {
                    cur.honestStatusNote
                }
            )
        }
    }

    fun startPrecisionInfiniteMode() {
        setPrecisionInfiniteMode(true)
        val curPhase = _precisionUiState.value.phase
        when (curPhase) {
            AutonomousPhase.ANALYZING -> Unit
            AutonomousPhase.SUSPENDED -> resumePrecisionDiscovery()
            else -> startPrecisionDiscovery(continueBeyond = precisionEvoState != null)
        }
    }

    fun startPrecisionDiscovery(continueBeyond: Boolean = false) {
        val st = _uiState.value
        if (st.allDraws.size < 8) {
            _precisionUiState.update {
                it.copy(honestStatusNote = "Minimum 8 tirages historiques requis (${st.allDraws.size}/8).")
            }
            return
        }
        val mode = _precisionUiState.value.targetMode
        precisionJob?.cancel()
        precisionJob = scope.launch {
            val pool = activePool(st.computeMode)
            val cache = getOrBuildPrecisionCache(st.allDraws, st.selectedPeriod, st.customPeriodDays, pool)
            if (cache.steps.size < 3) {
                _precisionUiState.update {
                    it.copy(
                        phase = AutonomousPhase.STOPPED,
                        honestStatusNote = "Fenêtre causale trop courte (${cache.steps.size} tirages)."
                    )
                }
                return@launch
            }
            if (!continueBeyond || precisionEvoState == null || precisionEvoState?.mode != mode) {
                val popSize = st.populationSize.coerceIn(64, 4096)
                precisionEvoState = PrecisionTargetEngine.createInitialState(
                    mode = mode,
                    populationSize = popSize,
                    sessionSalt = (System.currentTimeMillis() % 100_000L) + 17L
                )
                precisionWinnerEval = null
                precisionBestGenome = repository.loadPrecisionBestGenome(mode)
                val hwNote = if (st.computeMode == ComputeMode.GPU_AND_CPU_MAX && gpuAccelerator.isPrecisionKernelReady) {
                    "Multi-GPU OpenCL + ${pool.parallelism} cœurs CPU (Pop $popSize)"
                } else {
                    "${pool.parallelism} cœurs CPU (Pop $popSize)"
                }
                appendPrecisionLog("Démarrage Moteur 2 Précision [${mode.label}] sur ${cache.periodLabel} (${cache.steps.size} tirages • $hwNote)...")
            } else {
                appendPrecisionLog("Reprise Moteur 2 Précision [${mode.label}] depuis Gén.${precisionEvoState?.generation ?: 0}...")
            }

            _precisionUiState.update { cur ->
                cur.copy(
                    phase = AutonomousPhase.ANALYZING,
                    honestStatusNote = "Recherche Précision [${mode.label}] en cours (${st.computeMode.label})...",
                    liveLogs = precisionLogsBuffer.toList()
                )
            }
            runPrecisionLoop(cache, pool)
        }
    }

    fun suspendPrecisionDiscovery() {
        precisionJob?.cancel()
        precisionJob = null
        val g = precisionEvoState?.generation ?: 0
        appendPrecisionLog("Moteur Précision mis en pause (Gén.$g — population conservée).")
        _precisionUiState.update { cur ->
            cur.copy(
                phase = AutonomousPhase.SUSPENDED,
                honestStatusNote = "En pause à Gén.$g — cliquez Continuer pour reprendre.",
                liveLogs = precisionLogsBuffer.toList()
            )
        }
    }

    fun resumePrecisionDiscovery() {
        val cur = _precisionUiState.value
        if (cur.phase == AutonomousPhase.SUSPENDED || cur.phase == AutonomousPhase.STOPPED || cur.phase == AutonomousPhase.VALIDATED) {
            startPrecisionDiscovery(continueBeyond = precisionEvoState != null)
        } else {
            startPrecisionDiscovery(continueBeyond = false)
        }
    }

    fun stopPrecisionDiscovery() {
        precisionJob?.cancel()
        precisionJob = null
        appendPrecisionLog("Moteur Précision arrêté (Stop).")
        _precisionUiState.update { cur ->
            cur.copy(
                phase = AutonomousPhase.STOPPED,
                honestStatusNote = "Recherche Précision fermée proprement.",
                liveLogs = precisionLogsBuffer.toList()
            )
        }
    }

    fun restartPrecisionFromZero() {
        precisionJob?.cancel()
        precisionJob = null
        precisionEvoState = null
        precisionWinnerEval = null
        precisionBestGenome = null
        synchronized(precisionLogsBuffer) { precisionLogsBuffer.clear() }
        appendPrecisionLog("Remise à zéro complète du Moteur Précision.")
        _precisionUiState.update { cur ->
            PrecisionUiState(
                targetMode = cur.targetMode,
                phase = AutonomousPhase.IDLE,
                infiniteMode = cur.infiniteMode,
                liveLogs = precisionLogsBuffer.toList()
            )
        }
        startPrecisionDiscovery(continueBeyond = false)
    }

    fun launchPrecisionBacktestAndPrediction() {
        val st = _uiState.value
        val pState = _precisionUiState.value
        val draws = st.allDraws
        if (draws.size < 4) {
            _precisionUiState.update { it.copy(honestStatusNote = "Pas assez de tirages pour le backtest Précision (min. 4).") }
            return
        }
        val mode = pState.targetMode
        val genome = precisionWinnerEval?.genome
            ?: precisionBestGenome
            ?: precisionEvoState?.elites?.firstOrNull()?.genome
            ?: repository.loadPrecisionBestGenome(mode)

        if (genome == null) {
            _precisionUiState.update {
                it.copy(honestStatusNote = "Aucune formule Précision prête — lancez d'abord la recherche [${mode.label}].")
            }
            return
        }

        precisionBacktestJob?.cancel()
        precisionBacktestJob = scope.launch {
            val pool = activePool(st.computeMode)
            val cache = getOrBuildPrecisionCache(draws, st.selectedPeriod, st.customPeriodDays, pool)
            val ledgerKey = PrecisionTargetEngine.buildLedgerKey(mode, genome.signatureKey(), cache.periodLabel)
            val existing = repository.loadPrecisionLedger(ledgerKey)

            val (bt, updatedLedger) = withContext(Dispatchers.Default) {
                PrecisionTargetEngine.mergeAppendOnlyApprovedBacktest(
                    genome = genome,
                    cache = cache,
                    existingLedger = existing,
                    confirmationTestsCount = maxOf(1, _precisionUiState.value.confirmationTestsCount)
                )
            }
            val nowMs = clockProvider()
            val nextTarget = LotoDateUtils.resolveNextUpcomingDraw(draws, nowMs)
            val livePred = withContext(Dispatchers.Default) {
                PrecisionTargetEngine.predictNextUpcoming(
                    history = draws,
                    genome = genome,
                    nextTarget = nextTarget
                )
            }
            val finalLedger = PrecisionTargetEngine.recordLivePredictionInLedger(updatedLedger, livePred)
            repository.savePrecisionLedger(finalLedger)

            val note = if (bt.preservedStepsCount > 0) {
                "Ledger Précision verrouillé : ${bt.preservedStepsCount} pas intacts + ${bt.appendedStepsCount} nouveau(x) tirage(s) (${bt.liveMatchedCount} Live vérifié(s))."
            } else {
                "Ledger Précision verrouillé (${bt.totalTested} pas append-only • zéro fuite)."
            }
            appendPrecisionLog("$note | Prochain ${livePred.horaire} : ${livePred.predictedNumbers.joinToString("-") { it.toString().padStart(2, '0') }}")
            _precisionUiState.update { cur ->
                cur.copy(
                    backtestResult = bt,
                    livePrediction = livePred,
                    approvedNote = note,
                    activeFormulaDesc = genome.describeCompact(),
                    honestStatusNote = "Prédiction ${mode.label} (${livePred.horaire}) prête : ${livePred.predictedNumbers.joinToString("-") { it.toString().padStart(2, '0') }}",
                    liveLogs = precisionLogsBuffer.toList()
                )
            }
        }
    }

    fun formatPrecisionDetailedReport(): String {
        val pst = _precisionUiState.value
        val live = pst.livePrediction
        val bt = pst.backtestResult
        val sb = StringBuilder()
        sb.appendLine("===== LOTO BJ PC — MOTEUR 2 PRÉCISION (${pst.targetMode.label}) =====")
        sb.appendLine("Statut : ${pst.phase.name}${if (pst.infiniteMode) " (INFINI)" else ""}")
        sb.appendLine("Formule : ${pst.activeFormulaDesc.ifBlank { "—" }}")
        if (live != null) {
            val ballsStr = live.predictedNumbers.joinToString(" - ") { it.toString().padStart(2, '0') }
            val confStr = live.confidencesPercent.joinToString(", ") { (n, p) ->
                "${n.toString().padStart(2, '0')}:${String.format(Locale.FRANCE, "%.1f%%", p)}"
            }
            sb.appendLine("Prédiction Live (${live.horaire} • ${live.dayLabel}) : [ $ballsStr ] ($confStr)")
            sb.appendLine(live.causalAuditNote)
        }
        if (bt != null) {
            sb.appendLine("Fenêtre : ${bt.periodLabel} • Tirages : ${bt.totalTested} • Cibles exactes : ${bt.exactTargetCount} • ≥1 bon : ${bt.onePlusCount}")
            sb.appendLine("Blocs qualifiés : ${bt.qualifyingBlocksCount} • Série max : ${bt.maxConsecQualifyingBlocks}J • Brier : ${String.format(Locale.FRANCE, "%.5f", bt.averageBrierScore)} • p_adj : ${String.format(Locale.FRANCE, "%.4f", bt.pValueAdjusted)}")
            sb.appendLine("Ledger append-only : ${bt.preservedStepsCount} conservés + ${bt.appendedStepsCount} ajoutés • ${bt.liveMatchedCount} Live concordants")
            sb.appendLine("\n--- Blocs 6 Tirages / 1J ---")
            bt.dailyBlocks.forEach { b ->
                sb.appendLine("${b.blockLabel} | ${b.drawsTested} tir. | Exact=${b.exactTargetHits} | ≥1=${b.onePlusHits} | ${if (b.isQualifyingBlock) "QUALIFIÉ" else "Non"}")
            }
            sb.appendLine("\n--- Historique Tirage par Tirage (Strictement Causal) ---")
            bt.steps.forEach { s ->
                val predS = s.predictedNumbers.joinToString("-") { it.toString().padStart(2, '0') }
                val actS = s.actualNumbers.joinToString("-") { it.toString().padStart(2, '0') }
                sb.appendLine("#${s.stepIndex} ${s.drawDate} ${s.horaire} | Prédit=[$predS] | Pos1=${s.actualPos1.toString().padStart(2, '0')} [$actS] | ${if (s.exactTargetHit) "EXACT" else "${s.matchCount}"}")
            }
        }
        return sb.toString()
    }

    fun copyPrecisionReportToClipboard(): String {
        val text = formatPrecisionDetailedReport()
        return if (DesktopClipboard.copyText(text)) {
            "Rapport Moteur Précision copié dans le presse-papiers."
        } else {
            "Impossible d'accéder au presse-papiers."
        }
    }

    private suspend fun getOrBuildPrecisionCache(
        draws: List<LotoDraw>,
        period: BacktestPeriod,
        customDays: Int?,
        pool: ForkJoinPool
    ): PrecisionCausalCache = withContext(Dispatchers.Default) {
        val nowMs = clockProvider()
        val dayBucket = LotoDateUtils.formatDayKey(nowMs)
        val expectedKey = "PREC|${period.name}|${customDays ?: "ALL"}|${draws.size}|${draws.firstOrNull()?.uniqueId}|${draws.lastOrNull()?.uniqueId}|$dayBucket"
        if (precisionCache != null && precisionCacheKey == expectedKey) {
            precisionCache!!
        } else {
            PrecisionTargetEngine.buildCausalCache(
                history = draws,
                period = period,
                referenceNowMs = nowMs,
                customDays = customDays,
                pool = pool
            ).also {
                precisionCache = it
                precisionCacheKey = expectedKey
            }
        }
    }

    private suspend fun runPrecisionLoop(cache: PrecisionCausalCache, pool: ForkJoinPool) {
        var lastPrecUiEmitMs = 0L
        var bestPrecFitness = Double.NEGATIVE_INFINITY
        while (currentCoroutineContext().isActive) {
            val pState = _precisionUiState.value
            if (pState.phase != AutonomousPhase.ANALYZING) break
            val stateBefore = precisionEvoState ?: break
            val uiSnap = _uiState.value
            val cacheDaysCount = cache.steps.map { it.dayEpoch }.distinct().size.coerceAtLeast(1)
            val popSize = uiSnap.populationSize.coerceIn(64, 4096)

            val genRes = withContext(Dispatchers.Default) {
                PrecisionTargetEngine.executeEvolutionGeneration(
                    state = stateBefore,
                    cache = cache,
                    minObjectiveStreak = if (cacheDaysCount <= 1) 1 else minOf(3, cacheDaysCount),
                    populationSize = popSize,
                    computeMode = uiSnap.computeMode,
                    cpuPool = pool,
                    gpuAccelerator = gpuAccelerator
                )
            }
            precisionEvoState = genRes.nextState
            lastEvalPerSec = genRes.evaluationsPerSec
            lastGpuMs = genRes.gpuBatchMs
            lastCpuMs = genRes.cpuBatchMs
            if (genRes.gpuErrorNote.isNotBlank() && genRes.gpuErrorNote != lastLoggedGpuError) {
                lastLoggedGpuError = genRes.gpuErrorNote
                appendPrecisionLog("Alerte OpenCL GPU Moteur 2 (repli CPU propre) : ${genRes.gpuErrorNote}")
            }

            val top = genRes.winningEval ?: genRes.auditedEval ?: genRes.bestTrainEval
            if (genRes.winningEval != null) {
                precisionWinnerEval = genRes.winningEval
                precisionBestGenome = genRes.winningEval.genome
                bestPrecFitness = maxOf(bestPrecFitness, genRes.winningEval.fitnessScore)
                repository.savePrecisionBestGenome(genRes.nextState.mode, genRes.winningEval.genome)
                appendPrecisionLog("✔ GAGNANT PRÉCISION VALIDÉ (${genRes.nextState.mode.label}) : ${genRes.winningEval.genome.describeCompact()}")
            } else {
                top?.let {
                    if (precisionBestGenome == null ||
                        it.consolidatedStreak > pState.bestValidatedStreak ||
                        it.fitnessScore > bestPrecFitness + 0.25
                    ) {
                        bestPrecFitness = maxOf(bestPrecFitness, it.fitnessScore)
                        precisionBestGenome = it.genome
                        repository.savePrecisionBestGenome(genRes.nextState.mode, it.genome)
                    }
                }
                appendPrecisionLog("${genRes.generationNote} | ${String.format(Locale.FRANCE, "%.0f", genRes.evaluationsPerSec)} éval/s (GPU ${String.format(Locale.FRANCE, "%.1f", genRes.gpuBatchMs)}ms • CPU ${String.format(Locale.FRANCE, "%.1f", genRes.cpuBatchMs)}ms)")
            }

            val infinite = _precisionUiState.value.infiniteMode
            val nextPhase = if (genRes.winningEval != null && !infinite) AutonomousPhase.VALIDATED else AutonomousPhase.ANALYZING
            val nowMs = System.currentTimeMillis()
            val mustEmitPrecUi = genRes.winningEval != null ||
                genRes.nextState.generation <= 2 ||
                (nowMs - lastPrecUiEmitMs) >= 140L

            if (mustEmitPrecUi) {
                lastPrecUiEmitMs = nowMs
                _precisionUiState.update { cur ->
                    cur.copy(
                        phase = nextPhase,
                        generation = genRes.nextState.generation,
                        experimentsCount = genRes.nextState.experimentsCount,
                        confirmationTestsCount = genRes.nextState.confirmationTestsCount,
                        excludedCount = genRes.nextState.seenSignatures.size,
                        bestValidatedStreak = maxOf(cur.bestValidatedStreak, top?.consolidatedStreak ?: 0),
                        discoveryStats = top?.discStats ?: cur.discoveryStats,
                        rollingStats = top?.rollStats ?: cur.rollingStats,
                        confirmationStats = top?.confStats ?: cur.confirmationStats,
                        sealedStats = top?.sealedStats ?: cur.sealedStats,
                        activeFormulaDesc = top?.genome?.describeCompact() ?: cur.activeFormulaDesc,
                        pValueAdjusted = top?.pValueAdjusted ?: cur.pValueAdjusted,
                        honestStatusNote = genRes.generationNote,
                        evaluationsPerSec = genRes.evaluationsPerSec,
                        liveLogs = precisionLogsBuffer.toList()
                    )
                }
            }
            if (genRes.winningEval != null) {
                _uiState.update { it.copy(statusBanner = "Moteur 2 Précision (${genRes.nextState.mode.label}) validé (${genRes.winningEval.consolidatedStreak}j) !") }
                launchPrecisionBacktestAndPrediction()
                if (!infinite) break
            }
            yield()
        }
    }

    private fun appendPrecisionLog(line: String) {
        synchronized(precisionLogsBuffer) {
            if (precisionLogsBuffer.size >= 48) precisionLogsBuffer.removeFirst()
            precisionLogsBuffer.addLast("[${LotoDateUtils.formatBeninTime(System.currentTimeMillis())}] $line")
        }
    }

    override fun close() {
        evolutionJob?.cancel()
        backtestJob?.cancel()
        precisionJob?.cancel()
        precisionBacktestJob?.cancel()
        telemetryJob?.cancel()
        scope.cancel()
        cpuPoolMax.shutdownNow()
        cpuPoolBalanced.shutdownNow()
        gpuAccelerator.close()
    }
}
