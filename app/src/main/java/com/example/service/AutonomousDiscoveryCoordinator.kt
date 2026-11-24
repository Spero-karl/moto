package com.example.service

import android.app.Application
import com.example.data.local.AppDatabase
import com.example.data.local.DiscoveredFormulaEntity
import com.example.data.local.DiscoverySessionEntity
import com.example.data.local.LotoResultEntity
import com.example.data.repository.DiscoveryRepository
import com.example.data.repository.LotoRepository
import com.example.domain.engine.AutonomousPhase
import com.example.domain.engine.BacktestPeriod
import com.example.domain.engine.BacktestResult
import com.example.domain.engine.ChercheurCycleOutput
import com.example.domain.engine.ChercheurEngine
import com.example.domain.engine.ChercheurStatusSummary
import com.example.domain.engine.ChercheurTeaching
import com.example.domain.engine.DiscoveryEngine
import com.example.domain.engine.EngineActionState
import com.example.domain.engine.FourStageCausalCache
import com.example.domain.engine.FourStageValidationMetrics
import com.example.domain.engine.GenomeCandidateEvaluation
import com.example.domain.engine.LearningEngine
import com.example.domain.engine.NextDrawTarget
import com.example.domain.engine.PredictionMethod
import com.example.domain.engine.PredictionResult
import com.example.domain.engine.PrecisionCandidateEval
import com.example.domain.engine.PrecisionCausalCache
import com.example.domain.engine.PrecisionEvolutionState
import com.example.domain.engine.PrecisionGenome
import com.example.domain.engine.PrecisionTargetEngine
import com.example.domain.engine.PrecisionTargetMode
import com.example.domain.engine.PrecisionUiState
import com.example.domain.engine.RandomReferenceComparison
import com.example.domain.engine.RejectedFormulaSummary
import com.example.domain.engine.SymbolicEvolutionState
import com.example.domain.engine.SymbolicFormula
import com.example.domain.engine.SymbolicFormulaSummary
import com.example.domain.engine.SymbolicGenome
import com.example.domain.util.LotoDateUtils
import com.example.domain.util.StrategyBackupExport
import com.example.domain.util.StrategyBackupHelper
import com.example.domain.util.StrategyBackupImportResult
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlinx.coroutines.yield
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.add
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import java.util.Locale
import java.util.WeakHashMap
import java.util.concurrent.atomic.AtomicLong

class AutonomousDiscoveryCoordinator private constructor(
    private val application: Application
) {
    private val db = AppDatabase.getDatabase(application)
    private val lotoDao = db.lotoDao()
    private val repository = LotoRepository(lotoDao)
    private val discoveryRepository = DiscoveryRepository(db.discoveryDao())
    private val helperEngine = LearningEngine()

    private val coordinatorScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    private val _selectedMethod = MutableStateFlow(PredictionMethod.AUTO_SEARCH)
    val selectedMethod: StateFlow<PredictionMethod> = _selectedMethod.asStateFlow()

    private val _selectedPeriod = MutableStateFlow(BacktestPeriod.THREE_WEEKS)
    val selectedPeriod: StateFlow<BacktestPeriod> = _selectedPeriod.asStateFlow()

    private val _customPeriodDays = MutableStateFlow<Int?>(BacktestPeriod.THREE_WEEKS.days)
    val customPeriodDays: StateFlow<Int?> = _customPeriodDays.asStateFlow()

    private val _nextDrawTarget = MutableStateFlow(LotoDateUtils.resolveNextUpcomingDraw(emptyList()))
    val nextDrawTarget: StateFlow<NextDrawTarget> = _nextDrawTarget.asStateFlow()

    private val _selectedHoraire = MutableStateFlow(_nextDrawTarget.value.horaire)
    val selectedHoraire: StateFlow<String> = _selectedHoraire.asStateFlow()

    private val _availableHoraires = MutableStateFlow(listOf("00H", "8H", "11H", "14H", "18H", "20H"))
    val availableHoraires: StateFlow<List<String>> = _availableHoraires.asStateFlow()

    private val _isBacktestBusy = MutableStateFlow(false)
    val isBacktestBusy: StateFlow<Boolean> = _isBacktestBusy.asStateFlow()

    private val _discoveryState = MutableStateFlow(EngineActionState())
    val discoveryState: StateFlow<EngineActionState> = _discoveryState.asStateFlow()

    private val _prediction = MutableStateFlow<PredictionResult?>(null)
    val prediction: StateFlow<PredictionResult?> = _prediction.asStateFlow()

    private val _backtestResult = MutableStateFlow<BacktestResult?>(null)
    val backtestResult: StateFlow<BacktestResult?> = _backtestResult.asStateFlow()

    private val _precisionUiState = MutableStateFlow(PrecisionUiState())
    val precisionUiState: StateFlow<PrecisionUiState> = _precisionUiState.asStateFlow()

    private val precisionStorageDir: File by lazy { File(application.filesDir, "precision") }
    private var precisionJob: Job? = null
    private var precisionBacktestJob: Job? = null
    private var precisionCausalCacheKey: String = ""
    private var precisionCausalCache: PrecisionCausalCache? = null
    private var precisionEvoState: PrecisionEvolutionState? = null
    private var precisionBestGenome: PrecisionGenome? = null
    private var precisionWinnerEval: PrecisionCandidateEval? = null
    private val precisionLogs = ArrayDeque<String>(36)

    private var discoveryJob: Job? = null
    private var backtestJob: Job? = null
    private var restoreJob: Job? = null
    @Volatile
    private var lastObservedDatasetSig: String = ""
    private val stateLock = Any()
    private val actionEpoch = AtomicLong(0L)
    private var sessionSalt: Long = 1L
    private val accumulatedLogs = ArrayDeque<String>(MAX_LOG_ENTRIES)

    private var evolutionState: SymbolicEvolutionState = DiscoveryEngine.createInitialState(sessionSalt)
    private var cachedCausalCache: FourStageCausalCache? = null
    private var cachedCacheKey: String? = null
    private var cachedChronologicalHistory: List<LotoResultEntity> = emptyList()
    private var latestHistorySnapshot: List<LotoResultEntity> = emptyList()
    private var validatedGenome: SymbolicGenome? = null
    private var activeWinnerId: String = ""
    private var bestValidatedStreak: Int = 0
    private var bestValidatedFitness: Double = Double.NEGATIVE_INFINITY
    private var savedWinnersCache: List<SymbolicFormulaSummary> = emptyList()
    private var targetObjectiveStreak: Int = 4
    private var preferredStrategyDays: Int? = null
    private var continueBeyondMode: Boolean = false
    private var userPinnedWinner: Boolean = false
    private var latestAuditedEval: GenomeCandidateEvaluation? = null
    private var latestRandomRef: RandomReferenceComparison? = null
    private val savedMilestoneSignatures = HashSet<String>()

    private var bestMilestoneGenome: SymbolicGenome? = null
    private var bestMilestoneEval: GenomeCandidateEvaluation? = null
    private var bestMilestoneCombinedStreak: Int = 0
    private var bestMilestoneDiscStreak: Int = 0
    private var bestMilestoneRollStreak: Int = 0
    private var bestMilestoneConfStreak: Int = 0
    private var bestMilestoneFitness: Double = Double.NEGATIVE_INFINITY

    // État du moteur "Le Chercheur" et mode infini / ledger immuable
    private val chercheurLock = Any()
    private val pendingChercheurGenomes = mutableListOf<SymbolicGenome>()
    private val genomeToTeachingId = mutableMapOf<String, String>()
    private val activeTeachings = mutableListOf<ChercheurTeaching>()
    private var chercheurCycles: Int = 0
    private var totalProducedTeachings: Int = 0
    private var totalInjectedStrategies: Int = 0
    private var totalKeptStrategies: Int = 0
    private var totalDiscardedStrategies: Int = 0
    @Volatile
    private var infiniteModeFlag: Boolean = false
    @Volatile
    private var approvedBacktestNote: String = ""

    @Volatile
    internal var testInjectedErrorOnce: (() -> Unit)? = null

    @Volatile
    internal var clockProvider: () -> Long = { System.currentTimeMillis() }

    init {
        coordinatorScope.launch {
            repository.allResults.collect { results ->
                if (results.isNotEmpty()) {
                    val newest = results.maxByOrNull { it.timestamp }
                    val newSig = "${results.size}|${results.firstOrNull()?.uniqueId}|${newest?.uniqueId}|${newest?.timestamp}"
                    val changed = lastObservedDatasetSig.isNotEmpty() && newSig != lastObservedDatasetSig
                    lastObservedDatasetSig = newSig
                    latestHistorySnapshot = results
                    if (changed) {
                        cachedCausalCache = null
                        cachedCacheKey = null
                        precisionCausalCache = null
                        precisionCausalCacheKey = ""
                    }
                    val avail = helperEngine.availableHoraires(results)
                        .ifEmpty { _availableHoraires.value }
                    _availableHoraires.value = avail
                    refreshNextDrawTarget(results)
                    if (changed) {
                        onNewDrawsArrivedAutoRefresh()
                    }
                }
            }
        }
        restoreJob = coordinatorScope.launch {
            restoreInitialSession(actionEpoch.get())
        }
    }

    internal fun injectHistoryForTest(results: List<LotoResultEntity>, nowMs: Long = clockProvider()) {
        val newest = results.maxByOrNull { it.timestamp }
        val newSig = "${results.size}|${results.firstOrNull()?.uniqueId}|${newest?.uniqueId}|${newest?.timestamp}"
        val changed = lastObservedDatasetSig.isNotEmpty() && newSig != lastObservedDatasetSig
        lastObservedDatasetSig = newSig
        latestHistorySnapshot = results
        if (changed) {
            cachedCausalCache = null
            cachedCacheKey = null
        }
        if (results.isNotEmpty()) {
            val avail = helperEngine.availableHoraires(results)
                .ifEmpty { _availableHoraires.value }
            _availableHoraires.value = avail
        }
        refreshNextDrawTarget(results, nowMs)
        if (changed) {
            onNewDrawsArrivedAutoRefresh()
        }
    }

    fun refreshNextDrawTarget(
        history: List<LotoResultEntity> = latestHistorySnapshot,
        nowMs: Long = clockProvider()
    ): NextDrawTarget {
        val resolved = LotoDateUtils.resolveNextUpcomingDraw(history, nowMs)
        _nextDrawTarget.value = resolved
        _selectedHoraire.value = resolved.horaire
        return resolved
    }

    fun selectFocusAxis(method: PredictionMethod) {
        _selectedMethod.value = method
    }

    fun selectPeriod(period: BacktestPeriod) {
        if (_selectedPeriod.value == period && _customPeriodDays.value == period.days) return
        applyPeriodSelection(period = period, customDays = period.days)
    }

    fun selectCustomPeriodDays(days: Int) {
        val clamped = days.coerceIn(1, 3650)
        if (_customPeriodDays.value == clamped) return
        val mapped = BacktestPeriod.entries.firstOrNull { it.days == clamped } ?: BacktestPeriod.ONE_DAY
        applyPeriodSelection(period = mapped, customDays = clamped)
    }

    private fun applyPeriodSelection(period: BacktestPeriod, customDays: Int?) {
        val epoch = actionEpoch.incrementAndGet()
        restoreJob?.cancel()
        val prevPeriod = _selectedPeriod.value
        val prevState = _discoveryState.value
        val wasAnalyzing = prevState.phase == AutonomousPhase.ANALYZING
        if (wasAnalyzing) {
            val saved = prevState.copy(phase = AutonomousPhase.SUSPENDED)
            persistSnapshotAsync(saved, prevPeriod)
        }
        discoveryJob?.cancel()
        discoveryJob = null
        DiscoveryForegroundService.stopService(application)

        _selectedPeriod.value = period
        _customPeriodDays.value = customDays
        cachedCausalCache = null
        cachedCacheKey = null
        precisionCausalCache = null
        precisionCausalCacheKey = ""
        validatedGenome = null
        activeWinnerId = ""
        bestValidatedStreak = 0
        bestValidatedFitness = Double.NEGATIVE_INFINITY
        savedWinnersCache = emptyList()
        targetObjectiveStreak = 4
        preferredStrategyDays = null
        continueBeyondMode = false
        userPinnedWinner = false
        latestAuditedEval = null
        latestRandomRef = null
        savedMilestoneSignatures.clear()
        resetMilestoneTracking()
        _prediction.value = null
        _backtestResult.value = null
        synchronized(stateLock) { accumulatedLogs.clear() }
        evolutionState = DiscoveryEngine.createInitialState(sessionSalt)
        val rangeLabel = customDays?.let { "$it jours glissants jusqu'au jour J" }
            ?: "Historique complet jusqu'au jour J"
        appendLog("Période de travail configurée : ${customDays?.let { "$it jours" } ?: "Tout l'historique"}.")
        _discoveryState.value = EngineActionState(
            phase = AutonomousPhase.IDLE,
            periodDateRangeLabel = rangeLabel,
            liveLogs = snapshotLogs(),
            honestStatusNote = "Fenêtre configurée sur ${customDays?.let { "$it jours" } ?: "Tout"} — prête pour l'analyse."
        )

        if (period != prevPeriod || customDays == period.days) {
            restoreJob = coordinatorScope.launch {
                restorePersistedSession(period, autoResumeIfAnalyzing = wasAnalyzing, expectedEpoch = epoch)
            }
        } else if (wasAnalyzing) {
            startContinuousDiscovery(reset = false)
        }
    }

    fun selectHoraire(@Suppress("UNUSED_PARAMETER") horaire: String) {
        // Verrouille sur le prochain tirage uniquement selon l'heure Bénin en cours
        val resolvedTarget = refreshNextDrawTarget()
        val genome = validatedGenome
        val currentBt = _backtestResult.value
        val cleanHistory = cachedChronologicalHistory
        if (_discoveryState.value.canLaunchBacktest &&
            _discoveryState.value.backtestTriggered &&
            genome != null &&
            currentBt != null &&
            cleanHistory.isNotEmpty()
        ) {
            coordinatorScope.launch {
                val updatedPred = DiscoveryEngine.predictNextWithGenome(
                    genome = genome,
                    cleanHistory = cleanHistory,
                    targetHoraire = resolvedTarget.horaire,
                    validatedBacktest = currentBt,
                    targetTimestamp = resolvedTarget.targetTimestamp
                )
                _prediction.value = updatedPred
            }
        }
    }

    fun suspendDiscovery() {
        actionEpoch.incrementAndGet()
        restoreJob?.cancel()
        discoveryJob?.cancel()
        discoveryJob = null
        val current = _discoveryState.value
        if (current.phase == AutonomousPhase.ANALYZING) {
            appendLog("Analyse mise en pause à Gén.${evolutionState.generation} (${evolutionState.experimentsCount} formules) — mémoire sauvegardée.")
            val suspendedState = current.copy(
                phase = AutonomousPhase.SUSPENDED,
                liveLogs = snapshotLogs(),
                honestStatusNote = "En pause : mémoire génétique conservée intacte. Appuyez sur « Continuer » pour reprendre ou « Stop » pour fermer.",
                infiniteMode = infiniteModeFlag,
                approvedBacktestNote = approvedBacktestNote,
                chercheurStatus = currentChercheurStatus(isRunning = false)
            )
            _discoveryState.value = suspendedState
            persistSnapshotAsync(suspendedState, _selectedPeriod.value)
        }
        DiscoveryForegroundService.stopService(application)
    }

    fun resumeDiscovery() {
        startContinuousDiscovery(reset = false)
    }

    fun continueBeyondObjective() {
        continueBeyondMode = true
        infiniteModeFlag = true
        val baseline = maxOf(4, bestValidatedStreak, _discoveryState.value.bestValidatedStreak)
        targetObjectiveStreak = baseline + 1
        appendLog("Poursuite jusqu'à l'infini activée : recherche d'un gagnant ≥${targetObjectiveStreak}j (sans triche)...")
        startContinuousDiscovery(reset = false)
    }

    fun setInfiniteMode(enabled: Boolean) {
        infiniteModeFlag = enabled
        if (enabled) continueBeyondMode = true
        _discoveryState.update { cur ->
            cur.copy(
                infiniteMode = enabled,
                honestStatusNote = if (enabled) {
                    "Mode infini actif : l'analyse cherche en continu jusqu'à Pause ou Stop."
                } else {
                    cur.honestStatusNote
                }
            )
        }
    }

    fun startInfiniteMode() {
        setInfiniteMode(true)
        val curPhase = _discoveryState.value.phase
        when {
            curPhase == AutonomousPhase.ANALYZING -> Unit
            curPhase == AutonomousPhase.SUSPENDED -> resumeDiscovery()
            else -> startContinuousDiscovery(reset = evolutionState.generation == 0)
        }
    }

    fun restartDiscoveryFromZero() {
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
        startContinuousDiscovery(reset = true)
    }

    fun runChercheurAndInject() {
        val currentData = when {
            cachedChronologicalHistory.isNotEmpty() -> cachedChronologicalHistory
            latestHistorySnapshot.isNotEmpty() -> latestHistorySnapshot
            else -> emptyList()
        }
        if (currentData.size in 1..7) {
            appendLog("Le Chercheur nécessite au moins 8 tirages historiques.")
            publishChercheurStatus(isRunning = false, note = "Historique insuffisant (${currentData.size}/8 tirages).")
            return
        }

        coordinatorScope.launch {
            val loadedData = if (currentData.size >= 8) {
                currentData
            } else {
                withContext(Dispatchers.IO) { lotoDao.getAllResultsOnce() }.also {
                    if (it.isNotEmpty()) latestHistorySnapshot = it
                }
            }
            if (loadedData.size < 8) {
                publishChercheurStatus(isRunning = false, note = "Historique insuffisant (${loadedData.size}/8 tirages).")
                return@launch
            }

            val period = _selectedPeriod.value
            val customDays = _customPeriodDays.value
            val nowMs = clockProvider()
            val clean = loadedData
                .distinctBy { it.uniqueId }
                .sortedWith(compareBy<LotoResultEntity> { it.timestamp }.thenBy { it.uniqueId })
            cachedChronologicalHistory = clean
            val cache = cachedCausalCache ?: DiscoveryEngine.buildCausalCache(
                history = clean,
                period = period,
                referenceNowMs = nowMs,
                customDays = customDays
            ).also { cachedCausalCache = it }

            if (cache.steps.size < 4) {
                publishChercheurStatus(isRunning = false, note = "Fenêtre causale trop courte pour Le Chercheur.")
                return@launch
            }

            val nextCycle = synchronized(chercheurLock) { ++chercheurCycles }
            publishChercheurStatus(
                isRunning = true,
                note = "Le Chercheur étudie chaque numéro, Pos1..5 et tirage complet..."
            )

            val seenSigs = (evolutionState.confirmationTestedSignatures + savedMilestoneSignatures)
            val cycleOutput = withContext(Dispatchers.Default) {
                ChercheurEngine.runResearchAndDistill(
                    cache = cache,
                    cycleIndex = nextCycle,
                    generation = evolutionState.generation.coerceAtLeast(1),
                    seenSignatures = seenSigs
                )
            }

            if (_discoveryState.value.phase == AutonomousPhase.ANALYZING) {
                synchronized(chercheurLock) {
                    pendingChercheurGenomes.addAll(cycleOutput.synthesizedGenomes)
                    genomeToTeachingId.putAll(cycleOutput.teachingByGenomeId)
                    totalProducedTeachings += cycleOutput.teachings.size
                    totalInjectedStrategies += cycleOutput.synthesizedGenomes.size
                    activeTeachings.addAll(0, cycleOutput.teachings)
                    while (activeTeachings.size > 16) activeTeachings.removeAt(activeTeachings.lastIndex)
                }
                appendLog("${cycleOutput.cycleSummaryNote} (injecté dans le moteur d'analyse)")
                publishChercheurStatus(isRunning = false, note = cycleOutput.cycleSummaryNote)
            } else {
                val genResult = withContext(Dispatchers.Default) {
                    DiscoveryEngine.executeEvolutionGeneration(
                        state = evolutionState,
                        cache = cache,
                        periodName = period.name,
                        minObjectiveStreak = targetObjectiveStreak,
                        injectedGenomes = cycleOutput.synthesizedGenomes
                    )
                }
                evolutionState = genResult.nextState
                withContext(Dispatchers.IO + NonCancellable) {
                    discoveryRepository.saveFormulas(genResult.persistedEntities)
                }
                genResult.bestTrainEval?.let { top ->
                    if (top.rollingStabilityPassed && top.rollStats.avgMatches >= BacktestResult.RANDOM_EXPECTED_MATCHES) {
                        val s = maxOf(top.discStats.maxConsecutiveQualifyingDays, top.rollStats.maxConsecutiveQualifyingDays)
                        if (s >= 2) {
                            val audited = DiscoveryEngine.auditCandidateReadOnly(top, cache, evolutionState.confirmationTestsCount)
                            val id = "ch_${s}j_${top.genome.describeCompact().hashCode().toUInt()}"
                            val entity = audited.toEntity(period.name, System.currentTimeMillis()).copy(id = id)
                            withContext(Dispatchers.IO + NonCancellable) {
                                discoveryRepository.saveFormulas(listOf(entity))
                                savedWinnersCache = discoveryRepository.loadValidatedFormulas(period, 20).map { it.toSummary() }
                            }
                            promoteStrongestSavedWinner(fallbackEval = audited, fallbackEntityId = id)
                        }
                    }
                }
                recordChercheurOutcomes(cycleOutput, genResult.injectedOutcomes)
                appendLog(cycleOutput.cycleSummaryNote)
                _discoveryState.update { cur ->
                    cur.copy(
                        generation = evolutionState.generation,
                        experimentsCount = evolutionState.experimentsCount,
                        savedWinners = savedWinnersCache,
                        activeWinnerId = activeWinnerId,
                        bestValidatedStreak = bestValidatedStreak,
                        liveLogs = snapshotLogs(),
                        chercheurStatus = currentChercheurStatus(isRunning = false, note = cycleOutput.cycleSummaryNote)
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
                while (activeTeachings.size > 16) activeTeachings.removeAt(activeTeachings.lastIndex)
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
        val note = cycleOutput?.cycleSummaryNote
            ?: "Audit Chercheur : $totalKeptStrategies gardée(s) / $totalDiscardedStrategies jetée(s) sur $totalInjectedStrategies injectée(s)."
        publishChercheurStatus(isRunning = false, note = note)
    }

    private fun currentChercheurStatus(isRunning: Boolean = false, note: String? = null): ChercheurStatusSummary {
        return synchronized(chercheurLock) {
            val resolvedNote = note
                ?: _discoveryState.value.chercheurStatus.lastLessonNote.takeIf { it.isNotBlank() }
                ?: "Le Chercheur est prêt à analyser numéros, positions (1..5) et tirages complets."
            ChercheurStatusSummary(
                isRunning = isRunning,
                cyclesCompleted = chercheurCycles,
                teachingsProduced = totalProducedTeachings,
                teachingsInjected = totalInjectedStrategies,
                strategiesKept = totalKeptStrategies,
                strategiesDiscarded = totalDiscardedStrategies,
                lastLessonNote = resolvedNote,
                teachings = activeTeachings.toList()
            )
        }
    }

    private fun publishChercheurStatus(isRunning: Boolean, note: String) {
        val snap = currentChercheurStatus(isRunning = isRunning, note = note)
        _discoveryState.update { cur ->
            cur.copy(
                liveLogs = snapshotLogs(),
                chercheurStatus = snap
            )
        }
    }

    fun stopDiscovery() {
        actionEpoch.incrementAndGet()
        restoreJob?.cancel()
        discoveryJob?.cancel()
        discoveryJob = null
        val current = _discoveryState.value
        if (current.phase == AutonomousPhase.ANALYZING || current.phase == AutonomousPhase.SUSPENDED) {
            appendLog("Analyse fermée (Stop) par l'utilisateur à Gén.${evolutionState.generation} (${evolutionState.experimentsCount} expériences).")
            val stoppedState = current.copy(
                phase = AutonomousPhase.STOPPED,
                liveLogs = snapshotLogs(),
                honestStatusNote = "Analyse arrêtée et fermée proprement. Appuyez sur « Continuer » ou « Reprendre à zéro ».",
                infiniteMode = infiniteModeFlag,
                approvedBacktestNote = approvedBacktestNote,
                chercheurStatus = currentChercheurStatus(isRunning = false)
            )
            _discoveryState.value = stoppedState
            persistSnapshotAsync(stoppedState, _selectedPeriod.value)
        }
        DiscoveryForegroundService.stopService(application)
    }

    fun resumeActiveSessionFromStorageIfNeeded() {
        if (discoveryJob?.isActive == true) return
        val epoch = actionEpoch.get()
        restoreJob?.cancel()
        restoreJob = coordinatorScope.launch {
            restoreInitialSession(epoch)
        }
    }

    fun flushCheckpointAsync() {
        val state = _discoveryState.value
        if (state.phase == AutonomousPhase.ANALYZING || state.phase == AutonomousPhase.VALIDATED) {
            persistSnapshotAsync(state, _selectedPeriod.value)
        }
    }

    fun startContinuousDiscovery(reset: Boolean = true) {
        val epoch = actionEpoch.incrementAndGet()
        restoreJob?.cancel()
        discoveryJob?.cancel()

        val initialSnapshot = if (latestHistorySnapshot.isNotEmpty()) latestHistorySnapshot else emptyList()
        if (initialSnapshot.size in 1..7) {
            _prediction.value = null
            _backtestResult.value = null
            validatedGenome = null
            _discoveryState.value = EngineActionState(
                phase = AutonomousPhase.IDLE,
                liveLogs = listOf("Minimum 8 tirages requis pour démarrer la recherche symbolique."),
                honestStatusNote = "Historique insuffisant (${initialSnapshot.size}/8 tirages)."
            )
            DiscoveryForegroundService.stopService(application)
            return
        }

        val period = _selectedPeriod.value
        val activeCustomDays = _customPeriodDays.value
        val periodSummaryLabel = activeCustomDays?.let { "${it} Jours" } ?: period.label
        if (reset) {
            _prediction.value = null
            _backtestResult.value = null
            continueBeyondMode = false
            userPinnedWinner = false
            val selectedDaysBound = (activeCustomDays ?: period.days)?.coerceAtLeast(1) ?: 4
            targetObjectiveStreak = (preferredStrategyDays ?: 4).coerceIn(1, minOf(4, selectedDaysBound))
            latestAuditedEval = null
            latestRandomRef = null
            savedMilestoneSignatures.clear()
            resetMilestoneTracking()
            sessionSalt += 1L
            synchronized(stateLock) { accumulatedLogs.clear() }
            evolutionState = DiscoveryEngine.createInitialState(sessionSalt)
        } else {
            evolutionState = DiscoveryEngine.ensureEliteInPopulation(evolutionState, bestMilestoneGenome ?: validatedGenome)
        }

        appendLog(
            when {
                reset -> "Démarrage recherche symbolique libre ($periodSummaryLabel) — objectif ${targetObjectiveStreak}j continu..."
                continueBeyondMode -> "Poursuite illimitée depuis Gén.${evolutionState.generation} vers ≥${targetObjectiveStreak}j..."
                else -> "Reprise recherche continue depuis Gén.${evolutionState.generation} (objectif ≥${targetObjectiveStreak}j)..."
            }
        )
        val initialAnalyzingUi = _discoveryState.value.copy(
            phase = AutonomousPhase.ANALYZING,
            backtestTriggered = if (reset) false else _discoveryState.value.backtestTriggered,
            targetObjectiveStreak = targetObjectiveStreak,
            bestValidatedStreak = bestValidatedStreak,
            activeWinnerId = activeWinnerId,
            savedWinners = savedWinnersCache,
            preferredStrategyDays = preferredStrategyDays,
            liveLogs = snapshotLogs(),
            honestStatusNote = if (continueBeyondMode) {
                "Recherche continue au-delà de l'objectif (cible ≥${targetObjectiveStreak}j) — gagnants conservés."
            } else {
                "Évolution symbolique active en continu (objectif ${targetObjectiveStreak}j+ — paliers enregistrés)..."
            }
        )
        _discoveryState.value = initialAnalyzingUi

        DiscoveryForegroundService.startOrUpdate(
            context = application,
            generation = evolutionState.generation,
            experiments = evolutionState.experimentsCount,
            bestStreak = maxOf(initialAnalyzingUi.bestCombinedStreak, bestValidatedStreak)
        )

        discoveryJob = coordinatorScope.launch {
            val currentData = if (latestHistorySnapshot.isNotEmpty()) {
                latestHistorySnapshot
            } else {
                withContext(Dispatchers.IO) { lotoDao.getAllResultsOnce() }.also {
                    if (it.isNotEmpty()) latestHistorySnapshot = it
                }
            }

            if (currentData.size < 8) {
                _prediction.value = null
                _backtestResult.value = null
                _discoveryState.update {
                    EngineActionState(
                        phase = AutonomousPhase.IDLE,
                        savedWinners = savedWinnersCache,
                        activeWinnerId = activeWinnerId,
                        bestValidatedStreak = bestValidatedStreak,
                        liveLogs = listOf("Minimum 8 tirages requis pour démarrer la recherche symbolique."),
                        honestStatusNote = "Historique insuffisant (${currentData.size}/8 tirages)."
                    )
                }
                DiscoveryForegroundService.stopService(application)
                return@launch
            }

            if (reset) {
                val preservedWinners = withContext(Dispatchers.IO) {
                    discoveryRepository.clearSessionAndFormulas(period)
                    discoveryRepository.loadValidatedFormulas(period, 20).map { it.toSummary() }
                }
                savedWinnersCache = preservedWinners
                if (validatedGenome == null && preservedWinners.isNotEmpty()) {
                    val topEntity = withContext(Dispatchers.IO) { discoveryRepository.loadFormulaById(preservedWinners.first().id) }
                    if (topEntity != null) {
                        validatedGenome = parseGenomeOrNull(topEntity.genomeJson)
                        activeWinnerId = topEntity.id
                        bestValidatedStreak = maxOf(bestValidatedStreak, topEntity.combinedStreak)
                        bestValidatedFitness = topEntity.fitnessScore
                    }
                }
                _discoveryState.update {
                    it.copy(
                        savedWinners = savedWinnersCache,
                        activeWinnerId = activeWinnerId,
                        bestValidatedStreak = bestValidatedStreak
                    )
                }
            }

            val nowMs = clockProvider()
            val anchorDayKey = LotoDateUtils.formatDayKey(nowMs)
            val customDays = _customPeriodDays.value
            val datasetKey = "${period.name}_${customDays ?: "ALL"}_${anchorDayKey}_${currentData.size}_${currentData.firstOrNull()?.uniqueId}_${currentData.lastOrNull()?.uniqueId}"
            val cleanChronological = currentData
                .distinctBy { it.uniqueId }
                .sortedWith(compareBy<LotoResultEntity> { it.timestamp }.thenBy { it.uniqueId })
            cachedChronologicalHistory = cleanChronological
            refreshNextDrawTarget(cleanChronological, nowMs)

            val cache = if (cachedCacheKey == datasetKey && cachedCausalCache != null) {
                cachedCausalCache!!
            } else {
                DiscoveryEngine.buildCausalCache(
                    history = cleanChronological,
                    period = period,
                    referenceNowMs = nowMs,
                    customDays = customDays
                ).also {
                    cachedCausalCache = it
                    cachedCacheKey = datasetKey
                }
            }

            if (cache.steps.size < 3) {
                appendLog("Période trop courte (${cache.steps.size} tirages) pour 4 étapes causales.")
                val stopped = _discoveryState.value.copy(
                    phase = AutonomousPhase.STOPPED,
                    periodDateRangeLabel = cache.dateRangeLabel,
                    liveLogs = snapshotLogs(),
                    honestStatusNote = "Historique insuffisant sur $periodSummaryLabel pour garantir une séparation 4 niveaux."
                )
                _discoveryState.value = stopped
                persistSessionBlocking(stopped, period)
                DiscoveryForegroundService.stopService(application)
                return@launch
            }

            val cacheDistinctDays = cache.steps.asSequence().map { it.dayKey }.distinct().count().coerceAtLeast(1)
            targetObjectiveStreak = (preferredStrategyDays ?: 4).coerceIn(1, cacheDistinctDays)

            appendLog(
                "4 niveaux prêts : Déc(${cache.discoverySteps.size}) → Gliss(${cache.rollingSteps.size}) → " +
                    "Conf(${cache.confirmationSteps.size}) → Scellé(${cache.sealedSteps.size})"
            )

            // Sauvegarder immediatement phase=ANALYZING au demarrage pour survie en cas de redemarrage OS
            persistSessionBlocking(_discoveryState.value.copy(periodDateRangeLabel = cache.dateRangeLabel), period)

            var lastPersistMs = System.currentTimeMillis()
            var lastNotificationMs = 0L
            var lastUiEmitMs = 0L

            while (isActive && actionEpoch.get() == epoch) {
                ensureActive()

                try {
                    testInjectedErrorOnce?.let { hook ->
                        testInjectedErrorOnce = null
                        hook()
                    }

                    var autoCycleOutput: ChercheurCycleOutput? = null
                    val injectedBatch = ArrayList<SymbolicGenome>()
                    synchronized(chercheurLock) {
                        if (pendingChercheurGenomes.isNotEmpty()) {
                            injectedBatch.addAll(pendingChercheurGenomes)
                            pendingChercheurGenomes.clear()
                        }
                    }
                    if (injectedBatch.isEmpty() && evolutionState.generation % 3 == 0 && cache.steps.size >= 4) {
                        val nextCycle = synchronized(chercheurLock) { ++chercheurCycles }
                        val seenSigs = (evolutionState.confirmationTestedSignatures + savedMilestoneSignatures)
                        autoCycleOutput = ChercheurEngine.runResearchAndDistill(
                            cache = cache,
                            cycleIndex = nextCycle,
                            generation = (evolutionState.generation + 1).coerceAtLeast(1),
                            seenSignatures = seenSigs
                        )
                        injectedBatch.addAll(autoCycleOutput.synthesizedGenomes)
                    }

                    val genResult = DiscoveryEngine.executeEvolutionGeneration(
                        state = evolutionState,
                        cache = cache,
                        periodName = period.name,
                        minObjectiveStreak = targetObjectiveStreak,
                        injectedGenomes = injectedBatch
                    )
                    if (autoCycleOutput != null || genResult.injectedOutcomes.isNotEmpty()) {
                        recordChercheurOutcomes(autoCycleOutput, genResult.injectedOutcomes)
                    }
                    evolutionState = genResult.nextState

                    withContext(Dispatchers.IO) {
                        discoveryRepository.saveFormulas(genResult.persistedEntities)
                        if (evolutionState.generation % 25 == 0) {
                            discoveryRepository.pruneOldFormulas(period)
                        }
                    }

                    if (genResult.auditedEval != null) {
                        latestAuditedEval = genResult.auditedEval
                        latestRandomRef = genResult.randomComparison
                    } else if (latestRandomRef == null) {
                        latestRandomRef = genResult.randomComparison
                    }

                    val bestTrain = genResult.bestTrainEval
                    val bestSummaries = evolutionState.bestGenomes.take(6).map { it.toSummary() }
                    val winning = genResult.winningEval

                    if (winning != null) {
                        val winnerConfStreak = winning.confStats?.maxConsecutiveQualifyingDays ?: 0
                        val winnerRollStreak = winning.rollStats.maxConsecutiveQualifyingDays
                        val winnerDiscStreak = winning.discStats.maxConsecutiveQualifyingDays
                        val winnerStreak = maxOf(winnerConfStreak, winnerRollStreak, winnerDiscStreak, winning.consolidatedTrainStreak, winning.consolidatedFullStreak)
                        val sig = winning.genome.describeCompact()
                        savedMilestoneSignatures.add(sig)
                        val winnerEntityId = "strat_${winnerStreak}j_${sig.hashCode().toUInt()}"
                        val winnerEntity = winning.toEntity(period.name, System.currentTimeMillis()).copy(
                            id = winnerEntityId,
                            status = "VALIDATED"
                        )
                        val prevBestValidated = bestValidatedStreak

                        withContext(Dispatchers.IO + NonCancellable) {
                            discoveryRepository.saveFormulas(listOf(winnerEntity))
                            discoveryRepository.saveCheckpointBestFormula(
                                period = period,
                                entity = winnerEntity
                            )
                            savedWinnersCache = discoveryRepository.loadValidatedFormulas(period, 20)
                                .map { it.toSummary() }
                        }

                        latestAuditedEval = winning
                        latestRandomRef = genResult.randomComparison
                        promoteStrongestSavedWinner(
                            fallbackEval = winning,
                            fallbackEntityId = winnerEntityId
                        )

                        bestMilestoneGenome = winning.genome
                        bestMilestoneEval = winning
                        bestMilestoneCombinedStreak = maxOf(bestMilestoneCombinedStreak, winnerStreak)
                        bestMilestoneDiscStreak = maxOf(bestMilestoneDiscStreak, winnerDiscStreak)
                        bestMilestoneRollStreak = maxOf(bestMilestoneRollStreak, winnerRollStreak)
                        bestMilestoneConfStreak = maxOf(bestMilestoneConfStreak, winnerConfStreak)
                        bestMilestoneFitness = winning.fitnessScore
                        evolutionState = DiscoveryEngine.ensureEliteInPopulation(evolutionState, winning.genome)

                        val reachedFourPlus = winnerStreak >= 4
                        val nextTarget = if (cacheDistinctDays <= 4) {
                            minOf(cacheDistinctDays, maxOf(targetObjectiveStreak, winnerStreak))
                        } else {
                            maxOf(
                                4,
                                if (maxOf(winnerStreak, bestValidatedStreak) >= 4) maxOf(winnerStreak, bestValidatedStreak) + 1 else 4,
                                targetObjectiveStreak
                            )
                        }
                        targetObjectiveStreak = nextTarget
                        if (reachedFourPlus && winnerStreak > prevBestValidated) {
                            DiscoveryForegroundService.notifyMilestoneWhileRunning(
                                context = application,
                                streak = winnerStreak,
                                generation = evolutionState.generation,
                                experiments = evolutionState.experimentsCount
                            )
                        }

                        val milestoneMsg = if (reachedFourPlus) {
                            "OBJECTIF ${winnerStreak}J ENREGISTRÉ (Gén.${evolutionState.generation}) — Backtest débloqué (${bestValidatedStreak}J), recherche continue vers ≥${nextTarget}j..."
                        } else {
                            "Palier ${winnerStreak}J validé & enregistré (Gén.${evolutionState.generation}) — Backtest débloqué, poursuite vers ${nextTarget}j..."
                        }
                        appendLog(milestoneMsg)

                        val fourStage = buildFourStageMetrics(winning, cache, latestAuditedEval)
                        val continueUi = EngineActionState(
                            phase = AutonomousPhase.ANALYZING,
                            generation = evolutionState.generation,
                            experimentsCount = evolutionState.experimentsCount,
                            confirmationTestsCount = evolutionState.confirmationTestsCount,
                            discoveryDrawsCount = cache.discoverySteps.size,
                            rollingDrawsCount = cache.rollingSteps.size,
                            confirmationDrawsCount = cache.confirmationSteps.size,
                            sealedDrawsCount = cache.sealedSteps.size,
                            periodDateRangeLabel = cache.dateRangeLabel,
                            currentPrimaryExpr = winning.genome.describePrimary(),
                            currentSecondaryExpr = winning.genome.describeSecondary(),
                            currentSlotSplit = "${winning.genome.primarySlots}+${5 - winning.genome.primarySlots}",
                            bestDiscoveryStreak = maxOf(_discoveryState.value.bestDiscoveryStreak, bestMilestoneDiscStreak),
                            bestRollingStreak = maxOf(_discoveryState.value.bestRollingStreak, bestMilestoneRollStreak),
                            bestConfirmationStreak = maxOf(_discoveryState.value.bestConfirmationStreak, bestMilestoneConfStreak),
                            bestSealedStreak = maxOf(_discoveryState.value.bestSealedStreak, winning.sealedStats?.maxConsecutiveQualifyingDays ?: 0),
                            bestCombinedStreak = maxOf(_discoveryState.value.bestCombinedStreak, bestMilestoneCombinedStreak, bestValidatedStreak),
                            bestQualifyingDays = (winning.confStats?.qualifyingDays ?: 0) + winning.rollStats.qualifyingDays,
                            bestTwoPlusCount = winning.discStats.twoPlusCount + winning.rollStats.twoPlusCount + (winning.confStats?.twoPlusCount ?: 0),
                            bestMaxZeroStreak = winning.confStats?.maxZeroStreak ?: winning.rollStats.maxZeroStreak,
                            discoveredFeaturesHistogram = genResult.featureHistogram,
                            bestFormulas = (listOf(winning.toSummary().copy(id = winnerEntityId)) + bestSummaries).distinctBy { it.id }.take(6),
                            rejectedFormulas = evolutionState.rejectedHistory.take(8),
                            liveLogs = snapshotLogs(),
                            fourStageMetrics = fourStage,
                            randomReference = genResult.randomComparison,
                            backtestTriggered = _discoveryState.value.backtestTriggered,
                            honestStatusNote = milestoneMsg,
                            targetObjectiveStreak = nextTarget,
                            bestValidatedStreak = bestValidatedStreak,
                            activeWinnerId = activeWinnerId,
                            savedWinners = savedWinnersCache,
                            preferredStrategyDays = preferredStrategyDays,
                            infiniteMode = infiniteModeFlag,
                            approvedBacktestNote = approvedBacktestNote,
                            chercheurStatus = currentChercheurStatus()
                        )
                        _discoveryState.value = continueUi
                        lastUiEmitMs = System.currentTimeMillis()
                        persistSessionBlocking(continueUi, period)
                        lastPersistMs = lastUiEmitMs
                        DiscoveryForegroundService.updateProgressNotificationIfPermitted(
                            context = application,
                            generation = evolutionState.generation,
                            experiments = evolutionState.experimentsCount,
                            bestStreak = continueUi.bestCombinedStreak
                        )
                        lastNotificationMs = lastUiEmitMs
                        yield()
                        continue
                    }

                    val genDiscStreak = bestTrain?.discStats?.maxConsecutiveQualifyingDays ?: 0
                    val genRollStreak = bestTrain?.rollStats?.maxConsecutiveQualifyingDays ?: 0
                    val genConfStreak = bestTrain?.confStats?.maxConsecutiveQualifyingDays ?: 0
                    val genCombinedStreak = maxOf(genDiscStreak, genRollStreak, genConfStreak, bestTrain?.consolidatedTrainStreak ?: 0, bestTrain?.consolidatedFullStreak ?: 0)

                    val prevCombined = bestMilestoneCombinedStreak
                    val prevDisc = bestMilestoneDiscStreak
                    val prevRoll = bestMilestoneRollStreak

                    val streakAdvanced = genCombinedStreak > prevCombined ||
                        genDiscStreak > prevDisc ||
                        genRollStreak > prevRoll ||
                        genConfStreak > bestMilestoneConfStreak

                    val betterFormulaFound = bestTrain != null && (
                        bestMilestoneGenome == null ||
                            genCombinedStreak > prevCombined ||
                            (genCombinedStreak == prevCombined && bestTrain.fitnessScore > bestMilestoneFitness)
                        )

                    if (streakAdvanced) {
                        bestMilestoneCombinedStreak = maxOf(bestMilestoneCombinedStreak, genCombinedStreak)
                        bestMilestoneDiscStreak = maxOf(bestMilestoneDiscStreak, genDiscStreak)
                        bestMilestoneRollStreak = maxOf(bestMilestoneRollStreak, genRollStreak)
                        bestMilestoneConfStreak = maxOf(bestMilestoneConfStreak, genConfStreak)
                    }
                    if (betterFormulaFound) {
                        val safeTrain = bestTrain!!
                        bestMilestoneGenome = safeTrain.genome
                        bestMilestoneEval = safeTrain
                        bestMilestoneFitness = safeTrain.fitnessScore
                    } else if (bestMilestoneGenome == null) {
                        bestMilestoneGenome = evolutionState.population.firstOrNull()
                    }

                    val cacheDaysCount = cache.steps.asSequence().map { it.dayKey }.distinct().count().coerceAtLeast(1)
                    val milestonePool = buildList {
                        bestTrain?.let { add(it) }
                        addAll(evolutionState.bestGenomes.take(6))
                    }.distinctBy { it.genome.describeCompact() }
                    var milestoneUnlockedNow = false
                    for (cand in milestonePool) {
                        if (!cand.rollingStabilityPassed || cand.rollStats.avgMatches < BacktestResult.RANDOM_EXPECTED_MATCHES) continue
                        val candCombinedStreak = maxOf(
                            cand.discStats.maxConsecutiveQualifyingDays,
                            cand.rollStats.maxConsecutiveQualifyingDays,
                            cand.confStats?.maxConsecutiveQualifyingDays ?: 0,
                            cand.consolidatedTrainStreak,
                            cand.consolidatedFullStreak
                        )
                        val trainTwoPlusTotal = cand.discStats.twoPlusCount + cand.rollStats.twoPlusCount
                        val oneDayMilestoneReady = cacheDaysCount == 1 &&
                            (cand.consolidatedTrainStreak >= 1 || trainTwoPlusTotal >= 2)
                        val tier = when {
                            candCombinedStreak >= 2 -> candCombinedStreak
                            candCombinedStreak == 1 || oneDayMilestoneReady -> 1
                            else -> 0
                        }
                        if (tier < 1) continue
                        val sig = cand.genome.describeCompact()
                        val existingAtTier = savedWinnersCache.firstOrNull { it.combinedStreak == tier }
                        val shouldSaveMilestone = sig !in savedMilestoneSignatures && (
                            existingAtTier == null ||
                                tier > bestValidatedStreak ||
                                cand.fitnessScore > existingAtTier.fitnessScore + 0.25
                            )
                        if (shouldSaveMilestone) {
                            savedMilestoneSignatures.add(sig)
                            val audited = DiscoveryEngine.auditCandidateReadOnly(
                                candidate = cand,
                                cache = cache,
                                confirmationTestsCount = evolutionState.confirmationTestsCount
                            )
                            latestAuditedEval = audited
                            latestRandomRef = RandomReferenceComparison(
                                simulationsCount = cache.confirmationMonteCarlo.simulationsCount,
                                confirmationMeanAvgMatches = cache.confirmationMonteCarlo.meanAvgMatches,
                                confirmationCiLow = cache.confirmationMonteCarlo.avgMatchesCiLow,
                                confirmationCiHigh = cache.confirmationMonteCarlo.avgMatchesCiHigh,
                                confirmationMeanQualifyingDays = cache.confirmationMonteCarlo.meanQualifyingDays,
                                pValueRaw = audited.pValueRaw,
                                pValueAdjusted = audited.pValueAdjusted,
                                beatsRandomAfterCorrection = audited.status == "VALIDATED"
                            )
                            val milestoneId = "strat_${tier}j_${sig.hashCode().toUInt()}"
                            val milestoneEntity = audited.toEntity(period.name, System.currentTimeMillis()).copy(
                                id = milestoneId,
                                combinedStreak = maxOf(audited.toSummary().combinedStreak, tier)
                            )
                            val prevBestVal = bestValidatedStreak

                            withContext(Dispatchers.IO + NonCancellable) {
                                discoveryRepository.saveFormulas(listOf(milestoneEntity))
                                discoveryRepository.saveCheckpointBestFormula(
                                    period = period,
                                    entity = milestoneEntity
                                )
                                savedWinnersCache = discoveryRepository.loadValidatedFormulas(period, 20)
                                    .map { it.toSummary() }
                            }

                            promoteStrongestSavedWinner(
                                fallbackEval = audited,
                                fallbackEntityId = milestoneId
                            )
                            if (tier >= 4) {
                                targetObjectiveStreak = maxOf(4, maxOf(tier, bestValidatedStreak) + 1, targetObjectiveStreak)
                                if (tier > prevBestVal) {
                                    DiscoveryForegroundService.notifyMilestoneWhileRunning(
                                        context = application,
                                        streak = tier,
                                        generation = evolutionState.generation,
                                        experiments = evolutionState.experimentsCount
                                    )
                                }
                            }
                            milestoneUnlockedNow = true
                            appendLog(
                                "Stratégie ${tier}J enregistrée (Gén.${evolutionState.generation}) — Backtest & Prédiction débloqués (${bestValidatedStreak}J), recherche continue vers ${targetObjectiveStreak}j..."
                            )
                        }
                    }

                    // Garantir que le meilleur genome sauvegarde reste present dans la population active
                    evolutionState = DiscoveryEngine.ensureEliteInPopulation(evolutionState, bestMilestoneGenome)

                    if (!milestoneUnlockedNow) {
                        if (streakAdvanced) {
                            appendLog(
                                "Palier ${bestMilestoneCombinedStreak}/${targetObjectiveStreak}j sauvegardé (Gén.${evolutionState.generation}) — meilleur génome conservé [${bestMilestoneGenome?.describeCompact().orEmpty()}]."
                            )
                        } else {
                            appendLog(genResult.generationNote)
                        }
                    }

                    val displayedGenome = bestMilestoneGenome ?: bestTrain?.genome ?: evolutionState.population.firstOrNull()
                    val effectiveRandomRef = if (genResult.auditedEval != null) {
                        genResult.randomComparison
                    } else {
                        latestRandomRef ?: genResult.randomComparison
                    }
                    val runningUi = EngineActionState(
                        phase = AutonomousPhase.ANALYZING,
                        generation = evolutionState.generation,
                        experimentsCount = evolutionState.experimentsCount,
                        confirmationTestsCount = evolutionState.confirmationTestsCount,
                        discoveryDrawsCount = cache.discoverySteps.size,
                        rollingDrawsCount = cache.rollingSteps.size,
                        confirmationDrawsCount = cache.confirmationSteps.size,
                        sealedDrawsCount = cache.sealedSteps.size,
                        periodDateRangeLabel = cache.dateRangeLabel,
                        currentPrimaryExpr = displayedGenome?.describePrimary().orEmpty(),
                        currentSecondaryExpr = displayedGenome?.describeSecondary().orEmpty(),
                        currentSlotSplit = displayedGenome?.let { "${it.primarySlots}+${5 - it.primarySlots}" }.orEmpty(),
                        bestDiscoveryStreak = maxOf(_discoveryState.value.bestDiscoveryStreak, bestMilestoneDiscStreak),
                        bestRollingStreak = maxOf(_discoveryState.value.bestRollingStreak, bestMilestoneRollStreak),
                        bestConfirmationStreak = maxOf(_discoveryState.value.bestConfirmationStreak, bestMilestoneConfStreak, latestAuditedEval?.confStats?.maxConsecutiveQualifyingDays ?: 0),
                        bestSealedStreak = maxOf(_discoveryState.value.bestSealedStreak, latestAuditedEval?.sealedStats?.maxConsecutiveQualifyingDays ?: 0),
                        bestCombinedStreak = maxOf(_discoveryState.value.bestCombinedStreak, bestMilestoneCombinedStreak, bestValidatedStreak),
                        bestQualifyingDays = bestTrain?.rollStats?.qualifyingDays ?: _discoveryState.value.bestQualifyingDays,
                        bestTwoPlusCount = maxOf(
                            _discoveryState.value.bestTwoPlusCount,
                            (bestTrain?.discStats?.twoPlusCount ?: 0) + (bestTrain?.rollStats?.twoPlusCount ?: 0)
                        ),
                        bestMaxZeroStreak = bestTrain?.rollStats?.maxZeroStreak ?: 0,
                        discoveredFeaturesHistogram = genResult.featureHistogram,
                        bestFormulas = bestSummaries,
                        rejectedFormulas = evolutionState.rejectedHistory.take(8),
                        liveLogs = snapshotLogs(),
                        fourStageMetrics = (bestMilestoneEval ?: bestTrain)?.let { buildFourStageMetrics(it, cache, latestAuditedEval) },
                        randomReference = effectiveRandomRef,
                        backtestTriggered = _discoveryState.value.backtestTriggered,
                        honestStatusNote = when {
                            milestoneUnlockedNow -> "Stratégie ${bestValidatedStreak}J enregistrée — Backtest/Prédiction débloqués, recherche continue vers ${targetObjectiveStreak}j."
                            streakAdvanced -> "Palier ${bestMilestoneCombinedStreak}/${targetObjectiveStreak}j sauvegardé — poursuite avec le meilleur génome."
                            else -> genResult.generationNote
                        },
                        targetObjectiveStreak = targetObjectiveStreak,
                        bestValidatedStreak = bestValidatedStreak,
                        activeWinnerId = activeWinnerId,
                        savedWinners = savedWinnersCache,
                        preferredStrategyDays = preferredStrategyDays,
                        infiniteMode = infiniteModeFlag,
                        approvedBacktestNote = approvedBacktestNote,
                        chercheurStatus = currentChercheurStatus()
                    )

                    val nowMs = System.currentTimeMillis()
                    if (milestoneUnlockedNow ||
                        streakAdvanced ||
                        betterFormulaFound ||
                        evolutionState.generation <= 5 ||
                        evolutionState.generation % 5 == 0 ||
                        nowMs - lastUiEmitMs >= 160L
                    ) {
                        _discoveryState.value = runningUi
                        lastUiEmitMs = nowMs
                    }
                    if (milestoneUnlockedNow ||
                        streakAdvanced ||
                        betterFormulaFound ||
                        evolutionState.generation == 1 ||
                        evolutionState.generation % 10 == 0 ||
                        nowMs - lastPersistMs >= 2_000L
                    ) {
                        val checkpointEval = bestMilestoneEval
                        if (checkpointEval != null && (streakAdvanced || betterFormulaFound)) {
                            withContext(Dispatchers.IO + NonCancellable) {
                                discoveryRepository.saveCheckpointBestFormula(
                                    period = period,
                                    entity = checkpointEval.toEntity(period.name, nowMs)
                                )
                            }
                        }
                        persistSessionBlocking(runningUi, period)
                        lastPersistMs = nowMs
                    }

                    if (milestoneUnlockedNow || streakAdvanced || nowMs - lastNotificationMs >= 1_200L) {
                        DiscoveryForegroundService.updateProgressNotificationIfPermitted(
                            context = application,
                            generation = evolutionState.generation,
                            experiments = evolutionState.experimentsCount,
                            bestStreak = runningUi.bestCombinedStreak
                        )
                        lastNotificationMs = nowMs
                    }
                } catch (ce: CancellationException) {
                    throw ce
                } catch (err: Throwable) {
                    val recoveryGenome = resolveBestRecoveryGenome(period)
                    if (recoveryGenome != null) {
                        bestMilestoneGenome = recoveryGenome
                        evolutionState = DiscoveryEngine.ensureEliteInPopulation(evolutionState, recoveryGenome)
                    }
                    appendLog(
                        "Erreur récupérée (${err.javaClass.simpleName}) — poursuite avec le meilleur génome sauvegardé (${bestMilestoneCombinedStreak}/${targetObjectiveStreak}j)."
                    )
                    val recoveredUi = _discoveryState.value.copy(
                        phase = AutonomousPhase.ANALYZING,
                        generation = evolutionState.generation,
                        experimentsCount = evolutionState.experimentsCount,
                        currentPrimaryExpr = recoveryGenome?.describePrimary() ?: _discoveryState.value.currentPrimaryExpr,
                        currentSecondaryExpr = recoveryGenome?.describeSecondary() ?: _discoveryState.value.currentSecondaryExpr,
                        currentSlotSplit = recoveryGenome?.let { "${it.primarySlots}+${5 - it.primarySlots}" }
                            ?: _discoveryState.value.currentSlotSplit,
                        bestDiscoveryStreak = maxOf(_discoveryState.value.bestDiscoveryStreak, bestMilestoneDiscStreak),
                        bestRollingStreak = maxOf(_discoveryState.value.bestRollingStreak, bestMilestoneRollStreak),
                        bestConfirmationStreak = maxOf(_discoveryState.value.bestConfirmationStreak, bestMilestoneConfStreak),
                        bestCombinedStreak = maxOf(_discoveryState.value.bestCombinedStreak, bestMilestoneCombinedStreak, bestValidatedStreak),
                        targetObjectiveStreak = targetObjectiveStreak,
                        bestValidatedStreak = bestValidatedStreak,
                        activeWinnerId = activeWinnerId,
                        savedWinners = savedWinnersCache,
                        preferredStrategyDays = preferredStrategyDays,
                        liveLogs = snapshotLogs(),
                        honestStatusNote = "Reprise automatique après erreur avec le meilleur génome sauvegardé (${bestMilestoneCombinedStreak}/${targetObjectiveStreak}j)."
                    )
                    _discoveryState.value = recoveredUi
                    runCatching { persistSessionBlocking(recoveredUi, period) }
                    delay(250L)
                    continue
                }

                yield()
            }
        }
    }

    fun selectPreferredStrategyDays(days: Int?) {
        val normalized = days?.coerceIn(1, 14)
        preferredStrategyDays = normalized
        val baseline = maxOf(bestValidatedStreak, bestMilestoneCombinedStreak)
        targetObjectiveStreak = if (normalized == null) {
            if (baseline >= 4) baseline + 1 else 4
        } else {
            maxOf(normalized, if (baseline >= 4) baseline + 1 else normalized)
        }
        if (normalized == null) {
            userPinnedWinner = false
        }

        coordinatorScope.launch {
            val matchingPool = resolvePoolForTier(savedWinnersCache, normalized)
            val topMatch = matchingPool.firstOrNull()
            if (topMatch != null) {
                val entity = withContext(Dispatchers.IO) {
                    discoveryRepository.loadFormulaById(topMatch.id)
                }
                val genome = entity?.genomeJson?.let { parseGenomeOrNull(it) }
                if (genome != null) {
                    validatedGenome = genome
                    activeWinnerId = topMatch.id
                    bestValidatedFitness = topMatch.fitnessScore
                    userPinnedWinner = normalized != null
                    val tierLabel = normalized?.let { "${it}J" } ?: "Auto (${topMatch.combinedStreak}J)"
                    appendLog("Palier backtest $tierLabel sélectionné [${genome.describeCompact()}] — prédiction débloquée.")
                    _discoveryState.update {
                        it.copy(
                            preferredStrategyDays = normalized,
                            targetObjectiveStreak = targetObjectiveStreak,
                            activeWinnerId = topMatch.id,
                            currentPrimaryExpr = genome.describePrimary(),
                            currentSecondaryExpr = genome.describeSecondary(),
                            currentSlotSplit = "${genome.primarySlots}+${5 - genome.primarySlots}",
                            liveLogs = snapshotLogs(),
                            honestStatusNote = "Stratégie $tierLabel active — Backtest & Prédiction débloqués."
                        )
                    }
                    if (_discoveryState.value.backtestTriggered) {
                        launchBacktestAndPrediction()
                    }
                    return@launch
                }
            }

            if (normalized != null) {
                validatedGenome = null
                activeWinnerId = ""
                userPinnedWinner = false
                appendLog("Cible backtest définie sur ${normalized}J consécutifs — recherche en cours pour débloquer.")
                _discoveryState.update {
                    it.copy(
                        preferredStrategyDays = normalized,
                        targetObjectiveStreak = targetObjectiveStreak,
                        activeWinnerId = "",
                        liveLogs = snapshotLogs(),
                        honestStatusNote = "Palier ${normalized}J sélectionné — modélisation verrouillée jusqu'à découverte d'un motif ≥${normalized}J."
                    )
                }
            } else {
                _discoveryState.update {
                    it.copy(
                        preferredStrategyDays = null,
                        targetObjectiveStreak = targetObjectiveStreak,
                        liveLogs = snapshotLogs()
                    )
                }
            }
        }
    }

    fun selectSavedWinner(winnerId: String) {
        coordinatorScope.launch {
            val entity = withContext(Dispatchers.IO) {
                discoveryRepository.loadFormulaById(winnerId)
            } ?: return@launch
            val genome = parseGenomeOrNull(entity.genomeJson) ?: return@launch
            userPinnedWinner = true
            validatedGenome = genome
            activeWinnerId = entity.id
            bestValidatedStreak = maxOf(bestValidatedStreak, entity.combinedStreak)
            bestValidatedFitness = entity.fitnessScore
            val currentPref = preferredStrategyDays
            val resolvedPref = if (currentPref != null && entity.combinedStreak < currentPref) {
                entity.combinedStreak
            } else {
                currentPref
            }
            preferredStrategyDays = resolvedPref

            appendLog("Stratégie ${entity.combinedStreak}J sélectionnée [${genome.describeCompact()}] pour Backtest & Prédiction.")
            _discoveryState.update {
                it.copy(
                    preferredStrategyDays = resolvedPref,
                    activeWinnerId = entity.id,
                    bestValidatedStreak = maxOf(it.bestValidatedStreak, entity.combinedStreak),
                    currentPrimaryExpr = genome.describePrimary(),
                    currentSecondaryExpr = genome.describeSecondary(),
                    currentSlotSplit = "${genome.primarySlots}+${5 - genome.primarySlots}",
                    liveLogs = snapshotLogs(),
                    honestStatusNote = "Stratégie ${entity.combinedStreak}J sélectionnée (Gén.${entity.generation}) — prêt pour Backtest & Prédiction."
                )
            }
            launchBacktestAndPrediction()
        }
    }

    suspend fun buildSavedStrategiesBackup(): Result<StrategyBackupExport> = runCatching {
        val period = _selectedPeriod.value
        val (dbWinners, dbCheckpoint) = withContext(Dispatchers.IO) {
            discoveryRepository.loadValidatedFormulas(period, 50) to
                discoveryRepository.loadCheckpointBestFormula(period)
        }

        val nowMs = System.currentTimeMillis()
        val memoryEval = latestAuditedEval ?: bestMilestoneEval
        val fallbackBestList = if (dbWinners.isEmpty() && validatedGenome != null) {
            val g = validatedGenome!!
            val sig = g.describeCompact()
            val streak = maxOf(bestValidatedStreak, 2)
            val entity = memoryEval?.takeIf { it.genome.describeCompact() == sig }
                ?.toEntity(period.name, nowMs)
                ?.copy(
                    id = activeWinnerId.ifBlank { "strat_${streak}j_${sig.hashCode().toUInt()}" },
                    status = if (streak >= 4) "VALIDATED" else "MILESTONE_${streak}J",
                    combinedStreak = streak
                )
                ?: DiscoveredFormulaEntity(
                    id = activeWinnerId.ifBlank { "strat_${streak}j_${sig.hashCode().toUInt()}" },
                    periodName = period.name,
                    generation = g.generation,
                    parentIdsCsv = g.parentIds.joinToString(","),
                    primaryExprText = g.describePrimary(),
                    secondaryExprText = g.describeSecondary(),
                    slotSplitLabel = "${g.primarySlots}+${5 - g.primarySlots}",
                    genomeJson = SymbolicFormula.genomeToJson(g),
                    usedFeaturesCsv = (
                        SymbolicFormula.usedFeatures(g.primaryExpr) +
                            SymbolicFormula.usedFeatures(g.secondaryExpr)
                        ).joinToString(",") { it.symbol },
                    status = if (streak >= 4) "VALIDATED" else "MILESTONE_${streak}J",
                    rejectionStage = "",
                    rejectionReason = "Stratégie ${streak}J enregistrée",
                    discoveryStreak = bestMilestoneDiscStreak,
                    rollingStreak = bestMilestoneRollStreak,
                    confirmationStreak = bestMilestoneConfStreak,
                    sealedStreak = 0,
                    combinedStreak = streak,
                    discoveryAvgMatches = 0f,
                    rollingAvgMatches = 0f,
                    confirmationAvgMatches = 0f,
                    sealedAvgMatches = 0f,
                    pValueAdjusted = 1.0,
                    fitnessScore = bestValidatedFitness.takeIf { it.isFinite() } ?: streak.toDouble(),
                    updatedAt = nowMs
                )
            listOf(entity)
        } else {
            dbWinners
        }

        val resolvedCheckpoint = dbCheckpoint ?: bestMilestoneEval?.toEntity(period.name, nowMs)
            ?.copy(id = DiscoveryRepository.checkpointIdFor(period))

        require(fallbackBestList.isNotEmpty() || resolvedCheckpoint != null) {
            "Aucune meilleure méthode enregistrée à sauvegarder pour le moment."
        }

        val payload = StrategyBackupHelper.serializeBackup(
            strategies = fallbackBestList,
            checkpointBest = resolvedCheckpoint,
            periodName = period.name,
            exportedAtMs = nowMs
        )
        val bestStreak = maxOf(
            fallbackBestList.maxOfOrNull { it.combinedStreak } ?: 0,
            resolvedCheckpoint?.combinedStreak ?: 0
        )
        StrategyBackupExport(
            jsonPayload = payload,
            strategiesCount = fallbackBestList.size,
            hasCheckpoint = resolvedCheckpoint != null,
            bestStreak = bestStreak,
            fileName = StrategyBackupHelper.buildBackupFileName(bestStreak, nowMs)
        )
    }

    suspend fun importSavedStrategiesBackup(rawPayload: String): Result<StrategyBackupImportResult> = runCatching {
        val period = _selectedPeriod.value
        val parsed = StrategyBackupHelper.parseBackup(
            rawText = rawPayload,
            targetPeriodName = period.name
        ).getOrThrow()

        var insertedCount = 0
        var updatedCount = 0
        var checkpointRestored = false

        val refreshedWinners = withContext(Dispatchers.IO + NonCancellable) {
            val existingById = discoveryRepository.loadValidatedFormulas(period, 50).associateBy { it.id }
            val existingBySig = existingById.values.associateBy { entity ->
                runCatching { SymbolicFormula.genomeFromJson(entity.genomeJson).describeCompact() }
                    .getOrDefault(entity.id)
            }

            val entitiesToSave = parsed.strategies.map { incoming ->
                val sig = runCatching { SymbolicFormula.genomeFromJson(incoming.genomeJson).describeCompact() }
                    .getOrDefault(incoming.id)
                val existing = existingById[incoming.id] ?: existingBySig[sig]
                if (existing == null) {
                    insertedCount++
                    incoming
                } else {
                    updatedCount++
                    if (incoming.combinedStreak > existing.combinedStreak ||
                        (incoming.combinedStreak == existing.combinedStreak && incoming.fitnessScore > existing.fitnessScore)
                    ) {
                        incoming.copy(id = existing.id)
                    } else {
                        existing
                    }
                }
            }

            if (entitiesToSave.isNotEmpty()) {
                discoveryRepository.saveFormulas(entitiesToSave)
            }

            val incomingCheckpoint = parsed.checkpointBest
            if (incomingCheckpoint != null) {
                val existingCp = discoveryRepository.loadCheckpointBestFormula(period)
                if (existingCp == null ||
                    incomingCheckpoint.combinedStreak > existingCp.combinedStreak ||
                    (incomingCheckpoint.combinedStreak == existingCp.combinedStreak &&
                        incomingCheckpoint.fitnessScore > existingCp.fitnessScore)
                ) {
                    discoveryRepository.saveCheckpointBestFormula(period, incomingCheckpoint)
                    checkpointRestored = true
                }
            }

            discoveryRepository.loadValidatedFormulas(period, 50).map { it.toSummary() }
        }

        savedWinnersCache = refreshedWinners

        // Injecter les genomes importes dans l'elite active pour l'evolution continue
        val allImportedEntities = parsed.strategies + listOfNotNull(parsed.checkpointBest)
        allImportedEntities.forEach { entity ->
            parseGenomeOrNull(entity.genomeJson)?.let { g ->
                evolutionState = DiscoveryEngine.ensureEliteInPopulation(evolutionState, g)
            }
        }

        val bestImportedEntity = parsed.strategies.maxWithOrNull(
            compareBy<DiscoveredFormulaEntity> { it.combinedStreak }
                .thenBy { it.fitnessScore }
        )
        val bestImportedGenome = (bestImportedEntity ?: parsed.checkpointBest)
            ?.let { parseGenomeOrNull(it.genomeJson) }

        if (bestImportedEntity != null) {
            bestValidatedStreak = maxOf(
                bestValidatedStreak,
                refreshedWinners.maxOfOrNull { it.combinedStreak } ?: bestImportedEntity.combinedStreak
            )
            if (validatedGenome == null ||
                bestImportedEntity.combinedStreak >= (refreshedWinners.firstOrNull { it.id == activeWinnerId }?.combinedStreak ?: 0)
            ) {
                validatedGenome = bestImportedGenome
                activeWinnerId = refreshedWinners.firstOrNull {
                    it.id == bestImportedEntity.id || it.primaryExprText == bestImportedEntity.primaryExprText
                }?.id ?: refreshedWinners.firstOrNull()?.id ?: bestImportedEntity.id
                bestValidatedFitness = bestImportedEntity.fitnessScore
            }
            bestMilestoneCombinedStreak = maxOf(bestMilestoneCombinedStreak, bestImportedEntity.combinedStreak)
            bestMilestoneDiscStreak = maxOf(bestMilestoneDiscStreak, bestImportedEntity.discoveryStreak)
            bestMilestoneRollStreak = maxOf(bestMilestoneRollStreak, bestImportedEntity.rollingStreak)
            bestMilestoneConfStreak = maxOf(bestMilestoneConfStreak, bestImportedEntity.confirmationStreak)
            bestMilestoneFitness = maxOf(bestMilestoneFitness, bestImportedEntity.fitnessScore)
            if (bestImportedGenome != null) {
                bestMilestoneGenome = bestImportedGenome
            }
        } else if (parsed.checkpointBest != null && bestImportedGenome != null) {
            val cp = parsed.checkpointBest
            if (cp.combinedStreak >= bestMilestoneCombinedStreak) {
                bestMilestoneGenome = bestImportedGenome
                bestMilestoneCombinedStreak = maxOf(bestMilestoneCombinedStreak, cp.combinedStreak)
                bestMilestoneDiscStreak = maxOf(bestMilestoneDiscStreak, cp.discoveryStreak)
                bestMilestoneRollStreak = maxOf(bestMilestoneRollStreak, cp.rollingStreak)
                bestMilestoneConfStreak = maxOf(bestMilestoneConfStreak, cp.confirmationStreak)
                bestMilestoneFitness = maxOf(bestMilestoneFitness, cp.fitnessScore)
            }
        }

        val totalStrategiesImported = parsed.strategies.size
        val bestStreakImported = maxOf(
            parsed.strategies.maxOfOrNull { it.combinedStreak } ?: 0,
            parsed.checkpointBest?.combinedStreak ?: 0
        )
        val summaryMsg = when {
            totalStrategiesImported > 0 ->
                "$totalStrategiesImported meilleure(s) méthode(s) importée(s) (max ${bestStreakImported}J) — prêtes pour Backtest & Prédiction."
            else ->
                "Meilleur checkpoint (${bestStreakImported}J) importé dans le moteur."
        }
        appendLog(summaryMsg)

        val currentUi = _discoveryState.value
        val updatedPhase = when {
            currentUi.phase == AutonomousPhase.ANALYZING -> AutonomousPhase.ANALYZING
            bestValidatedStreak >= 4 -> AutonomousPhase.VALIDATED
            else -> currentUi.phase
        }
        val activeSummary = refreshedWinners.firstOrNull { it.id == activeWinnerId } ?: refreshedWinners.firstOrNull()
        val updatedUi = currentUi.copy(
            phase = updatedPhase,
            generation = maxOf(currentUi.generation, bestImportedEntity?.generation ?: parsed.checkpointBest?.generation ?: 0),
            currentPrimaryExpr = bestImportedGenome?.describePrimary()
                ?: currentUi.currentPrimaryExpr,
            currentSecondaryExpr = bestImportedGenome?.describeSecondary()
                ?: currentUi.currentSecondaryExpr,
            currentSlotSplit = bestImportedGenome?.let { "${it.primarySlots}+${5 - it.primarySlots}" }
                ?: currentUi.currentSlotSplit,
            bestDiscoveryStreak = maxOf(currentUi.bestDiscoveryStreak, bestMilestoneDiscStreak),
            bestRollingStreak = maxOf(currentUi.bestRollingStreak, bestMilestoneRollStreak),
            bestConfirmationStreak = maxOf(currentUi.bestConfirmationStreak, bestMilestoneConfStreak),
            bestCombinedStreak = maxOf(currentUi.bestCombinedStreak, bestMilestoneCombinedStreak, bestValidatedStreak),
            bestValidatedStreak = bestValidatedStreak,
            activeWinnerId = activeWinnerId,
            savedWinners = refreshedWinners,
            bestFormulas = if (currentUi.bestFormulas.isEmpty()) refreshedWinners.take(6) else currentUi.bestFormulas,
            liveLogs = snapshotLogs(),
            honestStatusNote = summaryMsg
        )
        _discoveryState.value = updatedUi
        persistSessionBlocking(updatedUi, period)

        if (currentUi.backtestTriggered && bestImportedEntity != null) {
            launchBacktestAndPrediction()
        }

        StrategyBackupImportResult(
            importedCount = totalStrategiesImported,
            updatedCount = updatedCount,
            checkpointRestored = checkpointRestored,
            bestStreakImported = bestStreakImported,
            primaryWinnerId = activeSummary?.id ?: activeWinnerId.ifBlank { null },
            message = summaryMsg
        )
    }

    private fun onNewDrawsArrivedAutoRefresh() {
        precisionCausalCache = null
        precisionCausalCacheKey = ""
        val st = _discoveryState.value
        if (st.backtestTriggered && (validatedGenome != null || activeWinnerId.isNotBlank())) {
            launchBacktestAndPrediction()
        }
        val pst = _precisionUiState.value
        if (pst.backtestResult != null || pst.livePrediction != null) {
            launchPrecisionBacktestAndPrediction()
        }
    }

    // =========================================================================
    // MOTEUR PRÉCISION 100 % INDÉPENDANT (1er N° Pos1 & 2 N° Exacts)
    // =========================================================================

    fun setPrecisionTargetMode(mode: PrecisionTargetMode) {
        val cur = _precisionUiState.value
        if (cur.targetMode == mode) return
        precisionJob?.cancel()
        precisionJob = null
        precisionEvoState = null
        precisionWinnerEval = null
        precisionBestGenome = PrecisionTargetEngine.loadSavedBestGenome(precisionStorageDir, mode)
        appendPrecisionLog("Mode Précision sélectionné : ${mode.label}.")
        _precisionUiState.value = PrecisionUiState(
            targetMode = mode,
            phase = AutonomousPhase.IDLE,
            infiniteMode = cur.infiniteMode,
            activeFormulaDesc = precisionBestGenome?.describeCompact().orEmpty(),
            honestStatusNote = "Mode ${mode.label} prêt (indépendant du moteur 5 boules).",
            liveLogs = synchronized(precisionLogs) { precisionLogs.toList() }
        )
    }

    fun setPrecisionInfiniteMode(enabled: Boolean) {
        _precisionUiState.update { cur ->
            cur.copy(
                infiniteMode = enabled,
                honestStatusNote = if (enabled) "Mode infini Précision actif." else cur.honestStatusNote
            )
        }
    }

    fun startPrecisionDiscovery(continueBeyond: Boolean = false) {
        val draws = when {
            latestHistorySnapshot.isNotEmpty() -> latestHistorySnapshot
            cachedChronologicalHistory.isNotEmpty() -> cachedChronologicalHistory
            else -> emptyList()
        }
        val mode = _precisionUiState.value.targetMode
        precisionJob?.cancel()
        precisionJob = coordinatorScope.launch {
            val loadedDraws = if (draws.size >= 8) {
                draws
            } else {
                withContext(Dispatchers.IO) { lotoDao.getAllResultsOnce() }.also {
                    if (it.isNotEmpty()) latestHistorySnapshot = it
                }
            }
            if (loadedDraws.size < 8) {
                _precisionUiState.update {
                    it.copy(
                        phase = AutonomousPhase.IDLE,
                        honestStatusNote = "Minimum 8 tirages historiques requis (${loadedDraws.size}/8)."
                    )
                }
                return@launch
            }

            val period = _selectedPeriod.value
            val customDays = _customPeriodDays.value
            val nowMs = clockProvider()
            val anchorDayKey = LotoDateUtils.formatDayKey(nowMs)
            val expectedKey = "PREC|${period.name}|${customDays ?: "ALL"}|$anchorDayKey|${loadedDraws.size}|${loadedDraws.firstOrNull()?.uniqueId}|${loadedDraws.lastOrNull()?.uniqueId}"
            val cache = if (precisionCausalCache != null && precisionCausalCacheKey == expectedKey) {
                precisionCausalCache!!
            } else {
                PrecisionTargetEngine.buildCausalCache(
                    history = loadedDraws,
                    period = period,
                    referenceNowMs = nowMs,
                    customDays = customDays
                ).also {
                    precisionCausalCache = it
                    precisionCausalCacheKey = expectedKey
                }
            }

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
                precisionEvoState = PrecisionTargetEngine.createInitialState(
                    mode = mode,
                    populationSize = 96,
                    sessionSalt = (System.currentTimeMillis() % 100_000L) + 23L
                )
                precisionWinnerEval = null
                precisionBestGenome = withContext(Dispatchers.IO) {
                    PrecisionTargetEngine.loadSavedBestGenome(precisionStorageDir, mode)
                }
                appendPrecisionLog("Démarrage Précision [${mode.label}] sur ${cache.periodLabel} (${cache.steps.size} tirages)...")
            } else {
                appendPrecisionLog("Continuer Précision [${mode.label}] depuis Gén.${precisionEvoState?.generation ?: 0}...")
            }

            _precisionUiState.update { cur ->
                cur.copy(
                    phase = AutonomousPhase.ANALYZING,
                    honestStatusNote = "Recherche Précision [${mode.label}] en cours...",
                    liveLogs = synchronized(precisionLogs) { precisionLogs.toList() }
                )
            }

            val cacheDaysCount = cache.steps.map { it.dayEpoch }.distinct().size.coerceAtLeast(1)
            var lastPrecUiEmitMs = 0L
            var bestPrecFitness = Double.NEGATIVE_INFINITY
            while (currentCoroutineContext().isActive && _precisionUiState.value.phase == AutonomousPhase.ANALYZING) {
                val stBefore = precisionEvoState ?: break
                val res = withContext(Dispatchers.Default) {
                    PrecisionTargetEngine.executeEvolutionGeneration(
                        state = stBefore,
                        cache = cache,
                        minObjectiveStreak = if (cacheDaysCount <= 1) 1 else minOf(3, cacheDaysCount),
                        populationSize = 96
                    )
                }
                precisionEvoState = res.nextState

                val top = res.winningEval ?: res.auditedEval ?: res.bestTrainEval
                if (res.winningEval != null) {
                    precisionWinnerEval = res.winningEval
                    precisionBestGenome = res.winningEval.genome
                    bestPrecFitness = maxOf(bestPrecFitness, res.winningEval.fitnessScore)
                    withContext(Dispatchers.IO + NonCancellable) {
                        PrecisionTargetEngine.saveBestGenome(precisionStorageDir, res.nextState.mode, res.winningEval.genome)
                    }
                    appendPrecisionLog("✔ GAGNANT PRÉCISION VALIDÉ (${res.nextState.mode.label}) : ${res.winningEval.genome.describeCompact()}")
                } else {
                    top?.let {
                        if (precisionBestGenome == null ||
                            it.consolidatedStreak > _precisionUiState.value.bestValidatedStreak ||
                            it.fitnessScore > bestPrecFitness + 0.25
                        ) {
                            bestPrecFitness = maxOf(bestPrecFitness, it.fitnessScore)
                            precisionBestGenome = it.genome
                            withContext(Dispatchers.IO + NonCancellable) {
                                PrecisionTargetEngine.saveBestGenome(precisionStorageDir, res.nextState.mode, it.genome)
                            }
                        }
                    }
                    appendPrecisionLog(res.generationNote)
                }

                val infinite = _precisionUiState.value.infiniteMode
                val nextPhase = if (res.winningEval != null && !infinite) AutonomousPhase.VALIDATED else AutonomousPhase.ANALYZING
                val nowPrecMs = System.currentTimeMillis()
                if (res.winningEval != null ||
                    res.nextState.generation <= 5 ||
                    res.nextState.generation % 5 == 0 ||
                    nowPrecMs - lastPrecUiEmitMs >= 160L
                ) {
                    lastPrecUiEmitMs = nowPrecMs
                    _precisionUiState.update { cur ->
                        cur.copy(
                            phase = nextPhase,
                            generation = res.nextState.generation,
                            experimentsCount = res.nextState.experimentsCount,
                            confirmationTestsCount = res.nextState.confirmationTestsCount,
                            excludedCount = res.nextState.seenSignatures.size,
                            bestValidatedStreak = maxOf(cur.bestValidatedStreak, top?.consolidatedStreak ?: 0),
                            discoveryStats = top?.discStats ?: cur.discoveryStats,
                            rollingStats = top?.rollStats ?: cur.rollingStats,
                            confirmationStats = top?.confStats ?: cur.confirmationStats,
                            sealedStats = top?.sealedStats ?: cur.sealedStats,
                            activeFormulaDesc = top?.genome?.describeCompact() ?: cur.activeFormulaDesc,
                            pValueAdjusted = top?.pValueAdjusted ?: cur.pValueAdjusted,
                            honestStatusNote = res.generationNote,
                            liveLogs = synchronized(precisionLogs) { precisionLogs.toList() }
                        )
                    }
                }
                if (res.winningEval != null) {
                    launchPrecisionBacktestAndPrediction()
                    if (!infinite) break
                }
                yield()
            }
        }
    }

    fun suspendPrecisionDiscovery() {
        precisionJob?.cancel()
        precisionJob = null
        val g = precisionEvoState?.generation ?: 0
        appendPrecisionLog("Moteur Précision en pause (Gén.$g — mémoire conservée).")
        _precisionUiState.update { cur ->
            cur.copy(
                phase = AutonomousPhase.SUSPENDED,
                honestStatusNote = "En pause à Gén.$g — appuyez sur Continuer pour reprendre.",
                liveLogs = synchronized(precisionLogs) { precisionLogs.toList() }
            )
        }
    }

    fun resumePrecisionDiscovery() {
        val cur = _precisionUiState.value
        startPrecisionDiscovery(continueBeyond = precisionEvoState != null && cur.phase != AutonomousPhase.IDLE)
    }

    fun stopPrecisionDiscovery() {
        precisionJob?.cancel()
        precisionJob = null
        appendPrecisionLog("Moteur Précision fermé (Stop).")
        _precisionUiState.update { cur ->
            cur.copy(
                phase = AutonomousPhase.STOPPED,
                honestStatusNote = "Analyse Précision arrêtée proprement.",
                liveLogs = synchronized(precisionLogs) { precisionLogs.toList() }
            )
        }
    }

    fun restartPrecisionFromZero() {
        precisionJob?.cancel()
        precisionJob = null
        precisionEvoState = null
        precisionWinnerEval = null
        precisionBestGenome = null
        synchronized(precisionLogs) { precisionLogs.clear() }
        appendPrecisionLog("Remise à zéro complète du Moteur Précision.")
        _precisionUiState.update { cur ->
            PrecisionUiState(
                targetMode = cur.targetMode,
                phase = AutonomousPhase.IDLE,
                infiniteMode = cur.infiniteMode,
                liveLogs = synchronized(precisionLogs) { precisionLogs.toList() }
            )
        }
        startPrecisionDiscovery(continueBeyond = false)
    }

    fun launchPrecisionBacktestAndPrediction() {
        val mode = _precisionUiState.value.targetMode
        precisionBacktestJob?.cancel()
        precisionBacktestJob = coordinatorScope.launch {
            val currentData = when {
                latestHistorySnapshot.isNotEmpty() -> latestHistorySnapshot
                cachedChronologicalHistory.isNotEmpty() -> cachedChronologicalHistory
                else -> withContext(Dispatchers.IO) { lotoDao.getAllResultsOnce() }
            }
            if (currentData.size < 4) {
                _precisionUiState.update {
                    it.copy(honestStatusNote = "Pas assez de tirages pour le backtest Précision (${currentData.size}/4).")
                }
                return@launch
            }
            val genome = precisionWinnerEval?.genome
                ?: precisionBestGenome
                ?: precisionEvoState?.elites?.firstOrNull()?.genome
                ?: withContext(Dispatchers.IO) { PrecisionTargetEngine.loadSavedBestGenome(precisionStorageDir, mode) }

            if (genome == null) {
                _precisionUiState.update {
                    it.copy(honestStatusNote = "Lancez d'abord la recherche Précision [${mode.label}] pour obtenir une formule.")
                }
                return@launch
            }

            val period = _selectedPeriod.value
            val customDays = _customPeriodDays.value
            val nowMs = clockProvider()
            val anchorDayKey = LotoDateUtils.formatDayKey(nowMs)
            val expectedKey = "PREC|${period.name}|${customDays ?: "ALL"}|$anchorDayKey|${currentData.size}|${currentData.firstOrNull()?.uniqueId}|${currentData.lastOrNull()?.uniqueId}"
            val cache = if (precisionCausalCache != null && precisionCausalCacheKey == expectedKey) {
                precisionCausalCache!!
            } else {
                PrecisionTargetEngine.buildCausalCache(
                    history = currentData,
                    period = period,
                    referenceNowMs = nowMs,
                    customDays = customDays
                ).also {
                    precisionCausalCache = it
                    precisionCausalCacheKey = expectedKey
                }
            }

            val ledgerKey = PrecisionTargetEngine.buildLedgerKey(mode, genome.signatureKey(), cache.periodLabel)
            val existing = withContext(Dispatchers.IO) {
                PrecisionTargetEngine.loadLedger(precisionStorageDir, ledgerKey)
            }
            val (bt, updatedLedger) = withContext(Dispatchers.Default) {
                PrecisionTargetEngine.mergeAppendOnlyApprovedBacktest(
                    genome = genome,
                    cache = cache,
                    existingLedger = existing,
                    confirmationTestsCount = maxOf(1, _precisionUiState.value.confirmationTestsCount)
                )
            }
            val nextTarget = refreshNextDrawTarget(currentData, nowMs)
            val livePred = withContext(Dispatchers.Default) {
                PrecisionTargetEngine.predictNextUpcoming(
                    history = currentData,
                    genome = genome,
                    nextTarget = nextTarget
                )
            }
            val finalLedger = PrecisionTargetEngine.recordLivePredictionInLedger(updatedLedger, livePred)
            withContext(Dispatchers.IO + NonCancellable) {
                PrecisionTargetEngine.saveLedger(precisionStorageDir, finalLedger)
            }

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
                    honestStatusNote = "Prédiction ${mode.label} (${livePred.horaire}) : ${livePred.predictedNumbers.joinToString("-") { it.toString().padStart(2, '0') }}",
                    liveLogs = synchronized(precisionLogs) { precisionLogs.toList() }
                )
            }
        }
    }

    private fun appendPrecisionLog(line: String) {
        synchronized(precisionLogs) {
            if (precisionLogs.size >= 36) precisionLogs.removeFirst()
            precisionLogs.addLast("[${LotoDateUtils.formatShortDate(System.currentTimeMillis())}] $line")
        }
    }

    fun launchBacktestAndPrediction() {
        val currentState = _discoveryState.value
        val period = _selectedPeriod.value

        if (!currentState.canLaunchBacktest && validatedGenome == null) return

        backtestJob?.cancel()
        backtestJob = coordinatorScope.launch {
            _isBacktestBusy.value = true
            var genome = validatedGenome
            if (genome == null) {
                val fallbackEntity = withContext(Dispatchers.IO) {
                    val byId = activeWinnerId.takeIf { it.isNotBlank() }?.let { discoveryRepository.loadFormulaById(it) }
                    byId ?: discoveryRepository.loadValidatedFormulas(period, 1).firstOrNull()
                }
                if (fallbackEntity != null) {
                    genome = parseGenomeOrNull(fallbackEntity.genomeJson)
                    if (genome != null) {
                        validatedGenome = genome
                        activeWinnerId = fallbackEntity.id
                        bestValidatedStreak = maxOf(bestValidatedStreak, fallbackEntity.combinedStreak)
                    }
                }
            }
            val resolvedGenome = genome
            if (resolvedGenome == null) {
                _isBacktestBusy.value = false
                return@launch
            }

            val currentData = when {
                latestHistorySnapshot.isNotEmpty() -> latestHistorySnapshot
                cachedChronologicalHistory.isNotEmpty() -> cachedChronologicalHistory
                else -> withContext(Dispatchers.IO) { lotoDao.getAllResultsOnce() }
            }
            if (currentData.isEmpty()) {
                _isBacktestBusy.value = false
                return@launch
            }

            val nowMs = clockProvider()
            val cleanChronological = currentData
                .distinctBy { it.uniqueId }
                .sortedWith(compareBy<LotoResultEntity> { it.timestamp }.thenBy { it.uniqueId })
            cachedChronologicalHistory = cleanChronological
            val nextTarget = refreshNextDrawTarget(cleanChronological, nowMs)

            val anchorDayKey = LotoDateUtils.formatDayKey(nowMs)
            val customDays = _customPeriodDays.value
            val expectedKey = "${period.name}_${customDays ?: "ALL"}_${anchorDayKey}_${cleanChronological.size}_${cleanChronological.firstOrNull()?.uniqueId}_${cleanChronological.lastOrNull()?.uniqueId}"
            val cache = if (cachedCacheKey == expectedKey && cachedCausalCache != null) {
                cachedCausalCache!!
            } else {
                DiscoveryEngine.buildCausalCache(
                    history = cleanChronological,
                    period = period,
                    referenceNowMs = nowMs,
                    customDays = customDays
                ).also {
                    cachedCausalCache = it
                    cachedCacheKey = expectedKey
                }
            }
            val latestSnap = _discoveryState.value
            val strategySig = resolvedGenome.describeCompact()
            val ledgerKey = DiscoveryEngine.buildLedgerKey(strategySig, cache.periodLabel)
            val existingLedger = withContext(Dispatchers.IO) {
                discoveryRepository.loadApprovedBacktestLedger(ledgerKey)
            }
            val (bt, updatedLedger) = DiscoveryEngine.mergeAppendOnlyApprovedBacktest(
                genome = resolvedGenome,
                cache = cache,
                existingLedger = existingLedger,
                generationsExecuted = latestSnap.generation,
                experimentsCount = latestSnap.experimentsCount
            )
            val pred = DiscoveryEngine.predictNextWithGenome(
                genome = resolvedGenome,
                cleanHistory = cleanChronological,
                targetHoraire = nextTarget.horaire,
                validatedBacktest = bt,
                targetTimestamp = nextTarget.targetTimestamp
            )
            val finalLedger = DiscoveryEngine.recordLivePredictionInLedger(
                ledger = updatedLedger,
                prediction = pred,
                targetTimestamp = nextTarget.targetTimestamp
            )
            withContext(Dispatchers.IO + NonCancellable) {
                discoveryRepository.saveApprovedBacktestLedger(finalLedger)
            }
            approvedBacktestNote = if (bt.approvedStepsPreservedCount > 0) {
                "Backtest approuvé verrouillé : ${bt.approvedStepsPreservedCount} étapes intactes + ${bt.newlyAppendedStepsCount} nouveau(x) tirage(s) complété(s) (${bt.livePredictionsMatchedCount} Live vérifié(s))."
            } else {
                "Backtest approuvé verrouillé (${bt.totalTested} étapes en mode append-only ; les anciens résultats restent intacts)."
            }
            _backtestResult.value = bt
            _prediction.value = pred
            _discoveryState.update {
                it.copy(
                    backtestTriggered = true,
                    periodDateRangeLabel = cache.dateRangeLabel,
                    activeWinnerId = activeWinnerId,
                    bestValidatedStreak = bestValidatedStreak,
                    savedWinners = savedWinnersCache,
                    approvedBacktestNote = approvedBacktestNote,
                    infiniteMode = infiniteModeFlag,
                    chercheurStatus = currentChercheurStatus()
                )
            }
            _isBacktestBusy.value = false
        }
    }

    private suspend fun restoreInitialSession(expectedEpoch: Long) {
        val activeSession = withContext(Dispatchers.IO) {
            discoveryRepository.loadActiveAnalyzingSession()
        }
        if (actionEpoch.get() != expectedEpoch) return

        val targetPeriod = activeSession?.periodName?.let { name ->
            BacktestPeriod.entries.firstOrNull { it.name == name }
        } ?: BacktestPeriod.THREE_WEEKS
        if (_selectedPeriod.value != targetPeriod) {
            _selectedPeriod.value = targetPeriod
        }
        _customPeriodDays.value = targetPeriod.days
        restorePersistedSession(targetPeriod, autoResumeIfAnalyzing = true, expectedEpoch = expectedEpoch)
    }

    private suspend fun restorePersistedSession(
        period: BacktestPeriod,
        autoResumeIfAnalyzing: Boolean,
        expectedEpoch: Long
    ) {
        val (session, topEntities, rejectedEntities, validatedEntity, savedWinnerEntities) = withContext(Dispatchers.IO) {
            RestoreBundle(
                session = discoveryRepository.loadSession(period),
                top = discoveryRepository.loadTopFormulas(period, 6),
                rejected = discoveryRepository.loadRejectedFormulas(period, 8),
                validated = discoveryRepository.loadValidatedFormula(period),
                savedWinners = discoveryRepository.loadValidatedFormulas(period, 20)
            )
        }
        if (actionEpoch.get() != expectedEpoch) return
        if (session == null && topEntities.isEmpty() && validatedEntity == null && savedWinnerEntities.isEmpty()) return

        val restoredPop = session?.let { SymbolicFormula.genomeListFromJson(it.populationJson) }.orEmpty()
        val restoredLogs = session?.let { decodeLogs(it.logsJson) }.orEmpty()
        val effectiveValidatedEntity = savedWinnerEntities.firstOrNull() ?: validatedEntity
        val restoredValidated = (effectiveValidatedEntity?.genomeJson ?: session?.validatedGenomeJson)?.let { raw ->
            parseGenomeOrNull(raw)
        }
        val restoredActive = session?.activeGenomeJson?.takeIf { it.isNotBlank() }?.let { raw ->
            parseGenomeOrNull(raw)
        }

        val wasAnalyzing = session?.phaseName == AutonomousPhase.ANALYZING.name
        val restoredPhase = when {
            session?.phaseName == AutonomousPhase.VALIDATED.name && restoredValidated != null -> AutonomousPhase.VALIDATED
            wasAnalyzing -> if (autoResumeIfAnalyzing) AutonomousPhase.ANALYZING else AutonomousPhase.SUSPENDED
            session?.phaseName == AutonomousPhase.SUSPENDED.name -> AutonomousPhase.SUSPENDED
            session?.phaseName == AutonomousPhase.STOPPED.name -> AutonomousPhase.STOPPED
            restoredValidated != null && session == null -> AutonomousPhase.VALIDATED
            else -> AutonomousPhase.IDLE
        }

        synchronized(stateLock) {
            accumulatedLogs.clear()
            restoredLogs.forEach { accumulatedLogs.addLast(it) }
        }

        val checkpointEntity = withContext(Dispatchers.IO) {
            discoveryRepository.loadCheckpointBestFormula(period)
        }
        val restoredCheckpointGenome = checkpointEntity?.genomeJson?.takeIf { it.isNotBlank() }?.let { raw ->
            parseGenomeOrNull(raw)
        }
        val effectiveBestGenome = restoredValidated ?: restoredCheckpointGenome ?: restoredActive
        bestMilestoneGenome = effectiveBestGenome
        bestMilestoneCombinedStreak = maxOf(
            session?.bestCombinedStreak ?: 0,
            checkpointEntity?.combinedStreak ?: 0
        )
        bestMilestoneDiscStreak = maxOf(
            session?.bestDiscoveryStreak ?: 0,
            checkpointEntity?.discoveryStreak ?: 0
        )
        bestMilestoneRollStreak = maxOf(
            session?.bestRollingStreak ?: 0,
            checkpointEntity?.rollingStreak ?: 0
        )
        bestMilestoneConfStreak = maxOf(
            session?.bestConfirmationStreak ?: 0,
            checkpointEntity?.confirmationStreak ?: 0
        )
        bestMilestoneFitness = checkpointEntity?.fitnessScore ?: Double.NEGATIVE_INFINITY

        sessionSalt = session?.sessionSalt ?: sessionSalt
        validatedGenome = restoredValidated
        savedWinnersCache = savedWinnerEntities.map { it.toSummary() }
        bestValidatedStreak = savedWinnerEntities.maxOfOrNull { it.combinedStreak } ?: 0
        bestValidatedFitness = effectiveValidatedEntity?.fitnessScore ?: Double.NEGATIVE_INFINITY
        activeWinnerId = effectiveValidatedEntity?.id.orEmpty()
        userPinnedWinner = false
        continueBeyondMode = bestValidatedStreak >= 4 && wasAnalyzing
        targetObjectiveStreak = if (maxOf(bestValidatedStreak, bestMilestoneCombinedStreak) >= 4) {
            maxOf(bestValidatedStreak, bestMilestoneCombinedStreak) + 1
        } else {
            4
        }

        val bestSummaries = topEntities.map { it.toSummary() }
        val rejectedSummaries = rejectedEntities.map { it.toRejectedSummary() }

        val baseRestoredState = DiscoveryEngine.createInitialState(
            sessionSalt = sessionSalt,
            restoredPopulation = restoredPop
        ).copy(
            generation = session?.generation ?: 0,
            experimentsCount = session?.experimentsCount ?: 0,
            confirmationTestsCount = session?.confirmationTestsCount ?: 0,
            rejectedHistory = rejectedSummaries
        )
        evolutionState = DiscoveryEngine.ensureEliteInPopulation(baseRestoredState, effectiveBestGenome)

        _discoveryState.value = EngineActionState(
            phase = restoredPhase,
            generation = session?.generation ?: 0,
            experimentsCount = session?.experimentsCount ?: 0,
            confirmationTestsCount = session?.confirmationTestsCount ?: 0,
            currentPrimaryExpr = effectiveBestGenome?.describePrimary().orEmpty(),
            currentSecondaryExpr = effectiveBestGenome?.describeSecondary().orEmpty(),
            currentSlotSplit = effectiveBestGenome?.let { "${it.primarySlots}+${5 - it.primarySlots}" }.orEmpty(),
            bestDiscoveryStreak = bestMilestoneDiscStreak,
            bestRollingStreak = bestMilestoneRollStreak,
            bestConfirmationStreak = bestMilestoneConfStreak,
            bestSealedStreak = session?.bestSealedStreak ?: 0,
            bestCombinedStreak = maxOf(bestMilestoneCombinedStreak, bestValidatedStreak),
            discoveredFeaturesHistogram = DiscoveryEngine.computeFeatureHistogram(evolutionState.population),
            bestFormulas = bestSummaries,
            rejectedFormulas = rejectedSummaries,
            liveLogs = restoredLogs,
            honestStatusNote = session?.honestStatusNote.orEmpty(),
            targetObjectiveStreak = targetObjectiveStreak,
            bestValidatedStreak = bestValidatedStreak,
            activeWinnerId = activeWinnerId,
            savedWinners = savedWinnersCache,
            preferredStrategyDays = preferredStrategyDays
        )

        if (wasAnalyzing && autoResumeIfAnalyzing && discoveryJob?.isActive != true) {
            val savedDraws = withContext(Dispatchers.IO) {
                if (latestHistorySnapshot.isNotEmpty()) latestHistorySnapshot else lotoDao.getAllResultsOnce()
            }
            if (savedDraws.size >= 8 && actionEpoch.get() == expectedEpoch) {
                latestHistorySnapshot = savedDraws
                startContinuousDiscovery(reset = false)
            }
        }
    }

    private fun resolvePoolForTier(
        winners: List<SymbolicFormulaSummary>,
        days: Int?
    ): List<SymbolicFormulaSummary> {
        val targetDays = days ?: return winners
        val exact = winners.filter { it.combinedStreak == targetDays }
        return if (exact.isNotEmpty()) exact else winners.filter { it.combinedStreak >= targetDays }
    }

    private suspend fun promoteStrongestSavedWinner(
        fallbackEval: GenomeCandidateEvaluation? = null,
        fallbackEntityId: String = ""
    ) {
        val prevBest = bestValidatedStreak
        bestValidatedStreak = maxOf(prevBest, savedWinnersCache.maxOfOrNull { it.combinedStreak } ?: 0)
        val pool = resolvePoolForTier(savedWinnersCache, preferredStrategyDays)
        val topSaved = pool.firstOrNull() ?: return
        val currentActiveSummary = pool.firstOrNull { it.id == activeWinnerId }

        val shouldPromote = validatedGenome == null ||
            (preferredStrategyDays == null && topSaved.combinedStreak > prevBest) ||
            (!userPinnedWinner && (
                currentActiveSummary == null ||
                    topSaved.combinedStreak > currentActiveSummary.combinedStreak ||
                    (topSaved.combinedStreak == currentActiveSummary.combinedStreak &&
                        topSaved.fitnessScore >= currentActiveSummary.fitnessScore)
                ))

        if (!shouldPromote) return

        val resolvedGenome = if (topSaved.id == fallbackEntityId && fallbackEval != null) {
            fallbackEval.genome
        } else {
            val loaded = withContext(Dispatchers.IO + NonCancellable) {
                discoveryRepository.loadFormulaById(topSaved.id)
            }
            loaded?.let { parseGenomeOrNull(it.genomeJson) }
                ?: if (fallbackEval != null) fallbackEval.genome else null
        }

        if (resolvedGenome != null) {
            validatedGenome = resolvedGenome
            activeWinnerId = topSaved.id
            bestValidatedFitness = topSaved.fitnessScore
        }
    }

    private fun buildFourStageMetrics(
        eval: GenomeCandidateEvaluation,
        cache: FourStageCausalCache,
        fallbackAudit: GenomeCandidateEvaluation? = null
    ): FourStageValidationMetrics {
        val conf = eval.confStats ?: fallbackAudit?.confStats
        val sealed = eval.sealedStats ?: fallbackAudit?.sealedStats
        return FourStageValidationMetrics(
            discoveryDraws = cache.discoverySteps.size,
            discoveryStreak = eval.discStats.maxConsecutiveQualifyingDays,
            discoveryQualifyingDays = eval.discStats.qualifyingDays,
            discoveryTwoPlusCount = eval.discStats.twoPlusCount,
            discoveryAvgMatches = eval.discStats.avgMatches,
            rollingDraws = cache.rollingSteps.size,
            rollingStreak = eval.rollStats.maxConsecutiveQualifyingDays,
            rollingQualifyingDays = eval.rollStats.qualifyingDays,
            rollingTwoPlusCount = eval.rollStats.twoPlusCount,
            rollingAvgMatches = eval.rollStats.avgMatches,
            rollingStabilityPassed = eval.rollingStabilityPassed,
            rollingTwoPlusRate = eval.rollStats.twoPlusRate,
            confirmationDraws = cache.confirmationSteps.size,
            confirmationStreak = conf?.maxConsecutiveQualifyingDays ?: 0,
            confirmationQualifyingDays = conf?.qualifyingDays ?: 0,
            confirmationTwoPlusCount = conf?.twoPlusCount ?: 0,
            confirmationAvgMatches = conf?.avgMatches ?: 0f,
            sealedDraws = cache.sealedSteps.size,
            sealedStreak = sealed?.maxConsecutiveQualifyingDays ?: 0,
            sealedQualifyingDays = sealed?.qualifyingDays ?: 0,
            sealedTwoPlusCount = sealed?.twoPlusCount ?: 0,
            sealedAvgMatches = sealed?.avgMatches ?: 0f,
            sealedCausalVerified = sealed?.causalVerified ?: false,
            combinedStreak = maxOf(
                eval.discStats.maxConsecutiveQualifyingDays,
                eval.rollStats.maxConsecutiveQualifyingDays,
                conf?.maxConsecutiveQualifyingDays ?: 0
            ),
            combinedQualifyingDays = eval.rollStats.qualifyingDays + (conf?.qualifyingDays ?: 0),
            combinedMaxZeroStreak = conf?.maxZeroStreak ?: eval.rollStats.maxZeroStreak,
            combinedBrierScore = conf?.brierScore ?: eval.rollStats.brierScore,
            isFourStageValidated = eval.status == "VALIDATED" || fallbackAudit?.status == "VALIDATED"
        )
    }

    private fun persistSnapshotAsync(uiState: EngineActionState, period: BacktestPeriod) {
        coordinatorScope.launch(Dispatchers.IO + NonCancellable) {
            persistSessionBlocking(uiState, period)
        }
    }

    private fun resetMilestoneTracking() {
        bestMilestoneGenome = null
        bestMilestoneEval = null
        bestMilestoneCombinedStreak = 0
        bestMilestoneDiscStreak = 0
        bestMilestoneRollStreak = 0
        bestMilestoneConfStreak = 0
        bestMilestoneFitness = Double.NEGATIVE_INFINITY
    }

    private suspend fun resolveBestRecoveryGenome(period: BacktestPeriod): SymbolicGenome? {
        bestMilestoneGenome?.let { return it }
        return withContext(Dispatchers.IO + NonCancellable) {
            val checkpointRaw = discoveryRepository.loadCheckpointBestFormula(period)?.genomeJson
            val sessionRaw = discoveryRepository.loadSession(period)?.activeGenomeJson
            val topRaw = discoveryRepository.loadTopFormulas(period, 1).firstOrNull()?.genomeJson
            listOfNotNull(checkpointRaw, sessionRaw, topRaw)
                .firstOrNull { it.isNotBlank() }
                ?.let { raw ->
                    try {
                        SymbolicFormula.genomeFromJson(raw)
                    } catch (_: Exception) {
                        null
                    }
                }
        } ?: evolutionState.population.firstOrNull()
    }

    private suspend fun persistSessionBlocking(uiState: EngineActionState, period: BacktestPeriod) {
        val activeGenome = validatedGenome ?: bestMilestoneGenome ?: evolutionState.population.firstOrNull()
        val session = DiscoverySessionEntity(
            periodName = period.name,
            phaseName = uiState.phase.name,
            generation = evolutionState.generation,
            experimentsCount = evolutionState.experimentsCount,
            confirmationTestsCount = evolutionState.confirmationTestsCount,
            sessionSalt = sessionSalt,
            populationJson = SymbolicFormula.genomeListToJson(evolutionState.population),
            activeGenomeJson = activeGenome?.let { SymbolicFormula.genomeToJson(it) }.orEmpty(),
            validatedGenomeJson = validatedGenome?.let { SymbolicFormula.genomeToJson(it) },
            bestDiscoveryStreak = maxOf(uiState.bestDiscoveryStreak, bestMilestoneDiscStreak),
            bestRollingStreak = maxOf(uiState.bestRollingStreak, bestMilestoneRollStreak),
            bestConfirmationStreak = maxOf(uiState.bestConfirmationStreak, bestMilestoneConfStreak),
            bestSealedStreak = uiState.bestSealedStreak,
            bestCombinedStreak = maxOf(uiState.bestCombinedStreak, bestMilestoneCombinedStreak),
            honestStatusNote = uiState.honestStatusNote,
            logsJson = encodeLogs(uiState.liveLogs),
            updatedAt = System.currentTimeMillis()
        )
        withContext(Dispatchers.IO + NonCancellable) {
            discoveryRepository.saveSession(session)
        }
    }

    private fun appendLog(entry: String) {
        synchronized(stateLock) {
            if (accumulatedLogs.size >= MAX_LOG_ENTRIES) {
                accumulatedLogs.removeFirst()
            }
            accumulatedLogs.addLast(entry)
        }
    }

    private fun snapshotLogs(): List<String> = synchronized(stateLock) { accumulatedLogs.toList() }

    private fun encodeLogs(logs: List<String>): String =
        buildJsonArray { logs.forEach { add(it) } }.toString()

    private fun decodeLogs(raw: String): List<String> {
        if (raw.isBlank()) return emptyList()
        return try {
            Json.parseToJsonElement(raw).jsonArray.mapNotNull { it.jsonPrimitive.contentOrNull }
        } catch (_: Exception) {
            emptyList()
        }
    }

    private fun parseGenomeOrNull(raw: String): SymbolicGenome? = try {
        if (raw.isBlank()) null else SymbolicFormula.genomeFromJson(raw)
    } catch (_: Exception) {
        null
    }

    private data class RestoreBundle(
        val session: DiscoverySessionEntity?,
        val top: List<DiscoveredFormulaEntity>,
        val rejected: List<DiscoveredFormulaEntity>,
        val validated: DiscoveredFormulaEntity?,
        val savedWinners: List<DiscoveredFormulaEntity>
    )

    companion object {
        private const val MAX_LOG_ENTRIES = 12
        private val instances = WeakHashMap<Application, AutonomousDiscoveryCoordinator>()

        fun getInstance(application: Application): AutonomousDiscoveryCoordinator {
            synchronized(this) {
                return instances.getOrPut(application) {
                    AutonomousDiscoveryCoordinator(application)
                }
            }
        }
    }
}

private fun DiscoveredFormulaEntity.toSummary(): SymbolicFormulaSummary = SymbolicFormulaSummary(
    id = id,
    generation = generation,
    primaryExprText = primaryExprText,
    secondaryExprText = secondaryExprText,
    slotSplitLabel = slotSplitLabel,
    usedVariables = usedFeaturesCsv.split(",").filter { it.isNotBlank() },
    status = status,
    statusReason = rejectionReason.ifBlank { status },
    discoveryStreak = discoveryStreak,
    rollingStreak = rollingStreak,
    confirmationStreak = confirmationStreak,
    sealedStreak = sealedStreak,
    combinedStreak = combinedStreak,
    discoveryAvgMatches = discoveryAvgMatches,
    rollingAvgMatches = rollingAvgMatches,
    confirmationAvgMatches = confirmationAvgMatches,
    pValueAdjusted = pValueAdjusted,
    fitnessScore = fitnessScore,
    isValidated = status == "VALIDATED"
)

private fun DiscoveredFormulaEntity.toRejectedSummary(): RejectedFormulaSummary = RejectedFormulaSummary(
    id = id,
    generation = generation,
    formulaPreview = "[$slotSplitLabel] A=($primaryExprText) | B=($secondaryExprText)",
    rejectedAtStage = rejectionStage.ifBlank { "Validation" },
    reason = rejectionReason,
    trainAvgMatches = rollingAvgMatches.takeIf { it > 0f } ?: discoveryAvgMatches,
    validationAvgMatches = confirmationAvgMatches.takeIf { it > 0f } ?: rollingAvgMatches
)
