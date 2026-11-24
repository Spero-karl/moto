package com.example.ui.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.data.local.AppDatabase
import com.example.data.local.LotoResultEntity
import com.example.data.repository.LotoRepository
import com.example.domain.engine.BacktestPeriod
import com.example.domain.engine.BacktestResult
import com.example.domain.engine.EngineActionState
import com.example.domain.engine.NextDrawTarget
import com.example.domain.engine.PredictionMethod
import com.example.domain.engine.PredictionResult
import com.example.domain.engine.PrecisionTargetMode
import com.example.domain.engine.PrecisionUiState
import com.example.domain.util.StrategyBackupExport
import com.example.domain.util.StrategyBackupImportResult
import com.example.service.AutonomousDiscoveryCoordinator
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn

class DiscoveryViewModel(application: Application) : AndroidViewModel(application) {

    private val coordinator = AutonomousDiscoveryCoordinator.getInstance(application)
    private val repository = LotoRepository(AppDatabase.getDatabase(application).lotoDao())

    val allResults = repository.allResults.stateIn(
        scope = viewModelScope,
        started = SharingStarted.WhileSubscribed(5000),
        initialValue = emptyList()
    )

    val selectedMethod: StateFlow<PredictionMethod> = coordinator.selectedMethod
    val selectedPeriod: StateFlow<BacktestPeriod> = coordinator.selectedPeriod
    val customPeriodDays: StateFlow<Int?> = coordinator.customPeriodDays
    val nextDrawTarget: StateFlow<NextDrawTarget> = coordinator.nextDrawTarget
    val selectedHoraire: StateFlow<String> = coordinator.selectedHoraire
    val availableHoraires: StateFlow<List<String>> = coordinator.availableHoraires
    val isBacktestBusy: StateFlow<Boolean> = coordinator.isBacktestBusy
    val discoveryState: StateFlow<EngineActionState> = coordinator.discoveryState
    val prediction: StateFlow<PredictionResult?> = coordinator.prediction
    val backtestResult: StateFlow<BacktestResult?> = coordinator.backtestResult
    val precisionUiState: StateFlow<PrecisionUiState> = coordinator.precisionUiState

    internal fun injectHistoryForTest(results: List<LotoResultEntity>, nowMs: Long = coordinator.clockProvider()) {
        coordinator.injectHistoryForTest(results, nowMs)
    }

    fun refreshNextDrawTarget() {
        coordinator.refreshNextDrawTarget()
    }

    fun selectFocusAxis(method: PredictionMethod) {
        coordinator.selectFocusAxis(method)
    }

    fun selectPeriod(period: BacktestPeriod) {
        coordinator.selectPeriod(period)
    }

    fun selectCustomPeriodDays(days: Int) {
        coordinator.selectCustomPeriodDays(days)
    }

    fun selectHoraire(horaire: String) {
        coordinator.selectHoraire(horaire)
    }

    fun suspendDiscovery() {
        coordinator.suspendDiscovery()
    }

    fun resumeDiscovery() {
        coordinator.resumeDiscovery()
    }

    fun continueBeyondObjective() {
        coordinator.continueBeyondObjective()
    }

    fun selectSavedWinner(winnerId: String) {
        coordinator.selectSavedWinner(winnerId)
    }

    fun selectPreferredStrategyDays(days: Int?) {
        coordinator.selectPreferredStrategyDays(days)
    }

    fun stopDiscovery() {
        coordinator.stopDiscovery()
    }

    fun startContinuousDiscovery(reset: Boolean = true) {
        coordinator.startContinuousDiscovery(reset = reset)
    }

    fun setInfiniteMode(enabled: Boolean) {
        coordinator.setInfiniteMode(enabled)
    }

    fun startInfiniteMode() {
        coordinator.startInfiniteMode()
    }

    fun restartDiscoveryFromZero() {
        coordinator.restartDiscoveryFromZero()
    }

    fun runChercheurAndInject() {
        coordinator.runChercheurAndInject()
    }

    fun launchBacktestAndPrediction() {
        coordinator.launchBacktestAndPrediction()
    }

    fun setPrecisionTargetMode(mode: PrecisionTargetMode) {
        coordinator.setPrecisionTargetMode(mode)
    }

    fun setPrecisionInfiniteMode(enabled: Boolean) {
        coordinator.setPrecisionInfiniteMode(enabled)
    }

    fun startPrecisionDiscovery(continueBeyond: Boolean = false) {
        coordinator.startPrecisionDiscovery(continueBeyond)
    }

    fun suspendPrecisionDiscovery() {
        coordinator.suspendPrecisionDiscovery()
    }

    fun resumePrecisionDiscovery() {
        coordinator.resumePrecisionDiscovery()
    }

    fun stopPrecisionDiscovery() {
        coordinator.stopPrecisionDiscovery()
    }

    fun restartPrecisionFromZero() {
        coordinator.restartPrecisionFromZero()
    }

    fun launchPrecisionBacktestAndPrediction() {
        coordinator.launchPrecisionBacktestAndPrediction()
    }

    suspend fun prepareStrategiesBackupExport(): Result<StrategyBackupExport> {
        return coordinator.buildSavedStrategiesBackup()
    }

    suspend fun importStrategiesBackup(rawPayload: String): Result<StrategyBackupImportResult> {
        return coordinator.importSavedStrategiesBackup(rawPayload)
    }

    override fun onCleared() {
        // Ne jamais suspendre ou arreter le moteur ici : il vit dans AutonomousDiscoveryCoordinator
        // et DiscoveryForegroundService pour tourner indefiniment meme hors de l'application.
        super.onCleared()
    }
}
