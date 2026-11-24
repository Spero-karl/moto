package com.example.ui.viewmodel

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.example.data.local.AppDatabase
import com.example.data.repository.DiscoveryRepository
import com.example.data.repository.LotoRepository
import com.example.domain.engine.BacktestPeriod
import com.example.domain.engine.LabSavedModelInput
import com.example.domain.engine.LearningEngine
import com.example.domain.engine.ScientificReport
import com.example.domain.engine.SymbolicFormula
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class LotoViewModel(application: Application) : AndroidViewModel(application) {

    private val db = AppDatabase.getDatabase(application)
    private val repository = LotoRepository(db.lotoDao())
    private val discoveryRepository = DiscoveryRepository(db.discoveryDao())
    private val engine = LearningEngine()

    val allResults = repository.allResults.stateIn(
        scope = viewModelScope,
        started = SharingStarted.WhileSubscribed(5000),
        initialValue = emptyList()
    )

    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    private val _isComparing = MutableStateFlow(false)
    val isComparing: StateFlow<Boolean> = _isComparing.asStateFlow()

    private val _errorMessage = MutableStateFlow<String?>(null)
    val errorMessage: StateFlow<String?> = _errorMessage.asStateFlow()

    private val _statusMessage = MutableStateFlow<String?>(null)
    val statusMessage: StateFlow<String?> = _statusMessage.asStateFlow()

    private val _scientificReport = MutableStateFlow<ScientificReport?>(null)
    val scientificReport: StateFlow<ScientificReport?> = _scientificReport.asStateFlow()

    init {
        refreshData()
    }

    fun refreshData() {
        viewModelScope.launch {
            _isLoading.value = true
            _errorMessage.value = null
            val result = repository.fetchLatestResults()
            result.onSuccess { summary ->
                if (summary.fetchedUniqueCount > 0) {
                    _statusMessage.value = "${summary.fetchedUniqueCount} tirages synchronisés (${summary.pagesLoaded} pages)"
                }
            }.onFailure { throwable ->
                _errorMessage.value = throwable.localizedMessage ?: "Erreur lors du chargement des résultats"
            }
            _isLoading.value = false
        }
    }

    fun importArchive(rawContent: String, fileName: String) {
        viewModelScope.launch {
            _isLoading.value = true
            val result = repository.importArchive(rawContent, fileName)
            result.onSuccess { res ->
                _statusMessage.value = if (res.acceptedCount > 0) {
                    "Archive importée : ${res.acceptedCount} tirages valides (${res.rejectedCount} rejetés)"
                } else {
                    "Aucun tirage valide trouvé (${res.errors.firstOrNull() ?: "format invalide"})"
                }
            }.onFailure { err ->
                _errorMessage.value = err.localizedMessage ?: "Échec de l'importation"
            }
            _isLoading.value = false
        }
    }

    fun clearError() {
        _errorMessage.value = null
    }

    fun clearStatus() {
        _statusMessage.value = null
    }

    fun runScientificComparison(
        period: BacktestPeriod = BacktestPeriod.THREE_WEEKS,
        customDays: Int? = period.days
    ) {
        val currentData = allResults.value
        if (currentData.size < 8) {
            _errorMessage.value = "Minimum 8 tirages requis pour l'évaluation scientifique."
            return
        }
        viewModelScope.launch {
            _isComparing.value = true
            val savedModels = withContext(Dispatchers.IO) {
                val periodsToCheck = (listOf(period) + BacktestPeriod.entries).distinct()
                val seenSigs = HashSet<String>()
                val out = ArrayList<LabSavedModelInput>()
                for (p in periodsToCheck) {
                    val entities = discoveryRepository.loadValidatedFormulas(p, 20)
                    for (e in entities) {
                        if (e.id.startsWith("checkpoint_best_")) continue
                        val genome = runCatching { SymbolicFormula.genomeFromJson(e.genomeJson) }.getOrNull() ?: continue
                        val sig = genome.describeCompact()
                        if (!seenSigs.add(sig)) continue
                        val label = "Modèle ${e.combinedStreak}J (Gén.${e.generation})"
                        out.add(
                            LabSavedModelInput(
                                id = e.id,
                                displayName = label,
                                savedStreak = e.combinedStreak,
                                slotSplitLabel = e.slotSplitLabel,
                                genome = genome
                            )
                        )
                    }
                }
                out.sortedByDescending { it.savedStreak }.take(12)
            }
            val report = withContext(Dispatchers.Default) {
                engine.runScientificComparison(
                    data = currentData,
                    period = period,
                    customDays = customDays,
                    referenceNowMs = System.currentTimeMillis(),
                    savedModels = savedModels
                )
            }
            _scientificReport.value = report
            _isComparing.value = false
        }
    }
}
