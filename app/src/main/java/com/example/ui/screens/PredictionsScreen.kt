package com.example.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.DateRange
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.minimumInteractiveComponentSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.example.domain.engine.AutonomousPhase
import com.example.domain.engine.BacktestPeriod
import com.example.domain.engine.BacktestResult
import com.example.domain.engine.BacktestStepResult
import com.example.domain.engine.ChercheurStatusSummary
import com.example.domain.engine.DailyBacktestSummary
import com.example.domain.engine.EngineActionState
import com.example.domain.engine.FourStageValidationMetrics
import com.example.domain.engine.HorairePerformance
import com.example.domain.engine.NextDrawTarget
import com.example.domain.engine.PredictionMethod
import com.example.domain.engine.PredictionResult
import com.example.domain.engine.PrecisionBacktestResult
import com.example.domain.engine.PrecisionLivePrediction
import com.example.domain.engine.PrecisionTargetMode
import com.example.domain.engine.PrecisionUiState
import com.example.domain.engine.RandomReferenceComparison
import com.example.domain.engine.RejectedFormulaSummary
import com.example.domain.engine.SymbolicFormulaSummary
import com.example.domain.util.StrategyBackupExport
import com.example.domain.util.StrategyBackupImportResult
import com.example.ui.components.BacktestExportActions
import com.example.ui.components.BacktestStrategySelectorCard
import com.example.ui.components.FeaturedLotoBall
import com.example.ui.components.LotoBall
import com.example.ui.components.StrategyBackupActionsCard
import com.example.ui.components.formatLotoNumber
import java.util.Locale

private val CardShape20 = RoundedCornerShape(20.dp)
private val CardShape16 = RoundedCornerShape(16.dp)
private val CardShape12 = RoundedCornerShape(12.dp)
private val PillShape = RoundedCornerShape(10.dp)

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun PredictionsScreen(
    engineState: EngineActionState,
    prediction: PredictionResult?,
    backtestResult: BacktestResult?,
    selectedMethod: PredictionMethod,
    selectedPeriod: BacktestPeriod,
    selectedHoraire: String,
    availableHoraires: List<String>,
    resultsCount: Int,
    isLoading: Boolean,
    onSelectPeriod: (BacktestPeriod) -> Unit,
    onSelectHoraire: (String) -> Unit,
    onStartAnalysis: () -> Unit,
    onSuspendAnalysis: () -> Unit,
    onResumeAnalysis: () -> Unit,
    onStopAnalysis: () -> Unit,
    onLaunchBacktestAndPrediction: () -> Unit,
    onContinueBeyondObjective: () -> Unit = {},
    onSelectSavedWinner: (String) -> Unit = {},
    onSelectPreferredStrategyDays: (Int?) -> Unit = {},
    onOpenEngineOutput: () -> Unit = {},
    onExportStrategiesBackup: (suspend () -> Result<StrategyBackupExport>)? = null,
    onImportStrategiesBackup: (suspend (String) -> Result<StrategyBackupImportResult>)? = null,
    nextDrawTarget: NextDrawTarget = NextDrawTarget(horaire = selectedHoraire),
    customPeriodDays: Int? = selectedPeriod.days,
    onSelectCustomPeriodDays: (Int) -> Unit = {},
    onStartInfiniteMode: () -> Unit = onContinueBeyondObjective,
    onRestartFromZero: () -> Unit = onStartAnalysis,
    onSetInfiniteMode: (Boolean) -> Unit = {},
    onRunChercheur: () -> Unit = {},
    precisionState: PrecisionUiState = PrecisionUiState(),
    onSetPrecisionTargetMode: (PrecisionTargetMode) -> Unit = {},
    onStartPrecisionDiscovery: () -> Unit = {},
    onSuspendPrecisionDiscovery: () -> Unit = {},
    onResumePrecisionDiscovery: () -> Unit = {},
    onStopPrecisionDiscovery: () -> Unit = {},
    onRestartPrecisionFromZero: () -> Unit = {},
    onSetPrecisionInfiniteMode: (Boolean) -> Unit = {},
    onLaunchPrecisionBacktest: () -> Unit = {},
    modifier: Modifier = Modifier
) {
    val listState = rememberLazyListState()
    var selectedFilter by rememberSaveable { mutableIntStateOf(0) }
    var showAdvancedTechnicalCards by rememberSaveable { mutableStateOf(false) }

    val showResults = engineState.canLaunchBacktest &&
        engineState.backtestTriggered &&
        prediction != null &&
        backtestResult != null

    val filteredSteps = remember(backtestResult, selectedFilter, showResults) {
        if (!showResults) {
            emptyList()
        } else {
            val steps = backtestResult?.steps.orEmpty()
            when (selectedFilter) {
                1 -> steps.filter { it.matchCount >= 1 }
                2 -> steps.filter { it.matchCount >= 2 }
                else -> steps
            }
        }
    }

    LazyColumn(
        state = listState,
        modifier = modifier
            .fillMaxSize()
            .testTag("predictions_list"),
        contentPadding = PaddingValues(horizontal = 14.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        item(key = "three_weeks_scope", contentType = "selector") {
            val activeDays = customPeriodDays ?: selectedPeriod.days
            val activeWindowLabel = activeDays?.let { "$it jours" } ?: selectedPeriod.label
            val rangeDesc = engineState.periodDateRangeLabel.takeIf { it.isNotBlank() }
                ?: activeDays?.let { "$it j jusqu'au jour J" }
                ?: "Historique complet"
            var draftCustomDays by rememberSaveable(activeDays) {
                mutableIntStateOf((activeDays ?: 21).coerceIn(1, 3650))
            }
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("three_weeks_window_card"),
                shape = CardShape16,
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.35f))
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 6.dp),
                    verticalArrangement = Arrangement.spacedBy(2.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.DateRange,
                            contentDescription = "Fenêtre de travail configurable",
                            tint = MaterialTheme.colorScheme.primary,
                            modifier = Modifier.size(18.dp)
                        )
                        Text(
                            text = "Fenêtre : $activeWindowLabel",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "• $rangeDesc ($resultsCount tir.)",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = Modifier.weight(1f)
                        )
                    }

                    LazyRow(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        items(BacktestPeriod.entries, key = { it.name }) { preset ->
                            val chipLabel = preset.days?.let { "${it}J" } ?: "Tout"
                            val isSelected = selectedPeriod == preset && customPeriodDays == preset.days
                            FilterChip(
                                selected = isSelected,
                                onClick = { onSelectPeriod(preset) },
                                enabled = !isLoading,
                                label = { Text(chipLabel) },
                                modifier = Modifier.testTag("working_period_chip_${preset.name}")
                            )
                        }
                    }

                    FlowRow(
                        modifier = Modifier
                            .fillMaxWidth()
                            .clip(CardShape12)
                            .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.72f))
                            .padding(horizontal = 8.dp, vertical = 4.dp),
                        verticalArrangement = Arrangement.spacedBy(6.dp),
                        horizontalArrangement = Arrangement.spacedBy(8.dp)
                    ) {
                        Text(
                            text = "Libre (≥1J) :",
                            style = MaterialTheme.typography.labelSmall,
                            fontWeight = FontWeight.SemiBold,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.align(Alignment.CenterVertically)
                        )
                        Row(
                            verticalAlignment = Alignment.CenterVertically,
                            horizontalArrangement = Arrangement.spacedBy(6.dp)
                        ) {
                            OutlinedButton(
                                onClick = { if (draftCustomDays > 1) draftCustomDays -= 1 },
                                enabled = !isLoading && draftCustomDays > 1,
                                contentPadding = PaddingValues(0.dp),
                                modifier = Modifier
                                    .minimumInteractiveComponentSize()
                                    .size(48.dp)
                                    .testTag("working_period_minus")
                            ) {
                                Text(
                                    text = "−",
                                    style = MaterialTheme.typography.titleMedium,
                                    fontWeight = FontWeight.Bold
                                )
                            }
                            Text(
                                text = "${draftCustomDays}J",
                                style = MaterialTheme.typography.labelLarge,
                                fontWeight = FontWeight.ExtraBold,
                                modifier = Modifier.testTag("working_period_value")
                            )
                            OutlinedButton(
                                onClick = { if (draftCustomDays < 3650) draftCustomDays += 1 },
                                enabled = !isLoading && draftCustomDays < 3650,
                                contentPadding = PaddingValues(0.dp),
                                modifier = Modifier
                                    .minimumInteractiveComponentSize()
                                    .size(48.dp)
                                    .testTag("working_period_plus")
                            ) {
                                Icon(
                                    imageVector = Icons.Default.Add,
                                    contentDescription = "Augmenter le nombre de jours",
                                    modifier = Modifier.size(18.dp)
                                )
                            }
                        }
                        FilledTonalButton(
                            onClick = { onSelectCustomPeriodDays(draftCustomDays) },
                            enabled = !isLoading && customPeriodDays != draftCustomDays,
                            modifier = Modifier
                                .heightIn(min = 48.dp)
                                .testTag("working_period_apply")
                        ) {
                            Text("Appliquer")
                        }
                    }
                }
            }
        }

        item(key = "next_draw_only_badge", contentType = "selector") {
            val targetHoraireLabel = nextDrawTarget.horaire.ifBlank { selectedHoraire }
            val whenLabel = if (nextDrawTarget.isTomorrow) "Demain" else "Aujourd'hui"
            val beninTimeText = nextDrawTarget.currentBeninTimeLabel.takeIf { it.isNotBlank() }?.let { " • $it Bénin" } ?: ""
            val dateSuffix = nextDrawTarget.dayLabel.takeIf { it.isNotBlank() }?.let { " ($it)" } ?: ""
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .semantics(mergeDescendants = true) {}
                    .testTag("next_draw_horaire_badge"),
                shape = CardShape16,
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer),
                border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.4f))
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 12.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Lock,
                        contentDescription = "Prochain tirage uniquement",
                        tint = MaterialTheme.colorScheme.primary
                    )
                    Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                        Text(
                            text = "Prochain tirage ciblé uniquement : $targetHoraireLabel",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.ExtraBold,
                            color = MaterialTheme.colorScheme.onSecondaryContainer
                        )
                        Text(
                            text = "$whenLabel$dateSuffix$beninTimeText • Basé sur l'heure actuelle",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.85f)
                        )
                    }
                }
            }
        }

        item(key = "engine_action_hero", contentType = "engine_hero") {
            AutonomousDiscoveryHeroCard(
                engineState = engineState,
                selectedMethod = selectedMethod,
                resultsCount = resultsCount,
                isLoadingData = isLoading,
                onStartAnalysis = onStartAnalysis,
                onSuspendAnalysis = onSuspendAnalysis,
                onResumeAnalysis = onResumeAnalysis,
                onContinueBeyondObjective = onContinueBeyondObjective,
                onStopAnalysis = onStopAnalysis,
                onLaunchBacktestAndPrediction = onLaunchBacktestAndPrediction,
                onStartInfiniteMode = onStartInfiniteMode,
                onRestartFromZero = onRestartFromZero,
                onSetInfiniteMode = onSetInfiniteMode
            )
        }

        val patternLength = engineState.selectedPatternLength
        val matchedDaysCount = engineState.matchedDaysCount(backtestResult)

        if (showResults) {
            val unlockedPrediction = prediction!!
            item(key = "prediction_header", contentType = "header") {
                Text(
                    text = "Prochain Tirage Uniquement (${unlockedPrediction.horaire})",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.tertiary
                )
            }

            item(key = "prediction_card", contentType = "prediction") {
                UnlockedSymbolicPredictionCard(
                    prediction = unlockedPrediction,
                    matchedDays = matchedDaysCount,
                    patternLength = patternLength,
                    approvedNote = engineState.approvedBacktestNote,
                    onOpenEngineOutput = onOpenEngineOutput
                )
            }
        }

        item(key = "chercheur_card", contentType = "chercheur") {
            LeChercheurCard(
                status = engineState.chercheurStatus,
                isBusy = isLoading || resultsCount < 8,
                onRunChercheur = onRunChercheur
            )
        }

        item(key = "precision_target_engine", contentType = "precision_engine") {
            PrecisionTargetEngineCard(
                state = precisionState,
                resultsCount = resultsCount,
                isBusy = isLoading,
                onSelectMode = onSetPrecisionTargetMode,
                onStart = onStartPrecisionDiscovery,
                onPause = onSuspendPrecisionDiscovery,
                onResume = onResumePrecisionDiscovery,
                onStop = onStopPrecisionDiscovery,
                onRestartZero = onRestartPrecisionFromZero,
                onToggleInfinite = onSetPrecisionInfiniteMode,
                onLaunchBacktest = onLaunchPrecisionBacktest
            )
        }

        item(key = "strategy_selector", contentType = "strategy_selector") {
            BacktestStrategySelectorCard(
                engineState = engineState,
                backtestResult = backtestResult,
                isBusy = isLoading,
                onSelectPreferredDays = onSelectPreferredStrategyDays,
                onSelectWinner = onSelectSavedWinner
            )
        }

        if (onExportStrategiesBackup != null && onImportStrategiesBackup != null) {
            item(key = "strategy_backup_actions", contentType = "strategy_backup") {
                StrategyBackupActionsCard(
                    engineState = engineState,
                    isBusy = isLoading,
                    onExportStrategiesBackup = onExportStrategiesBackup,
                    onImportStrategiesBackup = onImportStrategiesBackup
                )
            }
        }

        if (engineState.rejectedFormulas.isNotEmpty()) {
            item(key = "rejected_formulas", contentType = "rejected") {
                RejectedFormulasCard(rejected = engineState.rejectedFormulas)
            }
        }

        if (engineState.fourStageMetrics != null ||
            engineState.discoveredFeaturesHistogram.isNotEmpty() ||
            engineState.bestFormulas.isNotEmpty() ||
            engineState.liveLogs.isNotEmpty()
        ) {
            item(key = "advanced_details_toggle", contentType = "toggle") {
                OutlinedButton(
                    onClick = { showAdvancedTechnicalCards = !showAdvancedTechnicalCards },
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 48.dp)
                        .testTag("toggle_advanced_details_button")
                ) {
                    Text(
                        if (showAdvancedTechnicalCards) {
                            "Masquer Détails Techniques (4 étapes, variables, top formules, logs)"
                        } else {
                            "Afficher Détails Techniques (4 étapes, variables, top formules, logs)"
                        }
                    )
                }
            }
        }

        if (showAdvancedTechnicalCards || engineState.isAnalyzing) {
            if (engineState.fourStageMetrics != null) {
                item(key = "four_stage_validation", contentType = "validation") {
                    FourStageValidationCard(
                        metrics = engineState.fourStageMetrics,
                        randomRef = engineState.randomReference,
                        confirmationTests = engineState.confirmationTestsCount
                    )
                }
            }

            if (engineState.discoveredFeaturesHistogram.isNotEmpty()) {
                item(key = "discovered_features", contentType = "traits") {
                    DiscoveredFeaturesCard(features = engineState.discoveredFeaturesHistogram)
                }
            }

            if (engineState.bestFormulas.isNotEmpty()) {
                item(key = "best_formulas", contentType = "discovery") {
                    BestSymbolicFormulasCard(
                        entries = engineState.bestFormulas,
                        generation = engineState.generation,
                        experimentsCount = engineState.experimentsCount
                    )
                }
            }

            if (engineState.liveLogs.isNotEmpty()) {
                item(key = "engine_live_logs", contentType = "logs") {
                    LiveEngineLogsCard(logs = engineState.liveLogs)
                }
            }
        }

        if (showResults) {
            val unlockedBacktest = backtestResult!!

            item(key = "divider_bt", contentType = "divider") {
                HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp))
            }

            item(key = "security_badge", contentType = "security") {
                SecurityVerificationCard(unlockedBacktest)
            }

            item(key = "backtest_summary", contentType = "summary") {
                BacktestSummaryCard(
                    bt = unlockedBacktest,
                    patternLength = patternLength
                )
            }

            item(key = "backtest_export_actions", contentType = "export") {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = CardShape16,
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
                    border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.35f))
                ) {
                    BacktestExportActions(
                        backtestResult = unlockedBacktest,
                        engineState = engineState,
                        prediction = prediction,
                        nextDrawTarget = nextDrawTarget
                    )
                }
            }

            if (unlockedBacktest.horaireBreakdown.isNotEmpty()) {
                item(key = "horaire_breakdown", contentType = "horaire") {
                    HoraireBreakdownCard(unlockedBacktest.horaireBreakdown)
                }
            }

            if (unlockedBacktest.dailySummaries.isNotEmpty()) {
                item(key = "daily_qualification", contentType = "daily") {
                    DailyQualificationCard(
                        dailySummaries = unlockedBacktest.dailySummaries,
                        matchedDays = unlockedBacktest.maxConsecutiveQualifyingDays,
                        patternLength = patternLength
                    )
                }
            }

            item(key = "backtest_filters", contentType = "filters") {
                Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text(
                        text = "Backtest Validé (Tirage après Tirage)",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                    FlowRow(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalArrangement = Arrangement.spacedBy(6.dp)
                    ) {
                        FilterChip(
                            selected = selectedFilter == 0,
                            onClick = { selectedFilter = 0 },
                            label = { Text("Tous (${unlockedBacktest.steps.size})") },
                            modifier = Modifier.testTag("filter_all")
                        )
                        FilterChip(
                            selected = selectedFilter == 1,
                            onClick = { selectedFilter = 1 },
                            label = { Text("≥ 1 bon (${unlockedBacktest.partialMatches})") },
                            modifier = Modifier.testTag("filter_matches")
                        )
                        FilterChip(
                            selected = selectedFilter == 2,
                            onClick = { selectedFilter = 2 },
                            label = { Text("2+ bons (${unlockedBacktest.drawsWithTwoPlusMatches})") },
                            modifier = Modifier.testTag("filter_two_plus")
                        )
                    }
                }
            }

            if (filteredSteps.isEmpty()) {
                item(key = "empty_filter", contentType = "empty") {
                    Text(
                        text = "Aucun tirage dans cette catégorie.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(vertical = 12.dp)
                    )
                }
            } else {
                items(
                    items = filteredSteps,
                    key = { it.drawId },
                    contentType = { "backtest_step" }
                ) { step ->
                    BacktestStepCard(step)
                }
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun AutonomousDiscoveryHeroCard(
    engineState: EngineActionState,
    selectedMethod: PredictionMethod,
    resultsCount: Int,
    isLoadingData: Boolean,
    onStartAnalysis: () -> Unit,
    onSuspendAnalysis: () -> Unit,
    onResumeAnalysis: () -> Unit,
    onContinueBeyondObjective: () -> Unit,
    onStopAnalysis: () -> Unit,
    onLaunchBacktestAndPrediction: () -> Unit,
    onStartInfiniteMode: () -> Unit,
    onRestartFromZero: () -> Unit,
    onSetInfiniteMode: (Boolean) -> Unit
) {
    val pulseTransition = rememberInfiniteTransition(label = "engine_pulse")
    val pulseAlpha by pulseTransition.animateFloat(
        initialValue = 0.35f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(750, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulse_alpha"
    )

    val accentBorder = when (engineState.phase) {
        AutonomousPhase.VALIDATED -> MaterialTheme.colorScheme.tertiary
        AutonomousPhase.ANALYZING -> MaterialTheme.colorScheme.primary
        AutonomousPhase.SUSPENDED -> MaterialTheme.colorScheme.secondary
        AutonomousPhase.STOPPED -> MaterialTheme.colorScheme.outline
        AutonomousPhase.IDLE -> MaterialTheme.colorScheme.outline.copy(alpha = 0.45f)
    }

    val headerBrush = Brush.linearGradient(
        colors = listOf(
            MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.65f),
            MaterialTheme.colorScheme.surfaceVariant
        )
    )

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .animateContentSize()
            .testTag("engine_action_card"),
        shape = CardShape20,
        border = BorderStroke(1.5.dp, accentBorder),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
    ) {
        Column(
            modifier = Modifier
                .background(headerBrush)
                .padding(horizontal = 14.dp, vertical = 10.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Moteur de Découverte Symbolique",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.ExtraBold
                    )
                    Text(
                        text = if (engineState.discoveryDrawsCount > 0) {
                            "Déc(${engineState.discoveryDrawsCount}) → Gliss(${engineState.rollingDrawsCount}) → Conf(${engineState.confirmationDrawsCount}) → Scellé(${engineState.sealedDrawsCount})"
                        } else {
                            "Programmation génétique libre sur variables causales brutes ($resultsCount tirages)"
                        },
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                val statusLabel = when (engineState.phase) {
                    AutonomousPhase.IDLE -> "PRÊT"
                    AutonomousPhase.ANALYZING -> if (engineState.infiniteMode) "INFINI ACTIF" else "RECHERCHE"
                    AutonomousPhase.SUSPENDED -> "EN PAUSE"
                    AutonomousPhase.STOPPED -> "ARRÊTÉ"
                    AutonomousPhase.VALIDATED -> "VALIDÉ 4 NIVEAUX"
                }
                val pillBg = when (engineState.phase) {
                    AutonomousPhase.VALIDATED -> MaterialTheme.colorScheme.tertiaryContainer
                    AutonomousPhase.ANALYZING -> MaterialTheme.colorScheme.primaryContainer
                    AutonomousPhase.SUSPENDED -> MaterialTheme.colorScheme.secondaryContainer
                    else -> MaterialTheme.colorScheme.surface
                }
                val pillFg = when (engineState.phase) {
                    AutonomousPhase.VALIDATED -> MaterialTheme.colorScheme.onTertiaryContainer
                    AutonomousPhase.ANALYZING -> MaterialTheme.colorScheme.onPrimaryContainer
                    AutonomousPhase.SUSPENDED -> MaterialTheme.colorScheme.onSecondaryContainer
                    else -> MaterialTheme.colorScheme.onSurfaceVariant
                }
                Row(
                    modifier = Modifier
                        .clip(PillShape)
                        .background(pillBg)
                        .padding(horizontal = 10.dp, vertical = 5.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    if (engineState.phase == AutonomousPhase.ANALYZING) {
                        Box(
                            modifier = Modifier
                                .size(8.dp)
                                .alpha(pulseAlpha)
                                .clip(CircleShape)
                                .background(MaterialTheme.colorScheme.onPrimaryContainer)
                        )
                    }
                    Text(
                        text = statusLabel,
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.ExtraBold,
                        color = pillFg,
                        maxLines = 1
                    )
                }
            }

            if (engineState.phase == AutonomousPhase.ANALYZING) {
                LinearProgressIndicator(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(6.dp)
                        .clip(PillShape)
                )
                Text(
                    text = if (engineState.infiniteMode) {
                        "Mode Infini actif : cherche de meilleurs paliers sans arrêt jusqu'à Pause ou Stop."
                    } else {
                        "Service arrière-plan actif : recherche continue jusqu'à validation."
                    },
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.primary
                )
            }

            if (engineState.generation > 0 || engineState.phase != AutonomousPhase.IDLE) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    MetricItem(
                        label = "Évolution GP",
                        value = "Gén. ${engineState.generation}",
                        sub = "${engineState.experimentsCount} exp.",
                        modifier = Modifier.weight(1f)
                    )
                    MetricItem(
                        label = "Tests Hors-Éch.",
                        value = "${engineState.confirmationTestsCount} audits",
                        sub = "p=${String.format(Locale.FRANCE, "%.3f", engineState.randomReference.pValueAdjusted)}",
                        modifier = Modifier.weight(1f)
                    )
                    MetricItem(
                        label = "Série Objective",
                        value = "${engineState.bestCombinedStreak} / ${engineState.targetObjectiveStreak} j",
                        sub = "D:${engineState.bestDiscoveryStreak} G:${engineState.bestRollingStreak} C:${engineState.bestConfirmationStreak}",
                        modifier = Modifier.weight(1f)
                    )
                }

                // Jauge dynamique des jours consecutifs (label au-dessus pour éviter toute collision)
                val gaugeTotalDays = maxOf(4, engineState.targetObjectiveStreak.coerceAtMost(7))
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(
                        text = if (engineState.bestValidatedStreak >= 4) {
                            "Objectif 4j OK • Cible actuelle : ${engineState.targetObjectiveStreak}j"
                        } else {
                            "Progression vers l'objectif ${engineState.targetObjectiveStreak}j :"
                        },
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        for (dayIdx in 1..gaugeTotalDays) {
                            val lit = engineState.bestCombinedStreak >= dayIdx
                            Box(
                                modifier = Modifier
                                    .weight(1f)
                                    .height(10.dp)
                                    .clip(PillShape)
                                    .background(
                                        if (lit) MaterialTheme.colorScheme.tertiary
                                        else MaterialTheme.colorScheme.surface
                                    )
                                    .border(
                                        1.dp,
                                        if (lit) MaterialTheme.colorScheme.tertiary else MaterialTheme.colorScheme.outline.copy(alpha = 0.4f),
                                        PillShape
                                    )
                            )
                        }
                    }
                }
            }

            if (engineState.currentPrimaryExpr.isNotBlank() || engineState.honestStatusNote.isNotBlank()) {
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(CardShape12)
                        .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.72f))
                        .padding(horizontal = 12.dp, vertical = 8.dp)
                ) {
                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        if (engineState.currentPrimaryExpr.isNotBlank()) {
                            Text(
                                text = "Formule A (${engineState.currentSlotSplit}): ${engineState.currentPrimaryExpr}",
                                style = MaterialTheme.typography.labelSmall.copy(fontFamily = FontFamily.Monospace),
                                fontWeight = FontWeight.SemiBold,
                                maxLines = 2,
                                overflow = TextOverflow.Ellipsis
                            )
                            Text(
                                text = "Formule B : ${engineState.currentSecondaryExpr}",
                                style = MaterialTheme.typography.labelSmall.copy(fontFamily = FontFamily.Monospace),
                                color = MaterialTheme.colorScheme.primary,
                                maxLines = 2,
                                overflow = TextOverflow.Ellipsis
                            )
                        }
                        if (engineState.honestStatusNote.isNotBlank()) {
                            Text(
                                text = engineState.honestStatusNote,
                                style = MaterialTheme.typography.labelSmall,
                                color = if (engineState.phase == AutonomousPhase.VALIDATED) {
                                    MaterialTheme.colorScheme.tertiary
                                } else {
                                    MaterialTheme.colorScheme.onSurfaceVariant
                                }
                            )
                        }
                    }
                }
            }

            // Guide rapide + Contrôles de cycle de vie clairs :
            // Pause = suspend en mémoire • Continuer = reprend • Stop = ferme l'analyse • Reprendre à zéro = réinitialise • Infini = cherche sans limite
            when (engineState.phase) {
                AutonomousPhase.IDLE -> {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(
                            onClick = onStartAnalysis,
                            enabled = !isLoadingData && resultsCount >= 8,
                            modifier = Modifier
                                .fillMaxWidth()
                                .heightIn(min = 48.dp)
                                .testTag("start_analysis_button")
                        ) {
                            Icon(Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(18.dp))
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(
                                if (resultsCount >= 8) "LANCER L'ANALYSE CONTINUE"
                                else "Chargement historique ($resultsCount/8)"
                            )
                        }
                        OutlinedButton(
                            onClick = onStartInfiniteMode,
                            enabled = !isLoadingData && resultsCount >= 8,
                            modifier = Modifier
                                .fillMaxWidth()
                                .heightIn(min = 48.dp)
                                .testTag("start_infinite_button")
                        ) {
                            Text("Continuer jusqu'à l'infini (sans arrêt auto)")
                        }
                    }
                }

                AutonomousPhase.ANALYZING -> {
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Button(
                                onClick = onSuspendAnalysis,
                                modifier = Modifier
                                    .weight(1f)
                                    .heightIn(min = 48.dp)
                                    .testTag("suspend_analysis_button")
                            ) {
                                Text("Mettre en pause")
                            }
                            OutlinedButton(
                                onClick = onStopAnalysis,
                                modifier = Modifier
                                    .weight(1f)
                                    .heightIn(min = 48.dp)
                                    .testTag("stop_analysis_button")
                            ) {
                                Icon(Icons.Default.Close, contentDescription = null, modifier = Modifier.size(18.dp))
                                Spacer(modifier = Modifier.width(4.dp))
                                Text("Stop (Fermer)")
                            }
                        }
                        FlowRow(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp),
                            verticalArrangement = Arrangement.Center
                        ) {
                            FilterChip(
                                selected = engineState.infiniteMode,
                                onClick = { onSetInfiniteMode(!engineState.infiniteMode) },
                                label = { Text("Continuer jusqu'à l'infini") },
                                modifier = Modifier.testTag("infinite_mode_toggle")
                            )
                        }
                        Text(
                            text = "Pause = garde la mémoire • Stop = ferme totalement l'analyse",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }

                AutonomousPhase.SUSPENDED, AutonomousPhase.STOPPED -> {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            Button(
                                onClick = onResumeAnalysis,
                                enabled = !isLoadingData && resultsCount >= 8,
                                modifier = Modifier
                                    .weight(1f)
                                    .heightIn(min = 48.dp)
                                    .testTag("resume_analysis_button")
                            ) {
                                Icon(Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(18.dp))
                                Spacer(modifier = Modifier.width(4.dp))
                                Text("Continuer")
                            }
                            OutlinedButton(
                                onClick = onRestartFromZero,
                                enabled = !isLoadingData && resultsCount >= 8,
                                modifier = Modifier
                                    .weight(1f)
                                    .heightIn(min = 48.dp)
                                    .testTag("start_analysis_button")
                            ) {
                                Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(18.dp))
                                Spacer(modifier = Modifier.width(4.dp))
                                Text("Reprendre à zéro")
                            }
                        }
                        OutlinedButton(
                            onClick = onStartInfiniteMode,
                            enabled = !isLoadingData && resultsCount >= 8,
                            modifier = Modifier
                                .fillMaxWidth()
                                .heightIn(min = 48.dp)
                                .testTag("start_infinite_button")
                        ) {
                            Text("Continuer jusqu'à l'infini")
                        }
                    }
                }

                AutonomousPhase.VALIDATED -> {
                    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(
                            onClick = onStartInfiniteMode,
                            enabled = !isLoadingData,
                            modifier = Modifier
                                .fillMaxWidth()
                                .heightIn(min = 48.dp)
                                .testTag("continue_beyond_button")
                        ) {
                            Icon(Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(18.dp))
                            Spacer(modifier = Modifier.width(6.dp))
                            Text("Continuer jusqu'à l'infini (≥ ${maxOf(4, engineState.bestValidatedStreak) + 1}j)")
                        }
                        OutlinedButton(
                            onClick = onRestartFromZero,
                            enabled = !isLoadingData,
                            modifier = Modifier
                                .fillMaxWidth()
                                .heightIn(min = 48.dp)
                                .testTag("restart_analysis_button")
                        ) {
                            Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(18.dp))
                            Spacer(modifier = Modifier.width(6.dp))
                            Text("Reprendre à zéro")
                        }
                    }
                }
            }

            val activeStrategy = engineState.activeStrategySummary
            val activeStreak = activeStrategy?.combinedStreak
                ?: engineState.preferredStrategyDays?.takeIf { engineState.canLaunchBacktest }
                ?: engineState.bestValidatedStreak

            Button(
                onClick = onLaunchBacktestAndPrediction,
                enabled = engineState.canLaunchBacktest && !isLoadingData,
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.tertiary,
                    contentColor = MaterialTheme.colorScheme.onTertiary
                ),
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 48.dp)
                    .testTag("launch_backtest_button")
            ) {
                Icon(Icons.Default.Star, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    when {
                        !engineState.canLaunchBacktest -> "LANCER BACKTEST & PRÉDICTION (Verrouillé)"
                        engineState.backtestTriggered && activeStreak >= 2 ->
                            "ACTUALISER BACKTEST & PRÉDICTION (${activeStreak}J)"
                        engineState.backtestTriggered -> "ACTUALISER BACKTEST & PRÉDICTION"
                        activeStreak >= 2 ->
                            "LANCER BACKTEST & PRÉDICTION (${activeStreak}J)"
                        else -> "LANCER BACKTEST & PRÉDICTION"
                    }
                )
            }
            if (!engineState.canLaunchBacktest) {
                val needed = engineState.preferredStrategyDays
                Text(
                    text = if (needed != null) {
                        "Verrouillé : aucune formule de palier ≥${needed}J n'est encore enregistrée. Sélectionnez « Auto » ou un palier déjà découvert ci-dessous."
                    } else {
                        "Verrouillé tant qu'aucune formule causale (palier 2J / 3J / objectif 4J+) n'a encore été enregistrée."
                    },
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
private fun FourStageValidationCard(
    metrics: FourStageValidationMetrics,
    randomRef: RandomReferenceComparison,
    confirmationTests: Int
) {
    val outlineColor = if (metrics.isFourStageValidated) {
        MaterialTheme.colorScheme.tertiary
    } else {
        MaterialTheme.colorScheme.outline.copy(alpha = 0.35f)
    }
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("four_stage_card"),
        shape = CardShape16,
        border = BorderStroke(1.dp, outlineColor),
        colors = CardDefaults.cardColors(
            containerColor = if (metrics.isFourStageValidated) {
                MaterialTheme.colorScheme.tertiaryContainer
            } else {
                MaterialTheme.colorScheme.surfaceVariant
            }
        )
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(
                text = if (metrics.isFourStageValidated) {
                    "Validation Chronologique 4 Niveaux : VALIDÉE"
                } else {
                    "Validation Chronologique 4 Niveaux (En cours)"
                },
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.ExtraBold
            )
            Text(
                text = "1. Découverte (${metrics.discoveryDraws} tir.) : moy ${fmt3(metrics.discoveryAvgMatches)} • ≥2: ${metrics.discoveryTwoPlusCount} • série ${metrics.discoveryStreak}j",
                style = MaterialTheme.typography.bodySmall
            )
            Text(
                text = "2. Validation Glissante (${metrics.rollingDraws} tir.) : moy ${fmt3(metrics.rollingAvgMatches)} • ≥2/tir ${(metrics.rollingTwoPlusRate * 100f).toInt()}% (${metrics.rollingTwoPlusCount}) • ${if (metrics.rollingStabilityPassed) "Stable" else "Rejetée"}",
                style = MaterialTheme.typography.bodySmall
            )
            Text(
                text = "3. Confirmation Indép. (${metrics.confirmationDraws} tir., $confirmationTests tests) : moy ${fmt3(metrics.confirmationAvgMatches)} vs MC ${fmt3(randomRef.confirmationMeanAvgMatches)} [IC ${fmt3(randomRef.confirmationCiLow)}–${fmt3(randomRef.confirmationCiHigh)}] • p_adj=${String.format(Locale.FRANCE, "%.4f", randomRef.pValueAdjusted)}",
                style = MaterialTheme.typography.bodySmall
            )
            Text(
                text = "4. Test Final Scellé (${metrics.sealedDraws} tir.) : moy ${fmt3(metrics.sealedAvgMatches)} • causal=${if (metrics.sealedCausalVerified) "100%" else "En attente"}",
                style = MaterialTheme.typography.bodySmall,
                fontWeight = FontWeight.SemiBold
            )
        }
    }
}

@Composable
private fun DiscoveredFeaturesCard(features: List<Pair<String, Float>>) {
    val outlineColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.35f)
    val borderStroke = remember(outlineColor) { BorderStroke(1.dp, outlineColor) }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("learned_traits_card"),
        shape = CardShape16,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        border = borderStroke
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = "Variables Brutes Découvertes dans la Population",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold
            )
            features.forEach { (label, ratio) ->
                Column(verticalArrangement = Arrangement.spacedBy(3.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(label, style = MaterialTheme.typography.labelMedium)
                        Text(
                            text = "${(ratio * 100f).toInt()}%",
                            style = MaterialTheme.typography.labelSmall,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                    LinearProgressIndicator(
                        progress = { ratio.coerceIn(0f, 1f) },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(6.dp)
                            .clip(PillShape)
                    )
                }
            }
        }
    }
}

@Composable
private fun BestSymbolicFormulasCard(
    entries: List<SymbolicFormulaSummary>,
    generation: Int,
    experimentsCount: Int
) {
    val outlineColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.35f)
    val borderStroke = remember(outlineColor) { BorderStroke(1.dp, outlineColor) }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("hall_of_fame_card"),
        shape = CardShape16,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        border = borderStroke
    ) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(
                text = "Meilleures Formules Retenues — Gén. $generation ($experimentsCount exp.)",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold
            )
            entries.take(5).forEach { cand ->
                Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "${if (cand.isValidated) "★ " else ""}Gén.${cand.generation} [${cand.slotSplitLabel}] ${cand.primaryExprText}",
                            style = MaterialTheme.typography.labelSmall.copy(fontFamily = FontFamily.Monospace),
                            fontWeight = FontWeight.Bold,
                            color = if (cand.isValidated) MaterialTheme.colorScheme.tertiary else MaterialTheme.colorScheme.onSurface,
                            modifier = Modifier.weight(1f),
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text(
                            text = "D:${fmt3(cand.discoveryAvgMatches)} G:${fmt3(cand.rollingAvgMatches)}",
                            style = MaterialTheme.typography.labelSmall,
                            fontWeight = FontWeight.Bold
                        )
                    }
                    Text(
                        text = "${cand.statusReason} • Vars: ${cand.usedVariables.joinToString(",")}",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                }
            }
        }
    }
}

@Composable
private fun RejectedFormulasCard(rejected: List<RejectedFormulaSummary>) {
    val outlineColor = MaterialTheme.colorScheme.error.copy(alpha = 0.35f)
    val borderStroke = remember(outlineColor) { BorderStroke(1.dp, outlineColor) }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("rejected_formulas_card"),
        shape = CardShape16,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = borderStroke
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(
                text = "Formules Abandonnées / Rejetées (${rejected.size})",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.error
            )
            rejected.take(5).forEach { item ->
                Column(verticalArrangement = Arrangement.spacedBy(1.dp)) {
                    Text(
                        text = "[${item.rejectedAtStage}] Gén.${item.generation} • ${item.formulaPreview}",
                        style = MaterialTheme.typography.labelSmall.copy(fontFamily = FontFamily.Monospace),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                    Text(
                        text = "Motif : ${item.reason}",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}

@Composable
private fun LiveEngineLogsCard(logs: List<String>) {
    val outlineColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.35f)
    val borderStroke = remember(outlineColor) { BorderStroke(1.dp, outlineColor) }
    val recentLogs = remember(logs) { logs.takeLast(5).reversed() }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("engine_logs_card"),
        shape = CardShape16,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        border = borderStroke
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(5.dp)
        ) {
            Text(
                text = "Journal d'Évolution en Direct",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold
            )
            recentLogs.forEachIndexed { idx, line ->
                Text(
                    text = line,
                    style = MaterialTheme.typography.labelSmall.copy(fontFamily = FontFamily.Monospace),
                    color = if (idx == 0) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 2,
                    overflow = TextOverflow.Ellipsis
                )
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun LeChercheurCard(
    status: ChercheurStatusSummary,
    isBusy: Boolean,
    onRunChercheur: () -> Unit
) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("chercheur_card"),
        shape = CardShape16,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.primary.copy(alpha = 0.45f))
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Le Chercheur — Auto-Apprentissage & Injection",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.ExtraBold,
                        color = MaterialTheme.colorScheme.primary
                    )
                    Text(
                        text = status.lastLessonNote,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                MetricItem(
                    label = "Cycles",
                    value = "${status.cyclesCompleted}",
                    sub = "${status.teachingsProduced} leçons",
                    modifier = Modifier.weight(1f)
                )
                MetricItem(
                    label = "Injectées",
                    value = "${status.teachingsInjected}",
                    sub = "Premium",
                    modifier = Modifier.weight(1f)
                )
                MetricItem(
                    label = "Gardées / Jetées",
                    value = "${status.strategiesKept} / ${status.strategiesDiscarded}",
                    sub = "Audit causal",
                    modifier = Modifier.weight(1f)
                )
            }

            if (status.isRunning) {
                LinearProgressIndicator(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(5.dp)
                        .clip(PillShape)
                )
            }

            FilledTonalButton(
                onClick = onRunChercheur,
                enabled = !isBusy && !status.isRunning,
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 48.dp)
                    .testTag("run_chercheur_button")
            ) {
                Icon(Icons.Default.Star, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    if (status.isRunning) "Le Chercheur étudie & distille..."
                    else "Lancer Le Chercheur & Injecter"
                )
            }

            status.teachings.take(4).forEach { t ->
                val verdictLabel = when (t.status) {
                    "KEPT" -> "GARDÉE"
                    "DISCARDED" -> "JETÉE"
                    else -> "INJECTÉE"
                }
                val verdictColor = when (t.status) {
                    "KEPT" -> MaterialTheme.colorScheme.tertiary
                    "DISCARDED" -> MaterialTheme.colorScheme.error
                    else -> MaterialTheme.colorScheme.primary
                }
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(CardShape12)
                        .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.72f))
                        .padding(10.dp),
                    verticalArrangement = Arrangement.spacedBy(3.dp)
                ) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "${t.title} (${t.scopeLabel})",
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.weight(1f)
                        )
                        Text(
                            text = verdictLabel,
                            style = MaterialTheme.typography.labelSmall,
                            fontWeight = FontWeight.ExtraBold,
                            color = verdictColor
                        )
                    }
                    Text(
                        text = t.principleSummary,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun UnlockedSymbolicPredictionCard(
    prediction: PredictionResult,
    matchedDays: Int,
    patternLength: Int,
    approvedNote: String = "",
    onOpenEngineOutput: () -> Unit
) {
    val confidenceByNumber = remember(prediction.numberConfidences) {
        prediction.numberConfidences.associateBy { it.number }
    }
    Card(
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.secondaryContainer,
            contentColor = MaterialTheme.colorScheme.onSecondaryContainer
        ),
        shape = CardShape20,
        modifier = Modifier
            .fillMaxWidth()
            .testTag("unlocked_prediction_card")
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Prochain tirage : ${prediction.horaire}",
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.ExtraBold,
                    color = MaterialTheme.colorScheme.onSecondaryContainer,
                    modifier = Modifier.weight(1f)
                )
                Box(
                    modifier = Modifier
                        .clip(PillShape)
                        .background(MaterialTheme.colorScheme.tertiaryContainer)
                        .padding(horizontal = 10.dp, vertical = 4.dp)
                        .semantics(mergeDescendants = true) {}
                        .testTag("prediction_matched_days_counter")
                ) {
                    Text(
                        text = "Jours réussis : $matchedDays/$patternLength",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.ExtraBold,
                        color = MaterialTheme.colorScheme.onTertiaryContainer
                    )
                }
            }

            if (approvedNote.isNotBlank()) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(CardShape12)
                        .background(MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.72f))
                        .padding(horizontal = 10.dp, vertical = 6.dp)
                        .testTag("approved_backtest_banner"),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Default.Lock,
                        contentDescription = "Backtest figé et complété",
                        tint = MaterialTheme.colorScheme.onTertiaryContainer,
                        modifier = Modifier.size(16.dp)
                    )
                    Text(
                        text = approvedNote,
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.SemiBold,
                        color = MaterialTheme.colorScheme.onTertiaryContainer,
                        modifier = Modifier.weight(1f)
                    )
                }
            }

            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp, Alignment.CenterHorizontally),
                verticalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                prediction.predictedDraw.forEach { num ->
                    val conf = confidenceByNumber[num]
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        FeaturedLotoBall(num)
                        if (conf != null) {
                            Text(
                                text = String.format(Locale.FRANCE, "%.1f%%", conf.marginalProbabilityPercent),
                                style = MaterialTheme.typography.labelSmall,
                                fontWeight = FontWeight.ExtraBold,
                                color = MaterialTheme.colorScheme.tertiary
                            )
                        }
                    }
                }
            }

            Text(
                text = prediction.discoveredStrategyName.ifBlank { prediction.method.displayName },
                style = MaterialTheme.typography.labelSmall.copy(fontFamily = FontFamily.Monospace),
                fontWeight = FontWeight.Bold
            )

            if (prediction.methodANumbers.isNotEmpty() || prediction.methodBNumbers.isNotEmpty()) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(CardShape12)
                        .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.45f))
                        .padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    Text(
                        text = "Répartition de la Formule Symbolique :",
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold
                    )
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            text = prediction.methodALabel,
                            style = MaterialTheme.typography.labelSmall,
                            modifier = Modifier.weight(1f),
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                        Text(
                            text = prediction.methodANumbers.joinToString(" - ") { formatLotoNumber(it) },
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.ExtraBold,
                            color = MaterialTheme.colorScheme.primary
                        )
                    }
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            text = prediction.methodBLabel,
                            style = MaterialTheme.typography.labelSmall,
                            modifier = Modifier.weight(1f),
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                        Text(
                            text = prediction.methodBNumbers.joinToString(" - ") { formatLotoNumber(it) },
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.ExtraBold,
                            color = MaterialTheme.colorScheme.tertiary
                        )
                    }
                }
            }

            val expStr = fmt3(prediction.expectedMatchesEstimate)
            val p1Str = String.format(Locale.FRANCE, "%.1f", prediction.calibratedProbAtLeastOne * 100f)
            Text(
                text = "Espérance : $expStr / 5 • P(≥1 bon) : $p1Str% • ${prediction.coldNumbersCount}/5 en écart",
                style = MaterialTheme.typography.bodySmall,
                fontWeight = FontWeight.Medium
            )
            if (prediction.auditNote.isNotBlank()) {
                Text(
                    text = "${prediction.auditNote} (${prediction.trainedOnCount} tirages passés)",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.85f)
                )
            }
            OutlinedButton(
                onClick = onOpenEngineOutput,
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 48.dp)
                    .testTag("open_engine_output_button")
            ) {
                Icon(Icons.Default.Star, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(modifier = Modifier.width(6.dp))
                Text("Voir Détails & Confiances (%)")
            }
        }
    }
}

@Composable
private fun SecurityVerificationCard(bt: BacktestResult) {
    val containerColor = if (bt.allStepsCausalVerified) {
        MaterialTheme.colorScheme.tertiaryContainer
    } else {
        MaterialTheme.colorScheme.errorContainer
    }
    val contentColor = if (bt.allStepsCausalVerified) {
        MaterialTheme.colorScheme.onTertiaryContainer
    } else {
        MaterialTheme.colorScheme.onErrorContainer
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("anti_cheat_badge"),
        shape = CardShape16,
        colors = CardDefaults.cardColors(
            containerColor = containerColor,
            contentColor = contentColor
        )
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.Top,
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Icon(
                imageVector = Icons.Default.Check,
                contentDescription = "Vérification anti-triche",
                modifier = Modifier.padding(top = 2.dp)
            )
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(
                    text = if (bt.allStepsCausalVerified) {
                        "Protocole Anti-Triche Vérifié (100%) • Série ${bt.maxConsecutiveQualifyingDays}j"
                    } else {
                        "Vérification incomplète"
                    },
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "Période : ${bt.periodLabel}. Zéro fuite future : t_train < t_cible vérifié sur ${bt.totalTested} tirages (${bt.totalStrategiesSearched} exp. en ${bt.searchRoundsExecuted} gén.).",
                    style = MaterialTheme.typography.bodySmall
                )
            }
        }
    }
}

@Composable
private fun BacktestSummaryCard(
    bt: BacktestResult,
    patternLength: Int
) {
    val outlineColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.35f)
    val borderStroke = remember(outlineColor) { BorderStroke(1.dp, outlineColor) }
    val matchedDays = bt.maxConsecutiveQualifyingDays

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("backtest_summary_card"),
        shape = CardShape16,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        border = borderStroke
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Résultats Réels Observés (${bt.totalTested} tirages)",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.weight(1f)
                )
                Box(
                    modifier = Modifier
                        .clip(PillShape)
                        .background(MaterialTheme.colorScheme.tertiaryContainer)
                        .padding(horizontal = 10.dp, vertical = 4.dp)
                        .semantics(mergeDescendants = true) {}
                        .testTag("backtest_matched_days_counter")
                ) {
                    Text(
                        text = "$matchedDays/$patternLength",
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.Black,
                        color = MaterialTheme.colorScheme.onTertiaryContainer
                    )
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                MetricItem(
                    label = "Tirages ≥ 2 bons",
                    value = "${bt.drawsWithTwoPlusMatches} / ${bt.totalTested}",
                    sub = "≥3: ${bt.drawsWithThreePlusMatches} • ≥4: ${bt.drawsWithFourPlusMatches}",
                    modifier = Modifier.weight(1f)
                )
                MetricItem(
                    label = "Jours réussis (${patternLength}J)",
                    value = "$matchedDays/$patternLength",
                    sub = "${bt.qualifyingDaysCount}/${bt.totalDaysTested} qualif.",
                    modifier = Modifier.weight(1f)
                )
                MetricItem(
                    label = "Taux ≥ 1 bon",
                    value = "${String.format(Locale.FRANCE, "%.1f", bt.successRate * 100f)}%",
                    sub = "IC [${(bt.rateOnePlusCiLow * 100).toInt()}–${(bt.rateOnePlusCiHigh * 100).toInt()}%]",
                    modifier = Modifier.weight(1f)
                )
            }

            Text(
                text = "Série max jours consécutifs qualifiants : $matchedDays/$patternLength j",
                style = MaterialTheme.typography.bodySmall,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.tertiary
            )
            Text(
                text = "Plus longue série à 0 correspondance : ${bt.maxZeroMatchStreak} tirage(s)",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            val avgStr = fmt3(bt.averageMatches)
            val avgCi = "${fmt3(bt.avgMatchesCiLow)}–${fmt3(bt.avgMatchesCiHigh)}"
            val brierStr = String.format(Locale.FRANCE, "%.5f", bt.averageBrierScore)
            val baseBrierStr = String.format(Locale.FRANCE, "%.5f", BacktestResult.RANDOM_BRIER_SCORE)
            val eceStr = String.format(Locale.FRANCE, "%.4f", bt.calibrationErrorEce)

            Text(
                text = "Moy. bons : $avgStr (IC 95% [$avgCi], hasard : 0,278)",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Text(
                text = "Brier : $brierStr (hasard : $baseBrierStr) • ECE : $eceStr",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )

            HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(alpha = 0.2f))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                bt.matchDistribution.forEachIndexed { count, freq ->
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = "$count/5",
                            style = MaterialTheme.typography.labelSmall,
                            color = if (count >= 2 && freq > 0) MaterialTheme.colorScheme.tertiary else MaterialTheme.colorScheme.onSurfaceVariant,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "$freq",
                            style = MaterialTheme.typography.bodyMedium,
                            fontWeight = FontWeight.SemiBold
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun HoraireBreakdownCard(horaires: List<HorairePerformance>) {
    val outlineColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.35f)
    val borderStroke = remember(outlineColor) { BorderStroke(1.dp, outlineColor) }
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = CardShape16,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        border = borderStroke
    ) {
        Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(
                text = "Performances par Horaire",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold
            )
            horaires.forEach { h ->
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "${h.horaire} (${h.testedCount} tir.)",
                        style = MaterialTheme.typography.bodySmall,
                        fontWeight = FontWeight.SemiBold,
                        modifier = Modifier.weight(1f)
                    )
                    Text(
                        text = "Moy: ${String.format(Locale.FRANCE, "%.2f", h.averageMatches)} • ≥1: ${h.onePlusCount} • ≥2: ${h.twoPlusCount}",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}

@Composable
private fun DailyQualificationCard(
    dailySummaries: List<DailyBacktestSummary>,
    matchedDays: Int,
    patternLength: Int
) {
    val outlineColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.35f)
    val borderStroke = remember(outlineColor) { BorderStroke(1.dp, outlineColor) }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = CardShape16,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        border = borderStroke
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Bilan Quotidien (≥3 tir. ≥2 + avant-dernier ≥2)",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.weight(1f)
                )
                Text(
                    text = "$matchedDays/$patternLength j",
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.Black,
                    color = MaterialTheme.colorScheme.tertiary,
                    modifier = Modifier.testTag("daily_matched_days_counter")
                )
            }
            if (dailySummaries.isEmpty()) {
                Text(
                    text = "Aucune journée disponible sur cette fenêtre.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            } else {
                dailySummaries.take(21).forEach { day ->
                    val statusColor = if (day.isQualifyingDay) {
                        MaterialTheme.colorScheme.tertiary
                    } else {
                        MaterialTheme.colorScheme.onSurfaceVariant
                    }
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = day.dayLabel,
                                style = MaterialTheme.typography.bodyMedium,
                                fontWeight = FontWeight.SemiBold,
                                color = statusColor
                            )
                            Text(
                                text = "Avant-dernier ≥2 : ${if (day.penultimateTwoPlus) "OUI" else "NON"} (${day.penultimateHitCount}/5)",
                                style = MaterialTheme.typography.labelSmall,
                                color = if (day.penultimateTwoPlus) MaterialTheme.colorScheme.tertiary else MaterialTheme.colorScheme.error
                            )
                        }
                        Text(
                            text = "${day.drawsWithTwoPlus}/${day.drawsTested} ≥2 • ${if (day.isQualifyingDay) "OUI" else "NON"}",
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.Bold,
                            color = statusColor
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun MetricItem(
    label: String,
    value: String,
    sub: String,
    modifier: Modifier = Modifier
) {
    Column(modifier = modifier) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
        Text(
            text = value,
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
        Text(
            text = sub,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.8f),
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
    }
}

@Composable
private fun BacktestStepCard(step: BacktestStepResult) {
    val hasMatch = step.matchCount >= 1
    val isTwoPlus = step.matchCount >= 2
    val outlineColor = when {
        isTwoPlus -> MaterialTheme.colorScheme.tertiary
        hasMatch -> MaterialTheme.colorScheme.tertiary.copy(alpha = 0.45f)
        else -> MaterialTheme.colorScheme.outline.copy(alpha = 0.28f)
    }
    val borderStroke = remember(outlineColor, isTwoPlus) {
        BorderStroke(if (isTwoPlus) 1.5.dp else 1.dp, outlineColor)
    }
    val matchedSet = remember(step.matchedNumbers) { step.matchedNumbers.toSet() }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("backtest_step_${step.drawId}"),
        shape = CardShape16,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        border = borderStroke,
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "${step.drawLabel} • ${step.drawDate}",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "Pas #${step.stepIndex} • ${step.trainSampleCount} tirages passés • Brier ${String.format(Locale.FRANCE, "%.4f", step.stepBrierScore)}",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Box(
                    modifier = Modifier
                        .clip(PillShape)
                        .background(
                            if (hasMatch) MaterialTheme.colorScheme.tertiaryContainer
                            else MaterialTheme.colorScheme.surface
                        )
                        .padding(horizontal = 10.dp, vertical = 4.dp)
                ) {
                    Text(
                        text = "${step.matchCount}/5",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.ExtraBold,
                        color = if (hasMatch) MaterialTheme.colorScheme.onTertiaryContainer else MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(
                    text = "Prédiction passée (Formules A + B avant tirage)",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    step.predictedNumbers.forEach { num ->
                        LotoBall(number = num, isMatched = num in matchedSet)
                    }
                }
                AnimatedVisibility(visible = step.methodANumbers.isNotEmpty() || step.methodBNumbers.isNotEmpty()) {
                    val aStr = step.methodANumbers.joinToString(",") { formatLotoNumber(it) }
                    val bStr = step.methodBNumbers.joinToString(",") { formatLotoNumber(it) }
                    Text(
                        text = "A : [$aStr]  •  B : [$bStr]",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.primary
                    )
                }
            }

            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text(
                    text = "Résultat réel du jeu",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    step.actualNumbers.forEach { num ->
                        LotoBall(number = num, isMatched = num in matchedSet)
                    }
                }
            }

            val matchText = remember(step.matchedNumbers) {
                if (step.matchedNumbers.isNotEmpty()) {
                    "Correspondance(s) : " + step.matchedNumbers.joinToString(", ") { formatLotoNumber(it) }
                } else {
                    "Aucune correspondance"
                }
            }
            Text(
                text = matchText,
                style = MaterialTheme.typography.labelMedium,
                fontWeight = if (hasMatch) FontWeight.Bold else FontWeight.Normal,
                color = if (hasMatch) MaterialTheme.colorScheme.tertiary else MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun PrecisionTargetEngineCard(
    state: PrecisionUiState,
    resultsCount: Int,
    isBusy: Boolean,
    onSelectMode: (PrecisionTargetMode) -> Unit,
    onStart: () -> Unit,
    onPause: () -> Unit,
    onResume: () -> Unit,
    onStop: () -> Unit,
    onRestartZero: () -> Unit,
    onToggleInfinite: (Boolean) -> Unit,
    onLaunchBacktest: () -> Unit
) {
    var showPrecisionSteps by rememberSaveable { mutableStateOf(false) }
    val analyzing = state.phase == AutonomousPhase.ANALYZING
    val hasData = resultsCount >= 8 && !isBusy

    val borderAccent = when (state.phase) {
        AutonomousPhase.VALIDATED -> MaterialTheme.colorScheme.tertiary
        AutonomousPhase.ANALYZING -> MaterialTheme.colorScheme.primary
        AutonomousPhase.SUSPENDED -> MaterialTheme.colorScheme.secondary
        else -> MaterialTheme.colorScheme.outline.copy(alpha = 0.45f)
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .animateContentSize()
            .testTag("precision_engine_card"),
        shape = CardShape20,
        border = BorderStroke(1.5.dp, borderAccent),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text(
                        text = "Moteur Précision Indépendant",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.ExtraBold
                    )
                    Text(
                        text = state.targetMode.description,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Box(
                    modifier = Modifier
                        .clip(PillShape)
                        .background(
                            if (state.phase == AutonomousPhase.VALIDATED) MaterialTheme.colorScheme.tertiaryContainer
                            else if (analyzing) MaterialTheme.colorScheme.primaryContainer
                            else MaterialTheme.colorScheme.surface
                        )
                        .padding(horizontal = 10.dp, vertical = 4.dp)
                ) {
                    Text(
                        text = if (state.infiniteMode && analyzing) "INFINI" else state.phase.name,
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.ExtraBold,
                        color = if (state.phase == AutonomousPhase.VALIDATED) {
                            MaterialTheme.colorScheme.onTertiaryContainer
                        } else if (analyzing) {
                            MaterialTheme.colorScheme.onPrimaryContainer
                        } else {
                            MaterialTheme.colorScheme.onSurfaceVariant
                        }
                    )
                }
            }

            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                PrecisionTargetMode.entries.forEach { mode ->
                    FilterChip(
                        selected = state.targetMode == mode,
                        onClick = { onSelectMode(mode) },
                        enabled = !analyzing,
                        label = { Text(mode.label) },
                        modifier = Modifier
                            .minimumInteractiveComponentSize()
                            .testTag("precision_mode_${mode.name}")
                    )
                }
                FilterChip(
                    selected = state.infiniteMode,
                    onClick = { onToggleInfinite(!state.infiniteMode) },
                    label = { Text("Infini") },
                    modifier = Modifier
                        .minimumInteractiveComponentSize()
                        .testTag("precision_infinite_toggle")
                )
            }

            if (analyzing) {
                LinearProgressIndicator(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(5.dp)
                        .clip(PillShape)
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                MetricItem(
                    label = "Génération",
                    value = "Gén. ${state.generation}",
                    sub = "${state.experimentsCount} exp.",
                    modifier = Modifier.weight(1f)
                )
                MetricItem(
                    label = "Palier",
                    value = "${state.bestValidatedStreak}J",
                    sub = "${state.excludedCount} testées",
                    modifier = Modifier.weight(1f)
                )
                MetricItem(
                    label = "Conf.",
                    value = "${state.confirmationTestsCount}",
                    sub = "p=${String.format(Locale.FRANCE, "%.3f", state.pValueAdjusted)}",
                    modifier = Modifier.weight(1f)
                )
            }

            if (state.activeFormulaDesc.isNotBlank() || state.honestStatusNote.isNotBlank()) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(CardShape12)
                        .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.72f))
                        .padding(horizontal = 10.dp, vertical = 8.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    if (state.activeFormulaDesc.isNotBlank()) {
                        Text(
                            text = state.activeFormulaDesc,
                            style = MaterialTheme.typography.labelSmall.copy(fontFamily = FontFamily.Monospace),
                            fontWeight = FontWeight.SemiBold,
                            maxLines = 2,
                            overflow = TextOverflow.Ellipsis
                        )
                    }
                    Text(
                        text = state.honestStatusNote,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            FlowRow(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                if (!analyzing) {
                    Button(
                        onClick = if (state.phase == AutonomousPhase.SUSPENDED || state.phase == AutonomousPhase.STOPPED) onResume else onStart,
                        enabled = hasData,
                        modifier = Modifier
                            .heightIn(min = 48.dp)
                            .testTag("precision_start_or_resume_btn")
                    ) {
                        Icon(Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(
                            if (state.phase == AutonomousPhase.SUSPENDED || state.phase == AutonomousPhase.STOPPED) "Continuer"
                            else "Démarrer"
                        )
                    }
                } else {
                    Button(
                        onClick = onPause,
                        modifier = Modifier
                            .heightIn(min = 48.dp)
                            .testTag("precision_pause_btn")
                    ) {
                        Text("Mettre en pause")
                    }
                    OutlinedButton(
                        onClick = onStop,
                        modifier = Modifier
                            .heightIn(min = 48.dp)
                            .testTag("precision_stop_btn")
                    ) {
                        Icon(Icons.Default.Close, contentDescription = null, modifier = Modifier.size(18.dp))
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Stop")
                    }
                }
                OutlinedButton(
                    onClick = onRestartZero,
                    enabled = hasData,
                    modifier = Modifier
                        .heightIn(min = 48.dp)
                        .testTag("precision_restart_btn")
                ) {
                    Icon(Icons.Default.Refresh, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Reprendre à zéro")
                }
                FilledTonalButton(
                    onClick = onLaunchBacktest,
                    enabled = hasData,
                    modifier = Modifier
                        .heightIn(min = 48.dp)
                        .testTag("precision_backtest_btn")
                ) {
                    Icon(Icons.Default.Star, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Backtest & Prédiction")
                }
            }

            val live = state.livePrediction
            if (live != null) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(CardShape12)
                        .background(MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.6f))
                        .padding(10.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        text = "Prédiction Précision ${live.mode.label} • ${live.horaire} (${live.dayLabel})",
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.ExtraBold,
                        color = MaterialTheme.colorScheme.onSecondaryContainer
                    )
                    Row(
                        horizontalArrangement = Arrangement.spacedBy(12.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        live.predictedNumbers.forEachIndexed { idx, num ->
                            val pct = live.confidencesPercent.getOrNull(idx)?.second ?: 0f
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                FeaturedLotoBall(number = num)
                                Spacer(modifier = Modifier.height(2.dp))
                                Text(
                                    text = String.format(Locale.FRANCE, "%.1f%%", pct),
                                    style = MaterialTheme.typography.labelSmall,
                                    fontWeight = FontWeight.Bold,
                                    color = MaterialTheme.colorScheme.primary
                                )
                            }
                        }
                    }
                    Text(
                        text = live.causalAuditNote,
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.85f)
                    )
                    if (state.approvedNote.isNotBlank()) {
                        Text(
                            text = state.approvedNote,
                            style = MaterialTheme.typography.labelSmall,
                            fontWeight = FontWeight.SemiBold,
                            color = MaterialTheme.colorScheme.tertiary
                        )
                    }
                }
            }

            val bt = state.backtestResult
            if (bt != null && bt.steps.isNotEmpty()) {
                OutlinedButton(
                    onClick = { showPrecisionSteps = !showPrecisionSteps },
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 48.dp)
                        .testTag("precision_toggle_steps_btn")
                ) {
                    Text(
                        if (showPrecisionSteps) {
                            "Masquer Backtest Précision (${bt.exactTargetCount}/${bt.totalTested} exacts)"
                        } else {
                            "Voir Backtest Précision (${bt.exactTargetCount}/${bt.totalTested} exacts • ≥1: ${bt.onePlusCount})"
                        }
                    )
                }
                AnimatedVisibility(visible = showPrecisionSteps) {
                    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        bt.steps.takeLast(12).reversed().forEach { step ->
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .clip(CardShape12)
                                    .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.72f))
                                    .padding(horizontal = 10.dp, vertical = 6.dp),
                                horizontalArrangement = Arrangement.SpaceBetween,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                Column(modifier = Modifier.weight(1f)) {
                                    Text(
                                        text = "#${step.stepIndex} • ${step.game} ${step.horaire} (${step.dayKey})",
                                        style = MaterialTheme.typography.labelSmall,
                                        fontWeight = FontWeight.Bold
                                    )
                                    Text(
                                        text = "Prédit: [${step.predictedNumbers.joinToString("-") { formatLotoNumber(it) }}] • Pos1=${formatLotoNumber(step.actualPos1)} [${step.actualNumbers.joinToString("-") { formatLotoNumber(it) }}]",
                                        style = MaterialTheme.typography.labelSmall,
                                        color = MaterialTheme.colorScheme.onSurfaceVariant
                                    )
                                }
                                Text(
                                    text = if (step.exactTargetHit) "EXACT" else "${step.matchCount} bon(s)",
                                    style = MaterialTheme.typography.labelSmall,
                                    fontWeight = FontWeight.ExtraBold,
                                    color = if (step.exactTargetHit) MaterialTheme.colorScheme.tertiary else MaterialTheme.colorScheme.onSurfaceVariant
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

private fun fmt3(v: Float): String = String.format(Locale.FRANCE, "%.3f", v)
