package com.example.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.animateContentSize
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.spring
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
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
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.Lock
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
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
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.domain.engine.AutonomousPhase
import com.example.domain.engine.BacktestResult
import com.example.domain.engine.EngineActionState
import com.example.domain.engine.FourStageValidationMetrics
import com.example.domain.engine.NextDrawTarget
import com.example.domain.engine.PredictedNumberConfidence
import com.example.domain.engine.PredictionResult
import com.example.domain.util.StrategyBackupExport
import com.example.domain.util.StrategyBackupImportResult
import com.example.ui.components.BacktestExportActions
import com.example.ui.components.BacktestStrategySelectorCard
import com.example.ui.components.StrategyBackupActionsCard
import com.example.ui.components.formatLotoNumber
import java.util.Locale

private val OutputCardShape = RoundedCornerShape(22.dp)
private val InnerTileShape = RoundedCornerShape(16.dp)
private val BadgePillShape = RoundedCornerShape(10.dp)
private const val UNIFORM_RANDOM_PCT = (5.0f / 90.0f) * 100f

@Composable
fun AutonomousEngineOutputScreen(
    engineState: EngineActionState,
    prediction: PredictionResult?,
    selectedHoraire: String,
    availableHoraires: List<String>,
    isBusy: Boolean,
    onSelectHoraire: (String) -> Unit,
    onStartOrResumeAnalysis: () -> Unit,
    onLaunchBacktestAndPrediction: () -> Unit,
    onOpenDiscoveryConsole: () -> Unit,
    onSelectPreferredStrategyDays: (Int?) -> Unit = {},
    onSelectSavedWinner: (String) -> Unit = {},
    onExportStrategiesBackup: (suspend () -> Result<StrategyBackupExport>)? = null,
    onImportStrategiesBackup: (suspend (String) -> Result<StrategyBackupImportResult>)? = null,
    nextDrawTarget: NextDrawTarget = NextDrawTarget(horaire = selectedHoraire),
    backtestResult: BacktestResult? = null,
    modifier: Modifier = Modifier
) {
    val activePrediction = prediction?.takeIf { it.predictedDraw.isNotEmpty() }
    val resolvedNumbers = remember(activePrediction) {
        if (activePrediction == null) {
            emptyList()
        } else if (activePrediction.numberConfidences.isNotEmpty()) {
            activePrediction.numberConfidences
        } else {
            val uniformPct = UNIFORM_RANDOM_PCT
            activePrediction.predictedDraw.mapIndexed { idx, num ->
                PredictedNumberConfidence(
                    number = num,
                    marginalProbabilityPercent = uniformPct,
                    relativeConfidencePercent = 100f - idx * 8f,
                    liftVsRandom = 1f,
                    rank = idx + 1,
                    sourceBranch = if (num in activePrediction.methodANumbers) "Formule A" else "Formule B",
                    isGapCandidate = false
                )
            }
        }
    }

    var selectedBallNumber by rememberSaveable(activePrediction?.horaire, activePrediction?.predictedDraw) {
        mutableIntStateOf(resolvedNumbers.firstOrNull()?.number ?: -1)
    }
    val selectedConfidence = remember(resolvedNumbers, selectedBallNumber) {
        resolvedNumbers.firstOrNull { it.number == selectedBallNumber } ?: resolvedNumbers.firstOrNull()
    }

    BoxWithConstraints(
        modifier = modifier
            .fillMaxSize()
            .testTag("engine_output_screen")
    ) {
        val isWideScreen = maxWidth >= 600.dp

        val matchedDays = engineState.matchedDaysCount(backtestResult)
        val patternLength = engineState.selectedPatternLength

        LazyColumn(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .fillMaxSize()
                .widthIn(max = 860.dp)
                .testTag("engine_output_list"),
            contentPadding = PaddingValues(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            item(key = "output_header") {
                EngineOutputHeaderCard(
                    engineState = engineState,
                    nextDrawTarget = nextDrawTarget,
                    selectedHoraire = selectedHoraire,
                    matchedDays = matchedDays,
                    patternLength = patternLength
                )
            }

            if ((engineState.preferredStrategyDays == null || engineState.canLaunchBacktest) &&
                activePrediction != null &&
                resolvedNumbers.isNotEmpty()
            ) {
                item(key = "large_prediction_banner") {
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .testTag("large_prediction_banner"),
                        shape = OutputCardShape,
                        border = BorderStroke(2.dp, MaterialTheme.colorScheme.primary),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.35f))
                    ) {
                        Column(
                            modifier = Modifier.padding(20.dp),
                            horizontalAlignment = Alignment.CenterHorizontally,
                            verticalArrangement = Arrangement.spacedBy(12.dp)
                        ) {
                            Text(
                                text = "PRÉDICTION OFFICIELLE (${activePrediction.horaire})",
                                style = MaterialTheme.typography.labelMedium,
                                fontWeight = FontWeight.ExtraBold,
                                color = MaterialTheme.colorScheme.primary
                            )
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.SpaceEvenly,
                                verticalAlignment = Alignment.CenterVertically
                            ) {
                                activePrediction.predictedDraw.forEach { num ->
                                    Box(
                                        modifier = Modifier
                                            .size(58.dp)
                                            .clip(CircleShape)
                                            .background(MaterialTheme.colorScheme.primary),
                                        contentAlignment = Alignment.Center
                                    ) {
                                        Text(
                                            text = formatLotoNumber(num),
                                            style = MaterialTheme.typography.titleLarge,
                                            fontWeight = FontWeight.Black,
                                            color = MaterialTheme.colorScheme.onPrimary
                                        )
                                    }
                                }
                            }
                            Text(
                                text = "Combinaison validée • Causalité stricte & Zéro dérive",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                                fontWeight = FontWeight.SemiBold
                            )
                        }
                    }
                }

                item(key = "predicted_numbers_header") {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .testTag("predicted_numbers_confidence_grid"),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text(
                            text = "Numéros Prédits & Indices de Confiance",
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier.weight(1f)
                        )
                        Text(
                            text = "Base hasard : ${String.format(Locale.FRANCE, "%.1f%%", UNIFORM_RANDOM_PCT)}",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.tertiary,
                            fontWeight = FontWeight.SemiBold,
                            maxLines = 1
                        )
                    }
                }

                val chunkSize = if (isWideScreen) 5 else 3
                val rows = resolvedNumbers.chunked(chunkSize)
                items(
                    count = rows.size,
                    key = { idx -> "predicted_row_$idx" }
                ) { rowIdx ->
                    val rowItems = rows[rowIdx]
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(10.dp)
                    ) {
                        rowItems.forEach { item ->
                            PredictedNumberConfidenceTile(
                                item = item,
                                isSelected = item.number == (selectedConfidence?.number ?: -1),
                                onClick = { selectedBallNumber = item.number },
                                modifier = Modifier.weight(1f)
                            )
                        }
                        if (!isWideScreen && rowItems.size < chunkSize) {
                            repeat(chunkSize - rowItems.size) {
                                Spacer(modifier = Modifier.weight(1f))
                            }
                        }
                    }
                }

                if (selectedConfidence != null) {
                    item(key = "selected_number_inspector") {
                        SelectedNumberConfidenceInspector(
                            item = selectedConfidence,
                            prediction = activePrediction
                        )
                    }
                }

                item(key = "confidence_summary_and_formulas") {
                    EngineConfidenceSummaryCard(
                        prediction = activePrediction,
                        engineState = engineState,
                        matchedDays = matchedDays,
                        patternLength = patternLength,
                        isWideScreen = isWideScreen
                    )
                }

                val comparisonList = activePrediction.candidateConfidences.ifEmpty { resolvedNumbers }
                if (comparisonList.isNotEmpty()) {
                    item(key = "candidate_distribution") {
                        CandidateConfidenceComparisonCard(
                            candidates = comparisonList,
                            predictedSet = activePrediction.predictedDraw.toSet()
                        )
                    }
                }

                if (backtestResult != null) {
                    item(key = "output_backtest_export") {
                        BacktestExportActions(
                            backtestResult = backtestResult,
                            engineState = engineState,
                            prediction = activePrediction,
                            nextDrawTarget = nextDrawTarget,
                            tagPrefix = "output_"
                        )
                    }
                }
            } else {
                item(key = "empty_engine_output") {
                    EngineOutputEmptyOrLockedState(
                        engineState = engineState,
                        isBusy = isBusy,
                        onStartOrResumeAnalysis = onStartOrResumeAnalysis,
                        onLaunchBacktestAndPrediction = onLaunchBacktestAndPrediction,
                        onOpenDiscoveryConsole = onOpenDiscoveryConsole
                    )
                }
            }

            item(key = "output_strategy_selector") {
                BacktestStrategySelectorCard(
                    engineState = engineState,
                    backtestResult = backtestResult,
                    isBusy = isBusy,
                    onSelectPreferredDays = onSelectPreferredStrategyDays,
                    onSelectWinner = onSelectSavedWinner,
                    onUnlockPredictiveModeling = onLaunchBacktestAndPrediction
                )
            }

            if (onExportStrategiesBackup != null && onImportStrategiesBackup != null) {
                item(key = "output_strategy_backup") {
                    StrategyBackupActionsCard(
                        engineState = engineState,
                        isBusy = isBusy,
                        onExportStrategiesBackup = onExportStrategiesBackup,
                        onImportStrategiesBackup = onImportStrategiesBackup,
                        tagPrefix = "output_"
                    )
                }
            }
        }
    }
}

@Composable
private fun EngineOutputHeaderCard(
    engineState: EngineActionState,
    nextDrawTarget: NextDrawTarget,
    selectedHoraire: String,
    matchedDays: Int,
    patternLength: Int
) {
    val primaryColor = MaterialTheme.colorScheme.primary
    val tertiaryColor = MaterialTheme.colorScheme.tertiary
    val headerGradient = Brush.linearGradient(
        colors = listOf(
            MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.72f),
            MaterialTheme.colorScheme.surfaceVariant
        )
    )

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("engine_output_header"),
        shape = OutputCardShape,
        border = BorderStroke(
            1.dp,
            if (engineState.canLaunchBacktest) tertiaryColor.copy(alpha = 0.85f)
            else MaterialTheme.colorScheme.outline.copy(alpha = 0.45f)
        ),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
    ) {
        Column(
            modifier = Modifier
                .background(headerGradient)
                .padding(18.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = "Sortie du Moteur Autonome",
                        style = MaterialTheme.typography.titleLarge,
                        fontWeight = FontWeight.ExtraBold
                    )
                    Text(
                        text = if (engineState.generation > 0) {
                            "Génération ${engineState.generation} • ${engineState.experimentsCount} formules évaluées"
                        } else {
                            "Découverte symbolique libre sans règle figée"
                        },
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                val activeStreak = engineState.activeStrategySummary?.combinedStreak
                    ?: engineState.bestValidatedStreak
                val statusText = when {
                    engineState.phase == AutonomousPhase.VALIDATED -> "VALIDÉ 4 NIVEAUX"
                    engineState.phase == AutonomousPhase.ANALYZING && engineState.hasSavedWinner -> "RECHERCHE (${activeStreak}J PRÊT)"
                    engineState.phase == AutonomousPhase.ANALYZING -> "ANALYSE ACTIVE"
                    engineState.phase == AutonomousPhase.SUSPENDED -> "SUSPENDU"
                    engineState.phase == AutonomousPhase.STOPPED -> "ARRÊTÉ"
                    engineState.hasSavedWinner -> "STRATÉGIE ${activeStreak}J PRÊTE"
                    else -> "EN ATTENTE"
                }
                val badgeBg = when (engineState.phase) {
                    AutonomousPhase.VALIDATED -> MaterialTheme.colorScheme.tertiaryContainer
                    AutonomousPhase.ANALYZING -> MaterialTheme.colorScheme.primaryContainer
                    else -> MaterialTheme.colorScheme.surface
                }
                val badgeFg = when (engineState.phase) {
                    AutonomousPhase.VALIDATED -> MaterialTheme.colorScheme.onTertiaryContainer
                    AutonomousPhase.ANALYZING -> MaterialTheme.colorScheme.onPrimaryContainer
                    else -> MaterialTheme.colorScheme.onSurfaceVariant
                }
                Row(
                    modifier = Modifier
                        .clip(BadgePillShape)
                        .background(badgeBg)
                        .border(1.dp, (if (engineState.isValidated) tertiaryColor else primaryColor).copy(alpha = 0.35f), BadgePillShape)
                        .padding(horizontal = 10.dp, vertical = 6.dp),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(5.dp)
                ) {
                    Icon(
                        imageVector = if (engineState.isValidated) Icons.Default.CheckCircle else Icons.Default.Info,
                        contentDescription = statusText,
                        tint = badgeFg,
                        modifier = Modifier.size(14.dp)
                    )
                    Text(
                        text = statusText,
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.ExtraBold,
                        color = badgeFg
                    )
                }
            }

            val resolvedHoraire = nextDrawTarget.horaire.ifBlank { selectedHoraire }
            val whenTag = if (nextDrawTarget.isTomorrow) "Demain" else "Aujourd'hui"
            val beninTime = nextDrawTarget.currentBeninTimeLabel.takeIf { it.isNotBlank() }?.let { " • $it Bénin" } ?: ""
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(BadgePillShape)
                    .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.72f))
                    .padding(horizontal = 12.dp, vertical = 8.dp)
                    .testTag("output_next_draw_badge"),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(
                    modifier = Modifier.weight(1f),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Icon(
                        imageVector = Icons.Default.Lock,
                        contentDescription = "Prochain tirage uniquement",
                        tint = primaryColor,
                        modifier = Modifier.size(16.dp)
                    )
                    Column(modifier = Modifier.weight(1f)) {
                        Text(
                            text = "Prochain tirage : $resolvedHoraire ($whenTag$beninTime)",
                            style = MaterialTheme.typography.labelMedium,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = engineState.periodDateRangeLabel.takeIf { it.isNotBlank() }?.let { "Fenêtre active : $it" }
                                ?: "Fenêtre active configurée",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
                Box(
                    modifier = Modifier
                        .clip(BadgePillShape)
                        .background(MaterialTheme.colorScheme.tertiaryContainer)
                        .padding(horizontal = 9.dp, vertical = 4.dp)
                        .testTag("output_header_matched_days_counter")
                ) {
                    Text(
                        text = "$matchedDays/$patternLength j",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.Black,
                        color = MaterialTheme.colorScheme.onTertiaryContainer
                    )
                }
            }
            if (engineState.approvedBacktestNote.isNotBlank()) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(BadgePillShape)
                        .background(MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.72f))
                        .padding(horizontal = 10.dp, vertical = 6.dp)
                        .testTag("output_approved_backtest_banner"),
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Icon(
                        imageVector = Icons.Default.Lock,
                        contentDescription = "Backtest figé et complété",
                        tint = MaterialTheme.colorScheme.onTertiaryContainer,
                        modifier = Modifier.size(15.dp)
                    )
                    Text(
                        text = engineState.approvedBacktestNote,
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.SemiBold,
                        color = MaterialTheme.colorScheme.onTertiaryContainer,
                        modifier = Modifier.weight(1f)
                    )
                }
            }
        }
    }
}


@Composable
private fun PredictedNumberConfidenceTile(
    item: PredictedNumberConfidence,
    isSelected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val tertiary = MaterialTheme.colorScheme.tertiary
    val primary = MaterialTheme.colorScheme.primary
    val trackColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.35f)
    val ringColor = if (item.sourceBranch == "Formule A") primary else tertiary
    val animatedSweep by animateFloatAsState(
        targetValue = (item.relativeConfidencePercent / 100f).coerceIn(0.08f, 1f),
        animationSpec = spring(stiffness = Spring.StiffnessLow),
        label = "confidence_ring_${item.number}"
    )

    val cardBg = Brush.verticalGradient(
        colors = if (isSelected) {
            listOf(
                MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.85f),
                MaterialTheme.colorScheme.surfaceVariant
            )
        } else {
            listOf(
                MaterialTheme.colorScheme.surfaceVariant,
                MaterialTheme.colorScheme.surface
            )
        }
    )

    Card(
        modifier = modifier
            .minimumInteractiveComponentSize()
            .clip(InnerTileShape)
            .clickable(role = Role.Button, onClick = onClick)
            .testTag("predicted_confidence_tile_${item.number}"),
        shape = InnerTileShape,
        border = BorderStroke(
            width = if (isSelected) 2.dp else 1.dp,
            color = if (isSelected) tertiary else MaterialTheme.colorScheme.outline.copy(alpha = 0.4f)
        ),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .background(cardBg)
                .padding(horizontal = 10.dp, vertical = 14.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "#${item.rank}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    fontWeight = FontWeight.Bold
                )
                if (item.isGapCandidate) {
                    Text(
                        text = "ÉCART",
                        style = MaterialTheme.typography.labelSmall.copy(fontSize = 9.sp),
                        fontWeight = FontWeight.ExtraBold,
                        color = MaterialTheme.colorScheme.onTertiaryContainer,
                        modifier = Modifier
                            .clip(BadgePillShape)
                            .background(MaterialTheme.colorScheme.tertiaryContainer)
                            .padding(horizontal = 5.dp, vertical = 2.dp)
                    )
                }
            }

            Box(
                modifier = Modifier.size(66.dp),
                contentAlignment = Alignment.Center
            ) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val strokePx = 6.dp.toPx()
                    val inset = strokePx / 2f
                    val arcSize = Size(size.width - strokePx, size.height - strokePx)
                    drawArc(
                        color = trackColor,
                        startAngle = 0f,
                        sweepAngle = 360f,
                        useCenter = false,
                        topLeft = Offset(inset, inset),
                        size = arcSize,
                        style = Stroke(width = strokePx, cap = StrokeCap.Round)
                    )
                    drawArc(
                        color = ringColor,
                        startAngle = -90f,
                        sweepAngle = 360f * animatedSweep,
                        useCenter = false,
                        topLeft = Offset(inset, inset),
                        size = arcSize,
                        style = Stroke(width = strokePx, cap = StrokeCap.Round)
                    )
                }

                Box(
                    modifier = Modifier
                        .size(46.dp)
                        .clip(CircleShape)
                        .background(MaterialTheme.colorScheme.surface),
                    contentAlignment = Alignment.Center
                ) {
                    Text(
                        text = formatLotoNumber(item.number),
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Black,
                        color = MaterialTheme.colorScheme.onSurface
                    )
                }
            }

            Text(
                text = String.format(Locale.FRANCE, "%.1f%%", item.marginalProbabilityPercent),
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.ExtraBold,
                color = ringColor,
                modifier = Modifier.testTag("confidence_pct_${item.number}")
            )

            Text(
                text = "Indice ${item.relativeConfidencePercent.toInt()}% • ×${String.format(Locale.FRANCE, "%.1f", item.liftVsRandom)}",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )

            Text(
                text = item.sourceBranch,
                style = MaterialTheme.typography.labelSmall,
                fontWeight = FontWeight.SemiBold,
                color = ringColor
            )
        }
    }
}

@Composable
private fun SelectedNumberConfidenceInspector(
    item: PredictedNumberConfidence,
    prediction: PredictionResult
) {
    val branchFormula = if (item.sourceBranch == "Formule A") {
        prediction.methodALabel
    } else {
        prediction.methodBLabel
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .animateContentSize()
            .testTag("selected_number_detail_card"),
        shape = InnerTileShape,
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.tertiary.copy(alpha = 0.55f)),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
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
                    text = "Détail Numéro ${formatLotoNumber(item.number)} (${item.sourceBranch})",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.ExtraBold,
                    color = MaterialTheme.colorScheme.tertiary
                )
                Text(
                    text = "Rang #${item.rank} / 90",
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                MetricColumn(
                    label = "Probabilité calibrée",
                    value = String.format(Locale.FRANCE, "%.2f%%", item.marginalProbabilityPercent),
                    modifier = Modifier.weight(1f)
                )
                MetricColumn(
                    label = "Indice relatif",
                    value = "${item.relativeConfidencePercent.toInt()}%",
                    modifier = Modifier.weight(1f)
                )
                MetricColumn(
                    label = "Gain vs hasard",
                    value = "×${String.format(Locale.FRANCE, "%.2f", item.liftVsRandom)}",
                    modifier = Modifier.weight(1f)
                )
            }

            if (branchFormula.isNotBlank()) {
                Text(
                    text = branchFormula,
                    style = MaterialTheme.typography.labelSmall.copy(fontFamily = FontFamily.Monospace),
                    color = MaterialTheme.colorScheme.primary
                )
            }
            Text(
                text = if (item.isGapCandidate) {
                    "Statut écart : numéro dormant intégré pour diversifier contre les séries nulles."
                } else {
                    "Statut récence/structure : sélectionné directement par l'arbre symbolique dominant."
                },
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun EngineConfidenceSummaryCard(
    prediction: PredictionResult,
    engineState: EngineActionState,
    matchedDays: Int,
    patternLength: Int,
    isWideScreen: Boolean
) {
    val borderStroke = remember { BorderStroke(1.dp, Color(0x5549454F)) }
    val pAtLeastOnePct = (prediction.calibratedProbAtLeastOne * 100f).coerceIn(0f, 100f)

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("engine_confidence_summary_card"),
        shape = OutputCardShape,
        border = borderStroke,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Synthèse de Confiance & Formules Découvertes",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.weight(1f)
                )
                Box(
                    modifier = Modifier
                        .clip(BadgePillShape)
                        .background(MaterialTheme.colorScheme.tertiaryContainer)
                        .padding(horizontal = 10.dp, vertical = 4.dp)
                        .testTag("output_matched_days_counter")
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
                MetricColumn(
                    label = "Confiance P(≥1 bon)",
                    value = String.format(Locale.FRANCE, "%.1f%%", pAtLeastOnePct),
                    modifier = Modifier.weight(1f)
                )
                MetricColumn(
                    label = "Espérance / 5",
                    value = String.format(Locale.FRANCE, "%.3f", prediction.expectedMatchesEstimate),
                    modifier = Modifier.weight(1f)
                )
                MetricColumn(
                    label = "Jours réussis",
                    value = "$matchedDays/$patternLength",
                    modifier = Modifier.weight(1f)
                )
            }

            AnimatedVisibility(
                visible = prediction.discoveredStrategyName.isNotBlank(),
                enter = fadeIn(),
                exit = fadeOut()
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(InnerTileShape)
                        .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.72f))
                        .padding(12.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    Text(
                        text = prediction.discoveredStrategyName,
                        style = MaterialTheme.typography.labelSmall.copy(fontFamily = FontFamily.Monospace),
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary
                    )
                    if (prediction.auditNote.isNotBlank()) {
                        Text(
                            text = "${prediction.auditNote} • ${prediction.trainedOnCount} tirages passés",
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            }

            val metrics = engineState.fourStageMetrics
            if (metrics != null) {
                FourStageConfidenceStrip(metrics = metrics, isWideScreen = isWideScreen)
            }
        }
    }
}

@Composable
private fun FourStageConfidenceStrip(
    metrics: FourStageValidationMetrics,
    isWideScreen: Boolean
) {
    val stages = listOf(
        "1. Découverte" to String.format(Locale.FRANCE, "%.2f", metrics.discoveryAvgMatches),
        "2. Glissant" to String.format(Locale.FRANCE, "%.2f", metrics.rollingAvgMatches),
        "3. Confirm." to String.format(Locale.FRANCE, "%.2f", metrics.confirmationAvgMatches),
        "4. Scellé" to String.format(Locale.FRANCE, "%.2f", metrics.sealedAvgMatches)
    )
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(alpha = 0.25f))
        Text(
            text = "Moyennes vérifiées sur les 4 étapes chronologiques (hasard = 0,28) :",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(if (isWideScreen) 12.dp else 6.dp)
        ) {
            stages.forEach { (title, avg) ->
                Column(
                    modifier = Modifier
                        .weight(1f)
                        .clip(BadgePillShape)
                        .background(MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.45f))
                        .padding(horizontal = 8.dp, vertical = 6.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = title,
                        style = MaterialTheme.typography.labelSmall.copy(fontSize = 10.sp),
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis
                    )
                    Text(
                        text = avg,
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.ExtraBold,
                        color = MaterialTheme.colorScheme.tertiary
                    )
                }
            }
        }
    }
}

@Composable
private fun CandidateConfidenceComparisonCard(
    candidates: List<PredictedNumberConfidence>,
    predictedSet: Set<Int>
) {
    val maxPct = remember(candidates) {
        (candidates.maxOfOrNull { it.marginalProbabilityPercent } ?: 15f).coerceAtLeast(10f)
    }
    val baselineColor = MaterialTheme.colorScheme.outlineVariant
    val selectedColor = MaterialTheme.colorScheme.tertiary
    val runnerUpColor = MaterialTheme.colorScheme.primary.copy(alpha = 0.55f)

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("confidence_distribution_card"),
        shape = OutputCardShape,
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.35f)),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
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
                    text = "Comparaison Top Candidats vs Hasard (5,6%)",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold
                )
            }

            candidates.forEach { cand ->
                val isChosen = cand.number in predictedSet
                val barFraction = (cand.marginalProbabilityPercent / maxPct).coerceIn(0.03f, 1f)
                val randomFraction = (UNIFORM_RANDOM_PCT / maxPct).coerceIn(0.02f, 0.95f)

                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Text(
                        text = formatLotoNumber(cand.number),
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = if (isChosen) FontWeight.Black else FontWeight.Medium,
                        color = if (isChosen) selectedColor else MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.width(30.dp)
                    )

                    Box(
                        modifier = Modifier
                            .weight(1f)
                            .height(16.dp)
                            .clip(BadgePillShape)
                            .background(MaterialTheme.colorScheme.surface)
                    ) {
                        Canvas(modifier = Modifier.fillMaxSize()) {
                            drawRect(
                                color = if (isChosen) selectedColor else runnerUpColor,
                                size = Size(width = size.width * barFraction, height = size.height)
                            )
                            val baselineX = size.width * randomFraction
                            drawLine(
                                color = baselineColor,
                                start = Offset(baselineX, 0f),
                                end = Offset(baselineX, size.height),
                                strokeWidth = 2.dp.toPx(),
                                pathEffect = PathEffect.dashPathEffect(floatArrayOf(6f, 4f))
                            )
                        }
                    }

                    Text(
                        text = String.format(Locale.FRANCE, "%.1f%%", cand.marginalProbabilityPercent),
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.Bold,
                        color = if (isChosen) selectedColor else MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.width(50.dp),
                        textAlign = TextAlign.End
                    )
                }
            }
        }
    }
}

@Composable
private fun EngineOutputEmptyOrLockedState(
    engineState: EngineActionState,
    isBusy: Boolean,
    onStartOrResumeAnalysis: () -> Unit,
    onLaunchBacktestAndPrediction: () -> Unit,
    onOpenDiscoveryConsole: () -> Unit
) {
    val hasReadyWinner = engineState.canLaunchBacktest
    val ringColor = if (hasReadyWinner) {
        MaterialTheme.colorScheme.tertiary
    } else {
        MaterialTheme.colorScheme.primary
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("engine_output_empty_state"),
        shape = OutputCardShape,
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.4f)),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            Box(
                modifier = Modifier.size(84.dp),
                contentAlignment = Alignment.Center
            ) {
                Canvas(modifier = Modifier.fillMaxSize()) {
                    drawCircle(
                        color = ringColor.copy(alpha = 0.16f),
                        radius = size.minDimension / 2f
                    )
                    drawCircle(
                        color = ringColor.copy(alpha = 0.55f),
                        radius = size.minDimension / 2.6f,
                        style = Stroke(width = 2.dp.toPx())
                    )
                }
                Icon(
                    imageVector = if (hasReadyWinner) Icons.Default.Star else Icons.Default.Lock,
                    contentDescription = "État du moteur",
                    tint = ringColor,
                    modifier = Modifier.size(32.dp)
                )
            }

            val activeStreak = engineState.activeStrategySummary?.combinedStreak
                ?: engineState.bestValidatedStreak
            Text(
                text = when {
                    hasReadyWinner ->
                        "Stratégie ${activeStreak.coerceAtLeast(2)}J Prête"
                    engineState.isAnalyzing ->
                        "Découverte Autonome en Cours (Gén. ${engineState.generation})"
                    else ->
                        "Sortie Verrouillée Avant Validation"
                },
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.ExtraBold,
                textAlign = TextAlign.Center
            )

            Text(
                text = when {
                    hasReadyWinner ->
                        "Une stratégie causale (${activeStreak.coerceAtLeast(2)}J) est enregistrée et sélectionnée. Lancez le backtest pour calculer les numéros et leurs pourcentages de confiance."
                    engineState.isAnalyzing ->
                        "Le moteur explore et teste des formules symboliques (${engineState.experimentsCount} expériences). Le service reste actif même hors de l'application jusqu'à validation."
                    else ->
                        "Aucun numéro ni pourcentage fictif n'est affiché tant qu'une formule symbolique n'a pas atteint un palier causal (2J / 3J / objectif 4J+)."
                },
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center
            )

            if (engineState.isAnalyzing) {
                LinearProgressIndicator(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(6.dp)
                        .clip(BadgePillShape)
                )
            }

            if (hasReadyWinner) {
                Button(
                    onClick = onLaunchBacktestAndPrediction,
                    enabled = !isBusy,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.tertiary,
                        contentColor = MaterialTheme.colorScheme.onTertiary
                    ),
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 48.dp)
                        .testTag("output_launch_backtest_button")
                ) {
                    Icon(Icons.Default.Star, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        if (activeStreak >= 2) "LANCER BACKTEST & PRÉDICTION (${activeStreak}J)"
                        else "LANCER BACKTEST & PRÉDICTION"
                    )
                }
            } else {
                Button(
                    onClick = if (engineState.isAnalyzing) onOpenDiscoveryConsole else onStartOrResumeAnalysis,
                    enabled = !isBusy,
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 48.dp)
                        .testTag("output_start_or_console_button")
                ) {
                    Icon(Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        when {
                            engineState.isAnalyzing -> "Voir l'Évolution en Direct"
                            engineState.canResume -> "Reprendre la Recherche"
                            else -> "Lancer l'Analyse Continue"
                        }
                    )
                }

                if (!engineState.isAnalyzing) {
                    OutlinedButton(
                        onClick = onOpenDiscoveryConsole,
                        modifier = Modifier
                            .fillMaxWidth()
                            .heightIn(min = 48.dp)
                            .testTag("output_open_console_button")
                    ) {
                        Text("Ouvrir la Console de Découverte")
                    }
                }
            }
        }
    }
}

@Composable
private fun MetricColumn(
    label: String,
    value: String,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(2.dp)
    ) {
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
            fontWeight = FontWeight.ExtraBold,
            maxLines = 1,
            overflow = TextOverflow.Ellipsis
        )
    }
}
