package com.example.ui.screens

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
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
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.example.domain.engine.BacktestPeriod
import com.example.domain.engine.ComplementarityPair
import com.example.domain.engine.MonteCarloBaseline
import com.example.domain.engine.PredictionMethod
import com.example.domain.engine.SavedModelLabAudit
import com.example.domain.engine.ScientificReport
import com.example.domain.engine.StrategyComparisonEntry
import com.example.domain.engine.WindowSplitInfo
import java.util.Locale

private val CardShape = RoundedCornerShape(16.dp)
private val InnerCardShape = RoundedCornerShape(12.dp)
private val BadgeShape = RoundedCornerShape(8.dp)

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ComparisonScreen(
    report: ScientificReport?,
    resultsCount: Int,
    isComparing: Boolean,
    onRunComparison: () -> Unit,
    onSelectMethod: (PredictionMethod) -> Unit,
    onSelectSavedModel: (String) -> Unit = {},
    selectedPeriod: BacktestPeriod = BacktestPeriod.THREE_WEEKS,
    customPeriodDays: Int? = selectedPeriod.days,
    modifier: Modifier = Modifier
) {
    val listState = rememberLazyListState()
    var selectedWindow by rememberSaveable { mutableIntStateOf(2) } // 0=Dev, 1=Val, 2=Test Independant
    var showTechnicalDetails by rememberSaveable { mutableStateOf(false) }

    val activePeriodText = report?.periodLabel?.takeIf { it.isNotBlank() }
        ?: customPeriodDays?.let { if (it == 1) "1 Jour (6 tirages)" else "$it Jours" }
        ?: selectedPeriod.label

    LazyColumn(
        state = listState,
        modifier = modifier.fillMaxSize(),
        contentPadding = PaddingValues(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        item(key = "comp_header", contentType = "header") {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(
                    text = "Laboratoire Scientifique (Zéro Triche)",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "Audit simple par OUI / NON sur la fenêtre active ($activePeriodText). " +
                        "Objectif obligatoire : ≥ 3 tirages/jour avec ≥ 2 boules exactes dont l'avant-dernier tirage ≥ 2.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Button(
                    onClick = onRunComparison,
                    enabled = !isComparing && resultsCount >= 8,
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 48.dp)
                        .testTag("run_comparison_button")
                ) {
                    if (isComparing) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(18.dp),
                            strokeWidth = 2.dp,
                            color = MaterialTheme.colorScheme.onPrimary
                        )
                        Spacer(modifier = Modifier.width(8.dp))
                        Text("Audit des modèles & 10 000 simulations...")
                    } else {
                        Text("Auditer mes Modèles & 13 Méthodes ($resultsCount tir.)")
                    }
                }
            }
        }

        if (report == null && !isComparing) {
            item(key = "comp_empty", contentType = "empty") {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = CardShape,
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
                ) {
                    Row(
                        modifier = Modifier.padding(16.dp),
                        horizontalArrangement = Arrangement.spacedBy(12.dp),
                        verticalAlignment = Alignment.Top
                    ) {
                        Icon(Icons.Default.Info, contentDescription = "Information laboratoire")
                        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                            Text(
                                text = "Comment lire ce laboratoire ?",
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.Bold
                            )
                            Text(
                                text = "1) Vos modèles IA / Chercheur conservés sont testés sans triche sur la fenêtre active.\n" +
                                    "2) Chaque modèle reçoit un verdict clair : OUI ou NON sur l'objectif et sur l'avant-dernier tirage.\n" +
                                    "3) Les 13 méthodes mathématiques classiques servent uniquement de repère de comparaison.",
                                style = MaterialTheme.typography.bodySmall
                            )
                        }
                    }
                }
            }
        }

        if (report != null) {
            item(key = "lab_verdict_card", contentType = "verdict") {
                val ok = report.anyObjectiveValidated
                val container = if (ok) {
                    MaterialTheme.colorScheme.tertiaryContainer
                } else {
                    MaterialTheme.colorScheme.errorContainer
                }
                val content = if (ok) {
                    MaterialTheme.colorScheme.onTertiaryContainer
                } else {
                    MaterialTheme.colorScheme.onErrorContainer
                }
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .testTag("lab_verdict_card"),
                    shape = CardShape,
                    colors = CardDefaults.cardColors(
                        containerColor = container,
                        contentColor = content
                    ),
                    border = BorderStroke(
                        1.5.dp,
                        if (ok) MaterialTheme.colorScheme.tertiary else MaterialTheme.colorScheme.error
                    )
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
                                text = "Verdict Global",
                                style = MaterialTheme.typography.titleMedium,
                                fontWeight = FontWeight.ExtraBold
                            )
                            OuiNonPill(
                                yes = ok,
                                yesLabel = "OUI — OBJECTIF VALIDÉ",
                                noLabel = "NON — OBJECTIF NON VALIDÉ"
                            )
                        }
                        Text(
                            text = report.recommendationReason,
                            style = MaterialTheme.typography.bodySmall,
                            fontWeight = FontWeight.SemiBold
                        )
                        HorizontalDivider(color = content.copy(alpha = 0.2f))
                        Text(
                            text = "Règle : ≥ 3 tirages/jour avec ≥ 2 boules exactes dont l'avant-dernier tirage ≥ 2 • Fenêtre : ${report.periodLabel.ifBlank { activePeriodText }}",
                            style = MaterialTheme.typography.labelSmall
                        )
                    }
                }
            }

            item(key = "saved_models_header", contentType = "header") {
                Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text(
                        text = "Modèles IA & Chercheur Conservés (${report.savedModelAudits.size})",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "Meilleurs modèles enregistrés par vos recherches et audités sans fuite future.",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            if (report.savedModelAudits.isEmpty()) {
                item(key = "saved_models_empty", contentType = "empty_saved") {
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .testTag("lab_saved_models_empty"),
                        shape = CardShape,
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
                    ) {
                        Text(
                            text = "Aucun modèle IA conservé pour l'instant. Lancez le Moteur dans l'onglet « Moteur » pour découvrir et enregistrer vos premiers modèles gagnants.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.padding(14.dp)
                        )
                    }
                }
            } else {
                items(
                    items = report.savedModelAudits,
                    key = { "saved_${it.id}" },
                    contentType = { "saved_model" }
                ) { audit ->
                    SavedModelAuditCard(
                        audit = audit,
                        onUseModel = { onSelectSavedModel(audit.id) }
                    )
                }
            }

            item(key = "classical_header", contentType = "header") {
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text(
                        text = "13 Méthodes Statistiques (Repères)",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "Comparaison honnête contre les formules classiques. Touchez une méthode pour la sélectionner.",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }

            item(key = "window_tabs", contentType = "tabs") {
                FlowRow(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    FilterChip(
                        selected = selectedWindow == 2,
                        onClick = { selectedWindow = 2 },
                        label = { Text("Test Indép. (${report.splitInfo.indepTestDrawsCount})") },
                        modifier = Modifier.testTag("window_indep")
                    )
                    FilterChip(
                        selected = selectedWindow == 1,
                        onClick = { selectedWindow = 1 },
                        label = { Text("Validation (${report.splitInfo.valDrawsCount})") },
                        modifier = Modifier.testTag("window_val")
                    )
                    FilterChip(
                        selected = selectedWindow == 0,
                        onClick = { selectedWindow = 0 },
                        label = { Text("Dév. (${report.splitInfo.devDrawsCount})") },
                        modifier = Modifier.testTag("window_dev")
                    )
                }
            }

            val activeEntries = when (selectedWindow) {
                0 -> report.devComparisons
                1 -> report.valComparisons
                else -> report.indepTestComparisons
            }

            items(
                items = activeEntries,
                key = { "${selectedWindow}_${it.method.name}" },
                contentType = { "strategy_row" }
            ) { entry ->
                StrategyComparisonCard(
                    entry = entry,
                    onClickSelect = { onSelectMethod(entry.method) }
                )
            }

            item(key = "toggle_technical_details", contentType = "toggle") {
                OutlinedButton(
                    onClick = { showTechnicalDetails = !showTechnicalDetails },
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 48.dp)
                        .testTag("toggle_lab_technical_details")
                ) {
                    Text(
                        if (showTechnicalDetails) {
                            "Masquer les détails techniques (Monte Carlo, Split, Cube90, Paires)"
                        } else {
                            "Afficher les détails techniques (Monte Carlo, Split, Cube90, Paires)"
                        }
                    )
                }
            }

            if (showTechnicalDetails) {
                item(key = "split_info", contentType = "split") {
                    SplitInfoCard(
                        split = report.splitInfo,
                        periodLabel = report.periodLabel.ifBlank { activePeriodText }
                    )
                }

                val activeMc = when (selectedWindow) {
                    0 -> report.monteCarloDev
                    1 -> report.monteCarloVal
                    else -> report.monteCarloIndepTest
                }
                item(key = "mc_baseline_$selectedWindow", contentType = "mc") {
                    MonteCarloCard(activeMc)
                }

                item(key = "cube90_technical", contentType = "cube90") {
                    Card(
                        modifier = Modifier.fillMaxWidth(),
                        shape = CardShape,
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
                    ) {
                        Column(
                            modifier = Modifier.padding(14.dp),
                            verticalArrangement = Arrangement.spacedBy(4.dp)
                        ) {
                            Text(
                                text = "Audit Mathématique Cube90",
                                style = MaterialTheme.typography.titleSmall,
                                fontWeight = FontWeight.Bold
                            )
                            Text(
                                text = report.cube90AuditSummary,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                            )
                        }
                    }
                }

                if (report.complementaryPairs.isNotEmpty()) {
                    item(key = "comp_pairs_header", contentType = "header") {
                        Text(
                            text = "Paires Complémentaires (Faible Corrélation d'Erreurs)",
                            style = MaterialTheme.typography.titleSmall,
                            fontWeight = FontWeight.Bold
                        )
                    }
                    items(
                        items = report.complementaryPairs,
                        key = { "${it.methodA.name}_${it.methodB.name}" },
                        contentType = { "pair_row" }
                    ) { pair ->
                        ComplementarityCard(pair)
                    }
                }
            }
        }
    }
}

@Composable
private fun OuiNonPill(
    yes: Boolean,
    yesLabel: String = "OUI",
    noLabel: String = "NON"
) {
    val bg = if (yes) MaterialTheme.colorScheme.tertiary else MaterialTheme.colorScheme.error
    val fg = if (yes) MaterialTheme.colorScheme.onTertiary else MaterialTheme.colorScheme.onError
    Box(
        modifier = Modifier
            .clip(BadgeShape)
            .background(bg)
            .padding(horizontal = 8.dp, vertical = 4.dp)
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Icon(
                imageVector = if (yes) Icons.Default.Check else Icons.Default.Close,
                contentDescription = if (yes) yesLabel else noLabel,
                modifier = Modifier.size(14.dp),
                tint = fg
            )
            Text(
                text = if (yes) yesLabel else noLabel,
                style = MaterialTheme.typography.labelSmall,
                fontWeight = FontWeight.ExtraBold,
                color = fg
            )
        }
    }
}

@Composable
private fun OuiNonLine(
    label: String,
    yes: Boolean,
    detail: String
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.weight(1f)
        )
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(6.dp)
        ) {
            Text(
                text = detail,
                style = MaterialTheme.typography.labelSmall,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                text = if (yes) "OUI" else "NON",
                style = MaterialTheme.typography.labelSmall,
                fontWeight = FontWeight.ExtraBold,
                color = if (yes) MaterialTheme.colorScheme.tertiary else MaterialTheme.colorScheme.error
            )
        }
    }
}

@Composable
private fun SavedModelAuditCard(
    audit: SavedModelLabAudit,
    onUseModel: () -> Unit
) {
    val accent = if (audit.objectiveValidated) {
        MaterialTheme.colorScheme.tertiary
    } else {
        MaterialTheme.colorScheme.outline.copy(alpha = 0.45f)
    }
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("saved_model_audit_${audit.id}"),
        shape = CardShape,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        border = BorderStroke(if (audit.objectiveValidated) 1.5.dp else 1.dp, accent)
    ) {
        Column(
            modifier = Modifier.padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = audit.displayName,
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.ExtraBold
                    )
                    Text(
                        text = "${audit.windowName} • Split ${audit.slotSplitLabel}",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                OuiNonPill(
                    yes = audit.objectiveValidated,
                    yesLabel = "VALIDÉ : OUI",
                    noLabel = "VALIDÉ : NON"
                )
            }

            Text(
                text = audit.formulaCompact,
                style = MaterialTheme.typography.labelSmall.copy(fontFamily = FontFamily.Monospace),
                color = MaterialTheme.colorScheme.primary,
                maxLines = 2,
                overflow = TextOverflow.Ellipsis
            )

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(InnerCardShape)
                    .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.7f))
                    .padding(10.dp),
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                OuiNonLine(
                    label = "Jours réussis (≥3 tir. à ≥2 boules)",
                    yes = audit.qualifyingDaysCount > 0,
                    detail = "${audit.qualifyingDaysCount}/${audit.totalDays} j"
                )
                OuiNonLine(
                    label = "Avant-dernier tirage ≥ 2 boules",
                    yes = audit.penultimateOkDaysCount > 0,
                    detail = "${audit.penultimateOkDaysCount}/${audit.totalDays} j"
                )
                OuiNonLine(
                    label = "Bat le hasard ?",
                    yes = audit.beatsRandom,
                    detail = "Moy. ${String.format(Locale.FRANCE, "%.2f", audit.averageMatches)}/5 • ≥2: ${audit.twoPlusDraws}/${audit.testedDraws}"
                )
                OuiNonLine(
                    label = "Zéro triche (causal strict)",
                    yes = audit.causalVerified,
                    detail = "0-boule: ${audit.zeroMatchRatePct}%"
                )
            }

            Text(
                text = audit.verdictReason,
                style = MaterialTheme.typography.labelSmall,
                color = if (audit.objectiveValidated) MaterialTheme.colorScheme.tertiary else MaterialTheme.colorScheme.onSurfaceVariant
            )

            Button(
                onClick = onUseModel,
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (audit.objectiveValidated) {
                        MaterialTheme.colorScheme.tertiary
                    } else {
                        MaterialTheme.colorScheme.primary
                    }
                ),
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 48.dp)
                    .testTag("lab_use_saved_model_${audit.id}")
            ) {
                Icon(Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(18.dp))
                Spacer(modifier = Modifier.width(6.dp))
                Text("Utiliser ce modèle (Backtest & Prédiction)")
            }
        }
    }
}

@Composable
private fun SplitInfoCard(
    split: WindowSplitInfo,
    periodLabel: String
) {
    val outlineColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.35f)
    val border = remember(outlineColor) { BorderStroke(1.dp, outlineColor) }
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = CardShape,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        border = border
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(
                text = "Séparation Chronologique — $periodLabel (${split.totalDrawsAvailable} tirages)",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold
            )
            Text("Warm-up initial : ${split.warmupDrawsCount} tirages", style = MaterialTheme.typography.bodySmall)
            Text("Développement : ${split.devDateRange} (${split.devDrawsCount} tirages)", style = MaterialTheme.typography.bodySmall)
            Text("Validation : ${split.valDateRange} (${split.valDrawsCount} tirages)", style = MaterialTheme.typography.bodySmall)
            Text(
                "Test Final Indépendant : ${split.indepTestDateRange} (${split.indepTestDrawsCount} tirages)",
                style = MaterialTheme.typography.bodySmall,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.tertiary
            )
        }
    }
}

@Composable
private fun MonteCarloCard(mc: MonteCarloBaseline) {
    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("monte_carlo_card"),
        shape = CardShape,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.tertiaryContainer,
            contentColor = MaterialTheme.colorScheme.onTertiaryContainer
        )
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(
                text = "Référence Aléatoire (${mc.simulationsCount} simulations, graine=${mc.seed})",
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold
            )
            val avgStr = String.format(Locale.FRANCE, "%.3f", mc.meanAvgMatches)
            val ciAvg = "${String.format(Locale.FRANCE, "%.3f", mc.avgMatchesCiLow)}–${String.format(Locale.FRANCE, "%.3f", mc.avgMatchesCiHigh)}"
            val rateStr = String.format(Locale.FRANCE, "%.1f", mc.meanOnePlusRate * 100f)
            Text(
                text = "Moy. bons : $avgStr (IC 95% [$ciAvg]) • Taux ≥1 : $rateStr%",
                style = MaterialTheme.typography.bodySmall
            )
            Text(
                text = "Tirages ≥2 bons attendus : ${String.format(Locale.FRANCE, "%.1f", mc.meanTwoPlusCount)} (IC 95% [${mc.twoPlusCiLow}, ${mc.twoPlusCiHigh}]) • Jours qualifiants : ${String.format(Locale.FRANCE, "%.2f", mc.meanQualifyingDays)} (IC 95% [${mc.qualifyingDaysCiLow}, ${mc.qualifyingDaysCiHigh}])",
                style = MaterialTheme.typography.bodySmall
            )
        }
    }
}

@Composable
private fun StrategyComparisonCard(
    entry: StrategyComparisonEntry,
    onClickSelect: () -> Unit
) {
    val outlineColor = when {
        entry.objectiveValidated -> MaterialTheme.colorScheme.tertiary
        entry.isRecommendedParsimonious -> MaterialTheme.colorScheme.primary.copy(alpha = 0.55f)
        else -> MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)
    }
    val border = remember(outlineColor, entry.objectiveValidated) {
        BorderStroke(if (entry.objectiveValidated) 1.5.dp else 1.dp, outlineColor)
    }

    Card(
        onClick = onClickSelect,
        modifier = Modifier
            .fillMaxWidth()
            .testTag("compare_${entry.method.name}"),
        shape = CardShape,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        border = border
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f)) {
                    Text(
                        text = entry.method.displayName,
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "${entry.method.family} • ${entry.testedCount} tirages",
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
                Row(
                    horizontalArrangement = Arrangement.spacedBy(6.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    if (entry.isRecommendedParsimonious && !entry.objectiveValidated) {
                        Box(
                            modifier = Modifier
                                .clip(BadgeShape)
                                .background(MaterialTheme.colorScheme.secondaryContainer)
                                .padding(horizontal = 8.dp, vertical = 4.dp)
                        ) {
                            Text(
                                text = "Repère Stat.",
                                style = MaterialTheme.typography.labelSmall,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.onSecondaryContainer
                            )
                        }
                    }
                    OuiNonPill(
                        yes = entry.objectiveValidated,
                        yesLabel = "OUI • VALIDÉ",
                        noLabel = "NON"
                    )
                }
            }

            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(InnerCardShape)
                    .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.6f))
                    .padding(10.dp),
                verticalArrangement = Arrangement.spacedBy(3.dp)
            ) {
                OuiNonLine(
                    label = "Objectif ≥3 tir./j à ≥2 + avant-dernier",
                    yes = entry.objectiveValidated,
                    detail = "${entry.qualifyingDaysCount}/${entry.totalDays} j"
                )
                OuiNonLine(
                    label = "Avant-dernier tirage ≥ 2",
                    yes = entry.penultimateOkDaysCount > 0,
                    detail = "${entry.penultimateOkDaysCount}/${entry.totalDays} j"
                )
                OuiNonLine(
                    label = "Bat le hasard (statistique) ?",
                    yes = entry.adjustedSignificant,
                    detail = "Moy. ${String.format(Locale.FRANCE, "%.2f", entry.averageMatches)}/5 • ≥2: ${entry.twoPlusCount}"
                )
            }
        }
    }
}

@Composable
private fun ComplementarityCard(pair: ComplementarityPair) {
    val outlineColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)
    val border = remember(outlineColor) { BorderStroke(1.dp, outlineColor) }
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = CardShape,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant),
        border = border
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(14.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "${pair.methodA.shortLabel} + ${pair.methodB.shortLabel}",
                    style = MaterialTheme.typography.bodyMedium,
                    fontWeight = FontWeight.SemiBold
                )
                Text(
                    text = "Chevauchement Jaccard : ${String.format(Locale.FRANCE, "%.1f", pair.jaccardOverlap * 100f)}% • Corr. erreurs : ${String.format(Locale.FRANCE, "%.2f", pair.errorCorrelation)}",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
            Text(
                text = "${pair.complementaryHitsCount} tirages compl.",
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.tertiary
            )
        }
    }
}
