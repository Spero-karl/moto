package com.example.ui.components

import androidx.compose.animation.animateContentSize
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
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.Lock
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
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.example.domain.engine.BacktestResult
import com.example.domain.engine.EngineActionState
import com.example.domain.engine.SymbolicFormulaSummary
import java.util.Locale

private val SelectorCardShape = RoundedCornerShape(18.dp)
private val SelectorInnerShape = RoundedCornerShape(12.dp)
private val SelectorPillShape = RoundedCornerShape(10.dp)

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun BacktestStrategySelectorCard(
    engineState: EngineActionState,
    isBusy: Boolean,
    onSelectPreferredDays: (Int?) -> Unit,
    onSelectWinner: (String) -> Unit,
    onUnlockPredictiveModeling: (() -> Unit)? = null,
    backtestResult: BacktestResult? = null,
    modifier: Modifier = Modifier
) {
    val preferredDays = engineState.preferredStrategyDays
    val matchingWinners = engineState.matchingSavedWinners
    val activeStrategy = engineState.activeStrategySummary
    val isUnlocked = engineState.canLaunchBacktest
    val requiredDays = preferredDays ?: engineState.targetObjectiveStreak.coerceAtLeast(2)
    val bestObservedStreak = maxOf(engineState.bestCombinedStreak, engineState.bestValidatedStreak)
    val matchedDaysCount = engineState.matchedDaysCount(backtestResult)
    val patternLength = engineState.selectedPatternLength

    val dynamicTiers = remember(engineState.savedWinners, preferredDays) {
        val base = sortedSetOf(1, 2, 3, 4, 5)
        engineState.savedWinners.forEach { if (it.combinedStreak >= 1) base.add(it.combinedStreak) }
        if (preferredDays != null && preferredDays >= 1) base.add(preferredDays)
        base.toList()
    }

    var customDays by rememberSaveable(preferredDays) {
        mutableIntStateOf((preferredDays ?: 5).coerceIn(2, 14))
    }

    val tertiary = MaterialTheme.colorScheme.tertiary
    val primary = MaterialTheme.colorScheme.primary
    val borderStroke = remember(isUnlocked, tertiary, primary) {
        BorderStroke(
            width = 1.25.dp,
            color = if (isUnlocked) tertiary.copy(alpha = 0.65f) else primary.copy(alpha = 0.35f)
        )
    }
    val cardGradient = Brush.linearGradient(
        colors = listOf(
            if (isUnlocked) MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.28f)
            else MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.35f),
            MaterialTheme.colorScheme.surfaceVariant
        )
    )

    Card(
        modifier = modifier
            .fillMaxWidth()
            .animateContentSize()
            .testTag("strategy_selector_card"),
        shape = SelectorCardShape,
        border = borderStroke,
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
    ) {
        Column(
            modifier = Modifier
                .background(cardGradient)
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column(modifier = Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
                    Text(
                        text = "Stratégie de Backtest & Modélisation",
                        style = MaterialTheme.typography.titleSmall,
                        fontWeight = FontWeight.ExtraBold
                    )
                    Text(
                        text = if (preferredDays == null) {
                            "Auto : active automatiquement la plus longue série découverte"
                        } else {
                            "Motif requis : ≥2 bons sur ≥3 tirages/jour × $preferredDays j consécutifs"
                        },
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }

                val statusBadgeBg = if (isUnlocked) {
                    MaterialTheme.colorScheme.tertiaryContainer
                } else {
                    MaterialTheme.colorScheme.surface
                }
                val statusBadgeFg = if (isUnlocked) {
                    MaterialTheme.colorScheme.onTertiaryContainer
                } else {
                    MaterialTheme.colorScheme.onSurfaceVariant
                }
                Column(horizontalAlignment = Alignment.End, verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Row(
                        modifier = Modifier
                            .clip(SelectorPillShape)
                            .background(statusBadgeBg)
                            .padding(horizontal = 9.dp, vertical = 5.dp)
                            .testTag("strategy_unlock_badge"),
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                    ) {
                        Icon(
                            imageVector = if (isUnlocked) Icons.Default.CheckCircle else Icons.Default.Lock,
                            contentDescription = if (isUnlocked) "Débloqué" else "Verrouillé",
                            tint = statusBadgeFg,
                            modifier = Modifier.size(14.dp)
                        )
                        Text(
                            text = if (isUnlocked) {
                                "PRÊT • ${activeStrategy?.combinedStreak ?: engineState.bestValidatedStreak}J"
                            } else {
                                "CIBLE ${requiredDays}J"
                            },
                            style = MaterialTheme.typography.labelSmall,
                            fontWeight = FontWeight.ExtraBold,
                            color = statusBadgeFg,
                            maxLines = 1
                        )
                    }
                    Text(
                        text = "Jours réussis : $matchedDaysCount/$patternLength",
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.ExtraBold,
                        color = if (matchedDaysCount >= patternLength) tertiary else primary,
                        modifier = Modifier.testTag("strategy_matched_days_counter")
                    )
                }
            }

            LazyRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                modifier = Modifier
                    .fillMaxWidth()
                    .testTag("strategy_tier_list")
            ) {
                item(key = "tier_auto") {
                    val maxSaved = engineState.savedWinners.firstOrNull()?.combinedStreak
                    FilterChip(
                        selected = preferredDays == null,
                        onClick = { onSelectPreferredDays(null) },
                        enabled = !isBusy,
                        leadingIcon = if (preferredDays == null) {
                            { Icon(Icons.Default.Check, contentDescription = null, modifier = Modifier.size(16.dp)) }
                        } else null,
                        label = {
                            Text(
                                if (maxSaved != null) "Auto (${maxSaved}J max)"
                                else "Auto (Plus forte)"
                            )
                        },
                        modifier = Modifier
                            .heightIn(min = 48.dp)
                            .testTag("strategy_tier_auto")
                    )
                }

                items(dynamicTiers, key = { it }) { tierDays ->
                    val tierMatchesCount = remember(engineState.savedWinners, tierDays) {
                        val exact = engineState.savedWinners.count { it.combinedStreak == tierDays }
                        if (exact > 0) exact else engineState.savedWinners.count { it.combinedStreak >= tierDays }
                    }
                    val isSelected = preferredDays == tierDays
                    FilterChip(
                        selected = isSelected,
                        onClick = { onSelectPreferredDays(tierDays) },
                        enabled = !isBusy,
                        leadingIcon = if (tierMatchesCount > 0) {
                            {
                                Icon(
                                    imageVector = Icons.Default.Star,
                                    contentDescription = "Palier ${tierDays}J disponible",
                                    tint = if (isSelected) MaterialTheme.colorScheme.onSecondaryContainer else tertiary,
                                    modifier = Modifier.size(15.dp)
                                )
                            }
                        } else null,
                        label = {
                            Text(
                                if (tierMatchesCount > 0) "${tierDays}J ($tierMatchesCount)"
                                else "${tierDays}J"
                            )
                        },
                        modifier = Modifier
                            .heightIn(min = 48.dp)
                            .testTag("strategy_tier_$tierDays")
                    )
                }
            }

            // Selecteur personnalise N-Jours consecutifs (2..14 jours) sans debordement sur petit ecran
            FlowRow(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(SelectorInnerShape)
                    .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.72f))
                    .padding(horizontal = 10.dp, vertical = 6.dp),
                verticalArrangement = Arrangement.spacedBy(6.dp),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Text(
                    text = "Palier N jours :",
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.align(Alignment.CenterVertically)
                )

                OutlinedButton(
                    onClick = { if (customDays > 2) customDays -= 1 },
                    enabled = !isBusy && customDays > 2,
                    contentPadding = PaddingValues(0.dp),
                    modifier = Modifier
                        .minimumInteractiveComponentSize()
                        .size(48.dp)
                        .testTag("strategy_n_stepper_minus")
                ) {
                    Text(
                        text = "−",
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.Bold
                    )
                }

                Text(
                    text = "${customDays}J",
                    style = MaterialTheme.typography.labelLarge,
                    fontWeight = FontWeight.ExtraBold,
                    modifier = Modifier
                        .align(Alignment.CenterVertically)
                        .testTag("strategy_n_stepper_value")
                )

                OutlinedButton(
                    onClick = { if (customDays < 14) customDays += 1 },
                    enabled = !isBusy && customDays < 14,
                    contentPadding = PaddingValues(0.dp),
                    modifier = Modifier
                        .minimumInteractiveComponentSize()
                        .size(48.dp)
                        .testTag("strategy_n_stepper_plus")
                ) {
                    Icon(
                        imageVector = Icons.Default.Add,
                        contentDescription = "Augmenter N jours",
                        modifier = Modifier.size(18.dp)
                    )
                }

                FilledTonalButton(
                    onClick = { onSelectPreferredDays(customDays) },
                    enabled = !isBusy && preferredDays != customDays,
                    modifier = Modifier
                        .heightIn(min = 48.dp)
                        .testTag("strategy_n_apply")
                ) {
                    Text(if (preferredDays == customDays) "Actif" else "Appliquer ${customDays}J")
                }
            }

            if (!isUnlocked) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clip(SelectorInnerShape)
                        .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.65f))
                        .padding(10.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    val ratio = (bestObservedStreak.toFloat() / requiredDays.coerceAtLeast(1).toFloat())
                        .coerceIn(0f, 1f)
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween
                    ) {
                        Text(
                            text = if (preferredDays != null) {
                                "Palier ${preferredDays}J en attente de découverte"
                            } else {
                                "Progression vers le premier palier causal"
                            },
                            style = MaterialTheme.typography.labelSmall,
                            fontWeight = FontWeight.SemiBold
                        )
                        Text(
                            text = "$bestObservedStreak / ${requiredDays}j",
                            style = MaterialTheme.typography.labelSmall,
                            fontWeight = FontWeight.Bold,
                            color = primary
                        )
                    }
                    LinearProgressIndicator(
                        progress = { ratio },
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(6.dp)
                            .clip(SelectorPillShape)
                    )
                    if (preferredDays != null && engineState.savedWinners.isNotEmpty()) {
                        OutlinedButton(
                            onClick = { onSelectPreferredDays(null) },
                            enabled = !isBusy,
                            modifier = Modifier
                                .fillMaxWidth()
                                .heightIn(min = 48.dp)
                                .testTag("strategy_fallback_auto_button")
                        ) {
                            Text(
                                "Utiliser la stratégie ${engineState.savedWinners.first().combinedStreak}J déjà débloquée"
                            )
                        }
                    }
                }
            } else if (onUnlockPredictiveModeling != null && activeStrategy != null) {
                Button(
                    onClick = onUnlockPredictiveModeling,
                    enabled = !isBusy,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = tertiary,
                        contentColor = MaterialTheme.colorScheme.onTertiary
                    ),
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(min = 48.dp)
                        .testTag("strategy_unlock_cta")
                ) {
                    Icon(Icons.Default.Star, contentDescription = null, modifier = Modifier.size(18.dp))
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("MODÉLISER AVEC LA STRATÉGIE ${activeStrategy.combinedStreak}J")
                }
            }

            if (matchingWinners.isNotEmpty()) {
                HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(alpha = 0.25f))
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .testTag("saved_winners_card"),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        text = if (preferredDays == null) {
                            "Gagnants Enregistrés (${matchingWinners.size}) — Choisir (2J / 3J / 4J+)"
                        } else {
                            "Formules ${preferredDays}J Disponibles (${matchingWinners.size})"
                        },
                        style = MaterialTheme.typography.labelLarge,
                        fontWeight = FontWeight.ExtraBold,
                        color = tertiary
                    )
                    matchingWinners.take(8).forEach { winner ->
                        StrategyFormulaRow(
                            winner = winner,
                            isSelected = winner.id == (activeStrategy?.id ?: engineState.activeWinnerId),
                            isBusy = isBusy,
                            onSelectWinner = onSelectWinner
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun StrategyFormulaRow(
    winner: SymbolicFormulaSummary,
    isSelected: Boolean,
    isBusy: Boolean,
    onSelectWinner: (String) -> Unit
) {
    val isPrimaryGoal = winner.combinedStreak >= 4
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(SelectorInnerShape)
            .background(
                if (isSelected) MaterialTheme.colorScheme.tertiaryContainer.copy(alpha = 0.55f)
                else MaterialTheme.colorScheme.surface.copy(alpha = 0.72f)
            )
            .padding(horizontal = 10.dp, vertical = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .clip(SelectorPillShape)
                .background(
                    if (isPrimaryGoal) MaterialTheme.colorScheme.tertiaryContainer
                    else MaterialTheme.colorScheme.primaryContainer
                )
                .padding(horizontal = 8.dp, vertical = 6.dp),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = "${winner.combinedStreak}J",
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Black,
                color = if (isPrimaryGoal) {
                    MaterialTheme.colorScheme.onTertiaryContainer
                } else {
                    MaterialTheme.colorScheme.onPrimaryContainer
                }
            )
        }
        Column(
            modifier = Modifier.weight(1f),
            verticalArrangement = Arrangement.spacedBy(2.dp)
        ) {
            Text(
                text = "Gén.${winner.generation} [${winner.slotSplitLabel}] ${winner.primaryExprText}",
                style = MaterialTheme.typography.labelSmall.copy(fontFamily = FontFamily.Monospace),
                fontWeight = FontWeight.Bold,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
            Text(
                text = "${if (winner.isValidated) "Validé 4 niv." else "Palier ${winner.combinedStreak}J"} • Gliss=${formatAvg3(winner.rollingAvgMatches)} • Conf=${formatAvg3(winner.confirmationAvgMatches)}",
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis
            )
        }
        OutlinedButton(
            onClick = { onSelectWinner(winner.id) },
            enabled = !isBusy,
            modifier = Modifier
                .heightIn(min = 48.dp)
                .testTag("use_saved_winner_${winner.id}")
        ) {
            if (isSelected) {
                Icon(Icons.Default.Check, contentDescription = null, modifier = Modifier.size(16.dp))
                Spacer(modifier = Modifier.width(4.dp))
                Text("Actif")
            } else {
                Text("Choisir")
            }
        }
    }
}

private fun formatAvg3(value: Float): String = String.format(Locale.FRANCE, "%.3f", value)
