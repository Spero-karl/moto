package com.example.ui.screens

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.KeyboardArrowRight
import androidx.compose.material.icons.filled.Star
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.data.local.LotoResultEntity
import com.example.domain.util.LotoDateUtils
import com.example.ui.components.FeaturedLotoBall
import com.example.ui.components.LotoBall

private val FeaturedCardShape = RoundedCornerShape(24.dp)
private val DrawCardShape = RoundedCornerShape(16.dp)
private val StarBadgeShape = RoundedCornerShape(12.dp)

@Composable
fun ResultsScreen(
    results: List<LotoResultEntity>,
    isLoading: Boolean,
    errorMessage: String?,
    onRefresh: () -> Unit,
    modifier: Modifier = Modifier
) {
    val listState = rememberLazyListState()

    Box(modifier = modifier.fillMaxSize()) {
        if (isLoading && results.isEmpty()) {
            CircularProgressIndicator(modifier = Modifier.align(Alignment.Center))
        } else if (results.isEmpty()) {
            Column(
                modifier = Modifier
                    .align(Alignment.Center)
                    .padding(24.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text(
                    text = errorMessage ?: "Aucun résultat. Appuyez sur le bouton actualiser.",
                    color = if (errorMessage != null) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.onBackground,
                    textAlign = TextAlign.Center
                )
                Button(
                    onClick = onRefresh,
                    modifier = Modifier.testTag("retry_button")
                ) {
                    Text("Actualiser")
                }
            }
        } else {
            val threeWeekWindow = remember(results) {
                LotoDateUtils.resolveThreeWeekWindow(results)
            }
            val displayedResults = remember(results, threeWeekWindow) {
                threeWeekWindow.draws.asReversed().ifEmpty { results }
            }

            LazyColumn(
                state = listState,
                modifier = Modifier.fillMaxSize(),
                contentPadding = PaddingValues(16.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp)
            ) {
                item(key = "results_three_weeks_banner", contentType = "banner") {
                    Card(
                        modifier = Modifier
                            .fillMaxWidth()
                            .testTag("results_three_weeks_banner"),
                        shape = DrawCardShape,
                        colors = CardDefaults.cardColors(
                            containerColor = MaterialTheme.colorScheme.primaryContainer.copy(alpha = 0.7f),
                            contentColor = MaterialTheme.colorScheme.onPrimaryContainer
                        )
                    ) {
                        Column(
                            modifier = Modifier.padding(horizontal = 14.dp, vertical = 12.dp),
                            verticalArrangement = Arrangement.spacedBy(2.dp)
                        ) {
                            Text(
                                text = "Historique Ciblé — 3 Semaines (21j à la date J)",
                                style = MaterialTheme.typography.labelLarge,
                                fontWeight = FontWeight.Bold
                            )
                            Text(
                                text = "${threeWeekWindow.startLabel} → ${threeWeekWindow.endLabel} (${threeWeekWindow.anchorTimeLabel} Bénin) • ${displayedResults.size} tirages actifs (${results.size} total)",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.85f)
                            )
                        }
                    }
                }
                itemsIndexed(
                    items = displayedResults,
                    key = { _, item -> item.uniqueId },
                    contentType = { index, _ -> if (index == 0) 0 else 1 }
                ) { index, result ->
                    if (index == 0) {
                        FeaturedDrawCard(result)
                    } else {
                        DrawCard(result)
                    }
                }
            }
        }
    }
}

@Composable
fun FeaturedDrawCard(result: LotoResultEntity) {
    val title = remember(result.game, result.time) {
        if (result.time.isNotBlank()) "${result.game} ${result.time}" else result.game
    }
    val dateWithDay = remember(result.timestamp) {
        LotoDateUtils.formatFrenchDayAndDate(result.timestamp)
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("featured_draw_card_${result.uniqueId}"),
        shape = FeaturedCardShape,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.secondaryContainer,
            contentColor = MaterialTheme.colorScheme.onSecondaryContainer
        ),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(modifier = Modifier.padding(20.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.Top
            ) {
                Column {
                    Text(
                        text = "DERNIER TIRAGE",
                        style = MaterialTheme.typography.labelSmall,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.8f),
                        letterSpacing = 1.sp
                    )
                    Text(
                        text = title,
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold
                    )
                }
                Box(
                    modifier = Modifier
                        .clip(StarBadgeShape)
                        .background(MaterialTheme.colorScheme.primaryContainer)
                        .padding(8.dp)
                ) {
                    Icon(
                        Icons.Default.Star,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.onPrimaryContainer
                    )
                }
            }
            Spacer(modifier = Modifier.height(16.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                FeaturedLotoBall(result.num1)
                FeaturedLotoBall(result.num2)
                FeaturedLotoBall(result.num3)
                FeaturedLotoBall(result.num4)
                FeaturedLotoBall(result.num5)
            }
            Spacer(modifier = Modifier.height(14.dp))
            Text(
                text = dateWithDay,
                style = MaterialTheme.typography.labelMedium,
                fontWeight = FontWeight.Medium,
                color = MaterialTheme.colorScheme.onSecondaryContainer.copy(alpha = 0.85f)
            )
        }
    }
}

@Composable
fun DrawCard(result: LotoResultEntity) {
    val outlineColor = MaterialTheme.colorScheme.outline.copy(alpha = 0.3f)
    val borderStroke = remember(outlineColor) { BorderStroke(1.dp, outlineColor) }
    val title = remember(result.game, result.time) {
        if (result.time.isNotBlank()) "${result.game} ${result.time}" else result.game
    }
    val dateWithDay = remember(result.timestamp) {
        LotoDateUtils.formatFrenchDayAndDate(result.timestamp)
    }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .testTag("draw_card_${result.uniqueId}"),
        shape = DrawCardShape,
        colors = CardDefaults.cardColors(
            containerColor = MaterialTheme.colorScheme.surfaceVariant,
            contentColor = MaterialTheme.colorScheme.onSurface
        ),
        border = borderStroke,
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 14.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    modifier = Modifier.padding(bottom = 8.dp)
                ) {
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .clip(CircleShape)
                            .background(MaterialTheme.colorScheme.primary)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = title,
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.SemiBold
                    )
                }
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    LotoBall(result.num1)
                    LotoBall(result.num2)
                    LotoBall(result.num3)
                    LotoBall(result.num4)
                    LotoBall(result.num5)
                }
            }
            Column(horizontalAlignment = Alignment.End) {
                Text(
                    text = dateWithDay,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Icon(
                    Icons.AutoMirrored.Filled.KeyboardArrowRight,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.padding(top = 4.dp)
                )
            }
        }
    }
}
