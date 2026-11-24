package com.example.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

private val PADDED_NUMBERS = Array(91) { it.toString().padStart(2, '0') }

fun formatLotoNumber(number: Int): String {
    return if (number in 0..90) PADDED_NUMBERS[number] else number.toString().padStart(2, '0')
}

@Composable
fun FeaturedLotoBall(
    number: Int,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .size(48.dp)
            .clip(CircleShape)
            .background(MaterialTheme.colorScheme.primaryContainer),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = formatLotoNumber(number),
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Black,
            color = MaterialTheme.colorScheme.onPrimaryContainer
        )
    }
}

@Composable
fun LotoBall(
    number: Int,
    isMatched: Boolean = false,
    modifier: Modifier = Modifier
) {
    val bgColor = if (isMatched) {
        MaterialTheme.colorScheme.tertiaryContainer
    } else {
        MaterialTheme.colorScheme.surface
    }
    val borderColor = if (isMatched) {
        MaterialTheme.colorScheme.tertiary
    } else {
        MaterialTheme.colorScheme.outlineVariant
    }
    val textColor = if (isMatched) {
        MaterialTheme.colorScheme.onTertiaryContainer
    } else {
        MaterialTheme.colorScheme.onSurface
    }

    Box(
        modifier = modifier
            .size(34.dp)
            .clip(CircleShape)
            .background(bgColor)
            .border(if (isMatched) 1.5.dp else 1.dp, borderColor, CircleShape),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = formatLotoNumber(number),
            style = MaterialTheme.typography.labelMedium,
            fontWeight = if (isMatched) FontWeight.ExtraBold else FontWeight.Bold,
            color = textColor
        )
    }
}
