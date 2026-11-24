package com.example.ui.components

import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Create
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.example.domain.engine.BacktestResult
import com.example.domain.engine.EngineActionState
import com.example.domain.engine.NextDrawTarget
import com.example.domain.engine.PredictionResult
import com.example.domain.util.BacktestExportHelper

private val ExportBoxShape = RoundedCornerShape(12.dp)

@Composable
fun BacktestExportActions(
    backtestResult: BacktestResult,
    engineState: EngineActionState,
    prediction: PredictionResult?,
    nextDrawTarget: NextDrawTarget?,
    tagPrefix: String = "",
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    var feedbackMessage by rememberSaveable(
        backtestResult.periodLabel,
        backtestResult.totalTested,
        backtestResult.maxConsecutiveQualifyingDays
    ) { mutableStateOf<String?>(null) }

    val reportContent = remember(backtestResult, engineState, prediction, nextDrawTarget) {
        BacktestExportHelper.formatDetailedBacktestReport(
            backtestResult = backtestResult,
            engineState = engineState,
            prediction = prediction,
            nextDrawTarget = nextDrawTarget
        )
    }
    val suggestedFileName = remember(prediction, backtestResult) {
        BacktestExportHelper.buildExportFileName(prediction = prediction, backtestResult = backtestResult)
    }

    val saveDocumentLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("text/plain")
    ) { uri ->
        if (uri != null) {
            BacktestExportHelper.writeReportToUri(context, uri, reportContent)
                .onSuccess {
                    val msg = "Fichier texte détaillé enregistré ($suggestedFileName)."
                    feedbackMessage = msg
                    Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                }
                .onFailure {
                    val msg = "Impossible d'enregistrer le fichier texte."
                    feedbackMessage = msg
                    Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                }
        }
    }

    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(ExportBoxShape)
            .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.72f))
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        Text(
            text = "Copier ou Transférer le Backtest Détaillé (.txt)",
            style = MaterialTheme.typography.labelMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onSurface
        )
        Text(
            text = "Inclut la synthèse, les 4 niveaux chronologiques, les jours qualifiants et les ${backtestResult.steps.size} tirages détaillés.",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            OutlinedButton(
                onClick = {
                    val copied = BacktestExportHelper.copyToClipboard(context, reportContent)
                    val msg = if (copied) {
                        "Backtest détaillé copié dans le presse-papiers."
                    } else {
                        "Presse-papiers indisponible."
                    }
                    feedbackMessage = msg
                    Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                },
                modifier = Modifier
                    .weight(1f)
                    .heightIn(min = 48.dp)
                    .testTag("${tagPrefix}copy_backtest_button")
            ) {
                Icon(Icons.Default.Check, contentDescription = null, modifier = Modifier.size(16.dp))
                Spacer(modifier = Modifier.width(4.dp))
                Text("Copier")
            }

            Button(
                onClick = {
                    runCatching {
                        val shareIntent = BacktestExportHelper.createShareTextFileIntent(
                            context = context,
                            content = reportContent,
                            fileName = suggestedFileName
                        )
                        context.startActivity(shareIntent)
                    }.onSuccess {
                        feedbackMessage = "Fichier .txt prêt pour transfert : $suggestedFileName"
                    }.onFailure {
                        val msg = "Aucune application de transfert trouvée."
                        feedbackMessage = msg
                        Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                    }
                },
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.primary,
                    contentColor = MaterialTheme.colorScheme.onPrimary
                ),
                modifier = Modifier
                    .weight(1.3f)
                    .heightIn(min = 48.dp)
                    .testTag("${tagPrefix}transfer_backtest_txt_button")
            ) {
                Icon(Icons.Default.Share, contentDescription = null, modifier = Modifier.size(16.dp))
                Spacer(modifier = Modifier.width(6.dp))
                Text("Transférer .txt")
            }
        }

        feedbackMessage?.let { msg ->
            Text(
                text = msg,
                style = MaterialTheme.typography.labelSmall,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.tertiary,
                modifier = Modifier.testTag("${tagPrefix}backtest_export_feedback")
            )
        }

        OutlinedButton(
            onClick = { saveDocumentLauncher.launch(suggestedFileName) },
            modifier = Modifier
                .fillMaxWidth()
                .heightIn(min = 48.dp)
                .testTag("${tagPrefix}save_backtest_txt_button")
        ) {
            Icon(Icons.Default.Create, contentDescription = null, modifier = Modifier.size(16.dp))
            Spacer(modifier = Modifier.width(6.dp))
            Text("Enregistrer le fichier texte (.txt) sur l'appareil")
        }
    }
}
