package com.example.ui.components

import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.animateContentSize
import androidx.compose.foundation.BorderStroke
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
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Create
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import com.example.domain.engine.EngineActionState
import com.example.domain.util.StrategyBackupExport
import com.example.domain.util.StrategyBackupHelper
import com.example.domain.util.StrategyBackupImportResult
import kotlinx.coroutines.launch

private val BackupCardShape = RoundedCornerShape(18.dp)

@Composable
fun StrategyBackupActionsCard(
    engineState: EngineActionState,
    isBusy: Boolean,
    onExportStrategiesBackup: suspend () -> Result<StrategyBackupExport>,
    onImportStrategiesBackup: suspend (String) -> Result<StrategyBackupImportResult>,
    tagPrefix: String = "",
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    var localWorking by remember { mutableStateOf(false) }
    var feedbackMessage by rememberSaveable(tagPrefix) { mutableStateOf<String?>(null) }
    var feedbackIsError by rememberSaveable(tagPrefix) { mutableStateOf(false) }
    var pendingSavePayload by remember { mutableStateOf<String?>(null) }
    var pendingSaveFileName by remember { mutableStateOf("loto_bj_methodes.json") }

    var showPasteDialog by rememberSaveable(tagPrefix) { mutableStateOf(false) }
    var manualPasteText by rememberSaveable(tagPrefix) { mutableStateOf("") }
    var pasteDialogError by rememberSaveable(tagPrefix) { mutableStateOf<String?>(null) }

    val hasExportable = engineState.savedWinners.isNotEmpty() ||
        engineState.hasSavedWinner ||
        engineState.currentPrimaryExpr.isNotBlank()

    val saveDocumentLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("application/json")
    ) { uri ->
        val payload = pendingSavePayload
        val fileName = pendingSaveFileName
        pendingSavePayload = null
        if (uri != null && payload != null) {
            StrategyBackupHelper.writeBackupToUri(context, uri, payload)
                .onSuccess {
                    val msg = "Sauvegarde enregistrée ($fileName)."
                    feedbackMessage = msg
                    feedbackIsError = false
                    Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                }
                .onFailure { err ->
                    val msg = err.message ?: "Impossible d'enregistrer la sauvegarde."
                    feedbackMessage = msg
                    feedbackIsError = true
                    Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                }
        }
    }

    val openDocumentLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            scope.launch {
                localWorking = true
                StrategyBackupHelper.readBackupFromUri(context, uri)
                    .fold(
                        onSuccess = { raw ->
                            onImportStrategiesBackup(raw)
                                .onSuccess { imported ->
                                    feedbackMessage = imported.message
                                    feedbackIsError = false
                                    Toast.makeText(context, imported.message, Toast.LENGTH_SHORT).show()
                                }
                                .onFailure { err ->
                                    val msg = err.message ?: "Fichier de sauvegarde invalide."
                                    feedbackMessage = msg
                                    feedbackIsError = true
                                    Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                                }
                        },
                        onFailure = { err ->
                            val msg = err.message ?: "Impossible de lire le fichier sélectionné."
                            feedbackMessage = msg
                            feedbackIsError = true
                            Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                        }
                    )
                localWorking = false
            }
        }
    }

    val actionDisabled = isBusy || localWorking
    val tertiary = MaterialTheme.colorScheme.tertiary
    val primary = MaterialTheme.colorScheme.primary
    val cardGradient = Brush.linearGradient(
        colors = listOf(
            MaterialTheme.colorScheme.surfaceVariant,
            MaterialTheme.colorScheme.secondaryContainer.copy(alpha = 0.28f)
        )
    )

    Card(
        modifier = modifier
            .fillMaxWidth()
            .animateContentSize()
            .testTag("${tagPrefix}strategy_backup_card"),
        shape = BackupCardShape,
        border = BorderStroke(
            width = 1.dp,
            color = if (hasExportable) tertiary.copy(alpha = 0.45f)
            else MaterialTheme.colorScheme.outline.copy(alpha = 0.35f)
        ),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)
    ) {
        Column(
            modifier = Modifier
                .background(cardGradient)
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            Column(verticalArrangement = Arrangement.spacedBy(2.dp)) {
                Text(
                    text = "Sauvegarde & Transfert des Meilleures Méthodes",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.ExtraBold
                )
                Text(
                    text = if (engineState.savedWinners.isNotEmpty()) {
                        val maxStreak = engineState.savedWinners.maxOf { it.combinedStreak }
                        "${engineState.savedWinners.size} méthode(s) gagnante(s) enregistrée(s) (max ${maxStreak}J) — transférable vers une autre appli."
                    } else if (hasExportable) {
                        "Meilleur génome en cours prêt à sauvegarder ou restaurer depuis une autre appli."
                    } else {
                        "Importez une sauvegarde copiée ou reçue depuis une autre installation Loto BJ."
                    },
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                OutlinedButton(
                    onClick = {
                        scope.launch {
                            localWorking = true
                            onExportStrategiesBackup()
                                .onSuccess { export ->
                                    val copied = StrategyBackupHelper.copyBackupToClipboard(
                                        context = context,
                                        payload = export.jsonPayload
                                    )
                                    val msg = if (copied) {
                                        if (export.strategiesCount > 0) {
                                            "Sauvegarde de ${export.strategiesCount} méthode(s) (max ${export.bestStreak}J) copiée — collez dans l'autre appli."
                                        } else {
                                            "Checkpoint sauvegardé copié — collez dans l'autre appli."
                                        }
                                    } else {
                                        "Presse-papiers indisponible."
                                    }
                                    feedbackMessage = msg
                                    feedbackIsError = !copied
                                    Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                                }
                                .onFailure { err ->
                                    val msg = err.message ?: "Aucune méthode à copier."
                                    feedbackMessage = msg
                                    feedbackIsError = true
                                    Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                                }
                            localWorking = false
                        }
                    },
                    enabled = !actionDisabled && hasExportable,
                    modifier = Modifier
                        .weight(1f)
                        .heightIn(min = 48.dp)
                        .testTag("${tagPrefix}copy_strategies_backup_button")
                ) {
                    Icon(Icons.Default.Check, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Copier", maxLines = 1, overflow = TextOverflow.Ellipsis)
                }

                Button(
                    onClick = {
                        scope.launch {
                            localWorking = true
                            onExportStrategiesBackup()
                                .onSuccess { export ->
                                    runCatching {
                                        val shareIntent = StrategyBackupHelper.createShareBackupIntent(
                                            context = context,
                                            payload = export.jsonPayload,
                                            fileName = export.fileName,
                                            strategiesCount = export.strategiesCount,
                                            bestStreak = export.bestStreak
                                        )
                                        context.startActivity(shareIntent)
                                    }.onSuccess {
                                        feedbackMessage = "Sauvegarde prête pour transfert : ${export.fileName}"
                                        feedbackIsError = false
                                    }.onFailure {
                                        val msg = "Aucune application de transfert disponible."
                                        feedbackMessage = msg
                                        feedbackIsError = true
                                        Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                                    }
                                }
                                .onFailure { err ->
                                    val msg = err.message ?: "Aucune méthode à envoyer."
                                    feedbackMessage = msg
                                    feedbackIsError = true
                                    Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                                }
                            localWorking = false
                        }
                    },
                    enabled = !actionDisabled && hasExportable,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = primary,
                        contentColor = MaterialTheme.colorScheme.onPrimary
                    ),
                    modifier = Modifier
                        .weight(1.25f)
                        .heightIn(min = 48.dp)
                        .testTag("${tagPrefix}share_strategies_backup_button")
                ) {
                    Icon(Icons.Default.Share, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Envoyer .json", maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
            }

            HorizontalDivider(color = MaterialTheme.colorScheme.outline.copy(alpha = 0.2f))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                FilledTonalButton(
                    onClick = {
                        val clipText = StrategyBackupHelper.readTextFromClipboard(context)
                        if (clipText.isNullOrBlank()) {
                            manualPasteText = ""
                            pasteDialogError = null
                            showPasteDialog = true
                        } else {
                            scope.launch {
                                localWorking = true
                                onImportStrategiesBackup(clipText)
                                    .onSuccess { imported ->
                                        feedbackMessage = imported.message
                                        feedbackIsError = false
                                        Toast.makeText(context, imported.message, Toast.LENGTH_SHORT).show()
                                    }
                                    .onFailure { err ->
                                        manualPasteText = clipText
                                        pasteDialogError = err.message ?: "Sauvegarde non reconnue."
                                        showPasteDialog = true
                                    }
                                localWorking = false
                            }
                        }
                    },
                    enabled = !actionDisabled,
                    modifier = Modifier
                        .weight(1f)
                        .heightIn(min = 48.dp)
                        .testTag("${tagPrefix}paste_strategies_backup_button")
                ) {
                    Icon(Icons.Default.Add, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Coller", maxLines = 1, overflow = TextOverflow.Ellipsis)
                }

                OutlinedButton(
                    onClick = {
                        openDocumentLauncher.launch(arrayOf("application/json", "text/plain", "*/*"))
                    },
                    enabled = !actionDisabled,
                    modifier = Modifier
                        .weight(1.25f)
                        .heightIn(min = 48.dp)
                        .testTag("${tagPrefix}import_strategies_file_button")
                ) {
                    Icon(Icons.Default.Create, contentDescription = null, modifier = Modifier.size(16.dp))
                    Spacer(modifier = Modifier.width(6.dp))
                    Text("Importer fichier", maxLines = 1, overflow = TextOverflow.Ellipsis)
                }
            }

            OutlinedButton(
                onClick = {
                    scope.launch {
                        localWorking = true
                        onExportStrategiesBackup()
                            .onSuccess { export ->
                                pendingSavePayload = export.jsonPayload
                                pendingSaveFileName = export.fileName
                                saveDocumentLauncher.launch(export.fileName)
                            }
                            .onFailure { err ->
                                val msg = err.message ?: "Aucune méthode à enregistrer."
                                feedbackMessage = msg
                                feedbackIsError = true
                                Toast.makeText(context, msg, Toast.LENGTH_SHORT).show()
                            }
                        localWorking = false
                    }
                },
                enabled = !actionDisabled && hasExportable,
                modifier = Modifier
                    .fillMaxWidth()
                    .heightIn(min = 48.dp)
                    .testTag("${tagPrefix}save_strategies_backup_button")
            ) {
                Icon(Icons.Default.Create, contentDescription = null, modifier = Modifier.size(16.dp))
                Spacer(modifier = Modifier.width(6.dp))
                Text("Enregistrer la sauvegarde (.json)")
            }

            feedbackMessage?.let { msg ->
                Text(
                    text = msg,
                    style = MaterialTheme.typography.labelSmall,
                    fontWeight = FontWeight.SemiBold,
                    color = if (feedbackIsError) MaterialTheme.colorScheme.error else tertiary,
                    modifier = Modifier.testTag("${tagPrefix}strategies_backup_feedback")
                )
            }
        }
    }

    if (showPasteDialog) {
        AlertDialog(
            onDismissRequest = { showPasteDialog = false },
            modifier = Modifier.testTag("${tagPrefix}strategies_paste_dialog"),
            title = { Text("Importer une sauvegarde") },
            text = {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        text = "Collez ici le bloc JSON de sauvegarde copié depuis une autre application Loto BJ :",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    OutlinedTextField(
                        value = manualPasteText,
                        onValueChange = {
                            manualPasteText = it
                            pasteDialogError = null
                        },
                        label = { Text("Contenu JSON de sauvegarde") },
                        minLines = 4,
                        maxLines = 8,
                        isError = pasteDialogError != null,
                        modifier = Modifier
                            .fillMaxWidth()
                            .testTag("${tagPrefix}strategies_paste_input")
                    )
                    pasteDialogError?.let { errText ->
                        Text(
                            text = errText,
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.error
                        )
                    }
                }
            },
            confirmButton = {
                Button(
                    onClick = {
                        scope.launch {
                            localWorking = true
                            onImportStrategiesBackup(manualPasteText)
                                .onSuccess { imported ->
                                    showPasteDialog = false
                                    pasteDialogError = null
                                    feedbackMessage = imported.message
                                    feedbackIsError = false
                                    Toast.makeText(context, imported.message, Toast.LENGTH_SHORT).show()
                                }
                                .onFailure { err ->
                                    pasteDialogError = err.message ?: "Format de sauvegarde invalide."
                                }
                            localWorking = false
                        }
                    },
                    enabled = !actionDisabled && manualPasteText.isNotBlank(),
                    modifier = Modifier.testTag("${tagPrefix}strategies_paste_confirm")
                ) {
                    Text("Importer")
                }
            },
            dismissButton = {
                TextButton(onClick = { showPasteDialog = false }) {
                    Text("Annuler")
                }
            }
        )
    }
}
