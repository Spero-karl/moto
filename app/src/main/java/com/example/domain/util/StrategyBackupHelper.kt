package com.example.domain.util

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.core.content.FileProvider
import com.example.data.local.DiscoveredFormulaEntity
import com.example.domain.engine.BacktestPeriod
import com.example.domain.engine.SymbolicFormula
import com.example.domain.engine.SymbolicGenome
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.floatOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull
import kotlinx.serialization.json.put
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class StrategyBackupExport(
    val jsonPayload: String,
    val strategiesCount: Int,
    val hasCheckpoint: Boolean,
    val bestStreak: Int,
    val fileName: String
)

data class StrategyBackupImportResult(
    val importedCount: Int,
    val updatedCount: Int,
    val checkpointRestored: Boolean,
    val bestStreakImported: Int,
    val primaryWinnerId: String?,
    val message: String
)

data class ParsedStrategyBackup(
    val sourcePeriodName: String,
    val exportedAtMs: Long,
    val strategies: List<DiscoveredFormulaEntity>,
    val checkpointBest: DiscoveredFormulaEntity?
)

object StrategyBackupHelper {

    const val BACKUP_FORMAT = "loto_bj_strategies_backup_v1"
    const val BACKUP_VERSION = 1

    private val prettyJson = Json { prettyPrint = true }

    fun serializeBackup(
        strategies: List<DiscoveredFormulaEntity>,
        checkpointBest: DiscoveredFormulaEntity? = null,
        periodName: String = BacktestPeriod.THREE_WEEKS.name,
        exportedAtMs: Long = System.currentTimeMillis()
    ): String {
        val validStrategies = strategies
            .filter { it.genomeJson.isNotBlank() }
            .distinctBy { entity ->
                runCatching { SymbolicFormula.genomeFromJson(entity.genomeJson).describeCompact() }
                    .getOrDefault(entity.id)
            }
            .sortedWith(
                compareByDescending<DiscoveredFormulaEntity> { it.combinedStreak }
                    .thenByDescending { it.fitnessScore }
                    .thenByDescending { it.updatedAt }
            )

        val validCheckpoint = checkpointBest?.takeIf { cp ->
            cp.genomeJson.isNotBlank() && validStrategies.none {
                runCatching { SymbolicFormula.genomeFromJson(it.genomeJson).describeCompact() }.getOrNull() ==
                    runCatching { SymbolicFormula.genomeFromJson(cp.genomeJson).describeCompact() }.getOrNull()
            }
        }

        val maxStreak = maxOf(
            validStrategies.maxOfOrNull { it.combinedStreak } ?: 0,
            validCheckpoint?.combinedStreak ?: 0
        )

        val root = buildJsonObject {
            put("format", BACKUP_FORMAT)
            put("version", BACKUP_VERSION)
            put("app", "Loto BJ")
            put("periodName", periodName)
            put("exportedAtMs", exportedAtMs)
            put(
                "exportedAtLabel",
                "${LotoDateUtils.formatShortDate(exportedAtMs)} ${LotoDateUtils.formatBeninTime(exportedAtMs)} Bénin"
            )
            put("bestStreak", maxStreak)
            put("strategiesCount", validStrategies.size)
            put("strategies", buildJsonArray {
                validStrategies.forEach { entity ->
                    add(entityToJsonObject(entity))
                }
            })
            if (validCheckpoint != null) {
                put("checkpointBest", entityToJsonObject(validCheckpoint))
            }
        }
        return prettyJson.encodeToString(JsonObject.serializer(), root)
    }

    fun parseBackup(
        rawText: String,
        targetPeriodName: String = BacktestPeriod.THREE_WEEKS.name
    ): Result<ParsedStrategyBackup> = runCatching {
        val trimmed = rawText.trim()
        require(trimmed.isNotEmpty()) {
            "Le contenu collé ou sélectionné est vide."
        }

        val rootObj = extractJsonObject(trimmed)
            ?: error("Format de sauvegarde invalide : aucun bloc JSON Loto BJ détecté.")

        val format = rootObj["format"]?.jsonPrimitive?.contentOrNull.orEmpty()
        val hasBundleStructure = format == BACKUP_FORMAT ||
            rootObj.containsKey("strategies") ||
            rootObj.containsKey("checkpointBest")

        if (hasBundleStructure) {
            val sourcePeriod = rootObj["periodName"]?.jsonPrimitive?.contentOrNull ?: targetPeriodName
            val exportedAtMs = rootObj["exportedAtMs"]?.jsonPrimitive?.longOrNull ?: System.currentTimeMillis()
            val strategiesArray = (rootObj["strategies"] as? JsonArray).orEmpty()
            val nowMs = System.currentTimeMillis()

            val parsedStrategies = strategiesArray.mapNotNull { element ->
                val itemObj = element as? JsonObject ?: return@mapNotNull null
                parseEntityFromObject(
                    obj = itemObj,
                    targetPeriodName = targetPeriodName,
                    isCheckpoint = false,
                    fallbackTimestamp = nowMs
                )
            }.distinctBy { entity ->
                runCatching { SymbolicFormula.genomeFromJson(entity.genomeJson).describeCompact() }
                    .getOrDefault(entity.id)
            }

            val checkpointObj = rootObj["checkpointBest"] as? JsonObject
            val parsedCheckpoint = checkpointObj?.let {
                parseEntityFromObject(
                    obj = it,
                    targetPeriodName = targetPeriodName,
                    isCheckpoint = true,
                    fallbackTimestamp = nowMs
                )
            }

            require(parsedStrategies.isNotEmpty() || parsedCheckpoint != null) {
                "La sauvegarde ne contient aucune méthode symbolique valide."
            }

            ParsedStrategyBackup(
                sourcePeriodName = sourcePeriod,
                exportedAtMs = exportedAtMs,
                strategies = parsedStrategies,
                checkpointBest = parsedCheckpoint
            )
        } else if (rootObj.containsKey("primary") && rootObj.containsKey("secondary")) {
            val genome = validateGenome(SymbolicFormula.genomeFromJson(rootObj.toString()))
                ?: error("Génome symbolique invalide.")
            val sig = genome.describeCompact()
            val nowMs = System.currentTimeMillis()
            val featuresCsv = (
                SymbolicFormula.usedFeatures(genome.primaryExpr) +
                    SymbolicFormula.usedFeatures(genome.secondaryExpr)
                ).joinToString(",") { it.symbol }
            val entity = DiscoveredFormulaEntity(
                id = "strat_2j_${sig.hashCode().toUInt()}",
                periodName = targetPeriodName,
                generation = genome.generation,
                parentIdsCsv = genome.parentIds.joinToString(","),
                primaryExprText = genome.describePrimary(),
                secondaryExprText = genome.describeSecondary(),
                slotSplitLabel = "${genome.primarySlots}+${5 - genome.primarySlots}",
                genomeJson = SymbolicFormula.genomeToJson(genome),
                usedFeaturesCsv = featuresCsv,
                status = "MILESTONE_2J",
                rejectionStage = "",
                rejectionReason = "Importé depuis génome partagé",
                discoveryStreak = 2,
                rollingStreak = 2,
                confirmationStreak = 0,
                sealedStreak = 0,
                combinedStreak = 2,
                discoveryAvgMatches = 0f,
                rollingAvgMatches = 0f,
                confirmationAvgMatches = 0f,
                sealedAvgMatches = 0f,
                pValueAdjusted = 1.0,
                fitnessScore = 1.0,
                updatedAt = nowMs
            )
            ParsedStrategyBackup(
                sourcePeriodName = targetPeriodName,
                exportedAtMs = nowMs,
                strategies = listOf(entity),
                checkpointBest = null
            )
        } else {
            error("Sauvegarde non reconnue : format attendu 'loto_bj_strategies_backup_v1'.")
        }
    }

    fun buildBackupFileName(
        bestStreak: Int,
        exportedAtMs: Long = System.currentTimeMillis()
    ): String {
        val sdf = SimpleDateFormat("yyyyMMdd_HHmm", Locale.US).apply {
            timeZone = LotoDateUtils.beninTz
        }
        return "loto_bj_methodes_${bestStreak}j_${sdf.format(Date(exportedAtMs))}.json"
    }

    fun copyBackupToClipboard(
        context: Context,
        payload: String,
        label: String = "Sauvegarde Meilleures Méthodes Loto BJ"
    ): Boolean {
        val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
            ?: return false
        clipboard.setPrimaryClip(ClipData.newPlainText(label, payload))
        return true
    }

    fun readTextFromClipboard(context: Context): String? {
        val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
            ?: return null
        val clip = clipboard.primaryClip ?: return null
        if (clip.itemCount <= 0) return null
        return clip.getItemAt(0).coerceToText(context)?.toString()?.takeIf { it.isNotBlank() }
    }

    fun writeBackupToCacheFile(
        context: Context,
        payload: String,
        fileName: String
    ): File {
        val exportDir = File(context.cacheDir, "exports").apply { mkdirs() }
        val safeName = if (fileName.endsWith(".json", ignoreCase = true) || fileName.endsWith(".txt", ignoreCase = true)) {
            fileName
        } else {
            "$fileName.json"
        }
        val file = File(exportDir, safeName)
        file.writeText(payload, Charsets.UTF_8)
        return file
    }

    fun createShareBackupIntent(
        context: Context,
        payload: String,
        fileName: String,
        strategiesCount: Int,
        bestStreak: Int
    ): Intent {
        val file = writeBackupToCacheFile(context, payload, fileName)
        val uri = FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            file
        )
        val subject = "Sauvegarde Méthodes Loto BJ ($strategiesCount méthode(s), max ${bestStreak}J)"
        val sendIntent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_SUBJECT, subject)
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_TEXT, payload)
            clipData = ClipData.newRawUri(subject, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        return Intent.createChooser(sendIntent, "Envoyer la sauvegarde des méthodes")
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
    }

    fun writeBackupToUri(
        context: Context,
        uri: Uri,
        payload: String
    ): Result<Unit> = runCatching {
        context.contentResolver.openOutputStream(uri)?.use { stream ->
            stream.write(payload.toByteArray(Charsets.UTF_8))
            stream.flush()
        } ?: error("Impossible d'écrire le fichier de sauvegarde.")
    }

    fun readBackupFromUri(
        context: Context,
        uri: Uri
    ): Result<String> = runCatching {
        context.contentResolver.openInputStream(uri)?.bufferedReader(Charsets.UTF_8)?.use { reader ->
            reader.readText()
        } ?: error("Impossible de lire le fichier sélectionné.")
    }

    private fun entityToJsonObject(entity: DiscoveredFormulaEntity): JsonObject {
        val parsedGenomeElement: JsonElement? = runCatching {
            Json.parseToJsonElement(entity.genomeJson)
        }.getOrNull()

        return buildJsonObject {
            put("id", entity.id)
            put("periodName", entity.periodName)
            put("generation", entity.generation)
            put("parentIdsCsv", entity.parentIdsCsv)
            put("primaryExprText", entity.primaryExprText)
            put("secondaryExprText", entity.secondaryExprText)
            put("slotSplitLabel", entity.slotSplitLabel)
            if (parsedGenomeElement is JsonObject) {
                put("genome", parsedGenomeElement)
            } else {
                put("genomeJson", entity.genomeJson)
            }
            put("usedFeaturesCsv", entity.usedFeaturesCsv)
            put("status", entity.status)
            put("rejectionStage", entity.rejectionStage)
            put("rejectionReason", entity.rejectionReason)
            put("discoveryStreak", entity.discoveryStreak)
            put("rollingStreak", entity.rollingStreak)
            put("confirmationStreak", entity.confirmationStreak)
            put("sealedStreak", entity.sealedStreak)
            put("combinedStreak", entity.combinedStreak)
            put("discoveryAvgMatches", entity.discoveryAvgMatches)
            put("rollingAvgMatches", entity.rollingAvgMatches)
            put("confirmationAvgMatches", entity.confirmationAvgMatches)
            put("sealedAvgMatches", entity.sealedAvgMatches)
            put("pValueAdjusted", entity.pValueAdjusted)
            put("fitnessScore", entity.fitnessScore)
            put("updatedAt", entity.updatedAt)
        }
    }

    private fun parseEntityFromObject(
        obj: JsonObject,
        targetPeriodName: String,
        isCheckpoint: Boolean,
        fallbackTimestamp: Long
    ): DiscoveredFormulaEntity? {
        val genomeObj = obj["genome"] as? JsonObject
        val rawGenomeStr = genomeObj?.toString()
            ?: obj["genomeJson"]?.jsonPrimitive?.contentOrNull
            ?: return null

        val genome = runCatching { SymbolicFormula.genomeFromJson(rawGenomeStr) }
            .getOrNull()
            ?.let(::validateGenome)
            ?: return null

        val canonicalGenomeJson = SymbolicFormula.genomeToJson(genome)
        val sig = genome.describeCompact()
        val discStreak = (obj["discoveryStreak"]?.jsonPrimitive?.intOrNull ?: 0).coerceAtLeast(0)
        val rollStreak = (obj["rollingStreak"]?.jsonPrimitive?.intOrNull ?: 0).coerceAtLeast(0)
        val confStreak = (obj["confirmationStreak"]?.jsonPrimitive?.intOrNull ?: 0).coerceAtLeast(0)
        val sealedStreak = (obj["sealedStreak"]?.jsonPrimitive?.intOrNull ?: 0).coerceAtLeast(0)
        val combinedStreak = maxOf(
            obj["combinedStreak"]?.jsonPrimitive?.intOrNull ?: 0,
            discStreak,
            rollStreak,
            confStreak,
            sealedStreak
        ).coerceAtLeast(if (isCheckpoint) 0 else 2)

        val rawStatus = obj["status"]?.jsonPrimitive?.contentOrNull.orEmpty()
        val normalizedStatus = when {
            isCheckpoint -> rawStatus.ifBlank { "DISCOVERED" }
            rawStatus == "VALIDATED" || combinedStreak >= 4 -> "VALIDATED"
            rawStatus.startsWith("MILESTONE_") -> rawStatus
            else -> "MILESTONE_${combinedStreak}J"
        }

        val resolvedId = if (isCheckpoint) {
            "checkpoint_best_$targetPeriodName"
        } else {
            "strat_${combinedStreak}j_${sig.hashCode().toUInt()}"
        }

        val usedFeaturesCsv = (
            SymbolicFormula.usedFeatures(genome.primaryExpr) +
                SymbolicFormula.usedFeatures(genome.secondaryExpr)
            ).joinToString(",") { it.symbol }

        return DiscoveredFormulaEntity(
            id = resolvedId,
            periodName = targetPeriodName,
            generation = (obj["generation"]?.jsonPrimitive?.intOrNull ?: genome.generation).coerceAtLeast(1),
            parentIdsCsv = obj["parentIdsCsv"]?.jsonPrimitive?.contentOrNull
                ?: genome.parentIds.joinToString(","),
            primaryExprText = genome.describePrimary(),
            secondaryExprText = genome.describeSecondary(),
            slotSplitLabel = "${genome.primarySlots}+${5 - genome.primarySlots}",
            genomeJson = canonicalGenomeJson,
            usedFeaturesCsv = usedFeaturesCsv,
            status = normalizedStatus,
            rejectionStage = obj["rejectionStage"]?.jsonPrimitive?.contentOrNull.orEmpty(),
            rejectionReason = obj["rejectionReason"]?.jsonPrimitive?.contentOrNull
                ?: "Stratégie ${combinedStreak}J importée",
            discoveryStreak = discStreak,
            rollingStreak = rollStreak,
            confirmationStreak = confStreak,
            sealedStreak = sealedStreak,
            combinedStreak = combinedStreak,
            discoveryAvgMatches = obj["discoveryAvgMatches"]?.jsonPrimitive?.floatOrNull ?: 0f,
            rollingAvgMatches = obj["rollingAvgMatches"]?.jsonPrimitive?.floatOrNull ?: 0f,
            confirmationAvgMatches = obj["confirmationAvgMatches"]?.jsonPrimitive?.floatOrNull ?: 0f,
            sealedAvgMatches = obj["sealedAvgMatches"]?.jsonPrimitive?.floatOrNull ?: 0f,
            pValueAdjusted = obj["pValueAdjusted"]?.jsonPrimitive?.doubleOrNull ?: 1.0,
            fitnessScore = obj["fitnessScore"]?.jsonPrimitive?.doubleOrNull ?: combinedStreak.toDouble(),
            updatedAt = obj["updatedAt"]?.jsonPrimitive?.longOrNull ?: fallbackTimestamp
        )
    }

    private fun validateGenome(genome: SymbolicGenome): SymbolicGenome? {
        val featuresCount = SymbolicFormula.usedFeatures(genome.primaryExpr).size +
            SymbolicFormula.usedFeatures(genome.secondaryExpr).size
        return if (featuresCount > 0) genome else null
    }

    private fun extractJsonObject(text: String): JsonObject? {
        runCatching { Json.parseToJsonElement(text) as? JsonObject }
            .getOrNull()
            ?.let { return it }

        // Si le texte a ete colle avec du texte autour (WhatsApp/SMS), extraire le bloc JSON equilibre
        var i = 0
        while (i < text.length) {
            if (text[i] == '{') {
                var depth = 0
                var inString = false
                var escaped = false
                var endIdx = -1
                for (j in i until text.length) {
                    val ch = text[j]
                    if (inString) {
                        if (escaped) {
                            escaped = false
                        } else if (ch == '\\') {
                            escaped = true
                        } else if (ch == '"') {
                            inString = false
                        }
                    } else {
                        if (ch == '"') {
                            inString = true
                        } else if (ch == '{') {
                            depth++
                        } else if (ch == '}') {
                            depth--
                            if (depth == 0) {
                                endIdx = j
                                break
                            }
                        }
                    }
                }
                if (endIdx > i) {
                    val candidate = text.substring(i, endIdx + 1)
                    val parsed = runCatching { Json.parseToJsonElement(candidate) as? JsonObject }.getOrNull()
                    if (parsed != null && (
                        parsed["format"]?.jsonPrimitive?.contentOrNull == BACKUP_FORMAT ||
                            parsed.containsKey("strategies") ||
                            (parsed.containsKey("primary") && parsed.containsKey("secondary"))
                        )
                    ) {
                        return parsed
                    }
                }
            }
            i++
        }
        return null
    }
}
