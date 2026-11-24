package com.lotobj.pc.util

import com.lotobj.pc.engine.SymbolicFormula
import com.lotobj.pc.model.BacktestPeriod
import com.lotobj.pc.model.BacktestResult
import com.lotobj.pc.model.DiscoveredFormulaRecord
import com.lotobj.pc.model.EngineActionState
import com.lotobj.pc.model.NextDrawTarget
import com.lotobj.pc.model.PredictionResult
import com.lotobj.pc.model.SymbolicGenome
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
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull
import kotlinx.serialization.json.put
import java.awt.Toolkit
import java.awt.datatransfer.DataFlavor
import java.awt.datatransfer.StringSelection
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class ParsedStrategyBackup(
    val sourcePeriodName: String,
    val exportedAtMs: Long,
    val strategies: List<DiscoveredFormulaRecord>,
    val checkpointBest: DiscoveredFormulaRecord?
)

object StrategyBackupHelper {

    const val BACKUP_FORMAT = "loto_bj_strategies_backup_v1"
    const val BACKUP_VERSION = 1
    private val prettyJson = Json { prettyPrint = true }

    fun serializeBackup(
        strategies: List<DiscoveredFormulaRecord>,
        checkpointBest: DiscoveredFormulaRecord? = null,
        periodName: String = BacktestPeriod.THREE_WEEKS.name,
        exportedAtMs: Long = System.currentTimeMillis()
    ): String {
        val validStrategies = strategies
            .filter { it.genomeJson.isNotBlank() }
            .distinctBy { record ->
                runCatching { SymbolicFormula.genomeFromJson(record.genomeJson).describeCompact() }
                    .getOrDefault(record.id)
            }
            .sortedWith(
                compareByDescending<DiscoveredFormulaRecord> { it.combinedStreak }
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
            put("app", "Loto BJ Workstation PC")
            put("periodName", periodName)
            put("exportedAtMs", exportedAtMs)
            put(
                "exportedAtLabel",
                "${LotoDateUtils.formatShortDate(exportedAtMs)} ${LotoDateUtils.formatBeninTime(exportedAtMs)} Bénin"
            )
            put("bestStreak", maxStreak)
            put("strategiesCount", validStrategies.size)
            put("strategies", buildJsonArray {
                validStrategies.forEach { record -> add(recordToJsonObject(record)) }
            })
            if (validCheckpoint != null) {
                put("checkpointBest", recordToJsonObject(validCheckpoint))
            }
        }
        return prettyJson.encodeToString(JsonObject.serializer(), root)
    }

    fun parseBackup(
        rawText: String,
        targetPeriodName: String = BacktestPeriod.THREE_WEEKS.name
    ): Result<ParsedStrategyBackup> = runCatching {
        val trimmed = rawText.trim()
        require(trimmed.isNotEmpty()) { "Le contenu collé ou sélectionné est vide." }

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
                parseRecordFromObject(itemObj, targetPeriodName, isCheckpoint = false, fallbackTimestamp = nowMs)
            }.distinctBy { record ->
                runCatching { SymbolicFormula.genomeFromJson(record.genomeJson).describeCompact() }
                    .getOrDefault(record.id)
            }

            val checkpointObj = rootObj["checkpointBest"] as? JsonObject
            val parsedCheckpoint = checkpointObj?.let {
                parseRecordFromObject(it, targetPeriodName, isCheckpoint = true, fallbackTimestamp = nowMs)
            }

            require(parsedStrategies.isNotEmpty() || parsedCheckpoint != null) {
                "La sauvegarde ne contient aucune méthode symbolique valide."
            }

            ParsedStrategyBackup(sourcePeriod, exportedAtMs, parsedStrategies, parsedCheckpoint)
        } else if (rootObj.containsKey("primary") && rootObj.containsKey("secondary")) {
            val genome = validateGenome(SymbolicFormula.genomeFromJson(rootObj.toString()))
                ?: error("Génome symbolique invalide.")
            val sig = genome.describeCompact()
            val nowMs = System.currentTimeMillis()
            val featuresCsv = (
                SymbolicFormula.usedFeatures(genome.primaryExpr) +
                    SymbolicFormula.usedFeatures(genome.secondaryExpr)
                ).joinToString(",") { it.symbol }
            val record = DiscoveredFormulaRecord(
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
            ParsedStrategyBackup(targetPeriodName, nowMs, listOf(record), null)
        } else {
            error("Sauvegarde non reconnue : format attendu 'loto_bj_strategies_backup_v1'.")
        }
    }

    fun buildBackupFileName(bestStreak: Int, exportedAtMs: Long = System.currentTimeMillis()): String {
        val sdf = SimpleDateFormat("yyyyMMdd_HHmm", Locale.US).apply { timeZone = LotoDateUtils.beninTz }
        return "loto_bj_methodes_${bestStreak}j_${sdf.format(Date(exportedAtMs))}.json"
    }

    private fun recordToJsonObject(record: DiscoveredFormulaRecord): JsonObject {
        val parsedGenomeElement: JsonElement? = runCatching {
            Json.parseToJsonElement(record.genomeJson)
        }.getOrNull()

        return buildJsonObject {
            put("id", record.id)
            put("periodName", record.periodName)
            put("generation", record.generation)
            put("parentIdsCsv", record.parentIdsCsv)
            put("primaryExprText", record.primaryExprText)
            put("secondaryExprText", record.secondaryExprText)
            put("slotSplitLabel", record.slotSplitLabel)
            if (parsedGenomeElement is JsonObject) {
                put("genome", parsedGenomeElement)
            } else {
                put("genomeJson", record.genomeJson)
            }
            put("usedFeaturesCsv", record.usedFeaturesCsv)
            put("status", record.status)
            put("rejectionStage", record.rejectionStage)
            put("rejectionReason", record.rejectionReason)
            put("discoveryStreak", record.discoveryStreak)
            put("rollingStreak", record.rollingStreak)
            put("confirmationStreak", record.confirmationStreak)
            put("sealedStreak", record.sealedStreak)
            put("combinedStreak", record.combinedStreak)
            put("discoveryAvgMatches", record.discoveryAvgMatches)
            put("rollingAvgMatches", record.rollingAvgMatches)
            put("confirmationAvgMatches", record.confirmationAvgMatches)
            put("sealedAvgMatches", record.sealedAvgMatches)
            put("pValueAdjusted", record.pValueAdjusted)
            put("fitnessScore", record.fitnessScore)
            put("updatedAt", record.updatedAt)
        }
    }

    private fun parseRecordFromObject(
        obj: JsonObject,
        targetPeriodName: String,
        isCheckpoint: Boolean,
        fallbackTimestamp: Long
    ): DiscoveredFormulaRecord? {
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
            discStreak, rollStreak, confStreak, sealedStreak
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

        return DiscoveredFormulaRecord(
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
        runCatching { Json.parseToJsonElement(text) as? JsonObject }.getOrNull()?.let { return it }
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
                        if (escaped) escaped = false
                        else if (ch == '\\') escaped = true
                        else if (ch == '"') inString = false
                    } else {
                        when (ch) {
                            '"' -> inString = true
                            '{' -> depth++
                            '}' -> {
                                depth--
                                if (depth == 0) {
                                    endIdx = j
                                    break
                                }
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
                    ) return parsed
                }
            }
            i++
        }
        return null
    }
}

object DesktopClipboard {
    fun copyText(text: String): Boolean = runCatching {
        val clipboard = Toolkit.getDefaultToolkit().systemClipboard
        clipboard.setContents(StringSelection(text), null)
        true
    }.getOrDefault(false)

    fun readText(): String? = runCatching {
        val clipboard = Toolkit.getDefaultToolkit().systemClipboard
        if (clipboard.isDataFlavorAvailable(DataFlavor.stringFlavor)) {
            clipboard.getData(DataFlavor.stringFlavor) as? String
        } else null
    }.getOrNull()?.takeIf { it.isNotBlank() }
}

object BacktestExportHelper {

    fun formatDetailedBacktestReport(
        backtestResult: BacktestResult,
        engineState: EngineActionState,
        prediction: PredictionResult? = null,
        nextDrawTarget: NextDrawTarget? = null,
        generatedAtMs: Long = System.currentTimeMillis()
    ): String {
        val sb = StringBuilder(4096)
        val matchedDays = engineState.matchedDaysCount(backtestResult)
        val patternLength = engineState.selectedPatternLength
        val dateStamp = "${LotoDateUtils.formatShortDate(generatedAtMs)} à ${LotoDateUtils.formatBeninTime(generatedAtMs)} (heure Bénin)"
        val activeStrategy = engineState.activeStrategySummary

        sb.appendLine("============================================================")
        sb.appendLine("RAPPORT DÉTAILLÉ DE BACKTEST & PRÉDICTION — LOTO BJ PC")
        sb.appendLine("============================================================")
        sb.appendLine("Généré le          : $dateStamp")
        sb.appendLine("Fenêtre analysée   : ${backtestResult.periodLabel}")
        sb.appendLine("Méthode / Stratégie: ${prediction?.discoveredStrategyName?.ifBlank { null } ?: backtestResult.methodName}")
        if (activeStrategy != null) {
            sb.appendLine("Palier actif       : ${activeStrategy.combinedStreak}J (${activeStrategy.statusReason})")
            sb.appendLine("Génération / ID    : Gén.${activeStrategy.generation} • ${activeStrategy.id}")
        }
        if (engineState.currentPrimaryExpr.isNotBlank()) {
            sb.appendLine("Formule A (${engineState.currentSlotSplit}): ${engineState.currentPrimaryExpr}")
        }
        if (engineState.currentSecondaryExpr.isNotBlank()) {
            sb.appendLine("Formule B          : ${engineState.currentSecondaryExpr}")
        }
        sb.appendLine("Jours réussis      : $matchedDays/$patternLength j")
        sb.appendLine("Causalité stricte  : ${if (backtestResult.allStepsCausalVerified) "VÉRIFIÉE (t_train < t_cible, 0 fuite future)" else "NON VÉRIFIÉE"}")
        sb.appendLine()

        if (nextDrawTarget != null || prediction != null) {
            sb.appendLine("------------------------------------------------------------")
            sb.appendLine("1. PROCHAIN TIRAGE CIBLÉ & PRÉDICTION")
            sb.appendLine("------------------------------------------------------------")
            val horaire = prediction?.horaire ?: nextDrawTarget?.horaire.orEmpty()
            val dayLabel = nextDrawTarget?.dayLabel?.takeIf { it.isNotBlank() }?.let { " ($it)" } ?: ""
            val whenLabel = if (nextDrawTarget?.isTomorrow == true) "Demain" else "Aujourd'hui"
            sb.appendLine("Tirage ciblé       : $horaire — $whenLabel$dayLabel")
            if (prediction != null) {
                sb.appendLine("Numéros prédits    : ${prediction.predictedDraw.joinToString(" - ") { formatTwoDigits(it) }}")
                if (prediction.methodANumbers.isNotEmpty() || prediction.methodBNumbers.isNotEmpty()) {
                    sb.appendLine("  • ${prediction.methodALabel.ifBlank { "Formule A" }} : ${prediction.methodANumbers.joinToString(" - ") { formatTwoDigits(it) }}")
                    sb.appendLine("  • ${prediction.methodBLabel.ifBlank { "Formule B" }} : ${prediction.methodBNumbers.joinToString(" - ") { formatTwoDigits(it) }}")
                }
                sb.appendLine(
                    "Espérance / 5      : ${fmt3(prediction.expectedMatchesEstimate)} • Confiance P(≥1 bon) : ${
                        String.format(Locale.FRANCE, "%.1f%%", prediction.calibratedProbAtLeastOne * 100f)
                    }"
                )
            }
            sb.appendLine()
        }

        sb.appendLine("------------------------------------------------------------")
        sb.appendLine("2. SYNTHÈSE DES PERFORMANCES DU BACKTEST")
        sb.appendLine("------------------------------------------------------------")
        sb.appendLine("Tirages testés     : ${backtestResult.totalTested}")
        sb.appendLine(
            "Moyenne bons / tir.: ${fmt3(backtestResult.averageMatches)} (IC 95% [${fmt3(backtestResult.avgMatchesCiLow)}–${fmt3(backtestResult.avgMatchesCiHigh)}], hasard = 0,278)"
        )
        sb.appendLine(
            "Taux ≥ 1 bon       : ${String.format(Locale.FRANCE, "%.1f%%", backtestResult.successRate * 100f)} (IC 95% [${(backtestResult.rateOnePlusCiLow * 100).toInt()}–${(backtestResult.rateOnePlusCiHigh * 100).toInt()}%])"
        )
        sb.appendLine("Tirages ≥ 2 bons   : ${backtestResult.drawsWithTwoPlusMatches} / ${backtestResult.totalTested}")
        sb.appendLine("Tirages ≥ 3 bons   : ${backtestResult.drawsWithThreePlusMatches} • ≥ 4 bons : ${backtestResult.drawsWithFourPlusMatches}")
        sb.appendLine("Jours qualifiants  : ${backtestResult.qualifyingDaysCount} / ${backtestResult.totalDaysTested} (critère ≥2 bons sur ≥3 tir./j + avant-dernier ≥2)")
        sb.appendLine("Max jours consécut.: $matchedDays / $patternLength j")
        sb.appendLine("Max série à 0 bon  : ${backtestResult.maxZeroMatchStreak} tirage(s)")
        sb.appendLine(
            "Brier / LogLoss/ECE: ${String.format(Locale.FRANCE, "%.5f", backtestResult.averageBrierScore)} • ${
                String.format(Locale.FRANCE, "%.4f", backtestResult.averageBernoulliLogLoss)
            } • ${String.format(Locale.FRANCE, "%.4f", backtestResult.calibrationErrorEce)}"
        )
        sb.appendLine()

        if (backtestResult.dailySummaries.isNotEmpty()) {
            sb.appendLine("------------------------------------------------------------")
            sb.appendLine("RÉCAPITULATIF JOUR PAR JOUR (${backtestResult.dailySummaries.size} jours/blocs)")
            sb.appendLine("------------------------------------------------------------")
            backtestResult.dailySummaries.forEach { day ->
                val tag = if (day.isQualifyingDay) "OUI — QUALIFIÉ" else "NON"
                val penTag = if (day.penultimateTwoPlus) "OUI" else "NON"
                sb.appendLine(
                    "  • ${day.dayLabel} (${day.dayKey}) : ${day.drawsWithTwoPlus}/${day.drawsTested} tir. ≥2 | avant-dernier=${penTag}(${day.penultimateHitCount}/5) | max=${day.bestMatchCount}/5 [$tag]"
                )
            }
            sb.appendLine()
        }

        val fourStage = engineState.fourStageMetrics
        if (fourStage != null) {
            sb.appendLine("------------------------------------------------------------")
            sb.appendLine("3. VALIDATION CHRONOLOGIQUE 4 NIVEAUX & MONTE CARLO")
            sb.appendLine("------------------------------------------------------------")
            sb.appendLine("1. Découverte (${fourStage.discoveryDraws} tir.) : moy ${fmt3(fourStage.discoveryAvgMatches)} • ≥2: ${fourStage.discoveryTwoPlusCount} • série ${fourStage.discoveryStreak}j")
            sb.appendLine("2. Glissante  (${fourStage.rollingDraws} tir.) : moy ${fmt3(fourStage.rollingAvgMatches)} • ≥2: ${fourStage.rollingTwoPlusCount} • série ${fourStage.rollingStreak}j • ${if (fourStage.rollingStabilityPassed) "Stable" else "Instable"}")
            sb.appendLine(
                "3. Confirm.   (${fourStage.confirmationDraws} tir., ${engineState.confirmationTestsCount} audits) : moy ${fmt3(fourStage.confirmationAvgMatches)} vs MC ${fmt3(engineState.randomReference.confirmationMeanAvgMatches)} • p_adj=${
                    String.format(Locale.FRANCE, "%.4f", engineState.randomReference.pValueAdjusted)
                }"
            )
            sb.appendLine("4. Scellé     (${fourStage.sealedDraws} tir.) : moy ${fmt3(fourStage.sealedAvgMatches)} • causal=${if (fourStage.sealedCausalVerified) "Vérifié" else "En attente"}")
            sb.appendLine()
        }

        sb.appendLine("------------------------------------------------------------")
        sb.appendLine("4. DÉTAIL TIRAGE PAR TIRAGE (${backtestResult.steps.size} étapes)")
        sb.appendLine("------------------------------------------------------------")
        backtestResult.steps.forEach { step ->
            val predStr = step.predictedNumbers.joinToString(" ") { formatTwoDigits(it) }
            val actualStr = step.actualNumbers.joinToString(" ") { formatTwoDigits(it) }
            val matchedStr = if (step.matchedNumbers.isEmpty()) "Aucun" else step.matchedNumbers.joinToString(" ") { formatTwoDigits(it) }
            sb.appendLine("#${step.stepIndex} | ${step.drawDate} ${step.horaire} (${step.game}) | Base:${step.trainSampleCount}")
            sb.appendLine("   Prédit : [$predStr] | Tirage : [$actualStr] -> (${step.matchCount}/5) : [$matchedStr]")
        }
        sb.appendLine("============================================================")
        return sb.toString().trimEnd()
    }

    fun buildExportFileName(prediction: PredictionResult? = null, backtestResult: BacktestResult? = null): String {
        val horaireSafe = (prediction?.horaire?.ifBlank { null } ?: "3sem").replace(Regex("[^A-Za-z0-9_-]"), "")
        val streakSafe = backtestResult?.maxConsecutiveQualifyingDays ?: 0
        val sdf = SimpleDateFormat("yyyyMMdd_HHmm", Locale.US).apply { timeZone = LotoDateUtils.beninTz }
        return "backtest_loto_bj_${horaireSafe}_${streakSafe}j_${sdf.format(Date())}.txt"
    }

    private fun formatTwoDigits(value: Int): String = value.toString().padStart(2, '0')
    private fun fmt3(value: Float): String = String.format(Locale.FRANCE, "%.3f", value)
}
