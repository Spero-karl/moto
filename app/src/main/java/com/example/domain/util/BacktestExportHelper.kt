package com.example.domain.util

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import androidx.core.content.FileProvider
import com.example.domain.engine.BacktestResult
import com.example.domain.engine.EngineActionState
import com.example.domain.engine.NextDrawTarget
import com.example.domain.engine.PredictionResult
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

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
        sb.appendLine("RAPPORT DÉTAILLÉ DE BACKTEST & PRÉDICTION — LOTO BJ")
        sb.appendLine("============================================================")
        sb.appendLine("Généré le          : $dateStamp")
        sb.appendLine("Fenêtre analysée   : ${backtestResult.periodLabel}")
        sb.appendLine("Méthode / Stratégie: ${prediction?.discoveredStrategyName?.ifBlank { null } ?: backtestResult.method.displayName}")
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
                if (prediction.numberConfidences.isNotEmpty()) {
                    sb.appendLine("Confiances détaillées :")
                    prediction.numberConfidences.forEach { conf ->
                        sb.appendLine(
                            "  - N°${formatTwoDigits(conf.number)} : prob=${
                                String.format(Locale.FRANCE, "%.1f%%", conf.marginalProbabilityPercent)
                            } • rel=${
                                String.format(Locale.FRANCE, "%.0f%%", conf.relativeConfidencePercent)
                            } • lift=${String.format(Locale.FRANCE, "%.2fx", conf.liftVsRandom)} [${conf.sourceBranch}]"
                        )
                    }
                }
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
        if (backtestResult.matchDistribution.isNotEmpty()) {
            val distLine = backtestResult.matchDistribution
                .mapIndexed { idx, count -> "$idx/5=$count" }
                .joinToString(" | ")
            sb.appendLine("Distribution hits  : $distLine")
        }
        sb.appendLine()

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

        if (backtestResult.horaireBreakdown.isNotEmpty()) {
            sb.appendLine("------------------------------------------------------------")
            sb.appendLine("4. PERFORMANCE PAR HORAIRE")
            sb.appendLine("------------------------------------------------------------")
            backtestResult.horaireBreakdown.forEach { h ->
                val ratePct = if (h.testedCount > 0) (h.onePlusCount * 100f) / h.testedCount else 0f
                sb.appendLine(
                    "  • %-4s : %2d tir. | moy %s | taux ≥1: %5.1f%% (%d) | ≥2 bons: %d".format(
                        Locale.FRANCE,
                        h.horaire,
                        h.testedCount,
                        fmt3(h.averageMatches),
                        ratePct,
                        h.onePlusCount,
                        h.twoPlusCount
                    )
                )
            }
            sb.appendLine()
        }

        if (backtestResult.dailySummaries.isNotEmpty()) {
            sb.appendLine("------------------------------------------------------------")
            sb.appendLine("5. RÉCAPITULATIF JOUR PAR JOUR (${backtestResult.dailySummaries.size} jours)")
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

        sb.appendLine("------------------------------------------------------------")
        sb.appendLine("6. DÉTAIL TIRAGE PAR TIRAGE (${backtestResult.steps.size} étapes)")
        sb.appendLine("------------------------------------------------------------")
        if (backtestResult.steps.isEmpty()) {
            sb.appendLine("Aucune étape détaillée disponible.")
        } else {
            backtestResult.steps.forEach { step ->
                val predStr = step.predictedNumbers.joinToString(" ") { formatTwoDigits(it) }
                val actualStr = step.actualNumbers.joinToString(" ") { formatTwoDigits(it) }
                val matchedStr = if (step.matchedNumbers.isEmpty()) "Aucun" else step.matchedNumbers.joinToString(" ") { formatTwoDigits(it) }
                val splitStr = if (step.methodANumbers.isNotEmpty() || step.methodBNumbers.isNotEmpty()) {
                    " [A:${step.methodANumbers.joinToString(",") { formatTwoDigits(it) }} | B:${step.methodBNumbers.joinToString(",") { formatTwoDigits(it) }}]"
                } else ""
                val causalTag = if (step.causalVerified) "OK" else "ALERTE"
                sb.appendLine(
                    "#${step.stepIndex + 1} | ${step.drawDate} ${step.horaire} (${step.game}) | Base:${step.trainSampleCount} [Causal:$causalTag]"
                )
                sb.appendLine("   Prédit : [$predStr]$splitStr")
                sb.appendLine("   Tirage : [$actualStr] -> Trouvés (${step.matchCount}/5) : [$matchedStr]")
                sb.appendLine(
                    "   Espér. : ${fmt3(step.expectedMatchesEstimate)} • Brier=${
                        String.format(Locale.FRANCE, "%.5f", step.stepBrierScore)
                    } • LogLoss=${String.format(Locale.FRANCE, "%.4f", step.stepBernoulliLogLoss)}"
                )
            }
        }
        sb.appendLine("============================================================")
        return sb.toString().trimEnd()
    }

    fun buildExportFileName(
        prediction: PredictionResult? = null,
        backtestResult: BacktestResult? = null,
        generatedAtMs: Long = System.currentTimeMillis()
    ): String {
        val horaireRaw = prediction?.horaire?.ifBlank { null } ?: "3sem"
        val horaireSafe = horaireRaw.replace(Regex("[^A-Za-z0-9_-]"), "")
        val streakSafe = backtestResult?.maxConsecutiveQualifyingDays ?: 0
        val sdf = SimpleDateFormat("yyyyMMdd_HHmm", Locale.US).apply {
            timeZone = LotoDateUtils.beninTz
        }
        return "backtest_loto_bj_${horaireSafe}_${streakSafe}j_${sdf.format(Date(generatedAtMs))}.txt"
    }

    fun copyToClipboard(
        context: Context,
        text: String,
        label: String = "Backtest détaillé Loto BJ"
    ): Boolean {
        val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager
            ?: return false
        clipboard.setPrimaryClip(ClipData.newPlainText(label, text))
        return true
    }

    fun writeReportToCacheFile(
        context: Context,
        content: String,
        fileName: String
    ): File {
        val exportDir = File(context.cacheDir, "exports").apply { mkdirs() }
        val safeName = if (fileName.endsWith(".txt", ignoreCase = true)) fileName else "$fileName.txt"
        val file = File(exportDir, safeName)
        file.writeText(content, Charsets.UTF_8)
        return file
    }

    fun createShareTextFileIntent(
        context: Context,
        content: String,
        fileName: String,
        summaryTitle: String = "Rapport détaillé Backtest Loto BJ"
    ): Intent {
        val file = writeReportToCacheFile(context, content, fileName)
        val uri = FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            file
        )
        val sendIntent = Intent(Intent.ACTION_SEND).apply {
            type = "text/plain"
            putExtra(Intent.EXTRA_SUBJECT, summaryTitle)
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_TEXT, "$summaryTitle ($fileName)")
            clipData = ClipData.newRawUri(summaryTitle, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        return Intent.createChooser(sendIntent, "Transférer le fichier backtest (.txt)")
            .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
    }

    fun writeReportToUri(
        context: Context,
        uri: Uri,
        content: String
    ): Result<Unit> = runCatching {
        context.contentResolver.openOutputStream(uri)?.use { stream ->
            stream.write(content.toByteArray(Charsets.UTF_8))
            stream.flush()
        } ?: error("Impossible d'ouvrir la destination du fichier texte.")
    }

    private fun formatTwoDigits(value: Int): String = value.toString().padStart(2, '0')
    private fun fmt3(value: Float): String = String.format(Locale.FRANCE, "%.3f", value)
}
