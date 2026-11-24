package com.lotobj.pc.model

enum class PrecisionTargetMode(val label: String, val outputCount: Int, val description: String) {
    FIRST_BALL(
        label = "1er Numéro (Pos1)",
        outputCount = 1,
        description = "Cible uniquement le 1er numéro tiré (num1) • Hasard 1/90 (1,11%)"
    ),
    TWO_NUMBERS(
        label = "2 Numéros Exacts (Duo)",
        outputCount = 2,
        description = "Cible 2 numéros exacts (≥3/jour dont l'avant-dernier) • Hasard E=0,111 / tirage"
    )
}

enum class PrecisionFeature(val symbol: String) {
    X_NORM("x"),
    POS1_FREQ_GLOBAL("p1fg"),
    POS1_FREQ_W5("p1f5"),
    POS1_FREQ_W15("p1f15"),
    POS1_GAP_NORM("p1gap"),
    POS1_HORAIRE_FREQ("p1fh"),
    POS1_PREV_TRANSITION("p1tr"),
    POS1_SAME_LAG1("p1lag1"),
    ANY_FREQ_GLOBAL("afg"),
    ANY_FREQ_W5("af5"),
    ANY_FREQ_W15("af15"),
    ANY_GAP_NORM("agap"),
    ANY_HORAIRE_FREQ("afh"),
    CO_PREV_NUM1("cop1"),
    CO_LAST_RAW("co1"),
    BASE7_NORM("b7"),
    LUCKY_ROOT("lky"),
    AFFINE_MOD90("aff")
}

enum class PrecisionTransform(val symbol: String) {
    LINEAR("Lin"),
    TANH("Tanh"),
    SIGMOID("Sigm"),
    SQRT_SIGNED("Sqrt")
}

data class PrecisionGenome(
    val id: String,
    val mode: PrecisionTargetMode,
    val generation: Int,
    val weights: List<Double>,
    val secondaryWeights: List<Double>,
    val transform: PrecisionTransform,
    val interactionBoost: Double,
    val samplingTemp: Double,
    val exploreNoise: Double = 0.0
) {
    fun signatureKey(): String {
        val w1 = weights.joinToString(",") { String.format(java.util.Locale.US, "%.2f", it) }
        val w2 = secondaryWeights.joinToString(",") { String.format(java.util.Locale.US, "%.2f", it) }
        return "${mode.name}|${transform.symbol}|${String.format(java.util.Locale.US, "%.2f", interactionBoost)}|$w1|$w2"
    }

    fun describeCompact(): String {
        val feats = PrecisionFeature.entries
        val topPos = weights.indices
            .sortedByDescending { kotlin.math.abs(weights[it]) }
            .take(4)
            .filter { it < feats.size && kotlin.math.abs(weights[it]) >= 0.05 }
            .joinToString(" + ") { idx ->
                val v = weights[idx]
                "${String.format(java.util.Locale.US, "%.2f", v)}·${feats[idx].symbol}"
            }
        return "[${mode.label} • ${transform.symbol}] $topPos"
    }
}

data class PrecisionStageStats(
    val tested: Int,
    val exactTargetCount: Int,
    val onePlusCount: Int,
    val qualifyingDays: Int,
    val maxConsecutiveQualifyingDays: Int,
    val avgHits: Float,
    val zeroMatchCount: Int,
    val brierScore: Double,
    val hitsPerStep: List<Int>,
    val exactHitsPerStep: List<Boolean>,
    val causalVerified: Boolean
) {
    val zeroRate: Float get() = if (tested > 0) zeroMatchCount.toFloat() / tested.toFloat() else 1f
    val exactTargetRate: Float get() = if (tested > 0) exactTargetCount.toFloat() / tested.toFloat() else 0f
}

data class PrecisionBacktestStep(
    val stepIndex: Int,
    val drawId: String,
    val drawDate: String,
    val dayKey: String,
    val horaire: String,
    val game: String,
    val strictPastCount: Int,
    val maxPastTs: Long,
    val targetTimestamp: Long,
    val causalVerified: Boolean,
    val predictedNumbers: List<Int>,
    val actualPos1: Int,
    val actualNumbers: List<Int>,
    val matchedNumbers: List<Int>,
    val matchCount: Int,
    val exactTargetHit: Boolean,
    val stepBrierScore: Double
)

data class PrecisionDailyBlockSummary(
    val blockKey: String,
    val blockLabel: String,
    val drawsTested: Int,
    val exactTargetHits: Int,
    val onePlusHits: Int,
    val isQualifyingBlock: Boolean,
    val penultimateExact: Boolean = false
)

data class PrecisionBacktestResult(
    val mode: PrecisionTargetMode,
    val periodLabel: String,
    val totalTested: Int,
    val exactTargetCount: Int,
    val onePlusCount: Int,
    val qualifyingBlocksCount: Int,
    val maxConsecQualifyingBlocks: Int,
    val avgHits: Float,
    val zeroRate: Float,
    val averageBrierScore: Double,
    val pValueRaw: Double,
    val pValueAdjusted: Double,
    val allStepsCausalVerified: Boolean,
    val isAppendOnlyApproved: Boolean,
    val preservedStepsCount: Int,
    val appendedStepsCount: Int,
    val liveMatchedCount: Int,
    val strategySignature: String,
    val steps: List<PrecisionBacktestStep>,
    val dailyBlocks: List<PrecisionDailyBlockSummary>
)

data class PrecisionLivePrediction(
    val mode: PrecisionTargetMode,
    val horaire: String,
    val dayLabel: String,
    val targetTimestamp: Long,
    val predictedNumbers: List<Int>,
    val confidencesPercent: List<Pair<Int, Float>>,
    val strategyDescription: String,
    val trainedOnCount: Int,
    val causalAuditNote: String
)

data class PrecisionPendingPrediction(
    val modeName: String,
    val dayKey: String,
    val horaire: String,
    val targetTimestamp: Long,
    val predictedNumbers: List<Int>
)

data class PrecisionLedgerRecord(
    val ledgerKey: String,
    val modeName: String,
    val strategySignature: String,
    val periodKey: String,
    val genomeJson: String,
    val frozenStepsJson: String,
    val pendingPredictionsJson: String,
    val lastApprovedTimestamp: Long,
    val updatedAt: Long
)

data class PrecisionUiState(
    val targetMode: PrecisionTargetMode = PrecisionTargetMode.FIRST_BALL,
    val phase: AutonomousPhase = AutonomousPhase.IDLE,
    val infiniteMode: Boolean = false,
    val generation: Int = 0,
    val experimentsCount: Int = 0,
    val confirmationTestsCount: Int = 0,
    val excludedCount: Int = 0,
    val bestValidatedStreak: Int = 0,
    val discoveryStats: PrecisionStageStats? = null,
    val rollingStats: PrecisionStageStats? = null,
    val confirmationStats: PrecisionStageStats? = null,
    val sealedStats: PrecisionStageStats? = null,
    val activeFormulaDesc: String = "",
    val pValueAdjusted: Double = 1.0,
    val honestStatusNote: String = "Moteur Précision Indépendant prêt (1er N° & 2 N° exacts).",
    val backtestResult: PrecisionBacktestResult? = null,
    val livePrediction: PrecisionLivePrediction? = null,
    val approvedNote: String = "",
    val evaluationsPerSec: Double = 0.0,
    val liveLogs: List<String> = emptyList()
)
