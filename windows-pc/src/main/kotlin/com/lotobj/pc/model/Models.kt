package com.lotobj.pc.model

import kotlin.math.ln

data class LotoDraw(
    val uniqueId: String,
    val date: String,
    val time: String,
    val game: String,
    val num1: Int,
    val num2: Int,
    val num3: Int,
    val num4: Int,
    val num5: Int,
    val timestamp: Long
)

data class DiscoveredFormulaRecord(
    val id: String,
    val periodName: String,
    val generation: Int,
    val parentIdsCsv: String,
    val primaryExprText: String,
    val secondaryExprText: String,
    val slotSplitLabel: String,
    val genomeJson: String,
    val usedFeaturesCsv: String,
    val status: String,
    val rejectionStage: String,
    val rejectionReason: String,
    val discoveryStreak: Int,
    val rollingStreak: Int,
    val confirmationStreak: Int,
    val sealedStreak: Int,
    val combinedStreak: Int,
    val discoveryAvgMatches: Float,
    val rollingAvgMatches: Float,
    val confirmationAvgMatches: Float,
    val sealedAvgMatches: Float,
    val pValueAdjusted: Double,
    val fitnessScore: Double,
    val updatedAt: Long
)

enum class RawFeature(val symbol: String) {
    X_NORM("x"),
    FREQ_GLOBAL("fg"),
    FREQ_W5("f5"),
    FREQ_W15("f15"),
    FREQ_W40("f40"),
    GAP_NORM("gap"),
    PREV_GAP_NORM("pgap"),
    LAST_HIT("hit1"),
    LAG2_HIT("hit2"),
    CO_LAST_RAW("co1"),
    HORAIRE_FREQ("fh"),
    SAME_DAY_REP("rep"),
    BASE7_NORM("b7"),
    MOD7_RES_DENSITY("m7r"),
    LUCKY_ROOT_NORM("lky"),
    MOD10_UNIT_NORM("m10"),
    MOD11_HARMONIC("m11"),
    CONGRUENCE_CUBE90("cub"),
    AFFINE_MOD90_RESONANCE("aff"),
    EXTRAPOL_VELOCITY("vel"),
    LIMIT_HAZARD("lim"),
    PROB_ZSCORE("zsc"),
    TRIANG_PYRAMID_DIFF("tri"),
    GRID_9X10_PROXIMITY("g9x10")
}

enum class UnaryOp(val symbol: String) {
    NEG("-"),
    ABS("abs"),
    SQRT_ABS("sqrt"),
    LOG1P_ABS("ln1p"),
    EXP_CLIP("exp"),
    SIN_PI("sin"),
    SIGMOID("sigm"),
    SQUARE("sq"),
    INV_SAFE("inv"),
    TANH("tanh"),
    COS_PI("cos"),
    CBRT("cbrt"),
    SOFTPLUS("splus"),
    LUCKY9_WAVE("sin9"),
    STEP_POS("step")
}

enum class BinaryOp(val symbol: String) {
    ADD("+"),
    SUB("-"),
    MUL("*"),
    DIV_SAFE("/"),
    MAX("max"),
    MIN("min"),
    POW_SAFE("^"),
    MOD_SAFE("mod"),
    HYPOT_NORM("hyp"),
    MEAN2("avg"),
    ATAN2_PI("at2"),
    TRIANG_BLEND("trb"),
    EXP_DIFF("exd")
}

sealed interface ExprNode {
    data class Var(val feature: RawFeature) : ExprNode
    data class Const(val value: Double) : ExprNode
    data class Unary(val op: UnaryOp, val child: ExprNode) : ExprNode
    data class Binary(val op: BinaryOp, val left: ExprNode, val right: ExprNode) : ExprNode
}

data class SymbolicGenome(
    val id: String,
    val generation: Int,
    val parentIds: List<String>,
    val primaryExpr: ExprNode,
    val secondaryExpr: ExprNode,
    val primarySlots: Int,
    val samplingTemp: Double,
    val exploreNoise: Double,
    val seed: Long
) {
    private val cachedPrimaryText: String by lazy(LazyThreadSafetyMode.PUBLICATION) {
        com.lotobj.pc.engine.SymbolicFormula.prettyPrint(primaryExpr)
    }
    private val cachedSecondaryText: String by lazy(LazyThreadSafetyMode.PUBLICATION) {
        com.lotobj.pc.engine.SymbolicFormula.prettyPrint(secondaryExpr)
    }
    private val cachedCompactSignature: String by lazy(LazyThreadSafetyMode.PUBLICATION) {
        "[${primarySlots}+${5 - primarySlots}] A=($cachedPrimaryText) | B=($cachedSecondaryText)"
    }

    fun describePrimary(): String = cachedPrimaryText
    fun describeSecondary(): String = cachedSecondaryText
    fun describeCompact(): String = cachedCompactSignature
}

class RawFeatureMatrix(
    val featureRows: Array<DoubleArray>
) {
    fun get(featureIndex: Int, number: Int): Double = featureRows[featureIndex][number]
}

enum class BacktestPeriod(val label: String, val days: Int?) {
    ONE_DAY("1 Jour", 1),
    THREE_DAYS("3 Jours", 3),
    FIVE_DAYS("5 Jours", 5),
    SEVEN_DAYS("7 Jours (1 Sem.)", 7),
    FOURTEEN_DAYS("14 Jours (2 Sem.)", 14),
    THREE_WEEKS("3 Semaines (21j)", 21),
    ONE_MONTH("1 Mois (30j)", 30),
    THREE_MONTHS("3 Mois (90j)", 90),
    ALL_HISTORY("Tout l'historique", null)
}

data class NextDrawTarget(
    val horaire: String = "11H",
    val targetTimestamp: Long = 0L,
    val dayLabel: String = "",
    val currentBeninTimeLabel: String = "",
    val isTomorrow: Boolean = false
)

enum class AutonomousPhase {
    IDLE,
    ANALYZING,
    SUSPENDED,
    STOPPED,
    VALIDATED
}

enum class ComputeMode(val label: String) {
    GPU_AND_CPU_MAX("Puissance Max (Multi-GPU OpenCL + 100% CPU + RAM)"),
    CPU_MAX_CORES("CPU Multi-Cœurs 100% + RAM"),
    BALANCED("Mode Équilibré (50% Cœurs CPU)")
}

data class HardwareCapabilities(
    val logicalCores: Int,
    val totalPhysicalRamMb: Long,
    val maxJvmHeapMb: Long,
    val gpuAvailable: Boolean,
    val gpuDeviceName: String,
    val gpuVendor: String,
    val gpuComputeUnits: Int,
    val gpuGlobalMemMb: Long,
    val gpuMaxWorkGroupSize: Long,
    val gpuBackendNote: String,
    val gpuCount: Int = if (gpuAvailable) 1 else 0,
    val nvidiaGpuCount: Int = 0,
    val amdGpuCount: Int = 0,
    val otherGpuCount: Int = 0
)

data class HardwareTelemetry(
    val processCpuPercent: Double = 0.0,
    val systemCpuPercent: Double = 0.0,
    val jvmUsedRamMb: Long = 0L,
    val jvmMaxRamMb: Long = 0L,
    val systemFreeRamMb: Long = 0L,
    val systemTotalRamMb: Long = 0L,
    val activeWorkerThreads: Int = 0,
    val gpuActive: Boolean = false,
    val gpuDeviceName: String = "Détection...",
    val evaluationsPerSecond: Double = 0.0,
    val lastGpuBatchMs: Double = 0.0,
    val lastCpuBatchMs: Double = 0.0,
    val gpuCount: Int = 0,
    val gpuErrorNote: String = ""
)

data class RandomReferenceComparison(
    val simulationsCount: Int = 0,
    val confirmationMeanAvgMatches: Float = BacktestResult.RANDOM_EXPECTED_MATCHES,
    val confirmationCiLow: Float = BacktestResult.RANDOM_EXPECTED_MATCHES,
    val confirmationCiHigh: Float = BacktestResult.RANDOM_EXPECTED_MATCHES,
    val confirmationMeanQualifyingDays: Float = 0f,
    val pValueRaw: Double = 1.0,
    val pValueAdjusted: Double = 1.0,
    val beatsRandomAfterCorrection: Boolean = false
)

data class FourStageValidationMetrics(
    val discoveryDraws: Int,
    val discoveryStreak: Int,
    val discoveryQualifyingDays: Int,
    val discoveryTwoPlusCount: Int,
    val discoveryAvgMatches: Float,
    val rollingDraws: Int,
    val rollingStreak: Int,
    val rollingQualifyingDays: Int,
    val rollingTwoPlusCount: Int,
    val rollingAvgMatches: Float,
    val rollingStabilityPassed: Boolean,
    val confirmationDraws: Int,
    val confirmationStreak: Int,
    val confirmationQualifyingDays: Int,
    val confirmationTwoPlusCount: Int,
    val confirmationAvgMatches: Float,
    val sealedDraws: Int,
    val sealedStreak: Int,
    val sealedQualifyingDays: Int,
    val sealedTwoPlusCount: Int,
    val sealedAvgMatches: Float,
    val sealedCausalVerified: Boolean,
    val combinedStreak: Int,
    val combinedQualifyingDays: Int,
    val combinedMaxZeroStreak: Int,
    val combinedBrierScore: Double,
    val isFourStageValidated: Boolean,
    val rollingZeroMatchRate: Float = 0f,
    val rollingThreePlusCount: Int = 0,
    val rollingFourPlusCount: Int = 0,
    val rollingFiveMatchCount: Int = 0,
    val rollingTwoPlusRate: Float = if (rollingDraws > 0) rollingTwoPlusCount.toFloat() / rollingDraws.toFloat() else 0f
)

data class SymbolicFormulaSummary(
    val id: String,
    val generation: Int,
    val primaryExprText: String,
    val secondaryExprText: String,
    val slotSplitLabel: String,
    val usedVariables: List<String>,
    val status: String,
    val statusReason: String,
    val discoveryStreak: Int,
    val rollingStreak: Int,
    val confirmationStreak: Int,
    val sealedStreak: Int,
    val combinedStreak: Int,
    val discoveryAvgMatches: Float,
    val rollingAvgMatches: Float,
    val confirmationAvgMatches: Float,
    val pValueAdjusted: Double,
    val fitnessScore: Double,
    val isValidated: Boolean,
    val rollingZeroMatchRate: Float = 0f,
    val rollingThreePlusCount: Int = 0,
    val rollingFourPlusCount: Int = 0,
    val rollingFiveMatchCount: Int = 0,
    val rollingTwoPlusRate: Float = 0f
)

data class RejectedFormulaSummary(
    val id: String,
    val generation: Int,
    val formulaPreview: String,
    val rejectedAtStage: String,
    val reason: String,
    val trainAvgMatches: Float,
    val validationAvgMatches: Float
)

data class MonteCarloBaseline(
    val simulationsCount: Int,
    val seed: Long,
    val drawsCount: Int,
    val daysCount: Int,
    val meanAvgMatches: Float,
    val avgMatchesCiLow: Float,
    val avgMatchesCiHigh: Float,
    val meanOnePlusRate: Float,
    val meanTwoPlusCount: Float,
    val twoPlusCiLow: Int,
    val twoPlusCiHigh: Int,
    val meanQualifyingDays: Float,
    val qualifyingDaysCiLow: Int,
    val qualifyingDaysCiHigh: Int
)

data class PredictedNumberConfidence(
    val number: Int,
    val marginalProbabilityPercent: Float,
    val relativeConfidencePercent: Float,
    val liftVsRandom: Float,
    val rank: Int,
    val sourceBranch: String,
    val isGapCandidate: Boolean
)

data class PredictionResult(
    val predictedDraw: List<Int>,
    val expectedMatchesEstimate: Float,
    val calibratedProbAtLeastOne: Float,
    val trainedOnCount: Int,
    val methodName: String,
    val horaire: String,
    val marginalSumCheck: Float,
    val auditNote: String = "",
    val discoveredStrategyName: String = "",
    val coldNumbersCount: Int = 0,
    val searchRoundsExecuted: Int = 0,
    val totalStrategiesSearched: Int = 0,
    val objectiveFourDaysAchieved: Boolean = false,
    val bestCausalStreakFound: Int = 0,
    val methodANumbers: List<Int> = emptyList(),
    val methodBNumbers: List<Int> = emptyList(),
    val methodALabel: String = "",
    val methodBLabel: String = "",
    val numberConfidences: List<PredictedNumberConfidence> = emptyList(),
    val candidateConfidences: List<PredictedNumberConfidence> = emptyList()
)

data class BacktestStepResult(
    val stepIndex: Int,
    val drawId: String,
    val drawLabel: String,
    val drawDate: String,
    val dayKey: String,
    val horaire: String,
    val game: String,
    val trainSampleCount: Int,
    val maxTrainTimestamp: Long,
    val targetTimestamp: Long,
    val causalVerified: Boolean,
    val predictedNumbers: List<Int>,
    val actualNumbers: List<Int>,
    val matchedNumbers: List<Int>,
    val matchCount: Int,
    val expectedMatchesEstimate: Float,
    val stepBrierScore: Double,
    val stepBernoulliLogLoss: Double,
    val methodANumbers: List<Int> = emptyList(),
    val methodBNumbers: List<Int> = emptyList()
)

data class DailyBacktestSummary(
    val dayKey: String,
    val dayLabel: String,
    val drawsTested: Int,
    val drawsWithTwoPlus: Int,
    val bestMatchCount: Int,
    val isQualifyingDay: Boolean,
    val penultimateHitCount: Int = 0,
    val penultimateTwoPlus: Boolean = false
)

data class HorairePerformance(
    val horaire: String,
    val testedCount: Int,
    val averageMatches: Float,
    val onePlusCount: Int,
    val twoPlusCount: Int
)

data class BacktestResult(
    val methodName: String = "Auto-Recherche Symbolique PC",
    val successRate: Float,
    val totalTested: Int,
    val exactMatches: Int,
    val partialMatches: Int,
    val drawsWithTwoPlusMatches: Int,
    val drawsWithThreePlusMatches: Int,
    val drawsWithFourPlusMatches: Int,
    val qualifyingDaysCount: Int,
    val totalDaysTested: Int,
    val averageMatches: Float,
    val avgMatchesCiLow: Float,
    val avgMatchesCiHigh: Float,
    val rateOnePlusCiLow: Float,
    val rateOnePlusCiHigh: Float,
    val averageBrierScore: Double,
    val averageBernoulliLogLoss: Double,
    val calibrationErrorEce: Double,
    val matchDistribution: List<Int>,
    val allStepsCausalVerified: Boolean,
    val periodLabel: String,
    val dailySummaries: List<DailyBacktestSummary>,
    val horaireBreakdown: List<HorairePerformance>,
    val steps: List<BacktestStepResult>,
    val maxConsecutiveQualifyingDays: Int = 0,
    val maxZeroMatchStreak: Int = 0,
    val currentZeroMatchStreak: Int = 0,
    val searchRoundsExecuted: Int = 1,
    val totalStrategiesSearched: Int = 0,
    val objectiveFourDaysAchieved: Boolean = maxConsecutiveQualifyingDays >= 4,
    val isAppendOnlyApproved: Boolean = false,
    val approvedStepsPreservedCount: Int = 0,
    val newlyAppendedStepsCount: Int = 0,
    val livePredictionsMatchedCount: Int = 0
) {
    companion object {
        const val RANDOM_EXPECTED_MATCHES = 25f / 90f
        const val RANDOM_AT_LEAST_ONE_RATE = 0.25184f
        const val RANDOM_TWO_PLUS_RATE = 0.02116f
        const val RANDOM_BRIER_SCORE = (5.0 / 90.0) * (85.0 / 90.0)
        val RANDOM_BERNOULLI_LOG_LOSS =
            -(5.0 / 90.0) * ln(5.0 / 90.0) - (85.0 / 90.0) * ln(85.0 / 90.0)
    }
}

data class EngineActionState(
    val phase: AutonomousPhase = AutonomousPhase.IDLE,
    val generation: Int = 0,
    val experimentsCount: Int = 0,
    val confirmationTestsCount: Int = 0,
    val discoveryDrawsCount: Int = 0,
    val rollingDrawsCount: Int = 0,
    val confirmationDrawsCount: Int = 0,
    val sealedDrawsCount: Int = 0,
    val periodDateRangeLabel: String = "",
    val currentPrimaryExpr: String = "",
    val currentSecondaryExpr: String = "",
    val currentSlotSplit: String = "",
    val bestDiscoveryStreak: Int = 0,
    val bestRollingStreak: Int = 0,
    val bestConfirmationStreak: Int = 0,
    val bestSealedStreak: Int = 0,
    val bestCombinedStreak: Int = 0,
    val bestQualifyingDays: Int = 0,
    val bestTwoPlusCount: Int = 0,
    val bestMaxZeroStreak: Int = 0,
    val discoveredFeaturesHistogram: List<Pair<String, Float>> = emptyList(),
    val bestFormulas: List<SymbolicFormulaSummary> = emptyList(),
    val rejectedFormulas: List<RejectedFormulaSummary> = emptyList(),
    val liveLogs: List<String> = emptyList(),
    val fourStageMetrics: FourStageValidationMetrics? = null,
    val randomReference: RandomReferenceComparison = RandomReferenceComparison(),
    val backtestTriggered: Boolean = false,
    val honestStatusNote: String = "",
    val targetObjectiveStreak: Int = 4,
    val bestValidatedStreak: Int = 0,
    val activeWinnerId: String = "",
    val savedWinners: List<SymbolicFormulaSummary> = emptyList(),
    val preferredStrategyDays: Int? = null,
    val bestZeroMatchRate: Float = 1f,
    val bestThreePlusCount: Int = 0,
    val bestFourPlusCount: Int = 0,
    val bestFiveMatchCount: Int = 0,
    val excludedMethodsCount: Int = 0,
    val stagnationGenerations: Int = 0,
    val activeParadigmName: String = "Exploration Hybride",
    val infiniteMode: Boolean = false,
    val approvedBacktestNote: String = "",
    val chercheurStatus: ChercheurStatusSummary = ChercheurStatusSummary()
) {
    val isAnalyzing: Boolean get() = phase == AutonomousPhase.ANALYZING
    val isSuspended: Boolean get() = phase == AutonomousPhase.SUSPENDED
    val isValidated: Boolean get() = phase == AutonomousPhase.VALIDATED

    val matchingSavedWinners: List<SymbolicFormulaSummary>
        get() {
            val targetDays = preferredStrategyDays ?: return savedWinners
            val exact = savedWinners.filter { it.combinedStreak == targetDays }
            return if (exact.isNotEmpty()) exact else savedWinners.filter { it.combinedStreak >= targetDays }
        }

    val activeStrategySummary: SymbolicFormulaSummary?
        get() {
            val pool = matchingSavedWinners
            return pool.firstOrNull { it.id == activeWinnerId } ?: pool.firstOrNull()
        }

    val isPreferredTierUnlocked: Boolean
        get() = if (preferredStrategyDays == null) {
            phase == AutonomousPhase.VALIDATED || savedWinners.isNotEmpty() || activeWinnerId.isNotBlank()
        } else {
            matchingSavedWinners.isNotEmpty()
        }

    val hasSavedWinner: Boolean get() = savedWinners.isNotEmpty() || activeWinnerId.isNotBlank()
    val canResume: Boolean get() = phase == AutonomousPhase.SUSPENDED || phase == AutonomousPhase.STOPPED
    val canLaunchBacktest: Boolean get() = isPreferredTierUnlocked
    val selectedPatternLength: Int
        get() = (preferredStrategyDays
            ?: activeStrategySummary?.combinedStreak
            ?: targetObjectiveStreak).coerceAtLeast(1)

    fun matchedDaysCount(backtestResult: BacktestResult? = null): Int {
        return backtestResult?.maxConsecutiveQualifyingDays
            ?: activeStrategySummary?.combinedStreak
            ?: bestCombinedStreak
    }
}

enum class PredictionMethod(
    val displayName: String,
    val shortLabel: String,
    val family: String,
    val isComplex: Boolean
) {
    AUTO_SEARCH("Auto-Recherche (Moteur Chercheur)", "Auto-Chercheur", "Découverte causale", true),
    ADAPTIVE_ML("ML Adaptatif Bayésien", "ML Bayésien", "Ensemble adaptatif", true),
    COLD_HOT_BALANCE("Équilibre Chaud/Froid & Écarts", "Chaud/Froid", "Retards & diversification", false),
    DIRICHLET_MULTINOMIAL("Dirichlet-Multinomial", "Dirichlet", "Statistique bayésienne", false),
    EMA_RECENCY("Moyenne Mobile Exponentielle", "EMA", "Processus temporel", false),
    MARKOV_STRATIFIED("Markov Stratifié (Jeu/Horaire)", "Markov Strat.", "Chaînes de Markov", true),
    COOCCURRENCE_GRAPH("Graphe de Co-occurrence", "Graphe Co-occ.", "Graphes de nombres", false),
    SPECTRAL_FOURIER("Décomposition Spectrale (DFT)", "Fourier DFT", "Méthode spectrale", true),
    MATRIX_9X10("Grille 9×10 (Voisins + Cavalier)", "Grille 9×10", "Géométrie matricielle", false),
    CHANCE_CROIX_PYRAMID("Chance-Croix & Pyramides", "Croix/Pyramide", "Arithmétique & Croix", false),
    LUNAR_ASTRONOMICAL("Phases Lunaires (Expérimental)", "Lunaire (Exp.)", "Astronomie (Exp.)", false),
    CUBE90("Cube90 (Audité : 30 images)", "Cube90", "Congruence cubique", false),
    HORAIRE_OPTIM("Horaire max ∑ C(p,n_p)", "Horaire C(p,n)", "Affectation hongroise", false)
}

data class ExpertWeightInfo(
    val method: PredictionMethod,
    val name: String,
    val weightPercent: Float,
    val recentBrier: Double,
    val weightDelta: Float
)

data class DrawContext(
    val targetTimestamp: Long,
    val targetHoraire: String,
    val targetGame: String,
    val targetDayOfWeek: Int,
    val sameDayPredictedCounts: Map<Int, Int> = emptyMap(),
    val currentZeroMatchStreak: Int = 0
)

data class StrategyComparisonEntry(
    val method: PredictionMethod,
    val windowName: String,
    val testedCount: Int,
    val averageMatches: Float,
    val onePlusRate: Float,
    val twoPlusCount: Int,
    val threePlusCount: Int,
    val qualifyingDaysCount: Int,
    val totalDays: Int,
    val brierScore: Double,
    val bernoulliLogLoss: Double,
    val calibrationError: Double,
    val pValueVsRandom: Double,
    val adjustedSignificant: Boolean,
    val isRecommendedParsimonious: Boolean,
    val penultimateOkDaysCount: Int = 0,
    val objectiveValidated: Boolean = false
)

data class ComplementarityPair(
    val methodA: PredictionMethod,
    val methodB: PredictionMethod,
    val jaccardOverlap: Float,
    val errorCorrelation: Float,
    val complementaryHitsCount: Int
)

data class WindowSplitInfo(
    val totalDrawsAvailable: Int,
    val warmupDrawsCount: Int,
    val devDateRange: String,
    val devDrawsCount: Int,
    val valDateRange: String,
    val valDrawsCount: Int,
    val indepTestDateRange: String,
    val indepTestDrawsCount: Int
)

data class AutoSearchCandidateSummary(
    val id: String,
    val label: String,
    val coldRatio: Float,
    val diversityStrength: Float,
    val testedCount: Int,
    val averageMatches: Float,
    val twoPlusCount: Int,
    val qualifyingDaysCount: Int,
    val maxConsecutiveQualifyingDays: Int,
    val maxZeroMatchStreak: Int,
    val brierScore: Double,
    val isWinner: Boolean
)

data class SavedModelLabAudit(
    val id: String,
    val displayName: String,
    val formulaCompact: String,
    val slotSplitLabel: String,
    val savedStreak: Int,
    val windowName: String,
    val testedDraws: Int,
    val twoPlusDraws: Int,
    val qualifyingDaysCount: Int,
    val penultimateOkDaysCount: Int,
    val totalDays: Int,
    val averageMatches: Float,
    val zeroMatchRatePct: Int,
    val pValueVsRandom: Double,
    val causalVerified: Boolean,
    val beatsRandom: Boolean,
    val objectiveValidated: Boolean,
    val verdictReason: String
)

data class ScientificReport(
    val splitInfo: WindowSplitInfo,
    val monteCarloDev: MonteCarloBaseline,
    val monteCarloVal: MonteCarloBaseline,
    val monteCarloIndepTest: MonteCarloBaseline,
    val devComparisons: List<StrategyComparisonEntry>,
    val valComparisons: List<StrategyComparisonEntry>,
    val indepTestComparisons: List<StrategyComparisonEntry>,
    val recommendedMethod: PredictionMethod,
    val recommendationReason: String,
    val complementaryPairs: List<ComplementarityPair>,
    val cube90AuditSummary: String,
    val periodLabel: String = "",
    val savedModelAudits: List<SavedModelLabAudit> = emptyList(),
    val anyObjectiveValidated: Boolean = false
)

data class ChercheurTeaching(
    val id: String,
    val title: String,
    val scopeLabel: String,
    val principleSummary: String,
    val favoredFeatures: List<String>,
    val preferredSlotSplit: String,
    val generalizedScore: Float,
    val piercedDrawsCount: Int,
    val status: String // INJECTED, KEPT, DISCARDED
)

data class ChercheurStatusSummary(
    val isRunning: Boolean = false,
    val cyclesCompleted: Int = 0,
    val teachingsProduced: Int = 0,
    val teachingsInjected: Int = 0,
    val strategiesKept: Int = 0,
    val strategiesDiscarded: Int = 0,
    val lastLessonNote: String = "Le Chercheur est prêt à analyser numéros, positions (1..5) et tirages complets.",
    val teachings: List<ChercheurTeaching> = emptyList()
)

data class ApprovedBacktestLedgerRecord(
    val ledgerKey: String,
    val strategySignature: String,
    val periodKey: String,
    val frozenStepsJson: String,
    val pendingPredictionsJson: String,
    val lastApprovedTimestamp: Long,
    val updatedAt: Long
)

