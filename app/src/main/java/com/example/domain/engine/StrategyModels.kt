package com.example.domain.engine

import androidx.compose.runtime.Immutable
import kotlin.math.ln

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

@Immutable
data class ExpertWeightInfo(
    val method: PredictionMethod,
    val name: String,
    val weightPercent: Float,
    val recentBrier: Double,
    val weightDelta: Float
)

@Immutable
data class DrawContext(
    val targetTimestamp: Long,
    val targetHoraire: String,
    val targetGame: String,
    val targetDayOfWeek: Int,
    val sameDayPredictedCounts: Map<Int, Int> = emptyMap(),
    val currentZeroMatchStreak: Int = 0
)

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

@Immutable
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

@Immutable
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

@Immutable
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
    val rollingTwoPlusRate: Float = if (rollingDraws > 0) rollingTwoPlusCount.toFloat() / rollingDraws else 0f,
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
    val isFourStageValidated: Boolean
)

@Immutable
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
    val rollingTwoPlusRate: Float = 0f
)

@Immutable
data class RejectedFormulaSummary(
    val id: String,
    val generation: Int,
    val formulaPreview: String,
    val rejectedAtStage: String,
    val reason: String,
    val trainAvgMatches: Float,
    val validationAvgMatches: Float
)

@Immutable
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

@Immutable
data class PredictedNumberConfidence(
    val number: Int,
    val marginalProbabilityPercent: Float,
    val relativeConfidencePercent: Float,
    val liftVsRandom: Float,
    val rank: Int,
    val sourceBranch: String,
    val isGapCandidate: Boolean
)

@Immutable
data class PredictionResult(
    val predictedDraw: List<Int>,
    val expectedMatchesEstimate: Float,
    val calibratedProbAtLeastOne: Float,
    val trainedOnCount: Int,
    val method: PredictionMethod,
    val horaire: String,
    val marginalSumCheck: Float,
    val expertWeights: List<ExpertWeightInfo> = emptyList(),
    val auditNote: String = "",
    val discoveredStrategyName: String = "",
    val coldNumbersCount: Int = 0,
    val diversityPenaltyApplied: Boolean = false,
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

@Immutable
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

@Immutable
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

@Immutable
data class BacktestResult(
    val method: PredictionMethod,
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
    val learnedExpertWeights: List<ExpertWeightInfo>,
    val weightSnapshots: List<List<ExpertWeightInfo>>,
    val dailySummaries: List<DailyBacktestSummary>,
    val horaireBreakdown: List<HorairePerformance>,
    val steps: List<BacktestStepResult>,
    val maxConsecutiveQualifyingDays: Int = 0,
    val maxZeroMatchStreak: Int = 0,
    val currentZeroMatchStreak: Int = 0,
    val discoveredStrategiesSummary: List<AutoSearchCandidateSummary> = emptyList(),
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
        // Brier uniforme : p = 5/90 => E[(p - Y)^2] = p*(1-p) = (5/90)*(85/90)
        const val RANDOM_BRIER_SCORE = (5.0 / 90.0) * (85.0 / 90.0)
        // Entropie binaire uniforme par numero : -(p*ln(p) + (1-p)*ln(1-p))
        val RANDOM_BERNOULLI_LOG_LOSS =
            -(5.0 / 90.0) * ln(5.0 / 90.0) - (85.0 / 90.0) * ln(85.0 / 90.0)
    }
}

@Immutable
data class HorairePerformance(
    val horaire: String,
    val testedCount: Int,
    val averageMatches: Float,
    val onePlusCount: Int,
    val twoPlusCount: Int
)

@Immutable
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

@Immutable
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

@Immutable
data class ComplementarityPair(
    val methodA: PredictionMethod,
    val methodB: PredictionMethod,
    val jaccardOverlap: Float,
    val errorCorrelation: Float,
    val complementaryHitsCount: Int
)

@Immutable
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

@Immutable
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

@Immutable
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

@Immutable
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

@Immutable
data class ChercheurTeaching(
    val id: String,
    val title: String,
    val scopeLabel: String,
    val principleSummary: String,
    val favoredFeatures: List<String>,
    val preferredSlotSplit: String,
    val generalizedScore: Float,
    val piercedDrawsCount: Int,
    val status: String
)

@Immutable
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
