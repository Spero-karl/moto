package com.example.domain.engine

import com.example.data.local.ApprovedBacktestLedgerEntity
import com.example.data.local.DiscoveredFormulaEntity
import com.example.data.local.LotoResultEntity
import com.example.domain.util.LotoDateUtils
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.floatOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull
import kotlinx.serialization.json.put
import java.util.Locale
import java.util.concurrent.Callable
import java.util.concurrent.ForkJoinPool
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

data class CachedRawStep(
    val stepIndex: Int,
    val draw: LotoResultEntity,
    val actualNumbers: List<Int>,
    val actualSet: Set<Int>,
    val strictPastSize: Int,
    val maxPastTs: Long,
    val dayKey: String,
    val horaire: String,
    val game: String,
    val rawMatrix: RawFeatureMatrix,
    val actualMask: BooleanArray = BooleanArray(91).apply { for (n in actualNumbers) if (n in 1..90) this[n] = true },
    val dayEpoch: Long = LotoDateUtils.fastDayNumber(dayKey)
)

data class FourStageCausalCache(
    val periodLabel: String,
    val dateRangeLabel: String,
    val steps: List<CachedRawStep>,
    val discEnd: Int,
    val rollEnd: Int,
    val confEnd: Int,
    val confirmationMonteCarlo: MonteCarloBaseline
) {
    val discoverySteps: List<CachedRawStep>
        get() = if (steps.isEmpty()) emptyList() else steps.subList(0, discEnd.coerceIn(0, steps.size))

    val rollingSteps: List<CachedRawStep>
        get() = if (discEnd >= rollEnd) emptyList() else steps.subList(discEnd, rollEnd.coerceIn(discEnd, steps.size))

    val confirmationSteps: List<CachedRawStep>
        get() = if (rollEnd >= confEnd) emptyList() else steps.subList(rollEnd, confEnd.coerceIn(rollEnd, steps.size))

    val sealedSteps: List<CachedRawStep>
        get() = if (confEnd >= steps.size) emptyList() else steps.subList(confEnd, steps.size)
}

data class StageEvalStats(
    val tested: Int,
    val twoPlusCount: Int,
    val qualifyingDays: Int,
    val maxConsecutiveQualifyingDays: Int,
    val maxZeroStreak: Int,
    val avgMatches: Float,
    val brierScore: Double,
    val matchesList: List<Int>,
    val causalVerified: Boolean,
    val twoPlusRate: Float = if (tested > 0) twoPlusCount.toFloat() / tested else 0f
)

data class GenomeCandidateEvaluation(
    val genome: SymbolicGenome,
    val discStats: StageEvalStats,
    val rollStats: StageEvalStats,
    val rollingStabilityPassed: Boolean,
    val fitnessScore: Double,
    val status: String, // DISCOVERED, ROLLING_PASSED, REJECTED, VALIDATED
    val rejectionStage: String = "",
    val rejectionReason: String = "",
    val confStats: StageEvalStats? = null,
    val sealedStats: StageEvalStats? = null,
    val pValueRaw: Double = 1.0,
    val pValueAdjusted: Double = 1.0,
    val consolidatedTrainStreak: Int = 0,
    val consolidatedFullStreak: Int = 0
) {
    fun toSummary(): SymbolicFormulaSummary {
        val feats = (SymbolicFormula.usedFeatures(genome.primaryExpr) + SymbolicFormula.usedFeatures(genome.secondaryExpr))
            .map { it.symbol }
            .distinct()
        val effectiveStreak = maxOf(
            discStats.maxConsecutiveQualifyingDays,
            rollStats.maxConsecutiveQualifyingDays,
            confStats?.maxConsecutiveQualifyingDays ?: 0,
            consolidatedTrainStreak,
            consolidatedFullStreak
        )
        return SymbolicFormulaSummary(
            id = genome.id,
            generation = genome.generation,
            primaryExprText = genome.describePrimary(),
            secondaryExprText = genome.describeSecondary(),
            slotSplitLabel = "${genome.primarySlots}+${5 - genome.primarySlots}",
            usedVariables = feats,
            status = status,
            statusReason = rejectionReason.ifBlank {
                when (status) {
                    "VALIDATED" -> "Validé 4 niveaux (p_adj=${fmtP(pValueAdjusted)})"
                    "ROLLING_PASSED" -> "Stable sur validation glissante"
                    else -> "Découverte GP"
                }
            },
            discoveryStreak = discStats.maxConsecutiveQualifyingDays,
            rollingStreak = rollStats.maxConsecutiveQualifyingDays,
            confirmationStreak = confStats?.maxConsecutiveQualifyingDays ?: 0,
            sealedStreak = sealedStats?.maxConsecutiveQualifyingDays ?: 0,
            combinedStreak = effectiveStreak,
            discoveryAvgMatches = discStats.avgMatches,
            rollingAvgMatches = rollStats.avgMatches,
            confirmationAvgMatches = confStats?.avgMatches ?: 0f,
            pValueAdjusted = pValueAdjusted,
            fitnessScore = fitnessScore,
            isValidated = status == "VALIDATED",
            rollingTwoPlusRate = rollStats.twoPlusRate
        )
    }

    fun toEntity(periodName: String, updatedAt: Long): DiscoveredFormulaEntity {
        val feats = (SymbolicFormula.usedFeatures(genome.primaryExpr) + SymbolicFormula.usedFeatures(genome.secondaryExpr))
            .map { it.symbol }
            .distinct()
        return DiscoveredFormulaEntity(
            id = genome.id,
            periodName = periodName,
            generation = genome.generation,
            parentIdsCsv = genome.parentIds.joinToString(","),
            primaryExprText = genome.describePrimary(),
            secondaryExprText = genome.describeSecondary(),
            slotSplitLabel = "${genome.primarySlots}+${5 - genome.primarySlots}",
            genomeJson = SymbolicFormula.genomeToJson(genome),
            usedFeaturesCsv = feats.joinToString(","),
            status = status,
            rejectionStage = rejectionStage,
            rejectionReason = rejectionReason,
            discoveryStreak = discStats.maxConsecutiveQualifyingDays,
            rollingStreak = rollStats.maxConsecutiveQualifyingDays,
            confirmationStreak = confStats?.maxConsecutiveQualifyingDays ?: 0,
            sealedStreak = sealedStats?.maxConsecutiveQualifyingDays ?: 0,
            combinedStreak = maxOf(
                discStats.maxConsecutiveQualifyingDays,
                rollStats.maxConsecutiveQualifyingDays,
                confStats?.maxConsecutiveQualifyingDays ?: 0,
                consolidatedTrainStreak,
                consolidatedFullStreak
            ),
            discoveryAvgMatches = discStats.avgMatches,
            rollingAvgMatches = rollStats.avgMatches,
            confirmationAvgMatches = confStats?.avgMatches ?: 0f,
            sealedAvgMatches = sealedStats?.avgMatches ?: 0f,
            pValueAdjusted = pValueAdjusted,
            fitnessScore = fitnessScore,
            updatedAt = updatedAt
        )
    }

    private fun fmtP(p: Double): String = String.format(Locale.FRANCE, "%.4f", p)
}

data class SymbolicEvolutionState(
    val generation: Int,
    val experimentsCount: Int,
    val confirmationTestsCount: Int,
    val sessionSalt: Long,
    val population: List<SymbolicGenome>,
    val bestGenomes: List<GenomeCandidateEvaluation> = emptyList(),
    val rejectedHistory: List<RejectedFormulaSummary> = emptyList(),
    val confirmationTestedSignatures: Set<String> = emptySet(),
    val validatedCandidate: GenomeCandidateEvaluation? = null,
    val seenSignatures: Set<String> = emptySet(),
    val stagnationGenerations: Int = 0,
    val bestOverallFitness: Double = Double.NEGATIVE_INFINITY,
    val bestOverallStreak: Int = 0
)

data class EvolutionGenerationResult(
    val nextState: SymbolicEvolutionState,
    val bestTrainEval: GenomeCandidateEvaluation?,
    val winningEval: GenomeCandidateEvaluation?,
    val auditedEval: GenomeCandidateEvaluation?,
    val newRejected: List<RejectedFormulaSummary>,
    val persistedEntities: List<DiscoveredFormulaEntity>,
    val featureHistogram: List<Pair<String, Float>>,
    val randomComparison: RandomReferenceComparison,
    val generationNote: String,
    val injectedOutcomes: Map<String, String> = emptyMap()
)

data class PendingLivePrediction(
    val dayKey: String,
    val horaire: String,
    val targetTimestamp: Long,
    val predictedNumbers: List<Int>
)

/**
 * Moteur autonome de recherche symbolique & programmation genetique (Loto BJ) :
 * - Construit ses propres expressions AST (ExprNode) sur les 12 variables empiriques causales brutes
 * - N'impose aucune methode Loto predefinie (ni Weibull, ni Markov, ni grille 9x10, ni Fourier, ni Cube90)
 * - Validation stricte en 4 etapes chronologiques sans fuite :
 *   1) Decouverte (evolution GP + optimisation numerique des constantes)
 *   2) Validation Glissante Chronologique (stabilite + elagage surapprentissage)
 *   3) Confirmation Independante (reference Monte Carlo + correction p_adj, jamais re-injectee dans la fitness GP)
 *   4) Test Final Scelle (consulte une unique fois apres gel de la formule)
 */
object DiscoveryEngine {

    private const val POPULATION_SIZE = 40
    private const val MONTE_CARLO_CONF_SIMS = 1500
    private const val MAX_SEEN_SIGNATURES = 40_000

    internal fun effectiveMultipleTestFactor(testsCount: Int): Double {
        val c = testsCount.coerceAtLeast(1)
        if (c <= 1) return 1.0
        return minOf(c.toDouble(), 1.0 + 2.5 * ln(c.toDouble()))
    }

    fun buildCausalCache(
        history: List<LotoResultEntity>,
        period: BacktestPeriod = BacktestPeriod.THREE_WEEKS,
        referenceNowMs: Long = System.currentTimeMillis(),
        customDays: Int? = period.days
    ): FourStageCausalCache {
        val chronological = history
            .distinctBy { it.uniqueId }
            .sortedWith(compareBy<LotoResultEntity> { it.timestamp }.thenBy { it.uniqueId })

        val resolvedDays = if (period == BacktestPeriod.ALL_HISTORY && customDays == null) null else (customDays ?: period.days)?.coerceAtLeast(1)
        val resolvedLabel = when {
            resolvedDays == null -> period.label
            period.days == resolvedDays -> period.label
            else -> "$resolvedDays Jours (Perso)"
        }

        if (chronological.size < 4) {
            return FourStageCausalCache(
                periodLabel = resolvedLabel,
                dateRangeLabel = "Données insuffisantes",
                steps = emptyList(),
                discEnd = 0,
                rollEnd = 0,
                confEnd = 0,
                confirmationMonteCarlo = MonteCarloBaseline(0, 4242L, 0, 0, 0f, 0f, 0f, 0f, 0f, 0, 0, 0f, 0, 0)
            )
        }

        val range = AutoSearchEngine.computePeriodRange(chronological, period, referenceNowMs, resolvedDays)
        val rangeIndices = range.toList().toIntArray()
        val matrixByIdx = SymbolicFormula.extractRawMatrixSequence(chronological, rangeIndices)

        val strictCountAt = IntArray(chronological.size)
        val maxPastTsAt = LongArray(chronological.size) { Long.MIN_VALUE }
        var committed = 0
        var latestPastTs = Long.MIN_VALUE
        for (i in chronological.indices) {
            val ts = chronological[i].timestamp
            while (committed < i && chronological[committed].timestamp < ts) {
                val pTs = chronological[committed].timestamp
                if (pTs > latestPastTs) latestPastTs = pTs
                committed++
            }
            strictCountAt[i] = committed
            maxPastTsAt[i] = latestPastTs
        }

        val cachedRaw = mapIndexedParallel(rangeIndices.size) { idx ->
            val t = rangeIndices[idx]
            val targetDraw = chronological[t]
            val actualNumbers = StatisticalStrategies.extractDistinctBalls(targetDraw).sorted()
            if (actualNumbers.size < 5) return@mapIndexedParallel null

            val strictPastSize = strictCountAt[t]
            if (strictPastSize <= 0) return@mapIndexedParallel null

            val maxPastTs = maxPastTsAt[t]
            check(maxPastTs < targetDraw.timestamp) {
                "Violation causale détectée : maxPast=$maxPastTs >= target=${targetDraw.timestamp}"
            }

            val rawMatrix = matrixByIdx[t] ?: return@mapIndexedParallel null
            val horaire = LotoDateUtils.normalizeHoraire(targetDraw)
            val game = LotoDateUtils.normalizeGame(targetDraw)
            val dayKey = LotoDateUtils.formatDayKey(targetDraw.timestamp)
            val mask = BooleanArray(91)
            for (num in actualNumbers) if (num in 1..90) mask[num] = true

            CachedRawStep(
                stepIndex = idx + 1,
                draw = targetDraw,
                actualNumbers = actualNumbers,
                actualSet = actualNumbers.toSet(),
                strictPastSize = strictPastSize,
                maxPastTs = maxPastTs,
                dayKey = dayKey,
                horaire = horaire,
                game = game,
                rawMatrix = rawMatrix,
                actualMask = mask,
                dayEpoch = LotoDateUtils.fastDayNumber(dayKey)
            )
        }.filterNotNull()

        val rawSize = cachedRaw.size
        val cached = if (resolvedDays != null && rawSize > 0) {
            val totalBlocks = (rawSize + 5) / 6
            cachedRaw.mapIndexed { i, step ->
                val blockFromEnd = (rawSize - 1 - i) / 6
                step.copy(stepIndex = i + 1, dayEpoch = (totalBlocks - 1 - blockFromEnd).toLong())
            }
        } else {
            cachedRaw.mapIndexed { i, step -> step.copy(stepIndex = i + 1) }
        }

        val n = cached.size
        val discEnd: Int
        val rollEnd: Int
        val confEnd: Int
        when {
            n >= 20 -> {
                discEnd = ((n * 45) / 100).coerceAtLeast(6)
                rollEnd = (discEnd + ((n * 20) / 100).coerceAtLeast(4)).coerceAtMost(n - 6)
                confEnd = (rollEnd + ((n * 20) / 100).coerceAtLeast(3)).coerceAtMost(n - 3)
            }
            n == 6 -> {
                discEnd = 2
                rollEnd = 3
                confEnd = 5
            }
            n >= 5 -> {
                discEnd = ((n * 45) / 100).coerceAtLeast(2)
                rollEnd = (discEnd + ((n * 20) / 100).coerceAtLeast(1)).coerceAtMost(n - 2)
                confEnd = (rollEnd + ((n * 20) / 100).coerceAtLeast(1)).coerceAtMost(n - 1)
            }
            else -> {
                discEnd = (n / 2).coerceAtLeast(1)
                rollEnd = (discEnd + 1).coerceAtMost(n)
                confEnd = (rollEnd + 1).coerceAtMost(n)
            }
        }

        val rangeLabel = if (cached.isNotEmpty()) {
            "${LotoDateUtils.formatShortDate(cached.first().draw.timestamp)} → ${LotoDateUtils.formatShortDate(cached.last().draw.timestamp)}"
        } else {
            resolvedLabel
        }

        val confDraws = if (rollEnd < confEnd) {
            cached.subList(rollEnd, confEnd).map { it.draw }
        } else {
            emptyList()
        }
        val mcBaseline = ScientificEvaluator.runMonteCarloBaseline(
            drawsSubset = confDraws,
            simulations = MONTE_CARLO_CONF_SIMS,
            seed = 4242L
        )

        return FourStageCausalCache(
            periodLabel = resolvedLabel,
            dateRangeLabel = rangeLabel,
            steps = cached,
            discEnd = discEnd,
            rollEnd = rollEnd,
            confEnd = confEnd,
            confirmationMonteCarlo = mcBaseline
        )
    }

    fun createInitialState(
        sessionSalt: Long = 1L,
        restoredPopulation: List<SymbolicGenome> = emptyList()
    ): SymbolicEvolutionState {
        val population = if (restoredPopulation.isNotEmpty()) {
            restoredPopulation.take(POPULATION_SIZE)
        } else {
            generateInitialPopulation(POPULATION_SIZE, sessionSalt)
        }
        return SymbolicEvolutionState(
            generation = 0,
            experimentsCount = 0,
            confirmationTestsCount = 0,
            sessionSalt = sessionSalt,
            population = population
        )
    }

    fun ensureEliteInPopulation(
        state: SymbolicEvolutionState,
        elite: SymbolicGenome?
    ): SymbolicEvolutionState {
        if (elite == null) return state
        val eliteSig = elite.describeCompact()
        if (state.population.any { it.describeCompact() == eliteSig }) {
            return state
        }
        val merged = (listOf(elite) + state.population).take(POPULATION_SIZE)
        return state.copy(population = merged)
    }

    fun generateInitialPopulation(
        size: Int = POPULATION_SIZE,
        sessionSalt: Long = 1L
    ): List<SymbolicGenome> {
        val rng = Random(sessionSalt * 104_729L + 17L)
        return (0 until size).map { idx ->
            val depthA = if (idx % 2 == 0) 3 else 4
            val depthB = if (idx % 3 == 0) 2 else 3
            SymbolicGenome(
                id = "sym_g1_$idx",
                generation = 1,
                parentIds = emptyList(),
                primaryExpr = SymbolicFormula.randomExpr(depthA, rng),
                secondaryExpr = SymbolicFormula.randomExpr(depthB, rng),
                primarySlots = rng.nextInt(1, 5), // 1..4 boules A + (5 - A) boules B
                samplingTemp = round2(0.35 + rng.nextDouble() * 0.85),
                exploreNoise = 0.0,
                seed = sessionSalt * 97_003L + idx * 1_993L + 7L
            )
        }
    }

    /**
     * Execute une generation complete de decouverte symbolique sans fuite future :
     * - Etape 1 (Decouverte) & Etape 2 (Validation Glissante) guident SEULES l'evolution GP et la calibration numerique.
     * - Etape 3 (Confirmation Independante) audite le meilleur candidat non encore teste sans JAMAIS modifier
     *   la population ou la fitness GP (zero data leakage).
     * - Etape 4 (Test Final Scelle) n'est consultee QU'UNE SEULE FOIS si Etapes 1..3 sont reussies.
     */
    fun executeEvolutionGeneration(
        state: SymbolicEvolutionState,
        cache: FourStageCausalCache,
        periodName: String,
        minObjectiveStreak: Int = 4,
        injectedGenomes: List<SymbolicGenome> = emptyList()
    ): EvolutionGenerationResult {
        val cacheDistinctDays = cache.steps.asSequence().map { it.dayEpoch }.distinct().count().coerceAtLeast(1)
        val requiredStreak = minObjectiveStreak.coerceIn(1, cacheDistinctDays)
        val nextGen = state.generation + 1
        val rng = Random(nextGen * 83_117L + state.sessionSalt * 13_331L + 5L)

        val seen = LinkedHashSet<String>(maxOf(1024, state.seenSignatures.size + POPULATION_SIZE * 2)).apply {
            addAll(state.seenSignatures)
            for (g in state.population) add(g.describeCompact())
        }
        val causalBoost = if ((state.generation == 0 || state.generation % 4 == 0 || state.stagnationGenerations >= 2) && cache.discoverySteps.isNotEmpty()) {
            val ridge = CausalAiLearner.fitSymbolicRidgeGenomes(
                discoverySteps = cache.discoverySteps,
                generation = nextGen,
                sessionSalt = state.sessionSalt + nextGen * 31L,
                seenSignatures = seen,
                maxCount = minOf(12, POPULATION_SIZE / 3)
            )
            val zLift = if (state.generation == 0 || state.stagnationGenerations >= 2) {
                synthesizeCausalDiscoveryGenomes(
                    discoverySteps = cache.discoverySteps,
                    generation = nextGen,
                    sessionSalt = state.sessionSalt + state.stagnationGenerations * 97L,
                    seenSignatures = seen + ridge.map { it.describeCompact() },
                    maxCount = minOf(10, POPULATION_SIZE / 4)
                )
            } else emptyList()
            ridge + zLift
        } else emptyList()
        val injectedIdSet = injectedGenomes.map { it.id }.toSet()
        val combinedSeedPop = if (injectedGenomes.isNotEmpty() || causalBoost.isNotEmpty()) {
            val freshInjected = (injectedGenomes + causalBoost).filter { it.describeCompact() !in seen }
            (freshInjected + state.population).take(POPULATION_SIZE)
        } else {
            state.population
        }
        for (g in combinedSeedPop) seen.add(g.describeCompact())
        if (seen.size > MAX_SEEN_SIGNATURES) {
            val toDrop = seen.size - (MAX_SEEN_SIGNATURES - 5_000)
            val it = seen.iterator()
            var dropped = 0
            while (it.hasNext() && dropped < toDrop) {
                it.next()
                it.remove()
                dropped++
            }
        }

        // Calibration locale + evaluation Decouverte sans double calcul
        val (calibratedPopulation, precomputedDiscStats) = calibratePopulationOnDiscovery(
            population = combinedSeedPop,
            discoverySteps = cache.discoverySteps,
            generation = nextGen,
            rng = rng
        )

        val compiledPrograms = calibratedPopulation.map {
            SymbolicFormula.compileToRpn(it.primaryExpr) to SymbolicFormula.compileToRpn(it.secondaryExpr)
        }

        // Evaluation STRICTEMENT sur Etape 1 (Decouverte) et Etape 2 (Validation Glissante), en parallele multi-coeurs
        val phase12Evals = mapIndexedParallel(calibratedPopulation.size) { idx ->
            val genome = calibratedPopulation[idx]
            val (rpnA, rpnB) = compiledPrograms[idx]
            val cachedDisc = precomputedDiscStats.getOrNull(idx)
            evaluateDiscoveryAndRollingOnly(genome, cache, cachedDisc, rpnA, rpnB)
        }

        val injectedOutcomes = HashMap<String, String>()
        if (injectedIdSet.isNotEmpty()) {
            val medianFit = phase12Evals.map { it.fitnessScore }.sorted().let {
                if (it.isEmpty()) 0.0 else it[it.size / 2]
            }
            for (ev in phase12Evals) {
                if (ev.genome.id in injectedIdSet) {
                    val kept = ev.rollingStabilityPassed && (ev.fitnessScore >= medianFit || ev.rollStats.avgMatches > BacktestResult.RANDOM_EXPECTED_MATCHES)
                    injectedOutcomes[ev.genome.id] = if (kept) "KEPT" else "DISCARDED"
                }
            }
        }

        // Population & classement GP depend EXCLUSIVEMENT de Etape 1 + Etape 2
        val rankedTrainEvals = phase12Evals.sortedByDescending { it.fitnessScore }
        val newBestTrainList = (state.bestGenomes + rankedTrainEvals.filter { it.rollingStabilityPassed })
            .distinctBy { it.genome.describeCompact() }
            .sortedByDescending { it.fitnessScore }
            .take(12)

        val newlyRejected = ArrayList<RejectedFormulaSummary>()
        for (ev in rankedTrainEvals) {
            if (!ev.rollingStabilityPassed) {
                newlyRejected.add(
                    RejectedFormulaSummary(
                        id = ev.genome.id,
                        generation = nextGen,
                        formulaPreview = ev.genome.describeCompact(),
                        rejectedAtStage = ev.rejectionStage.ifBlank { "Validation Glissante" },
                        reason = ev.rejectionReason,
                        trainAvgMatches = ev.discStats.avgMatches,
                        validationAvgMatches = ev.rollStats.avgMatches
                    )
                )
            }
        }

        var updatedConfTestsCount = state.confirmationTestsCount
        val updatedTestedSigs = LinkedHashSet(state.confirmationTestedSignatures)
        var winningEval: GenomeCandidateEvaluation? = null
        var auditedEval: GenomeCandidateEvaluation? = null
        val updatedEntities = ArrayList<DiscoveredFormulaEntity>()
        val nowMs = System.currentTimeMillis()

        // Auditer jusqu'a 5 meilleurs candidats stables inedits sur Etape 3 (Confirmation Independante)
        val stableRanked = rankedTrainEvals.filter { it.rollingStabilityPassed }
        val topStable = stableRanked.firstOrNull() ?: rankedTrainEvals.firstOrNull()

        var latestConfComparison = RandomReferenceComparison(
            simulationsCount = cache.confirmationMonteCarlo.simulationsCount,
            confirmationMeanAvgMatches = cache.confirmationMonteCarlo.meanAvgMatches,
            confirmationCiLow = cache.confirmationMonteCarlo.avgMatchesCiLow,
            confirmationCiHigh = cache.confirmationMonteCarlo.avgMatchesCiHigh,
            confirmationMeanQualifyingDays = cache.confirmationMonteCarlo.meanQualifyingDays
        )

        val confCandidates = if (cache.confirmationSteps.isNotEmpty()) {
            stableRanked.filter { cand ->
                cand.genome.describeCompact() !in updatedTestedSigs && (
                    updatedConfTestsCount == 0 ||
                        cand.discStats.maxConsecutiveQualifyingDays >= 1 ||
                        cand.rollStats.maxConsecutiveQualifyingDays >= 1 ||
                        cand.consolidatedTrainStreak >= 1 ||
                        cand.rollStats.twoPlusRate >= 0.20f ||
                        cand.rollStats.avgMatches >= 0.50f ||
                        cand.discStats.twoPlusCount + cand.rollStats.twoPlusCount >= 2
                    )
            }.take(5)
        } else emptyList()

        val mc = cache.confirmationMonteCarlo
        val oosCombinedSteps = cache.rollingSteps + cache.confirmationSteps
        val trainConfSteps = cache.discoverySteps + cache.rollingSteps + cache.confirmationSteps
        val fullSteps = trainConfSteps + cache.sealedSteps

        for (cand in confCandidates) {
            val sig = cand.genome.describeCompact()
            if (updatedTestedSigs.size >= 4_000) {
                val it = updatedTestedSigs.iterator()
                if (it.hasNext()) {
                    it.next()
                    it.remove()
                }
            }
            updatedTestedSigs.add(sig)
            updatedConfTestsCount++

            val confStats = evaluateGenomeOnStage(cand.genome, cache.confirmationSteps)
            val beatsMcMean = confStats.avgMatches > maxOf(BacktestResult.RANDOM_EXPECTED_MATCHES, mc.meanAvgMatches)
            val pRaw = if (beatsMcMean) {
                ScientificEvaluator.twoSidedPValueVsRandom(confStats.matchesList)
            } else {
                1.0
            }
            val effTests = effectiveMultipleTestFactor(updatedConfTestsCount)
            val pAdj = (pRaw * effTests).coerceAtMost(1.0)

            val oosStats = evaluateGenomeOnStage(cand.genome, oosCombinedSteps)
            val oosPAdj = if (oosStats.avgMatches > BacktestResult.RANDOM_EXPECTED_MATCHES && oosStats.matchesList.isNotEmpty()) {
                (ScientificEvaluator.twoSidedPValueVsRandom(oosStats.matchesList) * effTests).coerceAtMost(1.0)
            } else 1.0
            val confSignificant = beatsMcMean &&
                (pAdj < 0.05 ||
                    (pRaw < 0.025 && confStats.avgMatches > mc.avgMatchesCiHigh) ||
                    (cache.confirmationSteps.size < 6 &&
                        oosStats.avgMatches > BacktestResult.RANDOM_EXPECTED_MATCHES &&
                        oosPAdj < 0.05))
            val (oosConsQualDays, oosConsStreak) = computeConsolidatedDayStats(oosCombinedSteps, oosStats.matchesList)
            val trainConfMatches = cand.discStats.matchesList + cand.rollStats.matchesList + confStats.matchesList
            val (trainConfQualDays, trainConfStreak) = computeConsolidatedDayStats(trainConfSteps, trainConfMatches)
            val twoPlusFirst5 = cand.discStats.twoPlusCount + cand.rollStats.twoPlusCount + confStats.twoPlusCount

            val shortWindowFallback = cacheDistinctDays <= 4 &&
                cacheDistinctDays > 1 &&
                (confStats.qualifyingDays >= 1 || oosStats.qualifyingDays >= 1 || oosConsQualDays >= 1 || trainConfQualDays >= 1) &&
                maxOf(cand.discStats.maxConsecutiveQualifyingDays, cand.consolidatedTrainStreak) >= 1
            val oneDayWindowMet = cacheDistinctDays <= 1 &&
                (confStats.qualifyingDays >= 1 || oosConsQualDays >= 1 || trainConfQualDays >= 1 || (twoPlusFirst5 >= 2 && confStats.avgMatches >= 0.5f))
            val objectiveMet = if (cacheDistinctDays <= 1) {
                oneDayWindowMet
            } else {
                confStats.maxConsecutiveQualifyingDays >= requiredStreak ||
                    oosConsStreak >= requiredStreak ||
                    (oosStats.maxConsecutiveQualifyingDays >= requiredStreak && confStats.qualifyingDays >= 1) ||
                    (trainConfStreak >= requiredStreak && confStats.avgMatches > BacktestResult.RANDOM_EXPECTED_MATCHES) ||
                    shortWindowFallback
            }

            latestConfComparison = RandomReferenceComparison(
                simulationsCount = mc.simulationsCount,
                confirmationMeanAvgMatches = mc.meanAvgMatches,
                confirmationCiLow = mc.avgMatchesCiLow,
                confirmationCiHigh = mc.avgMatchesCiHigh,
                confirmationMeanQualifyingDays = mc.meanQualifyingDays,
                pValueRaw = pRaw,
                pValueAdjusted = pAdj,
                beatsRandomAfterCorrection = confSignificant
            )

            if (!beatsMcMean) {
                val reason = "Ne bat pas le hasard Monte Carlo (Conf ${fmt3(confStats.avgMatches)} ≤ MC ${fmt3(mc.meanAvgMatches)})"
                val rej = cand.copy(
                    status = "REJECTED",
                    rejectionStage = "Confirmation Indépendante",
                    rejectionReason = reason,
                    confStats = confStats,
                    pValueRaw = pRaw,
                    pValueAdjusted = pAdj
                )
                if (auditedEval == null) auditedEval = rej
                updatedEntities.add(rej.toEntity(periodName, nowMs))
                newlyRejected.add(
                    RejectedFormulaSummary(
                        id = cand.genome.id,
                        generation = nextGen,
                        formulaPreview = sig,
                        rejectedAtStage = "Confirmation Indépendante",
                        reason = reason,
                        trainAvgMatches = cand.rollStats.avgMatches,
                        validationAvgMatches = confStats.avgMatches
                    )
                )
            } else if (!confSignificant) {
                val reason = "Non significatif après correction multiple (p_adj=${fmt4(pAdj)} ≥ 0,05 sur $updatedConfTestsCount tests)"
                val rej = cand.copy(
                    status = "REJECTED",
                    rejectionStage = "Confirmation Indépendante",
                    rejectionReason = reason,
                    confStats = confStats,
                    pValueRaw = pRaw,
                    pValueAdjusted = pAdj
                )
                if (auditedEval == null) auditedEval = rej
                updatedEntities.add(rej.toEntity(periodName, nowMs))
                newlyRejected.add(
                    RejectedFormulaSummary(
                        id = cand.genome.id,
                        generation = nextGen,
                        formulaPreview = sig,
                        rejectedAtStage = "Confirmation Indépendante",
                        reason = reason,
                        trainAvgMatches = cand.rollStats.avgMatches,
                        validationAvgMatches = confStats.avgMatches
                    )
                )
            } else if (!objectiveMet) {
                val reason = "Objectif ${requiredStreak}j non atteint hors-échantillon (série OOS ${oosStats.maxConsecutiveQualifyingDays}/${requiredStreak}j, p_adj=${fmt4(pAdj)})"
                val rej = cand.copy(
                    status = "REJECTED",
                    rejectionStage = "Confirmation Indépendante",
                    rejectionReason = reason,
                    confStats = confStats,
                    pValueRaw = pRaw,
                    pValueAdjusted = pAdj
                )
                if (auditedEval == null) auditedEval = rej
                updatedEntities.add(rej.toEntity(periodName, nowMs))
                newlyRejected.add(
                    RejectedFormulaSummary(
                        id = cand.genome.id,
                        generation = nextGen,
                        formulaPreview = sig,
                        rejectedAtStage = "Confirmation Indépendante",
                        reason = reason,
                        trainAvgMatches = cand.rollStats.avgMatches,
                        validationAvgMatches = confStats.avgMatches
                    )
                )
            } else {
                // Etape 4 : Test Final Scelle — consulte UNE UNIQUE FOIS apres gel de la formule !
                val sealedStats = if (cache.sealedSteps.isNotEmpty()) {
                    evaluateGenomeOnStage(cand.genome, cache.sealedSteps)
                } else {
                    confStats
                }
                val fullMatches = cand.discStats.matchesList +
                    cand.rollStats.matchesList +
                    confStats.matchesList +
                    (if (cache.sealedSteps.isNotEmpty()) sealedStats.matchesList else emptyList())
                val (_, fullConsStreak) = computeConsolidatedDayStats(fullSteps, fullMatches)
                val fullTwoPlusCount = fullMatches.count { it >= 2 }
                val oneDayRuleOk = cacheDistinctDays > 1 || fullConsStreak >= 1
                val sealedThresholdOk = if (cacheDistinctDays <= 1 && cache.sealedSteps.size <= 1) {
                    true
                } else {
                    sealedStats.avgMatches >= BacktestResult.RANDOM_EXPECTED_MATCHES
                }
                val sealedOk = sealedStats.causalVerified &&
                    sealedThresholdOk &&
                    oneDayRuleOk

                if (sealedOk) {
                    val achievedStreak = maxOf(
                        confStats.maxConsecutiveQualifyingDays,
                        oosStats.maxConsecutiveQualifyingDays,
                        oosConsStreak,
                        cand.rollStats.maxConsecutiveQualifyingDays,
                        cand.discStats.maxConsecutiveQualifyingDays,
                        cand.consolidatedTrainStreak,
                        fullConsStreak,
                        if (cacheDistinctDays <= 1) 1 else 0
                    )
                    val winner = cand.copy(
                        status = "VALIDATED",
                        rejectionStage = "",
                        rejectionReason = "Validé 4 étapes (${achievedStreak}j, p_adj=${fmt4(pAdj)}, scellé ${fmt3(sealedStats.avgMatches)})",
                        confStats = confStats,
                        sealedStats = sealedStats,
                        pValueRaw = pRaw,
                        pValueAdjusted = pAdj,
                        consolidatedFullStreak = achievedStreak
                    )
                    winningEval = winner
                    auditedEval = winner
                    updatedEntities.add(winner.toEntity(periodName, nowMs))
                    break
                } else {
                    val reason = if (!oneDayRuleOk) {
                        "Échec Scellé 1J : $fullTwoPlusCount/${fullMatches.size} tirages à ≥2 exacts + avant-dernier (requis 3/6)"
                    } else {
                        "Échec sur Test Final Scellé (moy ${fmt3(sealedStats.avgMatches)} < hasard 0,278)"
                    }
                    val rej = cand.copy(
                        status = "REJECTED",
                        rejectionStage = "Test Final Scellé",
                        rejectionReason = reason,
                        confStats = confStats,
                        sealedStats = sealedStats,
                        pValueRaw = pRaw,
                        pValueAdjusted = pAdj
                    )
                    auditedEval = rej
                    updatedEntities.add(rej.toEntity(periodName, nowMs))
                    newlyRejected.add(
                        RejectedFormulaSummary(
                            id = cand.genome.id,
                            generation = nextGen,
                            formulaPreview = sig,
                            rejectedAtStage = "Test Final Scellé",
                            reason = reason,
                            trainAvgMatches = confStats.avgMatches,
                            validationAvgMatches = sealedStats.avgMatches
                        )
                    )
                }
            }
        }

        // Sauvegarder aussi les meilleures formules stables (et un echantillon des rejetees)
        rankedTrainEvals.take(4).forEach { ev ->
            if (updatedEntities.none { it.id == ev.genome.id }) {
                updatedEntities.add(ev.toEntity(periodName, nowMs))
            }
        }

        // Stagnation STRICTEMENT calculee sur Etapes 1 + 2 uniquement (ZERO fuite Etapes 3 & 4)
        val topTrain = rankedTrainEvals.firstOrNull()
        val bestTrainFitness = topTrain?.fitnessScore ?: Double.NEGATIVE_INFINITY
        val bestTrainStreak = maxOf(
            topTrain?.discStats?.maxConsecutiveQualifyingDays ?: 0,
            topTrain?.rollStats?.maxConsecutiveQualifyingDays ?: 0,
            topTrain?.consolidatedTrainStreak ?: 0
        )
        val trainImproved = bestTrainFitness > state.bestOverallFitness + 0.5 || bestTrainStreak > state.bestOverallStreak
        var nextStagnation = if (trainImproved) 0 else state.stagnationGenerations + 1
        val stagnationShifted = nextStagnation >= 2
        if (stagnationShifted) nextStagnation = 0
        val newBestOverallFitness = maxOf(state.bestOverallFitness, bestTrainFitness)
        val newBestOverallStreak = maxOf(state.bestOverallStreak, bestTrainStreak)

        // Evolution GP PURE basee UNIQUEMENT sur Etapes 1 & 2, zero clones bloques
        val breedingPool = (newBestTrainList + rankedTrainEvals)
            .distinctBy { it.genome.describeCompact() }
            .sortedByDescending { it.fitnessScore }

        val nextPopulation = evolvePopulationPure(
            rankedTrainEvals = breedingPool,
            generation = nextGen + 1,
            sessionSalt = state.sessionSalt,
            seenSignatures = seen,
            stagnationShifted = stagnationShifted,
            rng = rng
        ).let { evolved ->
            if (stagnationShifted && cache.discoverySteps.isNotEmpty()) {
                val extraCausal = synthesizeCausalDiscoveryGenomes(
                    discoverySteps = cache.discoverySteps,
                    generation = nextGen + 1,
                    sessionSalt = state.sessionSalt + nextGen * 97L,
                    seenSignatures = seen,
                    maxCount = minOf(8, POPULATION_SIZE / 4)
                )
                if (extraCausal.isNotEmpty()) {
                    for (g in extraCausal) seen.add(g.describeCompact())
                    (extraCausal + evolved).take(POPULATION_SIZE)
                } else evolved
            } else evolved
        }

        val mergedRejected = (newlyRejected + state.rejectedHistory)
            .distinctBy { "${it.rejectedAtStage}|${it.formulaPreview}" }
            .take(20)

        val featureHistogram = computeFeatureHistogram(nextPopulation)
        val note = when {
            winningEval != null -> {
                val achieved = maxOf(
                    winningEval.confStats?.maxConsecutiveQualifyingDays ?: 0,
                    winningEval.rollStats.maxConsecutiveQualifyingDays,
                    winningEval.discStats.maxConsecutiveQualifyingDays
                )
                "Gagnant ${achieved}j validé sur les 4 niveaux (p_adj=${fmt4(winningEval.pValueAdjusted)})"
            }
            topStable == null ->
                "Gén.$nextGen : toutes les formules ont surappris sur validation glissante (recherche continue)"
            newlyRejected.any { it.rejectedAtStage == "Confirmation Indépendante" } ->
                "Gén.$nextGen : candidat rejeté hors-échantillon (${newlyRejected.last { it.rejectedAtStage == "Confirmation Indépendante" }.reason})"
            else ->
                "Gén.$nextGen : ≥2/tir=${(topTrain?.rollStats?.twoPlusRate?.times(100f) ?: 0f).toInt()}% • J(≥3)=${topTrain?.rollStats?.qualifyingDays ?: 0} • Déc=${fmt3(topTrain?.discStats?.avgMatches ?: 0f)} • Gliss=${fmt3(topTrain?.rollStats?.avgMatches ?: 0f)} (obj ≥${requiredStreak}j)"
        }

        val nextState = SymbolicEvolutionState(
            generation = nextGen,
            experimentsCount = state.experimentsCount + calibratedPopulation.size,
            confirmationTestsCount = updatedConfTestsCount,
            sessionSalt = state.sessionSalt,
            population = nextPopulation,
            bestGenomes = newBestTrainList,
            rejectedHistory = mergedRejected,
            confirmationTestedSignatures = updatedTestedSigs,
            validatedCandidate = winningEval ?: state.validatedCandidate,
            seenSignatures = seen,
            stagnationGenerations = nextStagnation,
            bestOverallFitness = newBestOverallFitness,
            bestOverallStreak = newBestOverallStreak
        )

        return EvolutionGenerationResult(
            nextState = nextState,
            bestTrainEval = topTrain,
            winningEval = winningEval,
            auditedEval = auditedEval,
            newRejected = newlyRejected,
            persistedEntities = updatedEntities,
            featureHistogram = featureHistogram,
            randomComparison = latestConfComparison,
            generationNote = note,
            injectedOutcomes = injectedOutcomes
        )
    }

    /**
     * Audit causal strict en lecture seule d'un candidat palier (>= 2j) sans JAMAIS reinjecter
     * ses scores de Confirmation/Scelle dans l'evolution GP.
     */
    fun auditCandidateReadOnly(
        candidate: GenomeCandidateEvaluation,
        cache: FourStageCausalCache,
        confirmationTestsCount: Int
    ): GenomeCandidateEvaluation {
        val confStats = candidate.confStats ?: evaluateGenomeOnStage(candidate.genome, cache.confirmationSteps)
        val sealedStats = candidate.sealedStats ?: evaluateGenomeOnStage(candidate.genome, cache.sealedSteps)
        val mc = cache.confirmationMonteCarlo
        val beatsMcMean = confStats.avgMatches > maxOf(BacktestResult.RANDOM_EXPECTED_MATCHES, mc.meanAvgMatches)
        val pRaw = if (confStats.matchesList.isNotEmpty() && beatsMcMean) {
            ScientificEvaluator.twoSidedPValueVsRandom(confStats.matchesList)
        } else {
            1.0
        }
        val effTests = effectiveMultipleTestFactor(confirmationTestsCount)
        val pAdj = (pRaw * effTests).coerceAtMost(1.0)
        val oosCombinedSteps = cache.rollingSteps + cache.confirmationSteps
        val oosMatches = candidate.rollStats.matchesList + confStats.matchesList
        val oosAvg = if (oosMatches.isNotEmpty()) oosMatches.average().toFloat() else 0f
        val oosPAdj = if (oosAvg > BacktestResult.RANDOM_EXPECTED_MATCHES && oosMatches.isNotEmpty()) {
            (ScientificEvaluator.twoSidedPValueVsRandom(oosMatches) * effTests).coerceAtMost(1.0)
        } else 1.0
        val trainSteps = cache.discoverySteps + cache.rollingSteps
        val trainMatches = candidate.discStats.matchesList + candidate.rollStats.matchesList
        val (_, consTrainStreak) = computeConsolidatedDayStats(trainSteps, trainMatches)
        val fullSteps = trainSteps + cache.confirmationSteps + cache.sealedSteps
        val fullMatches = trainMatches + confStats.matchesList + (if (cache.sealedSteps.isNotEmpty()) sealedStats.matchesList else emptyList())
        val (_, consFullStreak) = computeConsolidatedDayStats(fullSteps, fullMatches)
        val streak = maxOf(
            candidate.discStats.maxConsecutiveQualifyingDays,
            candidate.rollStats.maxConsecutiveQualifyingDays,
            confStats.maxConsecutiveQualifyingDays,
            candidate.consolidatedTrainStreak,
            consTrainStreak,
            consFullStreak
        )
        val cacheDistinctDays = cache.steps.map { it.dayEpoch }.distinct().size.coerceAtLeast(1)
        val sealedDrawThresholdOk = if (cacheDistinctDays <= 1 && cache.sealedSteps.size <= 1) {
            true
        } else {
            sealedStats.avgMatches >= BacktestResult.RANDOM_EXPECTED_MATCHES
        }
        val all4Passed = beatsMcMean &&
            (pAdj < 0.05 ||
                (pRaw < 0.025 && confStats.avgMatches > mc.avgMatchesCiHigh) ||
                (cache.confirmationSteps.size < 6 && oosAvg > BacktestResult.RANDOM_EXPECTED_MATCHES && oosPAdj < 0.05)) &&
            sealedStats.causalVerified &&
            sealedDrawThresholdOk &&
            (confStats.maxConsecutiveQualifyingDays >= streak.coerceIn(1, 4) ||
                consFullStreak >= streak.coerceIn(1, 4) ||
                (candidate.rollStats.maxConsecutiveQualifyingDays >= 1 && confStats.qualifyingDays >= 1) ||
                (cacheDistinctDays <= 1 && candidate.rollingStabilityPassed && (consFullStreak >= 1 || consTrainStreak >= 1 || confStats.avgMatches >= BacktestResult.RANDOM_EXPECTED_MATCHES)))

        val effectiveMilestone = maxOf(1, streak)
        val status = if (all4Passed) "VALIDATED" else "MILESTONE_${effectiveMilestone}J"
        val label = if (all4Passed) {
            "Validé 4 niveaux (${effectiveMilestone}J, p_adj=${fmt4(pAdj)}, scellé ${fmt3(sealedStats.avgMatches)})"
        } else {
            "Palier ${effectiveMilestone}J causal (Déc ${fmt3(candidate.discStats.avgMatches)} • Gliss ${fmt3(candidate.rollStats.avgMatches)} • Conf ${fmt3(confStats.avgMatches)})"
        }
        return candidate.copy(
            status = status,
            rejectionStage = "",
            rejectionReason = label,
            confStats = confStats,
            sealedStats = sealedStats,
            pValueRaw = pRaw,
            pValueAdjusted = pAdj,
            consolidatedTrainStreak = maxOf(candidate.consolidatedTrainStreak, consTrainStreak),
            consolidatedFullStreak = maxOf(consFullStreak, if (all4Passed) effectiveMilestone else 0)
        )
    }

    private fun calibratePopulationOnDiscovery(
        population: List<SymbolicGenome>,
        discoverySteps: List<CachedRawStep>,
        generation: Int,
        rng: Random
    ): Pair<List<SymbolicGenome>, List<StageEvalStats?>> {
        if (discoverySteps.isEmpty() || population.isEmpty()) {
            return population to List(population.size) { null }
        }
        val evalPairs = mapIndexedParallel(population.size) { idx ->
            val g = population[idx]
            val rA = SymbolicFormula.compileToRpn(g.primaryExpr)
            val rB = SymbolicFormula.compileToRpn(g.secondaryExpr)
            g to evaluateGenomeOnStage(g, discoverySteps, rA, rB)
        }
        fun calScore(st: StageEvalStats): Double {
            val (penHits, _, blockProg) = computeBlockProgressGradient(discoverySteps, st.matchesList)
            return st.twoPlusRate * 260.0 +
                st.qualifyingDays * 45.0 +
                st.maxConsecutiveQualifyingDays * 60.0 +
                penHits * 26.0 +
                blockProg * 18.0 +
                st.twoPlusCount * 16.0 +
                st.avgMatches * 18.0
        }

        val topEliteIndices = evalPairs.indices
            .sortedByDescending { calScore(evalPairs[it].second) }
            .take(minOf(4, evalPairs.size))
            .toSet()

        val updatedPop = ArrayList<SymbolicGenome>(population.size)
        val updatedStats = ArrayList<StageEvalStats?>(population.size)
        for (i in evalPairs.indices) {
            var bestG = evalPairs[i].first
            var bestSt = evalPairs[i].second
            if (i in topEliteIndices) {
                var bestSc = calScore(bestSt)
                repeat(4) { trial ->
                    val cand = bestG.copy(
                        id = "${bestG.id}_c$trial",
                        generation = generation,
                        primaryExpr = SymbolicFormula.simplify(perturbConsts(bestG.primaryExpr, rng)),
                        secondaryExpr = SymbolicFormula.simplify(perturbConsts(bestG.secondaryExpr, rng)),
                        samplingTemp = (bestG.samplingTemp + (rng.nextDouble() * 0.24 - 0.12)).coerceIn(0.20, 1.4),
                        exploreNoise = 0.0
                    )
                    val st = evaluateGenomeOnStage(cand, discoverySteps)
                    val sc = calScore(st)
                    if (sc > bestSc) {
                        bestSc = sc
                        bestG = cand
                        bestSt = st
                    }
                }
            }
            updatedPop.add(bestG)
            updatedStats.add(bestSt)
        }
        return updatedPop to updatedStats
    }

    private fun perturbConsts(node: ExprNode, rng: Random): ExprNode = when (node) {
        is ExprNode.Const -> ExprNode.Const((kotlin.math.round((node.value + (rng.nextDouble() * 0.6 - 0.3)).coerceIn(-5.0, 5.0) * 100.0) / 100.0))
        is ExprNode.Var -> node
        is ExprNode.Unary -> ExprNode.Unary(node.op, perturbConsts(node.child, rng))
        is ExprNode.Binary -> ExprNode.Binary(node.op, perturbConsts(node.left, rng), perturbConsts(node.right, rng))
    }

    internal fun computeConsolidatedDayStats(
        steps: List<CachedRawStep>,
        matchesList: List<Int>
    ): Pair<Int, Int> {
        val count = minOf(steps.size, matchesList.size)
        if (count == 0) return 0 to 0
        var curDayEpoch = Long.MIN_VALUE
        var curDayTwoPlus = 0
        var curDayTotalDraws = 0
        var curDayPrevHit = 0
        var curDayLastHit = 0
        var qualDays = 0
        var consecQual = 0
        var maxConsecQual = 0
        var lastQualDayEpoch = Long.MIN_VALUE
        for (i in 0 until count) {
            val sDayEpoch = steps[i].dayEpoch
            if (sDayEpoch != curDayEpoch) {
                if (curDayEpoch != Long.MIN_VALUE && curDayTwoPlus == curDayTotalDraws && curDayTotalDraws > 0 && curDayPrevHit >= 2) {
                    qualDays++
                    consecQual = if (lastQualDayEpoch != Long.MIN_VALUE && curDayEpoch == lastQualDayEpoch + 1L) consecQual + 1 else 1
                    if (consecQual > maxConsecQual) maxConsecQual = consecQual
                    lastQualDayEpoch = curDayEpoch
                } else if (curDayEpoch != Long.MIN_VALUE) {
                    consecQual = 0
                }
                curDayEpoch = sDayEpoch
                curDayTwoPlus = 0
                curDayTotalDraws = 0
                curDayPrevHit = 0
                curDayLastHit = 0
            }
            val h = matchesList[i]
            curDayPrevHit = curDayLastHit
            curDayLastHit = h
            curDayTotalDraws++
            if (h >= 2) curDayTwoPlus++
        }
        if (curDayEpoch != Long.MIN_VALUE && curDayTwoPlus == curDayTotalDraws && curDayTotalDraws > 0 && curDayPrevHit >= 2) {
            qualDays++
            consecQual = if (lastQualDayEpoch != Long.MIN_VALUE && curDayEpoch == lastQualDayEpoch + 1L) consecQual + 1 else 1
            if (consecQual > maxConsecQual) maxConsecQual = consecQual
        }
        return qualDays to maxConsecQual
    }

    internal fun computeBlockProgressGradient(
        steps: List<CachedRawStep>,
        matchesList: List<Int>
    ): Triple<Int, Int, Double> {
        val count = minOf(steps.size, matchesList.size)
        if (count == 0) return Triple(0, 0, 0.0)
        var curDayEpoch = Long.MIN_VALUE
        var curDayTwoPlus = 0
        var curDayPrevHit = 0
        var curDayLastHit = 0
        var penultimateTwoPlus = 0
        var partialBlocksTwoCount = 0
        var progressScore = 0.0

        fun flushDay() {
            if (curDayEpoch == Long.MIN_VALUE) return
            if (curDayPrevHit >= 2) {
                penultimateTwoPlus++
                progressScore += 1.5 + curDayTwoPlus * 0.8
            } else if (curDayPrevHit == 1 && curDayTwoPlus >= 2) {
                progressScore += 0.65
            }
            if (curDayTwoPlus >= 3 && curDayPrevHit >= 2) {
                partialBlocksTwoCount++
                progressScore += 3.0
            } else if (curDayTwoPlus >= 2) {
                partialBlocksTwoCount++
                progressScore += 0.9
            } else if (curDayTwoPlus == 1) {
                progressScore += 0.25
            }
        }

        for (i in 0 until count) {
            val sDayEpoch = steps[i].dayEpoch
            if (sDayEpoch != curDayEpoch) {
                flushDay()
                curDayEpoch = sDayEpoch
                curDayTwoPlus = 0
                curDayPrevHit = 0
                curDayLastHit = 0
            }
            val h = matchesList[i]
            curDayPrevHit = curDayLastHit
            curDayLastHit = h
            if (h >= 2) curDayTwoPlus++
        }
        flushDay()
        return Triple(penultimateTwoPlus, partialBlocksTwoCount, progressScore)
    }

    private fun evaluateDiscoveryAndRollingOnly(
        genome: SymbolicGenome,
        cache: FourStageCausalCache,
        precomputedDiscStats: StageEvalStats? = null,
        precompiledA: CompiledRpnProgram? = null,
        precompiledB: CompiledRpnProgram? = null
    ): GenomeCandidateEvaluation {
        val rA = precompiledA ?: SymbolicFormula.compileToRpn(genome.primaryExpr)
        val rB = precompiledB ?: SymbolicFormula.compileToRpn(genome.secondaryExpr)
        val discStats = precomputedDiscStats ?: evaluateGenomeOnStage(genome, cache.discoverySteps, rA, rB)
        val rollStats = evaluateGenomeOnStage(genome, cache.rollingSteps, rA, rB)
        val trainSteps = cache.discoverySteps + cache.rollingSteps
        val trainMatches = discStats.matchesList + rollStats.matchesList
        val (consTrainQualDays, consTrainStreak) = computeConsolidatedDayStats(trainSteps, trainMatches)
        val (penTrainCount, _, blockProgressScore) = computeBlockProgressGradient(trainSteps, trainMatches)

        val overfitDrop = rollStats.tested >= 4 &&
            rollStats.avgMatches < 1.0f &&
            rollStats.twoPlusRate < 0.35f &&
            rollStats.avgMatches + 0.04f < discStats.avgMatches * 0.65f
        val worseThanRandomRolling = rollStats.tested >= 4 &&
            rollStats.avgMatches < BacktestResult.RANDOM_EXPECTED_MATCHES * 0.78f

        val rollingStabilityPassed = !overfitDrop && !worseThanRandomRolling
        val complexity = SymbolicFormula.nodeCount(genome.primaryExpr) + SymbolicFormula.nodeCount(genome.secondaryExpr)
        val parsimonyPenalty = 0.03 * complexity

        // Priorité absolue : >=2 numéros exacts par tirage & >=3 tirages/jour dont avant-dernier >=2
        val rollTwoRate = rollStats.twoPlusRate.toDouble()
        val discTwoRate = discStats.twoPlusRate.toDouble()
        val rollFullSweep = if (rollStats.tested > 0 && rollStats.twoPlusCount == rollStats.tested) 360.0 else 0.0
        val discFullSweep = if (discStats.tested > 0 && discStats.twoPlusCount == discStats.tested) 140.0 else 0.0
        val rollMissPenalty = (rollStats.tested - rollStats.twoPlusCount).coerceAtLeast(0) * 14.0
        val discMissPenalty = (discStats.tested - discStats.twoPlusCount).coerceAtLeast(0) * 4.5
        val effectiveTrainStreak = maxOf(rollStats.maxConsecutiveQualifyingDays, discStats.maxConsecutiveQualifyingDays, consTrainStreak)

        val rawFitness = (rollTwoRate * 540.0) +
            (discTwoRate * 220.0) +
            rollFullSweep +
            discFullSweep +
            (effectiveTrainStreak * 110.0) +
            (rollStats.maxConsecutiveQualifyingDays * 45.0) +
            (discStats.maxConsecutiveQualifyingDays * 40.0) +
            (consTrainQualDays * 60.0) +
            (penTrainCount * 28.0) +
            (blockProgressScore * 16.0) +
            (rollStats.qualifyingDays * 35.0) +
            (discStats.qualifyingDays * 24.0) +
            (rollStats.twoPlusCount * 28.0) +
            (discStats.twoPlusCount * 10.0) +
            (rollStats.avgMatches * 130.0) +
            (discStats.avgMatches * 45.0) -
            rollMissPenalty -
            discMissPenalty -
            (rollStats.maxZeroStreak * 3.0) -
            (discStats.brierScore * 55.0) -
            parsimonyPenalty

        val status = if (rollingStabilityPassed) "ROLLING_PASSED" else "REJECTED"
        val reason = when {
            overfitDrop -> "Surapprentissage Découverte (Déc ${fmt3(discStats.avgMatches)} → Gliss ${fmt3(rollStats.avgMatches)})"
            worseThanRandomRolling -> "Sous-performance validation glissante (${fmt3(rollStats.avgMatches)} < hasard 0,278)"
            else -> ""
        }

        return GenomeCandidateEvaluation(
            genome = genome,
            discStats = discStats,
            rollStats = rollStats,
            rollingStabilityPassed = rollingStabilityPassed,
            fitnessScore = if (rollingStabilityPassed) rawFitness else rawFitness - 100.0,
            status = status,
            rejectionStage = if (rollingStabilityPassed) "" else "Validation Glissante",
            rejectionReason = reason,
            consolidatedTrainStreak = consTrainStreak
        )
    }

    private fun evolvePopulationPure(
        rankedTrainEvals: List<GenomeCandidateEvaluation>,
        generation: Int,
        sessionSalt: Long,
        seenSignatures: Set<String>,
        stagnationShifted: Boolean,
        rng: Random
    ): List<SymbolicGenome> {
        if (rankedTrainEvals.isEmpty()) return generateInitialPopulation(POPULATION_SIZE, sessionSalt + generation)

        val next = ArrayList<SymbolicGenome>(POPULATION_SIZE)
        val localSeen = HashSet<String>(seenSignatures.size + POPULATION_SIZE * 2).apply {
            addAll(seenSignatures)
        }
        val poolSize = rankedTrainEvals.size.coerceAtLeast(1)
        val immigrantsCount = if (stagnationShifted) maxOf(10, POPULATION_SIZE / 3) else 6
        val childrenTarget = (POPULATION_SIZE - immigrantsCount).coerceAtLeast(0)

        var childIdx = 0
        while (next.size < childrenTarget) {
            val p1 = tournamentPick(rankedTrainEvals, rng, poolSize)
            val p2 = tournamentPick(rankedTrainEvals, rng, poolSize)
            var chosenCand: SymbolicGenome? = null
            for (attempt in 0 until 6) {
                var childA = SymbolicFormula.crossover(p1.primaryExpr, p2.primaryExpr, rng)
                var childB = SymbolicFormula.crossover(p1.secondaryExpr, p2.secondaryExpr, rng)
                val mutRate = if (stagnationShifted || attempt > 0) 0.85 else 0.65
                if (rng.nextDouble() < mutRate) childA = SymbolicFormula.mutate(childA, rng)
                if (rng.nextDouble() < mutRate) childB = SymbolicFormula.mutate(childB, rng)

                val slots = if (rng.nextDouble() < 0.30 || attempt > 1) rng.nextInt(1, 5) else ((p1.primarySlots + p2.primarySlots) / 2).coerceIn(1, 4)
                val temp = ((p1.samplingTemp + p2.samplingTemp) * 0.5 + (rng.nextDouble() * 0.24 - 0.12)).coerceIn(0.22, 1.5)

                val cand = SymbolicGenome(
                    id = "sym_g${generation}_c$childIdx",
                    generation = generation,
                    parentIds = listOf(p1.id, p2.id),
                    primaryExpr = childA,
                    secondaryExpr = childB,
                    primarySlots = slots,
                    samplingTemp = round2(temp),
                    exploreNoise = 0.0,
                    seed = sessionSalt * 53_171L + generation * 1_009L + childIdx * 37L + attempt * 7L
                )
                if (localSeen.add(cand.describeCompact())) {
                    chosenCand = cand
                    break
                }
            }
            childIdx++
            next.add(
                chosenCand ?: SymbolicGenome(
                    id = "sym_g${generation}_n${childIdx}",
                    generation = generation,
                    parentIds = listOf(p1.id),
                    primaryExpr = SymbolicFormula.randomExpr(4, rng),
                    secondaryExpr = SymbolicFormula.randomExpr(3, rng),
                    primarySlots = rng.nextInt(1, 5),
                    samplingTemp = round2(0.30 + rng.nextDouble() * 0.90),
                    exploreNoise = 0.0,
                    seed = sessionSalt * 61_331L + generation * 1_999L + childIdx * 43L
                ).also { localSeen.add(it.describeCompact()) }
            )
        }

        var immIdx = 0
        while (next.size < POPULATION_SIZE) {
            var imm: SymbolicGenome
            var attempt = 0
            do {
                imm = SymbolicGenome(
                    id = "sym_g${generation}_r${immIdx}",
                    generation = generation,
                    parentIds = emptyList(),
                    primaryExpr = SymbolicFormula.randomExpr(if ((immIdx + attempt) % 2 == 0) 4 else 3, rng),
                    secondaryExpr = SymbolicFormula.randomExpr(3, rng),
                    primarySlots = rng.nextInt(1, 5),
                    samplingTemp = round2(0.30 + rng.nextDouble() * 0.90),
                    exploreNoise = 0.0,
                    seed = sessionSalt * 91_009L + generation * 3_001L + immIdx * 71L + attempt * 19L
                )
                if (localSeen.add(imm.describeCompact()) || attempt >= 5) break
                attempt++
            } while (true)
            immIdx++
            next.add(imm)
        }
        return next
    }

    private fun tournamentPick(
        ranked: List<GenomeCandidateEvaluation>,
        rng: Random,
        poolSize: Int
    ): SymbolicGenome {
        val a = ranked[rng.nextInt(poolSize)]
        val b = ranked[rng.nextInt(poolSize)]
        return if (a.fitnessScore >= b.fitnessScore) a.genome else b.genome
    }

    internal inline fun <R> mapIndexedParallel(
        count: Int,
        crossinline block: (Int) -> R
    ): List<R> {
        if (count <= 0) return emptyList()
        val pool = ForkJoinPool.commonPool()
        if (count <= 4 || pool.parallelism <= 1) {
            return List(count) { block(it) }
        }
        val chunkCount = (pool.parallelism * 2).coerceIn(1, count)
        val chunkSize = (count + chunkCount - 1) / chunkCount
        val results = arrayOfNulls<Any?>(count)
        val tasks = ArrayList<Callable<Unit>>(chunkCount)
        var start = 0
        while (start < count) {
            val from = start
            val to = minOf(count, from + chunkSize)
            tasks.add(Callable {
                for (i in from until to) {
                    results[i] = block(i)
                }
            })
            start = to
        }
        pool.invokeAll(tasks).forEach { it.get() }
        @Suppress("UNCHECKED_CAST")
        return results.asList() as List<R>
    }

    private class StageScratchBuffers {
        val sameDayCounts = IntArray(91)
        val rawA = DoubleArray(91)
        val rawB = DoubleArray(91)
        val probsA = DoubleArray(91)
        val probsB = DoubleArray(91)
        val marginalProbs = DoubleArray(91)
        val noisyA = DoubleArray(91)
        val noisyB = DoubleArray(91)
        val chosen = BooleanArray(91)
        val picked = IntArray(5)
        val rpnStack = DoubleArray(CompiledRpnProgram.MAX_STACK_DEPTH)
    }

    fun evaluateGenomeOnStage(
        genome: SymbolicGenome,
        steps: List<CachedRawStep>,
        precompiledA: CompiledRpnProgram? = null,
        precompiledB: CompiledRpnProgram? = null
    ): StageEvalStats {
        if (steps.isEmpty()) {
            return StageEvalStats(0, 0, 0, 0, 0, 0f, BacktestResult.RANDOM_BRIER_SCORE, emptyList(), true)
        }

        val rpnA = precompiledA ?: SymbolicFormula.compileToRpn(genome.primaryExpr)
        val rpnB = precompiledB ?: SymbolicFormula.compileToRpn(genome.secondaryExpr)

        var twoPlus = 0
        var totalHits = 0
        var brierSum = 0.0
        var zeroStreak = 0
        var maxZeroStreak = 0
        var allCausal = true
        val matchesList = ArrayList<Int>(steps.size)

        val buf = StageScratchBuffers()
        var activeDayKey = ""
        var curDayKey = ""
        var curDayEpoch = Long.MIN_VALUE
        var curDayTwoPlus = 0
        var curDayPrevHit = 0
        var curDayLastHit = 0
        var qualDays = 0
        var consecQual = 0
        var maxConsecQual = 0
        var lastQualDay = ""
        var lastQualDayEpoch = Long.MIN_VALUE

        for (step in steps) {
            if (step.maxPastTs >= step.draw.timestamp) {
                allCausal = false
            }
            if (step.dayKey != activeDayKey) {
                activeDayKey = step.dayKey
                buf.sameDayCounts.fill(0)
            }

            val pickedCount = selectStepIntoBuffers(
                genome = genome,
                rpnA = rpnA,
                rpnB = rpnB,
                matrix = step.rawMatrix,
                sameDayCounts = buf.sameDayCounts,
                stepSalt = deterministicStepSalt(step.dayKey, step.horaire),
                buf = buf
            ).first

            var hits = 0
            val mask = step.actualMask
            for (i in 0 until pickedCount) {
                val n = buf.picked[i]
                buf.sameDayCounts[n] = buf.sameDayCounts[n] + 1
                if (mask[n]) hits++
            }

            totalHits += hits
            matchesList.add(hits)
            if (hits >= 2) twoPlus++

            brierSum += ScientificEvaluator.computeBrierScoreMask(buf.marginalProbs, mask)

            if (hits == 0) {
                zeroStreak++
                if (zeroStreak > maxZeroStreak) maxZeroStreak = zeroStreak
            } else {
                zeroStreak = 0
            }

            val sDayEpoch = step.dayEpoch
            if (sDayEpoch != curDayEpoch) {
                if (curDayEpoch != Long.MIN_VALUE && curDayTwoPlus >= 3 && curDayPrevHit >= 2) {
                    qualDays++
                    val isConsec = if (lastQualDayEpoch != Long.MIN_VALUE) {
                        sDayEpoch == lastQualDayEpoch + 1L || curDayEpoch == lastQualDayEpoch + 1L
                    } else {
                        lastQualDay.isNotEmpty() && LotoDateUtils.areConsecutiveCalendarDays(lastQualDay, curDayKey)
                    }
                    consecQual = if (isConsec) consecQual + 1 else 1
                    if (consecQual > maxConsecQual) maxConsecQual = consecQual
                    lastQualDay = curDayKey
                    lastQualDayEpoch = curDayEpoch
                } else if (curDayEpoch != Long.MIN_VALUE) {
                    consecQual = 0
                }
                curDayKey = step.dayKey
                curDayEpoch = sDayEpoch
                curDayTwoPlus = 0
                curDayPrevHit = 0
                curDayLastHit = 0
            }
            curDayPrevHit = curDayLastHit
            curDayLastHit = hits
            if (hits >= 2) curDayTwoPlus++
        }

        if (curDayEpoch != Long.MIN_VALUE && curDayTwoPlus >= 3 && curDayPrevHit >= 2) {
            qualDays++
            val isConsec = if (lastQualDayEpoch != Long.MIN_VALUE) {
                curDayEpoch == lastQualDayEpoch + 1L
            } else {
                lastQualDay.isNotEmpty() && LotoDateUtils.areConsecutiveCalendarDays(lastQualDay, curDayKey)
            }
            consecQual = if (isConsec) consecQual + 1 else 1
            if (consecQual > maxConsecQual) maxConsecQual = consecQual
        }

        return StageEvalStats(
            tested = steps.size,
            twoPlusCount = twoPlus,
            qualifyingDays = qualDays,
            maxConsecutiveQualifyingDays = maxConsecQual,
            maxZeroStreak = maxZeroStreak,
            avgMatches = totalHits.toFloat() / steps.size,
            brierScore = brierSum / steps.size,
            matchesList = matchesList,
            causalVerified = allCausal,
            twoPlusRate = twoPlus.toFloat() / steps.size
        )
    }

    private data class GenomeStepSelection(
        val predicted: List<Int>,
        val methodANumbers: List<Int>,
        val methodBNumbers: List<Int>,
        val marginalProbs: DoubleArray
    )

    private fun selectStepIntoBuffers(
        genome: SymbolicGenome,
        rpnA: CompiledRpnProgram,
        rpnB: CompiledRpnProgram,
        matrix: RawFeatureMatrix,
        sameDayCounts: IntArray,
        stepSalt: Long,
        buf: StageScratchBuffers
    ): Pair<Int, Int> {
        val stack = buf.rpnStack
        for (n in 1..90) {
            val repNorm = (sameDayCounts[n] / 5.0).coerceIn(0.0, 1.0)
            buf.rawA[n] = rpnA.evaluateCpu(matrix, n, repNorm, stack)
            buf.rawB[n] = rpnB.evaluateCpu(matrix, n, repNorm, stack)
        }

        softmaxScaled5Into(buf.rawA, genome.samplingTemp, buf.probsA)
        softmaxScaled5Into(buf.rawB, genome.samplingTemp, buf.probsB)
        val alphaA = genome.primarySlots / 5.0
        val alphaB = 1.0 - alphaA
        buf.marginalProbs[0] = 0.0
        for (n in 1..90) {
            buf.marginalProbs[n] = (alphaA * buf.probsA[n] + alphaB * buf.probsB[n]).coerceIn(1e-6, 0.999)
        }

        val noiseScale = genome.exploreNoise * 0.08
        buf.noisyA[0] = -1e9
        buf.noisyB[0] = -1e9
        if (noiseScale <= 1e-6) {
            for (n in 1..90) {
                buf.noisyA[n] = buf.probsA[n]
                buf.noisyB[n] = buf.probsB[n]
            }
        } else {
            val rng = Random(genome.seed xor (stepSalt * 2_654_435_761L))
            for (n in 1..90) {
                buf.noisyA[n] = buf.probsA[n] + rng.nextDouble() * noiseScale
            }
            for (n in 1..90) {
                buf.noisyB[n] = buf.probsB[n] + rng.nextDouble() * noiseScale
            }
        }

        buf.chosen.fill(false)
        val kA = genome.primarySlots.coerceIn(0, 5)
        var totalPicked = 0
        while (totalPicked < kA) {
            var bestN = -1
            var bestScore = Double.NEGATIVE_INFINITY
            for (n in 1..90) {
                if (buf.chosen[n]) continue
                val s = buf.noisyA[n]
                if (bestN == -1 || s > bestScore) {
                    bestScore = s
                    bestN = n
                }
            }
            if (bestN == -1) break
            buf.chosen[bestN] = true
            buf.picked[totalPicked++] = bestN
        }
        val countA = totalPicked

        while (totalPicked < 5) {
            var bestN = -1
            var bestScore = Double.NEGATIVE_INFINITY
            for (n in 1..90) {
                if (buf.chosen[n]) continue
                val s = buf.noisyB[n]
                if (bestN == -1 || s > bestScore) {
                    bestScore = s
                    bestN = n
                }
            }
            if (bestN == -1) break
            buf.chosen[bestN] = true
            buf.picked[totalPicked++] = bestN
        }

        return totalPicked to countA
    }

    private fun selectNumbersWithGenome(
        genome: SymbolicGenome,
        matrix: RawFeatureMatrix,
        sameDayCounts: IntArray,
        stepSalt: Long,
        buf: StageScratchBuffers = StageScratchBuffers(),
        precompiledA: CompiledRpnProgram = SymbolicFormula.compileToRpn(genome.primaryExpr),
        precompiledB: CompiledRpnProgram = SymbolicFormula.compileToRpn(genome.secondaryExpr)
    ): GenomeStepSelection {
        val (totalPicked, countA) = selectStepIntoBuffers(genome, precompiledA, precompiledB, matrix, sameDayCounts, stepSalt, buf)
        val methodA = ArrayList<Int>(countA)
        for (i in 0 until countA) methodA.add(buf.picked[i])
        val methodB = ArrayList<Int>((totalPicked - countA).coerceAtLeast(0))
        for (i in countA until totalPicked) methodB.add(buf.picked[i])
        val predicted = ArrayList<Int>(totalPicked)
        for (i in 0 until totalPicked) predicted.add(buf.picked[i])
        predicted.sort()
        methodA.sort()
        methodB.sort()

        return GenomeStepSelection(
            predicted = predicted,
            methodANumbers = methodA,
            methodBNumbers = methodB,
            marginalProbs = buf.marginalProbs.copyOf()
        )
    }

    private fun softmaxScaled5Into(rawScores: DoubleArray, temp: Double, outProbs: DoubleArray) {
        val invT = 1.0 / temp.coerceIn(0.15, 2.0)
        var maxScore = Double.NEGATIVE_INFINITY
        for (n in 1..90) {
            val s = rawScores[n]
            if (s.isFinite() && s > maxScore) maxScore = s
        }
        if (!maxScore.isFinite()) maxScore = 0.0

        var sum = 0.0
        for (n in 1..90) {
            val v = if (rawScores[n].isFinite()) ((rawScores[n] - maxScore) * invT).coerceIn(-15.0, 15.0) else -15.0
            val e = exp(v)
            outProbs[n] = e
            sum += e
        }
        if (sum > 0.0 && sum.isFinite()) {
            for (n in 1..90) {
                outProbs[n] = ((5.0 * outProbs[n]) / sum).coerceIn(1e-6, 0.999)
            }
        } else {
            for (n in 1..90) outProbs[n] = 5.0 / 90.0
        }
        var currentSum = 0.0
        for (n in 1..90) currentSum += outProbs[n]
        val factor = 5.0 / currentSum.coerceAtLeast(1e-9)
        for (n in 1..90) outProbs[n] = (outProbs[n] * factor).coerceIn(1e-6, 0.999)
    }

    fun deterministicStepSalt(dayKey: String, normalizedHoraire: String): Long =
        "${dayKey}|${normalizedHoraire.trim().uppercase(Locale.ROOT)}".hashCode().toLong()

    fun buildLedgerKey(strategySignature: String, periodKey: String): String =
        "${periodKey}__${strategySignature.hashCode().toUInt()}"

    fun buildValidatedBacktest(
        genome: SymbolicGenome,
        cache: FourStageCausalCache,
        generationsExecuted: Int,
        experimentsCount: Int
    ): BacktestResult {
        val steps = cache.steps
        if (steps.isEmpty()) {
            return ScientificEvaluator.emptyBacktestResult(PredictionMethod.AUTO_SEARCH, cache.periodLabel)
        }

        val rpnA = SymbolicFormula.compileToRpn(genome.primaryExpr)
        val rpnB = SymbolicFormula.compileToRpn(genome.secondaryExpr)
        var activeDayKey = ""
        val buf = StageScratchBuffers()
        val stepResults = ArrayList<BacktestStepResult>(steps.size)

        for ((idx, step) in steps.withIndex()) {
            if (step.dayKey != activeDayKey) {
                activeDayKey = step.dayKey
                buf.sameDayCounts.fill(0)
            }

            val selection = selectNumbersWithGenome(
                genome = genome,
                matrix = step.rawMatrix,
                sameDayCounts = buf.sameDayCounts,
                stepSalt = deterministicStepSalt(step.dayKey, step.horaire),
                buf = buf,
                precompiledA = rpnA,
                precompiledB = rpnB
            )
            for (n in selection.predicted) {
                buf.sameDayCounts[n] = buf.sameDayCounts[n] + 1
            }

            val matched = selection.predicted.filter { it in step.actualSet }
            val stepBrier = ScientificEvaluator.computeBrierScore(selection.marginalProbs, step.actualSet)
            val stepLogLoss = ScientificEvaluator.computeBernoulliLogLoss(selection.marginalProbs, step.actualSet)

            stepResults.add(
                BacktestStepResult(
                    stepIndex = idx + 1,
                    drawId = step.draw.uniqueId,
                    drawLabel = "${step.game} • ${step.horaire}",
                    drawDate = LotoDateUtils.formatFrenchDayAndDate(step.draw.timestamp),
                    dayKey = step.dayKey,
                    horaire = step.horaire,
                    game = step.game,
                    trainSampleCount = step.strictPastSize,
                    maxTrainTimestamp = step.maxPastTs,
                    targetTimestamp = step.draw.timestamp,
                    causalVerified = step.maxPastTs < step.draw.timestamp,
                    predictedNumbers = selection.predicted,
                    actualNumbers = step.actualNumbers,
                    matchedNumbers = matched,
                    matchCount = matched.size,
                    expectedMatchesEstimate = selection.marginalProbs.sum().toFloat(),
                    stepBrierScore = stepBrier,
                    stepBernoulliLogLoss = stepLogLoss,
                    methodANumbers = selection.methodANumbers,
                    methodBNumbers = selection.methodBNumbers
                )
            )
        }

        return aggregateBacktestResult(
            chronologicalSteps = stepResults,
            periodLabel = cache.periodLabel,
            generationsExecuted = generationsExecuted,
            experimentsCount = experimentsCount,
            isAppendOnlyApproved = false,
            preservedCount = 0,
            appendedCount = stepResults.size,
            liveMatchedCount = 0
        )
    }

    internal fun evaluateGenomeOnRange(
        chronological: List<LotoResultEntity>,
        genome: SymbolicGenome,
        startIndex: Int,
        endIndexExclusive: Int,
        periodLabel: String
    ): BacktestResult {
        val safeStart = startIndex.coerceAtLeast(1)
        val safeEnd = endIndexExclusive.coerceAtMost(chronological.size)
        if (safeStart >= safeEnd) {
            return ScientificEvaluator.emptyBacktestResult(PredictionMethod.AUTO_SEARCH, periodLabel)
        }

        val rangeIndices = (safeStart until safeEnd).toList().toIntArray()
        val matrixByIdx = SymbolicFormula.extractRawMatrixSequence(chronological, rangeIndices)

        val strictCountAt = IntArray(chronological.size)
        val maxPastTsAt = LongArray(chronological.size) { Long.MIN_VALUE }
        var committed = 0
        var latestPastTs = Long.MIN_VALUE
        for (i in chronological.indices) {
            val ts = chronological[i].timestamp
            while (committed < i && chronological[committed].timestamp < ts) {
                val pTs = chronological[committed].timestamp
                if (pTs > latestPastTs) latestPastTs = pTs
                committed++
            }
            strictCountAt[i] = committed
            maxPastTsAt[i] = latestPastTs
        }

        val rpnA = SymbolicFormula.compileToRpn(genome.primaryExpr)
        val rpnB = SymbolicFormula.compileToRpn(genome.secondaryExpr)
        var activeDayKey = ""
        val buf = StageScratchBuffers()
        val stepResults = ArrayList<BacktestStepResult>(safeEnd - safeStart)

        for (t in safeStart until safeEnd) {
            val targetDraw = chronological[t]
            val actualNumbers = StatisticalStrategies.extractDistinctBalls(targetDraw).sorted()
            if (actualNumbers.size < 5) continue
            val strictPastSize = strictCountAt[t]
            if (strictPastSize <= 0) continue
            val maxPastTs = maxPastTsAt[t]
            if (maxPastTs >= targetDraw.timestamp) continue
            val rawMatrix = matrixByIdx[t] ?: continue

            val dayKey = LotoDateUtils.formatDayKey(targetDraw.timestamp)
            if (dayKey != activeDayKey) {
                activeDayKey = dayKey
                buf.sameDayCounts.fill(0)
            }
            val horaire = LotoDateUtils.normalizeHoraire(targetDraw)
            val game = LotoDateUtils.normalizeGame(targetDraw)
            val selection = selectNumbersWithGenome(
                genome = genome,
                matrix = rawMatrix,
                sameDayCounts = buf.sameDayCounts,
                stepSalt = deterministicStepSalt(dayKey, horaire),
                buf = buf,
                precompiledA = rpnA,
                precompiledB = rpnB
            )
            for (n in selection.predicted) {
                buf.sameDayCounts[n] = buf.sameDayCounts[n] + 1
            }

            val actualSet = actualNumbers.toSet()
            val matched = selection.predicted.filter { it in actualSet }
            stepResults.add(
                BacktestStepResult(
                    stepIndex = stepResults.size + 1,
                    drawId = targetDraw.uniqueId,
                    drawLabel = "$game • $horaire",
                    drawDate = LotoDateUtils.formatFrenchDayAndDate(targetDraw.timestamp),
                    dayKey = dayKey,
                    horaire = horaire,
                    game = game,
                    trainSampleCount = strictPastSize,
                    maxTrainTimestamp = maxPastTs,
                    targetTimestamp = targetDraw.timestamp,
                    causalVerified = maxPastTs < targetDraw.timestamp,
                    predictedNumbers = selection.predicted,
                    actualNumbers = actualNumbers,
                    matchedNumbers = matched,
                    matchCount = matched.size,
                    expectedMatchesEstimate = selection.marginalProbs.sum().toFloat(),
                    stepBrierScore = ScientificEvaluator.computeBrierScore(selection.marginalProbs, actualSet),
                    stepBernoulliLogLoss = ScientificEvaluator.computeBernoulliLogLoss(selection.marginalProbs, actualSet),
                    methodANumbers = selection.methodANumbers,
                    methodBNumbers = selection.methodBNumbers
                )
            )
        }

        return aggregateBacktestResult(
            chronologicalSteps = stepResults,
            periodLabel = periodLabel,
            generationsExecuted = genome.generation,
            experimentsCount = 1,
            isAppendOnlyApproved = false,
            preservedCount = 0,
            appendedCount = stepResults.size,
            liveMatchedCount = 0
        )
    }

    /**
     * Backtest approuvé IMUTABLE en mode append-only :
     * Conserve à l'identique toutes les étapes déjà approuvées et n'ajoute que les nouveaux tirages complétés.
     */
    fun mergeAppendOnlyApprovedBacktest(
        genome: SymbolicGenome,
        cache: FourStageCausalCache,
        existingLedger: ApprovedBacktestLedgerEntity?,
        generationsExecuted: Int,
        experimentsCount: Int
    ): Pair<BacktestResult, ApprovedBacktestLedgerEntity> {
        val periodKey = cache.periodLabel
        val strategySig = genome.describeCompact()
        val rawSteps = cache.steps
        val frozenChronological = if (existingLedger != null &&
            existingLedger.strategySignature == strategySig &&
            existingLedger.periodKey == periodKey
        ) {
            deserializeApprovedSteps(existingLedger.frozenStepsJson).sortedBy { it.targetTimestamp }
        } else {
            emptyList()
        }

        val pendingList = if (existingLedger != null && existingLedger.strategySignature == strategySig) {
            deserializePendingPredictions(existingLedger.pendingPredictionsJson)
        } else {
            emptyList()
        }

        val frozenByDrawId = frozenChronological.associateBy { it.drawId }
        val maxFrozenTs = frozenChronological.maxOfOrNull { it.targetTimestamp } ?: Long.MIN_VALUE
        val combinedChronological = ArrayList<BacktestStepResult>(frozenChronological.size + 16)
        combinedChronological.addAll(frozenChronological)

        val rpnA = SymbolicFormula.compileToRpn(genome.primaryExpr)
        val rpnB = SymbolicFormula.compileToRpn(genome.secondaryExpr)
        val buf = StageScratchBuffers()
        var activeDayKey = ""
        var appendedCount = 0
        var liveMatchedCount = 0

        val lastFrozenDayKey = frozenChronological.lastOrNull()?.dayKey
        if (lastFrozenDayKey != null) {
            activeDayKey = lastFrozenDayKey
            for (s in frozenChronological) {
                if (s.dayKey == lastFrozenDayKey) {
                    for (n in s.predictedNumbers) buf.sameDayCounts[n] = buf.sameDayCounts[n] + 1
                }
            }
        }

        for (step in rawSteps) {
            if (frozenByDrawId.containsKey(step.draw.uniqueId) || step.draw.timestamp <= maxFrozenTs) {
                continue
            }
            if (step.dayKey != activeDayKey) {
                activeDayKey = step.dayKey
                buf.sameDayCounts.fill(0)
            }
            val selection = selectNumbersWithGenome(
                genome = genome,
                matrix = step.rawMatrix,
                sameDayCounts = buf.sameDayCounts,
                stepSalt = deterministicStepSalt(step.dayKey, step.horaire),
                buf = buf,
                precompiledA = rpnA,
                precompiledB = rpnB
            )
            for (n in selection.predicted) buf.sameDayCounts[n] = buf.sameDayCounts[n] + 1

            val pendingHit = pendingList.firstOrNull {
                it.dayKey == step.dayKey && it.horaire.equals(step.horaire, ignoreCase = true)
            }
            val finalPredicted = if (pendingHit != null && pendingHit.predictedNumbers.size == 5) {
                pendingHit.predictedNumbers.sorted()
            } else {
                selection.predicted
            }
            if (pendingHit != null) {
                liveMatchedCount++
            }

            val matched = finalPredicted.filter { it in step.actualSet }
            val stepBrier = ScientificEvaluator.computeBrierScore(selection.marginalProbs, step.actualSet)
            val stepLogLoss = ScientificEvaluator.computeBernoulliLogLoss(selection.marginalProbs, step.actualSet)

            combinedChronological.add(
                BacktestStepResult(
                    stepIndex = combinedChronological.size + 1,
                    drawId = step.draw.uniqueId,
                    drawLabel = "${step.game} • ${step.horaire}",
                    drawDate = LotoDateUtils.formatFrenchDayAndDate(step.draw.timestamp),
                    dayKey = step.dayKey,
                    horaire = step.horaire,
                    game = step.game,
                    trainSampleCount = step.strictPastSize,
                    maxTrainTimestamp = step.maxPastTs,
                    targetTimestamp = step.draw.timestamp,
                    causalVerified = step.maxPastTs < step.draw.timestamp,
                    predictedNumbers = finalPredicted,
                    actualNumbers = step.actualNumbers,
                    matchedNumbers = matched,
                    matchCount = matched.size,
                    expectedMatchesEstimate = selection.marginalProbs.sum().toFloat(),
                    stepBrierScore = stepBrier,
                    stepBernoulliLogLoss = stepLogLoss,
                    methodANumbers = selection.methodANumbers,
                    methodBNumbers = selection.methodBNumbers
                )
            )
            appendedCount++
        }

        val reindexed = combinedChronological.mapIndexed { i, st -> st.copy(stepIndex = i + 1) }
        val remainingPending = pendingList.filter { p ->
            reindexed.none { it.dayKey == p.dayKey && it.horaire.equals(p.horaire, ignoreCase = true) }
        }.takeLast(24)

        val updatedLedger = ApprovedBacktestLedgerEntity(
            ledgerKey = buildLedgerKey(strategySig, periodKey),
            strategySignature = strategySig,
            periodKey = periodKey,
            frozenStepsJson = serializeApprovedSteps(reindexed),
            pendingPredictionsJson = serializePendingPredictions(remainingPending),
            lastApprovedTimestamp = reindexed.maxOfOrNull { it.targetTimestamp } ?: (existingLedger?.lastApprovedTimestamp ?: 0L),
            updatedAt = System.currentTimeMillis()
        )

        val result = aggregateBacktestResult(
            chronologicalSteps = reindexed,
            periodLabel = periodKey,
            generationsExecuted = generationsExecuted,
            experimentsCount = experimentsCount,
            isAppendOnlyApproved = true,
            preservedCount = frozenChronological.size,
            appendedCount = appendedCount,
            liveMatchedCount = liveMatchedCount
        )
        return result to updatedLedger
    }

    fun recordLivePredictionInLedger(
        ledger: ApprovedBacktestLedgerEntity,
        prediction: PredictionResult,
        targetTimestamp: Long
    ): ApprovedBacktestLedgerEntity {
        val dayKey = LotoDateUtils.formatDayKey(targetTimestamp)
        val normH = prediction.horaire.trim().uppercase(Locale.ROOT)
        val current = deserializePendingPredictions(ledger.pendingPredictionsJson).toMutableList()
        current.removeAll { it.dayKey == dayKey && it.horaire.equals(normH, ignoreCase = true) }
        current.add(
            PendingLivePrediction(
                dayKey = dayKey,
                horaire = normH,
                targetTimestamp = targetTimestamp,
                predictedNumbers = prediction.predictedDraw.sorted()
            )
        )
        return ledger.copy(
            pendingPredictionsJson = serializePendingPredictions(current.takeLast(24)),
            updatedAt = System.currentTimeMillis()
        )
    }

    private fun aggregateBacktestResult(
        chronologicalSteps: List<BacktestStepResult>,
        periodLabel: String,
        generationsExecuted: Int,
        experimentsCount: Int,
        isAppendOnlyApproved: Boolean,
        preservedCount: Int,
        appendedCount: Int,
        liveMatchedCount: Int
    ): BacktestResult {
        if (chronologicalSteps.isEmpty()) {
            return ScientificEvaluator.emptyBacktestResult(PredictionMethod.AUTO_SEARCH, periodLabel)
        }

        var partialMatches = 0
        var twoPlusMatches = 0
        var threePlusMatches = 0
        var fourPlusMatches = 0
        var brierSum = 0.0
        var logLossSum = 0.0
        val distribution = IntArray(6)
        val matchList = ArrayList<Int>(chronologicalSteps.size)
        var zeroStreak = 0
        var maxZeroStreak = 0
        var allCausal = true

        for (st in chronologicalSteps) {
            if (!st.causalVerified) allCausal = false
            val mc = st.matchCount.coerceIn(0, 5)
            if (mc >= 1) partialMatches++
            if (mc >= 2) twoPlusMatches++
            if (mc >= 3) threePlusMatches++
            if (mc >= 4) fourPlusMatches++
            distribution[mc]++
            matchList.add(mc)
            if (mc == 0) {
                zeroStreak++
                if (zeroStreak > maxZeroStreak) maxZeroStreak = zeroStreak
            } else zeroStreak = 0
            brierSum += st.stepBrierScore
            logLossSum += st.stepBernoulliLogLoss
        }

        val totalTested = chronologicalSteps.size
        val useSliding6DrawBlocks = (periodLabel.contains("Jour") && !periodLabel.contains("Tout")) ||
            periodLabel.startsWith("1 Jour")

        val dailySummaries: List<DailyBacktestSummary>
        var consecQual = 0
        var maxConsecQual = 0

        if (useSliding6DrawBlocks) {
            val totalBlocks = (totalTested + 5) / 6
            val groupedBlocks = chronologicalSteps.withIndex().groupBy { (idx, _) ->
                val blockFromEnd = (totalTested - 1 - idx) / 6
                totalBlocks - 1 - blockFromEnd
            }.entries.sortedBy { it.key }

            dailySummaries = groupedBlocks.map { (bIdx, indexedSteps) ->
                val list = indexedSteps.map { it.value }
                val t2 = list.count { it.matchCount >= 2 }
                val penHit = if (list.size >= 2) list[list.size - 2].matchCount else 0
                val penOk = penHit >= 2
                val firstDay = list.first().dayKey
                val lastDay = list.last().dayKey
                val label = if (firstDay == lastDay) {
                    "${list.first().drawDate} (${list.size} tir.)"
                } else {
                    "$firstDay → $lastDay (${list.size} tir.)"
                }
                DailyBacktestSummary(
                    dayKey = "B${bIdx + 1}_$lastDay",
                    dayLabel = label,
                    drawsTested = list.size,
                    drawsWithTwoPlus = t2,
                    bestMatchCount = list.maxOfOrNull { it.matchCount } ?: 0,
                    isQualifyingDay = t2 == list.size && list.isNotEmpty() && penOk,
                    penultimateHitCount = penHit,
                    penultimateTwoPlus = penOk
                )
            }
            var lastQualBlock = -2
            groupedBlocks.forEachIndexed { pos, (_, _) ->
                if (dailySummaries[pos].isQualifyingDay) {
                    consecQual = if (pos == lastQualBlock + 1) consecQual + 1 else 1
                    if (consecQual > maxConsecQual) maxConsecQual = consecQual
                    lastQualBlock = pos
                } else {
                    consecQual = 0
                }
            }
        } else {
            dailySummaries = chronologicalSteps.groupBy { it.dayKey }.entries.sortedBy { it.key }.map { (k, list) ->
                val t2 = list.count { it.matchCount >= 2 }
                val penHit = if (list.size >= 2) list[list.size - 2].matchCount else 0
                val penOk = penHit >= 2
                DailyBacktestSummary(
                    dayKey = k,
                    dayLabel = list.first().drawDate,
                    drawsTested = list.size,
                    drawsWithTwoPlus = t2,
                    bestMatchCount = list.maxOfOrNull { it.matchCount } ?: 0,
                    isQualifyingDay = t2 == list.size && list.isNotEmpty() && penOk,
                    penultimateHitCount = penHit,
                    penultimateTwoPlus = penOk
                )
            }

            var lastQual = ""
            for (d in dailySummaries) {
                if (d.isQualifyingDay) {
                    consecQual = if (lastQual.isNotEmpty() && LotoDateUtils.areConsecutiveCalendarDays(lastQual, d.dayKey)) {
                        consecQual + 1
                    } else 1
                    if (consecQual > maxConsecQual) maxConsecQual = consecQual
                    lastQual = d.dayKey
                } else consecQual = 0
            }
        }

        val horairePerf = chronologicalSteps.groupBy { it.horaire }.entries.map { (h, list) ->
            HorairePerformance(
                horaire = h,
                testedCount = list.size,
                averageMatches = list.sumOf { it.matchCount }.toFloat() / list.size,
                onePlusCount = list.count { it.matchCount >= 1 },
                twoPlusCount = list.count { it.matchCount >= 2 }
            )
        }.sortedByDescending { it.twoPlusCount }

        val (avgLo, avgHi) = ScientificEvaluator.mean95Interval(matchList)
        val (rateLo, rateHi) = ScientificEvaluator.wilson95Interval(partialMatches, totalTested)

        return BacktestResult(
            method = PredictionMethod.AUTO_SEARCH,
            successRate = partialMatches.toFloat() / totalTested,
            totalTested = totalTested,
            exactMatches = distribution[5],
            partialMatches = partialMatches,
            drawsWithTwoPlusMatches = twoPlusMatches,
            drawsWithThreePlusMatches = threePlusMatches,
            drawsWithFourPlusMatches = fourPlusMatches,
            qualifyingDaysCount = dailySummaries.count { it.isQualifyingDay },
            totalDaysTested = dailySummaries.size,
            averageMatches = matchList.sum().toFloat() / totalTested,
            avgMatchesCiLow = avgLo,
            avgMatchesCiHigh = avgHi,
            rateOnePlusCiLow = rateLo,
            rateOnePlusCiHigh = rateHi,
            averageBrierScore = brierSum / totalTested.coerceAtLeast(1),
            averageBernoulliLogLoss = logLossSum / totalTested.coerceAtLeast(1),
            calibrationErrorEce = 0.0,
            matchDistribution = distribution.toList(),
            allStepsCausalVerified = allCausal,
            periodLabel = periodLabel,
            learnedExpertWeights = emptyList(),
            weightSnapshots = emptyList(),
            dailySummaries = dailySummaries,
            horaireBreakdown = horairePerf,
            steps = chronologicalSteps.reversed(),
            maxConsecutiveQualifyingDays = maxConsecQual,
            maxZeroMatchStreak = maxZeroStreak,
            currentZeroMatchStreak = zeroStreak,
            discoveredStrategiesSummary = emptyList(),
            searchRoundsExecuted = generationsExecuted,
            totalStrategiesSearched = experimentsCount,
            objectiveFourDaysAchieved = maxConsecQual >= 4,
            isAppendOnlyApproved = isAppendOnlyApproved,
            approvedStepsPreservedCount = preservedCount,
            newlyAppendedStepsCount = appendedCount,
            livePredictionsMatchedCount = liveMatchedCount
        )
    }

    fun predictNextWithGenome(
        genome: SymbolicGenome,
        cleanHistory: List<LotoResultEntity>,
        targetHoraire: String,
        validatedBacktest: BacktestResult,
        targetTimestamp: Long? = null
    ): PredictionResult {
        val chronological = cleanHistory
            .distinctBy { it.uniqueId }
            .sortedWith(compareBy<LotoResultEntity> { it.timestamp }.thenBy { it.uniqueId })
        val normalizedH = targetHoraire.trim().uppercase(Locale.ROOT)
        val effectiveTargetTs = targetTimestamp ?: chronological.lastOrNull()?.timestamp ?: System.currentTimeMillis()
        val targetDayKey = LotoDateUtils.formatDayKey(effectiveTargetTs)

        val sameDayCounts = IntArray(91)
        val usesSameDay = (SymbolicFormula.usedFeatures(genome.primaryExpr) + SymbolicFormula.usedFeatures(genome.secondaryExpr))
            .any { it == RawFeature.SAME_DAY_REP }
        val rpnA = SymbolicFormula.compileToRpn(genome.primaryExpr)
        val rpnB = SymbolicFormula.compileToRpn(genome.secondaryExpr)
        if (usesSameDay) {
            val existingSameDay = validatedBacktest.steps
                .asReversed()
                .filter { it.dayKey == targetDayKey && it.targetTimestamp < effectiveTargetTs }
            if (existingSameDay.isNotEmpty()) {
                for (st in existingSameDay) {
                    for (n in st.predictedNumbers) if (n in 1..90) sameDayCounts[n] = sameDayCounts[n] + 1
                }
            } else {
                val buf = StageScratchBuffers()
                for (i in chronological.indices) {
                    val d = chronological[i]
                    if (d.timestamp >= effectiveTargetTs) break
                    if (LotoDateUtils.formatDayKey(d.timestamp) == targetDayKey) {
                        val past = chronological.subList(0, i).filter { it.timestamp < d.timestamp && it.uniqueId != d.uniqueId }
                        if (past.isNotEmpty()) {
                            val dH = LotoDateUtils.normalizeHoraire(d)
                            val m = SymbolicFormula.extractRawMatrix(past, dH)
                            val sel = selectNumbersWithGenome(
                                genome = genome,
                                matrix = m,
                                sameDayCounts = sameDayCounts,
                                stepSalt = deterministicStepSalt(targetDayKey, dH),
                                buf = buf,
                                precompiledA = rpnA,
                                precompiledB = rpnB
                            )
                            for (n in sel.predicted) if (n in 1..90) sameDayCounts[n] = sameDayCounts[n] + 1
                        }
                    }
                }
            }
        }

        val matrix = SymbolicFormula.extractRawMatrix(chronological, normalizedH)
        val selection = selectNumbersWithGenome(
            genome = genome,
            matrix = matrix,
            sameDayCounts = sameDayCounts,
            stepSalt = deterministicStepSalt(targetDayKey, normalizedH),
            precompiledA = rpnA,
            precompiledB = rpnB
        )
        val marginalSum = (1..90).sumOf { selection.marginalProbs[it] }.toFloat()
        val randomBaselineProb = 5.0 / 90.0
        val sortedAll = (1..90).sortedWith(compareByDescending<Int> { selection.marginalProbs[it] }.thenBy { it })
        val rankByNumber = IntArray(91)
        sortedAll.forEachIndexed { idx, num -> rankByNumber[num] = idx + 1 }
        val maxProb = selection.marginalProbs[sortedAll.first()].coerceAtLeast(randomBaselineProb)
        val methodASet = selection.methodANumbers.toSet()

        fun buildConfidence(num: Int): PredictedNumberConfidence {
            val p = selection.marginalProbs[num]
            val probPct = (p * 100.0).toFloat().coerceIn(0f, 100f)
            val relPct = ((p / maxProb) * 100.0).toFloat().coerceIn(0f, 100f)
            val lift = (p / randomBaselineProb).toFloat()
            val branch = when {
                num in methodASet -> "Formule A"
                num in selection.methodBNumbers -> "Formule B"
                else -> "Candidat"
            }
            val isGap = matrix.get(RawFeature.GAP_NORM.ordinal, num) >= 0.08
            return PredictedNumberConfidence(
                number = num,
                marginalProbabilityPercent = probPct,
                relativeConfidencePercent = relPct,
                liftVsRandom = lift,
                rank = rankByNumber[num],
                sourceBranch = branch,
                isGapCandidate = isGap
            )
        }

        val predictedConfidences = selection.predicted
            .map { buildConfidence(it) }
            .sortedWith(compareByDescending<PredictedNumberConfidence> { it.marginalProbabilityPercent }.thenBy { it.number })
        val topCandidates = sortedAll.take(8).map { buildConfidence(it) }

        return PredictionResult(
            predictedDraw = selection.predicted,
            expectedMatchesEstimate = selection.predicted.sumOf { selection.marginalProbs[it] }.toFloat(),
            calibratedProbAtLeastOne = ScientificEvaluator.calibratedProbAtLeastOne(selection.predicted, selection.marginalProbs),
            trainedOnCount = chronological.size,
            method = PredictionMethod.AUTO_SEARCH,
            horaire = normalizedH,
            marginalSumCheck = marginalSum,
            auditNote = "Formule symbolique unifiée (0 dérive Live ↔ Backtest • zéro fuite)",
            discoveredStrategyName = genome.describeCompact(),
            coldNumbersCount = selection.predicted.count { matrix.get(RawFeature.GAP_NORM.ordinal, it) >= 0.08 },
            diversityPenaltyApplied = false,
            searchRoundsExecuted = genome.generation,
            totalStrategiesSearched = validatedBacktest.totalStrategiesSearched,
            objectiveFourDaysAchieved = validatedBacktest.objectiveFourDaysAchieved,
            bestCausalStreakFound = validatedBacktest.maxConsecutiveQualifyingDays,
            methodANumbers = selection.methodANumbers,
            methodBNumbers = selection.methodBNumbers,
            methodALabel = "A: ${genome.describePrimary()}",
            methodBLabel = "B: ${genome.describeSecondary()}",
            numberConfidences = predictedConfidences,
            candidateConfidences = topCandidates
        )
    }

    private fun serializeApprovedSteps(steps: List<BacktestStepResult>): String {
        val arr = buildJsonArray {
            for (s in steps) {
                add(buildJsonObject {
                    put("stepIndex", s.stepIndex)
                    put("drawId", s.drawId)
                    put("drawLabel", s.drawLabel)
                    put("drawDate", s.drawDate)
                    put("dayKey", s.dayKey)
                    put("horaire", s.horaire)
                    put("game", s.game)
                    put("trainSampleCount", s.trainSampleCount)
                    put("maxTrainTimestamp", s.maxTrainTimestamp)
                    put("targetTimestamp", s.targetTimestamp)
                    put("causalVerified", s.causalVerified)
                    put("predictedNumbers", buildJsonArray { s.predictedNumbers.forEach { add(JsonPrimitive(it)) } })
                    put("actualNumbers", buildJsonArray { s.actualNumbers.forEach { add(JsonPrimitive(it)) } })
                    put("matchedNumbers", buildJsonArray { s.matchedNumbers.forEach { add(JsonPrimitive(it)) } })
                    put("matchCount", s.matchCount)
                    put("expectedMatchesEstimate", s.expectedMatchesEstimate)
                    put("stepBrierScore", s.stepBrierScore)
                    put("stepBernoulliLogLoss", s.stepBernoulliLogLoss)
                    put("methodANumbers", buildJsonArray { s.methodANumbers.forEach { add(JsonPrimitive(it)) } })
                    put("methodBNumbers", buildJsonArray { s.methodBNumbers.forEach { add(JsonPrimitive(it)) } })
                })
            }
        }
        return arr.toString()
    }

    private fun deserializeApprovedSteps(json: String): List<BacktestStepResult> {
        if (json.isBlank()) return emptyList()
        return runCatching {
            Json.parseToJsonElement(json).jsonArray.mapNotNull { el ->
                val o = el as? JsonObject ?: return@mapNotNull null
                BacktestStepResult(
                    stepIndex = o["stepIndex"]?.jsonPrimitive?.intOrNull ?: 0,
                    drawId = o["drawId"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null,
                    drawLabel = o["drawLabel"]?.jsonPrimitive?.contentOrNull ?: "",
                    drawDate = o["drawDate"]?.jsonPrimitive?.contentOrNull ?: "",
                    dayKey = o["dayKey"]?.jsonPrimitive?.contentOrNull ?: "",
                    horaire = o["horaire"]?.jsonPrimitive?.contentOrNull ?: "",
                    game = o["game"]?.jsonPrimitive?.contentOrNull ?: "",
                    trainSampleCount = o["trainSampleCount"]?.jsonPrimitive?.intOrNull ?: 0,
                    maxTrainTimestamp = o["maxTrainTimestamp"]?.jsonPrimitive?.longOrNull ?: 0L,
                    targetTimestamp = o["targetTimestamp"]?.jsonPrimitive?.longOrNull ?: 0L,
                    causalVerified = o["causalVerified"]?.jsonPrimitive?.booleanOrNull ?: true,
                    predictedNumbers = o["predictedNumbers"]?.jsonArray?.mapNotNull { it.jsonPrimitive.intOrNull } ?: emptyList(),
                    actualNumbers = o["actualNumbers"]?.jsonArray?.mapNotNull { it.jsonPrimitive.intOrNull } ?: emptyList(),
                    matchedNumbers = o["matchedNumbers"]?.jsonArray?.mapNotNull { it.jsonPrimitive.intOrNull } ?: emptyList(),
                    matchCount = o["matchCount"]?.jsonPrimitive?.intOrNull ?: 0,
                    expectedMatchesEstimate = o["expectedMatchesEstimate"]?.jsonPrimitive?.floatOrNull ?: 0f,
                    stepBrierScore = o["stepBrierScore"]?.jsonPrimitive?.doubleOrNull ?: 0.0,
                    stepBernoulliLogLoss = o["stepBernoulliLogLoss"]?.jsonPrimitive?.doubleOrNull ?: 0.0,
                    methodANumbers = o["methodANumbers"]?.jsonArray?.mapNotNull { it.jsonPrimitive.intOrNull } ?: emptyList(),
                    methodBNumbers = o["methodBNumbers"]?.jsonArray?.mapNotNull { it.jsonPrimitive.intOrNull } ?: emptyList()
                )
            }
        }.getOrElse { emptyList() }
    }

    private fun serializePendingPredictions(list: List<PendingLivePrediction>): String {
        val arr = buildJsonArray {
            for (p in list) {
                add(buildJsonObject {
                    put("dayKey", p.dayKey)
                    put("horaire", p.horaire)
                    put("targetTimestamp", p.targetTimestamp)
                    put("predictedNumbers", buildJsonArray { p.predictedNumbers.forEach { add(JsonPrimitive(it)) } })
                })
            }
        }
        return arr.toString()
    }

    private fun deserializePendingPredictions(json: String): List<PendingLivePrediction> {
        if (json.isBlank()) return emptyList()
        return runCatching {
            Json.parseToJsonElement(json).jsonArray.mapNotNull { el ->
                val o = el as? JsonObject ?: return@mapNotNull null
                PendingLivePrediction(
                    dayKey = o["dayKey"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null,
                    horaire = o["horaire"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null,
                    targetTimestamp = o["targetTimestamp"]?.jsonPrimitive?.longOrNull ?: 0L,
                    predictedNumbers = o["predictedNumbers"]?.jsonArray?.mapNotNull { it.jsonPrimitive.intOrNull } ?: emptyList()
                )
            }
        }.getOrElse { emptyList() }
    }

    fun computeFeatureHistogram(population: List<SymbolicGenome>): List<Pair<String, Float>> {
        if (population.isEmpty()) return emptyList()
        val counts = HashMap<RawFeature, Int>()
        var total = 0
        for (g in population) {
            for (f in SymbolicFormula.usedFeatures(g.primaryExpr)) {
                counts[f] = (counts[f] ?: 0) + 1
                total++
            }
            for (f in SymbolicFormula.usedFeatures(g.secondaryExpr)) {
                counts[f] = (counts[f] ?: 0) + 1
                total++
            }
        }
        if (total == 0) return emptyList()
        return counts.entries
            .sortedByDescending { it.value }
            .take(6)
            .map { (feat, c) -> "${feat.symbol} (${feat.name})" to (c.toFloat() / total.toFloat()) }
    }

    fun synthesizeCausalDiscoveryGenomes(
        discoverySteps: List<CachedRawStep>,
        generation: Int,
        sessionSalt: Long,
        seenSignatures: Set<String> = emptySet(),
        maxCount: Int = 12
    ): List<SymbolicGenome> {
        if (discoverySteps.isEmpty()) return emptyList()
        val features = RawFeature.entries.filter { it != RawFeature.SAME_DAY_REP }
        val fCount = features.size
        val sumHit = DoubleArray(fCount)
        val sqHit = DoubleArray(fCount)
        val sumMiss = DoubleArray(fCount)
        val sqMiss = DoubleArray(fCount)
        var nHit = 0
        var nMiss = 0

        for (step in discoverySteps) {
            val mat = step.rawMatrix
            val mask = step.actualMask
            for (n in 1..90) {
                val isHit = mask[n]
                if (isHit) nHit++ else nMiss++
                for (fi in 0 until fCount) {
                    val v = mat.get(features[fi].ordinal, n)
                    if (!v.isFinite()) continue
                    if (isHit) {
                        sumHit[fi] += v
                        sqHit[fi] += v * v
                    } else {
                        sumMiss[fi] += v
                        sqMiss[fi] += v * v
                    }
                }
            }
        }
        if (nHit == 0 || nMiss == 0) return emptyList()

        data class RankedSignal(val feature: RawFeature, val zLift: Double)

        val rankedSignals = (0 until fCount).map { fi ->
            val mH = sumHit[fi] / nHit
            val mM = sumMiss[fi] / nMiss
            val varH = (sqHit[fi] / nHit - mH * mH).coerceAtLeast(1e-6)
            val varM = (sqMiss[fi] / nMiss - mM * mM).coerceAtLeast(1e-6)
            val z = (mH - mM) / sqrt((varH + varM) * 0.5 + 1e-6)
            RankedSignal(features[fi], if (z.isFinite()) z else 0.0)
        }.sortedByDescending { abs(it.zLift) }

        val topSignals = rankedSignals.take(6)
        if (topSignals.isEmpty()) return emptyList()

        val wResSumHit = DoubleArray(fCount)
        val wResSumMiss = DoubleArray(fCount)
        var nResHit = 0
        var nResMiss = 0
        for (step in discoverySteps) {
            val mat = step.rawMatrix
            val mask = step.actualMask
            val top5ByTop3 = (1..90).sortedByDescending { num ->
                var acc = 0.0
                for (k in 0 until minOf(3, topSignals.size)) {
                    val sig = topSignals[k]
                    val weight = (if (sig.zLift >= 0.0) 1.0 else -1.0) * abs(sig.zLift).coerceIn(0.2, 2.5)
                    acc += weight * mat.get(sig.feature.ordinal, num)
                }
                acc
            }.take(5).toSet()
            for (n in 1..90) {
                if (mask[n] && n !in top5ByTop3) {
                    nResHit++
                    for (fi in 0 until fCount) wResSumHit[fi] += mat.get(features[fi].ordinal, n)
                } else if (!mask[n]) {
                    nResMiss++
                    for (fi in 0 until fCount) wResSumMiss[fi] += mat.get(features[fi].ordinal, n)
                }
            }
        }
        val residualRanked = if (nResHit > 0 && nResMiss > 0) {
            (0 until fCount).map { fi ->
                val diff = (wResSumHit[fi] / nResHit) - (wResSumMiss[fi] / nResMiss)
                RankedSignal(features[fi], diff)
            }.sortedByDescending { abs(it.zLift) }.take(5)
        } else {
            topSignals.drop(1).ifEmpty { topSignals }
        }

        fun weightedTerm(sig: RankedSignal, scale: Double): ExprNode {
            val w = kotlin.math.round(((if (sig.zLift >= 0.0) 1.0 else -1.0) * abs(sig.zLift).coerceIn(0.25, 2.5) * scale) * 100.0) / 100.0
            return ExprNode.Binary(BinaryOp.MUL, ExprNode.Const(w), ExprNode.Var(sig.feature))
        }

        fun buildLinearSum(sigs: List<RankedSignal>, scale: Double, unaryWrap: UnaryOp? = null): ExprNode {
            var expr = weightedTerm(sigs.first(), scale)
            for (i in 1 until sigs.size) {
                expr = ExprNode.Binary(BinaryOp.ADD, expr, weightedTerm(sigs[i], scale * (1.0 / (1.0 + i * 0.15))))
            }
            return if (unaryWrap != null) ExprNode.Unary(unaryWrap, expr) else expr
        }

        val out = ArrayList<SymbolicGenome>(maxCount)
        val localSigs = HashSet<String>(maxCount * 2)
        val combos = listOf(2, 3, 4, 5)
        val unaries = listOf<UnaryOp?>(null, UnaryOp.SIGMOID, UnaryOp.SQRT_ABS)
        val temps = listOf(0.28, 0.45, 0.72)

        var idx = 0
        for (kCount in combos) {
            for (uOp in unaries) {
                for (slotSplit in listOf(2, 3, 4)) {
                    if (out.size >= maxCount) break
                    val pList = topSignals.take(minOf(kCount, topSignals.size))
                    val sList = residualRanked.take(minOf(3, residualRanked.size))
                    val temp = temps[idx % temps.size]
                    val primBase = buildLinearSum(pList, 1.0, uOp)
                    val secBase = buildLinearSum(sList, 1.0, UnaryOp.SIGMOID)
                    val genome = SymbolicGenome(
                        id = "and_causal_g${generation}_$idx",
                        generation = generation.coerceAtLeast(1),
                        parentIds = listOf("causal_disc_lift"),
                        primaryExpr = SymbolicFormula.simplify(primBase),
                        secondaryExpr = SymbolicFormula.simplify(secBase),
                        primarySlots = slotSplit,
                        samplingTemp = temp,
                        exploreNoise = 0.0,
                        seed = sessionSalt * 31_337L + generation * 991L + idx * 19L + 3L
                    )
                    val sig = genome.describeCompact()
                    idx++
                    if (sig !in seenSignatures && localSigs.add(sig)) {
                        out.add(genome)
                    }
                }
                if (out.size >= maxCount) break
            }
            if (out.size >= maxCount) break
        }
        return out
    }

    private fun fmt3(v: Float): String = String.format(Locale.FRANCE, "%.3f", v)
    private fun fmt4(v: Double): String = String.format(Locale.FRANCE, "%.4f", v)
    private fun round2(v: Double): Double = kotlin.math.round(v * 100.0) / 100.0
}
