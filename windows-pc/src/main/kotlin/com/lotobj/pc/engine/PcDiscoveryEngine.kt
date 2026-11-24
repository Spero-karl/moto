package com.lotobj.pc.engine

import com.lotobj.pc.model.ApprovedBacktestLedgerRecord
import com.lotobj.pc.model.BacktestPeriod
import com.lotobj.pc.model.BacktestResult
import com.lotobj.pc.model.BacktestStepResult
import com.lotobj.pc.model.BinaryOp
import com.lotobj.pc.model.ComputeMode
import com.lotobj.pc.model.DailyBacktestSummary
import com.lotobj.pc.model.DiscoveredFormulaRecord
import com.lotobj.pc.model.ExprNode
import com.lotobj.pc.model.HorairePerformance
import com.lotobj.pc.model.LotoDraw
import com.lotobj.pc.model.MonteCarloBaseline
import com.lotobj.pc.model.PredictedNumberConfidence
import com.lotobj.pc.model.PredictionResult
import com.lotobj.pc.model.RandomReferenceComparison
import com.lotobj.pc.model.RawFeature
import com.lotobj.pc.model.RawFeatureMatrix
import com.lotobj.pc.model.RejectedFormulaSummary
import com.lotobj.pc.model.SymbolicFormulaSummary
import com.lotobj.pc.model.SymbolicGenome
import com.lotobj.pc.model.UnaryOp
import com.lotobj.pc.util.LotoDateUtils
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.booleanOrNull
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
import java.util.Locale
import java.util.concurrent.Callable
import java.util.concurrent.Executors
import java.util.concurrent.ForkJoinPool
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.sqrt
import kotlin.random.Random

data class CachedRawStep(
    val stepIndex: Int,
    val draw: LotoDraw,
    val actualNumbers: List<Int>,
    val actualSet: Set<Int>,
    val strictPastSize: Int,
    val maxPastTs: Long,
    val dayKey: String,
    val horaire: String,
    val game: String,
    val rawMatrix: RawFeatureMatrix,
    val actualMask: BooleanArray = BooleanArray(91).apply { for (n in actualNumbers) if (n in 1..90) this[n] = true },
    val dayEpoch: Long = runCatching { java.time.LocalDate.parse(dayKey).toEpochDay() }.getOrDefault(0L)
)

data class FourStageCausalCache(
    val cacheKey: String,
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
    val zeroMatchCount: Int = matchesList.count { it == 0 },
    val threePlusCount: Int = matchesList.count { it >= 3 },
    val fourPlusCount: Int = matchesList.count { it >= 4 },
    val fiveMatchCount: Int = matchesList.count { it >= 5 }
) {
    val zeroMatchRate: Float
        get() = if (tested > 0) zeroMatchCount.toFloat() / tested.toFloat() else 0f

    val twoPlusRate: Float
        get() = if (tested > 0) twoPlusCount.toFloat() / tested.toFloat() else 0f

    /** Règle stricte : toujours strictement moins de 50 % de tirages à 0 boule trouvée (≥1 boule > 50 %) */
    val passesZeroCapStrict: Boolean
        get() = tested == 0 || (zeroMatchCount * 2 < tested)
}

data class GenomeCandidateEvaluation(
    val genome: SymbolicGenome,
    val discStats: StageEvalStats,
    val rollStats: StageEvalStats,
    val rollingStabilityPassed: Boolean,
    val fitnessScore: Double,
    val status: String,
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
                    "ROLLING_PASSED" -> "Priorité ≥2/tirage & ≥3/jour (<50% 0-boule)"
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
            rollingZeroMatchRate = rollStats.zeroMatchRate,
            rollingThreePlusCount = rollStats.threePlusCount,
            rollingFourPlusCount = rollStats.fourPlusCount,
            rollingFiveMatchCount = rollStats.fiveMatchCount,
            rollingTwoPlusRate = rollStats.twoPlusRate
        )
    }

    fun toRecord(periodName: String, updatedAt: Long): DiscoveredFormulaRecord {
        val feats = (SymbolicFormula.usedFeatures(genome.primaryExpr) + SymbolicFormula.usedFeatures(genome.secondaryExpr))
            .map { it.symbol }
            .distinct()
        return DiscoveredFormulaRecord(
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
    val seenSignatures: MutableSet<String> = LinkedHashSet(),
    val stagnationGenerations: Int = 0,
    val activeParadigmIndex: Int = 0,
    val bestOverallFitness: Double = Double.NEGATIVE_INFINITY,
    val bestOverallStreak: Int = 0
) {
    val excludedSignaturesCount: Int get() = seenSignatures.size
}

data class EvolutionGenerationResult(
    val nextState: SymbolicEvolutionState,
    val bestTrainEval: GenomeCandidateEvaluation?,
    val winningEval: GenomeCandidateEvaluation?,
    val auditedEval: GenomeCandidateEvaluation?,
    val newRejected: List<RejectedFormulaSummary>,
    val persistedRecords: List<DiscoveredFormulaRecord>,
    val featureHistogram: List<Pair<String, Float>>,
    val randomComparison: RandomReferenceComparison,
    val generationNote: String,
    val gpuBatchMs: Double,
    val cpuBatchMs: Double,
    val evaluationsPerSec: Double,
    val injectedOutcomes: Map<String, String> = emptyMap()
)

data class PendingLivePrediction(
    val dayKey: String,
    val horaire: String,
    val targetTimestamp: Long,
    val predictedNumbers: List<Int>
)

/**
 * Moteur de calcul PC haute performance :
 * - Exploite Multi-GPU OpenCL (NVIDIA + AMD + Intel simultanément) ET 100 % des cœurs CPU EN PARALLÈLE RÉEL
 * - Registre global zéro-réutilisation : aucune méthode déjà testée n'est ré-évaluée lors de la recherche
 * - Détection rapide de stagnation (>= 2 générations) + rotation dynamique de familles de théorèmes mathématiques
 * - Injection et audit des enseignements du Chercheur (< 50 % de 0-boule + gros coups 3/5, 4/5, 5/5)
 */
object PcDiscoveryEngine {

    private const val MAX_SEEN_SIGNATURES = 250_000

    private val threadScratch = ThreadLocal.withInitial { StageScratchBuffers() }

    private val gpuAsyncExecutor = Executors.newSingleThreadExecutor { r ->
        Thread(r, "opencl-heterogeneous-gpu").apply { isDaemon = true }
    }

    fun buildCausalCache(
        history: List<LotoDraw>,
        period: BacktestPeriod = BacktestPeriod.THREE_WEEKS,
        referenceNowMs: Long = System.currentTimeMillis(),
        monteCarloSims: Int = 5_000,
        pool: ForkJoinPool = ForkJoinPool.commonPool(),
        customDays: Int? = period.days
    ): FourStageCausalCache {
        val chronological = history
            .distinctBy { it.uniqueId }
            .sortedWith(compareBy<LotoDraw> { it.timestamp }.thenBy { it.uniqueId })

        val resolvedDays = if (period == BacktestPeriod.ALL_HISTORY && customDays == null) null else (customDays ?: period.days)?.coerceAtLeast(1)
        val resolvedLabel = when {
            resolvedDays == null -> period.label
            period.days == resolvedDays -> period.label
            else -> "$resolvedDays Jours (Perso)"
        }

        if (chronological.size < 4) {
            return FourStageCausalCache(
                cacheKey = "empty",
                periodLabel = resolvedLabel,
                dateRangeLabel = "Données insuffisantes",
                steps = emptyList(),
                discEnd = 0,
                rollEnd = 0,
                confEnd = 0,
                confirmationMonteCarlo = MonteCarloBaseline(0, 4242L, 0, 0, 0f, 0f, 0f, 0f, 0f, 0, 0, 0f, 0, 0)
            )
        }

        val range = SymbolicFormula.computePeriodRange(chronological, period, referenceNowMs, resolvedDays)
        val rangeIndices = range.toList()
        val matrixByIdx = SymbolicFormula.extractRawMatrixSequence(chronological, rangeIndices.toIntArray())

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

        val tasks = rangeIndices.map { t ->
            Callable {
                val targetDraw = chronological[t]
                val actualNumbers = SymbolicFormula.extractDistinctBalls(targetDraw).sorted()
                if (actualNumbers.size < 5) return@Callable null

                val strictPastCount = strictCountAt[t]
                if (strictPastCount == 0) return@Callable null

                val maxPastTs = maxPastTsAt[t]
                check(maxPastTs < targetDraw.timestamp) {
                    "Violation causale détectée : maxPast=$maxPastTs >= target=${targetDraw.timestamp}"
                }

                val horaire = LotoDateUtils.normalizeHoraire(targetDraw)
                val game = LotoDateUtils.normalizeGame(targetDraw)
                val rawMatrix = matrixByIdx[t] ?: return@Callable null
                val dayKey = LotoDateUtils.formatDayKey(targetDraw.timestamp)
                val actualMask = BooleanArray(91)
                for (num in actualNumbers) if (num in 1..90) actualMask[num] = true
                val dayEpoch = runCatching { java.time.LocalDate.parse(dayKey).toEpochDay() }.getOrDefault(0L)

                CachedRawStep(
                    stepIndex = t,
                    draw = targetDraw,
                    actualNumbers = actualNumbers,
                    actualSet = actualNumbers.toSet(),
                    strictPastSize = strictPastCount,
                    maxPastTs = maxPastTs,
                    dayKey = dayKey,
                    horaire = horaire,
                    game = game,
                    rawMatrix = rawMatrix,
                    actualMask = actualMask,
                    dayEpoch = dayEpoch
                )
            }
        }

        val rawList = pool.invokeAll(tasks).mapNotNull { it.get() }
        val rawSize = rawList.size
        val cached = if (resolvedDays != null && rawSize > 0) {
            val totalBlocks = (rawSize + 5) / 6
            rawList.mapIndexed { idx, s ->
                val blockFromEnd = (rawSize - 1 - idx) / 6
                s.copy(stepIndex = idx + 1, dayEpoch = (totalBlocks - 1 - blockFromEnd).toLong())
            }
        } else {
            rawList.mapIndexed { idx, s -> s.copy(stepIndex = idx + 1) }
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
            n >= 8 -> {
                discEnd = ((n * 45) / 100).coerceAtLeast(3)
                rollEnd = (discEnd + ((n * 20) / 100).coerceAtLeast(2)).coerceAtMost(n - 3)
                confEnd = (rollEnd + ((n * 20) / 100).coerceAtLeast(2)).coerceAtMost(n - 1)
            }
            n == 6 -> {
                discEnd = 2
                rollEnd = 3
                confEnd = 5
            }
            n >= 5 -> {
                discEnd = ((n * 45) / 100).coerceAtLeast(2)
                rollEnd = (discEnd + 1).coerceAtMost(n - 2)
                confEnd = (rollEnd + 1).coerceAtMost(n - 1)
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

        val confDraws = if (rollEnd < confEnd) cached.subList(rollEnd, confEnd).map { it.draw } else emptyList()
        val mcBaseline = ScientificMath.runMonteCarloParallel(
            drawsSubset = confDraws,
            simulations = monteCarloSims,
            seed = 4242L,
            pool = pool
        )

        val cacheKey = "${period.name}_${resolvedDays ?: "ALL"}_${chronological.size}_${cached.firstOrNull()?.draw?.uniqueId}_${cached.lastOrNull()?.draw?.uniqueId}"

        return FourStageCausalCache(
            cacheKey = cacheKey,
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
        populationSize: Int = 512,
        sessionSalt: Long = 1L,
        restoredPopulation: List<SymbolicGenome> = emptyList(),
        preExcludedSignatures: Set<String> = emptySet()
    ): SymbolicEvolutionState {
        val targetSize = populationSize.coerceIn(18, 4096)
        val seen = LinkedHashSet<String>(maxOf(2048, preExcludedSignatures.size + targetSize * 2))
        seen.addAll(preExcludedSignatures)
        val population = if (restoredPopulation.isNotEmpty()) {
            val base = restoredPopulation.take(targetSize)
            if (base.size < targetSize) {
                base + generateInitialPopulation(targetSize - base.size, sessionSalt + 99L, seen)
            } else base
        } else {
            generateInitialPopulation(targetSize, sessionSalt, seen)
        }
        return SymbolicEvolutionState(
            generation = 0,
            experimentsCount = 0,
            confirmationTestsCount = 0,
            sessionSalt = sessionSalt,
            population = population,
            seenSignatures = seen
        )
    }

    fun ensureEliteInPopulation(
        state: SymbolicEvolutionState,
        elite: SymbolicGenome?
    ): SymbolicEvolutionState {
        if (elite == null) return state
        val sig = elite.describeCompact()
        // Ne réinjecte pas à l'identique si déjà évalué, mais s'assure qu'un descendant muté neuf en dérive si déjà vu
        if (sig in state.seenSignatures) return state
        if (state.population.any { it.describeCompact() == sig }) return state
        val merged = (listOf(elite) + state.population).take(state.population.size.coerceAtLeast(1))
        return state.copy(population = merged)
    }

    fun generateInitialPopulation(
        size: Int = 512,
        sessionSalt: Long = 1L,
        seenSignatures: Set<String>? = null,
        startParadigm: Int = 0
    ): List<SymbolicGenome> {
        val rng = Random(sessionSalt * 104_729L + 17L)
        val localBatchSigs = HashSet<String>(size * 2)
        val out = ArrayList<SymbolicGenome>(size)

        // Amorçage via Strategy Factory : moitié de la population issue des fabriques spécialisées
        // (Arithmétique Modulaire, Densité Probabiliste, Triangulation, Synthèse Hybride)
        val factoryTarget = (size / 2).coerceAtLeast(1)
        val preferredStart = StrategyFactory.domainForParadigmIndex(startParadigm)
        val (factoryBatch, _) = StrategyFactory.generateBatch(
            count = factoryTarget,
            generation = 1,
            sessionSalt = sessionSalt,
            seenSignatures = seenSignatures ?: emptySet(),
            preferredDomain = if (startParadigm == 0) null else preferredStart,
            complexityDepth = 4
        )
        for (fg in factoryBatch) {
            if (localBatchSigs.add(fg.describeCompact())) {
                out.add(fg)
            }
        }

        for (idx in out.size until size) {
            var attempt = 0
            var candidate: SymbolicGenome
            do {
                val pIdx = startParadigm + idx + attempt
                val exprA = if ((idx + attempt) % 4 == 3) {
                    SymbolicFormula.randomExpr(if (idx % 2 == 0) 4 else 5, rng)
                } else {
                    SymbolicFormula.randomTheoremExpr(pIdx, rng)
                }
                val exprB = if ((idx + attempt) % 3 == 0) {
                    SymbolicFormula.randomExpr(3, rng)
                } else {
                    SymbolicFormula.randomTheoremExpr(pIdx + 3, rng)
                }
                candidate = SymbolicGenome(
                    id = "pc_sym_g1_$idx",
                    generation = 1,
                    parentIds = emptyList(),
                    primaryExpr = exprA,
                    secondaryExpr = exprB,
                    primarySlots = rng.nextInt(1, 5),
                    samplingTemp = round2(0.30 + rng.nextDouble() * 0.90),
                    exploreNoise = 0.0,
                    seed = sessionSalt * 97_003L + idx * 1_993L + attempt * 131L + 7L
                )
                val sig = candidate.describeCompact()
                val isNew = (seenSignatures == null || sig !in seenSignatures) && localBatchSigs.add(sig)
                if (isNew || attempt >= 8) break
                attempt++
            } while (true)
            out.add(candidate)
        }
        return out
    }

    fun executeEvolutionGeneration(
        state: SymbolicEvolutionState,
        cache: FourStageCausalCache,
        periodName: String,
        minObjectiveStreak: Int = 4,
        populationSize: Int = state.population.size.coerceIn(18, 4096),
        computeMode: ComputeMode = ComputeMode.GPU_AND_CPU_MAX,
        cpuPool: ForkJoinPool,
        gpuAccelerator: GpuOpenClAccelerator,
        injectedGenomes: List<SymbolicGenome> = emptyList()
    ): EvolutionGenerationResult {
        val genStartNs = System.nanoTime()
        val cacheDistinctDays = cache.steps.map { it.dayEpoch }.distinct().size.coerceAtLeast(1)
        val requiredStreak = minObjectiveStreak.coerceIn(1, cacheDistinctDays)
        val nextGen = state.generation + 1
        val rng = Random(nextGen * 83_117L + state.sessionSalt * 13_331L + 5L)

        // Fusionner d'abord les stratégies injectées par "Le Chercheur" dans le vivier actif
        val seen = state.seenSignatures
        val targetPopSize = populationSize.coerceIn(18, 4096)
        val causalBoostPop = if ((state.generation == 0 || state.generation % 5 == 0) && cache.discoverySteps.isNotEmpty()) {
            val maxRidge = minOf(28, maxOf(10, targetPopSize / 4))
            val ridgePop = CausalAiLearner.fitSymbolicRidgeGenomes(
                discoverySteps = cache.discoverySteps,
                generation = nextGen,
                sessionSalt = state.sessionSalt + nextGen * 31L,
                seenSignatures = seen,
                maxCount = maxRidge / 2 + 4,
                paradigmHint = state.activeParadigmIndex
            )
            val liftPop = synthesizeCausalDiscoveryGenomes(
                discoverySteps = cache.discoverySteps,
                generation = nextGen,
                sessionSalt = state.sessionSalt,
                seenSignatures = seen,
                maxCount = maxRidge - ridgePop.size
            )
            (ridgePop + liftPop).distinctBy { it.describeCompact() }
        } else emptyList()
        val mergedSeedPop = if (injectedGenomes.isNotEmpty() || causalBoostPop.isNotEmpty()) {
            val freshInjected = (injectedGenomes + causalBoostPop).filter { it.describeCompact() !in seen }
            (freshInjected + state.population).take(targetPopSize)
        } else {
            state.population
        }
        val activePop = buildUnseenActivePopulation(
            rawPop = mergedSeedPop,
            targetSize = targetPopSize,
            nextGen = nextGen,
            sessionSalt = state.sessionSalt,
            paradigmIdx = state.activeParadigmIndex,
            seenSignatures = seen,
            rng = rng
        )
        val injectedIdSet = injectedGenomes.map { it.id }.toSet()

        // Enregistre immédiatement toutes les méthodes de cette génération dans le registre zéro-réutilisation
        recordSeenSignatures(seen, activePop)

        var gpuMs = 0.0
        var cpuMs = 0.0
        val useGpu = computeMode == ComputeMode.GPU_AND_CPU_MAX && gpuAccelerator.isGpuReady

        // Pré-compile RPN en parallèle multi-cœurs une seule fois pour tous les génomes (GPU + CPU)
        val compiledActive = mapIndexedParallel(cpuPool, activePop.size) { i ->
            val g = activePop[i]
            SymbolicFormula.compileToRpn(g.primaryExpr) to SymbolicFormula.compileToRpn(g.secondaryExpr)
        }

        // Évaluation hétérogène simultanée (Multi-GPU + 100% CPU en parallèle réel) : Découverte
        val (discResults, dGpuMs, dCpuMs) = evaluateStageHeterogeneous(
            stageKey = "${cache.cacheKey}_disc",
            steps = cache.discoverySteps,
            genomes = activePop,
            compiled = compiledActive,
            useGpu = useGpu,
            cpuPool = cpuPool,
            gpuAccelerator = gpuAccelerator
        )
        gpuMs += dGpuMs
        cpuMs += dCpuMs

        // Calibration intensive parallélisée des élites de Découverte (élargie quand GPU actif)
        val calEliteLimit = if (useGpu) minOf(20, activePop.size) else minOf(12, activePop.size)
        val calTrialsPerElite = if (useGpu) 10 else 6
        val topIndicesForCalibration = discResults.indices
            .sortedByDescending { idx ->
                val ds = discResults[idx]
                val (_, penHits, nearProgress) = computeBlockProgressGradient(cache.discoverySteps, ds.matchesList)
                (ds.twoPlusRate * 180.0) +
                    (ds.maxConsecutiveQualifyingDays * 55.0) +
                    (ds.qualifyingDays * 32.0) +
                    (penHits * 24.0) +
                    (nearProgress * 18.0) +
                    (ds.avgMatches * 22.0) +
                    (ds.twoPlusCount * 6.0) +
                    (ds.threePlusCount * 14.0) -
                    (ds.zeroMatchRate * 12.0)
            }
            .take(calEliteLimit)

        val seenConcurrent = java.util.concurrent.ConcurrentHashMap.newKeySet<String>(seen.size + calEliteLimit * calTrialsPerElite + 64)
        seenConcurrent.addAll(seen)
        val baseCalSeed = rng.nextLong()
        val parallelCalSlots = mapIndexedParallel(cpuPool, topIndicesForCalibration.size * calTrialsPerElite) { flatIdx ->
            val ePos = flatIdx / calTrialsPerElite
            val trial = flatIdx % calTrialsPerElite
            val popIdx = topIndicesForCalibration[ePos]
            val elite = activePop[popIdx]
            val localRng = Random(nextGen * 9973L + ePos * 317L + trial * 41L + baseCalSeed + flatIdx * 13L)
            val cand = elite.copy(
                id = "${elite.id}_cal${ePos}_$trial",
                generation = nextGen,
                primaryExpr = SymbolicFormula.simplify(perturbConsts(elite.primaryExpr, localRng)),
                secondaryExpr = SymbolicFormula.simplify(perturbConsts(elite.secondaryExpr, localRng)),
                samplingTemp = round2((elite.samplingTemp + (localRng.nextDouble() * 0.24 - 0.12)).coerceIn(0.22, 1.5)),
                exploreNoise = 0.0
            )
            val sig = cand.describeCompact()
            if (seenConcurrent.add(sig)) {
                cand to (SymbolicFormula.compileToRpn(cand.primaryExpr) to SymbolicFormula.compileToRpn(cand.secondaryExpr))
            } else null
        }
        val acceptedCal = parallelCalSlots.filterNotNull()
        val calibratedTrials = acceptedCal.map { it.first }
        val compiledTrials = acceptedCal.map { it.second }
        for (c in calibratedTrials) seen.add(c.describeCompact())

        val combinedPop: List<SymbolicGenome>
        val combinedCompiled: List<Pair<CompiledRpnProgram, CompiledRpnProgram>>
        val combinedDiscStats: List<StageEvalStats>

        if (calibratedTrials.isNotEmpty()) {
            val (trialDiscStats, tGpuMs, tCpuMs) = evaluateStageHeterogeneous(
                stageKey = "${cache.cacheKey}_disc",
                steps = cache.discoverySteps,
                genomes = calibratedTrials,
                compiled = compiledTrials,
                useGpu = useGpu,
                cpuPool = cpuPool,
                gpuAccelerator = gpuAccelerator
            )
            gpuMs += tGpuMs
            cpuMs += tCpuMs

            combinedPop = activePop + calibratedTrials
            combinedCompiled = compiledActive + compiledTrials
            combinedDiscStats = discResults + trialDiscStats
        } else {
            combinedPop = activePop
            combinedCompiled = compiledActive
            combinedDiscStats = discResults
        }

        // Évaluation 2 hétérogène simultanée : Validation Glissante (Multi-GPU + 100% CPU en parallèle)
        val (rollStatsList, rGpuMs, rCpuMs) = evaluateStageHeterogeneous(
            stageKey = "${cache.cacheKey}_roll",
            steps = cache.rollingSteps,
            genomes = combinedPop,
            compiled = combinedCompiled,
            useGpu = useGpu,
            cpuPool = cpuPool,
            gpuAccelerator = gpuAccelerator
        )
        gpuMs += rGpuMs
        cpuMs += rCpuMs

        val phase12Raw = mapIndexedParallel(cpuPool, combinedPop.size) { gIdx ->
            scoreDiscoveryAndRolling(
                combinedPop[gIdx],
                combinedDiscStats[gIdx],
                rollStatsList[gIdx],
                cache.discoverySteps,
                cache.rollingSteps
            )
        }

        val injectedOutcomes = HashMap<String, String>()
        if (injectedIdSet.isNotEmpty()) {
            val medianFit = phase12Raw.map { it.fitnessScore }.sorted().let {
                if (it.isEmpty()) 0.0 else it[it.size / 2]
            }
            for (ev in phase12Raw) {
                if (ev.genome.id in injectedIdSet) {
                    val kept = ev.rollingStabilityPassed && (ev.fitnessScore >= medianFit || ev.rollStats.avgMatches > BacktestResult.RANDOM_EXPECTED_MATCHES)
                    injectedOutcomes[ev.genome.id] = if (kept) "KEPT" else "DISCARDED"
                }
            }
        }

        val preliminaryRanked = phase12Raw.sortedByDescending { it.fitnessScore }

        // Ré-audit CPU Double précision exacte sur les 12 meilleurs candidats si le kernel GPU rapide était actif
        val rankedTrainEvals = if (useGpu && gpuAccelerator.isFullStageKernelReady && preliminaryRanked.isNotEmpty()) {
            val topK = minOf(12, preliminaryRanked.size)
            val reAuditedTop = mapIndexedParallel(cpuPool, topK) { idx ->
                val ev = preliminaryRanked[idx]
                val ds = evaluateGenomeOnStage(ev.genome, cache.discoverySteps)
                val rs = evaluateGenomeOnStage(ev.genome, cache.rollingSteps)
                scoreDiscoveryAndRolling(ev.genome, ds, rs, cache.discoverySteps, cache.rollingSteps)
            }
            (reAuditedTop + preliminaryRanked.drop(topK)).sortedByDescending { it.fitnessScore }
        } else {
            preliminaryRanked
        }

        val newBestTrainList = (state.bestGenomes + rankedTrainEvals.filter { it.rollingStabilityPassed })
            .distinctBy { it.genome.describeCompact() }
            .sortedByDescending { it.fitnessScore }
            .take(32)

        val newlyRejected = ArrayList<RejectedFormulaSummary>()
        for (ev in rankedTrainEvals) {
            if (!ev.rollingStabilityPassed && newlyRejected.size < 40) {
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
        val updatedRecords = ArrayList<DiscoveredFormulaRecord>()
        val nowMs = System.currentTimeMillis()

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
                val sig = cand.genome.describeCompact()
                sig !in updatedTestedSigs && (
                    updatedConfTestsCount == 0 ||
                        cand.consolidatedTrainStreak >= 1 ||
                        cand.discStats.maxConsecutiveQualifyingDays >= 1 ||
                        cand.rollStats.maxConsecutiveQualifyingDays >= 1 ||
                        cand.rollStats.twoPlusCount >= 1 ||
                        (cand.discStats.twoPlusCount >= 2 && cand.rollStats.avgMatches >= 0.42f) ||
                        cand.fitnessScore >= state.bestOverallFitness * 0.96
                    )
            }.take(5)
        } else {
            emptyList()
        }

        val mc = cache.confirmationMonteCarlo
        val oosCombinedSteps = cache.rollingSteps + cache.confirmationSteps
        val trainAndConfSteps = cache.discoverySteps + cache.rollingSteps + cache.confirmationSteps
        val fullSteps = trainAndConfSteps + cache.sealedSteps

        for (cand in confCandidates) {
            val sig = cand.genome.describeCompact()
            if (updatedTestedSigs.size >= 8_000) {
                val it = updatedTestedSigs.iterator()
                if (it.hasNext()) {
                    it.next()
                    it.remove()
                }
            }
            updatedTestedSigs.add(sig)
            updatedConfTestsCount++

            val confStats = evaluateGenomeOnStage(cand.genome, cache.confirmationSteps)
            val oosStats = evaluateGenomeOnStage(cand.genome, oosCombinedSteps)
            val effTests = effectiveMultipleTestFactor(updatedConfTestsCount)
            val beatsMcMean = confStats.avgMatches > maxOf(BacktestResult.RANDOM_EXPECTED_MATCHES, mc.meanAvgMatches)
            val pRaw = if (beatsMcMean) ScientificMath.twoSidedPValueVsRandom(confStats.matchesList) else 1.0
            val pAdj = (pRaw * effTests).coerceAtMost(1.0)
            val oosBeatsMc = oosStats.avgMatches > maxOf(BacktestResult.RANDOM_EXPECTED_MATCHES, mc.meanAvgMatches)
            val pOosRaw = if (oosBeatsMc) ScientificMath.twoSidedPValueVsRandom(oosStats.matchesList) else 1.0
            val pOosAdj = (pOosRaw * effTests).coerceAtMost(1.0)
            val confSignificant = (beatsMcMean && (pAdj < 0.05 || (pRaw < 0.025 && confStats.avgMatches > mc.avgMatchesCiHigh))) ||
                (cache.confirmationSteps.size < 6 && beatsMcMean && oosBeatsMc && (pOosAdj < 0.05 || pOosRaw < 0.025))
            val (oosConsQualDays, oosConsStreak) = computeConsolidatedDayStats(oosCombinedSteps, oosStats.matchesList)
            val trainAndConfMatches = cand.discStats.matchesList + cand.rollStats.matchesList + confStats.matchesList
            val (trainConfQualDays, trainConfStreak) = computeConsolidatedDayStats(trainAndConfSteps, trainAndConfMatches)
            val twoPlusFirst5 = cand.discStats.twoPlusCount + cand.rollStats.twoPlusCount + confStats.twoPlusCount

            val objectiveMet = if (cacheDistinctDays <= 1) {
                confStats.qualifyingDays >= 1 ||
                    oosConsQualDays >= 1 ||
                    trainConfQualDays >= 1 ||
                    (twoPlusFirst5 >= 2 && confStats.avgMatches >= 0.5f)
            } else {
                confStats.maxConsecutiveQualifyingDays >= requiredStreak ||
                    oosConsStreak >= requiredStreak ||
                    (oosStats.maxConsecutiveQualifyingDays >= requiredStreak && confStats.qualifyingDays >= 1) ||
                    (trainConfStreak >= requiredStreak && confStats.avgMatches > BacktestResult.RANDOM_EXPECTED_MATCHES) ||
                    (cacheDistinctDays <= 4 && maxOf(cand.discStats.maxConsecutiveQualifyingDays, cand.consolidatedTrainStreak) >= requiredStreak && confStats.avgMatches > BacktestResult.RANDOM_EXPECTED_MATCHES)
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
                updatedRecords.add(rej.toRecord(periodName, nowMs))
                newlyRejected.add(RejectedFormulaSummary(cand.genome.id, nextGen, sig, "Confirmation Indépendante", reason, cand.rollStats.avgMatches, confStats.avgMatches))
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
                updatedRecords.add(rej.toRecord(periodName, nowMs))
                newlyRejected.add(RejectedFormulaSummary(cand.genome.id, nextGen, sig, "Confirmation Indépendante", reason, cand.rollStats.avgMatches, confStats.avgMatches))
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
                updatedRecords.add(rej.toRecord(periodName, nowMs))
                newlyRejected.add(RejectedFormulaSummary(cand.genome.id, nextGen, sig, "Confirmation Indépendante", reason, cand.rollStats.avgMatches, confStats.avgMatches))
            } else {
                val sealedStats = if (cache.sealedSteps.isNotEmpty()) {
                    evaluateGenomeOnStage(cand.genome, cache.sealedSteps)
                } else confStats

                val fullMatches = cand.discStats.matchesList +
                    cand.rollStats.matchesList +
                    confStats.matchesList +
                    (if (cache.sealedSteps.isNotEmpty()) sealedStats.matchesList else emptyList())
                val (_, fullConsStreak) = computeConsolidatedDayStats(fullSteps, fullMatches)
                val fullTwoPlusCount = fullMatches.count { it >= 2 }
                val oneDayRuleOk = cacheDistinctDays > 1 || fullConsStreak >= 1
                val sealedDrawThresholdOk = if (cacheDistinctDays <= 1 && cache.sealedSteps.size <= 1) {
                    true
                } else {
                    sealedStats.avgMatches >= BacktestResult.RANDOM_EXPECTED_MATCHES
                }
                val sealedOk = sealedStats.causalVerified &&
                    sealedDrawThresholdOk &&
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
                        rejectionReason = "Validé 4 étapes (${achievedStreak}j, 0-boule ${(sealedStats.zeroMatchRate * 100).toInt()}%, p_adj=${fmt4(pAdj)})",
                        confStats = confStats,
                        sealedStats = sealedStats,
                        pValueRaw = pRaw,
                        pValueAdjusted = pAdj,
                        consolidatedFullStreak = achievedStreak
                    )
                    winningEval = winner
                    auditedEval = winner
                    updatedRecords.add(winner.toRecord(periodName, nowMs))
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
                    updatedRecords.add(rej.toRecord(periodName, nowMs))
                    newlyRejected.add(RejectedFormulaSummary(cand.genome.id, nextGen, sig, "Test Final Scellé", reason, confStats.avgMatches, sealedStats.avgMatches))
                }
            }
        }

        rankedTrainEvals.take(6).forEach { ev ->
            if (updatedRecords.none { it.id == ev.genome.id }) {
                updatedRecords.add(ev.toRecord(periodName, nowMs))
            }
        }

        // Détection de stagnation & rotation dynamique de théorèmes
        val topTrain = rankedTrainEvals.firstOrNull()
        val bestGenFitness = topTrain?.fitnessScore ?: Double.NEGATIVE_INFINITY
        val bestGenStreak = maxOf(
            topTrain?.discStats?.maxConsecutiveQualifyingDays ?: 0,
            topTrain?.rollStats?.maxConsecutiveQualifyingDays ?: 0,
            topTrain?.consolidatedTrainStreak ?: 0,
            auditedEval?.confStats?.maxConsecutiveQualifyingDays ?: 0,
            auditedEval?.consolidatedFullStreak ?: 0
        )
        val improved = bestGenFitness > state.bestOverallFitness + 0.5 || bestGenStreak > state.bestOverallStreak
        var nextStagnation = if (improved) 0 else (state.stagnationGenerations + 1)
        var nextParadigm = state.activeParadigmIndex
        var paradigmShifted = false
        if (nextStagnation >= 2) {
            nextParadigm = (state.activeParadigmIndex + 1) % SymbolicFormula.THEOREM_PARADIGMS.size
            nextStagnation = 0
            paradigmShifted = true
        }
        val newBestFitnessOverall = maxOf(state.bestOverallFitness, bestGenFitness)
        val newBestStreakOverall = maxOf(state.bestOverallStreak, bestGenStreak)

        // Parents issus des meilleures connues + génération courante ;
        // Tous les enfants générés sont STRICTEMENT inédits (hors registre seenSignatures)
        val breedingPool = (newBestTrainList + rankedTrainEvals)
            .distinctBy { it.genome.describeCompact() }
            .sortedByDescending { it.fitnessScore }

        val nextPopulation = evolvePopulationZeroReuse(
            breedingPool = breedingPool,
            populationSize = targetPopSize,
            generation = nextGen + 1,
            sessionSalt = state.sessionSalt,
            paradigmIdx = nextParadigm,
            paradigmShifted = paradigmShifted,
            seenSignatures = seen,
            rng = rng,
            cpuPool = cpuPool
        ).let { evolved ->
            if (paradigmShifted && cache.discoverySteps.isNotEmpty()) {
                val extraCausal = synthesizeCausalDiscoveryGenomes(
                    discoverySteps = cache.discoverySteps,
                    generation = nextGen + 1,
                    sessionSalt = state.sessionSalt + nextParadigm * 97L,
                    seenSignatures = seen,
                    maxCount = minOf(16, maxOf(4, targetPopSize / 6))
                )
                if (extraCausal.isNotEmpty()) (extraCausal + evolved).take(targetPopSize) else evolved
            } else evolved
        }

        val mergedRejected = (newlyRejected + state.rejectedHistory)
            .distinctBy { "${it.rejectedAtStage}|${it.formulaPreview}" }
            .take(30)

        val featureHistogram = computeFeatureHistogram(nextPopulation)
        val paradigmLabel = SymbolicFormula.THEOREM_PARADIGMS[nextParadigm]
        val note = when {
            winningEval != null -> {
                val achieved = maxOf(
                    winningEval.confStats?.maxConsecutiveQualifyingDays ?: 0,
                    winningEval.rollStats.maxConsecutiveQualifyingDays,
                    winningEval.discStats.maxConsecutiveQualifyingDays
                )
                "Gagnant ${achieved}j validé 4 niveaux (0-boule ${(winningEval.rollStats.zeroMatchRate * 100).toInt()}%, p_adj=${fmt4(winningEval.pValueAdjusted)})"
            }
            paradigmShifted ->
                "⚡ Choc Évolutif Gén.$nextGen → Nouveau théorème [$paradigmLabel] • ${seen.size} méthodes explorées"
            topStable?.rollingStabilityPassed != true ->
                "Gén.$nextGen [$paradigmLabel] : surapprentissage Glissante • ${seen.size} méthodes uniques"
            newlyRejected.any { it.rejectedAtStage == "Confirmation Indépendante" } ->
                "Gén.$nextGen [$paradigmLabel] : rejet OOS (${newlyRejected.last { it.rejectedAtStage == "Confirmation Indépendante" }.reason})"
            else -> {
                val zPct = ((topTrain?.rollStats?.zeroMatchRate ?: 1f) * 100f).toInt()
                val twoPct = ((topTrain?.rollStats?.twoPlusRate ?: 0f) * 100f).toInt()
                val qDays = topTrain?.rollStats?.qualifyingDays ?: 0
                val b3 = topTrain?.rollStats?.threePlusCount ?: 0
                val b4 = topTrain?.rollStats?.fourPlusCount ?: 0
                "Gén.$nextGen [$paradigmLabel] : ≥2/tir=$twoPct% • Jours(≥3)=$qDays • Gliss=${fmt3(topTrain?.rollStats?.avgMatches ?: 0f)} • 0-boule=$zPct% • ≥3:$b3 ≥4:$b4 (cible ${requiredStreak}j)"
            }
        }

        val totalSecs = ((System.nanoTime() - genStartNs) / 1_000_000_000.0).coerceAtLeast(1e-4)
        val totalEvaluations = combinedPop.size * (cache.discoverySteps.size + cache.rollingSteps.size).coerceAtLeast(1)
        val evalPerSec = totalEvaluations / totalSecs

        val nextState = SymbolicEvolutionState(
            generation = nextGen,
            experimentsCount = state.experimentsCount + combinedPop.size,
            confirmationTestsCount = updatedConfTestsCount,
            sessionSalt = state.sessionSalt,
            population = nextPopulation,
            bestGenomes = newBestTrainList,
            rejectedHistory = mergedRejected,
            confirmationTestedSignatures = updatedTestedSigs,
            validatedCandidate = winningEval ?: state.validatedCandidate,
            seenSignatures = seen,
            stagnationGenerations = nextStagnation,
            activeParadigmIndex = nextParadigm,
            bestOverallFitness = newBestFitnessOverall,
            bestOverallStreak = newBestStreakOverall
        )

        return EvolutionGenerationResult(
            nextState = nextState,
            bestTrainEval = topTrain,
            winningEval = winningEval,
            auditedEval = auditedEval,
            newRejected = newlyRejected,
            persistedRecords = updatedRecords,
            featureHistogram = featureHistogram,
            randomComparison = latestConfComparison,
            generationNote = note,
            gpuBatchMs = gpuMs,
            cpuBatchMs = cpuMs,
            evaluationsPerSec = evalPerSec,
            injectedOutcomes = injectedOutcomes
        )
    }

    fun auditCandidateReadOnly(
        candidate: GenomeCandidateEvaluation,
        cache: FourStageCausalCache,
        confirmationTestsCount: Int
    ): GenomeCandidateEvaluation {
        val confStats = candidate.confStats ?: evaluateGenomeOnStage(candidate.genome, cache.confirmationSteps)
        val sealedStats = candidate.sealedStats ?: evaluateGenomeOnStage(candidate.genome, cache.sealedSteps)
        val mc = cache.confirmationMonteCarlo
        val oosCombinedSteps = cache.rollingSteps + cache.confirmationSteps
        val oosStats = if (oosCombinedSteps.isNotEmpty()) evaluateGenomeOnStage(candidate.genome, oosCombinedSteps) else confStats
        val effTests = effectiveMultipleTestFactor(confirmationTestsCount)
        val beatsMcMean = confStats.avgMatches > maxOf(BacktestResult.RANDOM_EXPECTED_MATCHES, mc.meanAvgMatches)
        val pRaw = if (confStats.matchesList.isNotEmpty() && beatsMcMean) {
            ScientificMath.twoSidedPValueVsRandom(confStats.matchesList)
        } else 1.0
        val pAdj = (pRaw * effTests).coerceAtMost(1.0)
        val oosBeatsMc = oosStats.avgMatches > maxOf(BacktestResult.RANDOM_EXPECTED_MATCHES, mc.meanAvgMatches)
        val pOosRaw = if (oosStats.matchesList.isNotEmpty() && oosBeatsMc) ScientificMath.twoSidedPValueVsRandom(oosStats.matchesList) else 1.0
        val pOosAdj = (pOosRaw * effTests).coerceAtMost(1.0)
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
        val oneDayQualifies = cacheDistinctDays <= 1 &&
            (consFullStreak >= 1 || consTrainStreak >= 1 || confStats.avgMatches >= 1.0f)
        val sealedDrawThresholdOk = if (cacheDistinctDays <= 1 && cache.sealedSteps.size <= 1) {
            true
        } else {
            sealedStats.avgMatches >= BacktestResult.RANDOM_EXPECTED_MATCHES
        }
        val all4Passed = beatsMcMean &&
            ((pAdj < 0.05 || (pRaw < 0.025 && confStats.avgMatches > mc.avgMatchesCiHigh)) ||
                (cache.confirmationSteps.size < 6 && oosBeatsMc && (pOosAdj < 0.05 || pOosRaw < 0.025))) &&
            sealedStats.causalVerified &&
            sealedDrawThresholdOk &&
            (confStats.maxConsecutiveQualifyingDays >= streak.coerceIn(1, 4) ||
                consFullStreak >= streak.coerceIn(1, 4) ||
                (candidate.rollStats.maxConsecutiveQualifyingDays >= 2 && confStats.qualifyingDays >= 1) ||
                oneDayQualifies)

        val effectiveMilestone = maxOf(1, streak)
        val status = if (all4Passed) "VALIDATED" else "MILESTONE_${effectiveMilestone}J"
        val zeroPct = (candidate.rollStats.zeroMatchRate * 100f).toInt()
        val label = if (all4Passed) {
            "Validé 4 niveaux (${effectiveMilestone}J, 0-boule ${zeroPct}%, p_adj=${fmt4(pAdj)})"
        } else {
            "Palier ${effectiveMilestone}J causal (Gliss ${fmt3(candidate.rollStats.avgMatches)} • 0-boule ${zeroPct}% • ≥3:${candidate.rollStats.threePlusCount})"
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

    private fun buildUnseenActivePopulation(
        rawPop: List<SymbolicGenome>,
        targetSize: Int,
        nextGen: Int,
        sessionSalt: Long,
        paradigmIdx: Int,
        seenSignatures: Set<String>,
        rng: Random
    ): List<SymbolicGenome> {
        val out = ArrayList<SymbolicGenome>(targetSize)
        val batchLocalSigs = HashSet<String>(targetSize * 2)

        for (i in rawPop.indices) {
            if (out.size >= targetSize) break
            var g = rawPop[i]
            var sig = g.describeCompact()
            if (sig !in seenSignatures && sig !in batchLocalSigs) {
                batchLocalSigs.add(sig)
                out.add(g)
            } else {
                var attempts = 0
                while (attempts < 6) {
                    val mutatedA = SymbolicFormula.mutate(g.primaryExpr, rng)
                    val mutatedB = SymbolicFormula.mutate(g.secondaryExpr, rng)
                    g = g.copy(
                        id = "pc_sym_g${nextGen}_u${out.size}_$attempts",
                        generation = nextGen,
                        primaryExpr = mutatedA,
                        secondaryExpr = mutatedB,
                        primarySlots = if (rng.nextDouble() < 0.35) rng.nextInt(1, 5) else g.primarySlots,
                        seed = g.seed xor (rng.nextLong() and 0x7FFF_FFFFL)
                    )
                    sig = g.describeCompact()
                    if (sig !in seenSignatures && sig !in batchLocalSigs) break
                    attempts++
                }
                if (sig in seenSignatures || sig in batchLocalSigs) {
                    g = SymbolicGenome(
                        id = "pc_sym_g${nextGen}_th${out.size}",
                        generation = nextGen,
                        parentIds = emptyList(),
                        primaryExpr = SymbolicFormula.randomTheoremExpr(paradigmIdx + out.size, rng),
                        secondaryExpr = SymbolicFormula.randomTheoremExpr(paradigmIdx + out.size + 2, rng),
                        primarySlots = rng.nextInt(1, 5),
                        samplingTemp = round2(0.30 + rng.nextDouble() * 0.90),
                        exploreNoise = 0.0,
                        seed = sessionSalt * 79_193L + nextGen * 997L + out.size * 37L
                    )
                    sig = g.describeCompact()
                }
                batchLocalSigs.add(sig)
                out.add(g)
            }
        }

        if (out.size < targetSize) {
            val extra = generateInitialPopulation(
                size = targetSize - out.size,
                sessionSalt = sessionSalt + nextGen * 17L,
                seenSignatures = seenSignatures,
                startParadigm = paradigmIdx
            )
            for (g in extra) {
                if (out.size >= targetSize) break
                val sig = g.describeCompact()
                if (batchLocalSigs.add(sig)) {
                    out.add(g)
                }
            }
            while (out.size < targetSize) {
                val idx = out.size
                out.add(
                    SymbolicGenome(
                        id = "pc_sym_g${nextGen}_fill$idx",
                        generation = nextGen,
                        parentIds = emptyList(),
                        primaryExpr = SymbolicFormula.randomTheoremExpr(paradigmIdx + idx, rng),
                        secondaryExpr = SymbolicFormula.randomExpr(4, rng),
                        primarySlots = rng.nextInt(1, 5),
                        samplingTemp = round2(0.30 + rng.nextDouble() * 0.90),
                        exploreNoise = 0.0,
                        seed = sessionSalt * 19_009L + nextGen * 313L + idx * 17L
                    )
                )
            }
        }
        return out
    }

    internal fun recordSeenSignatures(seen: MutableSet<String>, genomes: List<SymbolicGenome>) {
        if (seen.size + genomes.size > MAX_SEEN_SIGNATURES) {
            val toRemove = (seen.size + genomes.size) - (MAX_SEEN_SIGNATURES - 20_000)
            val it = seen.iterator()
            var removed = 0
            while (it.hasNext() && removed < toRemove) {
                it.next()
                it.remove()
                removed++
            }
        }
        for (g in genomes) {
            seen.add(g.describeCompact())
        }
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

    internal fun scoreDiscoveryAndRolling(
        genome: SymbolicGenome,
        discStats: StageEvalStats,
        rollStats: StageEvalStats,
        discSteps: List<CachedRawStep> = emptyList(),
        rollSteps: List<CachedRawStep> = emptyList()
    ): GenomeCandidateEvaluation {
        val (consTrainQualDays, consTrainStreak) = if (discSteps.isNotEmpty() || rollSteps.isNotEmpty()) {
            computeConsolidatedDayStats(
                discSteps + rollSteps,
                discStats.matchesList + rollStats.matchesList
            )
        } else {
            maxOf(discStats.qualifyingDays, rollStats.qualifyingDays) to
                maxOf(discStats.maxConsecutiveQualifyingDays, rollStats.maxConsecutiveQualifyingDays)
        }
        val (penTrainCount, _, blockProgressScore) = if (discSteps.isNotEmpty() || rollSteps.isNotEmpty()) {
            computeBlockProgressGradient(
                discSteps + rollSteps,
                discStats.matchesList + rollStats.matchesList
            )
        } else {
            Triple(0, 0, 0.0)
        }
        val overfitDrop = rollStats.tested >= 4 &&
            rollStats.avgMatches < 1.0f &&
            rollStats.twoPlusRate < 0.35f &&
            rollStats.avgMatches + 0.04f < discStats.avgMatches * 0.65f
        val worseThanRandomRolling = rollStats.tested >= 4 &&
            rollStats.avgMatches < BacktestResult.RANDOM_EXPECTED_MATCHES * 0.78f

        val rollingStabilityPassed = !overfitDrop && !worseThanRandomRolling
        val complexity = SymbolicFormula.nodeCount(genome.primaryExpr) + SymbolicFormula.nodeCount(genome.secondaryExpr)
        val parsimonyPenalty = 0.02 * complexity

        // PRIORITÉ ABSOLUE : >= 2 numéros exacts par prédiction sur TOUS les tirages + >= 3 tirages/jour dont avant-dernier >= 2
        val rollTwoPlusRate = rollStats.twoPlusRate.toDouble()
        val discTwoPlusRate = discStats.twoPlusRate.toDouble()
        val rollFullSweepBonus = if (rollStats.tested >= 4 && rollStats.twoPlusCount == rollStats.tested) 360.0 else 0.0
        val discFullSweepBonus = if (discStats.tested >= 4 && discStats.twoPlusCount == discStats.tested) 140.0 else 0.0
        val missedTwoPlusPenalty = ((1.0 - rollTwoPlusRate) * 260.0) + ((1.0 - discTwoPlusRate) * 110.0)
        val twoPlusPriorityScore = (rollTwoPlusRate * 540.0) +
            (discTwoPlusRate * 220.0) +
            rollFullSweepBonus +
            discFullSweepBonus -
            missedTwoPlusPenalty

        val coverageRoll = (1.0 - rollStats.zeroMatchRate) * 180.0
        val coverageDisc = (1.0 - discStats.zeroMatchRate) * 90.0
        val zeroSoftBonus = (if (rollStats.passesZeroCapStrict) 80.0 else 0.0) +
            (if (discStats.passesZeroCapStrict) 40.0 else 0.0) -
            maxOf(0.0, rollStats.zeroMatchRate - 0.50) * 120.0 -
            maxOf(0.0, discStats.zeroMatchRate - 0.50) * 60.0

        val effectiveTrainStreak = maxOf(rollStats.maxConsecutiveQualifyingDays, discStats.maxConsecutiveQualifyingDays, consTrainStreak)
        val rawFitness = twoPlusPriorityScore +
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
            (rollStats.avgMatches * 145.0) +
            (discStats.avgMatches * 55.0) +
            coverageRoll +
            coverageDisc +
            zeroSoftBonus +
            (rollStats.threePlusCount * 22.0) +
            (discStats.threePlusCount * 10.0) +
            (rollStats.fourPlusCount * 45.0) +
            (discStats.fourPlusCount * 20.0) +
            (rollStats.fiveMatchCount * 140.0) +
            (discStats.fiveMatchCount * 60.0) -
            (rollStats.maxZeroStreak * 12.0) -
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
            fitnessScore = if (rollingStabilityPassed) rawFitness else rawFitness - 150.0,
            status = status,
            rejectionStage = if (rollingStabilityPassed) "" else "Validation Glissante",
            rejectionReason = reason,
            consolidatedTrainStreak = consTrainStreak
        )
    }

    private fun evolvePopulationZeroReuse(
        breedingPool: List<GenomeCandidateEvaluation>,
        populationSize: Int,
        generation: Int,
        sessionSalt: Long,
        paradigmIdx: Int,
        paradigmShifted: Boolean,
        seenSignatures: Set<String>,
        rng: Random,
        cpuPool: ForkJoinPool = ForkJoinPool.commonPool()
    ): List<SymbolicGenome> {
        if (breedingPool.isEmpty()) {
            return generateInitialPopulation(populationSize, sessionSalt + generation, seenSignatures, paradigmIdx)
        }

        val poolSize = breedingPool.size.coerceAtLeast(1)
        val immigrantRatio = if (paradigmShifted) 0.55 else 0.25
        val immigrantsCount = (populationSize * immigrantRatio).toInt().coerceIn(6, populationSize / 2 + 4)
        val childrenTarget = (populationSize - immigrantsCount).coerceAtLeast(0)
        val seenConcurrent = java.util.concurrent.ConcurrentHashMap.newKeySet<String>(seenSignatures.size + populationSize * 2)
        seenConcurrent.addAll(seenSignatures)
        val baseSeed = rng.nextLong()

        val children = mapIndexedParallel(cpuPool, childrenTarget) { childIdx ->
            val localRng = Random(baseSeed xor (childIdx.toLong() * 6_364_136_223_846_793_005L + generation * 1009L))
            val p1 = tournamentPick(breedingPool, localRng, poolSize)
            val p2 = tournamentPick(breedingPool, localRng, poolSize)
            var candidate: SymbolicGenome? = null

            for (attempt in 0 until 8) {
                var childA = SymbolicFormula.crossover(p1.primaryExpr, p2.primaryExpr, localRng)
                var childB = SymbolicFormula.crossover(p1.secondaryExpr, p2.secondaryExpr, localRng)
                val mutRate = if (paradigmShifted || attempt > 0) 0.90 else 0.65
                if (localRng.nextDouble() < mutRate) childA = SymbolicFormula.mutate(childA, localRng)
                if (localRng.nextDouble() < mutRate) childB = SymbolicFormula.mutate(childB, localRng)

                val slots = if (localRng.nextDouble() < 0.35 || attempt > 1) localRng.nextInt(1, 5) else ((p1.primarySlots + p2.primarySlots) / 2).coerceIn(1, 4)
                val temp = ((p1.samplingTemp + p2.samplingTemp) * 0.5 + (localRng.nextDouble() * 0.24 - 0.12)).coerceIn(0.22, 1.5)

                val cand = SymbolicGenome(
                    id = "pc_sym_g${generation}_c$childIdx",
                    generation = generation,
                    parentIds = listOf(p1.id, p2.id),
                    primaryExpr = childA,
                    secondaryExpr = childB,
                    primarySlots = slots,
                    samplingTemp = round2(temp),
                    exploreNoise = 0.0,
                    seed = sessionSalt * 53_171L + generation * 1_009L + childIdx * 37L + attempt * 7L
                )
                val sig = cand.describeCompact()
                if (seenConcurrent.add(sig)) {
                    candidate = cand
                    break
                }
            }

            candidate ?: run {
                val fac = StrategyFactory.getFactory(StrategyFactory.domainForParadigmIndex(paradigmIdx + childIdx))
                fac.createGenome(
                    id = "pc_sym_g${generation}_n$childIdx",
                    generation = generation,
                    seed = sessionSalt * 61_331L + generation * 1_999L + childIdx * 43L,
                    complexityDepth = 4
                ).also { seenConcurrent.add(it.describeCompact()) }
            }
        }

        val immigrants = mapIndexedParallel(cpuPool, immigrantsCount) { immIdx ->
            val localRng = Random((baseSeed + 7919L) xor (immIdx.toLong() * 2_862_933_555_777_941_757L + generation * 31L))
            val thIdx = if (immIdx % 3 == 0) paradigmIdx + immIdx else paradigmIdx
            var cand: SymbolicGenome
            var attempt = 0
            do {
                val seed = sessionSalt * 91_009L + generation * 3_001L + immIdx * 71L + attempt * 19L
                cand = if ((immIdx + attempt) % 2 == 0) {
                    val fac = StrategyFactory.getFactory(StrategyFactory.domainForParadigmIndex(thIdx + attempt))
                    fac.createGenome(
                        id = "pc_sym_g${generation}_r$immIdx",
                        generation = generation,
                        seed = seed,
                        complexityDepth = if (paradigmShifted) 5 else 4
                    )
                } else {
                    SymbolicGenome(
                        id = "pc_sym_g${generation}_r$immIdx",
                        generation = generation,
                        parentIds = emptyList(),
                        primaryExpr = SymbolicFormula.randomTheoremExpr(thIdx + attempt, localRng),
                        secondaryExpr = SymbolicFormula.randomTheoremExpr(thIdx + 4, localRng),
                        primarySlots = localRng.nextInt(1, 5),
                        samplingTemp = round2(0.30 + localRng.nextDouble() * 0.90),
                        exploreNoise = 0.0,
                        seed = seed
                    )
                }
                val sig = cand.describeCompact()
                if (seenConcurrent.add(sig) || attempt >= 5) {
                    break
                }
                attempt++
            } while (true)
            cand
        }

        return children + immigrants
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
        pool: ForkJoinPool,
        count: Int,
        crossinline block: (Int) -> R
    ): List<R> {
        if (count <= 0) return emptyList()
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

    private fun splitMix64Next(state: Long): Long {
        var z = state + -7046029254386353131L // 0x9E3779B97F4A7C15L
        z = (z xor (z ushr 30)) * -4658895280553007687L // 0xBF58476D1CE4E5B9L
        z = (z xor (z ushr 27)) * -7723592293110705685L // 0x94D049BB133111EBL
        return z xor (z ushr 31)
    }

    private fun splitMixDouble(z: Long): Double = ((z ushr 11) * (1.0 / 9007199254740992.0))

    fun deterministicStepSalt(dayKey: String, normalizedHoraire: String): Long {
        return "${dayKey}|${normalizedHoraire.trim().uppercase(Locale.ROOT)}".hashCode().toLong()
    }

    @Volatile private var emaGpuBatchMs: Double = 25.0
    @Volatile private var emaCpuBatchMs: Double = 60.0

    /**
     * Évaluation hétérogène SIMULTANÉE adaptative :
     * - Les génomes utilisant SAME_DAY_REP (séquentiels) + le reliquat CPU sont calculés sur 100% des cœurs CPU
     * - Simultanément, 72-95 % des génomes sans dépendance séquentielle sont évalués sur TOUS les GPUs OpenCL
     *   via le kernel complet evalSymbolicFullStage (RPN + Softmax + SplitMix + Hits + Brier directement en VRAM).
     */
    fun evaluateStageHeterogeneous(
        stageKey: String,
        steps: List<CachedRawStep>,
        genomes: List<SymbolicGenome>,
        compiled: List<Pair<CompiledRpnProgram, CompiledRpnProgram>>,
        useGpu: Boolean,
        cpuPool: ForkJoinPool,
        gpuAccelerator: GpuOpenClAccelerator
    ): Triple<List<StageEvalStats>, Double, Double> {
        if (genomes.isEmpty()) return Triple(emptyList(), 0.0, 0.0)
        if (steps.isEmpty()) {
            return Triple(
                List(genomes.size) { evaluateGenomeOnStage(genomes[it], steps, precompiledA = compiled[it].first, precompiledB = compiled[it].second) },
                0.0,
                0.0
            )
        }

        if (!useGpu || !gpuAccelerator.isGpuReady || genomes.size < 2) {
            val t0 = System.nanoTime()
            val stats = mapIndexedParallel(cpuPool, genomes.size) { i ->
                evaluateGenomeOnStage(genomes[i], steps, null, -1, compiled[i].first, compiled[i].second)
            }
            val cpuMs = (System.nanoTime() - t0) / 1_000_000.0
            return Triple(stats, 0.0, cpuMs)
        }

        val gpuEligible = ArrayList<Int>(genomes.size)
        val cpuDirect = ArrayList<Int>(genomes.size / 4 + 4)
        for (i in genomes.indices) {
            val (rA, rB) = compiled[i]
            if (rA.usesSameDayRep || rB.usesSameDayRep) {
                cpuDirect.add(i)
            } else {
                gpuEligible.add(i)
            }
        }

        val baseRatio = if (gpuAccelerator.gpuCount >= 2) 0.90 else 0.82
        val throughputRatio = if (emaCpuBatchMs > emaGpuBatchMs * 1.15) {
            (baseRatio + 0.05).coerceAtMost(0.95)
        } else if (emaGpuBatchMs > emaCpuBatchMs * 1.35) {
            (baseRatio - 0.10).coerceAtLeast(0.70)
        } else {
            baseRatio
        }
        val gpuTargetCount = (gpuEligible.size * throughputRatio).toInt().coerceIn(0, gpuEligible.size)
        if (gpuTargetCount < 1) {
            val t0 = System.nanoTime()
            val stats = mapIndexedParallel(cpuPool, genomes.size) { i ->
                evaluateGenomeOnStage(genomes[i], steps, null, -1, compiled[i].first, compiled[i].second)
            }
            val cpuMs = (System.nanoTime() - t0) / 1_000_000.0
            return Triple(stats, 0.0, cpuMs)
        }

        val gpuIndices = gpuEligible.subList(0, gpuTargetCount)
        for (k in gpuTargetCount until gpuEligible.size) {
            cpuDirect.add(gpuEligible[k])
        }

        val gpuGenomes = gpuIndices.map { genomes[it] }
        val gpuCompiled = gpuIndices.map { compiled[it] }

        // Chemin ultra-rapide : kernel GPU complet evalSymbolicFullStage (0 transfert raw float90, calcul complet sur GPUs)
        if (gpuAccelerator.isFullStageKernelReady) {
            val gpuFuture = gpuAsyncExecutor.submit(Callable {
                gpuAccelerator.evaluatePopulationStageDirect(stageKey, steps, gpuGenomes, gpuCompiled)
            })

            val cpuStartNs = System.nanoTime()
            val cpuPartitionStats = mapIndexedParallel(cpuPool, cpuDirect.size) { cPos ->
                val origIdx = cpuDirect[cPos]
                val (rA, rB) = compiled[origIdx]
                evaluateGenomeOnStage(genomes[origIdx], steps, null, -1, rA, rB)
            }
            val cpuDirectMs = (System.nanoTime() - cpuStartNs) / 1_000_000.0

            val gpuDirectRes = runCatching { gpuFuture.get() }.getOrNull()
            if (gpuDirectRes != null) {
                val gpuMs = gpuDirectRes.elapsedMs
                val aggStartNs = System.nanoTime()
                val gpuPartitionStats = mapIndexedParallel(cpuPool, gpuIndices.size) { localIdx ->
                    if (gpuDirectRes.isGenomeEvaluated(localIdx)) {
                        buildStatsFromDirectGpu(localIdx, steps, gpuDirectRes.hits, gpuDirectRes.brier)
                    } else {
                        val origIdx = gpuIndices[localIdx]
                        val (rA, rB) = compiled[origIdx]
                        evaluateGenomeOnStage(genomes[origIdx], steps, null, -1, rA, rB)
                    }
                }
                val aggMs = (System.nanoTime() - aggStartNs) / 1_000_000.0
                val totalCpuMs = cpuDirectMs + aggMs
                emaGpuBatchMs = emaGpuBatchMs * 0.7 + gpuMs * 0.3
                emaCpuBatchMs = emaCpuBatchMs * 0.7 + totalCpuMs * 0.3

                val merged = arrayOfNulls<StageEvalStats>(genomes.size)
                for (i in cpuDirect.indices) merged[cpuDirect[i]] = cpuPartitionStats[i]
                for (i in gpuIndices.indices) merged[gpuIndices[i]] = gpuPartitionStats[i]
                return Triple(merged.map { it!! }, gpuMs, totalCpuMs)
            }
            // Si fullKernel a échoué (ex. pilote hybride), ne pas abandonner le GPU : basculer vers evalSymbolicBatch
        }

        // Chemin alternatif raw scores (si fullKernel non supporté)
        val gpuFuture = gpuAsyncExecutor.submit(Callable {
            gpuAccelerator.evaluatePopulationRawScores(stageKey, steps, gpuGenomes, gpuCompiled)
        })

        val cpuStartNs = System.nanoTime()
        val cpuPartitionStats = mapIndexedParallel(cpuPool, cpuDirect.size) { cPos ->
            val origIdx = cpuDirect[cPos]
            val (rA, rB) = compiled[origIdx]
            evaluateGenomeOnStage(genomes[origIdx], steps, null, -1, rA, rB)
        }
        val cpuDirectMs = (System.nanoTime() - cpuStartNs) / 1_000_000.0

        val gpuScores = runCatching { gpuFuture.get() }.getOrNull()
        val gpuMs = gpuScores?.elapsedMs ?: 0.0

        val postGpuStartNs = System.nanoTime()
        val gpuPartitionStats = mapIndexedParallel(cpuPool, gpuIndices.size) { localIdx ->
            val origIdx = gpuIndices[localIdx]
            val (rA, rB) = compiled[origIdx]
            evaluateGenomeOnStage(genomes[origIdx], steps, gpuScores, localIdx, rA, rB)
        }
        val postCpuMs = (System.nanoTime() - postGpuStartNs) / 1_000_000.0
        val totalCpuMs = cpuDirectMs + postCpuMs
        if (gpuMs > 0.0) emaGpuBatchMs = emaGpuBatchMs * 0.7 + gpuMs * 0.3
        emaCpuBatchMs = emaCpuBatchMs * 0.7 + totalCpuMs * 0.3

        val merged = arrayOfNulls<StageEvalStats>(genomes.size)
        for (i in cpuDirect.indices) merged[cpuDirect[i]] = cpuPartitionStats[i]
        for (i in gpuIndices.indices) merged[gpuIndices[i]] = gpuPartitionStats[i]
        return Triple(merged.map { it!! }, gpuMs, totalCpuMs)
    }

    private fun buildStatsFromDirectGpu(
        genomeLocalIdx: Int,
        steps: List<CachedRawStep>,
        flatHits: IntArray,
        flatBrier: FloatArray
    ): StageEvalStats {
        val stepCount = steps.size
        val base = genomeLocalIdx * stepCount
        var twoPlus = 0
        var threePlus = 0
        var fourPlus = 0
        var fiveMatch = 0
        var zeroCount = 0
        var totalHits = 0
        var brierSum = 0.0
        var zeroStreak = 0
        var maxZeroStreak = 0
        var allCausal = true
        val matchesList = ArrayList<Int>(stepCount)

        var curDayEpoch = Long.MIN_VALUE
        var curDayTwoPlus = 0
        var curDayPrevHit = 0
        var curDayLastHit = 0
        var qualDays = 0
        var consecQual = 0
        var maxConsecQual = 0
        var lastQualDayEpoch = Long.MIN_VALUE

        for (stepPos in 0 until stepCount) {
            val step = steps[stepPos]
            if (step.maxPastTs >= step.draw.timestamp) allCausal = false
            val hits = flatHits[base + stepPos].coerceIn(0, 5)
            brierSum += flatBrier[base + stepPos].toDouble().coerceAtLeast(0.0)

            totalHits += hits
            matchesList.add(hits)
            if (hits == 0) zeroCount++
            if (hits >= 2) twoPlus++
            if (hits >= 3) threePlus++
            if (hits >= 4) fourPlus++
            if (hits >= 5) fiveMatch++

            if (hits == 0) {
                zeroStreak++
                if (zeroStreak > maxZeroStreak) maxZeroStreak = zeroStreak
            } else zeroStreak = 0

            val sDayEpoch = step.dayEpoch
            if (sDayEpoch != curDayEpoch) {
                if (curDayEpoch != Long.MIN_VALUE && curDayTwoPlus >= 3 && curDayPrevHit >= 2) {
                    qualDays++
                    consecQual = if (lastQualDayEpoch != Long.MIN_VALUE && curDayEpoch == lastQualDayEpoch + 1L) {
                        consecQual + 1
                    } else 1
                    if (consecQual > maxConsecQual) maxConsecQual = consecQual
                    lastQualDayEpoch = curDayEpoch
                } else if (curDayEpoch != Long.MIN_VALUE) consecQual = 0
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
            consecQual = if (lastQualDayEpoch != Long.MIN_VALUE && curDayEpoch == lastQualDayEpoch + 1L) {
                consecQual + 1
            } else 1
            if (consecQual > maxConsecQual) maxConsecQual = consecQual
        }

        return StageEvalStats(
            tested = stepCount,
            twoPlusCount = twoPlus,
            qualifyingDays = qualDays,
            maxConsecutiveQualifyingDays = maxConsecQual,
            maxZeroStreak = maxZeroStreak,
            avgMatches = totalHits.toFloat() / stepCount,
            brierScore = brierSum / stepCount,
            matchesList = matchesList,
            causalVerified = allCausal,
            zeroMatchCount = zeroCount,
            threePlusCount = threePlus,
            fourPlusCount = fourPlus,
            fiveMatchCount = fiveMatch
        )
    }

    fun evaluateGenomeOnStage(
        genome: SymbolicGenome,
        steps: List<CachedRawStep>,
        gpuBatchScores: GpuBatchScores? = null,
        genomeIndexInBatch: Int = -1,
        precompiledA: CompiledRpnProgram? = null,
        precompiledB: CompiledRpnProgram? = null
    ): StageEvalStats {
        if (steps.isEmpty()) {
            return StageEvalStats(0, 0, 0, 0, 0, 0f, BacktestResult.RANDOM_BRIER_SCORE, emptyList(), true, 0, 0, 0, 0)
        }

        val rpnA = precompiledA ?: SymbolicFormula.compileToRpn(genome.primaryExpr)
        val rpnB = precompiledB ?: SymbolicFormula.compileToRpn(genome.secondaryExpr)
        val canUseGpuScores = gpuBatchScores != null &&
            gpuBatchScores.isGenomeEvaluated(genomeIndexInBatch) &&
            gpuBatchScores.stepCount == steps.size &&
            !rpnA.usesSameDayRep &&
            !rpnB.usesSameDayRep

        var twoPlus = 0
        var threePlus = 0
        var fourPlus = 0
        var fiveMatch = 0
        var zeroCount = 0
        var totalHits = 0
        var brierSum = 0.0
        var zeroStreak = 0
        var maxZeroStreak = 0
        var allCausal = true
        val stepCount = steps.size
        val matchesList = ArrayList<Int>(stepCount)

        val buf = threadScratch.get()
        buf.sameDayCounts.fill(0)
        var activeDayEpoch = Long.MIN_VALUE
        var curDayEpoch = Long.MIN_VALUE
        var curDayTwoPlus = 0
        var curDayPrevHit = 0
        var curDayLastHit = 0
        var qualDays = 0
        var consecQual = 0
        var maxConsecQual = 0
        var lastQualDayEpoch = Long.MIN_VALUE

        for (stepPos in 0 until stepCount) {
            val step = steps[stepPos]
            if (step.maxPastTs >= step.draw.timestamp) allCausal = false
            val sDayEpoch = step.dayEpoch
            if (sDayEpoch != activeDayEpoch) {
                activeDayEpoch = sDayEpoch
                buf.sameDayCounts.fill(0)
            }

            if (canUseGpuScores) {
                gpuBatchScores!!.copyStepRawA(genomeIndexInBatch, stepPos, buf.rawA)
                gpuBatchScores.copyStepRawB(genomeIndexInBatch, stepPos, buf.rawB)
            } else {
                val matrix = step.rawMatrix
                val rpnStack = buf.rpnStack
                for (n in 1..90) {
                    val repNorm = (buf.sameDayCounts[n] / 5.0).coerceIn(0.0, 1.0)
                    buf.rawA[n] = rpnA.evaluateCpu(matrix, n, repNorm, rpnStack)
                    buf.rawB[n] = rpnB.evaluateCpu(matrix, n, repNorm, rpnStack)
                }
            }

            val salt = deterministicStepSalt(step.dayKey, step.horaire)
            val pickedCount = selectFromRawBuffers(genome, salt, buf).first
            var hits = 0
            val actualMask = step.actualMask
            for (i in 0 until pickedCount) {
                val n = buf.picked[i]
                buf.sameDayCounts[n] = buf.sameDayCounts[n] + 1
                if (actualMask[n]) hits++
            }

            totalHits += hits
            matchesList.add(hits)
            if (hits == 0) zeroCount++
            if (hits >= 2) twoPlus++
            if (hits >= 3) threePlus++
            if (hits >= 4) fourPlus++
            if (hits >= 5) fiveMatch++
            brierSum += ScientificMath.computeBrierScoreMask(buf.marginalProbs, actualMask)

            if (hits == 0) {
                zeroStreak++
                if (zeroStreak > maxZeroStreak) maxZeroStreak = zeroStreak
            } else zeroStreak = 0

            if (sDayEpoch != curDayEpoch) {
                if (curDayEpoch != Long.MIN_VALUE && curDayTwoPlus >= 3 && curDayPrevHit >= 2) {
                    qualDays++
                    consecQual = if (lastQualDayEpoch != Long.MIN_VALUE && curDayEpoch == lastQualDayEpoch + 1L) {
                        consecQual + 1
                    } else 1
                    if (consecQual > maxConsecQual) maxConsecQual = consecQual
                    lastQualDayEpoch = curDayEpoch
                } else if (curDayEpoch != Long.MIN_VALUE) consecQual = 0
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
            consecQual = if (lastQualDayEpoch != Long.MIN_VALUE && curDayEpoch == lastQualDayEpoch + 1L) {
                consecQual + 1
            } else 1
            if (consecQual > maxConsecQual) maxConsecQual = consecQual
        }

        return StageEvalStats(
            tested = stepCount,
            twoPlusCount = twoPlus,
            qualifyingDays = qualDays,
            maxConsecutiveQualifyingDays = maxConsecQual,
            maxZeroStreak = maxZeroStreak,
            avgMatches = totalHits.toFloat() / stepCount,
            brierScore = brierSum / stepCount,
            matchesList = matchesList,
            causalVerified = allCausal,
            zeroMatchCount = zeroCount,
            threePlusCount = threePlus,
            fourPlusCount = fourPlus,
            fiveMatchCount = fiveMatch
        )
    }

    private fun selectFromRawBuffers(
        genome: SymbolicGenome,
        stepSalt: Long,
        buf: StageScratchBuffers
    ): Pair<Int, Int> {
        softmaxScaled5Into(buf.rawA, genome.samplingTemp, buf.probsA)
        softmaxScaled5Into(buf.rawB, genome.samplingTemp, buf.probsB)
        val alphaA = genome.primarySlots / 5.0
        val alphaB = 1.0 - alphaA
        buf.marginalProbs[0] = 0.0
        for (n in 1..90) {
            buf.marginalProbs[n] = (alphaA * buf.probsA[n] + alphaB * buf.probsB[n]).coerceIn(1e-6, 0.999)
        }

        val noiseScale = genome.exploreNoise * 0.08
        if (noiseScale <= 1e-6) {
            for (n in 1..90) {
                buf.noisyA[n] = buf.probsA[n]
                buf.noisyB[n] = buf.probsB[n]
            }
        } else {
            var prngState = genome.seed xor (stepSalt * 2_654_435_761L)
            for (n in 1..90) {
                prngState = splitMix64Next(prngState)
                buf.noisyA[n] = buf.probsA[n] + splitMixDouble(prngState) * noiseScale
            }
            for (n in 1..90) {
                prngState = splitMix64Next(prngState)
                buf.noisyB[n] = buf.probsB[n] + splitMixDouble(prngState) * noiseScale
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
        precompiledA: CompiledRpnProgram? = null,
        precompiledB: CompiledRpnProgram? = null
    ): Triple<List<Int>, List<Int>, Pair<List<Int>, DoubleArray>> {
        val rpnA = precompiledA ?: SymbolicFormula.compileToRpn(genome.primaryExpr)
        val rpnB = precompiledB ?: SymbolicFormula.compileToRpn(genome.secondaryExpr)
        val rpnStack = buf.rpnStack
        for (n in 1..90) {
            val repNorm = (sameDayCounts[n] / 5.0).coerceIn(0.0, 1.0)
            buf.rawA[n] = rpnA.evaluateCpu(matrix, n, repNorm, rpnStack)
            buf.rawB[n] = rpnB.evaluateCpu(matrix, n, repNorm, rpnStack)
        }
        val (totalPicked, countA) = selectFromRawBuffers(genome, stepSalt, buf)
        val methodA = ArrayList<Int>(countA)
        for (i in 0 until countA) methodA.add(buf.picked[i])
        val methodB = ArrayList<Int>((totalPicked - countA).coerceAtLeast(0))
        for (i in countA until totalPicked) methodB.add(buf.picked[i])
        val predicted = ArrayList<Int>(totalPicked)
        for (i in 0 until totalPicked) predicted.add(buf.picked[i])
        predicted.sort()
        methodA.sort()
        methodB.sort()
        return Triple(predicted, methodA, methodB to buf.marginalProbs.copyOf())
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
            for (n in 1..90) outProbs[n] = ((5.0 * outProbs[n]) / sum).coerceIn(1e-6, 0.999)
        } else {
            for (n in 1..90) outProbs[n] = 5.0 / 90.0
        }
        var currentSum = 0.0
        for (n in 1..90) currentSum += outProbs[n]
        val factor = 5.0 / currentSum.coerceAtLeast(1e-9)
        for (n in 1..90) outProbs[n] = (outProbs[n] * factor).coerceIn(1e-6, 0.999)
    }

    fun buildValidatedBacktest(
        genome: SymbolicGenome,
        cache: FourStageCausalCache,
        generationsExecuted: Int,
        experimentsCount: Int
    ): BacktestResult {
        val steps = cache.steps
        if (steps.isEmpty()) return ScientificMath.emptyBacktestResult(cache.periodLabel)

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

            val salt = deterministicStepSalt(step.dayKey, step.horaire)
            val (predicted, methodA, bAndProbs) = selectNumbersWithGenome(
                genome = genome,
                matrix = step.rawMatrix,
                sameDayCounts = buf.sameDayCounts,
                stepSalt = salt,
                buf = buf,
                precompiledA = rpnA,
                precompiledB = rpnB
            )
            val (methodB, marginalProbs) = bAndProbs
            for (n in predicted) buf.sameDayCounts[n] = buf.sameDayCounts[n] + 1

            val matched = predicted.filter { it in step.actualSet }
            val stepBrier = ScientificMath.computeBrierScore(marginalProbs, step.actualSet)
            val stepLogLoss = ScientificMath.computeBernoulliLogLoss(marginalProbs, step.actualSet)

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
                    predictedNumbers = predicted,
                    actualNumbers = step.actualNumbers,
                    matchedNumbers = matched,
                    matchCount = matched.size,
                    expectedMatchesEstimate = marginalProbs.sum().toFloat(),
                    stepBrierScore = stepBrier,
                    stepBernoulliLogLoss = stepLogLoss,
                    methodANumbers = methodA,
                    methodBNumbers = methodB
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
        chronological: List<LotoDraw>,
        genome: SymbolicGenome,
        startIndex: Int,
        endIndexExclusive: Int,
        periodLabel: String
    ): BacktestResult {
        val safeStart = startIndex.coerceAtLeast(1)
        val safeEnd = endIndexExclusive.coerceAtMost(chronological.size)
        if (safeStart >= safeEnd) return ScientificMath.emptyBacktestResult(periodLabel)

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
            val actualNumbers = SymbolicFormula.extractDistinctBalls(targetDraw).sorted()
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
            val salt = deterministicStepSalt(dayKey, horaire)
            val (predicted, methodA, bAndProbs) = selectNumbersWithGenome(
                genome = genome,
                matrix = rawMatrix,
                sameDayCounts = buf.sameDayCounts,
                stepSalt = salt,
                buf = buf,
                precompiledA = rpnA,
                precompiledB = rpnB
            )
            val (methodB, marginalProbs) = bAndProbs
            for (n in predicted) buf.sameDayCounts[n] = buf.sameDayCounts[n] + 1

            val actualSet = actualNumbers.toSet()
            val matched = predicted.filter { it in actualSet }
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
                    predictedNumbers = predicted,
                    actualNumbers = actualNumbers,
                    matchedNumbers = matched,
                    matchCount = matched.size,
                    expectedMatchesEstimate = marginalProbs.sum().toFloat(),
                    stepBrierScore = ScientificMath.computeBrierScore(marginalProbs, actualSet),
                    stepBernoulliLogLoss = ScientificMath.computeBernoulliLogLoss(marginalProbs, actualSet),
                    methodANumbers = methodA,
                    methodBNumbers = methodB
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
     * - Si un registre approuvé existe pour (strategySignature, periodKey), aucune étape historique déjà approuvée n'est modifiée.
     * - Seuls les nouveaux tirages récemment arrivés (> dernier tirage figé) sont calculés et ajoutés à la suite.
     * - Réconcilie aussi les prédictions Live enregistrées pour prouver 0 dérive Live ↔ Historique.
     */
    fun mergeAppendOnlyApprovedBacktest(
        genome: SymbolicGenome,
        cache: FourStageCausalCache,
        existingLedger: ApprovedBacktestLedgerRecord?,
        generationsExecuted: Int,
        experimentsCount: Int
    ): Pair<BacktestResult, ApprovedBacktestLedgerRecord> {
        val periodKey = cache.periodLabel
        val strategySig = genome.describeCompact()
        val rawSteps = cache.steps
        val frozenChronological = if (existingLedger != null &&
            existingLedger.strategySignature == strategySig &&
            existingLedger.periodKey == periodKey
        ) {
            deserializeApprovedSteps(existingLedger.frozenStepsJson)
                .sortedBy { it.targetTimestamp }
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

        // Pré-charger les sameDayCounts avec les dernières étapes figées de la journée active si nécessaire
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
            val salt = deterministicStepSalt(step.dayKey, step.horaire)
            val (predicted, methodA, bAndProbs) = selectNumbersWithGenome(
                genome = genome,
                matrix = step.rawMatrix,
                sameDayCounts = buf.sameDayCounts,
                stepSalt = salt,
                buf = buf,
                precompiledA = rpnA,
                precompiledB = rpnB
            )
            val (methodB, marginalProbs) = bAndProbs
            for (n in predicted) buf.sameDayCounts[n] = buf.sameDayCounts[n] + 1

            val matchingPending = pendingList.firstOrNull {
                it.dayKey == step.dayKey && it.horaire.equals(step.horaire, ignoreCase = true)
            }
            if (matchingPending != null && matchingPending.predictedNumbers.sorted() == predicted.sorted()) {
                liveMatchedCount++
            }

            val matched = predicted.filter { it in step.actualSet }
            val stepBrier = ScientificMath.computeBrierScore(marginalProbs, step.actualSet)
            val stepLogLoss = ScientificMath.computeBernoulliLogLoss(marginalProbs, step.actualSet)

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
                    predictedNumbers = predicted,
                    actualNumbers = step.actualNumbers,
                    matchedNumbers = matched,
                    matchCount = matched.size,
                    expectedMatchesEstimate = marginalProbs.sum().toFloat(),
                    stepBrierScore = stepBrier,
                    stepBernoulliLogLoss = stepLogLoss,
                    methodANumbers = methodA,
                    methodBNumbers = methodB
                )
            )
            appendedCount++
        }

        val reindexed = combinedChronological.mapIndexed { i, st -> st.copy(stepIndex = i + 1) }
        val remainingPending = pendingList.filter { p ->
            reindexed.none { it.dayKey == p.dayKey && it.horaire.equals(p.horaire, ignoreCase = true) }
        }.takeLast(24)

        val ledgerKey = buildLedgerKey(strategySig, periodKey)
        val updatedLedger = ApprovedBacktestLedgerRecord(
            ledgerKey = ledgerKey,
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
        ledger: ApprovedBacktestLedgerRecord,
        prediction: PredictionResult,
        targetTimestamp: Long
    ): ApprovedBacktestLedgerRecord {
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

    fun buildLedgerKey(strategySignature: String, periodKey: String): String =
        "${periodKey}__${strategySignature.hashCode().toUInt()}"

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
        if (chronologicalSteps.isEmpty()) return ScientificMath.emptyBacktestResult(periodLabel)

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

        val (avgLo, avgHi) = ScientificMath.mean95Interval(matchList)
        val (rateLo, rateHi) = ScientificMath.wilson95Interval(partialMatches, totalTested)

        return BacktestResult(
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
            dailySummaries = dailySummaries,
            horaireBreakdown = horairePerf,
            steps = chronologicalSteps.reversed(),
            maxConsecutiveQualifyingDays = maxConsecQual,
            maxZeroMatchStreak = maxZeroStreak,
            currentZeroMatchStreak = zeroStreak,
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
        cleanHistory: List<LotoDraw>,
        targetHoraire: String,
        validatedBacktest: BacktestResult,
        targetTimestamp: Long? = null
    ): PredictionResult {
        val chronological = cleanHistory
            .distinctBy { it.uniqueId }
            .sortedWith(compareBy<LotoDraw> { it.timestamp }.thenBy { it.uniqueId })
        val normalizedH = targetHoraire.trim().uppercase(Locale.ROOT)
        val effectiveTargetTs = targetTimestamp ?: chronological.lastOrNull()?.timestamp ?: System.currentTimeMillis()
        val targetDayKey = LotoDateUtils.formatDayKey(effectiveTargetTs)

        // Reconstruction causale exacte des compteurs sameDayCounts sur targetDayKey avant effectiveTargetTs
        // -> Garantit 100 % d'identité entre prédiction Live et futur Backtest historique
        val sameDayCounts = IntArray(91)
        val rpnA = SymbolicFormula.compileToRpn(genome.primaryExpr)
        val rpnB = SymbolicFormula.compileToRpn(genome.secondaryExpr)
        if (rpnA.usesSameDayRep || rpnB.usesSameDayRep) {
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
                            val dHoraire = LotoDateUtils.normalizeHoraire(d)
                            val m = SymbolicFormula.extractRawMatrix(past, dHoraire)
                            val (predBefore, _, _) = selectNumbersWithGenome(
                                genome = genome,
                                matrix = m,
                                sameDayCounts = sameDayCounts,
                                stepSalt = deterministicStepSalt(targetDayKey, dHoraire),
                                buf = buf,
                                precompiledA = rpnA,
                                precompiledB = rpnB
                            )
                            for (n in predBefore) if (n in 1..90) sameDayCounts[n] = sameDayCounts[n] + 1
                        }
                    }
                }
            }
        }

        val matrix = SymbolicFormula.extractRawMatrix(chronological, normalizedH)
        val salt = deterministicStepSalt(targetDayKey, normalizedH)
        val (predicted, methodA, bAndProbs) = selectNumbersWithGenome(
            genome = genome,
            matrix = matrix,
            sameDayCounts = sameDayCounts,
            stepSalt = salt,
            precompiledA = rpnA,
            precompiledB = rpnB
        )
        val (methodB, marginalProbs) = bAndProbs

        val marginalSum = (1..90).sumOf { marginalProbs[it] }.toFloat()
        val randomBaselineProb = 5.0 / 90.0
        val sortedAll = (1..90).sortedWith(compareByDescending<Int> { marginalProbs[it] }.thenBy { it })
        val rankByNumber = IntArray(91)
        sortedAll.forEachIndexed { idx, num -> rankByNumber[num] = idx + 1 }
        val maxProb = marginalProbs[sortedAll.first()].coerceAtLeast(randomBaselineProb)
        val methodASet = methodA.toSet()

        fun buildConfidence(num: Int): PredictedNumberConfidence {
            val p = marginalProbs[num]
            val probPct = (p * 100.0).toFloat().coerceIn(0f, 100f)
            val relPct = ((p / maxProb) * 100.0).toFloat().coerceIn(0f, 100f)
            val lift = (p / randomBaselineProb).toFloat()
            val branch = when {
                num in methodASet -> "Formule A"
                num in methodB -> "Formule B"
                else -> "Candidat"
            }
            val isGap = matrix.get(RawFeature.GAP_NORM.ordinal, num) >= 0.08
            return PredictedNumberConfidence(num, probPct, relPct, lift, rankByNumber[num], branch, isGap)
        }

        val predictedConfidences = predicted.map(::buildConfidence)
            .sortedWith(compareByDescending<PredictedNumberConfidence> { it.marginalProbabilityPercent }.thenBy { it.number })
        val topCandidates = sortedAll.take(8).map(::buildConfidence)

        return PredictionResult(
            predictedDraw = predicted,
            expectedMatchesEstimate = predicted.sumOf { marginalProbs[it] }.toFloat(),
            calibratedProbAtLeastOne = ScientificMath.calibratedProbAtLeastOne(predicted, marginalProbs),
            trainedOnCount = chronological.size,
            methodName = "Auto-Recherche Symbolique PC",
            horaire = normalizedH,
            marginalSumCheck = marginalSum,
            auditNote = "Formule symbolique unifiée (0 dérive Live ↔ Backtest • <50% 0-boule)",
            discoveredStrategyName = genome.describeCompact(),
            coldNumbersCount = predicted.count { matrix.get(RawFeature.GAP_NORM.ordinal, it) >= 0.08 },
            searchRoundsExecuted = genome.generation,
            totalStrategiesSearched = validatedBacktest.totalStrategiesSearched,
            objectiveFourDaysAchieved = validatedBacktest.objectiveFourDaysAchieved,
            bestCausalStreakFound = validatedBacktest.maxConsecutiveQualifyingDays,
            methodANumbers = methodA,
            methodBNumbers = methodB,
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
                    put("predictedNumbers", buildJsonArray { s.predictedNumbers.forEach { add(kotlinx.serialization.json.JsonPrimitive(it)) } })
                    put("actualNumbers", buildJsonArray { s.actualNumbers.forEach { add(kotlinx.serialization.json.JsonPrimitive(it)) } })
                    put("matchedNumbers", buildJsonArray { s.matchedNumbers.forEach { add(kotlinx.serialization.json.JsonPrimitive(it)) } })
                    put("matchCount", s.matchCount)
                    put("expectedMatchesEstimate", s.expectedMatchesEstimate)
                    put("stepBrierScore", s.stepBrierScore)
                    put("stepBernoulliLogLoss", s.stepBernoulliLogLoss)
                    put("methodANumbers", buildJsonArray { s.methodANumbers.forEach { add(kotlinx.serialization.json.JsonPrimitive(it)) } })
                    put("methodBNumbers", buildJsonArray { s.methodBNumbers.forEach { add(kotlinx.serialization.json.JsonPrimitive(it)) } })
                })
            }
        }
        return arr.toString()
    }

    private fun deserializeApprovedSteps(json: String): List<BacktestStepResult> {
        if (json.isBlank()) return emptyList()
        return runCatching {
            val arr = Json.parseToJsonElement(json).jsonArray
            arr.mapNotNull { el ->
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
                    put("predictedNumbers", buildJsonArray { p.predictedNumbers.forEach { add(kotlinx.serialization.json.JsonPrimitive(it)) } })
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
        return counts.entries.sortedByDescending { it.value }.take(8)
            .map { (feat, c) -> "${feat.symbol} (${feat.name})" to (c.toFloat() / total.toFloat()) }
    }

    /**
     * Synthétiseur causal STRICTEMENT sur Etape 1 (discoverySteps UNIQUEMENT — zéro accès Rolling/Conf/Scellé) :
     * Mesure l'effet empirique standardisé (z-lift) de chaque variable brute sur les boules réellement sorties
     * en Découverte + l'erreur résiduelle des boules manquées, puis génère des formules symboliques AST inédites.
     */
    fun synthesizeCausalDiscoveryGenomes(
        discoverySteps: List<CachedRawStep>,
        generation: Int,
        sessionSalt: Long,
        seenSignatures: Set<String> = emptySet(),
        maxCount: Int = 20
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

        // Second passe : erreur résiduelle (boules réelles manquées par le top-5 du signal #1/#2/#3)
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
        val combos = listOf(2, 3, 4, 5, 6)
        val unaries = listOf<UnaryOp?>(null, UnaryOp.TANH, UnaryOp.SIGMOID, UnaryOp.SQRT_ABS)
        val temps = listOf(0.28, 0.45, 0.72)

        var idx = 0
        for (kCount in combos) {
            for (uOp in unaries) {
                for (slotSplit in listOf(2, 3, 4, 1)) {
                    if (out.size >= maxCount) break
                    val pList = topSignals.take(minOf(kCount, topSignals.size))
                    val sList = residualRanked.take(minOf(3, residualRanked.size))
                    val temp = temps[idx % temps.size]
                    val primBase = buildLinearSum(pList, 1.0, uOp)
                    val primExpr = if (pList.size >= 2 && idx % 2 == 1) {
                        ExprNode.Binary(
                            BinaryOp.TRIANG_BLEND,
                            primBase,
                            ExprNode.Binary(BinaryOp.MUL, weightedTerm(pList[0], 0.8), ExprNode.Var(pList[1].feature))
                        )
                    } else primBase
                    val secBase = buildLinearSum(sList, 1.0, if (idx % 2 == 0) UnaryOp.TANH else UnaryOp.SIGMOID)

                    val genome = SymbolicGenome(
                        id = "pc_causal_g${generation}_$idx",
                        generation = generation.coerceAtLeast(1),
                        parentIds = listOf("causal_disc_lift"),
                        primaryExpr = SymbolicFormula.simplify(primExpr),
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

    internal fun effectiveMultipleTestFactor(confTests: Int): Double {
        val c = confTests.coerceAtLeast(1).toDouble()
        return minOf(c, 1.0 + 2.5 * kotlin.math.ln(c))
    }

    private fun fmt3(v: Float): String = String.format(Locale.FRANCE, "%.3f", v)
    private fun fmt4(v: Double): String = String.format(Locale.FRANCE, "%.4f", v)
    private fun round2(v: Double): Double = kotlin.math.round(v * 100.0) / 100.0
}
