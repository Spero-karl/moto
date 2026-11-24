package com.lotobj.pc.engine

import com.lotobj.pc.model.BacktestResult
import com.lotobj.pc.model.BinaryOp
import com.lotobj.pc.model.ComputeMode
import com.lotobj.pc.model.ExprNode
import com.lotobj.pc.model.RawFeature
import com.lotobj.pc.model.RejectedFormulaSummary
import com.lotobj.pc.model.SymbolicGenome
import com.lotobj.pc.model.UnaryOp
import java.util.Locale
import java.util.concurrent.ForkJoinPool
import kotlin.math.round
import kotlin.random.Random

/**
 * Domaines de théorèmes mathématiques exploités par l'Usine à Stratégies (Strategy Factory).
 */
enum class MathTheoremDomain(val label: String, val shortLabel: String) {
    MODULAR_ARITHMETIC("Arithmétique Modulaire (Base 7 / Lucky 9 / Cube90)", "Modulaire"),
    PROBABILITY_DENSITY("Densité de Probabilité (Weibull / Z-Score / Extrapol.)", "Proba & Densité"),
    TRIANGULATION("Triangulation & Géométrie (Grille 9×10 / Pyramides)", "Triangulation"),
    HYBRID_SYNTHESIS("Synthèse Hybride Multi-Théorèmes", "Hybride")
}

/**
 * Contrat Strategy Factory : chaque fabrique concrète synthétise à l'exécution des formules
 * symboliques inédites dans son domaine mathématique puis les assemble en SymbolicGenome testable.
 */
interface MathFormulaBuilder {
    val domain: MathTheoremDomain
    val builderName: String
    val primaryFeatures: List<RawFeature>
    val secondaryFeatures: List<RawFeature>
    val preferredUnaryOps: List<UnaryOp>
    val preferredBinaryOps: List<BinaryOp>

    fun buildPrimaryExpr(rng: Random, complexityDepth: Int = 4): ExprNode
    fun buildSecondaryExpr(rng: Random, complexityDepth: Int = 3): ExprNode

    fun createGenome(
        id: String,
        generation: Int,
        seed: Long,
        complexityDepth: Int = 4
    ): SymbolicGenome {
        val rng = Random(seed)
        val exprA = SymbolicFormula.simplify(buildPrimaryExpr(rng, complexityDepth))
        val exprB = SymbolicFormula.simplify(buildSecondaryExpr(rng, (complexityDepth - 1).coerceAtLeast(2)))
        return SymbolicGenome(
            id = id,
            generation = generation,
            parentIds = listOf("factory:${domain.name}"),
            primaryExpr = ensureHasFeature(exprA, primaryFeatures, rng),
            secondaryExpr = ensureHasFeature(exprB, secondaryFeatures, rng),
            primarySlots = rng.nextInt(1, 5),
            samplingTemp = round2(0.28 + rng.nextDouble() * 0.92),
            exploreNoise = 0.0,
            seed = seed
        )
    }

    private fun ensureHasFeature(node: ExprNode, pool: List<RawFeature>, rng: Random): ExprNode {
        return if (SymbolicFormula.usedFeatures(node).isEmpty()) {
            ExprNode.Binary(
                BinaryOp.MUL,
                ExprNode.Const(round3(0.4 + rng.nextDouble() * 1.4)),
                ExprNode.Var(pool[rng.nextInt(pool.size)])
            )
        } else {
            node
        }
    }

    private fun round2(v: Double): Double = round(v * 100.0) / 100.0
    private fun round3(v: Double): Double = round(v * 1000.0) / 1000.0
}

/**
 * Fabrique 1 : Arithmétique modulaire, congruences Base 7, racine numérique Lucky 9,
 * classes Cube90 (n^3 + 9 mod 90) et échos affines (7k + 9 mod 90).
 */
class ModularArithmeticStrategyFactory : MathFormulaBuilder {
    override val domain = MathTheoremDomain.MODULAR_ARITHMETIC
    override val builderName = "Usine Congruence & Arithmétique Modulaire"
    override val primaryFeatures = listOf(
        RawFeature.BASE7_NORM,
        RawFeature.MOD7_RES_DENSITY,
        RawFeature.LUCKY_ROOT_NORM,
        RawFeature.MOD10_UNIT_NORM,
        RawFeature.MOD11_HARMONIC,
        RawFeature.CONGRUENCE_CUBE90,
        RawFeature.AFFINE_MOD90_RESONANCE,
        RawFeature.X_NORM
    )
    override val secondaryFeatures = listOf(
        RawFeature.MOD7_RES_DENSITY,
        RawFeature.CONGRUENCE_CUBE90,
        RawFeature.AFFINE_MOD90_RESONANCE,
        RawFeature.HORAIRE_FREQ,
        RawFeature.GAP_NORM,
        RawFeature.LAST_HIT
    )
    override val preferredUnaryOps = listOf(
        UnaryOp.LUCKY9_WAVE, UnaryOp.COS_PI, UnaryOp.SIN_PI, UnaryOp.SIGMOID, UnaryOp.ABS, UnaryOp.STEP_POS
    )
    override val preferredBinaryOps = listOf(
        BinaryOp.MOD_SAFE, BinaryOp.ADD, BinaryOp.MUL, BinaryOp.EXP_DIFF, BinaryOp.MAX, BinaryOp.ATAN2_PI
    )

    override fun buildPrimaryExpr(rng: Random, complexityDepth: Int): ExprNode {
        fun f() = ExprNode.Var(primaryFeatures[rng.nextInt(primaryFeatures.size)])
        fun c(min: Double = -2.0, max: Double = 2.5) = ExprNode.Const(round(1000.0 * (min + rng.nextDouble() * (max - min))) / 1000.0)
        val u1 = preferredUnaryOps[rng.nextInt(preferredUnaryOps.size)]
        val u2 = preferredUnaryOps[rng.nextInt(preferredUnaryOps.size)]
        val b1 = preferredBinaryOps[rng.nextInt(preferredBinaryOps.size)]

        val base = when (rng.nextInt(4)) {
            0 -> ExprNode.Binary(
                BinaryOp.ADD,
                ExprNode.Unary(u1, ExprNode.Binary(BinaryOp.MOD_SAFE, ExprNode.Binary(BinaryOp.MUL, c(0.6, 2.2), f()), c(0.5, 1.8))),
                ExprNode.Unary(u2, ExprNode.Binary(b1, f(), f()))
            )
            1 -> ExprNode.Binary(
                BinaryOp.MUL,
                ExprNode.Unary(UnaryOp.SIGMOID, ExprNode.Binary(BinaryOp.SUB, f(), f())),
                ExprNode.Binary(BinaryOp.ADD, ExprNode.Unary(UnaryOp.LUCKY9_WAVE, f()), ExprNode.Unary(UnaryOp.COS_PI, f()))
            )
            2 -> ExprNode.Binary(
                BinaryOp.EXP_DIFF,
                ExprNode.Binary(BinaryOp.MOD_SAFE, f(), c(0.4, 1.5)),
                ExprNode.Unary(u1, f())
            )
            else -> ExprNode.Binary(
                BinaryOp.ADD,
                ExprNode.Binary(BinaryOp.MUL, c(0.5, 2.0), f()),
                ExprNode.Binary(BinaryOp.SUB, ExprNode.Unary(u2, f()), ExprNode.Binary(BinaryOp.MUL, c(0.3, 1.4), f()))
            )
        }
        return if (complexityDepth >= 5) {
            ExprNode.Binary(BinaryOp.ADD, base, ExprNode.Unary(UnaryOp.TANH, f()))
        } else {
            base
        }
    }

    override fun buildSecondaryExpr(rng: Random, complexityDepth: Int): ExprNode {
        fun f() = ExprNode.Var(secondaryFeatures[rng.nextInt(secondaryFeatures.size)])
        val u = preferredUnaryOps[rng.nextInt(preferredUnaryOps.size)]
        val b = preferredBinaryOps[rng.nextInt(preferredBinaryOps.size)]
        return ExprNode.Binary(b, ExprNode.Unary(u, f()), f())
    }
}

/**
 * Fabrique 2 : Densité de probabilité bayésienne, limite de Weibull sur les écarts,
 * score Z binomial et vélocité d'extrapolation multi-horizons.
 */
class ProbabilityDensityStrategyFactory : MathFormulaBuilder {
    override val domain = MathTheoremDomain.PROBABILITY_DENSITY
    override val builderName = "Usine Densité Probabiliste & Limites"
    override val primaryFeatures = listOf(
        RawFeature.LIMIT_HAZARD,
        RawFeature.PROB_ZSCORE,
        RawFeature.EXTRAPOL_VELOCITY,
        RawFeature.FREQ_GLOBAL,
        RawFeature.FREQ_W5,
        RawFeature.FREQ_W15,
        RawFeature.FREQ_W40,
        RawFeature.GAP_NORM,
        RawFeature.PREV_GAP_NORM,
        RawFeature.HORAIRE_FREQ
    )
    override val secondaryFeatures = listOf(
        RawFeature.LIMIT_HAZARD,
        RawFeature.PROB_ZSCORE,
        RawFeature.EXTRAPOL_VELOCITY,
        RawFeature.GAP_NORM,
        RawFeature.HORAIRE_FREQ,
        RawFeature.CO_LAST_RAW
    )
    override val preferredUnaryOps = listOf(
        UnaryOp.SIGMOID, UnaryOp.TANH, UnaryOp.SOFTPLUS, UnaryOp.EXP_CLIP, UnaryOp.LOG1P_ABS, UnaryOp.INV_SAFE
    )
    override val preferredBinaryOps = listOf(
        BinaryOp.ADD, BinaryOp.SUB, BinaryOp.MUL, BinaryOp.DIV_SAFE, BinaryOp.MEAN2, BinaryOp.EXP_DIFF
    )

    override fun buildPrimaryExpr(rng: Random, complexityDepth: Int): ExprNode {
        fun f() = ExprNode.Var(primaryFeatures[rng.nextInt(primaryFeatures.size)])
        fun c(min: Double = 0.2, max: Double = 2.4) = ExprNode.Const(round(1000.0 * (min + rng.nextDouble() * (max - min))) / 1000.0)
        val u1 = preferredUnaryOps[rng.nextInt(preferredUnaryOps.size)]
        val u2 = preferredUnaryOps[rng.nextInt(preferredUnaryOps.size)]
        val b1 = preferredBinaryOps[rng.nextInt(preferredBinaryOps.size)]

        val core = when (rng.nextInt(4)) {
            0 -> ExprNode.Binary(
                BinaryOp.MUL,
                ExprNode.Unary(UnaryOp.SIGMOID, ExprNode.Binary(BinaryOp.ADD, ExprNode.Binary(BinaryOp.MUL, c(), f()), f())),
                ExprNode.Unary(UnaryOp.SOFTPLUS, ExprNode.Binary(BinaryOp.SUB, f(), f()))
            )
            1 -> ExprNode.Binary(
                BinaryOp.DIV_SAFE,
                ExprNode.Binary(BinaryOp.ADD, ExprNode.Unary(u1, f()), ExprNode.Unary(UnaryOp.TANH, f())),
                ExprNode.Unary(UnaryOp.LOG1P_ABS, f())
            )
            2 -> ExprNode.Binary(
                BinaryOp.SUB,
                ExprNode.Unary(UnaryOp.EXP_CLIP, ExprNode.Binary(BinaryOp.MUL, c(0.4, 1.5), f())),
                ExprNode.Unary(u2, ExprNode.Binary(b1, f(), f()))
            )
            else -> ExprNode.Binary(
                BinaryOp.MEAN2,
                ExprNode.Unary(UnaryOp.SOFTPLUS, f()),
                ExprNode.Binary(BinaryOp.EXP_DIFF, f(), f())
            )
        }
        return if (complexityDepth >= 5) {
            ExprNode.Binary(BinaryOp.ADD, core, ExprNode.Unary(UnaryOp.SIGMOID, f()))
        } else {
            core
        }
    }

    override fun buildSecondaryExpr(rng: Random, complexityDepth: Int): ExprNode {
        fun f() = ExprNode.Var(secondaryFeatures[rng.nextInt(secondaryFeatures.size)])
        val u = preferredUnaryOps[rng.nextInt(preferredUnaryOps.size)]
        return ExprNode.Binary(BinaryOp.MEAN2, ExprNode.Unary(u, f()), ExprNode.Unary(UnaryOp.SIGMOID, f()))
    }
}

/**
 * Fabrique 3 : Triangulation géométrique (différences |a-b|, sommes modulaires,
 * proximité/cavalier grille 9×10, co-occurrences de tirages précédents).
 */
class TriangulationStrategyFactory : MathFormulaBuilder {
    override val domain = MathTheoremDomain.TRIANGULATION
    override val builderName = "Usine Triangulation & Géométrie 9×10"
    override val primaryFeatures = listOf(
        RawFeature.TRIANG_PYRAMID_DIFF,
        RawFeature.GRID_9X10_PROXIMITY,
        RawFeature.CO_LAST_RAW,
        RawFeature.LAST_HIT,
        RawFeature.LAG2_HIT,
        RawFeature.X_NORM,
        RawFeature.AFFINE_MOD90_RESONANCE
    )
    override val secondaryFeatures = listOf(
        RawFeature.TRIANG_PYRAMID_DIFF,
        RawFeature.GRID_9X10_PROXIMITY,
        RawFeature.CO_LAST_RAW,
        RawFeature.GAP_NORM,
        RawFeature.LIMIT_HAZARD
    )
    override val preferredUnaryOps = listOf(
        UnaryOp.SQRT_ABS, UnaryOp.ABS, UnaryOp.CBRT, UnaryOp.SQUARE, UnaryOp.TANH, UnaryOp.SIN_PI
    )
    override val preferredBinaryOps = listOf(
        BinaryOp.TRIANG_BLEND, BinaryOp.HYPOT_NORM, BinaryOp.ATAN2_PI, BinaryOp.MEAN2, BinaryOp.MAX, BinaryOp.MIN
    )

    override fun buildPrimaryExpr(rng: Random, complexityDepth: Int): ExprNode {
        fun f() = ExprNode.Var(primaryFeatures[rng.nextInt(primaryFeatures.size)])
        fun c(min: Double = 0.25, max: Double = 2.0) = ExprNode.Const(round(1000.0 * (min + rng.nextDouble() * (max - min))) / 1000.0)
        val u1 = preferredUnaryOps[rng.nextInt(preferredUnaryOps.size)]
        val u2 = preferredUnaryOps[rng.nextInt(preferredUnaryOps.size)]

        val core = when (rng.nextInt(4)) {
            0 -> ExprNode.Binary(
                BinaryOp.TRIANG_BLEND,
                ExprNode.Binary(BinaryOp.HYPOT_NORM, f(), f()),
                ExprNode.Unary(u1, ExprNode.Binary(BinaryOp.MUL, c(), f()))
            )
            1 -> ExprNode.Binary(
                BinaryOp.ADD,
                ExprNode.Unary(UnaryOp.SQRT_ABS, ExprNode.Binary(BinaryOp.TRIANG_BLEND, f(), f())),
                ExprNode.Binary(BinaryOp.ATAN2_PI, f(), f())
            )
            2 -> ExprNode.Binary(
                BinaryOp.HYPOT_NORM,
                ExprNode.Unary(u1, f()),
                ExprNode.Unary(u2, ExprNode.Binary(BinaryOp.SUB, f(), f()))
            )
            else -> ExprNode.Binary(
                BinaryOp.MAX,
                ExprNode.Binary(BinaryOp.TRIANG_BLEND, f(), f()),
                ExprNode.Binary(BinaryOp.MEAN2, f(), ExprNode.Unary(UnaryOp.CBRT, f()))
            )
        }
        return if (complexityDepth >= 5) {
            ExprNode.Binary(BinaryOp.ADD, core, ExprNode.Unary(UnaryOp.TANH, f()))
        } else {
            core
        }
    }

    override fun buildSecondaryExpr(rng: Random, complexityDepth: Int): ExprNode {
        fun f() = ExprNode.Var(secondaryFeatures[rng.nextInt(secondaryFeatures.size)])
        val b = preferredBinaryOps[rng.nextInt(preferredBinaryOps.size)]
        return ExprNode.Binary(b, ExprNode.Unary(UnaryOp.SQRT_ABS, f()), f())
    }
}

/**
 * Fabrique 4 : Synthèse hybride croisant Arithmétique Modulaire + Densité Probabiliste + Triangulation.
 */
class HybridSynthesisStrategyFactory(
    private val modularFactory: ModularArithmeticStrategyFactory = ModularArithmeticStrategyFactory(),
    private val probabilityFactory: ProbabilityDensityStrategyFactory = ProbabilityDensityStrategyFactory(),
    private val triangulationFactory: TriangulationStrategyFactory = TriangulationStrategyFactory()
) : MathFormulaBuilder {
    override val domain = MathTheoremDomain.HYBRID_SYNTHESIS
    override val builderName = "Usine Hybride (Modulaire × Proba × Triangulation)"
    override val primaryFeatures = SymbolicFormula.GPU_CAPABLE_FEATURES
    override val secondaryFeatures = SymbolicFormula.GPU_CAPABLE_FEATURES
    override val preferredUnaryOps = UnaryOp.entries
    override val preferredBinaryOps = BinaryOp.entries

    override fun buildPrimaryExpr(rng: Random, complexityDepth: Int): ExprNode {
        val modTerm = modularFactory.buildPrimaryExpr(rng, 3)
        val probTerm = probabilityFactory.buildPrimaryExpr(rng, 3)
        val triTerm = triangulationFactory.buildPrimaryExpr(rng, 3)
        val combiner1 = when (rng.nextInt(3)) {
            0 -> BinaryOp.ADD
            1 -> BinaryOp.HYPOT_NORM
            else -> BinaryOp.TRIANG_BLEND
        }
        val combiner2 = when (rng.nextInt(3)) {
            0 -> BinaryOp.MEAN2
            1 -> BinaryOp.ADD
            else -> BinaryOp.EXP_DIFF
        }
        return ExprNode.Binary(
            combiner1,
            ExprNode.Unary(UnaryOp.TANH, modTerm),
            ExprNode.Binary(combiner2, probTerm, triTerm)
        )
    }

    override fun buildSecondaryExpr(rng: Random, complexityDepth: Int): ExprNode {
        val fA = triangulationFactory.buildSecondaryExpr(rng, 2)
        val fB = probabilityFactory.buildSecondaryExpr(rng, 2)
        return ExprNode.Binary(BinaryOp.MEAN2, fA, fB)
    }
}

data class StrategyFactorySearchReport(
    val domainLabel: String,
    val generatedCount: Int,
    val unseenEvaluatedCount: Int,
    val zeroCapPassedCount: Int,
    val optimalCandidates: List<GenomeCandidateEvaluation>,
    val rejectedCandidates: List<RejectedFormulaSummary>,
    val auditedWinner: GenomeCandidateEvaluation?,
    val domainBreakdown: Map<MathTheoremDomain, Int>,
    val evaluationsPerSec: Double,
    val gpuBatchMs: Double,
    val summaryNote: String
)

/**
 * Orchestrateur central du pattern Strategy Factory :
 * Crée à l'exécution des formules inédites (arithmétique modulaire, densité de probabilité,
 * triangulation, et hybrides), exclut automatiquement toute formule déjà testée, et exécute
 * leur banc d'essai causal 4 étapes sur Multi-GPU OpenCL + 100% CPU.
 */
object StrategyFactory {

    private val factories: Map<MathTheoremDomain, MathFormulaBuilder> = mapOf(
        MathTheoremDomain.MODULAR_ARITHMETIC to ModularArithmeticStrategyFactory(),
        MathTheoremDomain.PROBABILITY_DENSITY to ProbabilityDensityStrategyFactory(),
        MathTheoremDomain.TRIANGULATION to TriangulationStrategyFactory(),
        MathTheoremDomain.HYBRID_SYNTHESIS to HybridSynthesisStrategyFactory()
    )

    fun getFactory(domain: MathTheoremDomain): MathFormulaBuilder =
        factories[domain] ?: error("Fabrique inconnue pour $domain")

    fun listFactories(): List<MathFormulaBuilder> = MathTheoremDomain.entries.map { getFactory(it) }

    fun domainForParadigmIndex(paradigmIndex: Int): MathTheoremDomain {
        val idx = ((paradigmIndex % MathTheoremDomain.entries.size) + MathTheoremDomain.entries.size) %
            MathTheoremDomain.entries.size
        return MathTheoremDomain.entries[idx]
    }

    fun generateBatch(
        count: Int,
        generation: Int,
        sessionSalt: Long,
        seenSignatures: Set<String> = emptySet(),
        preferredDomain: MathTheoremDomain? = null,
        complexityDepth: Int = 4
    ): Pair<List<SymbolicGenome>, Map<MathTheoremDomain, Int>> {
        val out = ArrayList<SymbolicGenome>(count.coerceAtLeast(0))
        val localBatchSigs = HashSet<String>(count.coerceAtLeast(1) * 2)
        val countsByDomain = LinkedHashMap<MathTheoremDomain, Int>()
        val domains = MathTheoremDomain.entries

        for (i in 0 until count.coerceAtLeast(0)) {
            val domain = preferredDomain ?: domains[i % domains.size]
            val factory = getFactory(domain)
            var chosen: SymbolicGenome? = null
            for (attempt in 0 until 10) {
                val seed = sessionSalt * 131_071L + generation * 7_919L + i * 263L + attempt * 59L
                val cand = factory.createGenome(
                    id = "fac_${domain.name.take(3).lowercase(Locale.US)}_g${generation}_$i",
                    generation = generation,
                    seed = seed,
                    complexityDepth = complexityDepth
                )
                val sig = cand.describeCompact()
                if (sig !in seenSignatures && localBatchSigs.add(sig)) {
                    chosen = cand
                    break
                }
            }
            if (chosen != null) {
                out.add(chosen)
                countsByDomain[domain] = (countsByDomain[domain] ?: 0) + 1
            }
        }
        return out to countsByDomain
    }

    fun generateAndTestOptimalPatterns(
        cache: FourStageCausalCache,
        batchSize: Int = 256,
        generation: Int = 1,
        sessionSalt: Long = System.currentTimeMillis(),
        preferredDomain: MathTheoremDomain? = null,
        seenSignatures: MutableSet<String> = LinkedHashSet(),
        minObjectiveStreak: Int = 4,
        confirmationTestsCount: Int = 1,
        computeMode: ComputeMode = ComputeMode.GPU_AND_CPU_MAX,
        cpuPool: ForkJoinPool = ForkJoinPool.commonPool(),
        gpuAccelerator: GpuOpenClAccelerator? = null
    ): StrategyFactorySearchReport {
        val startNs = System.nanoTime()
        val domainLabel = preferredDomain?.label ?: "Toutes Familles (Modulaire + Proba + Triangulation + Hybride)"
        if (cache.steps.isEmpty()) {
            return StrategyFactorySearchReport(
                domainLabel = domainLabel,
                generatedCount = 0,
                unseenEvaluatedCount = 0,
                zeroCapPassedCount = 0,
                optimalCandidates = emptyList(),
                rejectedCandidates = emptyList(),
                auditedWinner = null,
                domainBreakdown = emptyMap(),
                evaluationsPerSec = 0.0,
                gpuBatchMs = 0.0,
                summaryNote = "Historique insuffisant pour tester les formules de l'Usine à Stratégies."
            )
        }

        val (facCandidates, breakdown) = generateBatch(
            count = batchSize.coerceIn(12, 4096),
            generation = generation,
            sessionSalt = sessionSalt,
            seenSignatures = seenSignatures,
            preferredDomain = preferredDomain,
            complexityDepth = 4
        )
        val ridgeSeed = if (cache.discoverySteps.isNotEmpty() && (preferredDomain == null || preferredDomain == MathTheoremDomain.HYBRID_SYNTHESIS)) {
            CausalAiLearner.fitSymbolicRidgeGenomes(
                discoverySteps = cache.discoverySteps,
                generation = generation,
                sessionSalt = sessionSalt,
                seenSignatures = seenSignatures,
                maxCount = minOf(12, maxOf(4, batchSize / 16))
            )
        } else emptyList()
        val candidates = (ridgeSeed + facCandidates).distinctBy { it.describeCompact() }.take(batchSize.coerceIn(12, 4096))
        PcDiscoveryEngine.recordSeenSignatures(seenSignatures, candidates)

        if (candidates.isEmpty()) {
            return StrategyFactorySearchReport(
                domainLabel = domainLabel,
                generatedCount = 0,
                unseenEvaluatedCount = 0,
                zeroCapPassedCount = 0,
                optimalCandidates = emptyList(),
                rejectedCandidates = emptyList(),
                auditedWinner = null,
                domainBreakdown = emptyMap(),
                evaluationsPerSec = 0.0,
                gpuBatchMs = 0.0,
                summaryNote = "Toutes les variations de ce lot figuraient déjà dans le registre d'exclusion."
            )
        }

        val compiled = candidates.map {
            SymbolicFormula.compileToRpn(it.primaryExpr) to SymbolicFormula.compileToRpn(it.secondaryExpr)
        }

        val useGpu = computeMode == ComputeMode.GPU_AND_CPU_MAX && gpuAccelerator != null && gpuAccelerator.isGpuReady
        val (discStatsList, discGpuMs, _) = if (gpuAccelerator != null) {
            PcDiscoveryEngine.evaluateStageHeterogeneous(
                stageKey = "${cache.cacheKey}_disc",
                steps = cache.discoverySteps,
                genomes = candidates,
                compiled = compiled,
                useGpu = useGpu,
                cpuPool = cpuPool,
                gpuAccelerator = gpuAccelerator
            )
        } else {
            val list = PcDiscoveryEngine.mapIndexedParallel(cpuPool, candidates.size) { idx ->
                PcDiscoveryEngine.evaluateGenomeOnStage(candidates[idx], cache.discoverySteps, null, -1, compiled[idx].first, compiled[idx].second)
            }
            Triple(list, 0.0, 0.0)
        }
        val (rollStatsList, rollGpuMs, _) = if (gpuAccelerator != null) {
            PcDiscoveryEngine.evaluateStageHeterogeneous(
                stageKey = "${cache.cacheKey}_roll",
                steps = cache.rollingSteps,
                genomes = candidates,
                compiled = compiled,
                useGpu = useGpu,
                cpuPool = cpuPool,
                gpuAccelerator = gpuAccelerator
            )
        } else {
            val list = PcDiscoveryEngine.mapIndexedParallel(cpuPool, candidates.size) { idx ->
                PcDiscoveryEngine.evaluateGenomeOnStage(candidates[idx], cache.rollingSteps, null, -1, compiled[idx].first, compiled[idx].second)
            }
            Triple(list, 0.0, 0.0)
        }
        val gpuMs = discGpuMs + rollGpuMs

        val evals = PcDiscoveryEngine.mapIndexedParallel(cpuPool, candidates.size) { idx ->
            PcDiscoveryEngine.scoreDiscoveryAndRolling(
                candidates[idx],
                discStatsList[idx],
                rollStatsList[idx],
                cache.discoverySteps,
                cache.rollingSteps
            )
        }.sortedByDescending { it.fitnessScore }

        val passedZeroCap = evals.filter { it.rollingStabilityPassed }
        val rejected = evals.filter { !it.rollingStabilityPassed }.take(40).map { ev ->
            RejectedFormulaSummary(
                id = ev.genome.id,
                generation = generation,
                formulaPreview = ev.genome.describeCompact(),
                rejectedAtStage = ev.rejectionStage.ifBlank { "Usine à Stratégies" },
                reason = ev.rejectionReason,
                trainAvgMatches = ev.discStats.avgMatches,
                validationAvgMatches = ev.rollStats.avgMatches
            )
        }

        val topCandidatesToAudit = (passedZeroCap.take(3).ifEmpty { evals.take(3) })
        var auditedTop: GenomeCandidateEvaluation? = null
        for (cand in topCandidatesToAudit) {
            val confStats = if (cache.confirmationSteps.isNotEmpty()) {
                PcDiscoveryEngine.evaluateGenomeOnStage(cand.genome, cache.confirmationSteps)
            } else null
            val sealedStats = if (cache.sealedSteps.isNotEmpty()) {
                PcDiscoveryEngine.evaluateGenomeOnStage(cand.genome, cache.sealedSteps)
            } else null

            val mc = cache.confirmationMonteCarlo
            val oosSteps = cache.rollingSteps + cache.confirmationSteps
            val oosStats = if (oosSteps.isNotEmpty()) PcDiscoveryEngine.evaluateGenomeOnStage(cand.genome, oosSteps) else confStats
            val effTests = PcDiscoveryEngine.effectiveMultipleTestFactor(confirmationTestsCount)
            val beatsMc = confStats != null && confStats.avgMatches > maxOf(BacktestResult.RANDOM_EXPECTED_MATCHES, mc.meanAvgMatches)
            val pRaw = if (confStats != null && beatsMc && confStats.matchesList.isNotEmpty()) {
                ScientificMath.twoSidedPValueVsRandom(confStats.matchesList)
            } else 1.0
            val pAdj = (pRaw * effTests).coerceAtMost(1.0)
            val oosBeatsMc = oosStats != null && oosStats.avgMatches > maxOf(BacktestResult.RANDOM_EXPECTED_MATCHES, mc.meanAvgMatches)
            val pOosRaw = if (oosStats != null && oosBeatsMc && oosStats.matchesList.isNotEmpty()) ScientificMath.twoSidedPValueVsRandom(oosStats.matchesList) else 1.0
            val pOosAdj = (pOosRaw * effTests).coerceAtMost(1.0)
            val fullSteps = cache.discoverySteps + cache.rollingSteps + cache.confirmationSteps + cache.sealedSteps
            val fullMatches = cand.discStats.matchesList +
                cand.rollStats.matchesList +
                (confStats?.matchesList ?: emptyList()) +
                (if (cache.sealedSteps.isNotEmpty()) sealedStats?.matchesList ?: emptyList() else emptyList())
            val (_, fullConsStreak) = PcDiscoveryEngine.computeConsolidatedDayStats(fullSteps, fullMatches)
            val streak = maxOf(
                cand.discStats.maxConsecutiveQualifyingDays,
                cand.rollStats.maxConsecutiveQualifyingDays,
                confStats?.maxConsecutiveQualifyingDays ?: 0,
                cand.consolidatedTrainStreak,
                fullConsStreak
            )
            val cacheDistinctDays = cache.steps.map { it.dayEpoch }.distinct().size.coerceAtLeast(1)
            val requiredStreak = minObjectiveStreak.coerceIn(1, cacheDistinctDays)
            val streakPassed = streak >= requiredStreak ||
                (cacheDistinctDays == 1 && cand.rollingStabilityPassed && (fullConsStreak >= 1 || (confStats?.avgMatches ?: 0f) >= BacktestResult.RANDOM_EXPECTED_MATCHES))
            val confSigOk = (pAdj < 0.05 || (pRaw < 0.025 && (confStats?.avgMatches ?: 0f) > mc.avgMatchesCiHigh)) ||
                (cache.confirmationSteps.size < 6 && oosBeatsMc && (pOosAdj < 0.05 || pOosRaw < 0.025))
            val sealedDrawOk = if (cacheDistinctDays <= 1 && cache.sealedSteps.size <= 1) {
                (sealedStats?.causalVerified ?: true) && fullConsStreak >= 1
            } else {
                (sealedStats?.causalVerified ?: true) && ((sealedStats?.avgMatches ?: 0f) >= BacktestResult.RANDOM_EXPECTED_MATCHES)
            }
            val allPassed = cand.rollingStabilityPassed &&
                beatsMc &&
                confSigOk &&
                sealedDrawOk &&
                streakPassed

            val effectiveStreak = if (allPassed && cacheDistinctDays == 1) maxOf(1, streak) else streak
            val zeroPct = (cand.rollStats.zeroMatchRate * 100f).toInt()
            val evaluated = cand.copy(
                status = when {
                    allPassed -> "VALIDATED"
                    cand.rollingStabilityPassed && (effectiveStreak >= 1 || (cand.discStats.twoPlusCount + cand.rollStats.twoPlusCount >= 2 && cand.rollStats.avgMatches >= BacktestResult.RANDOM_EXPECTED_MATCHES)) -> "MILESTONE_${maxOf(1, effectiveStreak)}J"
                    else -> cand.status
                },
                rejectionReason = when {
                    allPassed -> "Validé Factory 4 niveaux (${effectiveStreak}J≥3, ≥2/tir ${(cand.rollStats.twoPlusRate * 100f).toInt()}%, 0-boule ${zeroPct}%, p_adj=${String.format(Locale.FRANCE, "%.4f", pAdj)})"
                    cand.rollingStabilityPassed -> "Optimale Factory (${effectiveStreak}J≥3, ≥2/tir ${(cand.rollStats.twoPlusRate * 100f).toInt()}%, Gliss ${String.format(Locale.FRANCE, "%.3f", cand.rollStats.avgMatches)}, 0-boule ${zeroPct}%, ≥3:${cand.rollStats.threePlusCount})"
                    else -> cand.rejectionReason
                },
                confStats = confStats,
                sealedStats = sealedStats,
                pValueRaw = pRaw,
                pValueAdjusted = pAdj,
                consolidatedFullStreak = if (allPassed) effectiveStreak else fullConsStreak
            )
            if (auditedTop == null || evaluated.status == "VALIDATED") {
                auditedTop = evaluated
            }
            if (evaluated.status == "VALIDATED") break
        }

        val elapsedSecs = ((System.nanoTime() - startNs) / 1_000_000_000.0).coerceAtLeast(1e-4)
        val totalStepEvals = candidates.size * (cache.discoverySteps.size + cache.rollingSteps.size).coerceAtLeast(1)
        val evalsPerSec = totalStepEvals / elapsedSecs

        val topSummary = auditedTop?.let {
            "Top [${it.genome.id}] : ≥2/tir ${(it.rollStats.twoPlusRate * 100f).toInt()}% • J(≥3)=${it.rollStats.qualifyingDays} • " +
                "Gliss ${String.format(Locale.FRANCE, "%.3f", it.rollStats.avgMatches)} • 0-boule ${(it.rollStats.zeroMatchRate * 100f).toInt()}% • ≥3:${it.rollStats.threePlusCount}"
        } ?: "Aucun candidat"

        val note = "Factory ($domainLabel) : ${candidates.size} formules inédites testées (${passedZeroCap.size} stables) • $topSummary"

        return StrategyFactorySearchReport(
            domainLabel = domainLabel,
            generatedCount = candidates.size,
            unseenEvaluatedCount = evals.size,
            zeroCapPassedCount = passedZeroCap.size,
            optimalCandidates = (if (auditedTop != null) listOf(auditedTop) else emptyList()) +
                evals.filter { it.genome.id != auditedTop?.genome?.id }.take(23),
            rejectedCandidates = rejected,
            auditedWinner = auditedTop,
            domainBreakdown = breakdown,
            evaluationsPerSec = evalsPerSec,
            gpuBatchMs = gpuMs,
            summaryNote = note
        )
    }
}
