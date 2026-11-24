package com.lotobj.pc.engine

import com.lotobj.pc.model.BinaryOp
import com.lotobj.pc.model.ChercheurTeaching
import com.lotobj.pc.model.ExprNode
import com.lotobj.pc.model.PrecisionFeature
import com.lotobj.pc.model.PrecisionGenome
import com.lotobj.pc.model.PrecisionTargetMode
import com.lotobj.pc.model.PrecisionTransform
import com.lotobj.pc.model.RawFeature
import com.lotobj.pc.model.RawFeatureMatrix
import com.lotobj.pc.model.SymbolicGenome
import com.lotobj.pc.model.UnaryOp
import java.util.Locale
import kotlin.math.abs
import kotlin.math.round
import kotlin.math.sqrt
import kotlin.random.Random

data class ChercheurHypothesis(
    val key: String,
    val title: String,
    val scope: String,
    val scoreBall: (matrix: RawFeatureMatrix, number: Int) -> Double,
    val primaryFeatures: List<RawFeature>,
    val secondaryFeatures: List<RawFeature>,
    val preferredPrimarySlots: Int
)

data class ChercheurCycleOutput(
    val teachings: List<ChercheurTeaching>,
    val synthesizedGenomes: List<SymbolicGenome>,
    val teachingByGenomeId: Map<String, String>,
    val cycleSummaryNote: String
)

/**
 * Moteur "Le Chercheur" (PC Workstation — IA Empirique Multi-Niveaux) :
 * - Strictement causal sur la fenêtre d'apprentissage (discoverySteps uniquement).
 * - Apprend séparément sur :
 *   1) Chaque numéro individuel du tirage
 *   2) Chaque position individuelle brute (Pos1 = num1 .. Pos5 = num5) par calcul de Z-Lift & Ridge
 *   3) Le tirage complet (synergie & complémentarité résiduelle)
 * - Cycle "Surapprendre ce qui perce sur t -> Tester sur t+1 -> Adapter sur l'erreur si <2 exacts"
 * - Distille des ENSEIGNEMENTS ABSTRAITS pour Moteur 1 (SymbolicGenome) ET Moteur 2 (PrecisionGenome).
 */
object ChercheurEngine {

    private val gpuFeatures: List<RawFeature> = SymbolicFormula.GPU_CAPABLE_FEATURES

    private fun buildDynamicHypotheses(
        window: List<CachedRawStep>,
        cycleIndex: Int
    ): Pair<List<ChercheurHypothesis>, IntArray> {
        val fCount = gpuFeatures.size
        // Calcul empirique des signaux par Position (Pos1..Pos5) + Tirage Complet sans aucune fuite future
        val posSumHit = Array(5) { DoubleArray(fCount) }
        val posCount = IntArray(5)
        val allSumHit = DoubleArray(fCount)
        val allSumMiss = DoubleArray(fCount)
        val allSqHit = DoubleArray(fCount)
        val allSqMiss = DoubleArray(fCount)
        var allNHit = 0
        var allNMiss = 0

        for (step in window) {
            val mat = step.rawMatrix
            val mask = step.actualMask
            val posBalls = intArrayOf(
                step.draw.num1, step.draw.num2, step.draw.num3, step.draw.num4, step.draw.num5
            )
            for (pos in 0 until 5) {
                val b = posBalls[pos]
                if (b in 1..90) {
                    posCount[pos]++
                    for (fi in 0 until fCount) {
                        val v = mat.get(gpuFeatures[fi].ordinal, b)
                        if (v.isFinite()) posSumHit[pos][fi] += v
                    }
                }
            }
            for (n in 1..90) {
                val hit = mask[n]
                if (hit) allNHit++ else allNMiss++
                for (fi in 0 until fCount) {
                    val v = mat.get(gpuFeatures[fi].ordinal, n)
                    if (!v.isFinite()) continue
                    if (hit) {
                        allSumHit[fi] += v
                        allSqHit[fi] += v * v
                    } else {
                        allSumMiss[fi] += v
                        allSqMiss[fi] += v * v
                    }
                }
            }
        }

        val dynHypotheses = ArrayList<ChercheurHypothesis>(9)

        // 1) Hypothèses empiriques dédiées à chaque Position Pos1..Pos5 apprises sur l'historique local
        val missMeans = DoubleArray(fCount) { fi ->
            if (allNMiss > 0) allSumMiss[fi] / allNMiss else 0.0
        }
        val missStd = DoubleArray(fCount) { fi ->
            if (allNMiss > 1) {
                val m = missMeans[fi]
                sqrt((allSqMiss[fi] / allNMiss - m * m).coerceAtLeast(1e-5))
            } else 1.0
        }

        val posBestFeatureIdx = IntArray(5)
        for (pos in 0 until 5) {
            val cnt = posCount[pos].coerceAtLeast(1)
            val weights = DoubleArray(fCount) { fi ->
                val meanPos = posSumHit[pos][fi] / cnt
                ((meanPos - missMeans[fi]) / missStd[fi]).coerceIn(-2.5, 2.5)
            }
            val topFeatsIdx = (0 until fCount).sortedByDescending { abs(weights[it]) }
            posBestFeatureIdx[pos] = topFeatsIdx.first()
            val topMain = topFeatsIdx.take(3).map { gpuFeatures[it] }
            val topSec = topFeatsIdx.drop(3).take(3).map { gpuFeatures[it] }.ifEmpty { topMain }
            val capturedWeights = weights.copyOf()
            val capturedTop = topFeatsIdx.take(5).toIntArray()

            dynHypotheses.add(
                ChercheurHypothesis(
                    key = "DYN_POS${pos + 1}",
                    title = "Signal Empirique Pos${pos + 1} (${topMain.joinToString("+") { it.symbol }})",
                    scope = "Position individuelle Pos${pos + 1}",
                    scoreBall = { m, n ->
                        var acc = 0.0
                        for (fi in capturedTop) {
                            acc += capturedWeights[fi] * m.get(gpuFeatures[fi].ordinal, n)
                        }
                        acc
                    },
                    primaryFeatures = topMain,
                    secondaryFeatures = topSec,
                    preferredPrimarySlots = if (pos <= 1) 2 else if (pos == 2) 3 else 4
                )
            )
        }

        // 2) Hypothèse Z-Lift Global Tirage Complet
        val globalZ = DoubleArray(fCount) { fi ->
            val mH = if (allNHit > 0) allSumHit[fi] / allNHit else 0.0
            val mM = missMeans[fi]
            val vH = if (allNHit > 1) (allSqHit[fi] / allNHit - mH * mH).coerceAtLeast(1e-5) else 1.0
            val vM = (missStd[fi] * missStd[fi]).coerceAtLeast(1e-5)
            ((mH - mM) / sqrt((vH + vM) * 0.5)).coerceIn(-3.0, 3.0)
        }
        val globTop = (0 until fCount).sortedByDescending { abs(globalZ[it]) }
        val globTopIdx = globTop.take(6).toIntArray()
        dynHypotheses.add(
            ChercheurHypothesis(
                key = "DYN_GLOBAL_ZLIFT",
                title = "Synthèse Z-Lift Globale (${globTop.take(3).joinToString("+") { gpuFeatures[it].symbol }})",
                scope = "Tirage complet (5 boules)",
                scoreBall = { m, n ->
                    var acc = 0.0
                    for (fi in globTopIdx) acc += globalZ[fi] * m.get(gpuFeatures[fi].ordinal, n)
                    acc
                },
                primaryFeatures = globTop.take(3).map { gpuFeatures[it] },
                secondaryFeatures = globTop.drop(3).take(3).map { gpuFeatures[it] }.ifEmpty { globTop.take(3).map { gpuFeatures[it] } },
                preferredPrimarySlots = 3
            )
        )

        // 3) Hypothèse Complémentarité Résiduelle (numéros manqués par Z-Lift Global)
        val xTx = Array(fCount) { DoubleArray(fCount) }
        val xTy = DoubleArray(fCount)
        val row = DoubleArray(fCount)
        for (step in window) {
            val mat = step.rawMatrix
            val mask = step.actualMask
            val top5Glob = (1..90).sortedByDescending { n ->
                var s = 0.0
                for (fi in globTopIdx) s += globalZ[fi] * mat.get(gpuFeatures[fi].ordinal, n)
                s
            }.take(5).toSet()
            for (n in 1..90) {
                val resHit = mask[n] && n !in top5Glob
                val y = if (resHit) 1.0 else 0.0
                val sw = if (resHit) 4.0 else 1.0
                for (fi in 0 until fCount) {
                    val v = mat.get(gpuFeatures[fi].ordinal, n)
                    row[fi] = if (v.isFinite()) v else 0.0
                }
                for (i in 0 until fCount) {
                    val vi = row[i]
                    if (vi == 0.0) continue
                    val wVi = vi * sw
                    xTy[i] += wVi * y
                    for (j in i until fCount) {
                        val vj = row[j]
                        if (vj != 0.0) xTx[i][j] += wVi * vj
                    }
                }
            }
        }
        for (i in 0 until fCount) for (j in 0 until i) xTx[i][j] = xTx[j][i]
        val wResid = CausalAiLearner.solveLinearRidge(xTx, xTy, fCount, lambda = (window.size * 2.5).coerceIn(8.0, 90.0))
        val resTop = (0 until fCount).sortedByDescending { abs(wResid[it]) }
        val resTopIdx = resTop.take(6).toIntArray()
        dynHypotheses.add(
            ChercheurHypothesis(
                key = "DYN_RESIDUAL_RIDGE",
                title = "Correcteur Résiduel Ridge (${resTop.take(3).joinToString("+") { gpuFeatures[it].symbol }})",
                scope = "Tirage complet (complémentarité)",
                scoreBall = { m, n ->
                    var acc = 0.0
                    for (fi in resTopIdx) acc += wResid[fi] * m.get(gpuFeatures[fi].ordinal, n)
                    acc
                },
                primaryFeatures = resTop.take(3).map { gpuFeatures[it] },
                secondaryFeatures = resTop.drop(3).take(3).map { gpuFeatures[it] }.ifEmpty { resTop.take(3).map { gpuFeatures[it] } },
                preferredPrimarySlots = 2
            )
        )

        // 4) Hypothèse Structurelle Modulaire + Triangulation (rotation selon cycleIndex)
        val rotateOffset = cycleIndex % 3
        val structPrimary = when (rotateOffset) {
            0 -> listOf(RawFeature.MOD7_RES_DENSITY, RawFeature.AFFINE_MOD90_RESONANCE, RawFeature.LUCKY_ROOT_NORM)
            1 -> listOf(RawFeature.CONGRUENCE_CUBE90, RawFeature.LIMIT_HAZARD, RawFeature.GAP_NORM)
            else -> listOf(RawFeature.TRIANG_PYRAMID_DIFF, RawFeature.CO_LAST_RAW, RawFeature.GRID_9X10_PROXIMITY)
        }
        val structSec = listOf(RawFeature.HORAIRE_FREQ, RawFeature.PROB_ZSCORE, RawFeature.EXTRAPOL_VELOCITY)
        dynHypotheses.add(
            ChercheurHypothesis(
                key = "DYN_STRUCT_$rotateOffset",
                title = "Théorème Structurel #${rotateOffset + 1} (${structPrimary.joinToString("+") { it.symbol }})",
                scope = "Chaque numéro individuel",
                scoreBall = { m, n ->
                    0.40 * m.get(structPrimary[0].ordinal, n) +
                        0.33 * m.get(structPrimary[1].ordinal, n) +
                        0.27 * m.get(structPrimary[2].ordinal, n)
                },
                primaryFeatures = structPrimary,
                secondaryFeatures = structSec,
                preferredPrimarySlots = 3
            )
        )

        return dynHypotheses to posBestFeatureIdx
    }

    fun runResearchAndDistill(
        cache: FourStageCausalCache,
        cycleIndex: Int,
        generation: Int,
        seenSignatures: Set<String>
    ): ChercheurCycleOutput {
        val window = if (cache.discoverySteps.size >= 4) {
            cache.discoverySteps
        } else {
            cache.steps.take(((cache.steps.size * 2) / 3).coerceAtLeast(minOf(cache.steps.size, 3)))
        }
        if (window.size < 3) {
            return ChercheurCycleOutput(
                teachings = emptyList(),
                synthesizedGenomes = emptyList(),
                teachingByGenomeId = emptyMap(),
                cycleSummaryNote = "Chercheur en attente : au moins 3 tirages causaux requis."
            )
        }

        val (hypotheses, _) = buildDynamicHypotheses(window, cycleIndex)
        val hypStats = HashMap<String, MutableList<Int>>(hypotheses.size * 2)
        val posHits = IntArray(5)
        val rng = Random(cycleIndex * 91_807L + generation * 4_999L + 19L)

        // Apprentissage séquentiel : Surapprend sur t -> teste sur t+1 -> si échoue (<2), cherche & corrige sur t+1
        var activeHyp = hypotheses[cycleIndex % hypotheses.size]
        val consecutivePierceScores = HashMap<String, Int>()
        val learnedWeights = DoubleArray(hypotheses.size) { 1.0 }

        for (stepIdx in 0 until window.size - 1) {
            val stepT = window[stepIdx]
            val stepNext = window[stepIdx + 1]

            // 1) Identifier l'hypothèse causale qui explique le mieux stepT
            val bestOnT = findBestHypothesisForDraw(hypotheses, stepT, posHits)
            val hypIdxT = hypotheses.indexOf(bestOnT)
            if (hypIdxT >= 0) learnedWeights[hypIdxT] += 0.50
            activeHyp = bestOnT

            // 2) Tester l'hypothèse apprise sur stepNext sans aucune fuite
            val nextTop5 = top5ForHypothesis(activeHyp, stepNext)
            val nextHits = nextTop5.count { it in stepNext.actualSet }
            hypStats.getOrPut(activeHyp.key) { ArrayList() }.add(nextHits)

            // Diagnostic positionnel exact (Pos1..Pos5)
            val predictedSet = nextTop5.toSet()
            if (stepNext.draw.num1 in predictedSet) posHits[0]++
            if (stepNext.draw.num2 in predictedSet) posHits[1]++
            if (stepNext.draw.num3 in predictedSet) posHits[2]++
            if (stepNext.draw.num4 in predictedSet) posHits[3]++
            if (stepNext.draw.num5 in predictedSet) posHits[4]++

            // 3) Si ne perce pas (<2 exacts), chercher l'hypothèse corrective pour stepNext
            if (nextHits < 2) {
                val corrective = findBestHypothesisForDraw(hypotheses, stepNext, posHits)
                val corrIdx = hypotheses.indexOf(corrective)
                if (corrIdx >= 0) learnedWeights[corrIdx] += 0.70
                activeHyp = corrective
                val corrHits = top5ForHypothesis(corrective, stepNext).count { it in stepNext.actualSet }
                hypStats.getOrPut(corrective.key) { ArrayList() }.add(corrHits)
            } else {
                consecutivePierceScores[activeHyp.key] = (consecutivePierceScores[activeHyp.key] ?: 0) + nextHits
            }
        }

        // Sélection multi-tirages (jamais copiée d'un tirage unique)
        val rankedHypotheses = hypotheses.indices.sortedByDescending { idx ->
            val h = hypotheses[idx]
            val hits = hypStats[h.key].orEmpty()
            val avg = if (hits.isNotEmpty()) hits.average() else 0.0
            val twoPlusRate = if (hits.isNotEmpty()) hits.count { it >= 2 }.toDouble() / hits.size else 0.0
            val zeroRate = if (hits.isNotEmpty()) hits.count { it == 0 }.toDouble() / hits.size else 1.0
            val pierce = consecutivePierceScores[h.key] ?: 0
            twoPlusRate * 160.0 + learnedWeights[idx] * 0.45 + avg * 28.0 + pierce * 2.5 - zeroRate * 18.0
        }

        val topPosIdx = posHits.indices.maxByOrNull { posHits[it] } ?: 0
        val teachings = ArrayList<ChercheurTeaching>()
        val genomes = ArrayList<SymbolicGenome>()
        val teachingMap = LinkedHashMap<String, String>()

        // Intégrer aussi un noyau Ridge directement issu du Learner pour synthétiser des enfants non triviaux
        val ridgeSeeds = CausalAiLearner.fitSymbolicRidgeGenomes(
            discoverySteps = window,
            generation = generation.coerceAtLeast(1),
            sessionSalt = cycleIndex * 13_007L + generation * 97L,
            seenSignatures = seenSignatures,
            maxCount = 6,
            paradigmHint = cycleIndex
        )

        val primaryTop = rankedHypotheses.take(3)
        for ((rank, hIdx) in primaryTop.withIndex()) {
            val h = hypotheses[hIdx]
            val partner = hypotheses[rankedHypotheses[(rank + 1) % rankedHypotheses.size]]
            val hits = hypStats[h.key].orEmpty()
            val twoPlusHits = hits.count { it >= 2 }
            val twoPlusPct = if (hits.isNotEmpty()) (twoPlusHits * 100f / hits.size).toInt() else 0
            val blendTwoPlus = window.parallelStream().mapToInt { step ->
                val blendTop5 = (1..90).sortedByDescending { n ->
                    0.58 * h.scoreBall(step.rawMatrix, n) + 0.42 * partner.scoreBall(step.rawMatrix, n)
                }.take(5)
                if (blendTop5.count { it in step.actualSet } >= 2) 1 else 0
            }.sum()
            val piercedCount = maxOf(twoPlusHits, blendTwoPlus) + (consecutivePierceScores[h.key] ?: 0) / 3
            val avgHits = if (hits.isNotEmpty()) hits.average().toFloat() else 0f
            val teachingId = "teach_c${cycleIndex}_g${generation}_$rank"

            val abstractPrinciple = buildString {
                append("Principe causal (≥2 exacts : $twoPlusPct%, duo $blendTwoPlus/${window.size} • ${h.scope} + Pos${topPosIdx + 1}) : ")
                append("croiser ${h.primaryFeatures.joinToString("+") { it.symbol }} ")
                append("avec l'équilibre résiduel ${partner.secondaryFeatures.take(2).joinToString("+") { it.symbol }} ")
                append("(moy. ${String.format(Locale.FRANCE, "%.2f", avgHits)} boules).")
            }

            val teaching = ChercheurTeaching(
                id = teachingId,
                title = "Leçon #$rank : ${h.title.substringBefore(" (")}",
                scopeLabel = "${h.scope} • Pos${topPosIdx + 1}",
                principleSummary = abstractPrinciple,
                favoredFeatures = (h.primaryFeatures + partner.secondaryFeatures).map { it.symbol }.distinct(),
                preferredSlotSplit = "${h.preferredPrimarySlots}+${5 - h.preferredPrimarySlots}",
                generalizedScore = ((twoPlusPct * 0.65f) + (avgHits * 26f) + piercedCount * 5f).coerceIn(1f, 99f),
                piercedDrawsCount = piercedCount.coerceAtLeast(1),
                status = "INJECTED"
            )
            teachings.add(teaching)

            // Injecter un mix : 1 variante issue du solveur Ridge distillée + 2 variantes symboliques inédites
            ridgeSeeds.getOrNull(rank)?.let { rSeed ->
                val distilled = SymbolicGenome(
                    id = "ch_${teachingId}_ridge_$rank",
                    generation = generation.coerceAtLeast(1),
                    parentIds = listOf("chercheur:$teachingId", "causal_ridge"),
                    primaryExpr = rSeed.primaryExpr,
                    secondaryExpr = buildAbstractTeachingExpr(partner.secondaryFeatures, h.secondaryFeatures, rng, rank + 1),
                    primarySlots = h.preferredPrimarySlots.coerceIn(1, 4),
                    samplingTemp = rSeed.samplingTemp,
                    exploreNoise = 0.0,
                    seed = rSeed.seed
                )
                if (distilled.describeCompact() !in seenSignatures && genomes.none { it.describeCompact() == distilled.describeCompact() }) {
                    genomes.add(distilled)
                    teachingMap[distilled.id] = teachingId
                }
            }

            repeat(3) { variant ->
                for (attempt in 0 until 8) {
                    val exprA = buildAbstractTeachingExpr(h.primaryFeatures, partner.primaryFeatures, rng, variant + attempt)
                    val exprB = buildAbstractTeachingExpr(partner.secondaryFeatures, h.secondaryFeatures, rng, variant + attempt + 3)
                    val genome = SymbolicGenome(
                        id = "ch_${teachingId}_v${variant}_$attempt",
                        generation = generation.coerceAtLeast(1),
                        parentIds = listOf("chercheur:$teachingId"),
                        primaryExpr = SymbolicFormula.simplify(exprA),
                        secondaryExpr = SymbolicFormula.simplify(exprB),
                        primarySlots = when (variant) {
                            0 -> h.preferredPrimarySlots
                            1 -> partner.preferredPrimarySlots
                            else -> ((h.preferredPrimarySlots + partner.preferredPrimarySlots) / 2).coerceIn(2, 4)
                        },
                        samplingTemp = round2(0.24 + rng.nextDouble() * 0.56),
                        exploreNoise = 0.0,
                        seed = cycleIndex * 71_011L + generation * 1_777L + rank * 193L + variant * 41L + attempt * 7L
                    )
                    if (genome.describeCompact() !in seenSignatures && genomes.none { it.describeCompact() == genome.describeCompact() }) {
                        genomes.add(genome)
                        teachingMap[genome.id] = teachingId
                        break
                    }
                }
            }
        }

        val note = "Chercheur IA Cycle #$cycleIndex : ${window.size} tirages analysés (Pos1..5 empirique + Résiduel Ridge) → ${teachings.size} leçons / ${genomes.size} stratégies injectées."
        return ChercheurCycleOutput(
            teachings = teachings,
            synthesizedGenomes = genomes,
            teachingByGenomeId = teachingMap,
            cycleSummaryNote = note
        )
    }

    /**
     * Distille des stratégies causales dédiées pour le Moteur 2 (Précision 1er N° / 2 N° exacts)
     * à partir des tirages manqués en Découverte uniquement (zéro fuite future).
     */
    fun distillPrecisionTeachings(
        mode: PrecisionTargetMode,
        discoverySteps: List<PrecisionRawStep>,
        cycleIndex: Int,
        generation: Int,
        seenSignatures: Set<String>
    ): List<PrecisionGenome> {
        if (discoverySteps.size < 3) return emptyList()
        val fCount = PrecisionFeature.entries.size
        val missLift = DoubleArray(fCount)
        val hitCounts = IntArray(1)
        val posHits = DoubleArray(fCount)
        val missVals = DoubleArray(fCount)
        var nHit = 0
        var nMiss = 0

        for (step in discoverySteps) {
            val mat = step.matrix
            for (n in 1..90) {
                val isTarget = if (mode == PrecisionTargetMode.FIRST_BALL) (n == step.actualPos1) else step.actualMask[n]
                if (isTarget) {
                    nHit++
                    for (fi in 0 until fCount) posHits[fi] += mat.get(fi, n)
                } else {
                    nMiss++
                    for (fi in 0 until fCount) missVals[fi] += mat.get(fi, n)
                }
            }
        }
        if (nHit == 0 || nMiss == 0) return emptyList()
        for (fi in 0 until fCount) {
            missLift[fi] = ((posHits[fi] / nHit) - (missVals[fi] / nMiss)).coerceIn(-2.5, 2.5)
        }
        var maxA = 1e-6
        for (v in missLift) if (abs(v) > maxA) maxA = abs(v)
        val gain = 2.2 / maxA
        val baseW = List(fCount) { fi -> round2((missLift[fi] * gain).coerceIn(-3.0, 3.0)) }

        val ridgeBatch = CausalAiLearner.fitPrecisionRidgeGenomes(
            mode = mode,
            discoverySteps = discoverySteps,
            generation = generation,
            sessionSalt = cycleIndex * 19_993L + generation * 31L,
            seenSignatures = seenSignatures,
            maxCount = 10
        )

        val out = ArrayList<PrecisionGenome>(14)
        out.addAll(ridgeBatch)

        val rng = Random(cycleIndex * 55_127L + generation * 101L)
        val transforms = PrecisionTransform.entries
        repeat(4) { idx ->
            val topK = if (idx % 2 == 0) 5 else 8
            val keep = baseW.indices.sortedByDescending { abs(baseW[it]) }.take(topK).toSet()
            val w1 = List(fCount) { fi ->
                if (fi in keep) round2((baseW[fi] + (rng.nextDouble() * 0.4 - 0.2)).coerceIn(-3.2, 3.2)) else 0.0
            }
            val w2 = List(fCount) { fi ->
                if (fi !in keep && abs(baseW[fi]) > 0.25) round2(baseW[fi] * 0.9) else if (fi in keep) round2(baseW[fi] * 0.45) else 0.0
            }
            val g = PrecisionGenome(
                id = "prec_${mode.name}_ch_c${cycleIndex}_g${generation}_$idx",
                mode = mode,
                generation = generation.coerceAtLeast(1),
                weights = w1,
                secondaryWeights = w2,
                transform = transforms[(cycleIndex + idx) % transforms.size],
                interactionBoost = round2(0.25 + idx * 0.18),
                samplingTemp = round2(0.28 + idx * 0.14),
                exploreNoise = 0.0
            )
            if (g.signatureKey() !in seenSignatures && out.none { it.signatureKey() == g.signatureKey() }) {
                out.add(g)
            }
        }
        hitCounts[0] = out.size
        return out
    }

    private fun findBestHypothesisForDraw(
        hypotheses: List<ChercheurHypothesis>,
        step: CachedRawStep,
        posHits: IntArray
    ): ChercheurHypothesis {
        var best = hypotheses.first()
        var bestObj = Double.NEGATIVE_INFINITY
        val rawPos = intArrayOf(step.draw.num1, step.draw.num2, step.draw.num3, step.draw.num4, step.draw.num5)
        for (h in hypotheses) {
            val top5 = top5ForHypothesis(h, step)
            val hits = top5.count { it in step.actualSet }
            var posMatchBonus = 0.0
            for (idx in 0 until 5) {
                if (rawPos[idx] in top5) posMatchBonus += 0.35 + posHits[idx] * 0.02
            }
            val actualScoreMass = step.actualNumbers.sumOf { n -> h.scoreBall(step.rawMatrix, n) }
            val obj = hits * 10.0 + posMatchBonus + actualScoreMass
            if (obj > bestObj) {
                bestObj = obj
                best = h
            }
        }
        return best
    }

    private fun top5ForHypothesis(hyp: ChercheurHypothesis, step: CachedRawStep): List<Int> {
        val m = step.rawMatrix
        return (1..90)
            .sortedWith(compareByDescending<Int> { hyp.scoreBall(m, it) }.thenBy { it })
            .take(5)
    }

    private fun buildAbstractTeachingExpr(
        mainFeatures: List<RawFeature>,
        supportFeatures: List<RawFeature>,
        rng: Random,
        variant: Int
    ): ExprNode {
        val f1 = ExprNode.Var(mainFeatures[rng.nextInt(mainFeatures.size)])
        val f2 = ExprNode.Var(mainFeatures[(variant + 1) % mainFeatures.size])
        val fs = ExprNode.Var(supportFeatures[rng.nextInt(supportFeatures.size)])
        val w1 = ExprNode.Const(round2(0.45 + rng.nextDouble() * 1.35))
        val w2 = ExprNode.Const(round2(0.35 + rng.nextDouble() * 1.15))

        val unaryMain = listOf(UnaryOp.SIGMOID, UnaryOp.TANH, UnaryOp.SOFTPLUS, UnaryOp.LUCKY9_WAVE, UnaryOp.SQRT_ABS)
        val binaryMix = listOf(BinaryOp.ADD, BinaryOp.HYPOT_NORM, BinaryOp.TRIANG_BLEND, BinaryOp.MEAN2, BinaryOp.EXP_DIFF)

        val uA = unaryMain[(variant + rng.nextInt(unaryMain.size)) % unaryMain.size]
        val uB = unaryMain[(variant + 2 + rng.nextInt(unaryMain.size)) % unaryMain.size]
        val bin = binaryMix[rng.nextInt(binaryMix.size)]

        val core = ExprNode.Binary(
            bin,
            ExprNode.Unary(uA, ExprNode.Binary(BinaryOp.MUL, w1, f1)),
            ExprNode.Binary(BinaryOp.ADD, ExprNode.Binary(BinaryOp.MUL, w2, f2), ExprNode.Unary(uB, fs))
        )
        return when (variant % 3) {
            0 -> core
            1 -> ExprNode.Binary(BinaryOp.SUB, core, ExprNode.Const(round2(rng.nextDouble() * 0.25)))
            else -> ExprNode.Unary(UnaryOp.TANH, core)
        }
    }

    private fun round2(v: Double): Double = round(v * 100.0) / 100.0
}
