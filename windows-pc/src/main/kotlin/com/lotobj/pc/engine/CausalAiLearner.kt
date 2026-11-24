package com.lotobj.pc.engine

import com.lotobj.pc.model.BinaryOp
import com.lotobj.pc.model.ExprNode
import com.lotobj.pc.model.PrecisionFeature
import com.lotobj.pc.model.PrecisionGenome
import com.lotobj.pc.model.PrecisionTargetMode
import com.lotobj.pc.model.PrecisionTransform
import com.lotobj.pc.model.RawFeature
import com.lotobj.pc.model.SymbolicGenome
import com.lotobj.pc.model.UnaryOp
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.round

/**
 * Apprentissage causal multivarié (Régression Ridge régularisée + Boost Résiduel + Spécialisation Horaire)
 * Strictement exécuté sur `discoverySteps` uniquement (zéro fuite Rolling / Confirmation / Scellé).
 */
object CausalAiLearner {

    fun solveLinearRidge(
        xTxInput: Array<DoubleArray>,
        xTyInput: DoubleArray,
        dim: Int,
        lambda: Double
    ): DoubleArray {
        val a = Array(dim) { i ->
            DoubleArray(dim) { j ->
                val v = xTxInput[i][j]
                if (i == j) v + lambda else v
            }
        }
        val b = DoubleArray(dim) { i -> xTyInput[i] }

        for (col in 0 until dim) {
            var pivotRow = col
            var maxAbs = abs(a[col][col])
            for (r in col + 1 until dim) {
                val cand = abs(a[r][col])
                if (cand > maxAbs) {
                    maxAbs = cand
                    pivotRow = r
                }
            }
            if (pivotRow != col) {
                val tmpRow = a[col]
                a[col] = a[pivotRow]
                a[pivotRow] = tmpRow
                val tmpB = b[col]
                b[col] = b[pivotRow]
                b[pivotRow] = tmpB
            }
            val pivot = a[col][col]
            if (abs(pivot) < 1e-10) continue

            val invPivot = 1.0 / pivot
            for (r in col + 1 until dim) {
                val factor = a[r][col] * invPivot
                if (factor == 0.0) continue
                a[r][col] = 0.0
                for (c in col + 1 until dim) {
                    a[r][c] -= factor * a[col][c]
                }
                b[r] -= factor * b[col]
            }
        }

        val w = DoubleArray(dim)
        for (i in dim - 1 downTo 0) {
            var sum = b[i]
            for (c in i + 1 until dim) {
                sum -= a[i][c] * w[c]
            }
            val diag = a[i][i]
            w[i] = if (abs(diag) > 1e-10 && sum.isFinite()) (sum / diag).coerceIn(-4.0, 4.0) else 0.0
        }
        return w
    }

    fun fitSymbolicRidgeGenomes(
        discoverySteps: List<CachedRawStep>,
        generation: Int,
        sessionSalt: Long,
        seenSignatures: Set<String> = emptySet(),
        maxCount: Int = 20,
        paradigmHint: Int = 0
    ): List<SymbolicGenome> {
        if (discoverySteps.isEmpty()) return emptyList()
        val features = SymbolicFormula.GPU_CAPABLE_FEATURES
        val dim = features.size
        val stepCount = discoverySteps.size

        // 1) Global Ridge + Recency-Weighted Ridge sur discoverySteps uniquement
        val xTxGlobal = Array(dim) { DoubleArray(dim) }
        val xTyGlobal = DoubleArray(dim)
        val xTxRecent = Array(dim) { DoubleArray(dim) }
        val xTyRecent = DoubleArray(dim)
        val xMean = DoubleArray(dim)
        var ySum = 0.0
        var totalSamples = 0.0

        val rowVals = DoubleArray(dim)
        for ((sIdx, step) in discoverySteps.withIndex()) {
            val mat = step.rawMatrix
            val mask = step.actualMask
            val age = (stepCount - 1 - sIdx).toDouble()
            val recencyW = exp(-age * 0.075)

            for (n in 1..90) {
                val y = if (mask[n]) 1.0 else 0.0
                val sampleWeight = if (mask[n]) 2.4 else 1.0 // Surpondère légèrement les boules positives pour précision Top-5
                totalSamples += 1.0
                ySum += y
                for (i in 0 until dim) {
                    val v = mat.get(features[i].ordinal, n)
                    val safeV = if (v.isFinite()) v else 0.0
                    rowVals[i] = safeV
                    xMean[i] += safeV
                }
                for (i in 0 until dim) {
                    val vi = rowVals[i]
                    if (vi == 0.0) continue
                    val wViGlobal = vi * sampleWeight
                    val wViRecent = wViGlobal * recencyW
                    xTyGlobal[i] += wViGlobal * y
                    xTyRecent[i] += wViRecent * y
                    for (j in i until dim) {
                        val vj = rowVals[j]
                        if (vj != 0.0) {
                            xTxGlobal[i][j] += wViGlobal * vj
                            xTxRecent[i][j] += wViRecent * vj
                        }
                    }
                }
            }
        }
        if (totalSamples <= 0.0) return emptyList()
        for (i in 0 until dim) {
            for (j in 0 until i) {
                xTxGlobal[i][j] = xTxGlobal[j][i]
                xTxRecent[i][j] = xTxRecent[j][i]
            }
        }

        val scaleLambda = (stepCount * 2.5).coerceIn(8.0, 120.0)
        val wGlobal = solveLinearRidge(xTxGlobal, xTyGlobal, dim, lambda = scaleLambda)
        val wRecent = solveLinearRidge(xTxRecent, xTyRecent, dim, lambda = (scaleLambda * 0.7).coerceAtLeast(6.0))

        // 2) Second passe : Residual-Boosted Complementary Ridge (Formule B)
        // Entraîné spécifiquement sur les boules réelles manquées par le Top-3 du modèle Global Ridge
        val xTxResid = Array(dim) { DoubleArray(dim) }
        val xTyResid = DoubleArray(dim)
        for (step in discoverySteps) {
            val mat = step.rawMatrix
            val mask = step.actualMask
            val top3Set = (1..90).sortedByDescending { n ->
                var acc = 0.0
                for (i in 0 until dim) acc += wGlobal[i] * mat.get(features[i].ordinal, n)
                acc
            }.take(3).toSet()
            val hitsTop3 = top3Set.count { it in step.actualSet }
            val drawHardBoost = if (hitsTop3 < 2) 1.8 else 0.65

            for (n in 1..90) {
                val missedActual = mask[n] && (n !in top3Set)
                val yRes = if (missedActual) 1.0 else 0.0
                val sampleW = (if (missedActual) 4.2 else 1.0) * drawHardBoost
                for (i in 0 until dim) {
                    val v = mat.get(features[i].ordinal, n)
                    rowVals[i] = if (v.isFinite()) v else 0.0
                }
                for (i in 0 until dim) {
                    val vi = rowVals[i]
                    if (vi == 0.0) continue
                    val wVi = vi * sampleW
                    xTyResid[i] += wVi * yRes
                    for (j in i until dim) {
                        val vj = rowVals[j]
                        if (vj != 0.0) xTxResid[i][j] += wVi * vj
                    }
                }
            }
        }
        for (i in 0 until dim) {
            for (j in 0 until i) xTxResid[i][j] = xTxResid[j][i]
        }
        val wResid = solveLinearRidge(xTxResid, xTyResid, dim, lambda = scaleLambda * 1.15)

        // Normalisation des vecteurs appris par amplitude maximale
        fun normalizeWeights(raw: DoubleArray): DoubleArray {
            var maxAbs = 1e-6
            for (v in raw) if (abs(v) > maxAbs) maxAbs = abs(v)
            val gain = 2.4 / maxAbs
            return DoubleArray(raw.size) { i -> (raw[i] * gain).coerceIn(-3.5, 3.5) }
        }

        val nwGlobal = normalizeWeights(wGlobal)
        val nwRecent = normalizeWeights(wRecent)
        val nwResid = normalizeWeights(wResid)
        val nwBlend = DoubleArray(dim) { i -> round2(0.60 * nwGlobal[i] + 0.40 * nwRecent[i]) }

        fun buildAstFromWeights(wVec: DoubleArray, topK: Int, wrap: UnaryOp?, nonLinearSynergy: Boolean): ExprNode {
            val rankedIdx = (0 until dim)
                .filter { abs(wVec[it]) >= 0.08 }
                .sortedByDescending { abs(wVec[it]) }
                .take(topK.coerceIn(2, dim))
                .ifEmpty { listOf(0, 1) }

            var expr: ExprNode = ExprNode.Binary(
                BinaryOp.MUL,
                ExprNode.Const(round2(wVec[rankedIdx[0]])),
                ExprNode.Var(features[rankedIdx[0]])
            )
            for (k in 1 until rankedIdx.size) {
                val fIdx = rankedIdx[k]
                val term = ExprNode.Binary(
                    BinaryOp.MUL,
                    ExprNode.Const(round2(wVec[fIdx])),
                    ExprNode.Var(features[fIdx])
                )
                expr = ExprNode.Binary(BinaryOp.ADD, expr, term)
            }
            if (nonLinearSynergy && rankedIdx.size >= 2) {
                val syn = ExprNode.Binary(
                    BinaryOp.MUL,
                    ExprNode.Const(round2(0.45 * kotlin.math.sign(wVec[rankedIdx[0]] * wVec[rankedIdx[1]]))),
                    ExprNode.Binary(
                        BinaryOp.MUL,
                        ExprNode.Var(features[rankedIdx[0]]),
                        ExprNode.Var(features[rankedIdx[1]])
                    )
                )
                expr = ExprNode.Binary(BinaryOp.ADD, expr, syn)
            }
            return SymbolicFormula.simplify(if (wrap != null) ExprNode.Unary(wrap, expr) else expr)
        }

        val profiles = listOf(
            nwGlobal to nwResid,
            nwRecent to nwResid,
            nwBlend to nwResid,
            nwGlobal to nwRecent
        )
        val out = ArrayList<SymbolicGenome>(maxCount)
        val localSigs = HashSet<String>(maxCount * 2)
        val kChoices = listOf(4, 6, 8, 5)
        val slotChoices = listOf(3, 2, 4)
        val tempChoices = listOf(0.28, 0.45, 0.68)
        val wraps = listOf<UnaryOp?>(null, UnaryOp.TANH, UnaryOp.SIGMOID)

        var idx = 0
        for ((pIdx, pair) in profiles.withIndex()) {
            val (wA, wB) = pair
            for (k in kChoices) {
                for (wrap in wraps) {
                    for (slots in slotChoices) {
                        if (out.size >= maxCount) break
                        val exprA = buildAstFromWeights(wA, k, wrap, nonLinearSynergy = (idx + paradigmHint) % 2 == 1)
                        val exprB = buildAstFromWeights(wB, max(3, k - 1), if (idx % 2 == 0) UnaryOp.TANH else UnaryOp.SIGMOID, nonLinearSynergy = false)
                        val genome = SymbolicGenome(
                            id = "ai_ridge_g${generation}_m${pIdx}_$idx",
                            generation = generation.coerceAtLeast(1),
                            parentIds = listOf("causal_ridge_solver"),
                            primaryExpr = exprA,
                            secondaryExpr = exprB,
                            primarySlots = slots,
                            samplingTemp = tempChoices[idx % tempChoices.size],
                            exploreNoise = 0.0,
                            seed = sessionSalt * 43_991L + generation * 1_117L + idx * 29L + 11L
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
            if (out.size >= maxCount) break
        }
        return out
    }

    fun fitPrecisionRidgeGenomes(
        mode: PrecisionTargetMode,
        discoverySteps: List<PrecisionRawStep>,
        generation: Int,
        sessionSalt: Long,
        seenSignatures: Set<String> = emptySet(),
        maxCount: Int = 24
    ): List<PrecisionGenome> {
        if (discoverySteps.isEmpty()) return emptyList()
        val fCount = PrecisionFeature.entries.size
        val stepCount = discoverySteps.size

        val xTxGlobal = Array(fCount) { DoubleArray(fCount) }
        val xTyGlobal = DoubleArray(fCount)
        val xTxRecent = Array(fCount) { DoubleArray(fCount) }
        val xTyRecent = DoubleArray(fCount)
        val row = DoubleArray(fCount)

        for ((sIdx, step) in discoverySteps.withIndex()) {
            val mat = step.matrix
            val age = (stepCount - 1 - sIdx).toDouble()
            val recencyW = exp(-age * 0.085)
            for (n in 1..90) {
                val isTarget = if (mode == PrecisionTargetMode.FIRST_BALL) (n == step.actualPos1) else step.actualMask[n]
                val y = if (isTarget) 1.0 else 0.0
                val sampleW = if (isTarget) (if (mode == PrecisionTargetMode.FIRST_BALL) 6.0 else 2.6) else 1.0
                for (fi in 0 until fCount) {
                    val v = mat.get(fi, n)
                    row[fi] = if (v.isFinite()) v else 0.0
                }
                for (i in 0 until fCount) {
                    val vi = row[i]
                    if (vi == 0.0) continue
                    val wVi = vi * sampleW
                    val wViR = wVi * recencyW
                    xTyGlobal[i] += wVi * y
                    xTyRecent[i] += wViR * y
                    for (j in i until fCount) {
                        val vj = row[j]
                        if (vj != 0.0) {
                            xTxGlobal[i][j] += wVi * vj
                            xTxRecent[i][j] += wViR * vj
                        }
                    }
                }
            }
        }
        for (i in 0 until fCount) {
            for (j in 0 until i) {
                xTxGlobal[i][j] = xTxGlobal[j][i]
                xTxRecent[i][j] = xTxRecent[j][i]
            }
        }

        val lambda = (stepCount * 2.2).coerceIn(6.0, 95.0)
        val wGlobal = solveLinearRidge(xTxGlobal, xTyGlobal, fCount, lambda)
        val wRecent = solveLinearRidge(xTxRecent, xTyRecent, fCount, (lambda * 0.65).coerceAtLeast(5.0))

        // Complémentarité Mode 2 (ou stabilisation Pos1 pour Mode 1) : apprentissage sur les cibles manquées par Top-1
        val xTxSec = Array(fCount) { DoubleArray(fCount) }
        val xTySec = DoubleArray(fCount)
        for (step in discoverySteps) {
            val mat = step.matrix
            var bestBall = 1
            var bestS = Double.NEGATIVE_INFINITY
            for (n in 1..90) {
                var s = 0.0
                for (fi in 0 until fCount) s += wGlobal[fi] * mat.get(fi, n)
                if (s > bestS) {
                    bestS = s
                    bestBall = n
                }
            }
            for (n in 1..90) {
                val residualTarget = if (mode == PrecisionTargetMode.FIRST_BALL) {
                    (n == step.actualPos1 && bestBall != step.actualPos1)
                } else {
                    (step.actualMask[n] && n != bestBall)
                }
                val y = if (residualTarget) 1.0 else 0.0
                val sw = if (residualTarget) 4.0 else 1.0
                for (fi in 0 until fCount) {
                    val v = mat.get(fi, n)
                    row[fi] = if (v.isFinite()) v else 0.0
                }
                for (i in 0 until fCount) {
                    val vi = row[i]
                    if (vi == 0.0) continue
                    val wVi = vi * sw
                    xTySec[i] += wVi * y
                    for (j in i until fCount) {
                        val vj = row[j]
                        if (vj != 0.0) xTxSec[i][j] += wVi * vj
                    }
                }
            }
        }
        for (i in 0 until fCount) {
            for (j in 0 until i) xTxSec[i][j] = xTxSec[j][i]
        }
        val wSecRaw = solveLinearRidge(xTxSec, xTySec, fCount, lambda * 1.1)

        fun norm(v: DoubleArray): List<Double> {
            var m = 1e-6
            for (x in v) if (abs(x) > m) m = abs(x)
            val scale = 2.6 / m
            return List(v.size) { i -> round2((v[i] * scale).coerceIn(-3.5, 3.5)) }
        }

        val nGlob = norm(wGlobal)
        val nRec = norm(wRecent)
        val nSec = norm(wSecRaw)
        val nBlend = List(fCount) { i -> round2(0.55 * nGlob[i] + 0.45 * nRec[i]) }

        fun sparsify(vec: List<Double>, k: Int): List<Double> {
            val keep = vec.indices.sortedByDescending { abs(vec[it]) }.take(k.coerceIn(2, fCount)).toSet()
            return List(fCount) { i -> if (i in keep) vec[i] else 0.0 }
        }

        val weightPairs = listOf(
            nGlob to nSec,
            nRec to nSec,
            nBlend to nSec,
            nGlob to nRec
        )
        val out = ArrayList<PrecisionGenome>(maxCount)
        val localSigs = HashSet<String>(maxCount * 2)
        val kList = listOf(4, 6, 9, fCount)
        val transforms = PrecisionTransform.entries
        val boosts = listOf(0.15, 0.45, 0.75)
        val temps = listOf(0.24, 0.40, 0.65, 0.90)

        var idx = 0
        for ((wpIdx, pair) in weightPairs.withIndex()) {
            val (w1Base, w2Base) = pair
            for (k in kList) {
                val w1 = sparsify(w1Base, k)
                val w2 = sparsify(w2Base, min(k + 2, fCount))
                for (tr in transforms) {
                    for (boost in boosts) {
                        if (out.size >= maxCount) break
                        val genome = PrecisionGenome(
                            id = "prec_${mode.name}_ai_g${generation}_m${wpIdx}_${idx++}",
                            mode = mode,
                            generation = generation.coerceAtLeast(1),
                            weights = w1,
                            secondaryWeights = w2,
                            transform = tr,
                            interactionBoost = boost,
                            samplingTemp = temps[idx % temps.size],
                            exploreNoise = 0.0
                        )
                        val sig = genome.signatureKey()
                        if (sig !in seenSignatures && localSigs.add(sig)) {
                            out.add(genome)
                        }
                    }
                    if (out.size >= maxCount) break
                }
                if (out.size >= maxCount) break
            }
            if (out.size >= maxCount) break
        }
        return out
    }

    private fun round2(v: Double): Double = round(v * 100.0) / 100.0
}
