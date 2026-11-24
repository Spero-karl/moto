package com.lotobj.pc.engine

import com.lotobj.pc.model.PrecisionFeature
import com.lotobj.pc.model.PrecisionGenome
import com.lotobj.pc.model.PrecisionTargetMode
import com.lotobj.pc.model.RawFeature
import com.lotobj.pc.model.SymbolicGenome
import org.jocl.CL
import org.jocl.Pointer
import org.jocl.Sizeof
import org.jocl.cl_command_queue
import org.jocl.cl_context
import org.jocl.cl_context_properties
import org.jocl.cl_device_id
import org.jocl.cl_kernel
import org.jocl.cl_mem
import org.jocl.cl_platform_id
import org.jocl.cl_program
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Callable
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

enum class GpuVendor(val shortLabel: String, val waveSize: Long, val compilerMacro: String) {
    NVIDIA("NVIDIA CUDA/OpenCL", 32L, "-D GPU_VENDOR_NVIDIA=1"),
    AMD("AMD Radeon/ROCm OpenCL", 64L, "-D GPU_VENDOR_AMD=1"),
    INTEL("Intel OpenCL", 32L, "-D GPU_VENDOR_INTEL=1"),
    OTHER("OpenCL", 32L, "-D GPU_VENDOR_OTHER=1")
}

data class OpenClDeviceInfo(
    val available: Boolean,
    val deviceName: String,
    val vendor: String,
    val computeUnits: Int,
    val globalMemMb: Long,
    val maxWorkGroupSize: Long,
    val statusNote: String,
    val gpuCount: Int = if (available) 1 else 0,
    val maxClockMhz: Int = 0,
    val kernelWorkGroupSize: Long = maxWorkGroupSize,
    val vendorType: GpuVendor = GpuVendor.OTHER,
    val recommendedWaveSize: Long = 32L,
    val vendorId: Int = 0
)

class GpuBatchScores(
    val genomeCount: Int,
    val stepCount: Int,
    val rawA: FloatArray,
    val rawB: FloatArray,
    val elapsedMs: Double,
    val gpusUsed: Int = 1,
    val evaluatedMask: BooleanArray? = null,
    val runtimeErrorNote: String = ""
) {
    fun isGenomeEvaluated(genomeIdx: Int): Boolean {
        if (genomeIdx !in 0 until genomeCount) return false
        return evaluatedMask?.get(genomeIdx) ?: true
    }

    fun copyStepRawA(genomeIdx: Int, stepIdx: Int, out: DoubleArray) {
        val base = ((genomeIdx * stepCount) + stepIdx) * 90
        for (i in 0 until 90) {
            out[i + 1] = rawA[base + i].toDouble()
        }
    }

    fun copyStepRawB(genomeIdx: Int, stepIdx: Int, out: DoubleArray) {
        val base = ((genomeIdx * stepCount) + stepIdx) * 90
        for (i in 0 until 90) {
            out[i + 1] = rawB[base + i].toDouble()
        }
    }
}

class GpuStageEvalResult(
    val genomeCount: Int,
    val stepCount: Int,
    val hits: IntArray,
    val brier: FloatArray,
    val elapsedMs: Double,
    val gpusUsed: Int = 1,
    val evaluatedMask: BooleanArray? = null,
    val runtimeErrorNote: String = ""
) {
    fun isGenomeEvaluated(genomeIdx: Int): Boolean {
        if (genomeIdx !in 0 until genomeCount) return false
        return evaluatedMask?.get(genomeIdx) ?: true
    }
}

class PrecisionGpuStageResult(
    val genomeCount: Int,
    val stepCount: Int,
    val exactHits: IntArray,
    val hits: IntArray,
    val brier: FloatArray,
    val elapsedMs: Double,
    val gpusUsed: Int = 1,
    val evaluatedMask: BooleanArray? = null,
    val runtimeErrorNote: String = ""
) {
    fun isGenomeEvaluated(genomeIdx: Int): Boolean {
        if (genomeIdx !in 0 until genomeCount) return false
        return evaluatedMask?.get(genomeIdx) ?: true
    }
}

/**
 * Accélérateur Multi-GPU (NVIDIA, AMD, Intel — y compris configurations hybrides iGPU + dGPU) basé sur OpenCL (JOCL).
 * Énumère TOUS les GPUs OpenCL disponibles, compile les arbres symboliques (ExprNode) en bytecode RPN
 * et répartit chaque génération entre tous les GPUs en parallèle proportionnellement à leur puissance (CU × MHz).
 * En absence de pilote OpenCL GPU (ex. conteneur headless), bascule proprement sur 100% des cœurs CPU.
 */
class GpuOpenClAccelerator : Closeable {

    private class GpuWorker(
        val info: OpenClDeviceInfo,
        val context: cl_context,
        val commandQueue: cl_command_queue,
        val program: cl_program,
        val kernel: cl_kernel,
        @Volatile var fullKernel: cl_kernel?,
        @Volatile var precisionKernel: cl_kernel?,
        val fullLocalWorkSize: Long = 32L,
        val precLocalWorkSize: Long = 32L
    ) : Closeable {
        private var cachedKey: String = ""
        private var cachedStepCount: Int = 0
        private var cachedFeaturesMem: cl_mem? = null
        private var cachedMasksMem: cl_mem? = null
        private var cachedSaltsMem: cl_mem? = null

        private var precCachedKey: String = ""
        private var precCachedStepCount: Int = 0
        private var precFeaturesMem: cl_mem? = null
        private var precPos1Mem: cl_mem? = null
        private var precMasksMem: cl_mem? = null

        @Synchronized
        fun uploadStageFeaturesIfNeeded(stageKey: String, steps: List<CachedRawStep>, flatFeatures: FloatArray): cl_mem? {
            if (cachedKey == stageKey && cachedStepCount == steps.size && cachedFeaturesMem != null) {
                return cachedFeaturesMem
            }
            return runCatching {
                cachedFeaturesMem?.let { CL.clReleaseMemObject(it) }
                cachedMasksMem?.let { CL.clReleaseMemObject(it) }
                cachedSaltsMem?.let { CL.clReleaseMemObject(it) }
                cachedFeaturesMem = null
                cachedMasksMem = null
                cachedSaltsMem = null
                val mem = CL.clCreateBuffer(
                    context,
                    CL.CL_MEM_READ_ONLY or CL.CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_float.toLong() * flatFeatures.size,
                    Pointer.to(flatFeatures),
                    null
                )
                val flatMasks = IntArray(steps.size * 90)
                val stepSalts = LongArray(steps.size)
                for (sIdx in steps.indices) {
                    val st = steps[sIdx]
                    stepSalts[sIdx] = PcDiscoveryEngine.deterministicStepSalt(st.dayKey, st.horaire)
                    val base = sIdx * 90
                    for (n in 0 until 90) {
                        flatMasks[base + n] = if (st.actualMask[n + 1]) 1 else 0
                    }
                }
                cachedMasksMem = CL.clCreateBuffer(
                    context,
                    CL.CL_MEM_READ_ONLY or CL.CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_int.toLong() * flatMasks.size,
                    Pointer.to(flatMasks),
                    null
                )
                cachedSaltsMem = CL.clCreateBuffer(
                    context,
                    CL.CL_MEM_READ_ONLY or CL.CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_long.toLong() * stepSalts.size.coerceAtLeast(1),
                    Pointer.to(stepSalts),
                    null
                )
                cachedFeaturesMem = mem
                cachedKey = stageKey
                cachedStepCount = steps.size
                mem
            }.getOrElse { ex ->
                cachedKey = ""
                throw IllegalStateException("Upload VRAM échoué sur ${info.deviceName}: ${ex.message ?: ex.javaClass.simpleName}", ex)
            }
        }

        private var capProgOps: Int = 0
        private var capProgs: Int = 0
        private var capConsts: Int = 0
        private var capOut: Int = 0
        private var capGenomes: Int = 0
        private var capFullOut: Int = 0

        private var hostOpCodes = IntArray(0)
        private var hostOpArgs = IntArray(0)
        private var hostProgLens = IntArray(0)
        private var hostConstOffsets = IntArray(0)
        private var hostConstArr = FloatArray(0)
        private var hostSlots = IntArray(0)
        private var hostTemps = FloatArray(0)
        private var hostNoises = FloatArray(0)
        private var hostSeeds = LongArray(0)

        private var memOpCodes: cl_mem? = null
        private var memOpArgs: cl_mem? = null
        private var memProgLens: cl_mem? = null
        private var memConstOffsets: cl_mem? = null
        private var memConsts: cl_mem? = null
        private var memOutA: cl_mem? = null
        private var memOutB: cl_mem? = null
        private var memSlots: cl_mem? = null
        private var memTemps: cl_mem? = null
        private var memNoises: cl_mem? = null
        private var memSeeds: cl_mem? = null
        private var memHits: cl_mem? = null
        private var memBrier: cl_mem? = null

        private var capPrecGenomes: Int = 0
        private var capPrecWeights: Int = 0
        private var capPrecOut: Int = 0
        private var hostPrecW1 = FloatArray(0)
        private var hostPrecW2 = FloatArray(0)
        private var hostPrecTr = IntArray(0)
        private var hostPrecBoost = FloatArray(0)
        private var hostPrecTemp = FloatArray(0)
        private var memPrecW1: cl_mem? = null
        private var memPrecW2: cl_mem? = null
        private var memPrecTr: cl_mem? = null
        private var memPrecBoost: cl_mem? = null
        private var memPrecTemp: cl_mem? = null
        private var memPrecExact: cl_mem? = null
        private var memPrecHits: cl_mem? = null
        private var memPrecBrier: cl_mem? = null

        @Synchronized
        fun uploadPrecisionStageIfNeeded(stageKey: String, steps: List<PrecisionRawStep>, flatFeatures: FloatArray): cl_mem? {
            if (precCachedKey == stageKey && precCachedStepCount == steps.size && precFeaturesMem != null) {
                return precFeaturesMem
            }
            return runCatching {
                precFeaturesMem?.let { CL.clReleaseMemObject(it) }
                precPos1Mem?.let { CL.clReleaseMemObject(it) }
                precMasksMem?.let { CL.clReleaseMemObject(it) }
                precFeaturesMem = null
                precPos1Mem = null
                precMasksMem = null

                val featMem = CL.clCreateBuffer(
                    context,
                    CL.CL_MEM_READ_ONLY or CL.CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_float.toLong() * flatFeatures.size,
                    Pointer.to(flatFeatures),
                    null
                )
                val pos1Arr = IntArray(steps.size) { idx -> (steps[idx].actualPos1 - 1).coerceIn(0, 89) }
                val flatMasks = IntArray(steps.size * 90)
                for (sIdx in steps.indices) {
                    val st = steps[sIdx]
                    val base = sIdx * 90
                    for (n in 0 until 90) {
                        flatMasks[base + n] = if (st.actualMask[n + 1]) 1 else 0
                    }
                }
                precPos1Mem = CL.clCreateBuffer(
                    context,
                    CL.CL_MEM_READ_ONLY or CL.CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_int.toLong() * pos1Arr.size.coerceAtLeast(1),
                    Pointer.to(pos1Arr),
                    null
                )
                precMasksMem = CL.clCreateBuffer(
                    context,
                    CL.CL_MEM_READ_ONLY or CL.CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_int.toLong() * flatMasks.size.coerceAtLeast(1),
                    Pointer.to(flatMasks),
                    null
                )
                precFeaturesMem = featMem
                precCachedKey = stageKey
                precCachedStepCount = steps.size
                featMem
            }.getOrElse { ex ->
                precCachedKey = ""
                throw IllegalStateException("Upload VRAM Précision échoué sur ${info.deviceName}: ${ex.message ?: ex.javaClass.simpleName}", ex)
            }
        }

        private fun ensurePrecisionBuffers(gCount: Int, featCount: Int, gsCount: Int) {
            val weightTotal = (gCount * featCount).coerceAtLeast(1)
            if (weightTotal > capPrecWeights) {
                memPrecW1?.let { CL.clReleaseMemObject(it) }
                memPrecW2?.let { CL.clReleaseMemObject(it) }
                capPrecWeights = weightTotal + weightTotal / 4 + 32
                hostPrecW1 = FloatArray(capPrecWeights)
                hostPrecW2 = FloatArray(capPrecWeights)
                memPrecW1 = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY, Sizeof.cl_float.toLong() * capPrecWeights, null, null)
                memPrecW2 = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY, Sizeof.cl_float.toLong() * capPrecWeights, null, null)
            }
            if (gCount > capPrecGenomes) {
                memPrecTr?.let { CL.clReleaseMemObject(it) }
                memPrecBoost?.let { CL.clReleaseMemObject(it) }
                memPrecTemp?.let { CL.clReleaseMemObject(it) }
                capPrecGenomes = gCount + gCount / 4 + 16
                hostPrecTr = IntArray(capPrecGenomes)
                hostPrecBoost = FloatArray(capPrecGenomes)
                hostPrecTemp = FloatArray(capPrecGenomes)
                memPrecTr = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY, Sizeof.cl_int.toLong() * capPrecGenomes, null, null)
                memPrecBoost = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY, Sizeof.cl_float.toLong() * capPrecGenomes, null, null)
                memPrecTemp = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY, Sizeof.cl_float.toLong() * capPrecGenomes, null, null)
            }
            if (gsCount > capPrecOut) {
                memPrecExact?.let { CL.clReleaseMemObject(it) }
                memPrecHits?.let { CL.clReleaseMemObject(it) }
                memPrecBrier?.let { CL.clReleaseMemObject(it) }
                capPrecOut = gsCount + gsCount / 4 + 64
                memPrecExact = CL.clCreateBuffer(context, CL.CL_MEM_WRITE_ONLY, Sizeof.cl_int.toLong() * capPrecOut, null, null)
                memPrecHits = CL.clCreateBuffer(context, CL.CL_MEM_WRITE_ONLY, Sizeof.cl_int.toLong() * capPrecOut, null, null)
                memPrecBrier = CL.clCreateBuffer(context, CL.CL_MEM_WRITE_ONLY, Sizeof.cl_float.toLong() * capPrecOut, null, null)
            }
        }

        private fun ensureBuffers(progOps: Int, progCount: Int, constSize: Int, totalOutSize: Int) {
            if (progOps > capProgOps) {
                memOpCodes?.let { CL.clReleaseMemObject(it) }
                memOpArgs?.let { CL.clReleaseMemObject(it) }
                capProgOps = progOps + progOps / 4
                hostOpCodes = IntArray(capProgOps)
                hostOpArgs = IntArray(capProgOps)
                memOpCodes = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY, Sizeof.cl_int.toLong() * capProgOps, null, null)
                memOpArgs = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY, Sizeof.cl_int.toLong() * capProgOps, null, null)
            }
            if (progCount > capProgs) {
                memProgLens?.let { CL.clReleaseMemObject(it) }
                memConstOffsets?.let { CL.clReleaseMemObject(it) }
                capProgs = progCount + progCount / 4
                hostProgLens = IntArray(capProgs)
                hostConstOffsets = IntArray(capProgs)
                memProgLens = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY, Sizeof.cl_int.toLong() * capProgs, null, null)
                memConstOffsets = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY, Sizeof.cl_int.toLong() * capProgs, null, null)
            }
            val safeConsts = constSize.coerceAtLeast(1)
            if (safeConsts > capConsts) {
                memConsts?.let { CL.clReleaseMemObject(it) }
                capConsts = safeConsts + safeConsts / 2
                hostConstArr = FloatArray(capConsts)
                memConsts = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY, Sizeof.cl_float.toLong() * capConsts, null, null)
            }
            if (totalOutSize > capOut) {
                memOutA?.let { CL.clReleaseMemObject(it) }
                memOutB?.let { CL.clReleaseMemObject(it) }
                capOut = totalOutSize + totalOutSize / 4
                memOutA = CL.clCreateBuffer(context, CL.CL_MEM_WRITE_ONLY, Sizeof.cl_float.toLong() * capOut, null, null)
                memOutB = CL.clCreateBuffer(context, CL.CL_MEM_WRITE_ONLY, Sizeof.cl_float.toLong() * capOut, null, null)
            }
        }

        private fun ensureFullStageBuffers(gCount: Int, gsCount: Int) {
            if (gCount > capGenomes) {
                memSlots?.let { CL.clReleaseMemObject(it) }
                memTemps?.let { CL.clReleaseMemObject(it) }
                memNoises?.let { CL.clReleaseMemObject(it) }
                memSeeds?.let { CL.clReleaseMemObject(it) }
                capGenomes = gCount + gCount / 4 + 16
                hostSlots = IntArray(capGenomes)
                hostTemps = FloatArray(capGenomes)
                hostNoises = FloatArray(capGenomes)
                hostSeeds = LongArray(capGenomes)
                memSlots = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY, Sizeof.cl_int.toLong() * capGenomes, null, null)
                memTemps = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY, Sizeof.cl_float.toLong() * capGenomes, null, null)
                memNoises = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY, Sizeof.cl_float.toLong() * capGenomes, null, null)
                memSeeds = CL.clCreateBuffer(context, CL.CL_MEM_READ_ONLY, Sizeof.cl_long.toLong() * capGenomes, null, null)
            }
            if (gsCount > capFullOut) {
                memHits?.let { CL.clReleaseMemObject(it) }
                memBrier?.let { CL.clReleaseMemObject(it) }
                capFullOut = gsCount + gsCount / 4 + 64
                memHits = CL.clCreateBuffer(context, CL.CL_MEM_WRITE_ONLY, Sizeof.cl_int.toLong() * capFullOut, null, null)
                memBrier = CL.clCreateBuffer(context, CL.CL_MEM_WRITE_ONLY, Sizeof.cl_float.toLong() * capFullOut, null, null)
            }
        }

        @Synchronized
        fun executeSlice(
            featMem: cl_mem,
            programs: List<Pair<CompiledRpnProgram, CompiledRpnProgram>>,
            stepCount: Int,
            maxOps: Int,
            featCount: Int,
            sameDayRepOrdinal: Int
        ): Pair<FloatArray, FloatArray> {
            val gCount = programs.size
            require(gCount > 0 && stepCount > 0) { "Slice GPU vide" }

            val progCount = gCount * 2
            val progOps = progCount * maxOps
            var totalConsts = 0
            for (gIdx in 0 until gCount) {
                val (pA, pB) = programs[gIdx]
                totalConsts += pA.constants.size + pB.constants.size
            }
            val totalOutSize = gCount * stepCount * 90
            ensureBuffers(progOps, progCount, totalConsts, totalOutSize)

            var constCursor = 0
            for (gIdx in 0 until gCount) {
                val (pA, pB) = programs[gIdx]
                val slotA = gIdx * 2
                val slotB = gIdx * 2 + 1

                val lenA = pA.instructionCount.coerceAtMost(maxOps)
                hostProgLens[slotA] = lenA
                hostConstOffsets[slotA] = constCursor
                for (c in pA.constants) hostConstArr[constCursor++] = c
                System.arraycopy(pA.opCodes, 0, hostOpCodes, slotA * maxOps, lenA)
                System.arraycopy(pA.opArgs, 0, hostOpArgs, slotA * maxOps, lenA)

                val lenB = pB.instructionCount.coerceAtMost(maxOps)
                hostProgLens[slotB] = lenB
                hostConstOffsets[slotB] = constCursor
                for (c in pB.constants) hostConstArr[constCursor++] = c
                System.arraycopy(pB.opCodes, 0, hostOpCodes, slotB * maxOps, lenB)
                System.arraycopy(pB.opArgs, 0, hostOpArgs, slotB * maxOps, lenB)
            }
            if (constCursor == 0) {
                hostConstArr[0] = 0f
                constCursor = 1
            }

            val mOpCodes = memOpCodes ?: error("Buffer opCodes non initialisé")
            val mOpArgs = memOpArgs ?: error("Buffer opArgs non initialisé")
            val mProgLens = memProgLens ?: error("Buffer progLens non initialisé")
            val mConstOffsets = memConstOffsets ?: error("Buffer constOffsets non initialisé")
            val mConsts = memConsts ?: error("Buffer consts non initialisé")
            val mOutA = memOutA ?: error("Buffer outA non initialisé")
            val mOutB = memOutB ?: error("Buffer outB non initialisé")

            // CRITICAL JOCL FIX : Pointer.to(tableau Java heap) EXIGE CL.CL_TRUE (transfert synchrone),
            // sinon JOCL lève IllegalArgumentException avant même de lancer le kernel GPU.
            CL.clEnqueueWriteBuffer(commandQueue, mOpCodes, CL.CL_TRUE, 0L, Sizeof.cl_int.toLong() * progOps, Pointer.to(hostOpCodes), 0, null, null)
            CL.clEnqueueWriteBuffer(commandQueue, mOpArgs, CL.CL_TRUE, 0L, Sizeof.cl_int.toLong() * progOps, Pointer.to(hostOpArgs), 0, null, null)
            CL.clEnqueueWriteBuffer(commandQueue, mProgLens, CL.CL_TRUE, 0L, Sizeof.cl_int.toLong() * progCount, Pointer.to(hostProgLens), 0, null, null)
            CL.clEnqueueWriteBuffer(commandQueue, mConstOffsets, CL.CL_TRUE, 0L, Sizeof.cl_int.toLong() * progCount, Pointer.to(hostConstOffsets), 0, null, null)
            CL.clEnqueueWriteBuffer(commandQueue, mConsts, CL.CL_TRUE, 0L, Sizeof.cl_float.toLong() * constCursor, Pointer.to(hostConstArr), 0, null, null)

            CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem.toLong(), Pointer.to(featMem))
            CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem.toLong(), Pointer.to(mOpCodes))
            CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem.toLong(), Pointer.to(mOpArgs))
            CL.clSetKernelArg(kernel, 3, Sizeof.cl_mem.toLong(), Pointer.to(mProgLens))
            CL.clSetKernelArg(kernel, 4, Sizeof.cl_mem.toLong(), Pointer.to(mConstOffsets))
            CL.clSetKernelArg(kernel, 5, Sizeof.cl_mem.toLong(), Pointer.to(mConsts))
            CL.clSetKernelArg(kernel, 6, Sizeof.cl_mem.toLong(), Pointer.to(mOutA))
            CL.clSetKernelArg(kernel, 7, Sizeof.cl_mem.toLong(), Pointer.to(mOutB))
            CL.clSetKernelArg(kernel, 8, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(gCount)))
            CL.clSetKernelArg(kernel, 9, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(stepCount)))
            CL.clSetKernelArg(kernel, 10, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(maxOps)))
            CL.clSetKernelArg(kernel, 11, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(featCount)))
            CL.clSetKernelArg(kernel, 12, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(sameDayRepOrdinal)))

            val localSizeVal = info.kernelWorkGroupSize.coerceIn(1L, 128L)
            val alignedGlobal = ((totalOutSize.toLong() + localSizeVal - 1L) / localSizeVal) * localSizeVal
            try {
                CL.clEnqueueNDRangeKernel(
                    commandQueue, kernel, 1, null,
                    longArrayOf(alignedGlobal), longArrayOf(localSizeVal),
                    0, null, null
                )
            } catch (_: Throwable) {
                CL.clEnqueueNDRangeKernel(
                    commandQueue, kernel, 1, null,
                    longArrayOf(totalOutSize.toLong()), null,
                    0, null, null
                )
            }
            CL.clFinish(commandQueue)

            val outRawA = FloatArray(totalOutSize)
            val outRawB = FloatArray(totalOutSize)
            CL.clEnqueueReadBuffer(commandQueue, mOutA, CL.CL_TRUE, 0L, Sizeof.cl_float.toLong() * totalOutSize, Pointer.to(outRawA), 0, null, null)
            CL.clEnqueueReadBuffer(commandQueue, mOutB, CL.CL_TRUE, 0L, Sizeof.cl_float.toLong() * totalOutSize, Pointer.to(outRawB), 0, null, null)
            return outRawA to outRawB
        }

        @Synchronized
        fun executeSliceFullStage(
            featMem: cl_mem,
            genomes: List<SymbolicGenome>,
            programs: List<Pair<CompiledRpnProgram, CompiledRpnProgram>>,
            stepCount: Int,
            maxOps: Int,
            featCount: Int,
            sameDayRepOrdinal: Int
        ): Pair<IntArray, FloatArray> {
            val fk = fullKernel ?: error("Kernel full-stage indisponible")
            val masksMem = cachedMasksMem ?: error("Buffer actualMasks non initialisé")
            val saltsMem = cachedSaltsMem ?: error("Buffer stepSalts non initialisé")
            val gCount = programs.size
            require(gCount > 0 && stepCount > 0) { "Slice GPU vide" }

            val maxChunkGenomes = (49_152 / stepCount).coerceAtLeast(1)
            if (gCount > maxChunkGenomes) {
                val totalGs = gCount * stepCount
                val mergedHits = IntArray(totalGs)
                val mergedBrier = FloatArray(totalGs)
                var offset = 0
                while (offset < gCount) {
                    val end = minOf(gCount, offset + maxChunkGenomes)
                    val (subHits, subBrier) = executeSliceFullStage(
                        featMem = featMem,
                        genomes = genomes.subList(offset, end),
                        programs = programs.subList(offset, end),
                        stepCount = stepCount,
                        maxOps = maxOps,
                        featCount = featCount,
                        sameDayRepOrdinal = sameDayRepOrdinal
                    )
                    System.arraycopy(subHits, 0, mergedHits, offset * stepCount, subHits.size)
                    System.arraycopy(subBrier, 0, mergedBrier, offset * stepCount, subBrier.size)
                    offset = end
                }
                return mergedHits to mergedBrier
            }

            val progCount = gCount * 2
            val progOps = progCount * maxOps
            var totalConsts = 0
            for (gIdx in 0 until gCount) {
                val (pA, pB) = programs[gIdx]
                totalConsts += pA.constants.size + pB.constants.size
            }
            val gsCount = gCount * stepCount
            ensureBuffers(progOps, progCount, totalConsts, 1)
            ensureFullStageBuffers(gCount, gsCount)

            var constCursor = 0
            for (gIdx in 0 until gCount) {
                val (pA, pB) = programs[gIdx]
                val g = genomes[gIdx]
                val slotA = gIdx * 2
                val slotB = gIdx * 2 + 1

                hostSlots[gIdx] = g.primarySlots.coerceIn(0, 5)
                hostTemps[gIdx] = g.samplingTemp.toFloat().coerceIn(0.15f, 2.0f)
                hostNoises[gIdx] = (g.exploreNoise * 0.08).toFloat()
                hostSeeds[gIdx] = g.seed

                val lenA = pA.instructionCount.coerceAtMost(maxOps)
                hostProgLens[slotA] = lenA
                hostConstOffsets[slotA] = constCursor
                for (c in pA.constants) hostConstArr[constCursor++] = c
                System.arraycopy(pA.opCodes, 0, hostOpCodes, slotA * maxOps, lenA)
                System.arraycopy(pA.opArgs, 0, hostOpArgs, slotA * maxOps, lenA)

                val lenB = pB.instructionCount.coerceAtMost(maxOps)
                hostProgLens[slotB] = lenB
                hostConstOffsets[slotB] = constCursor
                for (c in pB.constants) hostConstArr[constCursor++] = c
                System.arraycopy(pB.opCodes, 0, hostOpCodes, slotB * maxOps, lenB)
                System.arraycopy(pB.opArgs, 0, hostOpArgs, slotB * maxOps, lenB)
            }
            if (constCursor == 0) {
                hostConstArr[0] = 0f
                constCursor = 1
            }

            val mOpCodes = memOpCodes!!
            val mOpArgs = memOpArgs!!
            val mProgLens = memProgLens!!
            val mConstOffsets = memConstOffsets!!
            val mConsts = memConsts!!
            val mSlots = memSlots!!
            val mTemps = memTemps!!
            val mNoises = memNoises!!
            val mSeeds = memSeeds!!
            val mHits = memHits!!
            val mBrier = memBrier!!

            CL.clEnqueueWriteBuffer(commandQueue, mOpCodes, CL.CL_TRUE, 0L, Sizeof.cl_int.toLong() * progOps, Pointer.to(hostOpCodes), 0, null, null)
            CL.clEnqueueWriteBuffer(commandQueue, mOpArgs, CL.CL_TRUE, 0L, Sizeof.cl_int.toLong() * progOps, Pointer.to(hostOpArgs), 0, null, null)
            CL.clEnqueueWriteBuffer(commandQueue, mProgLens, CL.CL_TRUE, 0L, Sizeof.cl_int.toLong() * progCount, Pointer.to(hostProgLens), 0, null, null)
            CL.clEnqueueWriteBuffer(commandQueue, mConstOffsets, CL.CL_TRUE, 0L, Sizeof.cl_int.toLong() * progCount, Pointer.to(hostConstOffsets), 0, null, null)
            CL.clEnqueueWriteBuffer(commandQueue, mConsts, CL.CL_TRUE, 0L, Sizeof.cl_float.toLong() * constCursor, Pointer.to(hostConstArr), 0, null, null)
            CL.clEnqueueWriteBuffer(commandQueue, mSlots, CL.CL_TRUE, 0L, Sizeof.cl_int.toLong() * gCount, Pointer.to(hostSlots), 0, null, null)
            CL.clEnqueueWriteBuffer(commandQueue, mTemps, CL.CL_TRUE, 0L, Sizeof.cl_float.toLong() * gCount, Pointer.to(hostTemps), 0, null, null)
            CL.clEnqueueWriteBuffer(commandQueue, mNoises, CL.CL_TRUE, 0L, Sizeof.cl_float.toLong() * gCount, Pointer.to(hostNoises), 0, null, null)
            CL.clEnqueueWriteBuffer(commandQueue, mSeeds, CL.CL_TRUE, 0L, Sizeof.cl_long.toLong() * gCount, Pointer.to(hostSeeds), 0, null, null)

            CL.clSetKernelArg(fk, 0, Sizeof.cl_mem.toLong(), Pointer.to(featMem))
            CL.clSetKernelArg(fk, 1, Sizeof.cl_mem.toLong(), Pointer.to(masksMem))
            CL.clSetKernelArg(fk, 2, Sizeof.cl_mem.toLong(), Pointer.to(saltsMem))
            CL.clSetKernelArg(fk, 3, Sizeof.cl_mem.toLong(), Pointer.to(mOpCodes))
            CL.clSetKernelArg(fk, 4, Sizeof.cl_mem.toLong(), Pointer.to(mOpArgs))
            CL.clSetKernelArg(fk, 5, Sizeof.cl_mem.toLong(), Pointer.to(mProgLens))
            CL.clSetKernelArg(fk, 6, Sizeof.cl_mem.toLong(), Pointer.to(mConstOffsets))
            CL.clSetKernelArg(fk, 7, Sizeof.cl_mem.toLong(), Pointer.to(mConsts))
            CL.clSetKernelArg(fk, 8, Sizeof.cl_mem.toLong(), Pointer.to(mSlots))
            CL.clSetKernelArg(fk, 9, Sizeof.cl_mem.toLong(), Pointer.to(mTemps))
            CL.clSetKernelArg(fk, 10, Sizeof.cl_mem.toLong(), Pointer.to(mNoises))
            CL.clSetKernelArg(fk, 11, Sizeof.cl_mem.toLong(), Pointer.to(mSeeds))
            CL.clSetKernelArg(fk, 12, Sizeof.cl_mem.toLong(), Pointer.to(mHits))
            CL.clSetKernelArg(fk, 13, Sizeof.cl_mem.toLong(), Pointer.to(mBrier))
            CL.clSetKernelArg(fk, 14, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(gCount)))
            CL.clSetKernelArg(fk, 15, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(stepCount)))
            CL.clSetKernelArg(fk, 16, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(maxOps)))
            CL.clSetKernelArg(fk, 17, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(featCount)))
            CL.clSetKernelArg(fk, 18, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(sameDayRepOrdinal)))

            val lws = fullLocalWorkSize.coerceAtLeast(1L)
            val globalWorkSize = gsCount.toLong() * lws
            CL.clEnqueueNDRangeKernel(
                commandQueue, fk, 1, null,
                longArrayOf(globalWorkSize), longArrayOf(lws),
                0, null, null
            )
            CL.clFinish(commandQueue)

            val outHits = IntArray(gsCount)
            val outBrier = FloatArray(gsCount)
            CL.clEnqueueReadBuffer(commandQueue, mHits, CL.CL_TRUE, 0L, Sizeof.cl_int.toLong() * gsCount, Pointer.to(outHits), 0, null, null)
            CL.clEnqueueReadBuffer(commandQueue, mBrier, CL.CL_TRUE, 0L, Sizeof.cl_float.toLong() * gsCount, Pointer.to(outBrier), 0, null, null)
            return outHits to outBrier
        }

        @Synchronized
        fun executeSlicePrecision(
            featMem: cl_mem,
            genomes: List<PrecisionGenome>,
            stepCount: Int,
            featCount: Int,
            modeOrdinal: Int
        ): Triple<IntArray, IntArray, FloatArray> {
            val pk = precisionKernel ?: error("Kernel Précision indisponible")
            val pos1Mem = precPos1Mem ?: error("Buffer actualPos1 non initialisé")
            val masksMem = precMasksMem ?: error("Buffer actualMasks Précision non initialisé")
            val gCount = genomes.size
            require(gCount > 0 && stepCount > 0) { "Slice Précision vide" }
            val maxChunkGenomes = (49_152 / stepCount).coerceAtLeast(1)
            if (gCount > maxChunkGenomes) {
                val totalGs = gCount * stepCount
                val mergedExact = IntArray(totalGs)
                val mergedHits = IntArray(totalGs)
                val mergedBrier = FloatArray(totalGs)
                var offset = 0
                while (offset < gCount) {
                    val end = minOf(gCount, offset + maxChunkGenomes)
                    val (subExact, subHits, subBrier) = executeSlicePrecision(
                        featMem = featMem,
                        genomes = genomes.subList(offset, end),
                        stepCount = stepCount,
                        featCount = featCount,
                        modeOrdinal = modeOrdinal
                    )
                    val dest = offset * stepCount
                    System.arraycopy(subExact, 0, mergedExact, dest, subExact.size)
                    System.arraycopy(subHits, 0, mergedHits, dest, subHits.size)
                    System.arraycopy(subBrier, 0, mergedBrier, dest, subBrier.size)
                    offset = end
                }
                return Triple(mergedExact, mergedHits, mergedBrier)
            }
            val gsCount = gCount * stepCount
            ensurePrecisionBuffers(gCount, featCount, gsCount)

            for (gIdx in 0 until gCount) {
                val g = genomes[gIdx]
                val wBase = gIdx * featCount
                for (fi in 0 until featCount) {
                    hostPrecW1[wBase + fi] = g.weights.getOrElse(fi) { 0.0 }.toFloat()
                    hostPrecW2[wBase + fi] = g.secondaryWeights.getOrElse(fi) { 0.0 }.toFloat()
                }
                hostPrecTr[gIdx] = g.transform.ordinal
                hostPrecBoost[gIdx] = g.interactionBoost.toFloat()
                hostPrecTemp[gIdx] = g.samplingTemp.toFloat().coerceIn(0.15f, 2.0f)
            }

            val mW1 = memPrecW1!!
            val mW2 = memPrecW2!!
            val mTr = memPrecTr!!
            val mBoost = memPrecBoost!!
            val mTemp = memPrecTemp!!
            val mExact = memPrecExact!!
            val mHits = memPrecHits!!
            val mBrier = memPrecBrier!!

            CL.clEnqueueWriteBuffer(commandQueue, mW1, CL.CL_TRUE, 0L, Sizeof.cl_float.toLong() * gCount * featCount, Pointer.to(hostPrecW1), 0, null, null)
            CL.clEnqueueWriteBuffer(commandQueue, mW2, CL.CL_TRUE, 0L, Sizeof.cl_float.toLong() * gCount * featCount, Pointer.to(hostPrecW2), 0, null, null)
            CL.clEnqueueWriteBuffer(commandQueue, mTr, CL.CL_TRUE, 0L, Sizeof.cl_int.toLong() * gCount, Pointer.to(hostPrecTr), 0, null, null)
            CL.clEnqueueWriteBuffer(commandQueue, mBoost, CL.CL_TRUE, 0L, Sizeof.cl_float.toLong() * gCount, Pointer.to(hostPrecBoost), 0, null, null)
            CL.clEnqueueWriteBuffer(commandQueue, mTemp, CL.CL_TRUE, 0L, Sizeof.cl_float.toLong() * gCount, Pointer.to(hostPrecTemp), 0, null, null)

            CL.clSetKernelArg(pk, 0, Sizeof.cl_mem.toLong(), Pointer.to(featMem))
            CL.clSetKernelArg(pk, 1, Sizeof.cl_mem.toLong(), Pointer.to(pos1Mem))
            CL.clSetKernelArg(pk, 2, Sizeof.cl_mem.toLong(), Pointer.to(masksMem))
            CL.clSetKernelArg(pk, 3, Sizeof.cl_mem.toLong(), Pointer.to(mW1))
            CL.clSetKernelArg(pk, 4, Sizeof.cl_mem.toLong(), Pointer.to(mW2))
            CL.clSetKernelArg(pk, 5, Sizeof.cl_mem.toLong(), Pointer.to(mTr))
            CL.clSetKernelArg(pk, 6, Sizeof.cl_mem.toLong(), Pointer.to(mBoost))
            CL.clSetKernelArg(pk, 7, Sizeof.cl_mem.toLong(), Pointer.to(mTemp))
            CL.clSetKernelArg(pk, 8, Sizeof.cl_mem.toLong(), Pointer.to(mExact))
            CL.clSetKernelArg(pk, 9, Sizeof.cl_mem.toLong(), Pointer.to(mHits))
            CL.clSetKernelArg(pk, 10, Sizeof.cl_mem.toLong(), Pointer.to(mBrier))
            CL.clSetKernelArg(pk, 11, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(gCount)))
            CL.clSetKernelArg(pk, 12, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(stepCount)))
            CL.clSetKernelArg(pk, 13, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(featCount)))
            CL.clSetKernelArg(pk, 14, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(modeOrdinal)))
            CL.clSetKernelArg(pk, 15, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(PrecisionFeature.POS1_FREQ_W15.ordinal)))
            CL.clSetKernelArg(pk, 16, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(PrecisionFeature.POS1_PREV_TRANSITION.ordinal)))
            CL.clSetKernelArg(pk, 17, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(PrecisionFeature.CO_LAST_RAW.ordinal)))
            CL.clSetKernelArg(pk, 18, Sizeof.cl_int.toLong(), Pointer.to(intArrayOf(PrecisionFeature.ANY_FREQ_W15.ordinal)))

            val lws = precLocalWorkSize.coerceAtLeast(1L)
            val globalWorkSize = gsCount.toLong() * lws
            CL.clEnqueueNDRangeKernel(
                commandQueue, pk, 1, null,
                longArrayOf(globalWorkSize), longArrayOf(lws),
                0, null, null
            )
            CL.clFinish(commandQueue)

            val outExact = IntArray(gsCount)
            val outHits = IntArray(gsCount)
            val outBrier = FloatArray(gsCount)
            CL.clEnqueueReadBuffer(commandQueue, mExact, CL.CL_TRUE, 0L, Sizeof.cl_int.toLong() * gsCount, Pointer.to(outExact), 0, null, null)
            CL.clEnqueueReadBuffer(commandQueue, mHits, CL.CL_TRUE, 0L, Sizeof.cl_int.toLong() * gsCount, Pointer.to(outHits), 0, null, null)
            CL.clEnqueueReadBuffer(commandQueue, mBrier, CL.CL_TRUE, 0L, Sizeof.cl_float.toLong() * gsCount, Pointer.to(outBrier), 0, null, null)
            return Triple(outExact, outHits, outBrier)
        }

        override fun close() {
            runCatching { cachedFeaturesMem?.let { CL.clReleaseMemObject(it) } }
            runCatching { cachedMasksMem?.let { CL.clReleaseMemObject(it) } }
            runCatching { cachedSaltsMem?.let { CL.clReleaseMemObject(it) } }
            runCatching { precFeaturesMem?.let { CL.clReleaseMemObject(it) } }
            runCatching { precPos1Mem?.let { CL.clReleaseMemObject(it) } }
            runCatching { precMasksMem?.let { CL.clReleaseMemObject(it) } }
            runCatching { memOpCodes?.let { CL.clReleaseMemObject(it) } }
            runCatching { memOpArgs?.let { CL.clReleaseMemObject(it) } }
            runCatching { memProgLens?.let { CL.clReleaseMemObject(it) } }
            runCatching { memConstOffsets?.let { CL.clReleaseMemObject(it) } }
            runCatching { memConsts?.let { CL.clReleaseMemObject(it) } }
            runCatching { memOutA?.let { CL.clReleaseMemObject(it) } }
            runCatching { memOutB?.let { CL.clReleaseMemObject(it) } }
            runCatching { memSlots?.let { CL.clReleaseMemObject(it) } }
            runCatching { memTemps?.let { CL.clReleaseMemObject(it) } }
            runCatching { memNoises?.let { CL.clReleaseMemObject(it) } }
            runCatching { memSeeds?.let { CL.clReleaseMemObject(it) } }
            runCatching { memHits?.let { CL.clReleaseMemObject(it) } }
            runCatching { memBrier?.let { CL.clReleaseMemObject(it) } }
            runCatching { memPrecW1?.let { CL.clReleaseMemObject(it) } }
            runCatching { memPrecW2?.let { CL.clReleaseMemObject(it) } }
            runCatching { memPrecTr?.let { CL.clReleaseMemObject(it) } }
            runCatching { memPrecBoost?.let { CL.clReleaseMemObject(it) } }
            runCatching { memPrecTemp?.let { CL.clReleaseMemObject(it) } }
            runCatching { memPrecExact?.let { CL.clReleaseMemObject(it) } }
            runCatching { memPrecHits?.let { CL.clReleaseMemObject(it) } }
            runCatching { memPrecBrier?.let { CL.clReleaseMemObject(it) } }
            runCatching { precisionKernel?.let { CL.clReleaseKernel(it) } }
            runCatching { fullKernel?.let { CL.clReleaseKernel(it) } }
            runCatching { CL.clReleaseKernel(kernel) }
            runCatching { CL.clReleaseProgram(program) }
            runCatching { CL.clReleaseCommandQueue(commandQueue) }
            runCatching { CL.clReleaseContext(context) }
            cachedFeaturesMem = null
            cachedMasksMem = null
            cachedSaltsMem = null
            precFeaturesMem = null
            precPos1Mem = null
            precMasksMem = null
            memOpCodes = null
            memOpArgs = null
            memProgLens = null
            memConstOffsets = null
            memConsts = null
            memOutA = null
            memOutB = null
            memSlots = null
            memTemps = null
            memNoises = null
            memSeeds = null
            memHits = null
            memBrier = null
            memPrecW1 = null
            memPrecW2 = null
            memPrecTr = null
            memPrecBoost = null
            memPrecTemp = null
            memPrecExact = null
            memPrecHits = null
            memPrecBrier = null
        }
    }

    private val workers = ArrayList<GpuWorker>()
    private var dispatchExecutor: ExecutorService? = null
    private var cachedHostStageKey: String = ""
    private var cachedHostStepCount: Int = 0
    private var cachedHostFlatFeatures: FloatArray? = null
    private var cachedHostPrecKey: String = ""
    private var cachedHostPrecStepCount: Int = 0
    private var cachedHostPrecFeatures: FloatArray? = null

    @Volatile
    var lastRuntimeError: String = ""
        private set

    val gpuWorkersInfo: List<OpenClDeviceInfo>
    val deviceInfo: OpenClDeviceInfo
    val gpuCount: Int get() = workers.size

    init {
        val (summaryInfo, individualInfos) = initializeAllOpenClGpus()
        deviceInfo = summaryInfo
        gpuWorkersInfo = individualInfos
        if (workers.size >= 2) {
            dispatchExecutor = Executors.newFixedThreadPool(workers.size) { r ->
                Thread(r, "opencl-gpu-dispatch").apply { isDaemon = true }
            }
        }
    }

    val isGpuReady: Boolean
        get() = deviceInfo.available && workers.isNotEmpty()

    private fun initializeAllOpenClGpus(): Pair<OpenClDeviceInfo, List<OpenClDeviceInfo>> {
        return try {
            CL.setExceptionsEnabled(true)
            val numPlatformsArray = IntArray(1)
            CL.clGetPlatformIDs(0, null, numPlatformsArray)
            val numPlatforms = numPlatformsArray[0]
            if (numPlatforms <= 0) {
                val fallback = OpenClDeviceInfo(false, "Aucun GPU OpenCL", "-", 0, 0L, 0L, "Aucune plateforme OpenCL trouvée", 0)
                return fallback to emptyList()
            }

            val platforms = arrayOfNulls<cl_platform_id>(numPlatforms)
            CL.clGetPlatformIDs(platforms.size, platforms, null)

            data class CandidateDev(
                val platform: cl_platform_id,
                val device: cl_device_id,
                val name: String,
                val vendor: String,
                val vendorId: Int,
                val vendorType: GpuVendor,
                val cus: Int,
                val memMb: Long,
                val wg: Long,
                val clockMhz: Int
            )

            val discovered = ArrayList<CandidateDev>()
            for (platform in platforms) {
                if (platform == null) continue
                val platformName = getPlatformStringParam(platform, CL.CL_PLATFORM_NAME)
                val numDev = IntArray(1)
                val status = runCatching {
                    CL.clGetDeviceIDs(platform, CL.CL_DEVICE_TYPE_GPU, 0, null, numDev)
                }.getOrNull()
                if (status == null || numDev[0] <= 0) continue

                val devs = arrayOfNulls<cl_device_id>(numDev[0])
                CL.clGetDeviceIDs(platform, CL.CL_DEVICE_TYPE_GPU, devs.size, devs, null)
                for (dev in devs) {
                    if (dev == null) continue
                    val cus = getIntParam(dev, CL.CL_DEVICE_MAX_COMPUTE_UNITS).coerceAtLeast(1)
                    val memMb = getLongParam(dev, CL.CL_DEVICE_GLOBAL_MEM_SIZE) / (1024L * 1024L)
                    val wg = getSizeParam(dev, CL.CL_DEVICE_MAX_WORK_GROUP_SIZE).coerceAtLeast(32L)
                    val clockMhz = runCatching { getIntParam(dev, CL.CL_DEVICE_MAX_CLOCK_FREQUENCY) }.getOrDefault(1000).coerceAtLeast(300)
                    val vendorId = runCatching { getIntParam(dev, CL.CL_DEVICE_VENDOR_ID) }.getOrDefault(0)
                    val name = getStringParam(dev, CL.CL_DEVICE_NAME).trim().ifBlank { "GPU OpenCL" }
                    val vendor = getStringParam(dev, CL.CL_DEVICE_VENDOR).trim().ifBlank { "OpenCL" }
                    val vendorType = detectGpuVendor(vendorId, vendor, platformName, name)
                    discovered.add(CandidateDev(platform, dev, name, vendor, vendorId, vendorType, cus, memMb, wg, clockMhz))
                }
            }

            // Trier du GPU le plus puissant au moins puissant (CU × horloge + VRAM)
            val sortedCandidates = discovered.sortedByDescending {
                it.cus.toLong() * it.clockMhz.toLong() * 10L + it.memMb
            }
            val workerInfos = ArrayList<OpenClDeviceInfo>()
            val initErrors = ArrayList<String>()

            for (cand in sortedCandidates) {
                var ctx: cl_context? = null
                var queue: cl_command_queue? = null
                var prog: cl_program? = null
                var krn: cl_kernel? = null
                var fullKrn: cl_kernel? = null
                var precKrn: cl_kernel? = null
                val attempt = runCatching {
                    val ctxProps = cl_context_properties()
                    ctxProps.addProperty(CL.CL_CONTEXT_PLATFORM.toLong(), cand.platform)
                    val err = IntArray(1)
                    ctx = CL.clCreateContext(ctxProps, 1, arrayOf(cand.device), null, null, err)
                    queue = createCommandQueueCompat(ctx!!, cand.device)
                    prog = CL.clCreateProgramWithSource(ctx!!, 1, arrayOf(OPENCL_KERNEL_SOURCE), null, err)
                    buildProgramWithFallback(prog!!, cand.device, cand.name, cand.vendorType)
                    krn = CL.clCreateKernel(prog!!, "evalSymbolicBatch", err)
                    var fullWg = 32L
                    fullKrn = runCatching {
                        val fk = CL.clCreateKernel(prog!!, "evalSymbolicFullStage", err)
                        val fkWg = getKernelWorkGroupSize(fk, cand.device, cand.wg)
                        if (fkWg >= 16L) {
                            fullWg = recommendedLocalWorkSize(cand.vendorType, fkWg)
                            fk
                        } else {
                            runCatching { CL.clReleaseKernel(fk) }
                            null
                        }
                    }.getOrNull()
                    var precWg = 32L
                    precKrn = runCatching {
                        val pk = CL.clCreateKernel(prog!!, "evalPrecisionFullStage", err)
                        val pkWg = getKernelWorkGroupSize(pk, cand.device, cand.wg)
                        if (pkWg >= 16L) {
                            precWg = recommendedLocalWorkSize(cand.vendorType, pkWg)
                            pk
                        } else {
                            runCatching { CL.clReleaseKernel(pk) }
                            null
                        }
                    }.getOrNull()
                    val kernelWg = getKernelWorkGroupSize(krn!!, cand.device, cand.wg)
                    val recWave = recommendedLocalWorkSize(cand.vendorType, kernelWg)

                    val info = OpenClDeviceInfo(
                        available = true,
                        deviceName = cand.name,
                        vendor = cand.vendor,
                        computeUnits = cand.cus,
                        globalMemMb = cand.memMb,
                        maxWorkGroupSize = cand.wg,
                        statusNote = "${cand.name} [${cand.vendorType.shortLabel}] (${cand.cus} CU @ ${cand.clockMhz} MHz • ${cand.memMb} Mo VRAM • WG=$kernelWg • Wave=$recWave)",
                        gpuCount = 1,
                        maxClockMhz = cand.clockMhz,
                        kernelWorkGroupSize = kernelWg,
                        vendorType = cand.vendorType,
                        recommendedWaveSize = recWave,
                        vendorId = cand.vendorId
                    )
                    workers.add(GpuWorker(info, ctx!!, queue!!, prog!!, krn!!, fullKrn, precKrn, fullWg, precWg))
                    workerInfos.add(info)
                }
                attempt.onFailure { ex ->
                    runCatching { precKrn?.let { CL.clReleaseKernel(it) } }
                    runCatching { fullKrn?.let { CL.clReleaseKernel(it) } }
                    runCatching { krn?.let { CL.clReleaseKernel(it) } }
                    runCatching { prog?.let { CL.clReleaseProgram(it) } }
                    runCatching { queue?.let { CL.clReleaseCommandQueue(it) } }
                    runCatching { ctx?.let { CL.clReleaseContext(it) } }
                    initErrors.add("${cand.name}: ${ex.message ?: ex.javaClass.simpleName}")
                }
            }

            if (workers.isEmpty()) {
                val diag = if (initErrors.isNotEmpty()) " (${initErrors.joinToString("; ")})" else ""
                val fallback = OpenClDeviceInfo(
                    available = false,
                    deviceName = "Mode CPU Multi-Cœurs",
                    vendor = "OpenCL GPU absent",
                    computeUnits = 0,
                    globalMemMb = 0L,
                    maxWorkGroupSize = 0L,
                    statusNote = "Aucun GPU OpenCL exploitable$diag — moteur CPU 100% cœurs actif",
                    gpuCount = 0
                )
                return fallback to emptyList()
            }

            val totalCus = workerInfos.sumOf { it.computeUnits }
            val totalMemMb = workerInfos.sumOf { it.globalMemMb }
            val maxWg = workerInfos.maxOfOrNull { it.kernelWorkGroupSize } ?: 64L
            val nvidiaCnt = workerInfos.count { it.vendorType == GpuVendor.NVIDIA }
            val amdCnt = workerInfos.count { it.vendorType == GpuVendor.AMD }
            val joinedNames = if (workerInfos.size == 1) {
                workerInfos.first().deviceName
            } else {
                "${workerInfos.size} GPUs (${workerInfos.joinToString(" + ") { it.deviceName }})"
            }
            val joinedVendors = workerInfos.map { it.vendor }.distinct().joinToString(" / ")
            val vendorSummary = buildList {
                if (nvidiaCnt > 0) add("${nvidiaCnt}× NVIDIA")
                if (amdCnt > 0) add("${amdCnt}× AMD")
                val otherCnt = workerInfos.size - nvidiaCnt - amdCnt
                if (otherCnt > 0) add("${otherCnt}× Autre")
            }.joinToString(" + ")
            val warnSuffix = if (initErrors.isNotEmpty()) " [ignoré: ${initErrors.joinToString("; ")}]" else ""
            val note = if (workerInfos.size == 1) {
                "OpenCL GPU actif [$vendorSummary] : ${workerInfos.first().deviceName} ($totalCus CU • ${totalMemMb} Mo VRAM • Wave=${workerInfos.first().recommendedWaveSize})$warnSuffix"
            } else {
                "Cluster Multi-GPU OpenCL [$vendorSummary] (${workerInfos.size} GPUs • $totalCus CU • ${totalMemMb} Mo VRAM)$warnSuffix"
            }

            OpenClDeviceInfo(
                available = true,
                deviceName = joinedNames,
                vendor = joinedVendors,
                computeUnits = totalCus,
                globalMemMb = totalMemMb,
                maxWorkGroupSize = maxWg,
                statusNote = note,
                gpuCount = workerInfos.size,
                kernelWorkGroupSize = maxWg,
                vendorType = workerInfos.first().vendorType,
                recommendedWaveSize = workerInfos.first().recommendedWaveSize
            ) to workerInfos
        } catch (t: Throwable) {
            closeResourcesSafely()
            val fallback = OpenClDeviceInfo(
                available = false,
                deviceName = "Mode CPU Multi-Cœurs",
                vendor = "Fallback",
                computeUnits = 0,
                globalMemMb = 0L,
                maxWorkGroupSize = 0L,
                statusNote = "OpenCL indisponible (${t.javaClass.simpleName}) — calcul 100% CPU multi-cœurs actif",
                gpuCount = 0
            )
            fallback to emptyList()
        }
    }

    @Suppress("DEPRECATION")
    private fun createCommandQueueCompat(ctx: cl_context, device: cl_device_id): cl_command_queue {
        val err = IntArray(1)
        return try {
            CL.clCreateCommandQueueWithProperties(ctx, device, null, err)
        } catch (_: Throwable) {
            CL.clCreateCommandQueue(ctx, device, 0L, err)
        }
    }

    private fun buildProgramWithFallback(
        program: cl_program,
        device: cl_device_id,
        deviceName: String,
        vendorType: GpuVendor
    ) {
        val preferredOpts = "${vendorType.compilerMacro} -cl-mad-enable -cl-no-signed-zeros -cl-fast-relaxed-math"
        try {
            CL.clBuildProgram(program, 1, arrayOf(device), preferredOpts, null, null)
        } catch (_: Throwable) {
            val log1 = readBuildLog(program, device)
            try {
                CL.clBuildProgram(program, 1, arrayOf(device), "-cl-fast-relaxed-math", null, null)
            } catch (secondErr: Throwable) {
                val log2 = readBuildLog(program, device)
                try {
                    CL.clBuildProgram(program, 1, arrayOf(device), "", null, null)
                } catch (thirdErr: Throwable) {
                    val log3 = readBuildLog(program, device)
                    val detail = listOf(log1, log2, log3, thirdErr.message, secondErr.message)
                        .firstOrNull { !it.isNullOrBlank() }
                        .orEmpty()
                    throw IllegalStateException("Compilation kernel OpenCL échouée sur $deviceName: ${detail.take(240)}", thirdErr)
                }
            }
        }
    }

    private fun readBuildLog(program: cl_program, device: cl_device_id): String {
        return runCatching {
            val size = LongArray(1)
            CL.clGetProgramBuildInfo(program, device, CL.CL_PROGRAM_BUILD_LOG, 0, null, size)
            val len = size[0].toInt().coerceIn(1, 4096)
            val buffer = ByteArray(len)
            CL.clGetProgramBuildInfo(program, device, CL.CL_PROGRAM_BUILD_LOG, len.toLong(), Pointer.to(buffer), null)
            String(buffer, Charsets.UTF_8).trimEnd('\u0000').trim()
        }.getOrDefault("")
    }

    private fun getKernelWorkGroupSize(kernel: cl_kernel, device: cl_device_id, fallback: Long): Long {
        return runCatching {
            val buf = ByteBuffer.allocateDirect(Sizeof.size_t).order(ByteOrder.nativeOrder())
            CL.clGetKernelWorkGroupInfo(kernel, device, CL.CL_KERNEL_WORK_GROUP_SIZE, Sizeof.size_t.toLong(), Pointer.to(buf), null)
            val queried = if (Sizeof.size_t == 8) buf.getLong(0) else buf.getInt(0).toLong()
            queried.coerceIn(16L, fallback.coerceAtLeast(16L))
        }.getOrDefault(fallback.coerceIn(16L, 64L))
    }

    @Synchronized
    private fun getOrBuildFlatFeatures(stageKey: String, steps: List<CachedRawStep>): FloatArray {
        val existing = cachedHostFlatFeatures
        if (cachedHostStageKey == stageKey && cachedHostStepCount == steps.size && existing != null) {
            return existing
        }
        val featCount = RawFeature.entries.size
        val flat = FloatArray(steps.size * featCount * 90)
        var idx = 0
        for (step in steps) {
            for (f in 0 until featCount) {
                val row = step.rawMatrix.featureRows[f]
                for (n in 1..90) {
                    flat[idx++] = row[n].toFloat()
                }
            }
        }
        cachedHostFlatFeatures = flat
        cachedHostStageKey = stageKey
        cachedHostStepCount = steps.size
        return flat
    }

    fun evaluatePopulationRawScores(
        stageKey: String,
        steps: List<CachedRawStep>,
        genomes: List<SymbolicGenome>,
        precompiled: List<Pair<CompiledRpnProgram, CompiledRpnProgram>>? = null
    ): GpuBatchScores? {
        if (!isGpuReady || steps.isEmpty() || genomes.isEmpty()) return null

        return try {
            val startNs = System.nanoTime()
            val gCount = genomes.size
            val sCount = steps.size
            val featCount = RawFeature.entries.size
            val repOrd = RawFeature.SAME_DAY_REP.ordinal
            val flatFeatures = getOrBuildFlatFeatures(stageKey, steps)

            val compiledPairs = precompiled ?: genomes.map { g ->
                SymbolicFormula.compileToRpn(g.primaryExpr) to SymbolicFormula.compileToRpn(g.secondaryExpr)
            }
            val dynamicMaxOps = compiledPairs
                .maxOfOrNull { maxOf(it.first.instructionCount, it.second.instructionCount) }
                ?.coerceIn(8, 128)
                ?: 48

            // Mono-GPU ou tout petit batch : exécution directe sur le GPU le plus puissant
            if (workers.size == 1 || gCount < workers.size * 2 || dispatchExecutor == null) {
                val w = workers.first()
                val featMem = w.uploadStageFeaturesIfNeeded(stageKey, steps, flatFeatures)
                    ?: error("Buffer VRAM nul sur ${w.info.deviceName}")
                val (outA, outB) = w.executeSlice(featMem, compiledPairs, sCount, dynamicMaxOps, featCount, repOrd)
                val elapsedMs = ((System.nanoTime() - startNs) / 1_000_000.0).coerceAtLeast(0.1)
                lastRuntimeError = ""
                return GpuBatchScores(gCount, sCount, outA, outB, elapsedMs, 1, null, "")
            }

            // Multi-GPU : répartition pondérée par (CU × Fréquence MHz) sur TOUS les GPUs simultanément
            val workerWeights = workers.map { w ->
                (w.info.computeUnits.coerceAtLeast(1).toLong() * w.info.maxClockMhz.coerceAtLeast(300)).coerceAtLeast(100L)
            }
            val totalWeight = workerWeights.sum().coerceAtLeast(1L)
            val slices = ArrayList<Triple<GpuWorker, Int, Int>>(workers.size)
            var cursor = 0
            for (wIdx in workers.indices) {
                if (cursor >= gCount) break
                val w = workers[wIdx]
                val count = if (wIdx == workers.lastIndex) {
                    gCount - cursor
                } else {
                    val share = ((gCount.toLong() * workerWeights[wIdx]) / totalWeight).toInt().coerceAtLeast(1)
                    share.coerceAtMost(gCount - cursor)
                }
                if (count > 0) {
                    slices.add(Triple(w, cursor, cursor + count))
                    cursor += count
                }
            }

            data class SliceOutcome(
                val fromIdx: Int,
                val toIdx: Int,
                val gpuName: String,
                val outPair: Pair<FloatArray, FloatArray>?,
                val error: String?
            )

            val futures = slices.map { (worker, fromIdx, toIdx) ->
                dispatchExecutor!!.submit(Callable {
                    try {
                        val featMem = worker.uploadStageFeaturesIfNeeded(stageKey, steps, flatFeatures)
                            ?: error("Upload VRAM échoué sur ${worker.info.deviceName}")
                        val subProgs = compiledPairs.subList(fromIdx, toIdx)
                        val res = worker.executeSlice(featMem, subProgs, sCount, dynamicMaxOps, featCount, repOrd)
                        SliceOutcome(fromIdx, toIdx, worker.info.deviceName, res, null)
                    } catch (ex: Throwable) {
                        SliceOutcome(fromIdx, toIdx, worker.info.deviceName, null, "${worker.info.deviceName}: ${ex.message ?: ex.javaClass.simpleName}")
                    }
                })
            }

            val totalOutSize = gCount * sCount * 90
            val mergedA = FloatArray(totalOutSize)
            val mergedB = FloatArray(totalOutSize)
            val evaluatedMask = BooleanArray(gCount)
            val floatsPerGenome = sCount * 90
            val sliceErrors = ArrayList<String>()
            var gpusSucceeded = 0

            for (f in futures) {
                val outcome = f.get()
                if (outcome.outPair != null) {
                    gpusSucceeded++
                    val destOffset = outcome.fromIdx * floatsPerGenome
                    val copyLen = (outcome.toIdx - outcome.fromIdx) * floatsPerGenome
                    System.arraycopy(outcome.outPair.first, 0, mergedA, destOffset, copyLen)
                    System.arraycopy(outcome.outPair.second, 0, mergedB, destOffset, copyLen)
                    for (gIdx in outcome.fromIdx until outcome.toIdx) {
                        evaluatedMask[gIdx] = true
                    }
                } else if (!outcome.error.isNullOrBlank()) {
                    sliceErrors.add(outcome.error)
                }
            }

            val errSummary = sliceErrors.joinToString(" | ")
            lastRuntimeError = errSummary
            if (gpusSucceeded == 0) {
                return null
            }

            val elapsedMs = ((System.nanoTime() - startNs) / 1_000_000.0).coerceAtLeast(0.1)
            GpuBatchScores(
                genomeCount = gCount,
                stepCount = sCount,
                rawA = mergedA,
                rawB = mergedB,
                elapsedMs = elapsedMs,
                gpusUsed = gpusSucceeded,
                evaluatedMask = if (gpusSucceeded == slices.size) null else evaluatedMask,
                runtimeErrorNote = errSummary
            )
        } catch (ex: Throwable) {
            lastRuntimeError = ex.message ?: ex.javaClass.simpleName
            null
        }
    }

    val isFullStageKernelReady: Boolean
        get() = isGpuReady && workers.isNotEmpty() && workers.all { it.fullKernel != null }

    fun evaluatePopulationStageDirect(
        stageKey: String,
        steps: List<CachedRawStep>,
        genomes: List<SymbolicGenome>,
        precompiled: List<Pair<CompiledRpnProgram, CompiledRpnProgram>>? = null
    ): GpuStageEvalResult? {
        if (!isFullStageKernelReady || steps.isEmpty() || genomes.isEmpty()) return null

        return try {
            val startNs = System.nanoTime()
            val gCount = genomes.size
            val sCount = steps.size
            val featCount = RawFeature.entries.size
            val repOrd = RawFeature.SAME_DAY_REP.ordinal
            val flatFeatures = getOrBuildFlatFeatures(stageKey, steps)

            val compiledPairs = precompiled ?: genomes.map { g ->
                SymbolicFormula.compileToRpn(g.primaryExpr) to SymbolicFormula.compileToRpn(g.secondaryExpr)
            }
            val dynamicMaxOps = compiledPairs
                .maxOfOrNull { maxOf(it.first.instructionCount, it.second.instructionCount) }
                ?.coerceIn(8, 128)
                ?: 48

            if (workers.size == 1 || gCount < workers.size * 2 || dispatchExecutor == null) {
                val w = workers.first()
                return try {
                    val featMem = w.uploadStageFeaturesIfNeeded(stageKey, steps, flatFeatures)
                        ?: error("Buffer VRAM nul sur ${w.info.deviceName}")
                    val (hits, brier) = w.executeSliceFullStage(
                        featMem, genomes, compiledPairs, sCount, dynamicMaxOps, featCount, repOrd
                    )
                    val elapsedMs = ((System.nanoTime() - startNs) / 1_000_000.0).coerceAtLeast(0.1)
                    lastRuntimeError = ""
                    GpuStageEvalResult(gCount, sCount, hits, brier, elapsedMs, 1, null, "")
                } catch (ex: Throwable) {
                    lastRuntimeError = "${w.info.deviceName} fullKernel: ${ex.message ?: ex.javaClass.simpleName}"
                    runCatching { w.fullKernel?.let { CL.clReleaseKernel(it) } }
                    w.fullKernel = null
                    null
                }
            }

            val workerWeights = workers.map { w ->
                (w.info.computeUnits.coerceAtLeast(1).toLong() * w.info.maxClockMhz.coerceAtLeast(300)).coerceAtLeast(100L)
            }
            val totalWeight = workerWeights.sum().coerceAtLeast(1L)
            val slices = ArrayList<Triple<GpuWorker, Int, Int>>(workers.size)
            var cursor = 0
            for (wIdx in workers.indices) {
                if (cursor >= gCount) break
                val w = workers[wIdx]
                val count = if (wIdx == workers.lastIndex) {
                    gCount - cursor
                } else {
                    val share = ((gCount.toLong() * workerWeights[wIdx]) / totalWeight).toInt().coerceAtLeast(1)
                    share.coerceAtMost(gCount - cursor)
                }
                if (count > 0) {
                    slices.add(Triple(w, cursor, cursor + count))
                    cursor += count
                }
            }

            data class FullSliceOutcome(
                val fromIdx: Int,
                val toIdx: Int,
                val outPair: Pair<IntArray, FloatArray>?,
                val error: String?
            )

            val futures = slices.map { (worker, fromIdx, toIdx) ->
                dispatchExecutor!!.submit(Callable {
                    try {
                        val featMem = worker.uploadStageFeaturesIfNeeded(stageKey, steps, flatFeatures)
                            ?: error("Upload VRAM échoué sur ${worker.info.deviceName}")
                        val subGenomes = genomes.subList(fromIdx, toIdx)
                        val subProgs = compiledPairs.subList(fromIdx, toIdx)
                        val res = worker.executeSliceFullStage(
                            featMem, subGenomes, subProgs, sCount, dynamicMaxOps, featCount, repOrd
                        )
                        FullSliceOutcome(fromIdx, toIdx, res, null)
                    } catch (ex: Throwable) {
                        FullSliceOutcome(fromIdx, toIdx, null, "${worker.info.deviceName}: ${ex.message ?: ex.javaClass.simpleName}")
                    }
                })
            }

            val totalGs = gCount * sCount
            val mergedHits = IntArray(totalGs)
            val mergedBrier = FloatArray(totalGs)
            val evaluatedMask = BooleanArray(gCount)
            val sliceErrors = ArrayList<String>()
            var gpusSucceeded = 0

            for (f in futures) {
                val outcome = f.get()
                if (outcome.outPair != null) {
                    gpusSucceeded++
                    val destOffset = outcome.fromIdx * sCount
                    val copyLen = (outcome.toIdx - outcome.fromIdx) * sCount
                    System.arraycopy(outcome.outPair.first, 0, mergedHits, destOffset, copyLen)
                    System.arraycopy(outcome.outPair.second, 0, mergedBrier, destOffset, copyLen)
                    for (gIdx in outcome.fromIdx until outcome.toIdx) {
                        evaluatedMask[gIdx] = true
                    }
                } else if (!outcome.error.isNullOrBlank()) {
                    sliceErrors.add(outcome.error)
                    val matchingWorker = slices.firstOrNull { it.second == outcome.fromIdx }?.first
                    if (matchingWorker != null && matchingWorker.fullKernel != null) {
                        runCatching { matchingWorker.fullKernel?.let { CL.clReleaseKernel(it) } }
                        matchingWorker.fullKernel = null
                    }
                }
            }

            val errSummary = sliceErrors.joinToString(" | ")
            lastRuntimeError = errSummary
            if (gpusSucceeded == 0) return null

            val elapsedMs = ((System.nanoTime() - startNs) / 1_000_000.0).coerceAtLeast(0.1)
            GpuStageEvalResult(
                genomeCount = gCount,
                stepCount = sCount,
                hits = mergedHits,
                brier = mergedBrier,
                elapsedMs = elapsedMs,
                gpusUsed = gpusSucceeded,
                evaluatedMask = if (gpusSucceeded == slices.size) null else evaluatedMask,
                runtimeErrorNote = errSummary
            )
        } catch (ex: Throwable) {
            lastRuntimeError = ex.message ?: ex.javaClass.simpleName
            null
        }
    }

    val isPrecisionKernelReady: Boolean
        get() = isGpuReady && workers.isNotEmpty() && workers.all { it.precisionKernel != null }

    @Synchronized
    private fun getOrBuildPrecisionFlatFeatures(stageKey: String, steps: List<PrecisionRawStep>): FloatArray {
        val existing = cachedHostPrecFeatures
        if (cachedHostPrecKey == stageKey && cachedHostPrecStepCount == steps.size && existing != null) {
            return existing
        }
        val featCount = PrecisionFeature.entries.size
        val flat = FloatArray(steps.size * featCount * 90)
        var idx = 0
        for (step in steps) {
            for (f in 0 until featCount) {
                val row = step.matrix.rows[f]
                for (n in 1..90) {
                    flat[idx++] = row[n].toFloat()
                }
            }
        }
        cachedHostPrecFeatures = flat
        cachedHostPrecKey = stageKey
        cachedHostPrecStepCount = steps.size
        return flat
    }

    fun evaluatePrecisionStageDirect(
        stageKey: String,
        steps: List<PrecisionRawStep>,
        genomes: List<PrecisionGenome>,
        mode: PrecisionTargetMode
    ): PrecisionGpuStageResult? {
        if (!isPrecisionKernelReady || steps.isEmpty() || genomes.isEmpty()) return null
        return try {
            val startNs = System.nanoTime()
            val gCount = genomes.size
            val sCount = steps.size
            val featCount = PrecisionFeature.entries.size
            val flatFeatures = getOrBuildPrecisionFlatFeatures(stageKey, steps)

            if (workers.size == 1 || gCount < workers.size * 2 || dispatchExecutor == null) {
                val w = workers.first()
                return try {
                    val featMem = w.uploadPrecisionStageIfNeeded(stageKey, steps, flatFeatures)
                        ?: error("Buffer VRAM Précision nul sur ${w.info.deviceName}")
                    val (exact, hits, brier) = w.executeSlicePrecision(featMem, genomes, sCount, featCount, mode.ordinal)
                    val elapsedMs = ((System.nanoTime() - startNs) / 1_000_000.0).coerceAtLeast(0.1)
                    lastRuntimeError = ""
                    PrecisionGpuStageResult(gCount, sCount, exact, hits, brier, elapsedMs, 1, null, "")
                } catch (ex: Throwable) {
                    lastRuntimeError = "${w.info.deviceName} precisionKernel: ${ex.message ?: ex.javaClass.simpleName}"
                    runCatching { w.precisionKernel?.let { CL.clReleaseKernel(it) } }
                    w.precisionKernel = null
                    null
                }
            }

            val workerWeights = workers.map { w ->
                (w.info.computeUnits.coerceAtLeast(1).toLong() * w.info.maxClockMhz.coerceAtLeast(300)).coerceAtLeast(100L)
            }
            val totalWeight = workerWeights.sum().coerceAtLeast(1L)
            val slices = ArrayList<Triple<GpuWorker, Int, Int>>(workers.size)
            var cursor = 0
            for (wIdx in workers.indices) {
                if (cursor >= gCount) break
                val w = workers[wIdx]
                val count = if (wIdx == workers.lastIndex) {
                    gCount - cursor
                } else {
                    val share = ((gCount.toLong() * workerWeights[wIdx]) / totalWeight).toInt().coerceAtLeast(1)
                    share.coerceAtMost(gCount - cursor)
                }
                if (count > 0) {
                    slices.add(Triple(w, cursor, cursor + count))
                    cursor += count
                }
            }

            data class PrecSliceOutcome(
                val fromIdx: Int,
                val toIdx: Int,
                val outTriple: Triple<IntArray, IntArray, FloatArray>?,
                val error: String?
            )

            val futures = slices.map { (worker, fromIdx, toIdx) ->
                dispatchExecutor!!.submit(Callable {
                    try {
                        val featMem = worker.uploadPrecisionStageIfNeeded(stageKey, steps, flatFeatures)
                            ?: error("Upload VRAM Précision échoué sur ${worker.info.deviceName}")
                        val subGenomes = genomes.subList(fromIdx, toIdx)
                        val res = worker.executeSlicePrecision(featMem, subGenomes, sCount, featCount, mode.ordinal)
                        PrecSliceOutcome(fromIdx, toIdx, res, null)
                    } catch (ex: Throwable) {
                        PrecSliceOutcome(fromIdx, toIdx, null, "${worker.info.deviceName}: ${ex.message ?: ex.javaClass.simpleName}")
                    }
                })
            }

            val totalGs = gCount * sCount
            val mergedExact = IntArray(totalGs)
            val mergedHits = IntArray(totalGs)
            val mergedBrier = FloatArray(totalGs)
            val evaluatedMask = BooleanArray(gCount)
            val sliceErrors = ArrayList<String>()
            var gpusSucceeded = 0

            for (f in futures) {
                val outcome = f.get()
                if (outcome.outTriple != null) {
                    gpusSucceeded++
                    val destOffset = outcome.fromIdx * sCount
                    val copyLen = (outcome.toIdx - outcome.fromIdx) * sCount
                    System.arraycopy(outcome.outTriple.first, 0, mergedExact, destOffset, copyLen)
                    System.arraycopy(outcome.outTriple.second, 0, mergedHits, destOffset, copyLen)
                    System.arraycopy(outcome.outTriple.third, 0, mergedBrier, destOffset, copyLen)
                    for (gIdx in outcome.fromIdx until outcome.toIdx) {
                        evaluatedMask[gIdx] = true
                    }
                } else if (!outcome.error.isNullOrBlank()) {
                    sliceErrors.add(outcome.error)
                    val matchingWorker = slices.firstOrNull { it.second == outcome.fromIdx }?.first
                    if (matchingWorker != null && matchingWorker.precisionKernel != null) {
                        runCatching { matchingWorker.precisionKernel?.let { CL.clReleaseKernel(it) } }
                        matchingWorker.precisionKernel = null
                    }
                }
            }

            val errSummary = sliceErrors.joinToString(" | ")
            lastRuntimeError = errSummary
            if (gpusSucceeded == 0) return null

            val elapsedMs = ((System.nanoTime() - startNs) / 1_000_000.0).coerceAtLeast(0.1)
            PrecisionGpuStageResult(
                genomeCount = gCount,
                stepCount = sCount,
                exactHits = mergedExact,
                hits = mergedHits,
                brier = mergedBrier,
                elapsedMs = elapsedMs,
                gpusUsed = gpusSucceeded,
                evaluatedMask = if (gpusSucceeded == slices.size) null else evaluatedMask,
                runtimeErrorNote = errSummary
            )
        } catch (ex: Throwable) {
            lastRuntimeError = ex.message ?: ex.javaClass.simpleName
            null
        }
    }

    private fun closeResourcesSafely() {
        runCatching { dispatchExecutor?.shutdownNow() }
        dispatchExecutor = null
        workers.forEach { w -> runCatching { w.close() } }
        workers.clear()
        cachedHostFlatFeatures = null
        cachedHostStageKey = ""
        cachedHostPrecFeatures = null
        cachedHostPrecKey = ""
    }

    override fun close() {
        closeResourcesSafely()
    }

    private fun getIntParam(device: cl_device_id, param: Int): Int {
        val values = IntArray(1)
        CL.clGetDeviceInfo(device, param, Sizeof.cl_int.toLong(), Pointer.to(values), null)
        return values[0]
    }

    private fun getLongParam(device: cl_device_id, param: Int): Long {
        val values = LongArray(1)
        CL.clGetDeviceInfo(device, param, Sizeof.cl_long.toLong(), Pointer.to(values), null)
        return values[0]
    }

    private fun getSizeParam(device: cl_device_id, param: Int): Long {
        val buf = ByteBuffer.allocateDirect(Sizeof.size_t).order(ByteOrder.nativeOrder())
        CL.clGetDeviceInfo(device, param, Sizeof.size_t.toLong(), Pointer.to(buf), null)
        return if (Sizeof.size_t == 8) buf.getLong(0) else buf.getInt(0).toLong()
    }

    private fun getStringParam(device: cl_device_id, param: Int): String {
        val size = LongArray(1)
        CL.clGetDeviceInfo(device, param, 0, null, size)
        val buffer = ByteArray(size[0].toInt().coerceAtLeast(1))
        CL.clGetDeviceInfo(device, param, buffer.size.toLong(), Pointer.to(buffer), null)
        return String(buffer, Charsets.UTF_8).trimEnd('\u0000')
    }

    private fun getPlatformStringParam(platform: cl_platform_id, param: Int): String = runCatching {
        val size = LongArray(1)
        CL.clGetPlatformInfo(platform, param, 0, null, size)
        val buffer = ByteArray(size[0].toInt().coerceAtLeast(1))
        CL.clGetPlatformInfo(platform, param, buffer.size.toLong(), Pointer.to(buffer), null)
        String(buffer, Charsets.UTF_8).trimEnd('\u0000').trim()
    }.getOrDefault("")

    companion object {
        fun detectGpuVendor(
            vendorId: Int,
            vendorStr: String,
            platformName: String = "",
            deviceName: String = ""
        ): GpuVendor {
            when (vendorId) {
                0x10DE -> return GpuVendor.NVIDIA
                0x1002, 0x1022 -> return GpuVendor.AMD
                0x8086 -> return GpuVendor.INTEL
            }
            val combined = "$vendorStr $platformName $deviceName".lowercase()
            return when {
                "nvidia" in combined || "geforce" in combined || "quadro" in combined || "rtx" in combined || "gtx" in combined || "cuda" in combined -> GpuVendor.NVIDIA
                "amd" in combined || "advanced micro devices" in combined || "radeon" in combined || "rocm" in combined || "ati" in combined -> GpuVendor.AMD
                "intel" in combined || "arc" in combined || "uhd" in combined || "iris" in combined -> GpuVendor.INTEL
                else -> GpuVendor.OTHER
            }
        }

        fun recommendedLocalWorkSize(vendor: GpuVendor, kernelMaxWg: Long): Long {
            val safeMax = kernelMaxWg.coerceAtLeast(16L)
            return when (vendor) {
                GpuVendor.AMD -> if (safeMax >= 64L) 64L else 32L.coerceAtMost(safeMax)
                else -> 32L.coerceAtMost(safeMax)
            }
        }

        private val OPENCL_KERNEL_SOURCE = """
            inline float evalRpnSlot(
                int slotIdx,
                int stepIdx,
                int numIdx,
                __global const float* features,
                __global const int* opCodes,
                __global const int* opArgs,
                __global const int* progLens,
                __global const int* constOffsets,
                __global const float* consts,
                int maxOps,
                int featCount,
                int sameDayRepOrdinal
            ) {
                float stk[32];
                int sp = 0;
                int len = progLens[slotIdx];
                if (len > maxOps) len = maxOps;
                int baseOp = slotIdx * maxOps;
                int baseConst = constOffsets[slotIdx];
                int featBase = stepIdx * featCount * 90;

                for (int i = 0; i < len; i++) {
                    int code = opCodes[baseOp + i];
                    int arg = opArgs[baseOp + i];
                    if (code == 0) {
                        float val = (arg == sameDayRepOrdinal) ? 0.0f : features[featBase + arg * 90 + numIdx];
                        if (sp < 32) stk[sp++] = val;
                    } else if (code == 1) {
                        if (sp < 32) stk[sp++] = consts[baseConst + arg];
                    } else if (code == 2) {
                        if (sp < 1) continue;
                        float x = stk[--sp];
                        float res = 0.0f;
                        switch (arg) {
                            case 0: res = -x; break;
                            case 1: res = fabs(x); break;
                            case 2: res = sqrt(fabs(x)); break;
                            case 3: res = log1p(fabs(x)); break;
                            case 4: res = exp(clamp(x, -4.0f, 4.0f)); break;
                            case 5: res = sin(clamp(x, -10.0f, 10.0f) * 3.14159265f); break;
                            case 6: res = 1.0f / (1.0f + exp(-clamp(x, -12.0f, 12.0f))); break;
                            case 7: { float c = clamp(x, -100.0f, 100.0f); res = c * c; break; }
                            case 8: res = 1.0f / (1.0f + fabs(x)); break;
                            case 9: {
                                float e2 = exp(2.0f * clamp(x, -6.0f, 6.0f));
                                res = (e2 - 1.0f) / (e2 + 1.0f);
                                break;
                            }
                            case 10: res = cos(clamp(x, -10.0f, 10.0f) * 3.14159265f); break;
                            case 11: {
                                float ax = fabs(x);
                                float r = (ax > 1e-12f) ? pow(ax, 0.33333334f) : 0.0f;
                                res = (x < 0.0f) ? -r : r;
                                break;
                            }
                            case 12: res = log1p(exp(clamp(x, -10.0f, 6.0f))); break;
                            case 13: res = sin(clamp(x, -4.0f, 4.0f) * 9.0f * 3.14159265f); break;
                            case 14: res = (x > 0.0f) ? 1.0f : 0.0f; break;
                        }
                        stk[sp++] = isfinite(res) ? clamp(res, -10000.0f, 10000.0f) : 0.0f;
                    } else if (code == 3) {
                        if (sp < 2) continue;
                        float b = stk[--sp];
                        float a = stk[--sp];
                        float res = 0.0f;
                        switch (arg) {
                            case 0: res = a + b; break;
                            case 1: res = a - b; break;
                            case 2: res = a * b; break;
                            case 3: res = a / (fabs(b) + 0.1f); break;
                            case 4: res = fmax(a, b); break;
                            case 5: res = fmin(a, b); break;
                            case 6: {
                                float baseVal = clamp(fabs(a), 1e-6f, 10.0f);
                                float expVal = clamp(fabs(b), 0.2f, 2.5f);
                                float mag = pow(baseVal, expVal);
                                res = (a < 0.0f) ? -mag : mag;
                                break;
                            }
                            case 7: {
                                float m = fabs(b) + 0.5f;
                                res = fmod(fmod(a, m) + m, m);
                                break;
                            }
                            case 8: res = sqrt(fmin(a * a + b * b, 100000000.0f)) * 0.5f; break;
                            case 9: res = 0.5f * (a + b); break;
                            case 10: res = (fabs(a) < 1e-5f && fabs(b) < 1e-5f) ? 0.0f : (atan2(a, b) / 3.14159265f); break;
                            case 11: res = 0.5f * fabs(a - b) + fmin(a, b); break;
                            case 12: res = exp(-fmin(fabs(a - b), 8.0f)); break;
                        }
                        stk[sp++] = isfinite(res) ? clamp(res, -10000.0f, 10000.0f) : 0.0f;
                    }
                }
                float out = (sp > 0) ? stk[sp - 1] : 0.0f;
                return isfinite(out) ? clamp(out, -10000.0f, 10000.0f) : 0.0f;
            }

            __kernel void evalSymbolicBatch(
                __global const float* features,
                __global const int* opCodes,
                __global const int* opArgs,
                __global const int* progLens,
                __global const int* constOffsets,
                __global const float* consts,
                __global float* outRawA,
                __global float* outRawB,
                int genomeCount,
                int stepCount,
                int maxOps,
                int featCount,
                int sameDayRepOrdinal
            ) {
                int gid = get_global_id(0);
                int total = genomeCount * stepCount * 90;
                if (gid >= total) return;

                int numIdx = gid % 90;
                int rem = gid / 90;
                int stepIdx = rem % stepCount;
                int genomeIdx = rem / stepCount;

                outRawA[gid] = evalRpnSlot(
                    genomeIdx * 2, stepIdx, numIdx,
                    features, opCodes, opArgs, progLens, constOffsets, consts, maxOps, featCount, sameDayRepOrdinal
                );
                outRawB[gid] = evalRpnSlot(
                    genomeIdx * 2 + 1, stepIdx, numIdx,
                    features, opCodes, opArgs, progLens, constOffsets, consts, maxOps, featCount, sameDayRepOrdinal
                );
            }

            __kernel void evalSymbolicFullStage(
                __global const float* features,
                __global const int* actualMasks,
                __global const long* stepSalts,
                __global const int* opCodes,
                __global const int* opArgs,
                __global const int* progLens,
                __global const int* constOffsets,
                __global const float* consts,
                __global const int* slots,
                __global const float* temps,
                __global const float* noises,
                __global const long* genomeSeeds,
                __global int* outHits,
                __global float* outBrier,
                int genomeCount,
                int stepCount,
                int maxOps,
                int featCount,
                int sameDayRepOrdinal
            ) {
                int wgId = get_group_id(0);
                int lid = get_local_id(0);
                int lsize = get_local_size(0);
                int genomeIdx = wgId / stepCount;
                int stepIdx = wgId % stepCount;
                if (genomeIdx >= genomeCount) return;

                __local float lRawA[90];
                __local float lRawB[90];
                __local float lProbA[90];
                __local float lProbB[90];

                for (int numIdx = lid; numIdx < 90; numIdx += lsize) {
                    lRawA[numIdx] = evalRpnSlot(
                        genomeIdx * 2, stepIdx, numIdx,
                        features, opCodes, opArgs, progLens, constOffsets, consts, maxOps, featCount, sameDayRepOrdinal
                    );
                    lRawB[numIdx] = evalRpnSlot(
                        genomeIdx * 2 + 1, stepIdx, numIdx,
                        features, opCodes, opArgs, progLens, constOffsets, consts, maxOps, featCount, sameDayRepOrdinal
                    );
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                if (lid == 0) {
                    float tVal = temps[genomeIdx];
                    float invT = 1.0f / clamp(tVal, 0.15f, 2.0f);
                    float maxS = -1e30f;
                    int hasFinite = 0;
                    for (int n = 0; n < 90; n++) {
                        float s = lRawA[n];
                        if (isfinite(s) && (hasFinite == 0 || s > maxS)) {
                            maxS = s;
                            hasFinite = 1;
                        }
                    }
                    if (hasFinite == 0) maxS = 0.0f;

                    float expSum = 0.0f;
                    for (int n = 0; n < 90; n++) {
                        float s = lRawA[n];
                        float v = isfinite(s) ? clamp((s - maxS) * invT, -15.0f, 15.0f) : -15.0f;
                        float e = exp(v);
                        lProbA[n] = e;
                        expSum += e;
                    }
                    if (expSum > 0.0f && isfinite(expSum)) {
                        for (int n = 0; n < 90; n++) {
                            lProbA[n] = clamp((5.0f * lProbA[n]) / expSum, 1e-6f, 0.999f);
                        }
                    } else {
                        for (int n = 0; n < 90; n++) {
                            lProbA[n] = 5.0f / 90.0f;
                        }
                    }
                    float curSum = 0.0f;
                    for (int n = 0; n < 90; n++) curSum += lProbA[n];
                    float factor = 5.0f / fmax(curSum, 1e-9f);
                    for (int n = 0; n < 90; n++) {
                        lProbA[n] = clamp(lProbA[n] * factor, 1e-6f, 0.999f);
                    }
                } else if (lid == 1) {
                    float tVal = temps[genomeIdx];
                    float invT = 1.0f / clamp(tVal, 0.15f, 2.0f);
                    float maxS = -1e30f;
                    int hasFinite = 0;
                    for (int n = 0; n < 90; n++) {
                        float s = lRawB[n];
                        if (isfinite(s) && (hasFinite == 0 || s > maxS)) {
                            maxS = s;
                            hasFinite = 1;
                        }
                    }
                    if (hasFinite == 0) maxS = 0.0f;

                    float expSum = 0.0f;
                    for (int n = 0; n < 90; n++) {
                        float s = lRawB[n];
                        float v = isfinite(s) ? clamp((s - maxS) * invT, -15.0f, 15.0f) : -15.0f;
                        float e = exp(v);
                        lProbB[n] = e;
                        expSum += e;
                    }
                    if (expSum > 0.0f && isfinite(expSum)) {
                        for (int n = 0; n < 90; n++) {
                            lProbB[n] = clamp((5.0f * lProbB[n]) / expSum, 1e-6f, 0.999f);
                        }
                    } else {
                        for (int n = 0; n < 90; n++) {
                            lProbB[n] = 5.0f / 90.0f;
                        }
                    }
                    float curSum = 0.0f;
                    for (int n = 0; n < 90; n++) curSum += lProbB[n];
                    float factor = 5.0f / fmax(curSum, 1e-9f);
                    for (int n = 0; n < 90; n++) {
                        lProbB[n] = clamp(lProbB[n] * factor, 1e-6f, 0.999f);
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                if (lid == 0) {
                    float noiseScale = noises[genomeIdx];
                    if (noiseScale > 1e-6f) {
                        ulong prngState = ((ulong)genomeSeeds[genomeIdx]) ^ (((ulong)stepSalts[stepIdx]) * 2654435761UL);
                        for (int n = 0; n < 90; n++) {
                            prngState += 0x9E3779B97F4A7C15UL;
                            ulong z = prngState;
                            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
                            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
                            z = z ^ (z >> 31);
                            float rnd = ((float)(z >> 11)) * (1.0f / 9007199254740992.0f);
                            lRawA[n] = lProbA[n] + rnd * noiseScale;
                        }
                        for (int n = 0; n < 90; n++) {
                            prngState += 0x9E3779B97F4A7C15UL;
                            ulong z = prngState;
                            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
                            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
                            z = z ^ (z >> 31);
                            float rnd = ((float)(z >> 11)) * (1.0f / 9007199254740992.0f);
                            lRawB[n] = lProbB[n] + rnd * noiseScale;
                        }
                    } else {
                        for (int n = 0; n < 90; n++) {
                            lRawA[n] = lProbA[n];
                            lRawB[n] = lProbB[n];
                        }
                    }

                    uchar chosen[90];
                    for (int n = 0; n < 90; n++) chosen[n] = 0;
                    int kA = clamp(slots[genomeIdx], 0, 5);
                    int maskBase = stepIdx * 90;
                    int hits = 0;

                    for (int p = 0; p < kA; p++) {
                        int bestN = -1;
                        float bestS = -1e30f;
                        for (int n = 0; n < 90; n++) {
                            if (chosen[n] == 0 && (bestN == -1 || lRawA[n] > bestS)) {
                                bestS = lRawA[n];
                                bestN = n;
                            }
                        }
                        if (bestN >= 0) {
                            chosen[bestN] = 1;
                            if (actualMasks[maskBase + bestN] != 0) hits++;
                        }
                    }
                    for (int p = kA; p < 5; p++) {
                        int bestN = -1;
                        float bestS = -1e30f;
                        for (int n = 0; n < 90; n++) {
                            if (chosen[n] == 0 && (bestN == -1 || lRawB[n] > bestS)) {
                                bestS = lRawB[n];
                                bestN = n;
                            }
                        }
                        if (bestN >= 0) {
                            chosen[bestN] = 1;
                            if (actualMasks[maskBase + bestN] != 0) hits++;
                        }
                    }

                    float alphaA = ((float)kA) * 0.2f;
                    float alphaB = 1.0f - alphaA;
                    float brierSum = 0.0f;
                    for (int n = 0; n < 90; n++) {
                        float pMix = clamp(alphaA * lProbA[n] + alphaB * lProbB[n], 1e-6f, 0.999f);
                        float y = (actualMasks[maskBase + n] != 0) ? 1.0f : 0.0f;
                        float d = pMix - y;
                        brierSum += d * d;
                    }
                    outHits[wgId] = hits;
                    outBrier[wgId] = brierSum / 90.0f;
                }
            }

            inline float applyPrecTransform(int trOrd, float x) {
                float c = clamp(x, -12.0f, 12.0f);
                if (trOrd == 0) return c;
                if (trOrd == 1) {
                    float e2 = exp(2.0f * clamp(c, -6.0f, 6.0f));
                    return (e2 - 1.0f) / (e2 + 1.0f);
                }
                if (trOrd == 2) return 1.0f / (1.0f + exp(-c));
                float sgn = (c < 0.0f) ? -1.0f : ((c > 0.0f) ? 1.0f : 0.0f);
                return sgn * sqrt(fabs(c));
            }

            __kernel void evalPrecisionFullStage(
                __global const float* precFeatures,
                __global const int* actualPos1Arr,
                __global const int* actualMasks,
                __global const float* weights1,
                __global const float* weights2,
                __global const int* transforms,
                __global const float* boosts,
                __global const float* temps,
                __global int* outExact,
                __global int* outHits,
                __global float* outBrier,
                int genomeCount,
                int stepCount,
                int featCount,
                int modeOrdinal,
                int p1W15Idx,
                int p1TrIdx,
                int coRawIdx,
                int anyW15Idx
            ) {
                int wgId = get_group_id(0);
                int lid = get_local_id(0);
                int lsize = get_local_size(0);
                int genomeIdx = wgId / stepCount;
                int stepIdx = wgId % stepCount;
                if (genomeIdx >= genomeCount) return;

                __local float lS1[90];
                __local float lS2[90];
                __local float lP1[90];
                __local float lP2[90];

                int wBase = genomeIdx * featCount;
                int featBase = stepIdx * featCount * 90;
                int trOrd = transforms[genomeIdx];
                float boost = boosts[genomeIdx];

                for (int numIdx = lid; numIdx < 90; numIdx += lsize) {
                    float s1 = 0.0f;
                    float s2 = 0.0f;
                    for (int fi = 0; fi < featCount; fi++) {
                        float v = precFeatures[featBase + fi * 90 + numIdx];
                        s1 += weights1[wBase + fi] * v;
                        s2 += weights2[wBase + fi] * v;
                    }
                    if (boost != 0.0f) {
                        float syn = (modeOrdinal == 0)
                            ? (precFeatures[featBase + p1W15Idx * 90 + numIdx] * precFeatures[featBase + p1TrIdx * 90 + numIdx])
                            : (precFeatures[featBase + coRawIdx * 90 + numIdx] * precFeatures[featBase + anyW15Idx * 90 + numIdx]);
                        s1 += boost * syn * 4.0f;
                    }
                    lS1[numIdx] = applyPrecTransform(trOrd, s1);
                    lS2[numIdx] = applyPrecTransform(trOrd, s2);
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                if (lid == 0) {
                    float invT = 1.0f / clamp(temps[genomeIdx], 0.15f, 2.0f);
                    float maxS = -1e30f;
                    int hasFinite = 0;
                    for (int n = 0; n < 90; n++) {
                        float s = lS1[n];
                        if (isfinite(s) && (hasFinite == 0 || s > maxS)) { maxS = s; hasFinite = 1; }
                    }
                    if (hasFinite == 0) maxS = 0.0f;
                    float sum = 0.0f;
                    for (int n = 0; n < 90; n++) {
                        float e = exp(clamp((lS1[n] - maxS) * invT, -15.0f, 15.0f));
                        lP1[n] = e;
                        sum += e;
                    }
                    float safeSum = (sum > 0.0f && isfinite(sum)) ? sum : 90.0f;
                    for (int n = 0; n < 90; n++) {
                        lP1[n] = clamp(lP1[n] / safeSum, 1e-6f, 0.999f);
                    }
                } else if (lid == 1 && modeOrdinal == 1) {
                    float invT = 1.0f / clamp(temps[genomeIdx], 0.15f, 2.0f);
                    float maxS = -1e30f;
                    int hasFinite = 0;
                    for (int n = 0; n < 90; n++) {
                        float s = lS2[n];
                        if (isfinite(s) && (hasFinite == 0 || s > maxS)) { maxS = s; hasFinite = 1; }
                    }
                    if (hasFinite == 0) maxS = 0.0f;
                    float sum = 0.0f;
                    for (int n = 0; n < 90; n++) {
                        float e = exp(clamp((lS2[n] - maxS) * invT, -15.0f, 15.0f));
                        lP2[n] = e;
                        sum += e;
                    }
                    float safeSum = (sum > 0.0f && isfinite(sum)) ? sum : 90.0f;
                    for (int n = 0; n < 90; n++) {
                        lP2[n] = clamp(lP2[n] / safeSum, 1e-6f, 0.999f);
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);

                if (lid == 0) {
                    int b1 = 0;
                    float max1 = lS1[0];
                    for (int n = 1; n < 90; n++) {
                        if (lS1[n] > max1) { max1 = lS1[n]; b1 = n; }
                    }
                    int maskBase = stepIdx * 90;
                    if (modeOrdinal == 0) {
                        int targetIdx = actualPos1Arr[stepIdx];
                        int h = (b1 == targetIdx) ? 1 : 0;
                        float brierSum = 0.0f;
                        for (int n = 0; n < 90; n++) {
                            float y = (n == targetIdx) ? 1.0f : 0.0f;
                            float d = lP1[n] - y;
                            brierSum += d * d;
                        }
                        outHits[wgId] = h;
                        outExact[wgId] = h;
                        outBrier[wgId] = brierSum / 90.0f;
                    } else {
                        int b2 = (b1 == 0) ? 1 : 0;
                        float max2 = -1e30f;
                        for (int n = 0; n < 90; n++) {
                            if (n == b1) continue;
                            float blended = 0.65f * lS2[n] + 0.35f * lS1[n];
                            if (blended > max2) { max2 = blended; b2 = n; }
                        }
                        int h = ((actualMasks[maskBase + b1] != 0) ? 1 : 0) + ((actualMasks[maskBase + b2] != 0) ? 1 : 0);
                        float brierSum = 0.0f;
                        for (int n = 0; n < 90; n++) {
                            float marg = clamp(lP1[n] + lP2[n], 1e-6f, 0.999f);
                            float y = (actualMasks[maskBase + n] != 0) ? 1.0f : 0.0f;
                            float d = marg - y;
                            brierSum += d * d;
                        }
                        outHits[wgId] = h;
                        outExact[wgId] = (h == 2) ? 1 : 0;
                        outBrier[wgId] = brierSum / 90.0f;
                    }
                }
            }
        """.trimIndent()
    }
}
