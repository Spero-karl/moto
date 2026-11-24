package com.lotobj.pc.engine

import com.lotobj.pc.model.HardwareCapabilities
import com.lotobj.pc.model.HardwareTelemetry
import java.lang.management.ManagementFactory

object HardwareMonitor {

    fun detectCapabilities(gpuAccelerator: GpuOpenClAccelerator): HardwareCapabilities {
        val runtime = Runtime.getRuntime()
        val cores = runtime.availableProcessors().coerceAtLeast(1)
        val maxHeapMb = runtime.maxMemory() / (1024L * 1024L)
        val totalPhysicalRamMb = queryTotalPhysicalMemoryMb() ?: maxHeapMb
        val gpuInfo = gpuAccelerator.deviceInfo
        val workers = gpuAccelerator.gpuWorkersInfo
        val nvidiaCount = workers.count { it.vendorType == GpuVendor.NVIDIA }
        val amdCount = workers.count { it.vendorType == GpuVendor.AMD }
        val otherCount = (workers.size - nvidiaCount - amdCount).coerceAtLeast(0)

        return HardwareCapabilities(
            logicalCores = cores,
            totalPhysicalRamMb = totalPhysicalRamMb,
            maxJvmHeapMb = maxHeapMb,
            gpuAvailable = gpuInfo.available,
            gpuDeviceName = gpuInfo.deviceName,
            gpuVendor = gpuInfo.vendor,
            gpuComputeUnits = gpuInfo.computeUnits,
            gpuGlobalMemMb = gpuInfo.globalMemMb,
            gpuMaxWorkGroupSize = gpuInfo.maxWorkGroupSize,
            gpuBackendNote = gpuInfo.statusNote,
            gpuCount = gpuInfo.gpuCount,
            nvidiaGpuCount = nvidiaCount,
            amdGpuCount = amdCount,
            otherGpuCount = otherCount
        )
    }

    @Volatile private var prevWallNs: Long = 0L
    @Volatile private var prevProcCpuNs: Long = -1L
    @Volatile private var prevEmaProcCpu: Double = 0.0

    fun sampleTelemetry(
        activeWorkers: Int,
        gpuAccelerator: GpuOpenClAccelerator,
        evalPerSec: Double,
        lastGpuBatchMs: Double,
        lastCpuBatchMs: Double
    ): HardwareTelemetry {
        val rt = Runtime.getRuntime()
        val cores = rt.availableProcessors().coerceAtLeast(1)
        val usedHeapMb = (rt.totalMemory() - rt.freeMemory()) / (1024L * 1024L)
        val maxHeapMb = rt.maxMemory() / (1024L * 1024L)

        var procCpu = -1.0
        var sysCpu = -1.0
        var procCpuTimeNs = -1L
        var totalPhysMb = maxHeapMb
        var freePhysMb = (maxHeapMb - usedHeapMb).coerceAtLeast(0L)

        val sunOsBean = ManagementFactory.getOperatingSystemMXBean() as? com.sun.management.OperatingSystemMXBean
        if (sunOsBean != null) {
            val pLoad = runCatching { sunOsBean.processCpuLoad }.getOrDefault(-1.0)
            if (pLoad >= 0.0) {
                procCpu = (pLoad * 100.0).coerceIn(0.0, 100.0)
            }
            val sLoad = runCatching { sunOsBean.cpuLoad }.getOrDefault(-1.0)
            if (sLoad >= 0.0) {
                sysCpu = (sLoad * 100.0).coerceIn(0.0, 100.0)
            }
            procCpuTimeNs = runCatching { sunOsBean.processCpuTime }.getOrDefault(-1L)
            val totMem = runCatching { sunOsBean.totalMemorySize }.getOrDefault(-1L)
            if (totMem > 0L) {
                totalPhysMb = totMem / (1024L * 1024L)
            }
            val freeMem = runCatching { sunOsBean.freeMemorySize }.getOrDefault(-1L)
            if (freeMem >= 0L) {
                freePhysMb = freeMem / (1024L * 1024L)
            }
        } else {
            val osBean = ManagementFactory.getOperatingSystemMXBean()
            val cls = osBean.javaClass
            runCatching {
                val v = (cls.getMethod("getProcessCpuLoad").invoke(osBean) as? Double) ?: -1.0
                if (v >= 0.0) procCpu = (v * 100.0).coerceIn(0.0, 100.0)
            }
            runCatching {
                val v = (cls.getMethod("getCpuLoad").invoke(osBean) as? Double) ?: -1.0
                if (v >= 0.0) sysCpu = (v * 100.0).coerceIn(0.0, 100.0)
            }
            runCatching {
                val ns = (cls.getMethod("getProcessCpuTime").invoke(osBean) as? Long) ?: -1L
                if (ns >= 0L) procCpuTimeNs = ns
            }
            runCatching {
                val bytes = (cls.getMethod("getTotalMemorySize").invoke(osBean) as? Long) ?: -1L
                if (bytes > 0L) totalPhysMb = bytes / (1024L * 1024L)
            }
            runCatching {
                val bytes = (cls.getMethod("getFreeMemorySize").invoke(osBean) as? Long) ?: -1L
                if (bytes >= 0L) freePhysMb = bytes / (1024L * 1024L)
            }
        }

        if (procCpuTimeNs < 0L) {
            procCpuTimeNs = measureThreadBeanCpuNs()
        }

        val nowWallNs = System.nanoTime()
        var deltaCpuPct = -1.0
        if (procCpuTimeNs >= 0L) {
            val oldWall = prevWallNs
            val oldProc = prevProcCpuNs
            if (oldWall > 0L && oldProc >= 0L && nowWallNs > oldWall && procCpuTimeNs >= oldProc) {
                val wallDelta = (nowWallNs - oldWall).toDouble()
                val cpuDelta = (procCpuTimeNs - oldProc).toDouble()
                deltaCpuPct = ((cpuDelta / (wallDelta * cores)) * 100.0).coerceIn(0.0, 100.0)
            }
            prevWallNs = nowWallNs
            prevProcCpuNs = procCpuTimeNs
        }

        val rawProc = when {
            procCpu > 0.5 && deltaCpuPct >= 0.0 -> maxOf(procCpu, deltaCpuPct)
            deltaCpuPct >= 0.0 -> deltaCpuPct
            procCpu >= 0.0 -> procCpu
            else -> 0.0
        }
        val smoothedProc = if (rawProc > 0.1) {
            val nextEma = if (prevEmaProcCpu <= 0.0) rawProc else (prevEmaProcCpu * 0.25 + rawProc * 0.75)
            prevEmaProcCpu = nextEma
            maxOf(rawProc, nextEma)
        } else {
            val decayed = prevEmaProcCpu * 0.5
            prevEmaProcCpu = if (decayed < 0.2) 0.0 else decayed
            prevEmaProcCpu
        }
        val finalProcCpu = smoothedProc.coerceIn(0.0, 100.0)
        val finalSysCpu = when {
            sysCpu > 0.0 -> maxOf(sysCpu, finalProcCpu).coerceIn(0.0, 100.0)
            else -> finalProcCpu
        }

        val gpuInfo = gpuAccelerator.deviceInfo
        return HardwareTelemetry(
            processCpuPercent = finalProcCpu,
            systemCpuPercent = finalSysCpu,
            jvmUsedRamMb = usedHeapMb,
            jvmMaxRamMb = maxHeapMb,
            systemFreeRamMb = freePhysMb,
            systemTotalRamMb = totalPhysMb,
            activeWorkerThreads = activeWorkers,
            gpuActive = gpuInfo.available,
            gpuDeviceName = gpuInfo.deviceName,
            evaluationsPerSecond = evalPerSec,
            lastGpuBatchMs = lastGpuBatchMs,
            lastCpuBatchMs = lastCpuBatchMs,
            gpuCount = gpuInfo.gpuCount,
            gpuErrorNote = gpuAccelerator.lastRuntimeError
        )
    }

    private fun measureThreadBeanCpuNs(): Long {
        return runCatching {
            val tb = ManagementFactory.getThreadMXBean()
            if (!tb.isThreadCpuTimeSupported) return -1L
            if (!tb.isThreadCpuTimeEnabled) tb.isThreadCpuTimeEnabled = true
            var sum = 0L
            for (tid in tb.allThreadIds) {
                val t = tb.getThreadCpuTime(tid)
                if (t > 0L) sum += t
            }
            sum
        }.getOrDefault(-1L)
    }

    private fun queryTotalPhysicalMemoryMb(): Long? {
        val sunOsBean = ManagementFactory.getOperatingSystemMXBean() as? com.sun.management.OperatingSystemMXBean
        if (sunOsBean != null) {
            val bytes = runCatching { sunOsBean.totalMemorySize }.getOrDefault(-1L)
            if (bytes > 0L) return bytes / (1024L * 1024L)
        }
        return runCatching {
            val osBean = ManagementFactory.getOperatingSystemMXBean()
            val bytes = (osBean.javaClass.getMethod("getTotalMemorySize").invoke(osBean) as? Long) ?: return null
            if (bytes > 0L) bytes / (1024L * 1024L) else null
        }.getOrNull()
    }
}
