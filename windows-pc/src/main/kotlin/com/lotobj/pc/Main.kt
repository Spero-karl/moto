package com.lotobj.pc

import com.lotobj.pc.engine.PcWorkstationCoordinator
import com.lotobj.pc.ui.MainWindow
import java.awt.GraphicsEnvironment
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import javax.swing.JOptionPane
import javax.swing.JScrollPane
import javax.swing.JTextArea
import javax.swing.SwingUtilities

private fun writeCrashLog(prefix: String, error: Throwable): File? {
    return runCatching {
        val logDir = File(System.getProperty("user.home"), ".LotoBJ_PC")
        logDir.mkdirs()
        val logFile = File(logDir, "crash.log")
        val sw = StringWriter()
        error.printStackTrace(PrintWriter(sw))
        val stamp = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US).format(Date())
        val body = buildString {
            appendLine("=== [$stamp] $prefix ===")
            appendLine("OS: ${System.getProperty("os.name")} (${System.getProperty("os.arch")})")
            appendLine("Java: ${System.getProperty("java.version")} (${System.getProperty("java.vendor")})")
            appendLine("Max Heap: ${Runtime.getRuntime().maxMemory() / (1024 * 1024)} Mo")
            appendLine(sw.toString())
            appendLine()
        }
        logFile.appendText(body, Charsets.UTF_8)
        logFile
    }.getOrNull()
}

private fun showStartupErrorDialog(title: String, error: Throwable) {
    val logFile = writeCrashLog(title, error)
    System.err.println("[$title] ${error::class.java.name}: ${error.message}")
    error.printStackTrace(System.err)

    if (GraphicsEnvironment.isHeadless()) return

    val sw = StringWriter()
    error.printStackTrace(PrintWriter(sw))
    val msg = buildString {
        appendLine("Impossible de démarrer ou d'exécuter Loto BJ PC :")
        appendLine("${error::class.java.simpleName}: ${error.message ?: "Erreur inconnue"}")
        if (logFile != null) {
            appendLine()
            appendLine("Journal complet enregistré dans :")
            appendLine(logFile.absolutePath)
        }
        appendLine()
        appendLine(sw.toString().take(1800))
    }

    val show = {
        val area = JTextArea(msg, 16, 64).apply {
            isEditable = false
            lineWrap = true
            wrapStyleWord = true
        }
        JOptionPane.showMessageDialog(
            null,
            JScrollPane(area),
            "Erreur Loto BJ PC — $title",
            JOptionPane.ERROR_MESSAGE
        )
    }

    runCatching {
        if (SwingUtilities.isEventDispatchThread()) show() else SwingUtilities.invokeAndWait(show)
    }
}

fun main() {
    System.setProperty("awt.useSystemAAFontSettings", "on")
    System.setProperty("swing.aatext", "true")

    Thread.setDefaultUncaughtExceptionHandler { thread, throwable ->
        writeCrashLog("Exception Thread ${thread.name}", throwable)
        System.err.println("[Uncaught ${thread.name}] ${throwable::class.java.name}: ${throwable.message}")
        throwable.printStackTrace(System.err)
    }

    try {
        SwingUtilities.invokeAndWait {
            MainWindow.installLookAndFeel()
            val coordinator = PcWorkstationCoordinator()
            val window = MainWindow(coordinator)
            window.isVisible = true
        }
    } catch (t: Throwable) {
        showStartupErrorDialog("Échec d'initialisation", t)
        kotlin.system.exitProcess(1)
    }
}
