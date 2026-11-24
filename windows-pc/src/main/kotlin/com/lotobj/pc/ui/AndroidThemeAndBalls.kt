package com.lotobj.pc.ui

import com.formdev.flatlaf.FlatDarkLaf
import java.awt.BasicStroke
import java.awt.Color
import java.awt.Component
import java.awt.Dimension
import java.awt.Font
import java.awt.Graphics
import java.awt.Graphics2D
import java.awt.Insets
import java.awt.RenderingHints
import java.awt.geom.Arc2D
import java.awt.geom.Ellipse2D
import java.awt.geom.RoundRectangle2D
import javax.swing.BorderFactory
import javax.swing.JComponent
import javax.swing.JLabel
import javax.swing.JList
import javax.swing.JPanel
import javax.swing.ListCellRenderer
import javax.swing.UIManager
import javax.swing.border.AbstractBorder
import javax.swing.border.EmptyBorder

/**
 * Palette Material 3 Dark exacte reprise de l'application Android Loto BJ (Color.kt).
 */
object AndroidM3Colors {
    val Background = Color(0x1C1B1F)
    val OnBackground = Color(0xE6E1E5)
    val PrimaryContainer = Color(0xEADDFF)
    val OnPrimaryContainer = Color(0x21005D)
    val Primary = Color(0xD0BCFF)
    val OnPrimary = Color(0x381E72)
    val SecondaryContainer = Color(0x4F378B)
    val OnSecondaryContainer = Color(0xEADDFF)
    val SurfaceVariant = Color(0x2B2930)
    val SurfaceElevated = Color(0x34313C)
    val Outline = Color(0x49454F)
    val OutlineVariant = Color(0x938F99)
    val Surface = Color(0x1C1B1F)
    val OnSurface = Color(0xE6E1E5)
    val OnSurfaceVariant = Color(0xCAC4D0)
    val Tertiary = Color(0x6EE7B7)
    val OnTertiary = Color(0x003825)
    val TertiaryContainer = Color(0x0B4F38)
    val OnTertiaryContainer = Color(0xA7F3D0)
    val Gold = Color(0xF6B93B)
    val ErrorRed = Color(0xEF5350)
}

class RoundedBorder(
    private val color: Color,
    private val radius: Int = 16,
    private val thickness: Float = 1f
) : AbstractBorder() {
    override fun paintBorder(c: Component?, g: Graphics?, x: Int, y: Int, width: Int, height: Int) {
        val g2 = g?.create() as? Graphics2D ?: return
        try {
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
            g2.color = color
            g2.stroke = BasicStroke(thickness)
            g2.draw(
                RoundRectangle2D.Float(
                    x + thickness / 2f,
                    y + thickness / 2f,
                    width - thickness,
                    height - thickness,
                    radius.toFloat(),
                    radius.toFloat()
                )
            )
        } finally {
            g2.dispose()
        }
    }

    override fun getBorderInsets(c: Component?): Insets = Insets(4, 4, 4, 4)
}

enum class BallStyle {
    FEATURED,
    NORMAL,
    MATCHED,
    FORMULA_A,
    FORMULA_B
}

/**
 * Boule de loto circulaire anti-aliasée identique au composant Compose LotoBall / Confidence Ring.
 */
class LotoBallBadge(
    private val number: Int,
    private val style: BallStyle = BallStyle.NORMAL,
    private val sizePx: Int = 38,
    private val confidenceRingFraction: Float? = null
) : JComponent() {

    init {
        val dim = Dimension(sizePx, sizePx)
        preferredSize = dim
        minimumSize = dim
        maximumSize = dim
        isOpaque = false
    }

    override fun paintComponent(g: Graphics) {
        val g2 = g.create() as Graphics2D
        try {
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
            g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON)

            val pad = if (confidenceRingFraction != null) 5f else 2f
            val d = minOf(width, height) - pad * 2f
            val x = (width - d) / 2f
            val y = (height - d) / 2f

            if (confidenceRingFraction != null) {
                val trackD = minOf(width, height) - 4f
                val tx = (width - trackD) / 2f
                val ty = (height - trackD) / 2f
                g2.stroke = BasicStroke(3.5f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND)
                g2.color = Color(0x49454F)
                g2.draw(Arc2D.Float(tx, ty, trackD, trackD, 0f, 360f, Arc2D.OPEN))

                val sweep = (confidenceRingFraction.coerceIn(0.08f, 1f)) * 360f
                g2.color = when (style) {
                    BallStyle.FORMULA_A -> AndroidM3Colors.Primary
                    BallStyle.FORMULA_B, BallStyle.MATCHED -> AndroidM3Colors.Tertiary
                    else -> AndroidM3Colors.Gold
                }
                g2.draw(Arc2D.Float(tx, ty, trackD, trackD, 90f, -sweep, Arc2D.OPEN))
            }

            val bg = when (style) {
                BallStyle.FEATURED -> AndroidM3Colors.PrimaryContainer
                BallStyle.MATCHED -> AndroidM3Colors.TertiaryContainer
                BallStyle.FORMULA_A -> AndroidM3Colors.SecondaryContainer
                BallStyle.FORMULA_B -> Color(0x164E3F)
                BallStyle.NORMAL -> AndroidM3Colors.Surface
            }
            val border = when (style) {
                BallStyle.FEATURED -> AndroidM3Colors.Primary
                BallStyle.MATCHED -> AndroidM3Colors.Tertiary
                BallStyle.FORMULA_A -> AndroidM3Colors.Primary
                BallStyle.FORMULA_B -> AndroidM3Colors.Tertiary
                BallStyle.NORMAL -> AndroidM3Colors.OutlineVariant
            }
            val fg = when (style) {
                BallStyle.FEATURED -> AndroidM3Colors.OnPrimaryContainer
                BallStyle.MATCHED -> AndroidM3Colors.OnTertiaryContainer
                BallStyle.FORMULA_A -> AndroidM3Colors.OnSecondaryContainer
                BallStyle.FORMULA_B -> AndroidM3Colors.Tertiary
                BallStyle.NORMAL -> AndroidM3Colors.OnSurface
            }

            g2.color = bg
            g2.fill(Ellipse2D.Float(x, y, d, d))
            g2.color = border
            g2.stroke = BasicStroke(if (style == BallStyle.NORMAL) 1.2f else 1.8f)
            g2.draw(Ellipse2D.Float(x, y, d, d))

            val txt = number.toString().padStart(2, '0')
            val fontSize = (d * 0.42f).toInt().coerceIn(11, 20)
            g2.font = Font("Segoe UI", Font.BOLD, fontSize)
            g2.color = fg
            val fm = g2.fontMetrics
            val textX = (width - fm.stringWidth(txt)) / 2
            val textY = (height - fm.height) / 2 + fm.ascent
            g2.drawString(txt, textX, textY)
        } finally {
            g2.dispose()
        }
    }
}

/**
 * Barre de comparaison visuelle horizontale avec ligne pointillée hasard (5,6%).
 */
class ProbabilityBarCanvas(
    private val fraction: Float,
    private val randomFraction: Float,
    private val isSelected: Boolean
) : JComponent() {
    init {
        preferredSize = Dimension(180, 16)
        minimumSize = Dimension(80, 16)
        isOpaque = false
    }

    override fun paintComponent(g: Graphics) {
        val g2 = g.create() as Graphics2D
        try {
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
            g2.color = AndroidM3Colors.Surface
            g2.fillRoundRect(0, 0, width, height, 8, 8)

            val barW = (width * fraction.coerceIn(0.03f, 1f)).toInt()
            g2.color = if (isSelected) AndroidM3Colors.Tertiary else Color(0x88D0BCFF.toInt(), true)
            g2.fillRoundRect(0, 0, barW, height, 8, 8)

            val rx = (width * randomFraction.coerceIn(0.02f, 0.95f)).toInt()
            g2.color = AndroidM3Colors.OutlineVariant
            g2.stroke = BasicStroke(1.8f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0f, floatArrayOf(4f, 3f), 0f)
            g2.drawLine(rx, 0, rx, height)
        } finally {
            g2.dispose()
        }
    }
}

object AndroidSwingTheme {
    fun install() {
        runCatching {
            FlatDarkLaf.setup()
            UIManager.put("Panel.background", AndroidM3Colors.Background)
            UIManager.put("Viewport.background", AndroidM3Colors.Background)
            UIManager.put("ScrollPane.background", AndroidM3Colors.Background)
            UIManager.put("TabbedPane.background", AndroidM3Colors.SurfaceVariant)
            UIManager.put("TabbedPane.selectedBackground", AndroidM3Colors.SecondaryContainer)
            UIManager.put("TabbedPane.selectedForeground", AndroidM3Colors.OnSecondaryContainer)
            UIManager.put("TabbedPane.underlineColor", AndroidM3Colors.Primary)
            UIManager.put("Table.background", AndroidM3Colors.SurfaceVariant)
            UIManager.put("Table.alternateRowColor", AndroidM3Colors.Background)
            UIManager.put("Table.foreground", AndroidM3Colors.OnSurface)
            UIManager.put("Table.gridColor", AndroidM3Colors.Outline)
            UIManager.put("TableHeader.background", AndroidM3Colors.SurfaceElevated)
            UIManager.put("TableHeader.foreground", AndroidM3Colors.Primary)
            UIManager.put("TextArea.background", AndroidM3Colors.SurfaceVariant)
            UIManager.put("TextArea.foreground", AndroidM3Colors.OnSurface)
            UIManager.put("ComboBox.background", AndroidM3Colors.SurfaceVariant)
            UIManager.put("ComboBox.foreground", AndroidM3Colors.OnSurface)
            UIManager.put("Button.arc", 12)
            UIManager.put("Component.arc", 10)
            UIManager.put("ProgressBar.foreground", AndroidM3Colors.Tertiary)
            UIManager.put("ProgressBar.background", AndroidM3Colors.SurfaceVariant)
        }
    }

    fun makeCard(bg: Color = AndroidM3Colors.SurfaceVariant, borderColor: Color = AndroidM3Colors.Outline, radius: Int = 14): JPanel {
        return JPanel().apply {
            isOpaque = true
            background = bg
            border = BorderFactory.createCompoundBorder(
                RoundedBorder(borderColor, radius, 1f),
                EmptyBorder(8, 10, 8, 10)
            )
        }
    }

    fun <T> simpleComboRenderer(labelFn: (T) -> String): ListCellRenderer<T> =
        object : ListCellRenderer<T> {
            private val delegate = javax.swing.DefaultListCellRenderer()
            override fun getListCellRendererComponent(
                list: JList<out T>?,
                value: T?,
                index: Int,
                isSelected: Boolean,
                cellHasFocus: Boolean
            ): Component {
                val c = delegate.getListCellRendererComponent(
                    list,
                    value?.let(labelFn).orEmpty(),
                    index,
                    isSelected,
                    cellHasFocus
                )
                return c
            }
        }
}
