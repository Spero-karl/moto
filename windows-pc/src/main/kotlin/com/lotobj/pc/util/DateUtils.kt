package com.lotobj.pc.util

import com.lotobj.pc.model.LotoDraw
import com.lotobj.pc.model.NextDrawTarget
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import kotlin.math.floor

data class ThreeWeekWindow(
    val startMs: Long,
    val endMs: Long,
    val draws: List<LotoDraw>,
    val startLabel: String,
    val endLabel: String,
    val anchorTimeLabel: String
)

object LotoDateUtils {

    val beninTz: TimeZone = TimeZone.getTimeZone("Africa/Porto-Novo")
    const val THREE_WEEKS_DAYS = 21
    private const val THREE_WEEKS_MS = THREE_WEEKS_DAYS.toLong() * 86_400_000L
    private val HORAIRE_REGEX = Regex("""(\d{1,2})\s*H(?:\s*(\d{1,2}))?""", RegexOption.IGNORE_CASE)
    private val DEFAULT_BENIN_SCHEDULE = listOf("00H", "8H", "11H", "14H", "18H", "20H")

    private const val SYNODIC_MONTH_MS = 29.53058867 * 86_400_000.0
    private const val REF_NEW_MOON_MS = 947_182_440_000.0

    private val beninTimeFormat = ThreadLocal.withInitial {
        SimpleDateFormat("HH:mm", Locale.FRANCE).apply { timeZone = beninTz }
    }

    private val dayAndDateFormat = ThreadLocal.withInitial {
        SimpleDateFormat("EEEE dd/MM/yyyy", Locale.FRANCE).apply { timeZone = beninTz }
    }

    private val shortDateFormat = ThreadLocal.withInitial {
        SimpleDateFormat("dd/MM/yyyy", Locale.FRANCE).apply { timeZone = beninTz }
    }

    private val dayKeyFormat = ThreadLocal.withInitial {
        SimpleDateFormat("yyyy-MM-dd", Locale.US).apply { timeZone = beninTz }
    }

    fun formatFrenchDayAndDate(timestamp: Long): String {
        val raw = dayAndDateFormat.get().format(Date(timestamp))
        return raw.replaceFirstChar {
            if (it.isLowerCase()) it.titlecase(Locale.FRANCE) else it.toString()
        }
    }

    fun formatShortDate(timestamp: Long): String = shortDateFormat.get().format(Date(timestamp))
    fun formatDayKey(timestamp: Long): String = dayKeyFormat.get().format(Date(timestamp))
    fun formatBeninTime(timestamp: Long): String = beninTimeFormat.get().format(Date(timestamp))

    fun dayOfWeek(timestamp: Long): Int {
        val cal = Calendar.getInstance(beninTz, Locale.FRANCE)
        cal.timeInMillis = timestamp
        return cal.get(Calendar.DAY_OF_WEEK)
    }

    fun areConsecutiveCalendarDays(earlierDayKey: String, laterDayKey: String): Boolean {
        return try {
            val fmt = dayKeyFormat.get()
            val d1 = fmt.parse(earlierDayKey)?.time ?: return false
            val d2 = fmt.parse(laterDayKey)?.time ?: return false
            val diffHours = (d2 - d1) / 3_600_000.0
            diffHours in 20.0..28.0
        } catch (_: Exception) {
            false
        }
    }

    fun lunarPhaseOctant(timestamp: Long): Int {
        val elapsed = timestamp.toDouble() - REF_NEW_MOON_MS
        val phase = ((elapsed % SYNODIC_MONTH_MS) + SYNODIC_MONTH_MS) % SYNODIC_MONTH_MS
        val fraction = phase / SYNODIC_MONTH_MS
        return floor(fraction * 8.0).toInt().coerceIn(0, 7)
    }

    private fun canonicalHoraireToken(raw: String): String {
        return when (raw) {
            "0H", "00H", "24H" -> "00H"
            else -> raw
        }
    }

    fun normalizeHoraire(draw: LotoDraw): String {
        val candidate = canonicalHoraireToken(draw.time.trim().uppercase(Locale.ROOT))
        if (candidate.endsWith("H")) return candidate
        val fromGame = canonicalHoraireToken(draw.game.substringAfterLast(" ", "").trim().uppercase(Locale.ROOT))
        if (fromGame.endsWith("H")) return fromGame
        return candidate.ifBlank { draw.game.trim().uppercase(Locale.ROOT) }
    }

    fun normalizeGame(draw: LotoDraw): String {
        val g = draw.game.trim()
        return if (g.isNotBlank()) g else "Inconnu"
    }

    fun parseHoraireMinutes(horaire: String): Int? {
        val match = HORAIRE_REGEX.find(horaire.trim()) ?: return null
        val hour = match.groupValues[1].toIntOrNull() ?: return null
        val minute = match.groupValues.getOrNull(2)?.toIntOrNull() ?: 0
        if (hour !in 0..24 || minute !in 0..59) return null
        val normalizedHour = if (hour == 24) 0 else hour
        return normalizedHour * 60 + minute
    }

    fun resolveThreeWeekWindow(
        draws: List<LotoDraw>,
        referenceNowMs: Long = System.currentTimeMillis()
    ): ThreeWeekWindow {
        val sorted = draws
            .distinctBy { it.uniqueId }
            .sortedWith(compareBy<LotoDraw> { it.timestamp }.thenBy { it.uniqueId })

        if (sorted.isEmpty()) {
            val startMs = referenceNowMs - THREE_WEEKS_MS
            return ThreeWeekWindow(
                startMs = startMs,
                endMs = referenceNowMs,
                draws = emptyList(),
                startLabel = formatShortDate(startMs),
                endLabel = formatShortDate(referenceNowMs),
                anchorTimeLabel = formatBeninTime(referenceNowMs)
            )
        }

        val hasDrawsInLiveWindow = sorted.any {
            it.timestamp in (referenceNowMs - THREE_WEEKS_MS)..(referenceNowMs + 60_000L)
        }
        val anchorEndMs = if (hasDrawsInLiveWindow) referenceNowMs else sorted.last().timestamp
        val cutoffMs = anchorEndMs - THREE_WEEKS_MS
        val filtered = sorted.filter { it.timestamp in cutoffMs..anchorEndMs }

        return ThreeWeekWindow(
            startMs = cutoffMs,
            endMs = anchorEndMs,
            draws = filtered,
            startLabel = formatShortDate(cutoffMs),
            endLabel = formatShortDate(anchorEndMs),
            anchorTimeLabel = formatBeninTime(anchorEndMs)
        )
    }

    fun resolveNextUpcomingDraw(
        history: List<LotoDraw>,
        nowMs: Long = System.currentTimeMillis()
    ): NextDrawTarget {
        val threeWeeksDraws = resolveThreeWeekWindow(history, nowMs).draws
        val scheduleSource = threeWeeksDraws.ifEmpty { history }
        val observedSlots = scheduleSource
            .mapNotNull { draw ->
                val h = normalizeHoraire(draw)
                val mins = parseHoraireMinutes(h) ?: return@mapNotNull null
                h to mins
            }
            .distinctBy { it.first }

        val effectiveSlots = (if (observedSlots.isNotEmpty()) {
            observedSlots
        } else {
            DEFAULT_BENIN_SCHEDULE.map { it to (parseHoraireMinutes(it) ?: 0) }
        }).sortedWith(compareBy<Pair<String, Int>> { it.second }.thenBy { it.first })

        val todayKey = formatDayKey(nowMs)
        val todayDrawnHoraires = history
            .asSequence()
            .filter { formatDayKey(it.timestamp) == todayKey }
            .map { normalizeHoraire(it) }
            .toSet()

        val calNow = Calendar.getInstance(beninTz, Locale.FRANCE).apply {
            timeInMillis = nowMs
        }
        val nowMinutes = calNow.get(Calendar.HOUR_OF_DAY) * 60 + calNow.get(Calendar.MINUTE)

        val nextToday = effectiveSlots.firstOrNull { (horaire, slotMinutes) ->
            slotMinutes > nowMinutes && horaire !in todayDrawnHoraires
        }

        val isTomorrow = nextToday == null
        val chosenSlot = nextToday ?: effectiveSlots.first()
        val targetCal = Calendar.getInstance(beninTz, Locale.FRANCE).apply {
            timeInMillis = nowMs
            if (isTomorrow) add(Calendar.DAY_OF_YEAR, 1)
            set(Calendar.HOUR_OF_DAY, chosenSlot.second / 60)
            set(Calendar.MINUTE, chosenSlot.second % 60)
            set(Calendar.SECOND, 0)
            set(Calendar.MILLISECOND, 0)
        }

        return NextDrawTarget(
            horaire = chosenSlot.first,
            targetTimestamp = targetCal.timeInMillis,
            dayLabel = formatFrenchDayAndDate(targetCal.timeInMillis),
            currentBeninTimeLabel = formatBeninTime(nowMs),
            isTomorrow = isTomorrow
        )
    }
}
