package com.example.domain.util

import com.example.data.local.LotoResultEntity
import com.example.domain.engine.NextDrawTarget
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale
import java.util.TimeZone
import kotlin.math.floor

data class ThreeWeekWindow(
    val startMs: Long,
    val endMs: Long,
    val draws: List<LotoResultEntity>,
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

    // Moyenne synodique lunaire (Meeus) : 29.53058867 jours
    private const val SYNODIC_MONTH_MS = 29.53058867 * 86_400_000.0
    // Nouvelle lune astronomique de reference : 2000-01-06 18:14 UTC
    private const val REF_NEW_MOON_MS = 947_182_440_000.0

    private val beninTimeFormat = object : ThreadLocal<SimpleDateFormat>() {
        override fun initialValue(): SimpleDateFormat {
            return SimpleDateFormat("HH:mm", Locale.FRANCE).apply {
                timeZone = beninTz
            }
        }
    }

    private val dayAndDateFormat = object : ThreadLocal<SimpleDateFormat>() {
        override fun initialValue(): SimpleDateFormat {
            return SimpleDateFormat("EEEE dd/MM/yyyy", Locale.FRANCE).apply {
                timeZone = beninTz
            }
        }
    }

    private val shortDateFormat = object : ThreadLocal<SimpleDateFormat>() {
        override fun initialValue(): SimpleDateFormat {
            return SimpleDateFormat("dd/MM/yyyy", Locale.FRANCE).apply {
                timeZone = beninTz
            }
        }
    }

    private val dayKeyFormat = object : ThreadLocal<SimpleDateFormat>() {
        override fun initialValue(): SimpleDateFormat {
            return SimpleDateFormat("yyyy-MM-dd", Locale.US).apply {
                timeZone = beninTz
            }
        }
    }

    fun formatFrenchDayAndDate(timestamp: Long): String {
        val raw = dayAndDateFormat.get()!!.format(Date(timestamp))
        return raw.replaceFirstChar {
            if (it.isLowerCase()) it.titlecase(Locale.FRANCE) else it.toString()
        }
    }

    fun formatShortDate(timestamp: Long): String {
        return shortDateFormat.get()!!.format(Date(timestamp))
    }

    fun formatDayKey(timestamp: Long): String {
        return dayKeyFormat.get()!!.format(Date(timestamp))
    }

    fun dayOfWeek(timestamp: Long): Int {
        val cal = Calendar.getInstance(beninTz, Locale.FRANCE)
        cal.timeInMillis = timestamp
        return cal.get(Calendar.DAY_OF_WEEK)
    }

    fun fastDayNumber(dayKey: String): Long {
        if (dayKey.length < 10) return Long.MIN_VALUE
        fun digit(idx: Int): Int {
            val c = dayKey[idx]
            return if (c in '0'..'9') c - '0' else -1
        }
        val y0 = digit(0); val y1 = digit(1); val y2 = digit(2); val y3 = digit(3)
        val m0 = digit(5); val m1 = digit(6)
        val d0 = digit(8); val d1 = digit(9)
        if ((y0 or y1 or y2 or y3 or m0 or m1 or d0 or d1) < 0) return Long.MIN_VALUE
        val y = y0 * 1000 + y1 * 100 + y2 * 10 + y3
        val m = m0 * 10 + m1
        val d = d0 * 10 + d1
        val yAdj = y - (if (m <= 2) 1 else 0)
        val era = (if (yAdj >= 0) yAdj else yAdj - 399) / 400
        val yoe = yAdj - era * 400
        val doy = (153 * (m + (if (m > 2) -3 else 9)) + 2) / 5 + d - 1
        val doe = yoe * 365 + yoe / 4 - yoe / 100 + doy
        return era * 146097L + doe - 719468L
    }

    /**
     * Verifie si deux cles yyyy-MM-dd representent exactement deux jours calendaires consecutifs.
     */
    fun areConsecutiveCalendarDays(earlierDayKey: String, laterDayKey: String): Boolean {
        val e1 = fastDayNumber(earlierDayKey)
        val e2 = fastDayNumber(laterDayKey)
        if (e1 != Long.MIN_VALUE && e2 != Long.MIN_VALUE) {
            return e2 == e1 + 1L
        }
        return try {
            val fmt = dayKeyFormat.get()!!
            val d1 = fmt.parse(earlierDayKey)?.time ?: return false
            val d2 = fmt.parse(laterDayKey)?.time ?: return false
            val diffHours = (d2 - d1) / 3_600_000.0
            diffHours in 20.0..28.0
        } catch (_: Exception) {
            false
        }
    }

    /**
     * Retourne l'octant lunaire astronomique (0..7) a partir d'un timestamp UTC verifie.
     */
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

    fun normalizeHoraire(draw: LotoResultEntity): String {
        val candidate = canonicalHoraireToken(draw.time.trim().uppercase(Locale.ROOT))
        if (candidate.endsWith("H")) return candidate
        val fromGame = canonicalHoraireToken(draw.game.substringAfterLast(" ", "").trim().uppercase(Locale.ROOT))
        if (fromGame.endsWith("H")) return fromGame
        return candidate.ifBlank { draw.game.trim().uppercase(Locale.ROOT) }
    }

    fun normalizeGame(draw: LotoResultEntity): String {
        val g = draw.game.trim()
        return if (g.isNotBlank()) g else "Inconnu"
    }

    fun formatBeninTime(timestamp: Long): String {
        return beninTimeFormat.get()!!.format(Date(timestamp))
    }

    fun parseHoraireMinutes(horaire: String): Int? {
        val match = HORAIRE_REGEX.find(horaire.trim()) ?: return null
        val hour = match.groupValues[1].toIntOrNull() ?: return null
        val minute = match.groupValues.getOrNull(2)?.toIntOrNull() ?: 0
        if (hour !in 0..24 || minute !in 0..59) return null
        val normalizedHour = if (hour == 24) 0 else hour
        return normalizedHour * 60 + minute
    }

    fun resolveSlidingWindow(
        draws: List<LotoResultEntity>,
        days: Int = THREE_WEEKS_DAYS,
        referenceNowMs: Long = System.currentTimeMillis()
    ): ThreeWeekWindow {
        val safeDays = days.coerceAtLeast(1)
        val windowMs = safeDays.toLong() * 86_400_000L
        val sorted = draws
            .distinctBy { it.uniqueId }
            .sortedWith(compareBy<LotoResultEntity> { it.timestamp }.thenBy { it.uniqueId })

        if (sorted.isEmpty()) {
            val startMs = referenceNowMs - windowMs
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
            it.timestamp in (referenceNowMs - windowMs)..(referenceNowMs + 60_000L)
        }
        val anchorEndMs = if (hasDrawsInLiveWindow) referenceNowMs else sorted.last().timestamp
        val cutoffMs = anchorEndMs - windowMs
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

    /**
     * Retourne la fenetre stricte des 3 semaines (21 jours) ancree a la date X du jour J.
     * Si l'historique contient des tirages recents autour de referenceNowMs, ancre sur referenceNowMs
     * en excluant tout tirage futur; sinon ancre sur le dernier tirage de l'archive disponible.
     */
    fun resolveThreeWeekWindow(
        draws: List<LotoResultEntity>,
        referenceNowMs: Long = System.currentTimeMillis()
    ): ThreeWeekWindow = resolveSlidingWindow(draws, THREE_WEEKS_DAYS, referenceNowMs)

    /**
     * Determine automatiquement l'UNIQUE prochain tirage a venir en fonction de l'heure locale Benin
     * (Africa/Porto-Novo) et des tirages deja publies aujourd'hui.
     */
    fun resolveNextUpcomingDraw(
        history: List<LotoResultEntity>,
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
            if (isTomorrow) {
                add(Calendar.DAY_OF_YEAR, 1)
            }
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
