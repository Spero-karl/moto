package com.example.data.repository

import com.example.data.local.LotoResultEntity
import com.example.domain.util.LotoDateUtils
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull
import java.text.SimpleDateFormat
import java.util.Locale

data class ArchiveImportResult(
    val entities: List<LotoResultEntity>,
    val totalRows: Int,
    val acceptedCount: Int,
    val rejectedCount: Int,
    val errors: List<String>
)

object ArchiveImporter {

    private val datePatterns = listOf(
        "yyyy-MM-dd'T'HH:mm:ss",
        "yyyy-MM-dd HH:mm",
        "yyyy-MM-dd",
        "dd/MM/yyyy HH:mm",
        "dd/MM/yyyy",
        "dd-MM-yyyy HH:mm",
        "dd-MM-yyyy"
    )

    fun parseAndValidate(rawContent: String, fileName: String): ArchiveImportResult {
        val trimmed = rawContent.trim()
        if (trimmed.isEmpty()) {
            return ArchiveImportResult(emptyList(), 0, 0, 0, listOf("Fichier vide"))
        }

        return if (fileName.endsWith(".json", ignoreCase = true) || trimmed.startsWith("{") || trimmed.startsWith("[")) {
            parseJsonArchive(trimmed)
        } else {
            parseCsvArchive(trimmed)
        }
    }

    private fun parseJsonArchive(jsonStr: String): ArchiveImportResult {
        val errors = mutableListOf<String>()
        val accepted = LinkedHashMap<String, LotoResultEntity>()
        var total = 0

        val rootElement = try {
            Json.parseToJsonElement(jsonStr)
        } catch (e: Exception) {
            return ArchiveImportResult(emptyList(), 0, 0, 1, listOf("JSON invalide : ${e.message}"))
        }

        val array: JsonArray = when (rootElement) {
            is JsonArray -> rootElement
            is JsonObject -> rootElement["drawResults"]?.jsonArray
                ?: rootElement["draws"]?.jsonArray
                ?: rootElement["data"]?.jsonArray
                ?: return ArchiveImportResult(emptyList(), 0, 0, 1, listOf("Aucune liste de tirages trouvée dans le JSON"))
            else -> return ArchiveImportResult(emptyList(), 0, 0, 1, listOf("Structure JSON non supportée"))
        }

        for ((index, elem) in array.withIndex()) {
            total++
            val obj = elem as? JsonObject
            if (obj == null) {
                errors.add("Ligne ${index + 1} : objet attendu")
                continue
            }
            val entity = parseJsonObject(obj)
            if (entity == null) {
                errors.add("Ligne ${index + 1} : date ou 5 numéros distincts (1..90) invalides")
            } else {
                accepted[entity.uniqueId] = entity
            }
        }

        val list = accepted.values.toList()
        return ArchiveImportResult(
            entities = list,
            totalRows = total,
            acceptedCount = list.size,
            rejectedCount = total - list.size,
            errors = errors.take(10)
        )
    }

    private fun parseJsonObject(obj: JsonObject): LotoResultEntity? {
        val rawTs = obj["closureDate"]?.jsonPrimitive?.longOrNull
            ?: obj["timestamp"]?.jsonPrimitive?.longOrNull
            ?: obj["date"]?.jsonPrimitive?.contentOrNull?.let { parseDateStringToMillis(it, obj["time"]?.jsonPrimitive?.contentOrNull) }

        if (rawTs == null || rawTs <= 0L) return null

        val nums: List<Int> = when {
            obj["drawResults"] != null -> parseNumberString(obj["drawResults"]?.jsonPrimitive?.contentOrNull)
            obj["winningNumbers"] != null -> parseNumberString(obj["winningNumbers"]?.jsonPrimitive?.contentOrNull)
            obj["numbers"] is JsonArray -> obj["numbers"]!!.jsonArray.mapNotNull { it.jsonPrimitive.intOrNull }
            obj["num1"] != null -> listOfNotNull(
                obj["num1"]?.jsonPrimitive?.intOrNull,
                obj["num2"]?.jsonPrimitive?.intOrNull,
                obj["num3"]?.jsonPrimitive?.intOrNull,
                obj["num4"]?.jsonPrimitive?.intOrNull,
                obj["num5"]?.jsonPrimitive?.intOrNull
            )
            else -> emptyList()
        }

        if (nums.size != 5 || nums.distinct().size != 5 || nums.any { it !in 1..90 }) return null

        val name = obj["name"]?.jsonPrimitive?.contentOrNull?.trim().orEmpty()
        val type = obj["type"]?.jsonPrimitive?.contentOrNull?.trim()
            ?: obj["game"]?.jsonPrimitive?.contentOrNull?.trim()
            ?: name.substringBefore(" ").trim().ifBlank { "Loto" }

        val timePart = obj["time"]?.jsonPrimitive?.contentOrNull?.trim()?.takeIf { it.isNotBlank() }
            ?: if (name.startsWith(type, ignoreCase = true)) name.substring(type.length).trim() else name

        val id = obj["id"]?.jsonPrimitive?.contentOrNull?.takeIf { it.isNotBlank() }
        val uniqueId = if (id != null) "lnb_$id" else "arch_${rawTs}_${type}_${timePart}"

        return LotoResultEntity(
            uniqueId = uniqueId,
            date = LotoDateUtils.formatFrenchDayAndDate(rawTs),
            time = timePart,
            game = type,
            num1 = nums[0],
            num2 = nums[1],
            num3 = nums[2],
            num4 = nums[3],
            num5 = nums[4],
            timestamp = rawTs
        )
    }

    private fun parseCsvArchive(csvStr: String): ArchiveImportResult {
        val lines = csvStr.lineSequence()
            .map { it.trim() }
            .filter { it.isNotEmpty() }
            .toList()

        if (lines.isEmpty()) {
            return ArchiveImportResult(emptyList(), 0, 0, 0, listOf("CSV vide"))
        }

        val accepted = LinkedHashMap<String, LotoResultEntity>()
        val errors = mutableListOf<String>()
        var total = 0

        for ((idx, line) in lines.withIndex()) {
            // Ignore la ligne d'en-tete si presente
            if (idx == 0 && (line.contains("date", ignoreCase = true) || line.contains("num", ignoreCase = true))) {
                continue
            }
            total++
            val delimiter = if (line.count { it == ';' } >= line.count { it == ',' }) ';' else ','
            val cols = splitCsvLine(line, delimiter)

            val parsed = parseCsvColumns(cols)
            if (parsed == null) {
                errors.add("Ligne ${idx + 1} : format invalide ($line)")
            } else {
                accepted[parsed.uniqueId] = parsed
            }
        }

        val list = accepted.values.toList()
        return ArchiveImportResult(
            entities = list,
            totalRows = total,
            acceptedCount = list.size,
            rejectedCount = total - list.size,
            errors = errors.take(10)
        )
    }

    private fun parseCsvColumns(cols: List<String>): LotoResultEntity? {
        if (cols.size < 6) return null

        // Formats supportes :
        // A) date, game, time, n1, n2, n3, n4, n5 (8 cols)
        // B) date, name/time, n1, n2, n3, n4, n5 (7 cols)
        // C) date, n1, n2, n3, n4, n5 (6 cols)
        val ts: Long?
        val game: String
        val time: String
        val nums: List<Int>

        when {
            cols.size >= 8 -> {
                game = cols[1].trim().ifBlank { "Loto" }
                time = cols[2].trim()
                ts = parseDateStringToMillis(cols[0].trim(), time)
                nums = cols.subList(3, 8).mapNotNull { it.trim().toIntOrNull() }
            }
            cols.size == 7 -> {
                val label = cols[1].trim()
                game = label.substringBefore(" ").trim().ifBlank { "Loto" }
                time = label.substringAfter(" ", label).trim()
                ts = parseDateStringToMillis(cols[0].trim(), time)
                nums = cols.subList(2, 7).mapNotNull { it.trim().toIntOrNull() }
            }
            else -> {
                game = "Loto"
                time = ""
                ts = parseDateStringToMillis(cols[0].trim(), null)
                nums = cols.subList(1, 6).mapNotNull { it.trim().toIntOrNull() }
            }
        }

        if (ts == null || ts <= 0L) return null
        if (nums.size != 5 || nums.distinct().size != 5 || nums.any { it !in 1..90 }) return null

        val uniqueId = "csv_${ts}_${game}_${time}"
        return LotoResultEntity(
            uniqueId = uniqueId,
            date = LotoDateUtils.formatFrenchDayAndDate(ts),
            time = time,
            game = game,
            num1 = nums[0],
            num2 = nums[1],
            num3 = nums[2],
            num4 = nums[3],
            num5 = nums[4],
            timestamp = ts
        )
    }

    private fun splitCsvLine(line: String, delimiter: Char): List<String> {
        val result = mutableListOf<String>()
        val sb = StringBuilder()
        var inQuotes = false
        for (ch in line) {
            when {
                ch == '"' -> inQuotes = !inQuotes
                ch == delimiter && !inQuotes -> {
                    result.add(sb.toString())
                    sb.clear()
                }
                else -> sb.append(ch)
            }
        }
        result.add(sb.toString())
        return result
    }

    private fun parseNumberString(raw: String?): List<Int> {
        if (raw.isNullOrBlank()) return emptyList()
        return raw.split(Regex("[,\\s\\-;]+"))
            .mapNotNull { it.trim().toIntOrNull() }
    }

    fun parseDateStringToMillis(dateStr: String, horaireHint: String? = null): Long? {
        val clean = dateStr.trim()
        clean.toLongOrNull()?.let { epoch ->
            return if (epoch > 1_000_000_000_000L) epoch else epoch * 1000L
        }

        val hourOffset = horaireHint
            ?.uppercase(Locale.ROOT)
            ?.removeSuffix("H")
            ?.trim()
            ?.toIntOrNull()
            ?.coerceIn(0, 23)

        for (pattern in datePatterns) {
            try {
                val sdf = SimpleDateFormat(pattern, Locale.FRANCE).apply {
                    timeZone = LotoDateUtils.beninTz
                    isLenient = false
                }
                val parsed = sdf.parse(clean)
                if (parsed != null) {
                    val base = parsed.time
                    return if (!pattern.contains("HH") && hourOffset != null) {
                        base + hourOffset * 3_600_000L
                    } else {
                        base
                    }
                }
            } catch (_: Exception) {
                // Essaie le format suivant
            }
        }
        return null
    }
}
