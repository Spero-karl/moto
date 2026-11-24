package com.lotobj.pc.data

import com.lotobj.pc.engine.PrecisionTargetEngine
import com.lotobj.pc.model.ApprovedBacktestLedgerRecord
import com.lotobj.pc.model.DiscoveredFormulaRecord
import com.lotobj.pc.model.LotoDraw
import com.lotobj.pc.model.PrecisionGenome
import com.lotobj.pc.model.PrecisionLedgerRecord
import com.lotobj.pc.model.PrecisionTargetMode
import com.lotobj.pc.util.LotoDateUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.floatOrNull
import kotlinx.serialization.json.intOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull
import kotlinx.serialization.json.put
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.text.SimpleDateFormat
import java.util.Locale
import java.util.concurrent.TimeUnit

data class FetchSummary(
    val fetchedUniqueCount: Int,
    val pagesLoaded: Int,
    val rejectedCount: Int
)

data class ArchiveImportResult(
    val draws: List<LotoDraw>,
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
        val accepted = LinkedHashMap<String, LotoDraw>()
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
            draws = list,
            totalRows = total,
            acceptedCount = list.size,
            rejectedCount = total - list.size,
            errors = errors.take(10)
        )
    }

    private fun parseJsonObject(obj: JsonObject): LotoDraw? {
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
        val uniqueId = if (id != null) "lnb_$id" else obj["uniqueId"]?.jsonPrimitive?.contentOrNull?.takeIf { it.isNotBlank() }
            ?: "arch_${rawTs}_${type}_${timePart}"

        return LotoDraw(
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

        val accepted = LinkedHashMap<String, LotoDraw>()
        val errors = mutableListOf<String>()
        var total = 0

        for ((idx, line) in lines.withIndex()) {
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
            draws = list,
            totalRows = total,
            acceptedCount = list.size,
            rejectedCount = total - list.size,
            errors = errors.take(10)
        )
    }

    private fun parseCsvColumns(cols: List<String>): LotoDraw? {
        if (cols.size < 6) return null
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
        return LotoDraw(
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
        return raw.split(Regex("[,\\s\\-;]+")).mapNotNull { it.trim().toIntOrNull() }
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
            }
        }
        return null
    }
}

class LotoDataRepository(
    storageDir: File = defaultStorageDirectory(),
    private val okHttpClient: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(20, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()
) {
    private val rootDir = storageDir.apply { mkdirs() }
    private val drawsFile = File(rootDir, "draws_cache.json")
    private val formulasFile = File(rootDir, "discovered_formulas.json")
    private val approvedBacktestsFile = File(rootDir, "approved_backtests.json")
    private val precisionLedgersFile = File(rootDir, "precision_ledgers.json")
    private val precisionFormulasFile = File(rootDir, "precision_formulas.json")
    private val compactJson = Json

    @Synchronized
    fun loadSavedDraws(): List<LotoDraw> {
        if (!drawsFile.exists()) return emptyList()
        return runCatching {
            val arr = Json.parseToJsonElement(drawsFile.readText(Charsets.UTF_8)).jsonArray
            arr.mapNotNull { el ->
                val obj = el as? JsonObject ?: return@mapNotNull null
                LotoDraw(
                    uniqueId = obj["uniqueId"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null,
                    date = obj["date"]?.jsonPrimitive?.contentOrNull.orEmpty(),
                    time = obj["time"]?.jsonPrimitive?.contentOrNull.orEmpty(),
                    game = obj["game"]?.jsonPrimitive?.contentOrNull.orEmpty(),
                    num1 = obj["num1"]?.jsonPrimitive?.intOrNull ?: return@mapNotNull null,
                    num2 = obj["num2"]?.jsonPrimitive?.intOrNull ?: return@mapNotNull null,
                    num3 = obj["num3"]?.jsonPrimitive?.intOrNull ?: return@mapNotNull null,
                    num4 = obj["num4"]?.jsonPrimitive?.intOrNull ?: return@mapNotNull null,
                    num5 = obj["num5"]?.jsonPrimitive?.intOrNull ?: return@mapNotNull null,
                    timestamp = obj["timestamp"]?.jsonPrimitive?.longOrNull ?: return@mapNotNull null
                )
            }.sortedByDescending { it.timestamp }
        }.getOrDefault(emptyList())
    }

    @Synchronized
    fun mergeAndSaveDraws(incoming: List<LotoDraw>): List<LotoDraw> {
        val merged = LinkedHashMap<String, LotoDraw>()
        for (d in loadSavedDraws()) merged[d.uniqueId] = d
        for (d in incoming) merged[d.uniqueId] = d
        val sorted = merged.values.sortedByDescending { it.timestamp }

        val arr = buildJsonArray {
            for (d in sorted) {
                add(buildJsonObject {
                    put("uniqueId", d.uniqueId)
                    put("date", d.date)
                    put("time", d.time)
                    put("game", d.game)
                    put("num1", d.num1)
                    put("num2", d.num2)
                    put("num3", d.num3)
                    put("num4", d.num4)
                    put("num5", d.num5)
                    put("timestamp", d.timestamp)
                })
            }
        }
        drawsFile.writeText(compactJson.encodeToString(JsonArray.serializer(), arr), Charsets.UTF_8)
        return sorted
    }

    @Synchronized
    fun loadFormulas(periodName: String): List<DiscoveredFormulaRecord> {
        if (!formulasFile.exists()) return emptyList()
        return runCatching {
            val arr = Json.parseToJsonElement(formulasFile.readText(Charsets.UTF_8)).jsonArray
            arr.mapNotNull { el ->
                val obj = el as? JsonObject ?: return@mapNotNull null
                val rec = DiscoveredFormulaRecord(
                    id = obj["id"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null,
                    periodName = obj["periodName"]?.jsonPrimitive?.contentOrNull ?: periodName,
                    generation = obj["generation"]?.jsonPrimitive?.intOrNull ?: 1,
                    parentIdsCsv = obj["parentIdsCsv"]?.jsonPrimitive?.contentOrNull.orEmpty(),
                    primaryExprText = obj["primaryExprText"]?.jsonPrimitive?.contentOrNull.orEmpty(),
                    secondaryExprText = obj["secondaryExprText"]?.jsonPrimitive?.contentOrNull.orEmpty(),
                    slotSplitLabel = obj["slotSplitLabel"]?.jsonPrimitive?.contentOrNull.orEmpty(),
                    genomeJson = obj["genomeJson"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null,
                    usedFeaturesCsv = obj["usedFeaturesCsv"]?.jsonPrimitive?.contentOrNull.orEmpty(),
                    status = obj["status"]?.jsonPrimitive?.contentOrNull ?: "DISCOVERED",
                    rejectionStage = obj["rejectionStage"]?.jsonPrimitive?.contentOrNull.orEmpty(),
                    rejectionReason = obj["rejectionReason"]?.jsonPrimitive?.contentOrNull.orEmpty(),
                    discoveryStreak = obj["discoveryStreak"]?.jsonPrimitive?.intOrNull ?: 0,
                    rollingStreak = obj["rollingStreak"]?.jsonPrimitive?.intOrNull ?: 0,
                    confirmationStreak = obj["confirmationStreak"]?.jsonPrimitive?.intOrNull ?: 0,
                    sealedStreak = obj["sealedStreak"]?.jsonPrimitive?.intOrNull ?: 0,
                    combinedStreak = obj["combinedStreak"]?.jsonPrimitive?.intOrNull ?: 0,
                    discoveryAvgMatches = obj["discoveryAvgMatches"]?.jsonPrimitive?.floatOrNull ?: 0f,
                    rollingAvgMatches = obj["rollingAvgMatches"]?.jsonPrimitive?.floatOrNull ?: 0f,
                    confirmationAvgMatches = obj["confirmationAvgMatches"]?.jsonPrimitive?.floatOrNull ?: 0f,
                    sealedAvgMatches = obj["sealedAvgMatches"]?.jsonPrimitive?.floatOrNull ?: 0f,
                    pValueAdjusted = obj["pValueAdjusted"]?.jsonPrimitive?.doubleOrNull ?: 1.0,
                    fitnessScore = obj["fitnessScore"]?.jsonPrimitive?.doubleOrNull ?: 0.0,
                    updatedAt = obj["updatedAt"]?.jsonPrimitive?.longOrNull ?: 0L
                )
                rec
            }.filter { it.periodName == periodName }
                .sortedWith(
                    compareByDescending<DiscoveredFormulaRecord> { it.combinedStreak }
                        .thenByDescending { it.fitnessScore }
                        .thenByDescending { it.updatedAt }
                )
        }.getOrDefault(emptyList())
    }

    @Synchronized
    fun upsertFormulas(incoming: List<DiscoveredFormulaRecord>) {
        if (incoming.isEmpty()) return
        val existing = if (formulasFile.exists()) {
            runCatching {
                val arr = Json.parseToJsonElement(formulasFile.readText(Charsets.UTF_8)).jsonArray
                arr.mapNotNull { el ->
                    val obj = el as? JsonObject ?: return@mapNotNull null
                    val id = obj["id"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null
                    id to obj
                }.toMap(LinkedHashMap())
            }.getOrDefault(LinkedHashMap())
        } else {
            LinkedHashMap()
        }

        for (rec in incoming) {
            existing[rec.id] = buildJsonObject {
                put("id", rec.id)
                put("periodName", rec.periodName)
                put("generation", rec.generation)
                put("parentIdsCsv", rec.parentIdsCsv)
                put("primaryExprText", rec.primaryExprText)
                put("secondaryExprText", rec.secondaryExprText)
                put("slotSplitLabel", rec.slotSplitLabel)
                put("genomeJson", rec.genomeJson)
                put("usedFeaturesCsv", rec.usedFeaturesCsv)
                put("status", rec.status)
                put("rejectionStage", rec.rejectionStage)
                put("rejectionReason", rec.rejectionReason)
                put("discoveryStreak", rec.discoveryStreak)
                put("rollingStreak", rec.rollingStreak)
                put("confirmationStreak", rec.confirmationStreak)
                put("sealedStreak", rec.sealedStreak)
                put("combinedStreak", rec.combinedStreak)
                put("discoveryAvgMatches", rec.discoveryAvgMatches)
                put("rollingAvgMatches", rec.rollingAvgMatches)
                put("confirmationAvgMatches", rec.confirmationAvgMatches)
                put("sealedAvgMatches", rec.sealedAvgMatches)
                put("pValueAdjusted", rec.pValueAdjusted)
                put("fitnessScore", rec.fitnessScore)
                put("updatedAt", rec.updatedAt)
            }
        }

        val kept = existing.values.toList().takeLast(250)
        val arr = JsonArray(kept)
        formulasFile.writeText(compactJson.encodeToString(JsonArray.serializer(), arr), Charsets.UTF_8)
    }

    @Synchronized
    fun loadApprovedBacktestLedger(ledgerKey: String): ApprovedBacktestLedgerRecord? {
        if (!approvedBacktestsFile.exists()) return null
        return runCatching {
            val arr = Json.parseToJsonElement(approvedBacktestsFile.readText(Charsets.UTF_8)).jsonArray
            for (el in arr) {
                val obj = el as? JsonObject ?: continue
                if (obj["ledgerKey"]?.jsonPrimitive?.contentOrNull == ledgerKey) {
                    return@runCatching ApprovedBacktestLedgerRecord(
                        ledgerKey = ledgerKey,
                        strategySignature = obj["strategySignature"]?.jsonPrimitive?.contentOrNull.orEmpty(),
                        periodKey = obj["periodKey"]?.jsonPrimitive?.contentOrNull.orEmpty(),
                        frozenStepsJson = obj["frozenStepsJson"]?.jsonPrimitive?.contentOrNull ?: "[]",
                        pendingPredictionsJson = obj["pendingPredictionsJson"]?.jsonPrimitive?.contentOrNull ?: "[]",
                        lastApprovedTimestamp = obj["lastApprovedTimestamp"]?.jsonPrimitive?.longOrNull ?: 0L,
                        updatedAt = obj["updatedAt"]?.jsonPrimitive?.longOrNull ?: 0L
                    )
                }
            }
            null
        }.getOrNull()
    }

    @Synchronized
    fun saveApprovedBacktestLedger(record: ApprovedBacktestLedgerRecord) {
        val existing = if (approvedBacktestsFile.exists()) {
            runCatching {
                val arr = Json.parseToJsonElement(approvedBacktestsFile.readText(Charsets.UTF_8)).jsonArray
                arr.mapNotNull { el ->
                    val obj = el as? JsonObject ?: return@mapNotNull null
                    val key = obj["ledgerKey"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null
                    key to obj
                }.toMap(LinkedHashMap())
            }.getOrDefault(LinkedHashMap())
        } else {
            LinkedHashMap()
        }

        existing[record.ledgerKey] = buildJsonObject {
            put("ledgerKey", record.ledgerKey)
            put("strategySignature", record.strategySignature)
            put("periodKey", record.periodKey)
            put("frozenStepsJson", record.frozenStepsJson)
            put("pendingPredictionsJson", record.pendingPredictionsJson)
            put("lastApprovedTimestamp", record.lastApprovedTimestamp)
            put("updatedAt", record.updatedAt)
        }

        val kept = existing.values.toList().takeLast(60)
        approvedBacktestsFile.writeText(
            compactJson.encodeToString(JsonArray.serializer(), JsonArray(kept)),
            Charsets.UTF_8
        )
    }

    @Synchronized
    fun loadPrecisionBestGenome(mode: PrecisionTargetMode): PrecisionGenome? {
        if (!precisionFormulasFile.exists()) return null
        return runCatching {
            val arr = Json.parseToJsonElement(precisionFormulasFile.readText(Charsets.UTF_8)).jsonArray
            for (el in arr) {
                val obj = el as? JsonObject ?: continue
                if (obj["modeName"]?.jsonPrimitive?.contentOrNull == mode.name) {
                    val gJson = obj["genomeJson"]?.jsonPrimitive?.contentOrNull ?: continue
                    return@runCatching PrecisionTargetEngine.genomeFromJson(gJson)
                }
            }
            null
        }.getOrNull()
    }

    @Synchronized
    fun savePrecisionBestGenome(mode: PrecisionTargetMode, genome: PrecisionGenome) {
        val existing = if (precisionFormulasFile.exists()) {
            runCatching {
                Json.parseToJsonElement(precisionFormulasFile.readText(Charsets.UTF_8)).jsonArray.mapNotNull { el ->
                    val obj = el as? JsonObject ?: return@mapNotNull null
                    val m = obj["modeName"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null
                    m to obj
                }.toMap(LinkedHashMap())
            }.getOrDefault(LinkedHashMap())
        } else {
            LinkedHashMap()
        }

        existing[mode.name] = buildJsonObject {
            put("modeName", mode.name)
            put("signature", genome.signatureKey())
            put("genomeJson", PrecisionTargetEngine.genomeToJson(genome))
            put("updatedAt", System.currentTimeMillis())
        }
        precisionFormulasFile.writeText(
            compactJson.encodeToString(JsonArray.serializer(), JsonArray(existing.values.toList())),
            Charsets.UTF_8
        )
    }

    @Synchronized
    fun loadPrecisionLedger(ledgerKey: String): PrecisionLedgerRecord? {
        if (!precisionLedgersFile.exists()) return null
        return runCatching {
            val arr = Json.parseToJsonElement(precisionLedgersFile.readText(Charsets.UTF_8)).jsonArray
            for (el in arr) {
                val obj = el as? JsonObject ?: continue
                if (obj["ledgerKey"]?.jsonPrimitive?.contentOrNull == ledgerKey) {
                    return@runCatching PrecisionLedgerRecord(
                        ledgerKey = ledgerKey,
                        modeName = obj["modeName"]?.jsonPrimitive?.contentOrNull.orEmpty(),
                        strategySignature = obj["strategySignature"]?.jsonPrimitive?.contentOrNull.orEmpty(),
                        periodKey = obj["periodKey"]?.jsonPrimitive?.contentOrNull.orEmpty(),
                        genomeJson = obj["genomeJson"]?.jsonPrimitive?.contentOrNull ?: "{}",
                        frozenStepsJson = obj["frozenStepsJson"]?.jsonPrimitive?.contentOrNull ?: "[]",
                        pendingPredictionsJson = obj["pendingPredictionsJson"]?.jsonPrimitive?.contentOrNull ?: "[]",
                        lastApprovedTimestamp = obj["lastApprovedTimestamp"]?.jsonPrimitive?.longOrNull ?: 0L,
                        updatedAt = obj["updatedAt"]?.jsonPrimitive?.longOrNull ?: 0L
                    )
                }
            }
            null
        }.getOrNull()
    }

    @Synchronized
    fun savePrecisionLedger(record: PrecisionLedgerRecord) {
        val existing = if (precisionLedgersFile.exists()) {
            runCatching {
                Json.parseToJsonElement(precisionLedgersFile.readText(Charsets.UTF_8)).jsonArray.mapNotNull { el ->
                    val obj = el as? JsonObject ?: return@mapNotNull null
                    val k = obj["ledgerKey"]?.jsonPrimitive?.contentOrNull ?: return@mapNotNull null
                    k to obj
                }.toMap(LinkedHashMap())
            }.getOrDefault(LinkedHashMap())
        } else {
            LinkedHashMap()
        }

        existing[record.ledgerKey] = buildJsonObject {
            put("ledgerKey", record.ledgerKey)
            put("modeName", record.modeName)
            put("strategySignature", record.strategySignature)
            put("periodKey", record.periodKey)
            put("genomeJson", record.genomeJson)
            put("frozenStepsJson", record.frozenStepsJson)
            put("pendingPredictionsJson", record.pendingPredictionsJson)
            put("lastApprovedTimestamp", record.lastApprovedTimestamp)
            put("updatedAt", record.updatedAt)
        }
        val kept = existing.values.toList().takeLast(60)
        precisionLedgersFile.writeText(
            compactJson.encodeToString(JsonArray.serializer(), JsonArray(kept)),
            Charsets.UTF_8
        )
    }

    suspend fun fetchLatestResults(maxPages: Int = DEFAULT_PAGES_TO_FETCH): Result<Pair<FetchSummary, List<LotoDraw>>> {
        return withContext(Dispatchers.IO) {
            runCatching {
                val unique = LinkedHashMap<String, LotoDraw>()
                var totalRejected = 0
                var pagesLoaded = 0

                for (page in 0 until maxPages) {
                    val pageResult = fetchFromApiV3(pageNumber = page, pageSize = PAGE_SIZE)
                    pagesLoaded++
                    totalRejected += pageResult.second

                    var addedOnPage = 0
                    for (draw in pageResult.first) {
                        if (!unique.containsKey(draw.uniqueId)) {
                            unique[draw.uniqueId] = draw
                            addedOnPage++
                        }
                    }
                    if (pageResult.first.isEmpty() || addedOnPage == 0) break
                }

                val entities = unique.values.toList()
                val allSaved = if (entities.isNotEmpty()) mergeAndSaveDraws(entities) else loadSavedDraws()
                FetchSummary(entities.size, pagesLoaded, totalRejected) to allSaved
            }
        }
    }

    suspend fun importArchiveFile(file: File): Result<Pair<ArchiveImportResult, List<LotoDraw>>> {
        return withContext(Dispatchers.IO) {
            runCatching {
                val content = file.readText(Charsets.UTF_8)
                val parsed = ArchiveImporter.parseAndValidate(content, file.name)
                val allSaved = if (parsed.draws.isNotEmpty()) mergeAndSaveDraws(parsed.draws) else loadSavedDraws()
                parsed to allSaved
            }
        }
    }

    private fun fetchFromApiV3(pageNumber: Int, pageSize: Int): Pair<List<LotoDraw>, Int> {
        val url = "$BASE_URL?format=json&language=fr&pageNumber=$pageNumber&pageSize=$pageSize"
        val request = Request.Builder().url(url).header("Accept", "application/json").build()
        okHttpClient.newCall(request).execute().use { response ->
            if (!response.isSuccessful) throw IllegalStateException("HTTP ${response.code}: ${response.message}")
            val bodyStr = response.body?.string().orEmpty()
            val trimmed = bodyStr.trimStart()
            if (!trimmed.startsWith("{")) throw IllegalStateException("Réponse non-JSON reçue du serveur LNB")
            val root = Json.parseToJsonElement(trimmed).jsonObject
            val status = root["status"]?.jsonPrimitive?.contentOrNull
            if (status != null && status != "0") {
                throw IllegalStateException(root["message"]?.jsonPrimitive?.contentOrNull ?: "Erreur API ($status)")
            }
            val drawsArray = root["drawResults"]?.jsonArray
                ?: throw IllegalStateException("Le JSON ne contient pas 'drawResults'")
            val list = ArrayList<LotoDraw>()
            var rejected = 0
            for (element in drawsArray) {
                val drawObj = element as? JsonObject
                if (drawObj == null) {
                    rejected++
                    continue
                }
                val parsed = parseV3Draw(drawObj)
                if (parsed != null) list.add(parsed) else rejected++
            }
            return list to rejected
        }
    }

    internal fun parseV3Draw(drawObj: JsonObject): LotoDraw? {
        val name = drawObj["name"]?.jsonPrimitive?.contentOrNull?.trim()
        if (name.isNullOrEmpty() || name == "-") return null
        val timestamp = drawObj["closureDate"]?.jsonPrimitive?.longOrNull
        if (timestamp == null || timestamp <= 0L) return null
        val rawNumbers = drawObj["drawResults"]?.jsonPrimitive?.contentOrNull?.trim() ?: return null
        val nums = rawNumbers.split(Regex("[,\\s-]+")).mapNotNull { it.trim().toIntOrNull() }
        if (nums.size != 5 || nums.distinct().size != 5 || nums.any { it !in 1..90 }) return null

        val id = drawObj["id"]?.jsonPrimitive?.contentOrNull?.takeIf { it.isNotBlank() }
        val uniqueId = if (id != null) "lnb_$id" else "${timestamp}_$name"
        val type = drawObj["type"]?.jsonPrimitive?.contentOrNull?.trim()
        val game = if (!type.isNullOrEmpty()) type else name.substringBefore(" ").trim()
        val timePart = if (name.startsWith(game, ignoreCase = true)) name.substring(game.length).trim() else name

        return LotoDraw(
            uniqueId = uniqueId,
            date = LotoDateUtils.formatFrenchDayAndDate(timestamp),
            time = timePart,
            game = game,
            num1 = nums[0],
            num2 = nums[1],
            num3 = nums[2],
            num4 = nums[3],
            num5 = nums[4],
            timestamp = timestamp
        )
    }

    fun getStorageFolderPath(): String = rootDir.absolutePath

    companion object {
        private const val BASE_URL = "https://api-gateway2.lnbloto.bj/lotto-api/v3/590/drawResults"
        private const val PAGE_SIZE = 186
        private const val DEFAULT_PAGES_TO_FETCH = 8

        fun defaultStorageDirectory(): File {
            val home = System.getProperty("user.home").orEmpty().ifBlank { "." }
            return File(home, ".LotoBJ_PC")
        }
    }
}
