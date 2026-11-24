package com.example.data.repository

import android.util.Log
import com.example.data.local.LotoDao
import com.example.data.local.LotoResultEntity
import com.example.domain.util.LotoDateUtils
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull
import okhttp3.OkHttpClient
import okhttp3.Request

data class FetchSummary(
    val fetchedUniqueCount: Int,
    val pagesLoaded: Int,
    val rejectedCount: Int
)

class LotoRepository(
    private val lotoDao: LotoDao,
    private val okHttpClient: OkHttpClient = OkHttpClient()
) {

    val allResults = lotoDao.getAllResults()

    suspend fun fetchLatestResults(maxPages: Int = DEFAULT_PAGES_TO_FETCH): Result<FetchSummary> {
        return withContext(Dispatchers.IO) {
            try {
                val uniqueEntities = LinkedHashMap<String, LotoResultEntity>()
                var totalRejected = 0
                var pagesLoaded = 0

                for (page in 0 until maxPages) {
                    val pageResult = fetchFromApiV3(pageNumber = page, pageSize = PAGE_SIZE)
                    pagesLoaded++
                    totalRejected += pageResult.rejectedCount

                    var addedOnPage = 0
                    for (entity in pageResult.entities) {
                        if (!uniqueEntities.containsKey(entity.uniqueId)) {
                            uniqueEntities[entity.uniqueId] = entity
                            addedOnPage++
                        }
                    }

                    // Arret anticipe si la page est vide ou ne contient aucun nouveau tirage
                    if (pageResult.entities.isEmpty() || addedOnPage == 0) {
                        break
                    }
                }

                val entities = uniqueEntities.values.toList()
                if (entities.isNotEmpty()) {
                    lotoDao.insertAll(entities)
                }
                Result.success(
                    FetchSummary(
                        fetchedUniqueCount = entities.size,
                        pagesLoaded = pagesLoaded,
                        rejectedCount = totalRejected
                    )
                )
            } catch (e: Exception) {
                Log.e(TAG, "Error fetching data: ${e.message}", e)
                Result.failure(e)
            }
        }
    }

    suspend fun importArchive(rawContent: String, fileName: String): Result<ArchiveImportResult> {
        return withContext(Dispatchers.IO) {
            try {
                val parsed = ArchiveImporter.parseAndValidate(rawContent, fileName)
                if (parsed.entities.isNotEmpty()) {
                    lotoDao.insertAll(parsed.entities)
                }
                Result.success(parsed)
            } catch (e: Exception) {
                Log.e(TAG, "Archive import failed: ${e.message}", e)
                Result.failure(e)
            }
        }
    }

    private data class PageParseResult(
        val entities: List<LotoResultEntity>,
        val rejectedCount: Int
    )

    private fun fetchFromApiV3(pageNumber: Int, pageSize: Int): PageParseResult {
        val url = "$BASE_URL?format=json&language=fr&pageNumber=$pageNumber&pageSize=$pageSize"
        val request = Request.Builder()
            .url(url)
            .header("Accept", "application/json")
            .build()

        okHttpClient.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IllegalStateException("HTTP ${response.code}: ${response.message}")
            }

            val bodyStr = response.body?.string().orEmpty()
            if (bodyStr.isBlank()) {
                throw IllegalStateException("Réponse vide du serveur")
            }

            val trimmed = bodyStr.trimStart()
            if (!trimmed.startsWith("{")) {
                throw IllegalStateException("Réponse non-JSON reçue du serveur")
            }

            val root = Json.parseToJsonElement(trimmed).jsonObject
            val status = root["status"]?.jsonPrimitive?.contentOrNull
            if (status != null && status != "0") {
                val msg = root["message"]?.jsonPrimitive?.contentOrNull ?: "Erreur API ($status)"
                throw IllegalStateException(msg)
            }

            val drawsArray = root["drawResults"]?.jsonArray
                ?: throw IllegalStateException("Le JSON ne contient pas 'drawResults'")

            val entities = mutableListOf<LotoResultEntity>()
            var rejected = 0
            for (element in drawsArray) {
                val drawObj = element as? JsonObject
                if (drawObj == null) {
                    rejected++
                    continue
                }
                val parsed = parseV3Draw(drawObj)
                if (parsed != null) {
                    entities.add(parsed)
                } else {
                    rejected++
                }
            }
            return PageParseResult(entities, rejected)
        }
    }

    /**
     * Parse un tirage v3 avec validation stricte :
     * - Rejecte tout tirage sans closureDate valide (jamais de repli sur System.currentTimeMillis()!).
     * - Rejecte tout tirage ne contenant pas exactement 5 numeros distincts dans 1..90.
     */
    internal fun parseV3Draw(drawObj: JsonObject): LotoResultEntity? {
        val name = drawObj["name"]?.jsonPrimitive?.contentOrNull?.trim()
        if (name.isNullOrEmpty() || name == "-") return null

        val timestamp = drawObj["closureDate"]?.jsonPrimitive?.longOrNull
        if (timestamp == null || timestamp <= 0L) {
            safeWarn("Tirage rejeté ($name) : closureDate manquant ou invalide")
            return null
        }

        val rawNumbers = drawObj["drawResults"]?.jsonPrimitive?.contentOrNull?.trim()
        if (rawNumbers.isNullOrEmpty()) return null

        val nums = rawNumbers
            .split(Regex("[,\\s-]+"))
            .mapNotNull { it.trim().toIntOrNull() }

        if (nums.size != 5 || nums.distinct().size != 5 || nums.any { it !in 1..90 }) {
            safeWarn("Tirage rejeté ($name) : combinaison invalide ($rawNumbers)")
            return null
        }

        val id = drawObj["id"]?.jsonPrimitive?.contentOrNull?.takeIf { it.isNotBlank() }
        val uniqueId = if (id != null) "lnb_$id" else "${timestamp}_$name"

        val type = drawObj["type"]?.jsonPrimitive?.contentOrNull?.trim()
        val game = if (!type.isNullOrEmpty()) {
            type
        } else {
            name.substringBefore(" ").trim()
        }

        val timePart = if (name.startsWith(game, ignoreCase = true)) {
            name.substring(game.length).trim()
        } else {
            name
        }

        val formattedDate = LotoDateUtils.formatFrenchDayAndDate(timestamp)

        return LotoResultEntity(
            uniqueId = uniqueId,
            date = formattedDate,
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

    private fun safeWarn(message: String) {
        try {
            Log.w(TAG, message)
        } catch (_: Throwable) {
            // Pure JVM unit test environment
        }
    }

    companion object {
        private const val TAG = "LotoRepository"
        private const val BASE_URL = "https://api-gateway2.lnbloto.bj/lotto-api/v3/590/drawResults"
        private const val PAGE_SIZE = 186
        private const val DEFAULT_PAGES_TO_FETCH = 6
    }
}
