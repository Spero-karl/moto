package com.example

import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onRoot
import com.example.data.local.LotoResultEntity
import com.example.ui.screens.FeaturedDrawCard
import com.example.ui.theme.MyApplicationTheme
import com.github.takahirom.roborazzi.RobolectricDeviceQualifiers
import com.github.takahirom.roborazzi.captureRoboImage
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.annotation.Config
import org.robolectric.annotation.GraphicsMode

@RunWith(RobolectricTestRunner::class)
@GraphicsMode(GraphicsMode.Mode.NATIVE)
@Config(qualifiers = RobolectricDeviceQualifiers.Pixel8, sdk = [36])
class GreetingScreenshotTest {

  @get:Rule val composeTestRule = createComposeRule()

  @Test
  fun greeting_screenshot() {
    val sample = LotoResultEntity(
      uniqueId = "sample_1",
      date = "15/11/2026",
      time = "18H",
      game = "Fortune",
      num1 = 12,
      num2 = 24,
      num3 = 45,
      num4 = 67,
      num5 = 88,
      timestamp = 1794747600000L
    )
    composeTestRule.setContent { MyApplicationTheme { FeaturedDrawCard(sample) } }

    composeTestRule.onRoot().captureRoboImage(filePath = "src/test/screenshots/greeting.png")
  }
}
