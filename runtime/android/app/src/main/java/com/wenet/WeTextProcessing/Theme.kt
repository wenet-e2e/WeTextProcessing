package com.wenet.WeTextProcessing

import android.os.Build
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext

// Brand colors taken from the logo gradient: indigo (#4C5BF5) -> cyan (#2DD4D4).
private val BrandIndigo = Color(0xFF4C5BF5)
private val BrandCyan = Color(0xFF2DD4D4)

private val LightColors = lightColorScheme(
    primary = BrandIndigo,
    secondary = BrandCyan,
)

private val DarkColors = darkColorScheme(
    // Lightened indigo for sufficient contrast on dark surfaces.
    primary = Color(0xFF9AA5FF),
    secondary = BrandCyan,
)

@Composable
fun WenetTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    // Material You dynamic color (Android 12+) takes priority; the brand colors
    // that match the logo are the fallback on devices without dynamic color.
    dynamicColor: Boolean = true,
    content: @Composable () -> Unit,
) {
    val colorScheme = when {
        dynamicColor && Build.VERSION.SDK_INT >= Build.VERSION_CODES.S -> {
            val context = LocalContext.current
            if (darkTheme) dynamicDarkColorScheme(context) else dynamicLightColorScheme(context)
        }
        darkTheme -> DarkColors
        else -> LightColors
    }
    MaterialTheme(colorScheme = colorScheme, content = content)
}
