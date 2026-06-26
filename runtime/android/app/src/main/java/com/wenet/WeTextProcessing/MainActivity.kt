package com.wenet.WeTextProcessing

import android.content.Context
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import java.io.File
import java.io.IOException

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            assetsInit(this)
            WeTextProcessing.init(filesDir.path)
        } catch (e: IOException) {
            Log.e(LOG_TAG, "Error processing asset files to file path", e)
        }
        setContent {
            WenetTheme {
                WeTextProcessingScreen()
            }
        }
    }

    companion object {
        private const val LOG_TAG = "WETEXTPROCESSING"
        private val resource = listOf(
            "zh_tn_tagger.fst", "zh_tn_verbalizer.fst", "zh_itn_tagger.fst", "zh_itn_verbalizer.fst"
        )

        // Unzip the FST models bundled in assets into the app's files dir on first run.
        // Note: uninstalling the app removes these files.
        @Throws(IOException::class)
        fun assetsInit(context: Context) {
            val assetMgr = context.assets
            assetMgr.list("")?.forEach { file ->
                if (file in resource) {
                    val dst = File(context.filesDir, file)
                    if (!dst.exists() || dst.length() == 0L) {
                        Log.i(LOG_TAG, "Unzipping $file to ${dst.absolutePath}")
                        assetMgr.open(file).use { input ->
                            dst.outputStream().use { output -> input.copyTo(output) }
                        }
                    }
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun WeTextProcessingScreen(modifier: Modifier = Modifier) {
    var input by remember { mutableStateOf("") }
    var result by remember { mutableStateOf("") }

    Scaffold(
        modifier = modifier.fillMaxSize(),
        topBar = { TopAppBar(title = { Text("WeTextProcessing") }) }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            OutlinedTextField(
                value = input,
                onValueChange = { input = it },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Input text") },
            )

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(16.dp)
            ) {
                Button(
                    onClick = { result = WeTextProcessing.normalize(input) },
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Normalize")
                }
                Button(
                    onClick = { result = WeTextProcessing.inverse_normalize(input) },
                    modifier = Modifier.weight(1f)
                ) {
                    Text("Inverse normalize")
                }
            }

            Card(modifier = Modifier.fillMaxWidth()) {
                Text(
                    text = result,
                    modifier = Modifier
                        .fillMaxWidth()
                        .verticalScroll(rememberScrollState())
                        .padding(16.dp),
                    style = MaterialTheme.typography.bodyLarge
                )
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun WeTextProcessingScreenPreview() {
    WenetTheme {
        WeTextProcessingScreen()
    }
}
