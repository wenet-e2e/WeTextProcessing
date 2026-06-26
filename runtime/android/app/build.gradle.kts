plugins {
    // AGP 9 has built-in Kotlin support (it bundles KGP and enables it by default),
    // so the org.jetbrains.kotlin.android plugin must NOT be applied here.
    alias(libs.plugins.android.application)
    // Compose compiler plugin (required since Kotlin 2.0+).
    alias(libs.plugins.compose.compiler)
}

fun nativeAbis(): List<String> =
    (project.findProperty("abiFilters") as String? ?: "armeabi-v7a,arm64-v8a,x86,x86_64")
        .split(",").map { it.trim() }.filter { it.isNotEmpty() }

android {
    namespace = "com.wenet.WeTextProcessing"
    lint {
        abortOnError = false
    }
    signingConfigs {
        create("release") {
            storeFile = file("wenet.keystore")
            storePassword = "123456"
            keyAlias = "wenet"
            keyPassword = "123456"
        }
    }
    compileSdk = 36

    defaultConfig {
        applicationId = "com.wenet.WeTextProcessing"
        minSdk = 30
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters.addAll(nativeAbis())
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("release")
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    // Built-in Kotlin defaults jvmTarget to compileOptions.targetCompatibility (17).
    buildFeatures {
        compose = true
    }
}

dependencies {
    implementation(platform(libs.androidx.compose.bom))

    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)
    debugImplementation(libs.androidx.compose.ui.tooling)
}

// Native libs are NOT built by Gradle. Build them beforehand with CMake presets
// (from repo root) so the .so files exist in app/src/main/jniLibs/<abi>/:
//   export ANDROID_NDK_HOME=$ANDROID_HOME/ndk/<version>
//   cd runtime
//   cmake --preset android-arm64-v8a
//   cmake --build --preset android-arm64-v8a
//   cmake --install build/aarch64-linux-android --component jni
// Gradle only packages whatever is already present under jniLibs.
