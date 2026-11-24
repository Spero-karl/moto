plugins {
    kotlin("jvm") version "2.2.10"
    application
}

group = "com.lotobj.pc"
version = "1.0.0"

repositories {
    mavenCentral()
    google()
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(21))
    }
}

dependencies {
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.10.2")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-swing:1.10.2")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    implementation("com.formdev:flatlaf:3.5.4")
    implementation("org.jocl:jocl:2.0.5")

    testImplementation("junit:junit:4.13.2")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.10.2")
}

application {
    mainClass.set("com.lotobj.pc.MainKt")
    applicationDefaultJvmArgs = listOf(
        "-Xms256m",
        "-XX:+UseG1GC",
        "-XX:MaxRAMPercentage=65.0",
        "-Dfile.encoding=UTF-8"
    )
}

tasks.test {
    useJUnit()
    maxHeapSize = "2g"
}

val fatJar by tasks.registering(Jar::class) {
    group = "distribution"
    description = "Construit un JAR exécutable tout-en-un pour Windows 10/11 (CPU/RAM/GPU OpenCL)"
    archiveBaseName.set("LotoBJ-PC-AllInOne")
    archiveVersion.set("")
    duplicatesStrategy = DuplicatesStrategy.EXCLUDE
    manifest {
        attributes(
            "Main-Class" to "com.lotobj.pc.MainKt",
            "Implementation-Title" to "Loto BJ Workstation PC",
            "Implementation-Version" to project.version
        )
    }
    from(sourceSets.main.get().output)
    dependsOn(configurations.runtimeClasspath)
    from({
        configurations.runtimeClasspath.get()
            .filter { it.name.endsWith("jar") }
            .map { zipTree(it) }
    }) {
        exclude("META-INF/*.SF", "META-INF/*.DSA", "META-INF/*.RSA", "META-INF/versions/**")
    }
}

val copyBundledJar by tasks.registering(Copy::class) {
    group = "distribution"
    description = "Copie LotoBJ-PC-AllInOne.jar dans windows-pc/app/ pour exécution Windows sans Gradle"
    dependsOn(fatJar)
    from(fatJar.flatMap { it.archiveFile })
    into(layout.projectDirectory.dir("app"))
}
