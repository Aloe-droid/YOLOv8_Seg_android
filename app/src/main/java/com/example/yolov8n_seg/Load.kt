package com.example.yolov8n_seg

import android.content.res.AssetManager
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.InputStreamReader

interface Load {

    companion object {
        const val FILE_NAME = "yolov8n-seg.onnx"
        const val LABEL_NAME = "yolov8n.txt"
    }

    fun loadModel(assets: AssetManager, fileDir: String): Net {
        val outputFile = File("$fileDir/$FILE_NAME")
        assets.open(FILE_NAME).use { inputStream ->
            FileOutputStream(outputFile).use { outputStream ->
                val buffer = ByteArray(1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
            }
        }
        return Dnn.readNetFromONNX("$fileDir/$FILE_NAME")
    }

    fun loadLabel(assets: AssetManager): Array<String> {
        BufferedReader(InputStreamReader(assets.open(LABEL_NAME))).use { reader ->
            var line: String?
            val classList = mutableListOf<String>()
            while (reader.readLine().also { line = it } != null) {
                line?.let { l -> classList.add(l) }
            }
            return classList.toTypedArray()
        }
    }
}