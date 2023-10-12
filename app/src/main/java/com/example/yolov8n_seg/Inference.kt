package com.example.yolov8n_seg

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfRect2d
import org.opencv.core.Rect
import org.opencv.core.Rect2d
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc

interface Inference : Load, Segment {

    companion object {

        const val OUTPUT_NAME_0 = "output0"
        const val OUTPUT_NAME_1 = "output1"

        const val INPUT_SIZE = 640
        const val SCALE_FACTOR = 1 / 255.0

        const val OUTPUT_SIZE = 8400
        const val OUTPUT_MASK_SIZE = 160

        const val CONFIDENCE_THRESHOLD = 0.5f
        const val NMS_THRESHOLD = 0.5f
    }

    fun detect(mat: Mat, net: Net, labels: Array<String>): MutableList<Result> {

        val inputMat = mat.clone()
        Imgproc.resize(inputMat, inputMat, Size(INPUT_SIZE.toDouble(), INPUT_SIZE.toDouble()))
        inputMat.convertTo(inputMat, CvType.CV_32FC3)
        val blob = Dnn.blobFromImage(inputMat, SCALE_FACTOR)
        net.setInput(blob)

        val output0 = Mat()
        val output1 = Mat()
        val outputList = arrayListOf(output0, output1)
        val outputNameList = arrayListOf(OUTPUT_NAME_0, OUTPUT_NAME_1)

        net.forward(outputList, outputNameList)
        val lists =
            postProcess(outputList[0], outputList[1], labels.size, mat.width(), mat.height())

        blob.release()
        inputMat.release()
        output0.release()
        output1.release()

        return lists
    }

    private fun postProcess(
        output0: Mat, output1: Mat, labelSize: Int, width: Int, height: Int
    ): MutableList<Result> {
        val lists = boxOutput(output0, labelSize)
        resizeBox(lists, width, height)
        maskOutput(lists, output1, width, height)

        output0.release()
        output1.release()
        return lists
    }

    private fun boxOutput(output: Mat, labelSize: Int): MutableList<Result> {
        val detections = output.reshape(1, output.total().toInt() / OUTPUT_SIZE).t()

        val boxes = Array(detections.rows()) { Rect2d() }
        val maxScores = Array(detections.rows()) { 0f }
        val indexes = Array(detections.rows()) { 0 }

        for (i in 0 until detections.rows()) {
            val scores = detections.row(i).colRange(4, labelSize)
            val max = Core.minMaxLoc(scores)
            val xPos = detections.get(i, 0)[0]
            val yPos = detections.get(i, 1)[0]
            val width = detections.get(i, 2)[0]
            val height = detections.get(i, 3)[0]
            val left = 0.0.coerceAtLeast(xPos - width / 2.0)
            val top = 0.0.coerceAtLeast(yPos - height / 2.0)

            boxes[i] = Rect2d(left, top, width, height)
            maxScores[i] = max.maxVal.toFloat()
            indexes[i] = max.maxLoc.x.toInt()

            scores.release()
        }

        val rects = MatOfRect2d(*boxes)
        val floats = MatOfFloat(*maxScores.toFloatArray())
        val ints = MatOfInt(*indexes.toIntArray())
        val indices = MatOfInt()
        Dnn.NMSBoxesBatched(rects, floats, ints, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices)

        val list = mutableListOf<Result>()

        if (indices.total().toInt() == 0) return list

        indices.toList().forEach {
            val scores = detections.row(it).colRange(4, labelSize)
            val max = Core.minMaxLoc(scores)

            val xPos = detections.get(it, 0)[0]
            val yPos = detections.get(it, 1)[0]
            val width = detections.get(it, 2)[0]
            val height = detections.get(it, 3)[0]

            val x = 0.0.coerceAtLeast(xPos - width / 2.0).toInt()
            val y = 0.0.coerceAtLeast(yPos - height / 2.0).toInt()
            val w = INPUT_SIZE.toDouble().coerceAtMost(width).toInt()
            val h = INPUT_SIZE.toDouble().coerceAtMost(height).toInt()
            val rect = Rect(x, y, w, h)

            val score = max.maxVal.toFloat()
            val index = max.maxLoc.x.toInt()
            val mask = detections.row(it).colRange(4 + labelSize, detections.cols())
            val result = Result(rect, score, index, mask)
            list.add(result)
        }
        detections.release()
        return list
    }

    private fun resizeBox(list: MutableList<Result>, width: Int, height: Int) {
        list.forEach {
            val box = it.box
            val x = (box.x * width / INPUT_SIZE)
            val y = (box.y * height / INPUT_SIZE)
            var w = (box.width * width / INPUT_SIZE)
            var h = (box.height * height / INPUT_SIZE)

            if(w > width) w = width
            if(h > height) h = height

            if(x + w > width) w = width - x
            if(y + h > height) h = height - y

            val rect = Rect(x, y, w, h)
            it.box = rect
        }
    }
}