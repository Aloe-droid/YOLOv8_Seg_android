package com.example.yolov8n_seg

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import kotlin.math.round

interface Draw {
    companion object {
        const val ALPHA = 0.5
    }

    fun drawSeg(mat: Mat, lists: MutableList<Result>, labels: Array<String>): Mat {

        val maskImg = mat.clone()

        if (lists.size == 0) return maskImg

        lists.forEach {
            val box = it.box
            val color = getColor(it.index)
            val textPoint = Point(box.x.toDouble(), box.y.toDouble() - 5)
            val text = "${labels[it.index]} ${round(it.confidence * 100).toInt()}%"

            Imgproc.rectangle(maskImg, box, color, 5)
            Imgproc.putText(maskImg, text, textPoint, Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, color, 5)

            val cropMask = it.maskMat
            val cropMaskImg = Mat(maskImg, box)
            val cropMaskRGB = Mat(cropMask.size(), CvType.CV_8UC3)
            val list = List(3) { cropMask.clone() }
            Core.merge(list, cropMaskRGB)

            val temp1 = Mat.zeros(cropMaskRGB.size(), cropMaskRGB.type())
            Core.add(temp1, Scalar(1.0, 1.0, 1.0), temp1)
            Core.subtract(temp1, cropMaskRGB, temp1)
            Core.multiply(cropMaskImg, temp1, cropMaskImg)

            val temp2 = Mat()
            Core.multiply(cropMaskRGB, color, temp2)
            Core.add(cropMaskImg, temp2, cropMaskImg)

            cropMaskImg.release()
            temp1.release()
            temp2.release()
            cropMaskRGB.release()
            list.forEach { mat -> mat.release() }
        }

        val resultMat = Mat(mat.size(), mat.type())
        Core.addWeighted(maskImg, ALPHA, mat, 1 - ALPHA, 0.0, resultMat, CvType.CV_8UC3)

        maskImg.release()
        return resultMat
    }

    private fun getColor(index: Int): Scalar {
        return when (index) {
            // WHITE
            45, 18, 19, 22, 30, 42, 43, 44, 61, 71, 72 -> Scalar(255.0, 255.0, 255.0)
            // BLUE
            1, 3, 14, 25, 37, 38, 79 -> Scalar(0.0, 0.0, 255.0)
            // RED
            2, 9, 10, 11, 32, 47, 49, 51, 52 -> Scalar(255.0, 0.0, 0.0)
            // YELLOW
            5, 23, 46, 48 -> Scalar(0.0, 255.0, 255.0)
            // GRAY
            6, 13, 34, 35, 36, 54, 59, 60, 73, 77, 78 -> Scalar(128.0, 128.0, 128.0)
            // BLACK
            7, 24, 26, 27, 28, 62, 64, 65, 66, 67, 68, 69, 74, 75 -> Scalar(0.0, 0.0, 0.0)
            // GREEN
            12, 29, 33, 39, 41, 58, 50 -> Scalar(0.0, 255.0, 0.0)
            // DARK GRAY
            15, 16, 17, 20, 21, 31, 40, 55, 57, 63 -> Scalar(64.0, 64.0, 64.0)
            // LIGHT GRAY
            70, 76 -> Scalar(192.0, 192.0, 192.0)
            // PURPLE
            else -> Scalar(128.0, 0.0, 128.0)
        }
    }
}