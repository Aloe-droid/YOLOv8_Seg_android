package com.example.yolov8n_seg

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

interface Segment {

    fun maskOutput(
        boxOutputs: MutableList<Result>,
        output1: Mat,
        matWidth: Int,
        matHeight: Int
    ) {

        if (boxOutputs.size == 0) return

        val maskPredictionList = boxOutputs.map { it.maskMat }
        val maskPredictionMat = Mat()
        Core.vconcat(maskPredictionList, maskPredictionMat)
        val reshapeSize = Inference.OUTPUT_MASK_SIZE * Inference.OUTPUT_MASK_SIZE
        val outputMat = output1.reshape(1, output1.total().toInt() / reshapeSize)
        val matMul = Mat()

        Core.gemm(maskPredictionMat, outputMat, 1.0, Mat(), 0.0, matMul)
        val masks = sigmoid(matMul)
        val resizedBoxes = resizeBoxes(boxOutputs, matWidth, matHeight)
        val blurSize = Size(
            (matWidth / Inference.OUTPUT_MASK_SIZE).toDouble(),
            (matHeight / Inference.OUTPUT_MASK_SIZE).toDouble()
        )

        for (i in 0 until resizedBoxes.size) {
            val resizeBox = resizedBoxes[i]
            val scaleX = resizeBox.x
            val scaleY = resizeBox.y
            val scaleW = resizeBox.width
            val scaleH = resizeBox.height

            val w = boxOutputs[i].box.width
            val h = boxOutputs[i].box.height

            val mask = masks.row(i).reshape(1, Inference.OUTPUT_MASK_SIZE)
            val resizedCropMask = Mat(mask, Rect(scaleX, scaleY, scaleW, scaleH))
            val cropMask = Mat()
            val blurMask = Mat()
            val thresholdMask = Mat()
            val resizeSize = Size(w.toDouble(), h.toDouble())

            Imgproc.resize(resizedCropMask, cropMask, resizeSize, 0.0, 0.0, Imgproc.INTER_LINEAR)
            Imgproc.blur(cropMask, blurMask, blurSize)
            Imgproc.threshold(blurMask, thresholdMask, 0.5, 1.0, Imgproc.THRESH_BINARY)

            thresholdMask.convertTo(thresholdMask, CvType.CV_8UC1)
            boxOutputs[i].maskMat.release()
            boxOutputs[i].maskMat = thresholdMask

            mask.release()
            resizedCropMask.release()
            cropMask.release()
            blurMask.release()
        }

        maskPredictionMat.release()
        output1.release()
        outputMat.release()
        matMul.release()
        masks.release()
        maskPredictionList.forEach { it.release() }
    }

    private fun sigmoid(mat: Mat): Mat {
        val oneMat = Mat.ones(mat.size(), mat.type())
        val mulMat = Mat()
        val expMat = Mat()
        val outMat = Mat()

        Core.multiply(mat, Scalar(-1.0), mulMat)
        Core.exp(mulMat, expMat)
        Core.add(oneMat, expMat, outMat)
        Core.divide(oneMat, outMat, outMat)

        oneMat.release()
        mulMat.release()
        expMat.release()

        return outMat
    }

    private fun resizeBoxes(
        boxOutputs: MutableList<Result>,
        width: Int,
        height: Int
    ): MutableList<Rect> {
        val resizedBoxes = mutableListOf<Rect>()
        boxOutputs.forEach {
            val rect = it.box
            val x = rect.x * Inference.OUTPUT_MASK_SIZE / width
            val w = rect.width * Inference.OUTPUT_MASK_SIZE / width
            val y = rect.y * Inference.OUTPUT_MASK_SIZE / height
            val h = rect.height * Inference.OUTPUT_MASK_SIZE / height

            resizedBoxes.add(Rect(x, y, w, h))
        }
        return resizedBoxes
    }
}