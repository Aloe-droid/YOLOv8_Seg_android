package com.example.yolov8n_seg

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.view.ViewGroup
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import org.opencv.android.CameraBridgeViewBase
import org.opencv.android.JavaCameraView
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc

class MainActivity : ComponentActivity(), CameraBridgeViewBase.CvCameraViewListener2, Inference,
    Draw {

    companion object {
        // 전면 = 1, 후면 = 0
        const val CAMERA_ID = 0
    }

    private lateinit var net: Net
    private lateinit var labels: Array<String>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setPermissions()

        val openCVCameraView = (JavaCameraView(this, CAMERA_ID) as CameraBridgeViewBase).apply {
            setCameraPermissionGranted()
            enableView()
            setCvCameraViewListener(this@MainActivity)
            setMaxFrameSize(640, 640)
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT
            )

        }

        net = loadModel(assets, filesDir.toString())
        labels = loadLabel(assets)

        setContent {
            AndroidView(modifier = Modifier.fillMaxSize(), factory = { openCVCameraView })
        }
    }

    private fun setPermissions() {
        val requestPermissionLauncher =
            registerForActivityResult(ActivityResultContracts.RequestPermission()) {
                if (!it) {
                    Toast.makeText(this, "권한을 허용 하지 않으면 사용할 수 없습니다!", Toast.LENGTH_SHORT).show()
                    finish()
                }
            }

        val permissions = listOf(Manifest.permission.CAMERA)

        permissions.forEach {
            if (ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED) {
                requestPermissionLauncher.launch(it)
            }
        }
        OpenCVLoader.initDebug()
    }

    override fun onCameraViewStarted(width: Int, height: Int) {
    }

    override fun onCameraViewStopped() {
    }

    override fun onCameraFrame(inputFrame: CameraBridgeViewBase.CvCameraViewFrame?): Mat {
        val frameMat = inputFrame!!.rgba()

        val results = detect(frameMat, net, labels)
        Imgproc.cvtColor(frameMat, frameMat, Imgproc.COLOR_RGBA2RGB)
        val outMat = drawSeg(frameMat, results, labels)

        frameMat.release()
        return outMat
    }
}

