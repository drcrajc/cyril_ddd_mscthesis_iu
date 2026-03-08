package com.example.dddapp

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.media.MediaPlayer
import android.os.Bundle
import android.util.Log
import android.widget.Button
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {

    private lateinit var previewView: PreviewView
    private lateinit var overlayView: OverlayView
    private lateinit var statusText: TextView
    private lateinit var pauseBtn: Button
    private lateinit var exitBtn: Button
    private lateinit var tflite: Interpreter
    private lateinit var alarmPlayer: MediaPlayer

    private val INPUT_SIZE = 320
    private val NUM_DETECTIONS = 6300
    private val CONF_THRESHOLD = 0.6f
    private val CLOSED_TIME_THRESHOLD = 1000L // 1 second

    private var closedStartTime: Long? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var isCameraRunning = true

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        statusText = findViewById(R.id.statusText)
        pauseBtn = findViewById(R.id.pauseBtn)
        exitBtn = findViewById(R.id.exitBtn)

        alarmPlayer = MediaPlayer.create(this, R.raw.alarm)

        loadModel()

        pauseBtn.setOnClickListener {
            if (isCameraRunning) {
                cameraProvider?.unbindAll()
                isCameraRunning = false
                pauseBtn.text = "START"
            } else {
                startCamera()
                isCameraRunning = true
                pauseBtn.text = "PAUSE"
            }
        }

        exitBtn.setOnClickListener {
            finish()
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                100
            )
        }
    }

    private fun loadModel() {
        val model = assets.open("ddd.tflite").readBytes()
        val buffer = ByteBuffer.allocateDirect(model.size)
        buffer.order(ByteOrder.nativeOrder())
        buffer.put(model)
        buffer.rewind()

        tflite = Interpreter(buffer)
        Log.d("YOLO", "Model Loaded")
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({

            cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }

            val imageAnalysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(
                ContextCompat.getMainExecutor(this)
            ) { imageProxy ->
                analyzeFrame(imageProxy)
            }

            cameraProvider?.unbindAll()
            cameraProvider?.bindToLifecycle(
                this,
                CameraSelector.DEFAULT_FRONT_CAMERA,
                preview,
                imageAnalysis
            )

        }, ContextCompat.getMainExecutor(this))
    }

    private fun analyzeFrame(imageProxy: ImageProxy) {

        val inputBuffer = imageProxyToBuffer(imageProxy)
        val output = Array(1) { Array(NUM_DETECTIONS) { FloatArray(7) } }

        tflite.run(inputBuffer, output)

        val detections = mutableListOf<Detection>()

        var frameClosed = false

        for (i in 0 until NUM_DETECTIONS) {

            val confidence = output[0][i][4]
            if (confidence < CONF_THRESHOLD) continue

            val cx = output[0][i][0]
            val cy = output[0][i][1]
            val w = output[0][i][2]
            val h = output[0][i][3]

            val openScore = output[0][i][5]
            val closedScore = output[0][i][6]

            val label = if (closedScore > openScore) "closed_eye" else "open_eye"

            if (label == "closed_eye") frameClosed = true

            val left = (cx - w / 2) * overlayView.width
            val top = (cy - h / 2) * overlayView.height
            val right = (cx + w / 2) * overlayView.width
            val bottom = (cy + h / 2) * overlayView.height

            detections.add(
                Detection(left, top, right, bottom, confidence, label)
            )
        }

        handleDrowsiness(frameClosed, detections.isEmpty())

        runOnUiThread {
            overlayView.setResults(detections)
        }

        imageProxy.close()
    }

    private fun handleDrowsiness(frameClosed: Boolean, noDetection: Boolean) {

        if (noDetection) {
            closedStartTime = null
            runOnUiThread {
                statusText.text = "NO DRIVER"
                statusText.setBackgroundColor(Color.GRAY)
            }
            return
        }

        if (frameClosed) {
            if (closedStartTime == null) {
                closedStartTime = System.currentTimeMillis()
            } else {
                val elapsed = System.currentTimeMillis() - closedStartTime!!
                if (elapsed > CLOSED_TIME_THRESHOLD) {
                    runOnUiThread {
                        statusText.text = "DROWSY"
                        statusText.setBackgroundColor(Color.RED)
                    }
                    if (!alarmPlayer.isPlaying) {
                        alarmPlayer.start()
                    }
                }
            }
        } else {
            closedStartTime = null
            runOnUiThread {
                statusText.text = "ALERT"
                statusText.setBackgroundColor(Color.GREEN)
            }
            if (alarmPlayer.isPlaying) {
                alarmPlayer.pause()
                alarmPlayer.seekTo(0)
            }
        }
    }

    private fun imageProxyToBuffer(image: ImageProxy): ByteBuffer {

        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(
            nv21,
            ImageFormat.NV21,
            image.width,
            image.height,
            null
        )

        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 90, out)
        val bitmap = BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())

        val matrix = Matrix()
        matrix.postRotate(image.imageInfo.rotationDegrees.toFloat())

        val rotatedBitmap = Bitmap.createBitmap(
            bitmap, 0, 0,
            bitmap.width, bitmap.height,
            matrix, true
        )

        val resized = Bitmap.createScaledBitmap(
            rotatedBitmap, INPUT_SIZE, INPUT_SIZE, true
        )

        val buffer = ByteBuffer.allocateDirect(1 * INPUT_SIZE * INPUT_SIZE * 3 * 4)
        buffer.order(ByteOrder.nativeOrder())

        for (y in 0 until INPUT_SIZE) {
            for (x in 0 until INPUT_SIZE) {
                val pixel = resized.getPixel(x, y)
                buffer.putFloat(((pixel shr 16) and 0xFF) / 255f)
                buffer.putFloat(((pixel shr 8) and 0xFF) / 255f)
                buffer.putFloat((pixel and 0xFF) / 255f)
            }
        }

        buffer.rewind()
        return buffer
    }
}
