package com.example.dddapp

import android.content.Context
import android.graphics.*
import android.util.AttributeSet
import android.view.View

data class Detection(
    val x1: Float,
    val y1: Float,
    val x2: Float,
    val y2: Float,
    val score: Float,
    val label: String
)

class OverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null
) : View(context, attrs) {

    private val boxPaint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 8f
        isAntiAlias = true
    }

    private val textPaint = Paint().apply {
        color = Color.WHITE
        textSize = 55f
        typeface = Typeface.DEFAULT_BOLD
        isAntiAlias = true
    }

    private val bgPaint = Paint().apply {
        style = Paint.Style.FILL
        isAntiAlias = true
    }

    private var detections: List<Detection> = emptyList()

    fun setResults(results: List<Detection>) {
        detections = results
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        for (det in detections) {

            // Color based on class
            if (det.label == "closed_eye") {
                boxPaint.color = Color.RED
                bgPaint.color = Color.RED
            } else {
                boxPaint.color = Color.BLUE
                bgPaint.color = Color.BLUE
            }

            // Draw box
            canvas.drawRect(det.x1, det.y1, det.x2, det.y2, boxPaint)

            // Draw label background
            val text = "${det.label} ${"%.2f".format(det.score)}"
            val textWidth = textPaint.measureText(text)
            val textHeight = textPaint.textSize

            canvas.drawRect(
                det.x1,
                det.y1 - textHeight - 20,
                det.x1 + textWidth + 20,
                det.y1,
                bgPaint
            )

            // Draw label text
            canvas.drawText(
                text,
                det.x1 + 10,
                det.y1 - 10,
                textPaint
            )
        }
    }
}
