package com.example.masterapp

import android.os.Bundle
import android.view.View
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Spinner
import androidx.appcompat.app.AppCompatActivity

class SettingsActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_settings)

        val themeSpinner: Spinner = findViewById(R.id.theme_spinner)
        val themes = arrayOf("Light Theme", "Dark Theme")

        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, themes)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        themeSpinner.adapter = adapter

        themeSpinner.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>, view: View, position: Int, id: Long) {
                // Handle theme selection
            }

            override fun onNothingSelected(parent: AdapterView<*>) {
                // Do nothing
            }
        }

        findViewById<View>(R.id.apply_theme_button).setOnClickListener {
            val selectedTheme = themeSpinner.selectedItemPosition
            when (selectedTheme) {
                0 -> setTheme(R.style.AppTheme)
                1 -> setTheme(R.style.DarkTheme)
            }
            recreate()
        }
    }
}