package com.bikash.mobileide;

import android.os.Build;
import android.view.View;
import android.view.WindowInsets;

/**
 * Keeps application content clear of system bars on Android 15+,
 * where edge-to-edge layout is enforced for apps targeting API 35+.
 */
public final class SystemBarInsets {
    private SystemBarInsets() {}

    @SuppressWarnings("deprecation")
    public static void apply(View root) {
        if (root == null || Build.VERSION.SDK_INT < 35) {
            return;
        }

        final int baseLeft = root.getPaddingLeft();
        final int baseTop = root.getPaddingTop();
        final int baseRight = root.getPaddingRight();
        final int baseBottom = root.getPaddingBottom();

        root.setOnApplyWindowInsetsListener(new View.OnApplyWindowInsetsListener() {
            @Override
            public WindowInsets onApplyWindowInsets(View view, WindowInsets insets) {
                view.setPadding(
                    baseLeft + insets.getSystemWindowInsetLeft(),
                    baseTop + insets.getSystemWindowInsetTop(),
                    baseRight + insets.getSystemWindowInsetRight(),
                    baseBottom + insets.getSystemWindowInsetBottom()
                );
                return insets;
            }
        });

        root.post(new Runnable() {
            @Override
            public void run() {
                root.requestApplyInsets();
            }
        });
    }
}
