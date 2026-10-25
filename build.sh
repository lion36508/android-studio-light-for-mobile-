#!/bin/bash
echo "[*] Cleaning old build files..."
rm -rf obj/* bin/*

echo "[*] Compiling Resources (Packing internal assets)..."
aapt package -f -m -J src/main/java/ -S res/ -M AndroidManifest.xml -A assets/ -I android.jar

echo "[*] Compiling Java Code..."
ecj -d obj -classpath android.jar -sourcepath src/main/java/ src/main/java/com/bikash/mobileide/*.java

echo "[*] Converting to DEX..."
d8 --release --output bin/ obj/com/bikash/mobileide/*.class

echo "[*] Packaging APK..."
aapt package -f -M AndroidManifest.xml -S res/ -A assets/ -I android.jar -F bin/app-unsigned.apk
cd bin
zip -q -u app-unsigned.apk classes.dex
cd ..

echo "[*] Signing APK..."
apksigner sign --ks ide_keystore.jks --ks-pass pass:android --out bin/MobileIDE.apk bin/app-unsigned.apk

echo "[*] Copying to File Manager..."
cp bin/MobileIDE.apk ~/storage/downloads/MobileIDE.apk

echo "=========================================="
echo "✅ BUILD SUCCESSFUL! 'assets' folder is packed inside."
echo "=========================================="
