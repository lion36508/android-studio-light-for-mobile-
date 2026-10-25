#!/bin/bash

echo "================================================="
echo "🛡 Smart OxDroid 24/7 Background Guard Started"
echo "================================================="

# हमारी सुरक्षित (Whitelist) फाइल्स जिन्हें गार्ड इग्नोर करेगा
WHITELIST_APK="./bin/MobileIDE.apk"
WHITELIST_KEYSTORE="./ide_keystore.jks"

while true; do
    if ping -c 1 8.8.8.8 > /dev/null 2>&1; then
        
        # सिर्फ उन फाइल्स को ढूंढेगा जो हमारी सुरक्षित लिस्ट से बाहर हैं
        UNAUTHORIZED=$(find . -type f \( -name "*.apk" -o -name "*.keystore" -o -name "*.jks" \) ! -path "$WHITELIST_APK" ! -path "$WHITELIST_KEYSTORE")
        
        if [ -n "$UNAUTHORIZED" ]; then
            echo "[$(date '+%T')] 🚨 ALERT: नई अनधिकृत फाइल मिली -> $UNAUTHORIZED"
            
            # आपके फोन की स्क्रीन पर डायरेक्ट नोटिफिकेशन भेजेगा
            termux-notification --title "🛡️ OxDroid Security Alert" --content "प्रोजेक्ट में कोई अनजान APK या Keystore मिली है! तुरंत Termux चेक करें।" --priority high 2>/dev/null
        fi
        
        if [ -d ".git" ]; then
            git status > /dev/null 2>&1
        fi
        
        # हर 60 सेकंड में चेक करेगा (बैटरी बचाने के लिए)
        sleep 60
    else
        sleep 10
    fi
done
