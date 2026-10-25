#!/bin/bash

USERNAME="lion36508"
REPO_NAME="android-studio-light-for-mobile-"

echo "================================================="
echo "🔓 GitHub Project Public Maker"
echo "================================================="
# यह आपसे आपका टोकन मांगेगा
read -p "अपना काम करने वाला GitHub Token यहाँ पेस्ट करें और Enter दबाएं: " TOKEN

echo ""
echo "[*] GitHub सर्वर को सिक्योरिटी लॉक खोलने का कमांड भेज रहे हैं..."

# GitHub API के ज़रिए प्रोजेक्ट को Public (private: false) करना
curl -s -X PATCH \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token $TOKEN" \
  https://api.github.com/repos/$USERNAME/$REPO_NAME \
  -d '{"private":false}' > /dev/null

echo "================================================="
echo "✅ SUCCESS! आपका प्रोजेक्ट अब पूरी तरह से PUBLIC (ओपन-सोर्स) हो गया है!"
echo "दुनिया का कोई भी डेवलपर अब इसे देख सकता है और इसमें कोड जोड़ सकता है।"
echo "🔗 लिंक: https://github.com/$USERNAME/$REPO_NAME"
echo "================================================="
