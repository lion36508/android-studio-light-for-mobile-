#!/bin/bash

USERNAME="lion36508"
REPO_NAME="android-studio-light-for-mobile-"

echo "================================================="
echo "🌍 GitHub Open Community Setup"
echo "================================================="
read -p "अपना काम करने वाला नया GitHub Token यहाँ पेस्ट करें और Enter दबाएं: " TOKEN

echo ""
echo "[*] प्रोजेक्ट में सभी कम्युनिटी फीचर्स (Discussions, Wiki, Issues) चालू किए जा रहे हैं..."

# GitHub API के ज़रिए सभी ओपन-सोर्स सेटिंग्स को On (true) करना
curl -s -X PATCH \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token $TOKEN" \
  https://api.github.com/repos/$USERNAME/$REPO_NAME \
  -d '{
    "has_issues": true,
    "has_projects": true,
    "has_wiki": true,
    "has_discussions": true,
    "allow_forking": true
  }' > /dev/null

echo "================================================="
echo "✅ SUCCESS! तुम्हारा प्रोजेक्ट अब एक फुल ओपन-सोर्स हब बन गया है।"
echo "अब डेवलपर्स 'Discussions' में बात कर सकते हैं, 'Wiki' पढ़ सकते हैं, और आसानी से 'Fork' करके अपना कोड जमा कर सकते हैं।"
echo "================================================="
