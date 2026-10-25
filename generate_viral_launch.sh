#!/bin/bash

echo "================================================="
echo "🚀 OxDroid Global Launch Kit Generator"
echo "================================================="
echo "[*] दुनिया भर के प्लेटफॉर्म्स के लिए पोस्ट तैयार की जा रही हैं..."
sleep 2

mkdir -p LaunchKit

# 1. Reddit Post (For r/androiddev, r/opensource, r/programming)
cat << 'REDDIT' > LaunchKit/reddit_post.txt
Title: I built a complete Android Studio alternative that runs natively on Android (Open Source) 📱💻

Hey everyone,
Not everyone has a high-end PC to start coding. Millions of students only have an Android phone. 

To solve this, I built **OxDroid Workspace** - a 100% open-source, native mobile IDE built on Android, for Android. It lets you write, compile, and run apps directly on your phone without needing a PC.

🌟 Features:
- Native Compilation
- No PC required
- Completely Open Source

I would love for the community to try it, review the architecture, and contribute! 
Link: https://github.com/lion36508/android-studio-light-for-mobile-
REDDIT

# 2. Hacker News Post (Show HN)
cat << 'HN' > LaunchKit/hacker_news_post.txt
Title: Show HN: OxDroid - A 100% native Android Studio alternative for mobile
Body:
Hi HN, I built OxDroid to democratize app development. It's a lightweight, open-source IDE that runs directly on Android devices, allowing users without PCs to build apps natively. Would love your feedback on the build engine and architecture.
Repo: https://github.com/lion36508/android-studio-light-for-mobile-
HN

# 3. Twitter / X Post
cat << 'TWITTER' > LaunchKit/twitter_post.txt
Not everyone has a 16GB RAM PC to learn Android dev. So I built an open-source alternative to Android Studio that runs directly on your phone! 📱🔥

Meet OxDroid Workspace - code, compile, and build apps anywhere. 

Drop a ⭐ if you support open-source: https://github.com/lion36508/android-studio-light-for-mobile- 
#AndroidDev #OpenSource #100DaysOfCode #GitHub
TWITTER

echo "✅ SUCCESS! 'LaunchKit' फोल्डर में आपकी सारी पोस्ट तैयार हैं।"
echo "अब बस इन्हें कॉपी करें और Reddit, Hacker News, और Twitter पर पोस्ट कर दें!"
echo "================================================="
