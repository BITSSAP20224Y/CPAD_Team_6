# CPAD_Team_6

### 👤 Team Members

- **Member 1:** Palaniappan Sockalingam (2022HS70058)
- **Member 2:** Avinash Maurya (2022HS70029)
- **Member 3:** Shubham Singh (2022HS70025)
- **Member 4:** Prem Kumar Mandal (2022HS70027)
- **Member 4:** Abhisek Mohanty (2022HS70051)

## Assignment 2 - OpenAPI Spec
[Swagger UI](https://secretkeepers.github.io/secret-keeper-service/#/)

## About the Project

This repository contains the code for a Flutter-based mobile application that performs **live object counting using a reference object**. Designed as part of CPAD course project, this app uses a camera feed to detect and count objects in real time, making it useful for applications such as inventory management, agriculture, packaging, and more.

---

## 🧠 Key Idea

The core concept of this project is to use a **user-defined reference object** for scale calibration, and then count similar objects in real time using live camera input. This allows for consistent object recognition regardless of distance or angle, as the reference provides a basis for estimating object size dynamically.

---

## 🚀 Features

- 📷 **Live Object Counting** using your device's camera  
- 📏 **Reference-Based Calibration** to account for varying object sizes  
- 🤖 **MobileNetV2 Integration** for efficient object detection
- ✅ **User-friendly Interface** for capturing, confirming, and counting  
- 🔄 **Real-time Feedback** as objects enter or leave the camera view  

---

## 🎬 How It Works

1. **Launch the App**
2. **Click "Reference"** to capture the reference object (e.g., a coin, card, etc.).
3. **Confirm** the captured object to set the scale.
4. **Click "Start"** to begin live object counting via your device's camera.

---

## 🧩 Tech Stack

- Flutter (Frontend)
- **MobileNetV2** (TensorFlow Lite) for on-device object detection
- Dart language
- Native Android/iOS camera APIs

---

## 📲 Installation

### Prerequisites

- Flutter SDK
- Android Studio or VS Code
- Physical device or emulator (Camera required)

### Steps

```bash
git clone https://github.com/BITSSAP20224Y/CPAD_Team_6.git
cd CPAD_Team_6
flutter pub get
flutter run
