# TFLite - Prevent R8 from stripping needed GPU delegate classes
-keep class org.tensorflow.** { *; }
-keepclassmembers class org.tensorflow.** { *; }
-dontwarn org.tensorflow.**
