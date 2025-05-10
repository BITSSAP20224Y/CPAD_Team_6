import 'dart:typed_data';
import 'dart:io';
import 'dart:math';
import 'dart:async';

import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as img;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class ObjectMatchAndroidPage extends StatefulWidget {
  @override
  _ObjectMatchAndroidPageState createState() => _ObjectMatchAndroidPageState();
}

class _ObjectMatchAndroidPageState extends State<ObjectMatchAndroidPage>
    with WidgetsBindingObserver {
  late Interpreter _interpreter;
  late IsolateInterpreter _isolateInterpreter;
  bool _isModelReady = false;
  List<double>? referenceFeatures;
  CameraController? _cameraController;
  bool _isProcessing = false;
  bool _cameraInitialized = false;
  bool _isMatching = false; // Track if object matching is active
  int _objectCount = 0; // Count of matched objects

  @override
  void initState() {
    super.initState();
    _initModel();
    _initCamera();
    WidgetsBinding.instance.addObserver(this);
  }

  Future<void> _initModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/MobileNet-v2.tflite');
      _isolateInterpreter = await IsolateInterpreter.create(
        address: _interpreter.address,
      );

      setState(() {
        _isModelReady = true;
      });
    } catch (e) {
      print("Error loading model: $e");
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text("Failed to load model")));
    }
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      _cameraController = CameraController(
        cameras[0],
        ResolutionPreset.medium,
        enableAudio: false,
      );
      await _cameraController!.initialize();
      setState(() {
        _cameraInitialized = true;
      });
    } catch (e) {
      print("Camera initialization failed: $e");
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text("Camera failed to initialize")));
    }
  }

  Future<void> _ensureCameraReady() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      await _initCamera();
    }
  }

  Future<List<double>> _extractFeatures(img.Image image) async {
    if (!_isModelReady) throw Exception("Interpreter not initialized yet.");

    final resized = img.copyResize(image, width: 224, height: 224);
    final input = Float32List(224 * 224 * 3);
    int idx = 0;

    for (var y = 0; y < 224; y++) {
      for (var x = 0; x < 224; x++) {
        final pixel = resized.getPixel(x, y);
        input[idx++] = (img.getRed(pixel) - 127.5) / 127.5;
        input[idx++] = (img.getGreen(pixel) - 127.5) / 127.5;
        input[idx++] = (img.getBlue(pixel) - 127.5) / 127.5;
      }
    }

    final output = List.filled(1000, 0.0).reshape([1, 1000]);
    await _isolateInterpreter.run(input.reshape([1, 224, 224, 3]), output);

    return output[0];
  }

  Future<void> _takeReferencePhoto() async {
    if (!_isModelReady) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text("Model not loaded yet")));
      return;
    }

    final picker = ImagePicker();
    final picked = await picker.pickImage(source: ImageSource.camera);
    if (picked != null) {
      final imageBytes = await picked.readAsBytes();
      final decoded = img.decodeImage(imageBytes);
      if (decoded != null) {
        referenceFeatures = await _extractFeatures(decoded);
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text("Reference object saved.")));
      }
    }
  }

  double _cosineSimilarity(List<double> a, List<double> b) {
    double dot = 0.0, normA = 0.0, normB = 0.0;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    return dot / (sqrt(normA) * sqrt(normB));
  }

  Future<void> _captureAndCompare() async {
    await _ensureCameraReady();

    if (_cameraController == null ||
        !_cameraController!.value.isInitialized ||
        !_cameraInitialized) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text("Camera not ready.")));
      return;
    }

    if (referenceFeatures == null) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text("Take a reference photo first.")));
      return;
    }

    if (_isProcessing) return;
    _isProcessing = true;

    try {
      // Optional delay to ensure stability
      await Future.delayed(Duration(milliseconds: 300));

      if (_cameraController!.value.isTakingPicture) {
        print("Camera is currently busy.");
        return;
      }

      final picture = await _cameraController!.takePicture();
      final imageBytes = await File(picture.path).readAsBytes();
      final image = img.decodeImage(imageBytes);

      if (image != null) {
        final currentFeatures = await _extractFeatures(image);
        final similarity = _cosineSimilarity(
          referenceFeatures!,
          currentFeatures,
        );
        print('Similarity: $similarity');

        if (similarity > 0.60) {
          setState(() {
            _objectCount++; // Increase count when similarity is above threshold
          });
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Object matched! Count: $_objectCount')),
          );
        }
      }
    } catch (e) {
      print('Error during capture: $e');
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(SnackBar(content: Text("Capture failed: $e")));
    } finally {
      _isProcessing = false;
    }
  }

  void _startMatching() {
    setState(() {
      _isMatching = true;
    });
    _matchObjectsContinuously();
  }

  void _stopMatching() {
    setState(() {
      _isMatching = false;
    });
  }

  Future<void> _matchObjectsContinuously() async {
    while (_isMatching) {
      await _captureAndCompare();
      if (!_isMatching) break; // Stop if the user has pressed Stop
      await Future.delayed(Duration(seconds: 2)); // Adjust delay between checks
    }
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _interpreter.close();
    WidgetsBinding.instance.removeObserver(this);
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    if (_cameraController == null || !_cameraController!.value.isInitialized)
      return;

    if (state == AppLifecycleState.inactive) {
      _cameraController?.dispose();
    } else if (state == AppLifecycleState.resumed) {
      _initCamera();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey.shade100,
      appBar: AppBar(
        title: Text("Inventory Tracker"),
        backgroundColor: Colors.indigo,
        centerTitle: true,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Expanded(
              flex: 5,
              child:
                  _cameraInitialized &&
                          _cameraController?.value.isInitialized == true
                      ? ClipRRect(
                        borderRadius: BorderRadius.circular(12),
                        child: CameraPreview(_cameraController!),
                      )
                      : Center(child: CircularProgressIndicator()),
            ),
            SizedBox(height: 24),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: _takeReferencePhoto,
                  icon: Icon(Icons.camera_alt),
                  label: Text("Reference"),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.indigo,
                    foregroundColor: Colors.white,
                    padding: EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                  ),
                ),
                ElevatedButton.icon(
                  onPressed: _isMatching ? null : _startMatching,
                  icon: Icon(Icons.play_arrow),
                  label: Text("Start"),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.green,
                    foregroundColor: Colors.white,
                    padding: EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                  ),
                ),
                ElevatedButton.icon(
                  onPressed: _stopMatching,
                  icon: Icon(Icons.stop),
                  label: Text("Stop"),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.red,
                    foregroundColor: Colors.white,
                    padding: EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                  ),
                ),
              ],
            ),
            SizedBox(height: 24),
            Text(
              "Object Count: $_objectCount",
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                color: Colors.indigo.shade700,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
