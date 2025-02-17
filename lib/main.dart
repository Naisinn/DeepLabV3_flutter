import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image/image.dart' as img;

void main() => runApp(MyApp());

// 定数の定義
const String dlv3 = 'DeepLabv3';
late int _numClasses; // 自動検出されるクラス数

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: TfliteHome(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class TfliteHome extends StatefulWidget {
  @override
  _TfliteHomeState createState() => _TfliteHomeState();
}

class _TfliteHomeState extends State<TfliteHome> {
  String _model = dlv3;

  File? _image;

  double? _imageWidth;
  double? _imageHeight;
  bool _busy = false;
  String _errorMessage = ""; // エラーメッセージ用の変数

  Interpreter? _interpreter;
  img.Image? _segmentationMask;

  @override
  void initState() {
    super.initState();
    _busy = true;
    loadModel().then((_) {
      setState(() {
        _busy = false;
      });
    });
  }

  // 色チャンネルを抽出するヘルパー関数
  int getRed(color) => color is int ? (color >> 16) & 0xFF : color.r;
  int getGreen(color) => color is int ? (color >> 8) & 0xFF : color.g;
  int getBlue(color) => color is int ? color & 0xFF : color.b;
  int getAlpha(color) => color is int ? (color >> 24) & 0xFF : color.a;

  // ARGB の img.Color を生成する関数 (修正箇所)
  img.Color getColor(int r, int g, int b, [int a = 255]) {
    return img.ColorInt8.rgba(r, g, b, a);
  }

  // モデルの読み込み
  Future<void> loadModel() async {
    try {
      // TFLiteモデルをロード
      _interpreter = await Interpreter.fromAsset('assets/tflite/Audi_20250208-223050.tflite');
      // 出力テンソルの形状からクラス数を自動検出
      _numClasses = _interpreter!.getOutputTensor(0).shape.last;
      print('Detected num_classes: $_numClasses');
    } catch (e, stacktrace) {
      String errorDetails = "モデルの読み込みに失敗しました: $e\n$stacktrace";
      print(errorDetails);
      setState(() {
        _errorMessage = errorDetails;
      });
    }
  }

  // ギャラリーから画像を選択
  selectFromImagePicker() async {
    try {
      var imagePicker = ImagePicker();
      var pickedFile = await imagePicker.pickImage(source: ImageSource.gallery);
      if (pickedFile == null) return;
      setState(() {
        _busy = true;
        _errorMessage = "";
      });
      await predictImage(File(pickedFile.path));
    } catch (e, stacktrace) {
      String errorDetails = "ギャラリーからの画像選択中にエラーが発生しました: $e\n$stacktrace";
      print(errorDetails);
      setState(() {
        _errorMessage = errorDetails;
        _busy = false;
      });
    }
  }

  // カメラから画像を撮影
  selectFromCamera() async {
    try {
      var imagePicker = ImagePicker();
      var pickedFile = await imagePicker.pickImage(source: ImageSource.camera);
      if (pickedFile == null) return;
      setState(() {
        _busy = true;
        _errorMessage = "";
      });
      await predictImage(File(pickedFile.path));
    } catch (e, stacktrace) {
      String errorDetails = "カメラからの画像撮影中にエラーが発生しました: $e\n$stacktrace";
      print(errorDetails);
      setState(() {
        _errorMessage = errorDetails;
        _busy = false;
      });
    }
  }

  // 画像の予測を行う
  predictImage(File image) async {
    try {
      // 画像の幅と高さを取得
      FileImage(image).resolve(ImageConfiguration()).addListener(
        ImageStreamListener((ImageInfo info, bool _) {
          setState(() {
            _imageWidth = info.image.width.toDouble();
            _imageHeight = info.image.height.toDouble();
          });
        }),
      );

      if (_model == dlv3) {
        await dlv(image);
      }

      setState(() {
        _image = image;
        _busy = false;
      });
    } catch (e, stacktrace) {
      String errorDetails = "画像の予測中にエラーが発生しました: $e\n$stacktrace";
      print(errorDetails);
      setState(() {
        _errorMessage = errorDetails;
        _busy = false;
      });
    }
  }

  // DeepLabv3でセグメンテーションを実行
  dlv(File imageFile) async {
    try {
      // 画像を読み込み
      var imageBytes = await imageFile.readAsBytes();
      var inputImage = img.decodeImage(imageBytes);
      if (inputImage == null) {
        throw Exception("画像のデコードに失敗しました。");
      }

      // 画像をリサイズ
      var resizedImage = img.copyResize(inputImage, width: 257, height: 257);

      // 入力テンソルの作成：モデルの入力型に応じて変換
      var inputTensorInfo = _interpreter!.getInputTensor(0);
      dynamic inputTensor;
      if (inputTensorInfo.type.toString().toLowerCase().contains('float32')) {
        // --- ここで reshape して「[1, 257, 257, 3]」にする ---
        Float32List buffer = imageToByteListFloat32(resizedImage);
        inputTensor = buffer.reshape([1, 257, 257, 3]);
      } else if (inputTensorInfo.type.toString().toLowerCase().contains('uint8')) {
        // --- 同様に reshape ---
        Uint8List buffer = imageToByteListUint8(resizedImage);
        inputTensor = buffer.reshape([1, 257, 257, 3]);
      } else {
        throw Exception("Unsupported input tensor type: ${inputTensorInfo.type}");
      }

      // 出力テンソルの作成：モデルの出力型に応じて作成
      var outputTensorInfo = _interpreter!.getOutputTensor(0);
      dynamic output;
      if (outputTensorInfo.type.toString().toLowerCase().contains('float32')) {
        output = Float32List(1 * 257 * 257 * _numClasses)
            .reshape([1, 257, 257, _numClasses]);
      } else if (outputTensorInfo.type.toString().toLowerCase().contains('uint8')) {
        output = Uint8List(1 * 257 * 257 * _numClasses)
            .reshape([1, 257, 257, _numClasses]);
      } else {
        throw Exception("Unsupported output tensor type: ${outputTensorInfo.type}");
      }

      // 推論を実行
      _interpreter?.run(inputTensor, output);

      // セグメンテーションマスクを生成
      var maskBytes = await processOutput(output);

      setState(() {
        _segmentationMask = img.decodeImage(maskBytes);
      });
    } catch (e, stacktrace) {
      String errorDetails = "DeepLabv3の処理中にエラーが発生しました: $e\n$stacktrace";
      print(errorDetails);
      setState(() {
        _errorMessage = errorDetails;
      });
    }
  }

  // 画像をFloat32のバイトリストに変換
  Float32List imageToByteListFloat32(img.Image image) {
    var buffer = Float32List(1 * 257 * 257 * 3).buffer;
    var pixels = buffer.asFloat32List();
    int pixelIndex = 0;

    try {
      for (var y = 0; y < 257; y++) {
        for (var x = 0; x < 257; x++) {
          var pixel = image.getPixel(x, y);
          pixels[pixelIndex++] = (getRed(pixel) - 127.5) / 127.5;
          pixels[pixelIndex++] = (getGreen(pixel) - 127.5) / 127.5;
          pixels[pixelIndex++] = (getBlue(pixel) - 127.5) / 127.5;
        }
      }
    } catch (e, stacktrace) {
      String errorDetails =
          "画像をFloat32のバイトリストに変換中にエラーが発生しました: $e\n$stacktrace";
      print(errorDetails);
      setState(() {
        _errorMessage = errorDetails;
      });
    }
    return buffer.asFloat32List();
  }

  // 画像をUint8のバイトリストに変換する関数（deeplabv3_257_mv_gpu.tflite 用）
  Uint8List imageToByteListUint8(img.Image image) {
    var convertedBytes = Uint8List(1 * 257 * 257 * 3);
    int pixelIndex = 0;
    try {
      for (var y = 0; y < 257; y++) {
        for (var x = 0; x < 257; x++) {
          var pixel = image.getPixel(x, y);
          convertedBytes[pixelIndex++] = getRed(pixel);
          convertedBytes[pixelIndex++] = getGreen(pixel);
          convertedBytes[pixelIndex++] = getBlue(pixel);
        }
      }
    } catch (e, stacktrace) {
      String errorDetails =
          "画像をUint8のバイトリストに変換中にエラーが発生しました: $e\n$stacktrace";
      print(errorDetails);
      setState(() {
        _errorMessage = errorDetails;
      });
    }
    return convertedBytes;
  }

  // モデルの出力を処理してセグメンテーションマスクを生成
  Future<Uint8List> processOutput(dynamic output) async {
    try {
      var outputData = output.reshape([257, 257, _numClasses]);
      var mask = img.Image(width: 257, height: 257);

      for (int y = 0; y < 257; y++) {
        for (int x = 0; x < 257; x++) {
          double maxScore = outputData[y][x][0].toDouble();
          int label = 0;
          for (int c = 1; c < _numClasses; c++) {
            double score = outputData[y][x][c].toDouble();
            if (score > maxScore) {
              maxScore = score;
              label = c;
            }
          }
          if (label == 15) {
            // 赤色 (RGB: 255, 0, 0, Alpha: 255)
            mask.setPixel(x, y, getColor(255, 0, 0, 255));
          } else {
            // 透明 (RGB: 0, 0, 0, Alpha: 0)
            mask.setPixel(x, y, getColor(0, 0, 0, 0));
          }
        }
      }

      // 画像をPNG形式にエンコード
      return img.encodePng(mask);
    } catch (e, stacktrace) {
      String errorDetails =
          "モデル出力の処理中にエラーが発生しました: $e\n$stacktrace";
      print(errorDetails);
      setState(() {
        _errorMessage = errorDetails;
      });
      rethrow;
    }
  }

  @override
  Widget build(BuildContext context) {
    // 画面サイズを取得
    Size size = MediaQuery.of(context).size;

    // 画像の表示サイズを計算
    double finalW;
    double finalH;

    if (_imageWidth == null || _imageHeight == null) {
      finalW = size.width;
      finalH = size.height;
    } else {
      double ratioW = size.width / _imageWidth!;
      double ratioH = size.height / _imageHeight!;
      finalW = _imageWidth! * ratioW;
      finalH = _imageHeight! * ratioH;
    }

    List<Widget> stackChildren = [];

    // 読み込み中の場合のインジケーター
    if (_busy) {
      stackChildren.add(
        Positioned(
          top: 0,
          left: 0,
          child: Center(
            child: CircularProgressIndicator(),
          ),
        ),
      );
    }

    // 画像のプレビュー
    stackChildren.add(
      Positioned(
        top: 0.0,
        left: 0.0,
        width: finalW,
        height: finalH,
        child: _image == null
            ? Center(
          child: Text('カメラまたはギャラリーから画像を選択してください'),
        )
            : Image.file(
          _image!,
          fit: BoxFit.fill,
        ),
      ),
    );

    // セグメンテーションマスクの表示
    if (_segmentationMask != null) {
      stackChildren.add(
        Positioned(
          top: 0,
          left: 0,
          width: finalW,
          height: finalH,
          child: Opacity(
            opacity: 0.7,
            child: Image.memory(
              img.encodePng(_segmentationMask!),
              fit: BoxFit.fill,
            ),
          ),
        ),
      );
    }

    // エラーメッセージの表示
    if (_errorMessage.isNotEmpty) {
      stackChildren.add(
        Positioned(
          bottom: 20,
          left: 10,
          right: 10,
          child: Container(
            padding: EdgeInsets.all(8),
            color: Colors.black54,
            child: SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Text(
                _errorMessage,
                style: TextStyle(color: Colors.white, fontSize: 12),
              ),
            ),
          ),
        ),
      );
    }

    return Scaffold(
      appBar: AppBar(
        title: Text('オンデバイス画像セグメンテーション'),
        backgroundColor: Colors.redAccent,
      ),
      floatingActionButton: Stack(
        children: <Widget>[
          Padding(
            padding: EdgeInsets.all(10),
            child: Align(
              alignment: Alignment.bottomCenter,
              child: FloatingActionButton(
                child: Icon(Icons.image),
                tooltip: 'ギャラリーから画像を選択',
                backgroundColor: Colors.purpleAccent,
                onPressed: selectFromImagePicker,
              ),
            ),
          ),
          Padding(
            padding: EdgeInsets.all(10),
            child: Align(
              alignment: Alignment.bottomRight,
              child: FloatingActionButton(
                child: Icon(Icons.camera),
                backgroundColor: Colors.redAccent,
                tooltip: 'カメラから画像を撮影',
                onPressed: selectFromCamera,
              ),
            ),
          )
        ],
      ),
      body: Stack(
        children: stackChildren,
      ),
    );
  }
}