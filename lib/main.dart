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

  // DeepLabv3でセグメンテーションを実行 (入力形状を動的に反映)
  dlv(File imageFile) async {
    try {
      // 画像を読み込み
      var imageBytes = await imageFile.readAsBytes();
      var inputImage = img.decodeImage(imageBytes);
      if (inputImage == null) {
        throw Exception("画像のデコードに失敗しました。");
      }

      // --- モデルが期待する入力形状を取得 ---
      var inputTensorInfo = _interpreter!.getInputTensor(0);
      var inputShape = inputTensorInfo.shape; // 例: [1, 321, 321, 3]
      int batch      = inputShape[0];
      int inHeight   = inputShape[1];
      int inWidth    = inputShape[2];
      int inChannels = inputShape[3];

      // 画像をリサイズ (モデルに合わせる)
      var resizedImage = img.copyResize(
        inputImage,
        width: inWidth,
        height: inHeight,
      );

      // 入力テンソルの型判定 (Float32 / Uint8)
      dynamic inputTensor;
      if (inputTensorInfo.type.toString().toLowerCase().contains('float32')) {
        Float32List buffer = imageToByteListFloat32(resizedImage, inWidth, inHeight);
        // [batch, inHeight, inWidth, inChannels] に reshape
        inputTensor = buffer.reshape([batch, inHeight, inWidth, inChannels]);
      } else if (inputTensorInfo.type.toString().toLowerCase().contains('uint8')) {
        Uint8List buffer = imageToByteListUint8(resizedImage, inWidth, inHeight);
        inputTensor = buffer.reshape([batch, inHeight, inWidth, inChannels]);
      } else {
        throw Exception("Unsupported input tensor type: ${inputTensorInfo.type}");
      }

      // --- 出力形状を取得して、対応するバッファを作成 ---
      var outputTensorInfo = _interpreter!.getOutputTensor(0);
      var outputShape = outputTensorInfo.shape; // 例: [1, 321, 321, 55]
      int outBatch    = outputShape[0]; // ※ outBatch は使わないが、assertでチェックする
      int outHeight   = outputShape[1];
      int outWidth    = outputShape[2];
      int outChannels = outputShape[3];
      assert(outBatch == 1, "Batch size must be 1 for this model.");

      dynamic output;
      if (outputTensorInfo.type.toString().toLowerCase().contains('float32')) {
        output = Float32List(outBatch * outHeight * outWidth * outChannels)
            .reshape([outBatch, outHeight, outWidth, outChannels]);
      } else if (outputTensorInfo.type.toString().toLowerCase().contains('uint8')) {
        output = Uint8List(outBatch * outHeight * outWidth * outChannels)
            .reshape([outBatch, outHeight, outWidth, outChannels]);
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
  // 幅・高さを引数にしておく (念のため)
  Float32List imageToByteListFloat32(img.Image image, int inWidth, int inHeight) {
    var buffer = Float32List(inWidth * inHeight * 3).buffer;
    var pixels = buffer.asFloat32List();
    int pixelIndex = 0;

    try {
      for (var y = 0; y < inHeight; y++) {
        for (var x = 0; x < inWidth; x++) {
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

  // 画像をUint8のバイトリストに変換する関数
  Uint8List imageToByteListUint8(img.Image image, int inWidth, int inHeight) {
    var convertedBytes = Uint8List(inWidth * inHeight * 3);
    int pixelIndex = 0;
    try {
      for (var y = 0; y < inHeight; y++) {
        for (var x = 0; x < inWidth; x++) {
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
      // output は [batch, outHeight, outWidth, outChannels] の 4 次元リストであるが、
      // バッチは 1 と仮定し、最初の要素を取り出す
      var outputData;
      if (output is List && output.length == 1) {
        outputData = output[0];
      } else {
        outputData = output;
      }
      // 出力の次元を出力データから取得
      int outHeight = outputData.length;
      int outWidth = outputData[0].length;
      int outChannels = outputData[0][0].length;

      var mask = img.Image(width: outWidth, height: outHeight);

      for (int y = 0; y < outHeight; y++) {
        for (int x = 0; x < outWidth; x++) {
          double maxScore = (outputData[y][x][0] as num).toDouble();
          int label = 0;
          for (int c = 1; c < outChannels; c++) {
            double score = (outputData[y][x][c] as num).toDouble();
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