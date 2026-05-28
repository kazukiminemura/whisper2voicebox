# whisper2voicebox

マイク入力を OpenVINO GenAI の Whisper で文字起こしし、VOICEVOX のずんだもん音声で復唱するCLIです。

## 使い方

1. 依存関係を入れます。

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

2. 必要なら `config.example.json` を `config.json` にコピーして設定を変更します。

   ```powershell
   Copy-Item config.example.json config.json
   ```

3. 起動します。

   ```powershell
   python main.py
   ```

起動後は、Enterで録音開始、もう一度Enterで録音停止します。文字起こしされた内容が、ずんだもん「ノーマル」の声で再生されます。

VOICEVOX Engine が見つからない場合、初回実行時に GitHub Releases から Windows CPU版を `tools/voicevox_engine` へ自動ダウンロードして展開します。

初回実行時、`openai/whisper-large-v3` を Hugging Face から取得して OpenVINO 形式に変換します。変換済みモデルは `models/whisper-large-v3-openvino` に保存され、次回以降は再利用されます。

## 設定

主な設定は `config.json` で変更できます。

- `voicevox_auto_install`: `true` なら VOICEVOX Engine を自動セットアップ
- `voicevox_engine_package`: 既定は `windows-cpu`
- `voicevox_engine_path`: 既存の `run.exe` を使いたい場合だけ指定
- `openvino_device`: Intel GPUを使う場合は `GPU`
- `openvino_weight_format`: 既定は `fp16`。メモリが厳しい場合は `int8`
- `input_device`: マイクのデバイスID。通常は `null` で既定デバイス
- `output_device`: 再生デバイスID。通常は `null` で既定デバイス
- `language`: 日本語固定なら `<|ja|>`、自動判定なら `null`
- `speaker_name` / `speaker_style`: 既定は `ずんだもん` / `ノーマル`

デバイス一覧はPythonから確認できます。

```powershell
python -c "import sounddevice as sd; print(sd.query_devices())"
```
