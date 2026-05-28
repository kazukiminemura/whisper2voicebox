from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CONFIG_PATH = Path("config.json")
DEFAULT_CONFIG_PATH = Path("config.example.json")


@dataclass(frozen=True)
class AppConfig:
    voicevox_engine_path: str | None
    voicevox_auto_install: bool
    voicevox_engine_package: str
    voicevox_install_dir: str
    voicevox_engine_args: list[str]
    voicevox_url: str
    speaker_name: str
    speaker_style: str
    whisper_model_id: str
    whisper_model_dir: str
    openvino_weight_format: str | None
    openvino_device: str
    sample_rate: int
    input_device: int | None
    output_device: int | None
    max_new_tokens: int
    language: str | None


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    source_path = path if path.exists() else DEFAULT_CONFIG_PATH
    if not source_path.exists():
        raise FileNotFoundError(f"{path} と {DEFAULT_CONFIG_PATH} が見つかりません。")

    if source_path == DEFAULT_CONFIG_PATH:
        print(f"{path} がないため、{DEFAULT_CONFIG_PATH} の既定設定で起動します。")

    with source_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return AppConfig(
        voicevox_engine_path=data.get("voicevox_engine_path"),
        voicevox_auto_install=bool(data.get("voicevox_auto_install", True)),
        voicevox_engine_package=data.get("voicevox_engine_package", "windows-cpu"),
        voicevox_install_dir=data.get("voicevox_install_dir", "tools/voicevox_engine"),
        voicevox_engine_args=list(data.get("voicevox_engine_args", [])),
        voicevox_url=data.get("voicevox_url", "http://127.0.0.1:50021").rstrip("/"),
        speaker_name=data.get("speaker_name", "ずんだもん"),
        speaker_style=data.get("speaker_style", "ノーマル"),
        whisper_model_id=data.get("whisper_model_id", "openai/whisper-large-v3"),
        whisper_model_dir=data.get("whisper_model_dir", "models/whisper-large-v3-openvino"),
        openvino_weight_format=data.get("openvino_weight_format", "fp16"),
        openvino_device=data.get("openvino_device", "GPU"),
        sample_rate=int(data.get("sample_rate", 16000)),
        input_device=data.get("input_device"),
        output_device=data.get("output_device"),
        max_new_tokens=int(data.get("max_new_tokens", 256)),
        language=data.get("language"),
    )


def ensure_openvino_model(config: AppConfig) -> Path:
    model_dir = Path(config.whisper_model_dir)
    if (model_dir / "openvino_encoder_model.xml").exists():
        return model_dir

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"WhisperモデルをOpenVINO形式に変換します: {config.whisper_model_id}")
    print("初回はダウンロードと変換に時間がかかります。")

    command = [
        "optimum-cli",
        "export",
        "openvino",
        "--model",
        config.whisper_model_id,
        "--trust-remote-code",
        str(model_dir),
    ]
    if config.openvino_weight_format:
        command[-1:-1] = ["--weight-format", config.openvino_weight_format]
    subprocess.run(command, check=True)
    return model_dir


def wait_for_voicevox(url: str, timeout_seconds: float = 30.0) -> None:
    import requests

    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None

    while time.monotonic() < deadline:
        try:
            response = requests.get(f"{url}/version", timeout=1.0)
            if response.ok:
                return
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(0.5)

    detail = f" 最後のエラー: {last_error}" if last_error else ""
    raise TimeoutError(f"VOICEVOX Engine に接続できませんでした: {url}{detail}")


def find_installed_voicevox_engine(install_dir: Path) -> Path | None:
    if not install_dir.exists():
        return None

    candidates = sorted(install_dir.rglob("run.exe"))
    return candidates[0] if candidates else None


def download_file(url: str, path: Path) -> None:
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "whisper2voicebox"})
    with urllib.request.urlopen(request) as response, path.open("wb") as f:
        shutil.copyfileobj(response, f)


def combine_split_archive(parts: list[Path], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output)


def github_latest_voicevox_assets() -> tuple[str, list[dict[str, Any]]]:
    import urllib.error
    import urllib.request

    url = "https://api.github.com/repos/VOICEVOX/voicevox_engine/releases/latest"
    request = urllib.request.Request(url, headers={"User-Agent": "whisper2voicebox"})
    try:
        with urllib.request.urlopen(request) as response:
            release = json.load(response)
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "VOICEVOX Engineの自動取得に失敗しました。"
            "GitHubへ接続できるネットワークで再実行してください。"
        ) from exc
    return release["tag_name"], release["assets"]


def install_voicevox_engine(config: AppConfig) -> Path:
    install_dir = Path(config.voicevox_install_dir)
    existing = find_installed_voicevox_engine(install_dir)
    if existing:
        return existing

    tag, assets = github_latest_voicevox_assets()
    package = re.escape(config.voicevox_engine_package)
    pattern = re.compile(rf"^voicevox_engine-{package}-{re.escape(tag)}\.7z\.(\d{{3}})$")
    matching_assets = []
    for asset in assets:
        match = pattern.match(asset["name"])
        if match:
            matching_assets.append((int(match.group(1)), asset))

    if not matching_assets:
        raise RuntimeError(
            f"VOICEVOX Engineの配布物が見つかりません: {config.voicevox_engine_package} / {tag}"
        )

    matching_assets.sort(key=lambda item: item[0])
    downloads_dir = install_dir / "_downloads" / tag
    archive_paths: list[Path] = []

    print(f"VOICEVOX Engine {tag} ({config.voicevox_engine_package}) をダウンロードします。")
    for _, asset in matching_assets:
        archive_path = downloads_dir / asset["name"]
        archive_paths.append(archive_path)
        if archive_path.exists() and archive_path.stat().st_size == asset.get("size"):
            continue
        print(f"  download: {asset['name']}")
        download_file(asset["browser_download_url"], archive_path)

    extract_dir = install_dir / tag
    extract_dir.mkdir(parents=True, exist_ok=True)
    print("VOICEVOX Engineを展開します。")
    combined_archive = downloads_dir / f"voicevox_engine-{config.voicevox_engine_package}-{tag}.7z"
    if not combined_archive.exists():
        print("分割アーカイブを結合します。")
        combine_split_archive(archive_paths, combined_archive)

    tar_path = shutil.which("tar")
    if not tar_path:
        raise RuntimeError("7z展開に使う tar.exe が見つかりません。")
    subprocess.run(
        [tar_path, "-xf", str(combined_archive), "-C", str(extract_dir)],
        check=True,
    )

    engine_path = find_installed_voicevox_engine(extract_dir)
    if not engine_path:
        raise RuntimeError(f"展開後にrun.exeが見つかりませんでした: {extract_dir}")
    return engine_path


def resolve_voicevox_engine_path(config: AppConfig) -> Path:
    if config.voicevox_engine_path:
        engine_path = Path(config.voicevox_engine_path)
        if engine_path.exists():
            return engine_path
        if not config.voicevox_auto_install:
            raise FileNotFoundError(f"VOICEVOX Engine が見つかりません: {engine_path}")

    installed = find_installed_voicevox_engine(Path(config.voicevox_install_dir))
    if installed:
        return installed

    if not config.voicevox_auto_install:
        raise FileNotFoundError("VOICEVOX Engine が見つかりません。")

    return install_voicevox_engine(config)


def start_voicevox(config: AppConfig) -> subprocess.Popen[bytes] | None:
    try:
        wait_for_voicevox(config.voicevox_url, timeout_seconds=2.0)
        print("起動済みのVOICEVOX Engineに接続しました。")
        return None
    except TimeoutError:
        pass

    engine_path = resolve_voicevox_engine_path(config)

    print("VOICEVOX Engineを起動しています。")
    process = subprocess.Popen(
        [str(engine_path), *config.voicevox_engine_args],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    wait_for_voicevox(config.voicevox_url)
    return process


def find_speaker_id(config: AppConfig) -> int:
    import requests

    response = requests.get(f"{config.voicevox_url}/speakers", timeout=10)
    response.raise_for_status()

    for speaker in response.json():
        if speaker.get("name") != config.speaker_name:
            continue
        for style in speaker.get("styles", []):
            if style.get("name") == config.speaker_style:
                return int(style["id"])

    raise RuntimeError(
        f"話者が見つかりません: {config.speaker_name} / {config.speaker_style}"
    )


def record_until_enter(config: AppConfig) -> "np.ndarray":
    import numpy as np
    import sounddevice as sd

    frames: list[np.ndarray] = []

    def callback(indata: np.ndarray, frame_count: int, time_info: Any, status: sd.CallbackFlags) -> None:
        if status:
            print(f"録音ステータス: {status}", file=sys.stderr)
        frames.append(indata.copy())

    input("Enterで録音開始します。終了もEnterです。")
    print("録音中...")
    with sd.InputStream(
        samplerate=config.sample_rate,
        channels=1,
        dtype="float32",
        device=config.input_device,
        callback=callback,
    ):
        input()

    print("録音を停止しました。")
    if not frames:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(frames, axis=0).reshape(-1)
    return audio.astype(np.float32, copy=False)


def transcribe(pipe: Any, audio: "np.ndarray", config: AppConfig) -> str:
    kwargs: dict[str, Any] = {"max_new_tokens": config.max_new_tokens}
    if config.language:
        kwargs["language"] = config.language

    result = pipe.generate(audio.tolist(), **kwargs)
    text = getattr(result, "text", str(result)).strip()
    return text


def synthesize_voicevox(config: AppConfig, speaker_id: int, text: str) -> bytes:
    import requests

    query_response = requests.post(
        f"{config.voicevox_url}/audio_query",
        params={"speaker": speaker_id, "text": text},
        timeout=30,
    )
    query_response.raise_for_status()

    synthesis_response = requests.post(
        f"{config.voicevox_url}/synthesis",
        params={"speaker": speaker_id},
        json=query_response.json(),
        timeout=60,
    )
    synthesis_response.raise_for_status()
    return synthesis_response.content


def play_wav_bytes(wav_bytes: bytes, config: AppConfig) -> None:
    import io

    import sounddevice as sd
    import soundfile as sf

    data, samplerate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    sd.play(data, samplerate=samplerate, device=config.output_device)
    sd.wait()


def run_loop(config: AppConfig) -> None:
    import openvino_genai as ov_genai

    voicevox_process = start_voicevox(config)
    try:
        speaker_id = find_speaker_id(config)
        model_dir = ensure_openvino_model(config)

        print(f"WhisperPipelineを読み込みます: {model_dir} ({config.openvino_device})")
        pipe = ov_genai.WhisperPipeline(str(model_dir), config.openvino_device)

        print("準備完了です。Ctrl+Cで終了します。")
        while True:
            audio = record_until_enter(config)
            if audio.size == 0:
                print("音声が録音されませんでした。")
                continue

            text = transcribe(pipe, audio, config)
            if not text:
                print("文字起こし結果が空でした。")
                continue

            print(f"文字起こし: {text}")
            wav_bytes = synthesize_voicevox(config, speaker_id, text)
            play_wav_bytes(wav_bytes, config)
    finally:
        if voicevox_process and voicevox_process.poll() is None:
            voicevox_process.terminate()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="マイク入力をWhisper(OpenVINO)で文字起こしし、VOICEVOXのずんだもんで復唱します。"
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()

    try:
        run_loop(load_config(args.config))
    except KeyboardInterrupt:
        print("\n終了します。")
        return 0
    except Exception as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
