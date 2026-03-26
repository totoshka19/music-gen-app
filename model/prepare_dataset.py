import pandas as pd
import librosa
import soundfile as sf
import json
from pathlib import Path

TARGET_GENRE = 'Hip-Hop'
SAMPLE_RATE = 32000
SEGMENT_SEC = 20
MAX_TRACKS = 500

DATASET_DIR = Path('dataset')
FMA_DIR = DATASET_DIR / 'fma_small'
META_DIR = DATASET_DIR / 'metadata'
OUT_DIR = DATASET_DIR / 'processed'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_tracks() -> pd.DataFrame:
    tracks = pd.read_csv(META_DIR / 'tracks.csv', index_col=0, header=[0, 1])
    genre_col = ('track', 'genre_top')
    selected = tracks[tracks[genre_col] == TARGET_GENRE].head(MAX_TRACKS)
    print(f"Треков жанра '{TARGET_GENRE}': {len(selected)}")
    return selected


def load_echonest() -> tuple[pd.DataFrame | None, bool]:
    try:
        tags = pd.read_csv(META_DIR / 'echonest.csv', index_col=0, header=[0, 1])
        return tags, True
    except FileNotFoundError:
        print("echonest.csv не найден — описания будут базовыми")
        return None, False


def process_track(track_id: int) -> list[str]:
    folder = str(track_id).zfill(6)[:3]
    src = FMA_DIR / folder / f"{str(track_id).zfill(6)}.mp3"
    if not src.exists():
        return []

    try:
        audio, _ = librosa.load(str(src), sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        print(f"Ошибка загрузки {track_id}: {e}")
        return []

    seg_len = SEGMENT_SEC * SAMPLE_RATE
    saved = []
    for i, start in enumerate(range(0, len(audio) - seg_len, seg_len)):
        seg = audio[start:start + seg_len]
        seg = seg / (max(abs(seg)) + 1e-8) * 0.9
        out = str(OUT_DIR / f"{track_id}_{i:02d}.wav")
        sf.write(out, seg, SAMPLE_RATE)
        saved.append(out)
    return saved


def build_caption(track_id: int, tags: pd.DataFrame | None, has_echonest: bool) -> str:
    genre = TARGET_GENRE.lower()
    if not has_echonest or tags is None:
        return f"{genre} music"
    try:
        tempo = tags.loc[track_id, ('audio_features', 'tempo')]
        energy = tags.loc[track_id, ('audio_features', 'energy')]
        mood = 'energetic' if energy > 0.6 else 'relaxed'
        return f"{genre}, {int(tempo)} bpm, {mood}"
    except (KeyError, ValueError):
        return f"{genre} music"


def main():
    tracks = load_tracks()
    tags, has_echonest = load_echonest()

    all_segments: list[tuple[str, str]] = []  # (filename, caption)
    for tid in tracks.index:
        paths = process_track(tid)
        caption = build_caption(tid, tags, has_echonest)
        for p in paths:
            all_segments.append((Path(p).name, caption))

        if len(all_segments) % 100 == 0 and all_segments:
            print(f"Сегментов: {len(all_segments)}")

    print(f"\nИтого сегментов: {len(all_segments)}")

    meta_path = DATASET_DIR / 'metadata.jsonl'
    with open(meta_path, 'w', encoding='utf-8') as f:
        for filename, caption in all_segments:
            f.write(json.dumps({"audio": filename, "caption": caption}) + '\n')

    print(f"Метаданные сохранены: {meta_path}")
    print("\nСовет: откройте metadata.jsonl и вручную улучшите описания для лучшего качества.")
    print('Формат: "lofi hip hop, 85 bpm, mellow piano, soft kick, vinyl crackle"')


if __name__ == '__main__':
    main()
