import pandas as pd
import librosa
import soundfile as sf
import json
import random
from pathlib import Path

SAMPLE_RATE = 32000
SEGMENT_SEC = 20
MAX_PER_GENRE = 300  # 300 × 8 жанров = 2400 сегментов

DATASET_DIR = Path('dataset')
FMA_DIR = DATASET_DIR / 'fma_small'
META_DIR = DATASET_DIR / 'fma_metadata'
OUT_DIR = DATASET_DIR / 'processed'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Богатые описания по жанрам — случайно выбираются для каждого трека
GENRE_CAPTIONS = {
    'Hip-Hop': [
        'hip-hop beat, heavy kick drum, bass groove',
        'rap instrumental, boom bap drums, vinyl samples',
        'hip-hop, trap beat, 808 bass, dark atmosphere',
        'hip-hop instrumental, lo-fi samples, jazzy chords',
        'urban hip-hop, punchy drums, melodic hook',
        'hip-hop groove, syncopated rhythm, soulful samples',
    ],
    'Electronic': [
        'electronic music, synthesizer leads, driving beat',
        'electronic dance, pulsing bass, arpeggiated synth',
        'ambient electronic, atmospheric pads, slow tempo',
        'electronic, four-on-the-floor kick, energetic build',
        'IDM electronic, glitchy percussion, complex rhythm',
        'electronic groove, warm synth bass, melodic synth',
    ],
    'Rock': [
        'rock music, electric guitar, powerful drums',
        'indie rock, distorted guitar, energetic rhythm',
        'alternative rock, melodic guitar riff, strong beat',
        'rock instrumental, crunchy guitars, driving rhythm',
        'hard rock, heavy riff, aggressive drumming',
        'rock, clean guitar arpeggios, dynamic build',
    ],
    'Folk': [
        'folk music, acoustic guitar, warm melody',
        'folk song, fingerpicked guitar, gentle rhythm',
        'acoustic folk, strummed chords, organic feel',
        'folk instrumental, light percussion, natural sound',
        'indie folk, layered acoustics, soft atmosphere',
        'folk, banjo and guitar, upbeat rustic feel',
    ],
    'Pop': [
        'pop music, catchy melody, upbeat groove',
        'pop song, bright synths, danceable rhythm',
        'pop instrumental, polished production, hook-driven',
        'pop, piano melody, light percussion, modern sound',
        'indie pop, melodic progression, warm production',
        'pop groove, synth bass, bright atmosphere',
    ],
    'Experimental': [
        'experimental music, abstract textures, unconventional',
        'avant-garde, layered noise, evolving soundscape',
        'experimental electronic, drone and rhythm, complex',
        'noise music, textural layers, atmospheric',
        'experimental ambient, shifting tones, minimalist',
        'avant-garde composition, unusual timbre, exploratory',
    ],
    'International': [
        'world music, ethnic instruments, cultural groove',
        'international folk, traditional melody, organic rhythm',
        'world beat, percussion-driven, vibrant energy',
        'ethnic music, exotic scale, rhythmic pattern',
        'global music, fusion of styles, melodic chant',
        'world instrumental, traditional instruments, lively',
    ],
    'Instrumental': [
        'instrumental music, melodic composition, no vocals',
        'instrumental, orchestral arrangement, cinematic feel',
        'instrumental groove, jazz-influenced, smooth melody',
        'background instrumental, calm atmosphere, gentle melody',
        'instrumental, piano and strings, emotional progression',
        'instrumental composition, layered arrangement, dynamic',
    ],
}


def load_tracks() -> pd.DataFrame:
    tracks = pd.read_csv(META_DIR / 'tracks.csv', index_col=0, header=[0, 1])
    genre_col = ('track', 'genre_top')
    selected_parts = []
    for genre in GENRE_CAPTIONS:
        genre_tracks = tracks[tracks[genre_col] == genre].head(MAX_PER_GENRE * 2)
        selected_parts.append(genre_tracks)
        print(f"  {genre}: {len(genre_tracks)} треков в метаданных")
    return pd.concat(selected_parts)


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
    if len(audio) < seg_len:
        return []
    seg = audio[:seg_len]
    seg = seg / (max(abs(seg)) + 1e-8) * 0.9
    out = str(OUT_DIR / f"{track_id}.wav")
    sf.write(out, seg, SAMPLE_RATE)
    return [out]


def main():
    print("Загружаем метаданные...")
    tracks = load_tracks()

    all_segments: list[tuple[str, str]] = []
    genre_counts: dict[str, int] = {}

    for tid in tracks.index:
        genre = tracks.loc[tid, ('track', 'genre_top')]
        if genre_counts.get(genre, 0) >= MAX_PER_GENRE:
            continue

        paths = process_track(tid)
        if not paths:
            continue

        caption = random.choice(GENRE_CAPTIONS.get(genre, [f"{genre.lower()} music"]))
        all_segments.append((Path(paths[0]).name, caption))
        genre_counts[genre] = genre_counts.get(genre, 0) + 1

        if len(all_segments) % 200 == 0:
            print(f"Сегментов: {len(all_segments)}")

    print(f"\nИтого сегментов: {len(all_segments)}")
    print("По жанрам:", genre_counts)

    meta_path = DATASET_DIR / 'metadata.jsonl'
    with open(meta_path, 'w', encoding='utf-8') as f:
        for filename, caption in all_segments:
            f.write(json.dumps({"audio": filename, "caption": caption}) + '\n')

    print(f"Метаданные сохранены: {meta_path}")


if __name__ == '__main__':
    main()
