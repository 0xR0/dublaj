# MP3 Tabanlı Otomatik AI Türkçe Dublaj Sistemi — Tasarım Dokümanı

**Tarih:** 2026-06-25
**Durum:** Onaylandı (uygulamaya hazır)
**Hedef ortam:** Google Colab (GPU), tamamen ücretsiz araçlarla

## 1. Amaç

MP3/WAV ses dosyasını girdi alıp, konuşmacıları ayıran, cinsiyet tespiti yapan,
her karaktere tutarlı bir ses atayan, konuşmayı Türkçeye çeviren ve doğal Türkçe
dublaj üreten tam otomatik bir sistem. Çıktı: `output_dubbed.mp3`.

**Kısıt:** Hiçbir ücretli servis/API yok. Tüm bileşenler ücretsiz ve mümkünse
tokensiz çalışmalı. pyannote için ücretsiz HF token opsiyonel; yoksa otomatik
tokensiz yedeğe düşülür.

## 2. Komut Arayüzü

```
python dub.py input.mp3
python dub.py input.mp3 --output output/output_dubbed.mp3
python dub.py input.mp3 --no-background        # arka plan ayırmayı atla
python dub.py input.mp3 --diarizer speechbrain # diarization motorunu zorla
python dub.py input.mp3 --tts edge             # XTTS yerine Edge TTS yedeğini zorla
```

Colab notebook (`notebook/dublaj_colab.ipynb`) aynı pipeline'ı hücre hücre çalıştırır.

## 3. İşlem Hattı (Pipeline)

```
input.mp3
  → 1. preprocess   (FFmpeg: 16kHz mono WAV, loudness normalize)
  → 2. separate     (Demucs: vokal stem | arka plan stem)
  → 3. diarize      (pyannote VARSA HF_TOKEN, YOKSA speechbrain — otomatik)
  → 4. gender       (librosa F0/pitch → her konuşmacıya male/female)
  → 5. transcribe   (faster-whisper: zaman damgalı transkript, diarize ile hizalı)
  → 6. translate    (NLLB-200 → Türkçe; segment ve zaman damgaları korunur)
  → 7. synthesize   (XTTS: her konuşmacının orijinal sesinden klonlanmış TR ses, süreye sığdırma)
  → 8. reconstruct  (TR konuşma + arka plan stem mix, volume normalize)
  → output_dubbed.mp3
```

## 4. Modüller (dubber/ paketi)

Her modül tek sorumluluk taşır, net girdi/çıktı ile bağımsız test edilebilir.

| Modül | Girdi | Çıktı | Bağımlılık |
|-------|-------|-------|-----------|
| `preprocess.py` | input.mp3/wav | temp/audio_16k.wav | ffmpeg |
| `separate.py` | 16k wav | temp/vocals.wav, temp/background.wav | demucs, torch |
| `diarize.py` | vocals.wav | segment listesi [{start, end, speaker_id}] | pyannote / speechbrain |
| `gender.py` | vocals.wav + segmentler | {speaker_id: "male"/"female"} | librosa, numpy |
| `transcribe.py` | vocals.wav + segmentler | [{start, end, speaker_id, text}] | faster-whisper |
| `translate.py` | transkript segmentleri | aynı yapı, text→Türkçe | transformers (NLLB) |
| `synthesize.py` | TR segmentler + konuşmacı referans klipleri | temp/tts/*.wav | TTS (XTTS-v2) / edge-tts |
| `reconstruct.py` | TTS klipleri + background.wav | output_dubbed.mp3 | pydub/ffmpeg, numpy |

Ortak veri modeli (segment): `{start: float, end: float, speaker_id: str, gender: str, text: str, text_tr: str}`.

## 5. Önemli Tasarım Kararları

### 5.1 Diarization — çift motor (pluggable)
- `HF_TOKEN` ortam değişkeni varsa → **pyannote/speaker-diarization-3.1** (en iyi kalite).
- Token yoksa veya pyannote yüklenemezse → **speechbrain** konuşmacı vektörleri
  (ECAPA-TDNN) + agglomerative clustering. Token gerektirmez, %100 ücretsiz.
- Seçim otomatik; `--diarizer` bayrağıyla elle zorlanabilir.

### 5.2 Cinsiyet tespiti — hafif pitch heuristiği
- Her konuşmacının segmentlerinden ortalama temel frekans (F0) librosa `pyin` ile
  hesaplanır. Eşik ~165 Hz: altı → male, üstü → female. Ağır model yok, GPU şart değil.

### 5.3 Ses atama — XTTS ses klonlama (her karaktere farklı ses)
- Birincil motor: **XTTS-v2** (Coqui TTS), çapraz-dilli ses klonlama.
- Her `speaker_id` için, ayrılmış `vocals.wav` içinden o konuşmacının **en uzun/temiz
  segmenti** (mümkünse ≥6 sn) referans klip olarak çıkarılır (`temp/refs/<speaker_id>.wav`).
- XTTS bu referansla, çevrilmiş Türkçe metni o konuşmacının **kendi tınısında** üretir.
  Sonuç: her karakter hem birbirinden **farklı** hem de orijinaline benzer ses alır,
  tüm dosya boyunca **tutarlı** (speaker_id → referans sabit).
- Cinsiyet tespiti (5.2) bilgi/log ve Edge yedeği için tutulur; XTTS klonlama zaten
  cinsiyeti doğal olarak korur.
- **Otomatik yedek (Edge TTS):** XTTS yüklenemez/çökerse veya `--tts edge` verilirse,
  cinsiyete göre Ahmet/Emel + konuşmacıya özgü `pitch`/`rate` profili kullanılır.
- **Lisans notu:** XTTS-v2 modeli Coqui Public Model License (CPML) — **ücretsiz ama
  ticari olmayan kullanım**. Para/token gerekmez; çalışma anında `COQUI_TOS_AGREED=1`
  ortam değişkeniyle lisans kabul edilir.

### 5.4 Zamanlama / senkron
- TR çeviri orijinalden uzun/kısa olabilir. Üretilen TTS klibi orijinal segment
  süresiyle karşılaştırılır. XTTS'te doğrudan hız parametresi olmadığından, taşma varsa
  klip **ffmpeg `atempo`** ile (tını koruyarak, makul sınırda ~1.3x'e kadar) hızlandırılır.
  Aşırı kısa ise sonuna sessizlik (padding) eklenir. (Edge yedeğinde `rate` parametresi.)
- Segment başlangıç zaman damgaları korunur; klipler zaman çizelgesine yerleştirilir.

### 5.5 Arka plan koruma
- Demucs vokal/arka planı ayırır. Final mix = (yerleştirilmiş TR TTS klipleri) +
  (orijinal background stem). Sonra loudness normalize. `--no-background` ile atlanır.

## 6. Proje Yapısı

```
dublaj/
├── dub.py                      # CLI giriş noktası
├── dubber/
│   ├── __init__.py
│   ├── config.py               # yollar, ses haritası, eşik sabitleri
│   ├── pipeline.py             # aşamaları sırayla bağlar
│   ├── preprocess.py
│   ├── separate.py
│   ├── diarize.py
│   ├── gender.py
│   ├── transcribe.py
│   ├── translate.py
│   ├── synthesize.py
│   ├── reconstruct.py
│   └── utils.py                # logging, zaman biçimleme, ffmpeg sarmalayıcılar
├── notebook/
│   └── dublaj_colab.ipynb
├── requirements.txt
├── README.md
├── input/                      # girdi sesleri
├── temp/                       # ara dosyalar (wav, stem, tts klipleri)
├── output/                     # output_dubbed.mp3
├── models/                     # indirilen model cache
└── logs/                       # çalışma logları
```

## 7. Performans / Colab

- faster-whisper ve Demucs GPU'da (CUDA) çalışır; CPU'ya otomatik düşer.
- Uzun ses (1–2 saat): işleme segment/chunk bazlı; ara dosyalar diske yazılıp
  bellekten boşaltılır (RAM tasarrufu). Modeller bir kez yüklenir, tekrar kullanılır.
- XTTS GPU-yoğundur; model bir kez yüklenir, tüm segmentler aynı oturumda üretilir.
  Referans klipler önceden çıkarılır. Edge yedeği seçilirse asyncio ile paralel üretim.
- XTTS-v2 modeli ilk çalıştırmada ~1.8 GB indirilir, `models/` altında cache'lenir.

## 8. Hata Yönetimi

- Girdi dosyası yoksa/biçim desteklenmiyorsa net hata.
- pyannote yüklenemezse uyarı logu + speechbrain'e otomatik geçiş.
- Bir segmentte ASR boş metin dönerse o segment atlanır (sessizlik korunur).
- TTS bir segmentte başarısız olursa o segment sessizlikle doldurulur, pipeline durmaz.
- XTTS yüklenemezse uyarı logu + Edge TTS yedeğine otomatik geçiş.
- Bir konuşmacının referans klibi çok kısaysa (<3 sn) o konuşmacı için Edge yedeği kullanılır.
- Tüm aşamalar `logs/run.log` dosyasına ilerleme yazar.

## 9. Bağımlılıklar (requirements.txt)

```
ffmpeg (sistem) 
faster-whisper
pyannote.audio        # opsiyonel, token gerektirir
speechbrain           # tokensiz diarization yedeği
demucs
TTS                   # Coqui XTTS-v2 (birincil seslendirme)
edge-tts              # yedek seslendirme
transformers          # NLLB çeviri
sentencepiece
torch
torchaudio
librosa
numpy
pydub
soundfile
```

## 10. Kapsam Dışı (YAGNI)

- Video yeniden senkron (opsiyonel; ileride FFmpeg ile ses-değiştirme eklenebilir,
  ilk sürümde yok).
- Web arayüzü / UI yok; CLI + notebook yeterli.

## 11. Başarı Kriterleri

1. `python dub.py input.mp3` çalışır ve `output/output_dubbed.mp3` üretir.
2. Çıktıda Türkçe konuşma duyulur; arka plan müziği/efekt korunur (varsa).
3. Farklı konuşmacılara XTTS ile orijinalinden klonlanmış, birbirinden farklı ve
   tutarlı Türkçe sesler atanır.
4. Erkek konuşmacı erkek sesle, kadın konuşmacı kadın sesle seslendirilir (klonlama
   cinsiyeti korur; Edge yedeğinde cinsiyete göre ses seçilir).
5. HF token olmadan da (speechbrain yedeğiyle) uçtan uca çalışır.
6. Colab GPU üzerinde uzun ses dosyalarını çökmeden işler.
```
