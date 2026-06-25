# MP3 AI Türkçe Dublaj

MP3/WAV sesini konuşmacılara ayırıp, cinsiyet tespiti yapıp, her karaktere
orijinalinden klonlanmış (XTTS) sesle Türkçe dublaj üreten otomatik sistem.

## Yerel kullanım
```bash
pip install -r requirements.txt
python dub.py input.mp3
# çıktı: output/output_dubbed.mp3
```

## Bayraklar
- `--output PATH` çıktı yolu
- `--no-background` arka plan ayırmayı atla (daha hızlı)
- `--tts edge` XTTS yerine Edge TTS yedeği
- `--diarizer pyannote|speechbrain` diarization motorunu zorla

## Ortam değişkenleri
- `HF_TOKEN` (opsiyonel): pyannote için ücretsiz HF token. Yoksa speechbrain kullanılır.
- `COQUI_TOS_AGREED=1`: XTTS lisans kabulü (ticari olmayan kullanım).

## Colab
`notebook/dublaj_colab.ipynb` dosyasını Colab'da açın, GPU seçin, hücreleri çalıştırın.

## Testler
```bash
pip install pytest && pytest -v
```
