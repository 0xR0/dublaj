import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []

c.append(nbf.v4.new_markdown_cell(
    "# MP3 AI Türkçe Dublaj — Colab\n"
    "**Runtime > Change runtime type > GPU** seçin."))

c.append(nbf.v4.new_markdown_cell(
    "## 0. (Opsiyonel) Colab Secrets\n"
    "Repo public; clone için token GEREKMEZ. İstersen 🔑 (anahtar) simgesinden "
    "secret ekleyebilirsin (Notebook access açık olsun):\n"
    "- `HF_TOKEN` → (opsiyonel) pyannote için. Eklemezsen speechbrain kullanılır.\n"
    "- `GITHUB_TOKEN` → sadece 8. hücrede (sonucu repoya geri push) gerekir."))

c.append(nbf.v4.new_markdown_cell("## 1. Repo (private) + sistem bağımlılıkları"))
c.append(nbf.v4.new_code_cell(
    "from google.colab import userdata\n"
    "GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')\n"
    "!apt-get -qq install -y ffmpeg\n"
    "!pip install -q yt-dlp\n"
    "%cd /content\n"
    "!rm -rf dublaj\n"
    "!git clone https://{GITHUB_TOKEN}@github.com/0xR0/dublaj.git\n"
    "%cd dublaj"))

c.append(nbf.v4.new_markdown_cell("## 2. Python bağımlılıkları (birkaç dakika)"))
c.append(nbf.v4.new_code_cell("!pip install -q -r requirements.txt"))

c.append(nbf.v4.new_markdown_cell(
    "## 3. Ortam değişkenleri\n"
    "HF_TOKEN secret'ı varsa pyannote, yoksa speechbrain kullanılır."))
c.append(nbf.v4.new_code_cell(
    "import os\n"
    "try:\n"
    "    from google.colab import userdata\n"
    "    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN') or ''\n"
    "except Exception:\n"
    "    os.environ['HF_TOKEN'] = ''\n"
    "os.environ['COQUI_TOS_AGREED'] = '1'\n"
    "print('HF_TOKEN ayarli mi:', bool(os.environ['HF_TOKEN']))"))

c.append(nbf.v4.new_markdown_cell(
    "## 4-A. Girdi: YouTube linkinden ses indir"))
c.append(nbf.v4.new_code_cell(
    'YT_URL = "https://youtu.be/2hsSEWguOQE"  # <-- linki degistir\n'
    '!yt-dlp -x --audio-format mp3 --audio-quality 0 '
    '-o "input/yt_audio.%(ext)s" "$YT_URL"\n'
    'INPUT = "input/yt_audio.mp3"\n'
    'print("Girdi:", INPUT)'))

c.append(nbf.v4.new_markdown_cell(
    "## 4-B. (Alternatif) Kendi dosyanı yükle"))
c.append(nbf.v4.new_code_cell(
    "from google.colab import files\n"
    "import shutil\n"
    "up = files.upload()\n"
    "name = list(up.keys())[0]\n"
    "shutil.move(name, f'input/{name}')\n"
    "INPUT = f'input/{name}'\n"
    "print('Girdi:', INPUT)"))

c.append(nbf.v4.new_markdown_cell(
    "## 5. Dublajı çalıştır\n"
    "İlk sefer modeller iner (XTTS ~1.8GB). Bayraklar: `--no-background`, "
    "`--tts edge`, `--diarizer speechbrain|pyannote`."))
c.append(nbf.v4.new_code_cell('!python dub.py "$INPUT"'))

c.append(nbf.v4.new_markdown_cell("## 6. Sonucu incele"))
c.append(nbf.v4.new_code_cell(
    "!ls -la output/\n"
    "print('--- run.log (son 40 satir) ---')\n"
    "!tail -n 40 logs/run.log"))

c.append(nbf.v4.new_markdown_cell("## 7. Sonucu indir"))
c.append(nbf.v4.new_code_cell(
    "from google.colab import files\n"
    "files.download('output/output_dubbed.mp3')"))

c.append(nbf.v4.new_markdown_cell(
    "## 8. (Opsiyonel) Sonucu GitHub'a geri gönder\n"
    "Termux tarafında commitler + log + çıktı incelenebilsin diye. "
    "Bunun için `GITHUB_TOKEN` secret'i (repo write yetkili) gerekir."))
c.append(nbf.v4.new_code_cell(
    'from google.colab import userdata\n'
    'GITHUB_TOKEN = userdata.get("GITHUB_TOKEN")\n'
    '!git config user.email "colab@dublaj"\n'
    '!git config user.name "colab"\n'
    '!git add -f output/output_dubbed.mp3 logs/run.log\n'
    '!git commit -m "colab: dublaj sonucu + loglar" || echo "degisiklik yok"\n'
    '!git push https://{GITHUB_TOKEN}@github.com/0xR0/dublaj.git HEAD:colab-results'))

nb.cells = c
with open("notebook/dublaj_colab.ipynb", "w") as f:
    nbf.write(nb, f)
print("notebook yazildi:", len(c), "hucre")
