# Pengenalan Vowel Bahasa Indonesia dengan DTW dan Model Gaussian

Praktikum **IF4071 Pemrosesan Ucapan** untuk mengenali lima vowel bahasa Indonesia **A, I, U, E, O** dari rekaman audio. Dua pendekatan sistem pengenal ucapan (ASR) diimplementasikan dan dibandingkan:

1. **DTW (Dynamic Time Warping)** dengan *averaged template* per vowel.
2. **Model Gaussian** dengan mean dan full covariance per vowel, diskor menggunakan *negative log-likelihood* (NLL).

**Kelompok 7**

| Nama | NIM |
|---|---|
| Muhammad Alfansya | 13523005 |
| Shannon Aurellius Anastasya Lie | 13523019 |
| Dzaky Aurelia Fawwaz | 13523065 |
| Ferdin Arsenarendra Purtadi | 13523117 |
| Naomi Risaka Sitorus | 13523122 |

## Daftar Isi

- [Ringkasan Metode](#ringkasan-metode)
- [Dataset](#dataset)
- [Struktur Direktori](#struktur-direktori)
- [Instalasi](#instalasi)
- [Menjalankan Eksperimen](#menjalankan-eksperimen)
- [Evaluasi](#evaluasi)
- [Output](#output)
- [Notebook](#notebook)

## Ringkasan Metode

### 1. Preprocessing Audio

Setiap file rekaman audio diproses melalui modul `src/preprocessing.py`:

- Konversi stereo ke **mono**;
- Resampling sinyal ke **16 kHz** menggunakan `resample_poly`;
- Penghapusan **DC offset**;
- Pemotongan silence di awal dan akhir (*trimming*) menggunakan `librosa.effects.trim(top_db=25)`;
- Normalisasi puncak (*peak normalization*).

### 2. Ekstraksi Fitur MFCC 39 Dimensi

Ekstraksi fitur dilakukan menggunakan `python_speech_features` melalui modul `src/features.py`:

- **13 koefisien MFCC** (termasuk energi frame, `appendEnergy=True`);
- **13 fitur delta** (turunan pertama);
- **13 fitur delta-delta** (turunan kedua);

sehingga setiap frame menghasilkan vektor fitur **39 dimensi**. Fitur dinormalisasi per rekaman (*zero mean, unit variance*). Konfigurasi: window length 25 ms, window step 10 ms, `nfft=512`, 26 filter Mel, pre-emphasis 0.97, dan cepstral lifter 22.

### 3. Sistem 1: DTW (Averaged Template)

- Setiap rekaman template diinterpolasi linear ke **100 frame**.
- **Averaged template** dibentuk dengan merata-ratakan frame dari kelima speaker pada data latih untuk masing-masing vowel.
- Biaya lokal dihitung menggunakan Euclidean distance antar-frame dengan langkah transisi `(1,0)`, `(0,1)`, dan `(1,1)`.
- Total cost dinormalisasi terhadap panjang warping path, dan vowel dengan **biaya terkecil** dipilih sebagai hasil prediksi.

### 4. Sistem 2: Model Gaussian

- Seluruh frame fitur dari data template per vowel digabungkan untuk mengestimasi distribusi Gaussian multivariat.
- Setiap kelas vowel dimodelkan oleh **vektor mean 39 dimensi** dan **matriks kovarians 39 × 39** (diberi regularisasi diagonal `1e-3` untuk stabilitas numerik).
- Skor input dihitung dari rata-rata **Negative Log-Likelihood (NLL)** per frame.
- Vowel dengan **NLL terkecil** dipilih sebagai hasil prediksi.

### 5. Aturan Keputusan

Kedua sistem menggunakan prinsip jarak/ketidakmiripan minimum (*minimum cost/NLL*). Sistem juga menghitung nilai **margin** (selisih skor terbaik kedua dengan skor terbaik) sebagai metrik keyakinan prediksi.

## Dataset

Korpus data terdiri atas 5 speaker (**S01–S05**) dan 5 vowel (**A, I, U, E, O**).

| Skenario | Lokasi | Jumlah File | Keterangan |
|---|---|---:|---|
| Template (latih) | `data/template/` | 25 | 5 speaker × 5 vowel, kondisi `CR_N` |
| Closed recording | `data/test/closed/recording/` | 25 | Rekaman yang sama dengan template |
| Closed speaker | `data/test/closed/speaker/` | 125 | Speaker template dengan variasi kondisi rekaman |
| Open speaker | `data/test/open/` | - | Rekaman speaker dari kelompok lain |

Kondisi rekaman:

| Kode | Arti |
|---|---|
| `N` | normal |
| `HP` | *high pitch* |
| `LP` | *low pitch* |
| `SD` | durasi pendek (*short duration*) |
| `LD` | durasi panjang (*long duration*) |

Konvensi nama file:

```text
S{speaker}_{vowel}_{usage}_{condition}.wav
```

Contoh: `S01_A_CS_HP.wav`.

- `usage`: `CR` (*closed recording*), `CS` (*closed speaker*), `OS` (*open speaker*).
- Format audio: WAV, 16 kHz, mono, durasi sekitar 2–3 detik per file.

Folder `audio/` menyimpan rekaman mentah seluruh speaker. Folder `data/` berisi dataset yang sudah ditata dan dinamai sesuai skenario pengujian.

## Struktur Repository

```text
.
├── audio/                      # Rekaman mentah seluruh speaker
├── data/                       # Dataset terstruktur per skenario
│   ├── template/               # Data latih (5 speaker × 5 vowel, CR_N)
│   └── test/
│       ├── closed/
│       │   ├── recording/      # Closed recording test set
│       │   └── speaker/        # Closed speaker test set
│       └── open/               # Open speaker test set
├── src/                        # Implementasi modul Python
│   ├── data_loader.py          # Pembacaan metadata dan pemindaian dataset
│   ├── preprocessing.py        # Pembersihan dan normalisasi audio
│   ├── features.py             # Ekstraksi MFCC 39 dimensi
│   ├── template.py             # Pembentukan averaged template
│   ├── dtw.py                  # Algoritma Dynamic Time Warping
│   ├── gaussian.py             # Pembentukan Model Gaussian dan evaluasi NLL
│   ├── recognize.py            # Logika pengenalan dan penghitungan skor
│   ├── run_experiments.py      # Eksekusi pipeline pengujian eksperimen
│   └── evaluate.py             # Perhitungan akurasi dan confusion matrix
├── main.ipynb                  # Notebook demonstrasi dan eksplorasi eksperimen
├── requirements.txt            # Dependensi proyek
└── README.md
```

## Instalasi

Gunakan Python 3.10 atau versi yang lebih baru.

```bash
# Buat dan aktifkan virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

# Install dependensi
pip install -r requirements.txt
```

Atau instal langsung melalui pip:

```bash
pip install numpy pandas scipy soundfile python_speech_features librosa matplotlib seaborn scikit-learn
```

## Menjalankan Eksperimen

### 1. Menjalankan Seluruh Pengujian

Untuk menjalankan seluruh pipeline pengenalan vowel:

```bash
python -m src.run_experiments
```

Perintah di atas akan secara otomatis:

1. Memindai dan memvalidasi integritas dataset di folder `data/`;
2. Mengekstrak fitur MFCC 39 dimensi untuk setiap file audio;
3. Membangun averaged template DTW dan Model Gaussian dari `data/template`;
4. Menguji seluruh data pada skenario `closed_recording`, `closed_speaker`, dan `open` (jika folder `data/test/open` berisi data);
5. Menyimpan rincian prediksi per file ke `results/predictions.csv` serta konfigurasi run ke `results/run_info.json`.

Jika folder `data/test/open` kosong, skenario open akan dilewati secara otomatis tanpa menghentikan program.

### 2. Memilih Skenario dan Sistem Tertentu

Anda dapat membatasi sistem atau skenario yang diuji menggunakan argumen:

```bash
python -m src.run_experiments --scenarios closed_speaker --systems dtw gaussian
```

- Pilihan skenario: `closed_recording`, `closed_speaker`, `open`, `loso` (*leave-one-speaker-out*).
- Pilihan sistem: `dtw`, `gaussian`, `dtw_nn`.

### 3. Validasi Dataset

Fungsi `check_dataset()` pada runner memastikan pengujian berjalan adil:

- Penamaan file sesuai konvensi dan label folder cocok dengan isi file;
- Tidak ada file duplikat dengan label berbeda;
- Rekaman *closed recording* identik dengan template latih;
- Rekaman *closed speaker* tidak memakai rekaman template;
- Rekaman *open speaker* bukan berasal dari speaker template;
- Tidak ada rekaman yang belum dipangkas (durasi > 10 detik).

## Evaluasi

Setelah hasil prediksi tersedia di `results/predictions.csv`, jalankan modul evaluasi:

```bash
python -m src.evaluate
```

Program akan menghitung dan menampilkan tabel hasil evaluasi di terminal serta menyimpannya ke folder `results/evaluation/`:

- **Ringkasan Akurasi Keseluruhan (%)**: Akurasi tiap sistem pada skenario closed recording, closed speakers, open speakers, dan nilai rata-ratanya.
- **Tabel 1: Akurasi per Vowel (%)**: Akurasi setiap vowel (**A, I, U, E, O**) untuk masing-masing sistem dan skenario, dilengkapi baris `Average Accuracy`.
- **Tabel 2: Closed Experiments per Speaker (%)**: Akurasi per speaker (**S01–S05**) pada closed recording dan closed speakers untuk sistem DTW dan Gaussian, dilengkapi kolom dan baris `Average`.
- **Tabel 3: Open Experiments per Speaker (%)**: Akurasi per speaker untuk pengujian open speaker untuk `DTW Open` dan `Gaussian Open`, dilengkapi kolom dan baris `Average`.
- **Confusion Matrix**: Matriks konfusi aktual vs prediksi untuk setiap kombinasi sistem dan skenario (disimpan dalam bentuk mentah dan ternormalisasi persen).

Rumus perhitungan akurasi:

```text
Accuracy = (jumlah prediksi benar / jumlah seluruh data uji) × 100%
```

Contoh menjalankan evaluasi dengan path kustom:

```bash
python -m src.evaluate --predictions results/predictions.csv --out results/evaluation
```

## Output

### `results/predictions.csv`

File hasil prediksi rekaman uji per baris (file uji × sistem):

| Kolom | Isi |
|---|---|
| `file`, `speaker`, `condition` | Identitas rekaman uji |
| `scenario` | `closed_recording`, `closed_speaker`, `open`, atau `loso` |
| `system` | `dtw`, `gaussian`, atau `dtw_nn` |
| `vowel_true`, `pred`, `correct` | Label sebenarnya, prediksi sistem, dan status kebenaran (1/0) |
| `score_A`–`score_O` | Skor ketidakmiripan terhadap setiap vowel |
| `margin` | Selisih skor terbaik kedua dan terbaik (tingkat keyakinan) |
| `n_frames`, `time_ms` | Jumlah frame MFCC dan waktu inferensi dalam milidetik |

### `results/evaluation/`

File keluaran tabel dalam format CSV untuk analisis dan penyusunan laporan:

| File | Isi |
|---|---|
| `table_overall.csv` | Ringkasan: akurasi per sistem × skenario beserta rata-ratanya |
| `table1_per_vowel.csv` | Tabel 1: akurasi per vowel (format matrix) |
| `table1_per_vowel_detail.csv` | Tabel 1 versi long: rincian jumlah data dan akurasi per vowel |
| `table2_closed_speakers.csv` | Tabel 2: closed experiments per speaker (dengan baris dan kolom Average) |
| `table3_open_speakers.csv` | Tabel 3: open experiments per speaker (jika data open tersedia) |
| `table_detail_per_speaker.csv` | Detail akurasi per speaker × skenario × sistem |
| `table_detail_per_speaker_pivot.csv` | Matriks pivot detail per speaker |
| `confusion_matrices/*.csv` | Confusion matrix mentah dan ternormalisasi |

## Notebook

File `main.ipynb` memuat demonstrasi alur eksperimen yang terstruktur dari awal hingga akhir:

1. Instalasi dan konfigurasi lingkungan;
2. Pemuatan dataset dan validasi sinyal audio;
3. Preprocessing sinyal audio dan visualisasi waveform;
4. Ekstraksi fitur MFCC 39 dimensi dan visualisasi spektrogram/koefisien;
5. Pembentukan dan inspeksi averaged template DTW;
6. Implementasi dan evaluasi sistem DTW;
7. Implementasi dan evaluasi Model Gaussian;
8. Pengujian sistem pada ketiga skenario pengujian;
9. Perhitungan Ringkasan Akurasi, Tabel 1, Tabel 2, dan Tabel 3;
10. Visualisasi Confusion Matrix dengan heatmap interaktif;
11. Penyimpanan dan export seluruh tabel hasil evaluasi ke CSV.

Jalankan notebook dari direktori root repository agar path `data/` dan `src/` terbaca dengan benar:

```bash
jupyter notebook main.ipynb
```