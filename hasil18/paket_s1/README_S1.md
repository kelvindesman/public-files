# Paket S1 — data hipotetik berlabel 10.000 baris + model ANN-LM program 1

Dibuat oleh notebook Kaggle `kelvindsmn/paket-s1-data-hipotetik-10000`.

## Isi

| File | Isi |
|---|---|
| `data_hipotetik_berlabel_10000.csv` | 10.000 baris, 4 sensor, nilai dengan fault + nilai asli + label per sensor |
| `kamus_kolom.csv` | arti kolom |
| `model_ann_lm_4sensor.npz` | model ANN-LM, satu sensor per panggilan (dipakai di contoh) |
| `model_ann_lm_rata2.npz` | model untuk rata-rata 4 sensor |
| `muat_model.py` | `muat_model(path)`, `deteksi(model, deret)`; cukup numpy + pandas |
| `contoh_backend.py` | kelas `DetektorSensor`: satu pembacaan masuk, JSON keluar |
| `prediksi_10000.csv`, `skor_10000.csv` | keluaran model pada data 10.000 baris dan skornya |
| `model_card.json` | ringkasan model, preprocessing, skor resmi, batasan |

## Data

Sumber: 10.000 baris pertama `tabel_sensor4_generated.csv` (interval 30 detik).
Fault disuntikkan dengan simulator notebook 15 (drift 5 poin per episode, bias +5,
spike x1,30 pada 1,52 % titik, hardware 8 % macet + 5 % hilang), 3 episode x 0,5 hari.
Tiap sensor diberi skenario berbeda: kelembaban1 = S2 drift, kelembaban2 = S3 spike+bias, kelembaban3 = S4 drift+bias+malfunc, kelembaban4 = S5 spike+bias+malfunc+drift.
Label berasal dari fault injection (bukan gangguan nyata di lapangan).

## Preprocessing (wajib sama di backend)

1. Ambil deret SATU sensor, urut waktu. Nilai hilang = NaN.
2. Isi NaN dengan nilai valid sebelumnya (forward fill).
3. Fitur sampel t: `lag1..lag10` = nilai t-1..t-10, `diff` = x(t) - x(t-1),
   `isna` = 1 bila x(t) hilang, `isna_10` = jumlah data hilang di t-10..t-1.
4. Normalisasi: `z = (fitur - mean) / scale` dengan mean/scale DARI FILE MODEL:

| fitur | mean | scale |
|---|---|---|
| `lag1` | 50.1845 | 3.9444 |
| `lag2` | 50.1828 | 3.9410 |
| `lag3` | 50.1817 | 3.9293 |
| `lag4` | 50.1834 | 3.9401 |
| `lag5` | 50.1871 | 3.9523 |
| `lag6` | 50.1831 | 3.9462 |
| `lag7` | 50.1854 | 3.9502 |
| `lag8` | 50.1876 | 3.9533 |
| `lag9` | 50.1831 | 3.9450 |
| `lag10` | 50.1889 | 3.9633 |
| `diff` | -0.0016 | 1.2722 |
| `isna` | 0.0058 | 0.0762 |
| `isna_10` | 0.0564 | 0.2789 |

5. Forward pass: 13 -> 12 tanh -> 6 tanh -> 2 -> softmax. `prob_fault` = keluaran kelas 1.
6. `alarm = prob_fault > 0.38` (model 4sensor).

## Cara pakai

```python
from muat_model import muat_model, deteksi
m = muat_model("model_ann_lm_4sensor.npz")
prob, alarm = deteksi(m, deret_satu_sensor)        # array per sampel

from contoh_backend import DetektorSensor           # versi streaming untuk backend
det = DetektorSensor("model_ann_lm_4sensor.npz")
det.proses("kelembaban1", 51.27)
# {"sensor": "kelembaban1", "nilai": 51.27, "prob_fault": ..., "alarm": false, "siap": true}
```

`python contoh_backend.py data_hipotetik_berlabel_10000.csv` memutar seluruh data
dan mencetak 5 alarm pertama + jumlah alarm per sensor.

## Skor pada data 10.000 baris (baris 10 ke atas)

| sensor | kondisi | precision | recall | F1 | alarm_palsu_% (baris normal) |
|---|---|---|---|---|---|
| kelembaban1 | drift | 0.8849 | 0.2137 | 0.3442 | 2.12 |
| kelembaban2 | spike+bias | 0.9614 | 0.334 | 0.4958 | 1.02 |
| kelembaban3 | drift+bias+malfunc | 0.993 | 0.6609 | 0.7936 | 0.35 |
| kelembaban4 | spike+bias+malfunc+drift | 0.8656 | 0.9583 | 0.9096 | 11.34 |

Skor resmi model (data uji notebook 15): F1 0.6404,
precision 0.6621, recall 0.6201.

## Batasan

- Drift kecil sulit dideteksi per sampel (recall drift rendah).
- Episode di data ini 0,5 hari, model dilatih pada episode 3 hari.
- Model tidak mengidentifikasi sensor; panggil per sensor.
- Data harus interval kira-kira 30 detik, nilai persen 0-100.
