# TRCaptionNet++ Projection Adapter Deneyleri -- Colab Akışı

Bu rehber, `proj_exp` dalındaki (GitHub: `mhr871/TRCAP`, branch `proj_exp`)
P1-P7 Projection Adapter deneylerini Colab'da baştan sona çalıştırmak için
gereken **tüm** hücreleri sırasıyla verir. `COLAB_TASVIRET_BASELINE.md` ile
aynı üslubu izler: `!` işaretleri notebook hücreleri içindir, Colab
Terminal'de çalışıyorsan kaldırabilirsin.

## Deneyde sabit tutulanlar

- Encoder: `DINOv2 ViT-L/14`, 224x224 giriş, **frozen** (`requires_grad=False`)
- Decoder: `dbmdz/electra-base-turkish-mc4-cased-discriminator`
  (`ElectraForCausalLM`, cross-attention eklenmiş), **frozen**
- Eğitilen tek bileşen: Projection Adapter (P1-P7, bkz. `models/projection_adapters/`)
- Optimizer: AdamW, yalnızca projection LR `5e-5`, betas `(0.9, 0.99)`,
  weight decay `0.01`, gradient clipping `1.0` -- optimizer'da decoder/encoder
  parametresi yoktur
- Schedule: linear warmup 500 iterasyon, toplam **16.000 iterasyon**, ardından linear decay
- Batch size: 64, seed: 42 (7 deneyde de ortak)
- Validation: her 1.000 iterasyonda bir, hedef metrik `CIDEr`
- Metrikler: BLEU-1/2/3/4, METEOR, ROUGE-L, CIDEr (+ train/val loss); SPICE yok

## 1. Runtime kontrolü

Colab runtime'da GPU olarak L4/A100 seç. Notebook hücrelerinde:

```python
!python --version
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

## 2. Repoyu klonla

`proj_exp` dalının kendisi bağımsız bir kök commit'tir (repo köküyle
`proj_exp/` klasörünün içeriği birebir aynıdır); klonlandığında dosyalar
doğrudan `/content/TRCAP_projection_exp` altına gelir, ayrıca `proj_exp/`
alt klasörüne inmen gerekmez.

```python
%cd /content
!rm -rf /content/TRCAP_projection_exp
!git clone --branch proj_exp --single-branch https://github.com/mhr871/TRCAP.git /content/TRCAP_projection_exp
%cd /content/TRCAP_projection_exp
!git rev-parse --short HEAD
```

Repo zaten varsa ve silmeden güncellemek istersen:

```python
%cd /content/TRCAP_projection_exp
!git pull --ff-only
!git rev-parse --short HEAD
```

## 3. Kütüphaneleri kur ve kontrol et

```python
!python -m pip install -r requirements.txt
```

```python
!python -c "import torch, transformers, pycocotools; print('torch=', torch.__version__); print('transformers=', transformers.__version__); print('cuda=', torch.cuda.is_available())"
```

METEOR ve PTB tokenizer Java'ya ihtiyaç duyar (pycocoevalcap kendi
`meteor-1.5.jar`'ını taşır); Colab imajında Java genelde hazır gelir, yine de
doğrula:

```python
!java -version
```

Java yoksa veya METEOR yine de başarısız olursa eğitim **durmaz**: hata
`train.log`'a yazılır ve o değerlendirmede `METEOR: null` kaydedilir, diğer
metrikler etkilenmez (bkz. `README.md` -- "METEOR davranışı").

## 4. TasvirEt görüntülerini indir

Split JSON'ları (`Data/tasvir-et/tasvir_{train,val,test}.json`) repoyla
birlikte geldi; yalnızca eşleşen Flickr8K görüntüleri (~1.1 GB) indirilmeli:

```python
!python tools/download_tasviret_images.py
```

Beklenen çıktı satırı:

```text
TasvirEt image extraction is complete: 8000 images in Data/flickr8k/images
```

## 5. Drive'ı çıktı için hazırla

Uzun eğitimde Colab oturumu kapanırsa checkpoint'lerin kaybolmaması için
Drive'ı bağla ve çıktı klasörünü oluştur:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!mkdir -p /content/drive/MyDrive/TRCAP_projection_exp
```

## 6. P1-P7'nin tamamını tek komutla çalıştır

`run_projection_experiments.py`, her adapter için ayrı ayrı ve sırayla:
model+optimizer+scheduler'ı sıfırdan kurar, `run_preflight_checks()`
kontrol listesini (Encoder Frozen / Decoder Frozen / Projection Trainable /
Optimizer Parametre Sayısı / Adapter Registry 7/7 / Dataset Yüklendi /
Checkpoint Dizinleri Oluşturuldu) yazdırır, 16.000 iterasyon eğitir, ve bir
sonraki adapter'a otomatik geçer.

```python
!PYTHONPATH=/content/TRCAP_projection_exp python -u run_projection_experiments.py \
  --save-dir /content/drive/MyDrive/TRCAP_projection_exp
```

Bu tek komut sonunda `runs/` yerine doğrudan Drive altında şu 7 klasör
oluşmuş olur:

```text
/content/drive/MyDrive/TRCAP_projection_exp/P1_linear
/content/drive/MyDrive/TRCAP_projection_exp/P2_mlp
/content/drive/MyDrive/TRCAP_projection_exp/P3_residual
/content/drive/MyDrive/TRCAP_projection_exp/P4_cross_attention
/content/drive/MyDrive/TRCAP_projection_exp/P5_gated
/content/drive/MyDrive/TRCAP_projection_exp/P6_film
/content/drive/MyDrive/TRCAP_projection_exp/P7_bottleneck
```

Her birinde: `config.yaml`, `train.log`, `adapter_summary.json`,
`epoch_metrics.csv`, `metrics.json`, `best.pth`, `last.pth`.

Yalnızca bir alt küme çalıştırmak istersen:

```python
!PYTHONPATH=/content/TRCAP_projection_exp python -u run_projection_experiments.py \
  --save-dir /content/drive/MyDrive/TRCAP_projection_exp \
  --adapters cross_attention gated
```

Colab kesilirse: bir adapter'ın `runs/<Pn>_<adapter>/last.pth`'i zaten
varsa `run_projection_experiments.py` o adapter'ı atlar (log'da
"skipping" mesajı görülür); yarım kalan adapter'ı **baştan** tamamlamak
için o klasörü sil veya `--overwrite` geç:

```python
!PYTHONPATH=/content/TRCAP_projection_exp python -u run_projection_experiments.py \
  --save-dir /content/drive/MyDrive/TRCAP_projection_exp \
  --adapters film \
  --overwrite
```

## 7. (Alternatif) Tek bir adapter'ı elle çalıştırmak/devam ettirmek

```python
!PYTHONPATH=/content/TRCAP_projection_exp python -u train.py \
  --config configs/projection_exp/P1_linear.yaml \
  --save-dir /content/drive/MyDrive/TRCAP_projection_exp
```

Kaldığı yerden devam:

```python
!PYTHONPATH=/content/TRCAP_projection_exp python -u train.py \
  --config configs/projection_exp/P1_linear.yaml \
  --save-dir /content/drive/MyDrive/TRCAP_projection_exp \
  --resume /content/drive/MyDrive/TRCAP_projection_exp/P1_linear/last.pth
```

## 8. Bağımsız checkpoint değerlendirmesi (test split)

```python
!PYTHONPATH=/content/TRCAP_projection_exp python eval.py \
  --config configs/projection_exp/P1_linear.yaml \
  --weights /content/drive/MyDrive/TRCAP_projection_exp/P1_linear/best.pth \
  --test-json Data/tasvir-et/tasvir_test.json \
  --test-data Data/flickr8k/images \
  --output-dir /content/drive/MyDrive/TRCAP_projection_exp/P1_linear/test_eval
```

## 9. Sonuçları tek tabloda topla

7 deney tamamlandıktan sonra `metrics.json`'ların `best` bölümünü ve
`adapter_summary.json`'ların parametre sayılarını tek bir CSV'de toplamak
için:

```python
import json, csv
from pathlib import Path

root = Path('/content/drive/MyDrive/TRCAP_projection_exp')
rows = []
for run_dir in sorted(root.glob('P*_*')):
    metrics = json.loads((run_dir / 'metrics.json').read_text(encoding='utf-8'))
    summary = json.loads((run_dir / 'adapter_summary.json').read_text(encoding='utf-8'))
    best = metrics['best']
    rows.append({
        'run': run_dir.name,
        'adapter': summary['adapter_name'],
        'trainable_params': summary['trainable_parameters'],
        'best_iteration': best['iteration'],
        'best_CIDEr': best.get('CIDEr'),
    })

out_path = root / 'projection_results_summary.csv'
with open(out_path, 'w', newline='', encoding='utf-8') as fp:
    writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
print(out_path)
for row in rows:
    print(row)
```

## Yeniden üretilebilirlik notu

`run_projection_experiments.py` her adapter'ı bağımsız çalıştırır: model,
optimizer, scheduler ve iterasyon sayacı sıfırdan kurulur, hiçbir ağırlık
bir adapter'dan diğerine taşınmaz. 7 deneyin tamamı aynı seed (42), aynı
batch (64), aynı veri kümesi (TasvirEt) ve aynı değerlendirme aralığından
(1.000 iterasyon) geçer; değişen tek şey `model.projection_adapter`
değeridir.
