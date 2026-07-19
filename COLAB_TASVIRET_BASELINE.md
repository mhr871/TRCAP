# TRCaptionNet++ TasvirEt Baseline - Colab Akisi

Bu akista amac, public `TRCaptionNetpp_Large.pth` checkpoint'ini baslangic agirligi olarak alip mimariye dokunmadan TasvirEt uzerinde fine-tune etmek ve ayni evaluator ile metrikleri hesaplamaktir.

## 1. Colab Runtime

- Runtime type: GPU
- GPU tercihi: L4
- Python/Ubuntu: Colab default

## 2. Projeyi Colab'a Al

Bu klasoru Drive'a zip olarak koyduktan sonra:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
cd /content
mkdir -p /content/2025tasviret_upd
unzip -q "/content/drive/MyDrive/2025tasviret_upd_colab.zip" -d /content/2025tasviret_upd
cd /content/2025tasviret_upd
```

## 3. Kutuphaneler

```bash
pip install -r requirements_colab.txt
```

Colab'in kendi `torch`/`torchvision` paketlerini korumak icin bu dosyada torch yeniden kurulmaz.

## 4. Checkpoint

```bash
python tools/download_checkpoint.py --output checkpoints/TRCaptionNetpp_Large.pth
```

## 5. TasvirEt Caption Splitleri

Resmi TasvirEt caption JSON dosyasi otomatik indirilir ve su dosyalara cevrilir:

- `Data/tasvir-et/tasvir_train.json`
- `Data/tasvir-et/tasvir_val.json`
- `Data/tasvir-et/tasvir_test.json`

Flickr8K goruntulerini su klasore koy:

```text
Data/flickr8k/images
```

Eger goruntuler Drive'daysa symlink kullan:

```bash
mkdir -p Data/flickr8k
ln -s "/content/drive/MyDrive/datasets/flickr8k/images" Data/flickr8k/images
```

Sonra splitleri hazirla ve goruntu eslesmesini kontrol et:

```bash
python tools/prepare_tasviret.py --images-root Data/flickr8k/images
```

Beklenen split sayilari:

```text
train: 6000 images
val: 1000 images
test: 1000 images
```

Not: TasvirEt makalesinde veri kumesi istatistigi `8091` goruntu ve `12222` Turkce aciklama olarak verilir; Flickr8K kaynaklarinda `8092` goruntu ifadesi de gorulebilir. Bu repodaki `prepare_tasviret.py` scripti ise HUCVL tarafindan indirilebilir `tasviret8k_captions.json` dosyasinin icindeki resmi `train/val/test` alanlarini kullanir. Bu dosyada deney icin kullanilan splitler `6000/1000/1000`, yani toplam `8000` goruntudur. Makale degerleriyle birebir karsilastirmada, yazarlarin kullandigi kesin split dosyalari bulunursa bu dosyalar `Data/tasvir-et/` altina dogrudan konup ayni train/eval akisiyle kullanilmalidir.

## 6. Fine-tune Oncesi Public Checkpoint Testi

Bu adim, public checkpoint'in TasvirEt test setindeki dogrudan performansini kaydeder.

```bash
python eval.py \
  --config configs/tasviret/tasviretpp_large_tasviret.yaml \
  --weights checkpoints/TRCaptionNetpp_Large.pth \
  --test-json Data/tasvir-et/tasvir_test.json \
  --test-data Data/flickr8k/images \
  --dataset tasviret \
  --output-dir eval_outputs/tasviret_test_public_checkpoint
```

## 7. TasvirEt Fine-tune

```bash
python train.py --config configs/tasviret/tasviretpp_large_tasviret.yaml
```

Cikti klasoru:

```text
experiments/tasviretpp_large_tasviret_baseline
```

Kaydedilecek dosyalar:

- `model_best.pth`
- `model_last.pth`
- `prediction_*.json`
- `result_*.json`
- `log.txt`
- `tensorboard/`

## 8. Final Test

Fine-tune bittikten sonra:

```bash
python eval.py \
  --config configs/tasviret/tasviretpp_large_tasviret.yaml \
  --weights experiments/tasviretpp_large_tasviret_baseline/model_best.pth \
  --test-json Data/tasvir-et/tasvir_test.json \
  --test-data Data/flickr8k/images \
  --dataset tasviret \
  --output-dir eval_outputs/tasviret_test_finetuned_best
```

Metrik dosyasi:

```text
eval_outputs/tasviret_test_finetuned_best/metrics.json
```

## 9. Deney Protokolu Notu

Bu baseline'da degisen sey mimari degildir. Encoder, projection, decoder ve generation ayarlari korunur. Sadece public TRCaptionNet++ Large checkpoint'i TasvirEt train split'i ile fine-tune edilir ve test split'i uzerinde BLEU, METEOR, ROUGE-L, CIDEr ve SPICE ile degerlendirilir.
