# TRCaptionNet++ TasvirEt Baseline - Colab/L4 Akisi

Bu akis public `TRCaptionNetpp_Large.pth` checkpoint'ini baslangic agirligi olarak alir, mimariye dokunmadan TasvirEt train split'i uzerinde fine-tune eder ve test split'inde resmi caption metriklerini hesaplar.

Bu rehber Colab notebook hucreleri icindir. Colab Terminal acarsan `!` isaretlerini kaldirabilir ve `%cd` yerine normal `cd` kullanabilirsin. Notebook hucrelerinde calisirken `%cd /content/TRCAP` yapildiktan sonra checkpoint ve dataset klasorleri dogru repo icine olusur; sonradan `mv` ile tasima gerekmemelidir. Arac scriptleri relative pathleri repo kokune gore cozer, bu nedenle `/content/TRCAP/tools/...` seklinde cagrildiklarinda da dosyalari `/content/TRCAP` altinda okur/yazar.

## Deneyde sabit tutulanlar

- Encoder: `DINOv2 ViT-L/14`, 224x224 giris, egitim sirasinda frozen
- Projection: bir Transformer block, 16 attention head, ardindan `Linear(1024, 768)`
- Decoder: `BertLMHeadModel`; kaynak config/tokenizer `dbmdz/electra-base-turkish-mc4-cased-discriminator`
- Egitilen moduller: projection ve language decoder
- Optimizer: AdamW, decoder LR `2e-5`, projection LR `1e-4`, betas `(0.9, 0.99)`, weight decay `0.01`, gradient clipping `1.0`
- Schedule: linear warmup, 500 warmup iteration, toplam 10.000 iteration, ardindan linear decay
- Batch size: 64
- Validation: her 1.000 iteration, hedef metrik `Bleu_4`
- Generation: `max_length=35`, `min_length=12`, `num_beams=3`, `repetition_penalty=1.1`

## 1. Runtime kontrolu

Colab runtime'da GPU olarak L4 sec. Notebook hucrelerinde:

```python
!python --version
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
```

## 2. Repoyu klonla

Temiz baslangic icin:

```python
%cd /content
!rm -rf /content/TRCAP
!git clone https://github.com/mhr871/TRCAP.git
%cd /content/TRCAP
!git rev-parse --short HEAD
```

Repo zaten varsa ve silmeden guncellemek istersen:

```python
%cd /content/TRCAP
!git pull --ff-only
!git rev-parse --short HEAD
```

## 3. Kutuphaneleri kur ve kontrol et

```python
!python -m pip install -r requirements_colab.txt
```

```python
!python -c "import torch, transformers, tokenizers, cv2, pyarrow; print('torch=', torch.__version__); print('transformers=', transformers.__version__); print('tokenizers=', tokenizers.__version__); print('opencv=', cv2.__version__); print('pyarrow=', pyarrow.__version__); print('cuda=', torch.cuda.is_available())"
```

`transformers==4.38.2` ve `tokenizers==0.15.2`, guncel Colab Python 3.12 ile uyumluluk icin kullanilir. Public checkpoint strict yukleme ve caption uretimi, eski `transformers==4.27.3` ile ayni ciktiyi verecek sekilde test edilmistir.

## 4. Public checkpoint'i indir

```python
!python tools/download_checkpoint.py --output checkpoints/TRCaptionNetpp_Large.pth
```

```python
!sha256sum checkpoints/TRCaptionNetpp_Large.pth
```

Beklenen deger:

```text
c055ef247f968c86140b941506026721ca4c301ef3c7f6b421caec89ada8ebf3
```

## 5. Caption JSON ve splitleri hazirla

Ilk komut resmi HUCVL caption arsivini indirir ve public JSON'daki split alanlarini COCO bicimine cevirir. Goruntuler henuz olmadigi icin bu ilk adimda `--allow-missing-images` bilerek kullanilir.

```python
!python tools/prepare_tasviret.py --allow-missing-images
```

HUCVL caption sunucusunda gecici DNS/erisim hatasi gorursen ayni hucreyi tekrar calistir. Indirilmis `tasviret8k_captions.zip` veya `tasviret8k_captions.json` varsa arac onu kullanarak devam eder.

Beklenen sayilar:

```text
train: 6000 images, 12028 captions
val:   1000 images,  2006 captions
test:  1000 images,  2003 captions
```

## 6. Eslesen Flickr8K goruntulerini indir

Asagidaki arac `atasoglu/flickr8k-turkish` mirror'inin sabitlenmis `12424a4...` revizyonunu kullanir. Her satirdaki `imgid` ve ilk iki Turkce caption'i resmi HUCVL JSON ile karsilastirir; goruntuyu resmi JSON'daki dosya adiyla kaydeder. Yaklasik 1.1 GB indirir.

```python
!python tools/download_tasviret_images.py
```

Indirme bittikten sonra `--allow-missing-images` kullanmadan kesin eslesme kontrolunu calistir:

```python
!python tools/prepare_tasviret.py --images-root Data/flickr8k/images
```

Bu komut 8.000 kayittan tek bir goruntu bile eksikse durmalidir.

## 7. Drive'i cikti icin hazirla

Uzun egitimde Colab oturumu kapanirsa checkpoint'lerin kaybolmamasi icin once bir notebook hucre kisminda Drive'i bagla:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Ardindan notebook hucrelerinde:

```python
!mkdir -p /content/drive/MyDrive/TRCAP_runs
```

## 8. Tam preflight kontrolu

Bu kontrol GPU/VRAM, Java, config, checkpoint byte/SHA256, split sayilari, split cakismasi, 8.000 goruntunun acilabilirligi, modelin `strict=True` yuklenmesi ve bir gercek GPU caption uretimini test eder.

```python
!PYTHONPATH=/content/TRCAP python tools/preflight_colab.py
```

Son satir mutlaka su olmalidir:

```text
PREFLIGHT PASSED: baseline is ready for training.
```

## 9. Fine-tune oncesi public checkpoint testi

```python
!PYTHONPATH=/content/TRCAP python eval.py \
  --config configs/tasviret/tasviretpp_large_tasviret.yaml \
  --weights checkpoints/TRCaptionNetpp_Large.pth \
  --test-json Data/tasvir-et/tasvir_test.json \
  --test-data Data/flickr8k/images \
  --dataset tasviret \
  --output-dir /content/drive/MyDrive/TRCAP_runs/tasviret_test_public_checkpoint
```

Metrikler `metrics.json`, uretilen caption'lar `predictions.json` icinde saklanir.

## 10. TasvirEt fine-tune egitimini baslat

```python
!PYTHONPATH=/content/TRCAP python -u train.py \
  --config configs/tasviret/tasviretpp_large_tasviret.yaml \
  --save-dir /content/drive/MyDrive/TRCAP_runs
```

Egitim cikti klasoru:

```text
/content/drive/MyDrive/TRCAP_runs/tasviretpp_large_tasviret_baseline
```

Her 1.000 iteration sonunda validation yapilir ve devam edilebilir `model_last.pth` atomik olarak yenilenir. En iyi `Bleu_4` sonucu ayrica `model_best.pth` olur.

Colab kesilirse ayni runtime hazirliklarini yaptiktan sonra egitime su komutla devam et:

```python
!PYTHONPATH=/content/TRCAP python -u train.py \
  --config configs/tasviret/tasviretpp_large_tasviret.yaml \
  --save-dir /content/drive/MyDrive/TRCAP_runs \
  --resume /content/drive/MyDrive/TRCAP_runs/tasviretpp_large_tasviret_baseline/model_last.pth
```

## 11. Final test

```python
!PYTHONPATH=/content/TRCAP python eval.py \
  --config configs/tasviret/tasviretpp_large_tasviret.yaml \
  --weights /content/drive/MyDrive/TRCAP_runs/tasviretpp_large_tasviret_baseline/model_best.pth \
  --test-json Data/tasvir-et/tasvir_test.json \
  --test-data Data/flickr8k/images \
  --dataset tasviret \
  --output-dir /content/drive/MyDrive/TRCAP_runs/tasviret_test_finetuned_best
```

Final metrik dosyasi:

```text
/content/drive/MyDrive/TRCAP_runs/tasviret_test_finetuned_best/metrics.json
```

## Yeniden uretilebilirlik siniri

Public HUCVL JSON `6000/1000/1000`, toplam 8.000 goruntu ve 16.037 caption icerir. TasvirEt makalesindeki veri kumesi istatistigi ise 8.091 goruntu ve 12.222 Turkce aciklamadir. Yazarlarin TRCaptionNet++ deneyinde kullandigi kesin `tasvir_train/val/test` dosyalari ve eksiksiz fine-tune kodu public kaynaklarda paylasilmadigi icin, yayin tablosundaki sayilarin birebir cikacagi garanti edilemez. Bu akis, public checkpoint + indirilebilir resmi caption JSON + public TRCaptionNet hiperparametreleriyle kurulabilen denetlenebilir baseline'dir; sonraki projection deneylerinin tamaminda ayni akis korunmalidir.
