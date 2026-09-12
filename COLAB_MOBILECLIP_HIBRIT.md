# MobileCLIP Hibrit Mimari (S0/S1/S2 + BERTurk) - Colab/L4 Akisi

Bu akis, `plan.md`'de tanimlanan hibrit mimariyi (DINOv2-L yerine MobileCLIP
S0/S1/S2 gorsel encoder + 2 katmanli MLP projeksiyon + BERTurk decoder) Colab
uzerinde iki asamali olarak (on-hizalama -> ana egitim) egitmek icin gerekli
tum hucreleri icerir. Hucre sirasi, daha once `COLAB_TASVIRET_BASELINE.md`
ile calisitirilip dogrulanmis komut sirasiyla ayni yapidadir (once Drive
baglama, sonra klonlama, kurulum, veri, preflight, egitim, test, demo).

Kod `github` remote'unun **`MC0`** dalina yuklendi:
`https://github.com/mhr871/TRCAP.git` (dal: `MC0`).

## Bu akiste sabit tutulanlar

- Encoder: `mobileclip_s0` / `mobileclip_s1` / `mobileclip_s2` (config ile
  secilir), 256x256 giris, egitim sirasinda tamamen frozen
  (`Model/mobileclip/mobileclip_encoder.py`)
- Projection: 2 katmanli MLP (`Linear -> GELU -> Linear`), `proj_type: mlp2`
- Decoder: `BertLMHeadModel`; kaynak config/tokenizer
  `dbmdz/bert-base-turkish-cased` (eskiden Electra discriminator
  checkpoint'i kullaniliyordu, ama `BertLMHeadModel`'in "bert." state-dict
  on-ekiyle uyusmadigi icin hicbir agirlik yuklenmiyordu -- bkz. devam.md)
- Iki asamali egitim:
  - **Stage 1 (on-hizalama, ~2 epoch):** decoder frozen, sadece projeksiyon
    MLP egitilir
  - **Stage 2 (ana egitim, ~53 epoch, DINOv2 baseline'iyla ayni toplam veri
    goruntusu):** decoder ve projeksiyon birlikte egitilir, encoder hala
    frozen
- Batch size, DINOv2-L baseline'ina (64) gore MobileCLIP'in daha dusuk VRAM
  kullanimindan yararlanarak S0/S1 icin 128'e, S2 icin 96'ya cikarildi;
  `max_iter` de ayni toplam epoch sayisini korumak icin oranli kucultuldu
  (bkz. her config dosyasindaki yorum satiri)
- Diger tum hiperparametreler (`lr`, `lr_proj`, `weight_decay`, ...)
  `COLAB_TASVIRET_BASELINE.md`'deki dogrulanmis degerlerle ayni tutuldu

## 1. Drive'i bagla

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2. Repoyu klonla (MC0 dali)

```python
%cd /content
!rm -rf /content/TRCAP
!git clone -b MC0 https://github.com/mhr871/TRCAP.git
%cd /content/TRCAP
!git rev-parse --short HEAD
```

`Model/clip/bpe_simple_vocab_16e6.txt.gz` (hangi encoder secilirse secilsin
`Model/clip/clip.py` her zaman import edildigi icin gereklidir, MobileCLIP
dahil) **artik Git LFS ile tutulmuyor** -- eskiden `.gitattributes`'daki
genel `*.gz filter=lfs` kuralina takiliyordu, Colab'in git-lfs kurulumu
bazi imajlarda sessizce basarisiz oluyordu ve plain `git clone` gercek
dosya yerine kucuk bir "pointer" metni indirip `gzip.BadGzipFile: Not a
gzipped file` hatasina yol aciyordu (birden fazla oturumda tekrar
tekrar karsilasildi, bkz. devam.md). Bu dosya artik normal bir git
blob'u olarak commit'lendi, yani yukaridaki plain `git clone` tek
basina yeterli -- ayrica `git lfs install`/`git lfs pull` calistirmaya
GEREK YOK.

(Repoda hala LFS ile tutulan birkac demo goruntusu var --
`images/test*.png` -- ama bunlar egitim/preflight/eval akisinda
kullanilmiyor, sadece `app.py` demosu icin; egitim icin onlari
indirmenize gerek yok.)

Repo zaten varsa ve silmeden guncellemek istersen:

```python
%cd /content/TRCAP
!git checkout MC0
!git pull --ff-only
!git rev-parse --short HEAD
```

## 3. Kutuphaneleri kur

```python
!python -m pip install -r requirements_colab.txt
!python -m pip install --no-deps git+https://github.com/apple/ml-mobileclip.git
```

`--no-deps` bilerek kullanilir: `ml-mobileclip`'in kendi `requirements.txt`'i
`torch>=2.8.0` ister ve `datasets`/`clip-benchmark` gibi bu projede gereksiz
agir paketleri de kurmaya calisir; bunlar kurulunca mevcut Colab
`torch`/`transformers` surumleriyle celisebilir. `open-clip-torch` ise
**gereklidir** ve zaten `requirements_colab.txt` icinde ayri satir olarak
listelendi: `mobileclip/__init__.py`, biz sadece gorsel kuleyi kullansak bile
kendi metin/tokenizer kodu icin modul seviyesinde `import open_clip` yapiyor;
bu paket kurulu olmadan `import mobileclip` bile basarisiz olur (bu hata
Colab'da bir kez gercekten alindi ve boylece duzeltildi). `timm>=0.9.5` de
ayni sekilde `requirements_colab.txt` icinde.

```python
!python -c "import torch, transformers, timm, mobileclip; print('torch=', torch.__version__); print('transformers=', transformers.__version__); print('timm=', timm.__version__); print('cuda=', torch.cuda.is_available())"
```

## 4. TasvirEt veri hazirligi (COLAB_TASVIRET_BASELINE.md ile ayni)

```python
!python tools/prepare_tasviret.py --allow-missing-images
!python tools/download_tasviret_images.py
!python tools/prepare_tasviret.py --images-root Data/flickr8k/images
```

Beklenen sayilar: `train: 6000 images, 12028 captions`,
`val: 1000 images, 2006 captions`, `test: 1000 images, 2003 captions`.

## 5. MobileCLIP checkpoint'ini indir

Hangi boyutla baslayacaksan (**s0 ile pilot testten baslamaniz onerilir**):

```python
!python tools/download_mobileclip.py --model mobileclip_s0
# !python tools/download_mobileclip.py --model mobileclip_s1
# !python tools/download_mobileclip.py --model mobileclip_s2
```

## 6. Preflight: mimarinin hatasiz calistigini dogrula

```python
%cd /content/TRCAP
!PYTHONPATH=/content/TRCAP python tools/preflight_mobileclip.py \
  --config configs/tasviret/mobileclip_s0_stage1.yaml
```

Son satir mutlaka su olmalidir:

```text
PREFLIGHT PASSED: configs/tasviret/mobileclip_s0_stage1.yaml is ready for training.
```

## 7. Stage 1 - On hizalama (sadece projeksiyon MLP, ~2 epoch)

```python
!mkdir -p /content/drive/MyDrive/TRCAP_hibrit_runs
!rm -rf /content/drive/MyDrive/TRCAP_hibrit_runs/mobileclip_s0_stage1_tasviret
!PYTHONPATH=/content/TRCAP python -u train.py \
  --config configs/tasviret/mobileclip_s0_stage1.yaml \
  --save-dir /content/drive/MyDrive/TRCAP_hibrit_runs
```

Kesilirse devam (yukaridaki `rm -rf` temizligini BU sefer atla):

```python
!PYTHONPATH=/content/TRCAP python -u train.py \
  --config configs/tasviret/mobileclip_s0_stage1.yaml \
  --save-dir /content/drive/MyDrive/TRCAP_hibrit_runs \
  --resume /content/drive/MyDrive/TRCAP_hibrit_runs/mobileclip_s0_stage1_tasviret/model_last.pth
```

## 8. Stage 2 - Ana egitim (projeksiyon + decoder, ~53 epoch)

`mobileclip_s0_stage2.yaml` varsayilan olarak yerel
`experiments/mobileclip_s0_stage1_tasviret/model_last.pth` yolunu bekler;
Stage 1 ciktisini oraya kopyala:

```python
!mkdir -p experiments/mobileclip_s0_stage1_tasviret
!cp /content/drive/MyDrive/TRCAP_hibrit_runs/mobileclip_s0_stage1_tasviret/model_last.pth \
    experiments/mobileclip_s0_stage1_tasviret/model_last.pth
```

```python
!mkdir -p /content/drive/MyDrive/TRCAP_hibrit_runs
!rm -rf /content/drive/MyDrive/TRCAP_hibrit_runs/mobileclip_s0_stage2_tasviret
!PYTHONPATH=/content/TRCAP python -u train.py \
  --config configs/tasviret/mobileclip_s0_stage2.yaml \
  --save-dir /content/drive/MyDrive/TRCAP_hibrit_runs
```

Kesilirse devam:

```python
!PYTHONPATH=/content/TRCAP python -u train.py \
  --config configs/tasviret/mobileclip_s0_stage2.yaml \
  --save-dir /content/drive/MyDrive/TRCAP_hibrit_runs \
  --resume /content/drive/MyDrive/TRCAP_hibrit_runs/mobileclip_s0_stage2_tasviret/model_last.pth
```

## 9. Final test

```python
!PYTHONPATH=/content/TRCAP python eval.py \
  --config configs/tasviret/mobileclip_s0_stage2.yaml \
  --weights /content/drive/MyDrive/TRCAP_hibrit_runs/mobileclip_s0_stage2_tasviret/model_best.pth \
  --test-json Data/tasvir-et/tasvir_test.json \
  --test-data Data/flickr8k/images \
  --dataset tasviret \
  --output-dir /content/drive/MyDrive/TRCAP_hibrit_runs/mobileclip_s0_test
```

## 10. Demo / tek goruntu cikarimi

```python
%cd /content/TRCAP
!git pull
from google.colab import files
uploaded = files.upload()
image_path = next(iter(uploaded.keys()))
!PYTHONPATH=/content/TRCAP python tools/infer_image.py \
  --image "{image_path}" \
  --config configs/tasviret/mobileclip_s0_stage2.yaml \
  --weights /content/drive/MyDrive/TRCAP_hibrit_runs/mobileclip_s0_stage2_tasviret/model_best.pth
```

## 11. S1 ve S2 icin tekrarla

Adim 5-10'u sirasiyla `mobileclip_s1_stage1.yaml`/`mobileclip_s1_stage2.yaml`
ve `mobileclip_s2_stage1.yaml`/`mobileclip_s2_stage2.yaml` ile tekrarla
(model adi, checkpoint yolu ve klasor isimleri disinda akis birebir aynidir).

## Sonraki adim: hiperparametre gozden gecirme

S0 pilotu (Stage 1 + Stage 2) hatasiz tamamlanip loss/Bleu_4 eğrisi
mantikli gorundukten sonra, `lr`, `lr_proj`, `batch_size`, `max_iter`,
`warm_up_iter`, `num_eval_iter` degerleri her 3 boyut icin ayri ayri gozden
gecirilip guncellenecek (bkz. `hedef.md` ve `devam.md`).
