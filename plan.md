# Hibrit Mimari Plani: MobileCLIP + BERTurk (hibrit_0)

## Temel Fikir

Mevcut TRCaptionNet++ mimarisinde gorsel encoder olarak kullanilan DINOv2-L
(~300M parametre), MobileCLIP ailesinin cok daha kucuk varyantlariyla
degistirilecek. Hedef: "cok daha kucuk bir gorsel encoder ile, kabul
edilebilir/rekabetci basari" hipotezini asamali olarak (S0 -> S1 -> S2) test
etmek.

Dil tarafinda (BERTurk / Turkce BERT tabanli decoder, cross-attention ile
gorsel token'lara bakan `BertLMHeadModel`) ve genel egitim/degerlendirme
iskeletinde (train.py, trainer.py, eval.py, Datasets/*) **buyuk bir degisiklik
yapilmayacak**. Bu repo (`hibrit_0`), `2025tasviret_upd` calisma kopyasindan
turetildi cunku o dizin plana en yakin, en temiz ve zaten Colab uyumlu egitim
altyapisina (bkz. `COLAB_TASVIRET_BASELINE.md`, `tools/preflight_colab.py`)
sahip olan surumdu.

## Neden `2025tasviret_upd` Secildi

Elimizde 3 aday dizin vardi:

1. **`2025tasviret_upd`** (SECILEN) — Tam egitim/degerlendirme pipeline'i
   (`train.py`, `trainer.py`, `eval.py`), `Model/dino/dino.py` uzerinden
   DINOv2 patch-token cikarimi, `Model/TRCaptionNet.py` icinde
   encoder-agnostik yapi (`clip` veya `dino2` anahtarina gore secim), hazir
   Colab akisi ve TasvirEt veri seti entegrasyonu. En temiz, en guncel git
   gecmisine sahip (11 commit, calisan bir "TasvirEt fine-tuning" hattı).
2. `TRCaptionNetpp` — Sadece cikarim (inference) icin Gradio demo'lari
   iceriyor; egitim kodu, dataset loader'lari veya config yok.
3. `TRcaptionNET_baseline` — Orijinal TRCaptionNet makalesinin CLIP-tabanli
   (DINOv2 degil) baseline'i; cok sayida deneysel config/log/not dosyasiyla
   dagitilmis durumda, hedeflenen DINOv2/hibrit mimariyle dogrudan uyumlu
   degil.

Bu nedenle `2025tasviret_upd` icerigi (kod, config, tools, Data/tasvir-et
caption JSON'lari) `.git`, `__pycache__`, `checkpoints/` (2.4GB, tekrar
indirilebilir) ve `__MACOSX` haric olacak sekilde bu dizine kopyalandi.

## Indirilecek Model Listesi (Encoder Varyantlari)

Apple'in resmi MobileCLIP deposundan (Hugging Face / OpenCLIP / apple/ml-mobileclip)
sirayla indirilip denenecek varyantlar:

| # | Model | Parametre (yaklasik) | Rol |
|---|-------|----------------------|-----|
| 1 | MobileCLIP-S0 | ~11.4M | En kucuk/en hizli; pilot/hizalama testi |
| 2 | MobileCLIP-S1 | ~21.5M | Hiz/kalite dengesi; ana aday |
| 3 | MobileCLIP-S2 | ~35.7M | S ailesinde en yuksek kapasite (DINOv2-L'nin hala ~9 kati kucuk) |
| 4 (opsiyonel yedek) | SigLIP-Base | ~86M | MobileCLIP sonrasi karsilastirma icin dil-hizali guclu alternatif |

## Kod ve Altyapida Yapilacak Degisiklikler

### 1. Ozellik Cikarimi (Feature Extraction)

- Eski: `Model/dino/dino.py` -> `forward_features(x)['x_norm_patchtokens']`
  ile DINOv2'den patch token dizisi (`B x N x 1024`) aliniyordu.
- Yeni: `Model/mobileclip/` altinda yeni bir sarmalayici (`MobileCLIPEncoder`)
  eklenecek. MobileCLIP'in gorsel kulesindeki son Global Average Pooling (GAP)
  katmani bypass edilecek; pooling **oncesi** `B x C x H x W` ozellik haritasi
  cekilip `B x (H*W) x C` seklinde duzlestirilerek patch-token formatina
  donusturulecek. Bu, uzamsal (spatial) bilgiyi korur ve mevcut cross-attention
  decoder arayuzuyle (`encoder_hidden_states`) uyumlu kalir.
- `Model/TRCaptionNet.py` icindeki encoder secim mantigina (`"clip" in config`,
  `"dino2" in config`) yeni bir `"mobileclip" in config` dali eklenecek.

### 2. Projeksiyon Katmani (2 Katmanli MLP)

- Eski: `Proj` sinifi (bkz. `Model/TRCaptionNet.py:16-30`) bir Transformer
  bloku + tek `nn.Linear(encoder_output_size, 768)`.
- Yeni: MobileCLIP kanal boyutunu (`C`) BERTurk gizli boyutuna (768) baglayan
  `Linear -> GELU -> Linear` seklinde 2 katmanli MLP adaptoru eklenecek
  (`Model/TRCaptionNet.py` icine yeni bir `MlpProj` sinifi olarak, mevcut
  `Proj` sinifinin yaninda; config'te `proj_type: mlp2` gibi bir anahtarla
  secilebilir yapilacak). Bu katman, Ingilizce ile hizalanmis MobileCLIP
  gorsel uzayini BERTurk'un Turkce kelime-vektor uzayina esnek bicimde
  cevirir.

### 3. Iki Asamali Egitim / On Hizalama (Warmup Pipeline)

- **Asama 1 (On Hizalama, 1-2 epoch):** Encoder VE BERTurk decoder tamamen
  dondurulur (`requires_grad_(False)`); sadece yeni 2 katmanli MLP projeksiyon
  katmani egitilir. Amac: gorsel uzay ile Turkce dil uzayini birbirine
  alistirmak.
- **Asama 2 (Ana Egitim):** Encoder dondurulmus kalmaya devam eder; projeksiyon
  ve BERTurk (veya BERTurk + LoRA) katmanlari birlikte egitilir (mevcut
  `trainer.py` optimizer grup mantigina benzer, `decoder_decay/no_decay` +
  `proj_decay/no_decay` parametre gruplari zaten var — bu yapi buyuk olcude
  yeniden kullanilabilir).
- `trainer.py` icine bir `freeze_decoder` / `stage` parametresi eklenecek ki
  Asama 1'de decoder parametreleri optimizer'a hic girmesin (`lr=0` yerine
  `requires_grad=False` tercih edilecek, boylece BatchNorm/dropout durumu da
  dogru yonetilir).

### 4. Konfigurasyon ve Hiperparametreler

- Yeni config dosyalari: `configs/tasviret/mobileclip_s0_tasviret.yaml`,
  `..._s1_...yaml`, `..._s2_...yaml` (mevcut
  `configs/tasviret/tasviretpp_large_tasviret.yaml` sablon alinarak).
- Her config'e MobileCLIP giris kanal/vektor boyutu (`C`), `proj_type: mlp2`,
  ve iki asamali egitim icin `warmup_stage_iters` gibi alanlar eklenecek.
- MobileCLIP DINOv2-L'ye gore cok daha hafif oldugundan VRAM kullanimi ciddi
  dusecek; bu sayede `batch_size` artirilarak egitim hizlandirilabilir
  (mevcut `batch_size: 64` degeri her model icin yeniden ayarlanacak).

## Toplam Egitim Plani (Deney Sayisi)

| # | Deney | Encoder | Projeksiyon | Amac |
|---|-------|---------|-------------|------|
| 1 | Pilot / Hizalama Testi | MobileCLIP-S0 | 2-Layer MLP | Prototipin sorunsuz calistigini, loss'un dustugunu dogrulamak |
| 2 | S1 Deneyi | MobileCLIP-S1 | 2-Layer MLP | Ana aday; hiz/kalite dengesi |
| 3 | S2 Deneyi | MobileCLIP-S2 | 2-Layer MLP | Kapasite tavani; DINOv2-L baseline'ina en yakin sonuc |

(Opsiyonel 4. deney: SigLIP-Base, zaman/kaynak izin verirse.)

## Ilgili Dosyalar

- [amac.md](amac.md) — Bu hibrit mimari calismasinin tez baglamindaki amaci.
- [hedef.md](hedef.md) — Somut, olculebilir hedefler ve basari kriterleri.
- [baslangic.md](baslangic.md) — Baslangic durumu: hangi dizin neden secildi, mevcut kod envanteri.
- [devam.md](devam.md) — Oturumlar arasi devam/yapilacaklar listesi (canli checklist).
