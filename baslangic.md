# Baslangic Durumu

Bu dosya, `hibrit_0` calisma dizininin nasil olusturuldugunu ve devralindigi
noktadaki envanteri kaydeder.

## Kaynak Dizin Secimi

`Program/` altinda 3 aday dizin degerlendirildi:

| Dizin | Egitim kodu? | DINOv2/hibrit mimari? | Colab akisi? | Durum |
|-------|:---:|:---:|:---:|-------|
| `2025tasviret_upd` | Var (`train.py`, `trainer.py`, `eval.py`) | Var (`Model/dino/dino.py`, patch-token cikarimi) | Var (`COLAB_TASVIRET_BASELINE.md`, `tools/preflight_colab.py`) | **SECILDI** |
| `TRCaptionNetpp` | Yok (sadece Gradio demo) | Kismen (sadece cikarim) | Yok | Elendi |
| `TRcaptionNET_baseline` | Var | Yok (CLIP-only baseline, DINOv2 yok) | Yok | Elendi |

`2025tasviret_upd`, TRCaptionNet++ public checkpoint'ini baslangic agirligi
olarak alip TasvirEt uzerinde fine-tune eden, git gecmisi temiz (11 commit)
ve mimarisi (`clip`/`dino2` secimli encoder + `Proj` + `BertLMHeadModel`
decoder) planla dogrudan uyumlu tek dizindi.

## Kopyalama Detaylari

`2025tasviret_upd` icerigi asagidakiler **haric** bu dizine kopyalandi
(robocopy ile, 2026-09-12):

- `.git/` — yeni dizin icin ayri/temiz bir gecmis baslatilacak (henuz
  baslatilmadi; git init yapilmasi gerekiyorsa ayrica talep edilmeli).
- `__pycache__/` — derlenmis bytecode, gereksiz.
- `checkpoints/` (2.4 GB, `TRCaptionNetpp_Large.pth`) — `tools/download_checkpoint.py`
  ile tekrar indirilebilir; repo icinde tasinmasi gereksiz yer kaplar.
- `Data/tasvir-et/__MACOSX/` ve `*.zip` — zip artigi/gereksiz dosyalar.

Kopyalanan onemli klasor/dosyalar: `Model/` (bert, clip, dino alt
modulleriyle), `Datasets/`, `configs/tasviret/tasviretpp_large_tasviret.yaml`,
`tools/`, `transform/`, `train.py`, `trainer.py`, `eval.py`, `utils.py`,
`app.py`, `requirements.txt`, `requirements_colab.txt`,
`COLAB_TASVIRET_BASELINE.md`, `Data/tasvir-et/*.json` (caption/split
dosyalari), `images/` (demo test goruntuleri).

## Devralinan Mimarinin Ozeti (Degisiklik Yapilmadan Once)

- **Encoder:** `Model/TRCaptionNet.py` icinde config'e gore `clip` (OpenAI/CN-CLIP
  tarzi ViT) veya `dino2` (DINOv2, `Model/dino/dino.py` uzerinden
  `forward_features(x)['x_norm_patchtokens']`) secilebiliyor. Aktif config
  (`configs/tasviret/tasviretpp_large_tasviret.yaml`) `dino2: dinov2_vitl14`
  kullaniyor.
- **Projeksiyon:** `Proj` sinifi — 1 Transformer bloğu (16 head) + tek
  `nn.Linear(1024, 768)`.
- **Decoder:** `BertLMHeadModel` (cross-attention'li), tokenizer/config
  kaynak: `dbmdz/electra-base-turkish-mc4-cased-discriminator`.
- **Egitim:** Encoder her zaman `torch.no_grad()` ile dondurulmus; sadece
  `proj` ve `language_decoder` optimizer'a giriyor (ayri LR gruplariyla:
  decoder `lr`, proj `lr_proj`). Tek asamali (warmup yok, sadece linear LR
  warmup + decay).
- **Veri:** TasvirEt (Flickr8K-Turkish goruntuleri + HUCVL Turkce caption'lari),
  train/val/test split JSON'lari `Data/tasvir-et/` altinda hazir.

## Sonraki Adim

Bkz. [devam.md](devam.md).
