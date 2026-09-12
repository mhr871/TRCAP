# Devam / Yapilacaklar Listesi

Bu dosya oturumlar arasi "kaldigimiz yer" takibi icindir. Her oturum sonunda
guncellenmeli; her oturum basinda buradan devam edilmeli.

## Durum: Kod degisiklikleri tamamlandi, Colab'da calistirilip dogrulanmadi (2026-09-12)

### Tamamlananlar

- [x] 3 aday dizin incelendi, `2025tasviret_upd` en uygun taban olarak
      secildi ve `hibrit mimariler/hibrit_0` icine kopyalandi (bkz.
      `baslangic.md`).
- [x] `plan.md`, `amac.md`, `hedef.md`, `baslangic.md`, `devam.md` olusturuldu.
- [x] **`Model/mobileclip/` modulu yazildi** (`mobileclip_encoder.py` +
      `__init__.py`): `apple/ml-mobileclip` pip paketini sarmalar; FastViT
      backbone'unun `forward_embeddings -> forward_tokens -> conv_exp`
      zincirini GAP/head'e ugramadan cagirip `(B, N, C)` patch-token dizisi
      dondurur; `get_output_dim()` ile kanal sayisini calisirken belirler
      (s0/s1: 1024, s2: 1280, 256x256 girişte); encoder her zaman
      `requires_grad_(False)` ve `.train()` override'i ile daima eval modda
      kalir.
- [x] **`Model/TRCaptionNet.py` guncellendi:** `"mobileclip" in config` dali
      eklendi; `MlpProj` sinifi (`Linear -> GELU -> Linear`) eklendi;
      `proj_type: mlp2` config alaniyla secilebiliyor (mevcut DINOv2 `Proj`
      transformer siniifina dokunulmadi).
- [x] **`Datasets/dataset_utils.py` guncellendi:** `getMobileCLIPTransforms`
      eklendi (256x256, OpenAI-CLIP normalizasyon sabitleri — Model/clip/clip.py
      ile ayni degerler); `getTestTransforms` artik `"mobileclip" in model_config`
      icin bu transformu seciyor. `app.py`, `eval.py`, `tools/infer_image.py`
      zaten `getTestTransforms` uzerinden gectigi icin ek degisiklik gerekmedi.
- [x] **`trainer.py` guncellendi:** `freeze_decoder` config alani eklendi.
      True oldugunda: decoder parametreleri `requires_grad_(False)`, decoder
      `.eval()` kilitleniyor (train()/eval()/save_model() sonrasi tekrar
      eval'e donduruluyor), optimizer'a decoder param grubu hic girmiyor,
      `validate_optimizer_param_groups`/`get_current_lrs` bu duruma gore
      guncellendi.
- [x] **6 config dosyasi olusturuldu:**
      `configs/tasviret/mobileclip_{s0,s1,s2}_stage{1,2}.yaml`. Stage 1:
      `freeze_decoder: true`, `init_model_ckpt: ~` (mobileclip + HF pretrained
      Electra'dan baslar). Stage 2: `freeze_decoder: false`,
      `init_model_ckpt: experiments/mobileclip_<size>_stage1_tasviret/model_last.pth`.
      Tum sayisal hiperparametreler (`lr`, `lr_proj`, `batch_size`, `max_iter`,
      `warm_up_iter`, `num_eval_iter`) **yer tutucudur**, asagida "Hiperparametre
      gozden gecirme" maddesine kadar kesin degil.
- [x] **`tools/download_mobileclip.py`** eklendi: Apple'in resmi
      `docs-assets.developer.apple.com/.../mobileclip_{s0,s1,s2,b}.pt`
      URL'lerinden indirir (`tools/download_checkpoint.py` ile ayni desen).
- [x] **`tools/preflight_mobileclip.py`** eklendi: runtime + TasvirEt veri
      butunlugu kontrolu + `mobileclip` paketi/checkpoint kontrolu +
      gercek forward/backward/generate() smoke testi (proj'un gradyan aldigini,
      encoder'in HER ZAMAN, decoder'in sadece freeze_decoder=false iken
      gradyan aldigini dogrular).
- [x] **`requirements_colab.txt`** guncellendi: `timm>=0.9.5` eklendi;
      `mobileclip` paketinin `--no-deps` ile ayrica kurulmasi gerektigi not
      edildi (kendi requirements.txt'i `torch>=2.8.0` + `open-clip-torch`/
      `datasets`/`clip-benchmark` istiyor, bunlar gereksiz ve versiyon
      catismasi yaratir).
- [x] **`COLAB_MOBILECLIP_HIBRIT.md`** eklendi: klonlama, kutuphane kurulumu,
      veri hazirligi (COLAB_TASVIRET_BASELINE.md ile ayni), checkpoint indirme,
      preflight, Stage 1, Stage 2, final test ve S1/S2 icin tekrar adimlarini
      iceren tam Colab hucre akisi.
- [x] Tum yeni/degisen `.py` dosyalari `python -m py_compile` ile, tum yeni
      `.yaml` dosyalari `yaml.safe_load` ile dogrulandi (sozdizimi hatasi yok).

### Sirada (Oncelik Sirasiyla)

1. [ ] **Colab'da gercek dogrulama (henuz yapilmadi):** `COLAB_MOBILECLIP_HIBRIT.md`
   adim 1-7'yi calistirip `tools/preflight_mobileclip.py --config
   configs/tasviret/mobileclip_s0_stage1.yaml` ciktisinin
   `PREFLIGHT PASSED` ile bittigini dogrula. Bu adim gercek bir GPU/Colab
   ortaminda calistirilmadi; `mobileclip` paketinin gercekten
   `create_model_and_transforms`/`image_encoder.model` API'siyle beklenen
   sekilde davrandigi (repo kaynagindan okunarak) varsayildi ama ucundan
   ucuna hic calistirilmadi.
2. [ ] **Stage 1 pilot (MobileCLIP-S0):** Preflight gectikten sonra
   `mobileclip_s0_stage1.yaml` ile kisa bir egitim calistirilip loss'un
   dustugu gozlemlenecek.
3. [ ] **Stage 2 pilot (MobileCLIP-S0):** Stage 1 ciktisindan devam edip
   `mobileclip_s0_stage2.yaml` ile ana egitim calistirilacak, `eval.py` ile
   test split'inde metrik alinacak.
4. [ ] **S1 ve S2 deneyleri:** S0 pilotu basarili olursa ayni akisla sirayla
   calistirilacak.
5. [ ] **Hiperparametre gozden gecirme (kullanici tarafindan talep edildi,
   S0 pilotundan sonra yapilacak):** `lr`, `lr_proj`, `batch_size`,
   `max_iter`, `warm_up_iter`, `num_eval_iter` degerleri her 3 config icin
   pilot sonuclarina gore guncellenecek.
6. [ ] **Sonuc karsilastirmasi:** DINOv2-L baseline sonuclariyla BLEU/CIDEr/
   METEOR ve model boyutu karsilastirmasi yapilip tez icin tablo/grafik
   hazirlanacak.

### Acik Sorular / Karar Bekleyenler / Riskler

- `Model/mobileclip/mobileclip_encoder.py` icindeki `backbone.forward_embeddings
  -> forward_tokens -> conv_exp` cagri zinciri ve kanal boyutlari (s0/s1: 1024,
  s2: 1280 @ 256x256) `apple/ml-mobileclip` reposunun `main` branch kaynak
  kodunun okunmasina dayaniyor (WebFetch ile dogrulandi), ama gercek bir
  ortamda import edilip calistirilmadi. Colab'da ilk `preflight_mobileclip.py`
  calistirmasi bu varsayimlarin dogru oldugunu kanitlayacak; hata cikarsa
  once bu dosyayi guncellemek gerekecek.
- `pip install --no-deps git+...` sonrasi `timm` surumunun mobileclip'in
  kayitli `mci0/mci1/mci2` model isimlerini taniyip tanimadigi kontrol
  edilmedi (repo `timm>=0.9.5` istiyor, `requirements_colab.txt`'e ayni
  alt sinir eklendi, ama Colab'in mevcut `timm` surumuyle celiski olabilir).
- Bu dizin icin ayri bir git deposu baslatilip baslatilmayacagi (`git init`)
  kullanicidan onay bekliyor — henuz yapilmadi.
- SigLIP-Base (opsiyonel 4. deney) zaman/kaynak durumuna gore karar
  verilecek.

## Nasil Devam Edilir

Yeni bir oturuma baslarken: once bu dosyayi, sonra `plan.md` ve
`baslangic.md`'yi oku; "Sirada" listesindeki ilk isaretlenmemis maddeden
devam et. Bir madde tamamlaninca burada `[x]` olarak isaretle ve gerekirse
yeni alt maddeler ekle.
