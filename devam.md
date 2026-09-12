# Devam / Yapilacaklar Listesi

Bu dosya oturumlar arasi "kaldigimiz yer" takibi icindir. Her oturum sonunda
guncellenmeli; her oturum basinda buradan devam edilmeli.

## Durum: Stage 1 (MobileCLIP-S0) pilotu koktan bozuk cikti, checkpoint duzeltildi, YENIDEN calistirilmali (2026-09-12)

**Onceki "buyuk kilometre tasi" gecersiz:** `mobileclip_s0_stage1.yaml` 200/200
iterasyon tamamladi (Colab T4, gercek `mobileclip` paketiyle, mekanik olarak
sorunsuz calisti), ama sonuclar anlamsiz: Bleu_4 ~0.001-0.002, uretilen
caption'lar tamamen rastgele kelime yigini ("bir Kah kişilerle Yarar
gerçekleştirilmesi Vit İster..."). Kok neden mekanik degil, model init hatasi:
`model.bert: dbmdz/electra-base-turkish-mc4-cased-discriminator` idi, ama
`Model/TRCaptionNet.py`'deki `BertLMHeadModel` (`Model/bert/med.py`) kendi
ozel BLIP-tarzi implementasyonu ve `base_model_prefix = "bert"`. Electra
checkpoint'inin state-dict anahtarlari `electra.*` on-ekiyle geliyor, `bert.*`
degil, bu yuzden HF'nin `from_pretrained`'i hicbir agirligi eslestiremedi.
Bu, gercek Colab log'unda acikca goruluyordu: "newly initialized" listesi
`embeddings.word_embeddings.weight`, `embeddings.position_embeddings.weight`
ve `encoder.layer.0..11`'in TAMAMINI iceriyordu -- yani dil modeli sifirdan,
rastgele agirliklarla basladi. Bunun ustune `freeze_decoder: true` bu rastgele
decoder'i hemen dondurdu, yani decoder 200 iterasyon boyunca hic ogrenemedi;
sadece projeksiyon katmani egitildi ve rastgele/donmus bir decoder'i anlamli
caption uretecek hale getiremedi. Konfigurasyondaki "BERTurk/Electra'nin HF
pretrained agirliklari otomatik yukleniyor" varsayimi hicbir zaman gercek bir
calistirmanin log'undaki uyarilarla dogrulanmamisti -- yanlisti.

**Duzeltme uygulandi (henuz yeniden egitim YAPILMADI):** `model.bert` alani
6 mobileclip config dosyasinin tumunde (`mobileclip_{s0,s1,s2}_stage{1,2}.yaml`)
`dbmdz/bert-base-turkish-cased`'e cevrildi -- bu gercek bir `BertModel`
checkpoint'i (`bert.*` on-eki), yani embedding'ler ve tum 12 self-attention/FFN
katmani dogru sekilde pretrained agirliklarla yuklenecek; sadece yeni eklenen
cross-attention alt-katmanlari ve LM head rastgele kalacak (BLIP/ALBEF'in
decoder init yontemiyle ayni, bu normal ve beklenen). `tasviretpp_large_tasviret.yaml`
(DINOv2 baseline) bu sorundan ETKILENMEDI, cunku `init_model_ckpt` +
`strict_init: true` ile decoder zaten tam bir checkpoint'ten sonradan
tamamen uzerine yaziliyor.

**Sirada:** Stage 1 pilotu (MobileCLIP-S0) duzeltilmis config ile Colab'da
YENIDEN calistirilmali; asagidaki "Tamamlananlar" ve "Sirada" listesi bu
dogrultuda guncellendi (2. madde tekrar acildi).

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

1. [x] **Colab'da gercek dogrulama:** git-lfs ve open_clip sorunlari
   cozuldukten sonra `mobileclip_s0_stage1.yaml` Colab T4'te gercekten
   calistirildi ve ilerliyor (bkz. yukarida "Buyuk kilometre tasi").
2. [ ] **Stage 1 pilotu (MobileCLIP-S0) DUZELTILMIS bert checkpoint'iyle
   yeniden calistir:** onceki 200/200 iterasyonluk calistirma tamamlandi ama
   `model.bert`'in yanlis (Electra) checkpoint'i yuzunden decoder tamamen
   rastgele agirliklarla baslayip donduruldugu icin sonuclar gecersiz
   (Bleu_4 ~0.001, caption'lar anlamsiz). `dbmdz/bert-base-turkish-cased`'e
   gecildikten sonra Colab'da yeniden calistirilip `model_last.pth`/
   `model_best.pth` olustugu ve loss/Bleu_4 degerlerinin bu kez mantikli
   oldugu (ozellikle uretilen caption'larin gercek Turkce kelimelerden
   olustugu) dogrulanmali.
3. [ ] **Stage 2 pilot (MobileCLIP-S0):** Stage 1 ciktisini
   `experiments/mobileclip_s0_stage1_tasviret/model_last.pth`'e kopyalayip
   `mobileclip_s0_stage2.yaml` ile ana egitimi baslat, `eval.py` ile test
   split'inde final metrik al.
4. [ ] **S1 ve S2 deneyleri:** S0 pilotu basarili olursa ayni akisla sirayla
   calistirilacak.
5. [ ] **Hiperparametre gozden gecirme (kullanici tarafindan talep edildi,
   S0 pilotundan sonra yapilacak):** `lr`, `lr_proj`, `batch_size`,
   `max_iter`, `warm_up_iter`, `num_eval_iter` degerleri her 3 config icin
   pilot sonuclarina gore guncellenecek.
6. [ ] **Sonuc karsilastirmasi:** DINOv2-L baseline sonuclariyla BLEU/CIDEr/
   METEOR ve model boyutu karsilastirmasi yapilip tez icin tablo/grafik
   hazirlanacak.

### Colab'da Gercekten Karsilasilan ve Duzeltilen Sorunlar

- **METEOR skorlayici cokup tum egitimi durdurdu** (Colab'da, Stage 1'in ilk
  validation'inda -- iterasyon 100 -- gercekten alindi):
  `pycocoevalcap`'in METEOR sarmalayicisi bir Java alt-surecine (`meteor-*.jar`)
  yaziyor/okuyor; bazen (ozellikle Stage 1'deki gibi hemen hic egitilmemis,
  bozuk/tekrarli caption'lar uretilen bir modelde) tek bir float yerine
  bosluklarla ayrilmis birden fazla sayi iceren bir satir donduruyor, bu da
  `ValueError: could not convert string to float` ile trainer.py'nin `eval()`
  cagrisini ve dolayisiyla tum egitim surecini cokertiyordu. Bu, MobileCLIP'e
  ozgu degil; `eval.py`'deki paylasilan `evaluate_on_coco_caption` fonksiyonu
  DINOv2 baseline'i da dahil her config icin ayni riski tasiyordu, sadece
  simdiye kadar tetiklenmemisti. Duzeltme: `eval.py`'de her scorer'in
  `compute_score` cagrisi artik ayri ayri `try/except` ile sariliyor; bir
  scorer (ozellikle METEOR) hata verirse o metrik(ler) icin `0.0` ile devam
  ediliyor, egitim cokmeden surmeye devam ediyor. `Bleu_4` (target_metric)
  listede METEOR'dan once hesaplandigi icin bu degisiklik onu etkilemez.
- **`gzip.BadGzipFile: Not a gzipped file` (`Model/clip/simple_tokenizer.py`)**
  (Colab'da preflight sirasinda gercekten alindi): `Model/clip/bpe_simple_vocab_16e6.txt.gz`
  repo `.gitattributes`'ina gore Git LFS ile tutuluyor. Bu dosya, hangi
  encoder secilirse secilsin (`mobileclip` dahil) `Model/clip/clip.py` her
  zaman modul seviyesinde import edildigi icin her calistirmada gerekli.
  Colab'in duz `git clone`'u LFS farkinda olmadigi icin gercek binary yerine
  kucuk bir pointer metni indiriyor, gzip acilirken patlıyor. Duzeltme: kod
  degil, `COLAB_MOBILECLIP_HIBRIT.md`'ye klonlama sonrasi
  `apt-get install git-lfs && git lfs install && git lfs pull` adimi eklendi.
  Bu, MobileCLIP'e ozgu degil, reponun genel bir Colab-klonlama tuzagi;
  ayni sorun `COLAB_TASVIRET_BASELINE.md` akisinda da teorik olarak var ama
  o dosyaya dokunulmadi (kapsam disi, sadece MC0/hibrit rehberi guncellendi).
- **`ModuleNotFoundError: No module named 'open_clip'`** (Colab'da adim 3'te
  gercekten alindi): `pip install --no-deps git+...ml-mobileclip.git` sonrasi
  `import mobileclip` basarisiz oluyordu. Neden: `mobileclip/__init__.py`,
  vision-tower disindaki metin/tokenizer kodu icin modul seviyesinde
  `import open_clip` yapiyor; biz o kod yolunu hic kullanmasak da paket
  import edilirken bu satir calisiyor. Duzeltme: `open-clip-torch>=2.20.0`
  `requirements_colab.txt`'e normal (yani `--no-deps` olmayan) bir satir
  olarak eklendi; `datasets`/`clip-benchmark` gibi gercekten gereksiz agir
  bagimliliklar hala `--no-deps` sayesinde disarida birakiliyor. Bu, MC0
  dalina ayri bir commit ile push edildi.

### Acik Sorular / Karar Bekleyenler / Riskler

- `Model/mobileclip/mobileclip_encoder.py` icindeki `backbone.forward_embeddings
  -> forward_tokens -> conv_exp` cagri zinciri ve kanal boyutlari (s0/s1: 1024,
  s2: 1280 @ 256x256) `apple/ml-mobileclip` reposunun `main` branch kaynak
  kodunun okunmasina dayaniyor (WebFetch ile dogrulandi); `import mobileclip`
  ve paketin kurulumu artik Colab'da gercekten calisti (yukaridaki open_clip
  duzeltmesinden sonra), ama `MobileCLIPEncoder.forward()`'in gercek
  checkpoint'lerle (s0/s1/s2) beklenen sekli urettigi henuz `preflight_mobileclip.py`
  ile dogrulanmadi. Bir sonraki Colab calistirmasi bunu netlestirecek.
- `pip install --no-deps git+...` sonrasi `timm` surumunun mobileclip'in
  kayitli `mci0/mci1/mci2` model isimlerini taniyip tanimadigi kontrol
  edilmedi (repo `timm>=0.9.5` istiyor, `requirements_colab.txt`'e ayni
  alt sinir eklendi, ama Colab'in mevcut `timm` surumuyle celiski olabilir).
- Bu dizin icin ayri bir git deposu baslatilip baslatilmayacagi (`git init`)
  kullanicidan onay bekliyor — henuz yapilmadi. (Not: kod artik ayrica
  `2025tasviret_upd` reposunun `MC0` dalina da push edildi; bkz. `baslangic.md`.)
- SigLIP-Base (opsiyonel 4. deney) zaman/kaynak durumuna gore karar
  verilecek.

## Nasil Devam Edilir

Yeni bir oturuma baslarken: once bu dosyayi, sonra `plan.md` ve
`baslangic.md`'yi oku; "Sirada" listesindeki ilk isaretlenmemis maddeden
devam et. Bir madde tamamlaninca burada `[x]` olarak isaretle ve gerekirse
yeni alt maddeler ekle.
