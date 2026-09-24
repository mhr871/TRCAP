# TRCaptionNet++ -- Projection Adapter Deney Altyapısı

Bu klasör, `Program/hibrit mimariler/hibrit_0` altyapısı temel alınarak
oluşturulmuş, **Projection Adapter Ablation** deneyleri için bağımsız bir
deney ortamıdır. Amaç, aynı dondurulmuş (frozen) **DINOv2 ViT-L/14** encoder
ve aynı eğitilebilir **ELECTRA Turkish** (`dbmdz/electra-base-turkish-mc4-cased-discriminator`)
decoder sabit tutulurken, encoder ile decoder arasındaki projeksiyon
katmanının mimarisini değiştirerek P1-P7 deneylerini tek bir ortak pipeline
üzerinden tekrar üretilebilir şekilde çalıştırmaktır. Bkz.
`TRCaptionNet_Projection_Adapter_Deney_Dokumani.docx`,
`TasvirET_Deney_Tablosu.docx`, `TasvirET_Deney_Omurgasi_v1.docx`.

## Sabit bileşenler

| Bileşen | Durum | Not |
|---|---|---|
| DINOv2 ViT-L/14 | **Frozen** | `torch.hub` üzerinden pretrained ağırlıklarla yüklenir, `requires_grad=False` |
| ELECTRA Turkish decoder | **Trainable** | `transformers.ElectraForCausalLM(is_decoder=True, add_cross_attention=True)` |
| Tokenizer | Sabit | ELECTRA'nın kendi WordPiece tokenizer'ı |
| Projection Adapter | **Değişken** | `models/projection_adapters/` altında registry ile seçilir |

`ElectraForCausalLM` özellikle kullanılır: `transformers`'ın kendi ELECTRA
implementasyonu BERT gibi `add_cross_attention` destekler ve ELECTRA-base'in
`embedding_size == hidden_size` (768) olması sayesinde pretrained ağırlıklar
`electra.*` isimleriyle doğrudan yüklenir; sadece cross-attention alt
katmanları rastgele ilklendirilip eğitilir.

## Klasör yapısı

```
proj_exp/
  Model/
    TRCaptionNetPP.py        # DINOv2 -> adapter -> ELECTRA
    dino/dino.py              # frozen DINOv2 sarmalayıcı
  models/
    projection_adapters/
      base.py                 # ProjectionAdapter arayüzü + summary()
      linear.py                # P1
      mlp.py                   # P2
      residual.py               # P3
      cross_attention.py        # P4
      gated.py                  # P5
      film.py                   # P6
      bottleneck.py              # P7
      __init__.py               # ADAPTERS / ADAPTER_CODES registry
  Datasets/                  # TasvirEt dataset + transform'lar
  configs/projection_exp/    # P1-P7 için hazır yaml config'ler
  Data/tasvir-et/            # tasvir_{train,val,test}.json (kopyalandı)
  tools/download_tasviret_images.py  # Data/flickr8k/images'i indirir
  train.py                   # tek bir adapter'ı eğitir
  run_projection_experiments.py      # P1-P7'yi sırayla çalıştırır
  eval.py                    # bağımsız checkpoint değerlendirme
  trainer.py / utils.py
  runs/                      # her deneyin çıktısı (eğitimde oluşur)
```

## Kurulum

```
pip install -r requirements.txt
python tools/download_tasviret_images.py   # Data/flickr8k/images'i doldurur
```

`java` (METEOR ve PTB tokenizer için) ve internet erişimi (DINOv2/ELECTRA
ağırlıkları ilk çalıştırmada indirilir) gereklidir.

## Tek bir adapter'ı eğitmek

```
python train.py --config configs/projection_exp/P1_linear.yaml
```

`save_name` boş (`~`) bırakılırsa çalışma otomatik olarak
`runs/<Pn>_<adapter>/` altına kaydedilir (örn. `runs/P1_linear/`).

## P1-P7'yi sırayla çalıştırmak

```
python run_projection_experiments.py
python run_projection_experiments.py --save-dir /content/drive/MyDrive/TRCAP_projection_exp
python run_projection_experiments.py --adapters linear mlp residual
```

Her deney birbirinden bağımsızdır: model, optimizer, scheduler ve iterasyon
sayacı her adapter için sıfırdan kurulur; hiçbir ağırlık deneyler arasında
taşınmaz.

## Her runs/<Pn>_<adapter>/ klasöründe

| Dosya | İçerik |
|---|---|
| `config.yaml` | O çalışmanın efektif config'i |
| `train.log` | Konsol log çıktısı |
| `adapter_summary.json` | `adapter_name`, `trainable_parameters`, `total_parameters`, `input_dim`, `output_dim`, `hidden_dim`, `architecture` |
| `epoch_metrics.csv` | `epoch, train_loss, val_loss, bleu1..4, meteor, rouge, cider` -- her `num_eval_iter` sınırında bir satır |
| `metrics.json` | En güncel ve en iyi (`target_metric`'e göre) sonuçların özeti |
| `best.pth` / `last.pth` | En iyi / en güncel checkpoint (model+optimizer+scheduler state) |

## Config şeması

```yaml
model:
  dino2: dinov2_vitl14
  electra: dbmdz/electra-base-turkish-mc4-cased-discriminator
  projection_adapter: linear   # linear | mlp | residual | cross_attention | gated | film | bottleneck
  projection_adapter_kwargs: {}
  freeze_encoder: true
  train_decoder: true
  train_projection: true
```

Yeni bir adapter eklemek için sadece `models/projection_adapters/` altına bir
modül eklenip `__init__.py`'deki `ADAPTERS` / `ADAPTER_CODES` sözlüklerine
kaydedilmesi yeterlidir; `Model/TRCaptionNetPP.py`, `trainer.py` ve
`train.py` adapter adına göre dallanma (if/else) içermez, yalnızca registry
üzerinden çağırır.

## Önceki encoder/decoder deneyleriyle ilişki

Bu klasör `hibrit mimariler/hibrit_0` ve `deneyler_encoder` kod tabanlarını
bozmaz; onlardan yeniden kullanılan parçalar (TasvirEt dataset sınıfları,
DINOv2 sarmalayıcı, eğitim/loglama altyapısı) buraya adapte edilerek
kopyalanmıştır. Eski projelerdeki CLIP/MobileCLIP encoder dalları ve
`Model/bert/med.py` tabanlı BERT decoder, bu deney kapsamının dışında
olduğu için (encoder ve decoder artık sabit: DINOv2 + ELECTRA) buraya
taşınmamıştır.
