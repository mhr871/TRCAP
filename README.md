# TRCAP

TRCaptionNet++ TasvirEt baseline fine-tuning calisma kopyasi.

Bu repo, public `TRCaptionNetpp_Large.pth` checkpoint'ini baslangic agirligi olarak kullanarak TasvirEt uzerinde orijinal mimariye dokunmadan fine-tune ve test surecini hazirlamak icin duzenlenmistir.

Colab/L4 uzerinde notebook hucreleriyle checkpoint, TasvirEt goruntuleri, tam preflight, fine-tune, resume ve final test akisi:

```text
COLAB_TASVIRET_BASELINE.md
```

Not: checkpoint, dataset ve deney ciktilari GitHub'a dahil edilmez. Bu dosyalar Colab/Drive tarafinda ayrica indirilir veya baglanir.

## Projection Adapter Deneyleri (P1-P7)

Bu klasör aynı zamanda `configs/projection_exp/P1_linear.yaml` ... `P7_bottleneck.yaml`
ve `run_projection_experiments.py` üzerinden **Projection Adapter Ablation**
deneylerini de barındırır (bkz. `Model/projection_adapters/`). Sabit encoder
MobileCLIP-S2, sabit decoder BERTurk (`dbmdz/bert-base-turkish-cased`);
değişen tek şey projeksiyon katmanı (P1 Linear, P2 MLP, P3 Residual, P4
Cross-Attention, P5 Gated, P6 FiLM, P7 Bottleneck).

**P2 (MLP) varsayılan toplu koşuda atlanır:** `hibrit_0`'da daha önce
sağlıklı/geçerli sonuçlar alınmış bir P2 koşusu zaten var; bu yüzden
`run_projection_experiments.py` argümansız çalıştırıldığında P1, P3–P7'yi
çalıştırır, P2'yi atlar. P2'yi yine de çalıştırmak istersen
`--adapters mlp` ile açıkça belirtmen gerekir.

**`proj_exp` klasöründen tek farkı:** orada `freeze_decoder: true` (decoder
donuk, yalnızca adapter eğitiliyor); burada **`freeze_decoder: false`** --
her 7 deneyde de BERTurk decoder, projection adapter ile birlikte
eğitiliyor. Diğer her şey (encoder, dataset) birebir aynı.

```
python run_projection_experiments.py --save-dir /content/drive/MyDrive/TRCAP_projection_exp_all
```

Sonuçlar `runs`/`experiments` yerine `<save-dir>/P1_linear ... P7_bottleneck`
altında, ortak `projection_results.csv` ile birlikte oluşur (bkz.
`proj_exp/README.md`'deki komutlarla aynı kullanım).

## Batch/iterasyon ölçeklemesi (40-95GB VRAM'i kullanmak için)

`batch_size=64, max_iter=16000` ile eğitim ~6GB VRAM kullanıyordu (40-95GB
seçeneklerinin çoğu boşta kalıyordu). Tüm 7 config'te **aynı oranla**
ölçeklendi -- toplam görülen görüntü sayısı (`batch_size * max_iter` =
1.024.000) ve eval oranı (warmup/eval'in `max_iter`'a oranı) korunuyor,
sadece adım sayısı azalıp adım başına paralellik artıyor:

| | Önceki | Şimdiki |
|---|---|---|
| `batch_size` | 64 | **256** |
| `max_iter` | 16000 | **4000** |
| `warm_up_iter` | 2000 | **500** |
| `num_eval_iter` | 4000 | **1000** |
| `lr` / `lr_proj` | 1e-5 / 5e-5 | **4e-5 / 2e-4** (linear scaling rule, x4) |

Mixed precision (fp16/AMP) bilinçli olarak **eklenmedi** -- P4 gibi taze
ilklenmiş cross-attention katmanlarında kararsızlık (loss spike/NaN) riski
taşıdığı için, sadece batch/lr ölçeklemesi tercih edildi (sayısal olarak
tam fp32, model/mimari değişmedi).
