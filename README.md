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

**`proj_exp` klasöründen tek farkı:** orada `freeze_decoder: true` (decoder
donuk, yalnızca adapter eğitiliyor); burada **`freeze_decoder: false`** --
her 7 deneyde de BERTurk decoder, projection adapter ile birlikte
eğitiliyor. Diğer her şey (encoder, hiperparametreler, `max_iter: 16000`,
`batch_size: 64`, dataset) birebir aynı.

```
python run_projection_experiments.py --save-dir /content/drive/MyDrive/TRCAP_projection_exp_all
```

Sonuçlar `runs`/`experiments` yerine `<save-dir>/P1_linear ... P7_bottleneck`
altında, ortak `projection_results.csv` ile birlikte oluşur (bkz.
`proj_exp/README.md`'deki komutlarla aynı kullanım).
