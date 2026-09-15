# TRCAP -- MC1-epoch-based

`MC0` dalinin epoch-tabanli egitim altyapisina gecirilmis hali. `MC0`'a
DOKUNULMADI -- o hala eski (iterasyon-tabanli) haliyle duruyor ve
calisir durumda; bu dal ondan ayri bir git dalidir.

## Ne degisti, ne degismedi

**Degisen (sadece 2 dosya sinifi):**
- `trainer.py` -- baştan yazildi: sabit `max_iter`/`warm_up_iter`/
  `num_eval_iter` yerine gercek epoch dongusu (`DataLoader(shuffle=True,
  drop_last=True)`, veri setinin tamami her epoch'ta bir kez gezilir),
  `ReduceLROnPlateau` (validation Bleu_4 durgunlasinca LR'yi otomatik
  dusurur) ve erken durdurma (`early_stop_patience`) eklendi.
- `configs/tasviret/` -- SADECE iki yeni config var:
  `mobileclip_s0_stage1_epoch.yaml` ve `mobileclip_s0_stage2_epoch.yaml`.
  Eski (iterasyon-tabanli) config'ler bu dalda SILINDI -- yeni
  `trainer.py` onlari anlamiyor (`max_iter` gibi alanlar artik
  okunmuyor). Ihtiyac olursa orijinalleri `MC0` dalinda duruyor
  (`git show MC0:configs/tasviret/mobileclip_s0_stage2.yaml` gibi).

**Degismeyen (MC0 ile birebir ayni, `diff -rq` ile dogrulandi):**
`Model/`, `Datasets/`, `tools/` (3 bilesen testi dahil), `eval.py`,
`utils.py`, `app.py`, `transform/`. Yani daha once dogrulanmis olan
encoder/proj/decoder bilesen testleri ve eval pipeline'i HICBIR
SEKILDE degismedi.

**SCST eklenmedi.** Kullanici talebi uzerine bu tur (RL/CIDEr
optimizasyonu) bilinçli olarak kapsam disi birakildi; sadece CE-benzeri
tek asamali egitim epoch-tabanli hale getirildi.

## Neden

Kullanicinin daha once calistirdigi ResNet101+LSTM / ShuffleNetV2+GRU
deneyleri (https://github.com/mhr871/TIC.git, `ShuffleNet` dali)
saglikli bir egitim egrisiyle BLEU-4 ~22-24'e ulasmisti; o deneylerin
ortak noktasi epoch-tabanli egitim + `ReduceLROnPlateau` + erken
durdurmaydi. `MC0`'da bunlarin UCU DE yoktu -- sadece onceden
belirlenmis, validation'a hic bakmayan sabit bir 50000 iterasyonluk
program vardi. Bu, gozlemlenen "Bleu_4 egitim ilerledikce dusuyor"
belirtisini dogrudan acikliyordu.

**Onemli:** eski deneylerin SAYISAL degerleri (decoder_lr=5e-4,
grad_clip=5.0, batch_size=256, Adam, ...) BILINCLI OLARAK
kopyalanmadi -- o degerler sifirdan egitilen bir GRU decoder icin
ayarlanmisti, bizim decoder'imiz (pretrained BERT fine-tune) tamamen
farkli bir rejim. Sadece MEKANIZMA (epoch dongusu +
ReduceLROnPlateau + erken durdurma) alindi; sayisal degerler MC0'in
kendi (lr_exp1'den ONCEKI, orijinal) degerlerine yakin tutuldu. Her
sayinin gerekcesi `configs/tasviret/mobileclip_s0_stage2_epoch.yaml`
icinde yorum olarak yazili.

## Sirada

- Stage 1 (`mobileclip_s0_stage1_epoch.yaml`) once calistirilip
  `experiments/mobileclip_s0_stage1_epoch_tasviret/model_last.pth`
  uretilmeli, sonra Stage 2 (`mobileclip_s0_stage2_epoch.yaml`) o
  checkpoint'i baslangic noktasi olarak kullanir.
- Bu dal Colab'da henuz gercekten calistirilmadi.
