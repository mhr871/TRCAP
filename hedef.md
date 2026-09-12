# Hedefler ve Basari Kriterleri

## Somut Hedefler

1. **Encoder soyutlamasi:** `Model/TRCaptionNet.py` icine, mevcut `clip` ve
   `dino2` secimlerinin yaninda `mobileclip` secimini ekleyen, patch-token
   ciktisi ureten bir `MobileCLIPEncoder` sinifi yazmak (bkz. `plan.md` /
   "Ozellik Cikarimi").
2. **2 katmanli MLP projeksiyon:** Mevcut tek-katmanli/Transformer tabanli
   `Proj` sinifinin yaninda, `Linear -> GELU -> Linear` yapisinda yeni bir
   projeksiyon siniif eklemek ve config uzerinden secilebilir yapmak.
3. **Iki asamali egitim destegi:** `trainer.py` icine, Asama 1'de encoder+decoder
   dondurulup sadece projeksiyonun egitildigi bir "warmup" modu eklemek;
   Asama 2'de mevcut davranisa (encoder frozen, proj+decoder egitiliyor)
   gecmek.
4. **3 config + 3 deney:** MobileCLIP-S0, S1, S2 icin ayri config dosyalari
   olusturmak ve her biri icin TasvirEt uzerinde ucer deney (pilot, S1, S2)
   calistirmak.
5. **Colab uyumlulugu:** Yeni pipeline'in, mevcut `COLAB_TASVIRET_BASELINE.md`
   akisina benzer sekilde (checkpoint indirme, veri hazirlama, egitim,
   degerlendirme hucreleri) bir Colab/L4 notebook akisiyla calistirilabilir
   olmasi.

## Basari Kriterleri

- **Islevsel:** Her uc MobileCLIP varyanti icin egitim, hata vermeden
  baslayip en az bir validation dongusu tamamlayabilmeli (loss'un dustugu
  gozlemlenmeli — plan.md'deki "Pilot / Hizalama Testi" asamasi).
- **Performans:** S1 veya S2 varyanti, DINOv2-L baseline'inin BLEU-4/CIDEr
  metriklerine mumkun oldugunca yakin sonuc vermeli (tam esitlik sart
  degil; tez, boyut/performans odunlesimini (trade-off) tartisacak).
- **Verimlilik:** MobileCLIP varyantlari ile VRAM kullaniminda gozle
  gorulur bir dusus ve/veya daha yuksek `batch_size` ile egitim
  hizlaninda artis olcumu.
- **Tekrarlanabilirlik:** Her deneyin config'i, checkpoint'i ve sonuc
  JSON'lari (`prediction_*.json`, `result_*.json`) `experiments/` altinda
  duzenli tutulmali; hangi deneyin hangi config'ten uretildigi acik olmali.

## Olcum Metrikleri

`eval.py` / `evaluate_on_coco_caption` uzerinden zaten hesaplanan:
Bleu_1..4, METEOR, ROUGE_L, CIDEr, (opsiyonel SPICE) + `avg_caption_len` ve
`eos_rate` diagnostik degerleri. Ayrica model boyutu (parametre sayisi) ve
(mumkunse) cikarim suresi/VRAM kullanimi karsilastirmali olarak raporlanacak.
