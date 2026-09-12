# Amac

## Genel Amac

TRCaptionNet++ tabanli Turkce goruntu altyazilama (image captioning)
mimarisinde, gorsel encoder olarak kullanilan buyuk DINOv2-L (~300M parametre)
modelinin, cok daha kucuk ve mobil/edge dostu MobileCLIP varyantlariyla
(S0/S1/S2, ~11M-36M parametre) degistirilebilirligini arastirmak.

## Hipotez

Uygun bir projeksiyon/adaptasyon katmani (2 katmanli MLP) ve iki asamali bir
on-hizalama + ince ayar egitim stratejisi kullanildiginda, MobileCLIP gibi
10-40 kat daha kucuk bir gorsel encoder, DINOv2-L tabanli sisteme yakin
altyazilama kalitesi (BLEU-4, CIDEr, METEOR gibi metriklerle olculen)
saglayabilir; bu da modelin toplam boyutunu ve cikarim maliyetini onemli
olcude azaltir.

## Neden Onemli

- DINOv2-L, tek basina TRCaptionNet++ modelinin agirliginin buyuk bir
  kismini olusturuyor; bu da modeli mobil/kisitli kaynakli cihazlarda
  calistirmayi zorlastiriyor.
- MobileCLIP, Apple tarafindan tam da mobil dagitim icin tasarlanmis,
  gorsel-dil hizalamasi onceden ogrenilmis bir aile; bu da yeni encoder'in
  sifirdan degil, guclu bir on-egitimden baslamasini saglar.
- Encoder'i degistirirken dil tarafina (BERTurk decoder) ve genel egitim
  altyapisina dokunmadan, sadece gorsel-dil kopru katmanini (projeksiyon)
  yeniden tasarlamak, degisikligin etkisini izole bicimde olcmeyi mumkun
  kilar — bu da tez icin temiz bir ablasyon/karsilastirma zemini sunar.

## Kapsam Disi

- BERTurk decoder mimarisinde veya tokenizasyonda degisiklik yapilmayacak.
- Egitim/degerlendirme script'lerinin (train.py, trainer.py, eval.py) genel
  akisi korunacak; sadece encoder secimi, projeksiyon katmani ve iki asamali
  egitim mantigi icin genisletilecek.
- Veri seti (TasvirEt) ve degerlendirme metrikleri (COCO-caption metrikleri)
  degistirilmeyecek.
