# SentinelAI'ı açma ve teslim etme

**Öğrenci:** Ali Edip Mangtay · 210911044

Mac'indeki `SWE_Chat_project` klasöründe modeller ve ortam hazır. `Baslat.command` dosyasına çift tıkla. Finder dosyayı açmazsa klasörde Terminal açıp `zsh Baslat.command` yaz. İlk ekranda **Multilingual BERT · INT8** seçili olmalı. Başka bilgisayarda ZIP'i açtıysan kurulum için Python 3.12 ve ilk seferde internet gerekir.

1. `Thanks for your help!` yaz. Mesajın göründüğünü, model skorunu ve sağlanan süreyi göster.
2. `You are a stupid id1ot.` yaz. Sabit maskeyi göster ve **Compare all four models** düğmesine bas.
3. `docs/demo_script.md` ile en fazla iki dakikalık İngilizce sesli ekran kaydını yap. Mimariyi 4. slaytta, sonuçları 8. slaytta göster.
4. `python -m scripts.source_archive` ile kod paylaşım ZIP'ini üret. ZIP'in içindeki kaynak dosyalarını GitHub/GitLab'a yükle. Eğitimli modelleri dağıtmak için taşınabilir proje ZIP'ini Google Drive veya Hugging Face'te paylaş.
5. Videoyu liste dışı YouTube veya bağlantıyla erişilen Drive videosu olarak yükle. `docs/submission.json` içine gerçek adresleri gir. `python -m scripts.presentation` ile kapağı yenile ve `python -m scripts.verify_submission --online` ile denetimi yeniden çalıştır.
6. Kod ve video bağlantılarını gizli pencerede kendin aç. Kendi sesinin duyulduğunu ve ilk iki dakikada canlı tahmin, mimari, sonuç tablosu görüldüğünü kontrol et. Son tarih **22 Kasım 2026 Pazar**.

Hocana açıklayacağın sonuç şu: INT8 model BERT'in disk boyutunu 714,36 MB'tan 182,93 MB'a düşürdü. Bu Mac'te biraz daha yavaş çalıştı; ortalama 35,04 ms ile baştan belirlediğimiz 50 ms sınırını karşıladı. 50 saldırgan test yorumunun 34'ünü yakaladı, 16'sını kaçırdı.
