# 1 dakika 55 saniyede SentinelAI
İkinci bir ekranda veya telefonda bu sayfayı açık tut. Orta sütun elinin yapacağı hareketi, sağ sütun seslendireceğin İngilizce cümleyi gösterir. Yazarken susup Enter’dan sonra konuşabilirsin.

| Süre | Ekranda ne yapacaksın? | İngilizce ne söyleyeceksin? |
|---|---|---|
| 0:00–0:12 | Boş sohbetle başla. Soldaki **Multilingual BERT · INT8** seçimini göster. | Hi, I’m Ali Edip Mangtay. SentinelAI checks English game chat and hides offensive messages. It runs on the CPU. |
| 0:12–0:23 | **Type a message…** kutusuna `Thanks for your help!` yaz. **Enter**’a bas. | I’ll start with a friendly message to a teammate. |
| 0:23–0:32 | İlk kartta **CLEAN** etiketini ve okunabilen mesajı işaret et. | The model marks it clean, so the message stays visible. |
| 0:32–0:44 | Aynı kutuya `You are a stupid id1ot.` yaz. **1** rakamını göster, sonra **Enter**’a bas. | I’ll try an insult with a number in the middle of the word. |
| 0:44–0:54 | İkinci kartta **OFFENSIVE**, **Message hidden** ve **confidence** alanlarını göster. | The model still flags it. The chat hides the words and shows the confidence score. |
| 0:54–1:05 | Sağda **Compare all four models**’a tıkla. **4 flagged · 0 clean** yazısını ve dört satırı göster. | All four models flagged that same message. Their confidence scores and response times differ. |
| 1:05–1:13 | Solda **Response time log**’a tıkla. İki mesajın ayrı sürelerini göster. | The sidebar records how long each message took. |
| 1:13–1:28 | Sunum sekmesine geç. **4** numaralı slaytta girdiden INT8 modele ve maskeleme kararına doğru ilerle. | The tokenizer splits the text into pieces. BERT scores the message. If the score reaches the threshold, the app hides it. |
| 1:28–1:45 | Üstten **8**’e tıkla. INT8 satırında **0.8262** ve **35.04 ms** değerlerini işaret et. | I tested four models on the same five hundred comments. INT eight had the best macro F-one, about point eight three. It averaged thirty-five milliseconds, below the fifty-millisecond target. |
| 1:45–1:55 | Üstten **9**’a tıkla. **16 missed** sayısını göster. Son cümleden sonra menü çubuğundaki **Durdur** düğmesine bas. | It still missed sixteen offensive comments. Next, I’d test indirect insults in real game chat. Thank you. |

<!-- page -->
# Kayda hazırlanırken
**1. Uygulamayı ve sunumu aç.** Mac’te proje klasöründeki `Demoyu_Ac.command` dosyasına çift tıkla. Terminal açılırsa açık bırak. Uygulama ve sunum Safari’de açılır. Dosya açılmazsa proje klasöründe Terminal açıp `zsh Demoyu_Ac.command` çalıştır. Sunum sekmesinin üstünde **4**’ü seç, sonra uygulamaya dön.

**2. Ekranı hazırla.** Tarayıcı penceresini büyüt. Solda **Active model**, ortada **Message review**, sağda **Model comparison** görünmeli. Dar pencerede karşılaştırma altta yer alır. Sol panel kapalıysa sol üstteki açma düğmesine bas. Model olarak **Multilingual BERT · INT8** seç.

**3. Bir kez prova et.** Bir deneme mesajı gönder, **Compare all four models**’a bas ve sonuçları bekle. **Clear session**’a bas. Başlangıçta **Reviewed 0**, **Flagged 0** ve boş mesaj kutusunu gör. Kayıtta iki mesajı da yeniden klavyeden yaz.

**4. Sesini kontrol et.** Bildirimleri sessize al. **Command + Shift + 5** tuşlarına bas. **Tüm Ekranı Kaydet** simgesini seç. **Seçenekler → Mikrofon** altında Mac mikrofonunu işaretle. Beş saniyelik bir deneme çekip sesini dinle. Ardından aynı menüden asıl kaydı başlat. Ödev, ekran görüntüsüyle birlikte sesli anlatım istiyor. Yüz kamerası şartı belirtilmiyor.

### Hangi değerleri söyleyeceksin?
**Canlı örnekler.** Bu sürümde provalanan ilk mesaj CLEAN ve %100,0, ikinci mesaj OFFENSIVE ve %98,5 çıktı. Yüzdeler arayüzde yuvarlanır. Canlı CPU süreleri her denemede değişebilir. Kartlarda sayıları işaret etmen yeterli.

**Slayt 8.** 500 yorum sınıflandırma testine aittir. 35,04 ms, Apple M3 Pro’da INT8 için ayrı ölçülen 150 zamanlamanın ortalamasıdır. “Thirty-five milliseconds” cümlesini bu slayta geldiğinde söyle. INT8 model boyutu 182,93 MB, tam hassasiyetli BERT 714,36 MB. INT8’in kazancı bu deneyde boyut ve macro-F1 oldu.

**Slayt 9.** 50 saldırgan yorumun 34’ü yakalandı, 16’sı kaçtı. 450 temiz yorumun 15’i gereksiz yere gizlendi. Bir limitasyon ve mantıklı bir sonraki deney anlatmak için bu slaytı kullan.

### Teslimden önce
Kaydı oynat. İki yeni mesajın yazılması, sesin, mimari ve sonuç tablosu ilk iki dakikaya sığmalı. Videoyu YouTube’a **liste dışı** veya Drive’a **bağlantıya sahip herkes görüntüleyebilir** olarak yükle. Video ve depo bağlantılarını gizli pencerede dene. Gerçek video bağlantısını `docs/submission.json` içindeki `video_url` alanına girip proje ortamında `python -m scripts.presentation` çalıştır. **Yenilenen dosya docs/presentation.pdf**. Güncellenen PDF’yi ve bağlantıyı depoya gönder. Bu PDF’yi ve iki bağlantıyı dersin teslim formuna ekle. R8 gereği 10. slayttaki AI kullanım açıklamasını koru.
