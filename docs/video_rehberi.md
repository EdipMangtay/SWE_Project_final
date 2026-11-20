# SentinelAI sunum ve çekim rehberi
## Hocaya teslim edeceğin üç şey
Ali Edip Mangtay · 210911044 · SWE015 · Son teslim 22 Kasım 2026 Pazar

**1. PDF sunum.** Hoca 8–12 slayt istiyor. Hazır sunum 10 slayt. Kapakta adın, numaran, seçtiğin Use Case 4, GitHub ve video bağlantıları bulunmalı. Kayıt yüklenince kapaktaki video alanını tamamla.

**2. Sesli ekran kaydı.** Hoca ilk iki dakikayı izliyor. Uygulama açıkken iki yeni mesajı klavyeden gir. Bunlardan biri zor bir örnek olsun. Ardından mimari şemayı, dört modelin sonuç tablosunu ve çıkarımını göster. Aşağıdaki İngilizce metin yaklaşık iki dakikadan kısa bir anlatım için yazıldı.

**3. Kod deposu.** Hocanın açabildiği GitHub veya GitLab bağlantısını kapağa ve teslim formuna ekle. README, bağımlılıklar ve çalıştırma komutu depoda bulunmalı. Hazırlanan İngilizce teknik rapor ek materyaldir. Zorunlu üç çıktı sunum, video ve kod deposudur.

### On slaytın görevi
| Slayt | İçerik | Ekranda neyi bulacaksın? |
|---|---|---|
| 1 | Kapak | Adın, numaran, Use Case 4 ve teslim bağlantıları |
| 2–3 | Problem ve veri | Kimin için, 50 ms hedefi, Jigsaw verisi ve ayrımlar |
| 4 | Mimari | Mesajdan modele, oradan skora, maskeye ve süre kaydına giden yol |
| 5–7 | Yöntem | A–D teknik haritası, dört modelin eğitimi ve INT8 dönüşümü |
| 8 | Sonuç tablosu | Her modelin macro-F1, accuracy, MB ve ortalama CPU süresi |
| 9–10 | Sonuç ve kaynaklar | Gerçek bir hata, sonraki deney, yapay zekâ kullanımı ve kaynakça |

### Kayıttan önce
Mac’te sohbeti aç. Boş ekran ve **Multilingual BERT · INT8** seçili olsun. İstersen iki örneği bir kez dene, **Compare all four models** düğmesine bas, sonra **Clear session** ile sıfırla. Sunumu iki ayrı sekmede hazırla; birinde 4., diğerinde 8. slaytı açık bırak. Anlatım kartını telefonundan okuyabilirsin.

**Kayıt.** Command + Shift + 5 ile araç çubuğunu aç. Tüm ekranı kaydetmeyi seç. Seçenekler menüsünde Mac’in mikrofonunu işaretle. Sesini kısa bir deneme kaydında dinle. Sohbet sekmesine dönüp Kaydet’e bas. [Apple ekran kaydı kılavuzu](https://support.apple.com/en-us/102618)

<!-- page -->
## Çekim sırasında okuyacağın metin
Türkçe satırlar ekranda yapacağın hareketlerdir. Yalnızca İngilizce satırları seslendir. Tuşlara basarken kısa bir duraklama bırak. Saatler yaklaşık geçiş zamanlarıdır.

### 0:00–0:12  ·  Boş sohbet ekranı
Sol tarafta aktif INT8 modelini göster. Kameranı açman gerekmiyor.

**“Hi, I’m Ali Edip Mangtay. This is SentinelAI. It checks English game chat and hides offensive messages. The model runs on the CPU.”**

### 0:12–0:32  ·  Temiz mesaj
`Thanks for your help!` yaz ve Enter’a bas. Mesaj ekranda kalınca CLEAN etiketini ve sol taraftaki süreyi işaret et.

**“I’ll send a friendly message. The model marks it clean, so we can still read it. The sidebar shows how long the decision took.”**

### 0:32–0:58  ·  Zor örnek ve karşılaştırma
`You are a stupid id1ot.` yaz. “id1ot” içindeki karakter 1 rakamıdır. Enter’a bas. **Message hidden** ve **OFFENSIVE** görününce **Compare all four models** düğmesine tıkla. Tabloyu göstermek için gerekirse aşağı kaydır.

**“Now I’ll type an insult with the number one in the word. The model flags it and hides the text. Here are all four models on that same message.”**

### 0:58–1:18  ·  Mimari, slayt 4
İmleci soldan sağa hareket ettir. Chat input → model → threshold and mask → Streamlit session yolunu göster.

**“The app checks the input. BERT turns the message into tokens and gives it a score. If the score reaches the threshold, the app hides the message.”**

### 1:18–1:43  ·  Sonuç tablosu, slayt 8
En alttaki INT8 satırını göster. Macro-F1, Size ve Mean sütunlarına işaret et. 35 ms bu tablodaki ortalamadır.

**“I tested all four models on the same five hundred comments. INT eight had the best macro F-one, about zero point eight three. It used less space than full BERT and averaged thirty-five milliseconds.”**

### 1:43–1:55  ·  Kapanış, slayt 9
8. slayttan bir sayfa ilerle. 16 kaçırılan mesajı göster. Son cümleden sonra menü çubuğundaki Durdur düğmesine bas.

**“It met the fifty-millisecond target, but it missed sixteen offensive comments. Next, I would test it on real gaming conversations. Thank you.”**

<!-- page -->
## Kayıttan sonra
**İzle.** Dosyayı baştan sona oynat. İki mesajın canlı yazıldığı, ikinci mesajın maskelendiği ve sesinin duyulduğu açık olmalı. 4. slayttaki mimari ile 8. slayttaki tabloyu iki dakikadan önce göstermiş ol.

**Yükle.** Videoyu YouTube’a “Liste dışı” veya Google Drive’a “Bağlantıya sahip herkes görüntüleyebilir” seçeneğiyle yükle. Çıkış yapılmış bir gizli pencerede videonun oynadığını ve GitHub deposunun açıldığını dene.

**Kapağı tamamla.** Video bağlantısını paylaşınca PDF güncellenebilir. Kendin yapmak istersen proje kökündeki `docs/submission.json` dosyasında `video_url` alanına gerçek bağlantıyı yaz. Python ortamı açıkken `python -m scripts.presentation` çalıştır. Yenilenen dosya `docs/presentation.pdf`. Kapaktaki iki bağlantıya da tıkla ve PDF’yi güncel sürümüyle depoya gönder.

**Teslim et.** Dersin teslim formuna güncellenmiş PDF’yi ekle. Kod deposu ve video bağlantısını formdaki ilgili alanlara koy. Hoca ek dosyaya izin veriyorsa teknik raporu da ekleyebilirsin. [SWE_Project_final kod deposu](https://github.com/EdipMangtay/SWE_Project_final)

### Hoca sorarsa kısa cevaplar
**“Neden macro-F1?”** Testte 450 temiz ve yalnızca 50 saldırgan mesaj var. Her mesaja CLEAN demek bile %90 doğruluk verir. Macro-F1, iki sınıfın F1 puanına eşit ağırlık verir.

**“INT8 ne kazandırdı?”** Boyut 714,36 MB’tan 182,93 MB’a indi. Bu Mac’te tam hassasiyetli BERT daha hızlıydı. INT8 ortalamada 35,04 ms ile 50 ms hedefini yine de karşıladı.

**“Skor neyi gösteriyor?”** Model saldırganlık için bir skor üretir. INT8 eşiği 0,575. Uygulama eşik aşılınca metni maskeler. Arayüzdeki güven puanı modelin seçilen sınıfa verdiği skordur.

**“Nerede hata yaptı?”** Testteki 50 saldırgan yorumun 34’ünü yakaladı, 16’sını kaçırdı. 450 temiz yorumun 15’ini de gereksiz yere gizledi. Slayt 9 dolaylı düşmanlık içeren gerçek bir kaçırma örneğini gösteriyor.
