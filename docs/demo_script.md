# Live demo · target 1 minute 50 seconds

| Time | Screen | Say |
|---|---|---|
| 0:00–0:15 | Start with the empty chat, INT8 selected. | “Hi, I'm Ali Edip Mangtay. SentinelAI helps a gaming community moderate English messages. It decides whether to show a message or hide it, and records how long the decision takes.” |
| 0:15–0:38 | Type **Thanks for your help!** and send it. Point at the score and the sidebar. | “I'll type a friendly message. The model marks it clean, so the text stays visible. Here is its confidence estimate. The sidebar shows the time for this message.” |
| 0:38–1:08 | Type **You are a stupid id1ot.** Compare all four models. | “Now I'll change one letter in an insult to a number. The quantized model flags this message and hides the text. I can compare all four models on the same input. Their confidence and time can differ.” |
| 1:08–1:29 | Show slide 4. | “The app checks the input and sends it to the selected model. The classical models use normalized words. BERT uses its own tokenizer. After prediction, a threshold controls whether the message is masked.” |
| 1:29–1:48 | Show slide 8, point to the BERT rows. | “I tested all four models on the same five hundred comments. INT8 had the highest macro-F1 here, about zero point eight three. It used about one quarter of the storage. Its average time was thirty-five milliseconds.” |
| 1:48–1:56 | Show slide 9, then finish. | “It met my fifty-millisecond target, but missed sixteen offensive comments. I would test real gaming messages next. Thank you.” |

These two typed messages were checked against the saved INT8 model. Use the scores and times shown on your screen. The 35 ms result is the average from the formal benchmark.
