# Submission audit

NOT SUBMISSION READY — CRITICAL REQUIREMENT MISSING

| Gate | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | R1 / R5 · chat accepts new input with default INT8 | PASS | Streamlit AppTest submitted a fresh message and recorded an INT8 decision. |
| 1 | HTTP application startup | PASS | Application root and health both returned HTTP 200 in process. |
| 1 | Working bert artifact | PASS | Loaded local checkpoint and ran a real CPU forward pass. |
| 1 | Working int8 artifact | PASS | Loaded local checkpoint and ran a real CPU forward pass. |
| 1 | Use Case 4 and repository source | PASS | Chat moderation UI, implementation, requirements and setup present. |
| 1 | Presentation PDF | PASS | Parsed the PDF. |
| 1 | Accessible code repository | PASS | HTTPS availability; verify anonymous access in a private browser. |
| 1 | Accessible demo video | FAIL | A valid project-specific link is missing. |
| 1 | Original results and student explanation | MANUAL CHECK | Verify experimental provenance and explain the code. Numerical consistency is checked below. |
| 1 | Video contains live input | MANUAL CHECK | Watch the recorded stream and verify that new typed input returns a decision. |
| 2 | R2 · technique map and A/B/C/D | PASS | Slide 5, trained Word2Vec and neural net, lexicon results, BERT log and INT8 checks. |
| 2 | R3 · four models and identical test IDs | PASS | Recomputed four macro-F1 scores from the ordered held-out predictions. |
| 2 | R4 · metrics, size, CPU and latency | PASS | Read slide 8, hardware, model sizes and all raw timings. |
| 2 | Mean latency budget | PASS | Compared INT8 with the predeclared 50 ms mean budget. |
| 2 | R6 · architecture diagram | PASS | Slide 4 embeds the exported diagram. |
| 2 | Architecture matches the complete demo | MANUAL CHECK | Trace an actual message through the shown preprocessing branch, model, mask and timing log. |
| 2 | R7 · installation and launch instructions | PASS | README documents the Python environment, model restoration and one run command. |
| 2 | R8 · AI usage slide | PASS | Slide 10 names the assistant and how it was used. |
| 2 | PDF structure and length | PASS | Ten-page Introduction / Development / Conclusion deck. |
| 2 | Cover identity and both external links | FAIL | Evidence absent or inconsistent. |
| 2 | End-to-end test evidence | PASS | Executed pytest JUnit XML records model and Streamlit integration tests. |
| 2 | Recording guide and narration | PASS | Timed script and checklist present. |
| 2 | Narration, architecture, results and duration | MANUAL CHECK | Play the final upload, verify sound and a runtime of 2:00 or less. |
| 2 | Fresh remote setup and anonymous access | MANUAL CHECK | Use the published repo in a fresh checkout and open the final video and repo in a private browser. |
