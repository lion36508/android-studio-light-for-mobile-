# Q-Mini Light 0.7M — परियोजना इतिहास और सफ़ाई ऑडिट

- **ऑडिट की तारीख:** 19 नवम्बर 2026
- **जाँची गई शाखा:** `main`
- **मूल commit:** `da4df686b8d1073c43b26ad9e6e02c0278ac622c`
**उद्देश्य:** पुराने Q-Mini Light को सुरक्षित रखते हुए असली training data, test-only synthetic data, production runtime और Add/Sub model को अलग-अलग पहचानना।

**वर्तमानता नोट (25 नवम्बर 2026):** यह तारीख़वार ऐतिहासिक audit है। पुराने profile/config निर्णय उस समय की स्थिति बताते हैं; मौजूदा scope के लिए root README और वर्तमान architecture/training docs मान्य हैं। 25 नवम्बर cleanup ने inactive capacity profiles और speculative planning documents हटाए हैं।

## 1. सबसे ज़रूरी निष्कर्ष

1. **OASST1 data-preparation में synthetic-data filter छूटा हुआ था।** नए V12 pipeline की `scripts/prepare_conversation_datasets.py` में `synthetic`, `review_result`, `tree_state` और assistant `rank` को required/filter नहीं किया जा रहा था। पुराने Personal Assistant notebook में ये filters मौजूद थे। इस वजह से नए pipeline में model-generated rows, unreviewed rows या low-ranked alternative replies आ सकते थे। इस branch में strict filters और regression tests जोड़े गए हैं।
2. **V12 notebook पुराना source revision pin कर रही थी।** Notebook `e8ee0a1779443cf19cf336e56495cb0e5ca887e8` fetch करती थी। उस revision में DailyDialog के train/test SHA-256 strings क्रमशः 61 और 63 hexadecimal characters के थे, जबकि SHA-256 64 characters का होता है। 18 नवम्बर के main commit `da4df686...` में hashes ठीक थे, पर V12 notebook पुरानी revision पर थी। इस branch में V12 pin को सही commit पर लाया गया है।
3. **पुराने Q2-v2 model में पाँच-सिंबल या zero-symbol codebook नहीं है।** मौजूदा production matrix symbols `{-1.0, -0.5, +0.5, +1.0}` हैं। Non-zero row के भीतर exact zero weight को zero के रूप में store नहीं किया जा सकता; all-zero row का scale zero होने पर ही row शून्य बनती है।
4. **पुराने runtime में multiplication सचमुच इस्तेमाल होती है।** `qmini/model.py`, `qmini/packed_runtime.py` और `qmini/native_kernel.cpp` में linear/dot-product, convolution, state-space update, gating और scale operations multiplication करते हैं। पुराने model को Add/Sub-only बताना गलत होगा।
5. **पुराने model को बदले बिना नया model अलग रखना ज़रूरी है।** इस branch में `qmini_addsub/` को अलग architecture line के रूप में परिभाषित किया गया है। अभी यह specification है, trained model या चालू runtime नहीं।

## 2. Fake data की सही पहचान

हर synthetic sample खराब नहीं होता। उसका इस्तेमाल कहाँ हो रहा है, यह निर्णायक है।

| स्रोत | क्या data है | सही फैसला |
|---|---|---|
| `scripts/learning_proof.py` | दो token IDs से बना छोटा deterministic toy task | CI-only; इसे language training data या assistant-quality proof न कहें |
| `scripts/benchmark.py` | random/synthetic token IDs | speed microbenchmark only; real prompt benchmark का विकल्प नहीं |
| `qmini/preflight.py` | random token IDs से model graph check | development preflight; training corpus नहीं |
| `tests/` | छोटे synthetic text, random IDs और fixtures | आवश्यक regression tests; इन्हें हटाने से tests कमजोर होंगे |
| पुराने TinyStories notebooks | synthetic English stories | पुराने experiment; मौजूदा V12 conversation-training का source नहीं |
| Mistral distillation notebook | teacher-generated synthetic replies | experimental; production human-data corpus में शामिल नहीं |
| OASST1 parquet | human-reviewed conversation corpus, जिसमें `synthetic` flag भी मौजूद है | केवल `synthetic is False`, `review_result is True`, `deleted is False`, English और ready-for-export rows स्वीकार करें |

**निष्कर्ष:** test fixtures को production dataset से अलग रखें; उन्हें चुपचाप training pipeline में कभी न जोड़ें। पुरानी synthetic-data notebooks को active training route के रूप में advertise न करें।

## 3. Data pipeline में सुधार

### OASST1

अब तैयारी script में निम्न fields required हैं:

- `review_result`
- `deleted`
- `synthetic`
- `rank`
- `tree_state`

अब केवल वही row स्वीकार होगी जो:

- English हो (`lang == "en"`)
- `tree_state == "ready_for_export"` हो
- `review_result is True` हो
- `deleted is False` हो
- `synthetic is False` हो
- assistant reply के लिए `rank == 0` हो; missing rank भी अस्वीकार होगा

Missing safety flags fail-closed हैं। Dropped rows के कारण अलग-अलग counters रखे जाएँगे ताकि audit में पता चल सके कि कितना data synthetic, unreviewed, deleted, non-English या low-ranked होने के कारण बाहर हुआ।

Source references:
- [OpenAssistant/oasst1 dataset card](https://huggingface.co/datasets/OpenAssistant/oasst1)
- [Pinned OASST1 revision used by the pipeline](https://huggingface.co/datasets/OpenAssistant/oasst1/tree/fdf72ae0827c1cda404ff25b6603abec9e3399b)

### DailyDialog

DailyDialog train/validation/test ZIPs के लिए exact size और SHA-256 pins रखे गए हैं। Dataset license `CC-BY-NC-SA-4.0` है; commercial distribution से पहले license conditions की जाँच आवश्यक है।

## 4. पुराने model की वास्तविक सीमाएँ

| क्षेत्र | मौजूदा Q-Mini Light |
|---|---|
| Parameters | 705,742 estimated |
| Vocabulary | 4,096 |
| Width / hidden | 64 / 128 |
| Main blocks | 6 |
| Experts | 2, Top-1 |
| Context | 512 tokens |
| Sequence mixing | causal depthwise convolution + gated diagonal SSM |
| Matrix weight symbols | `-1, -0.5, +0.5, +1` |
| Production inference | multiplication/dot-product/convolution/gating सहित mixed arithmetic |
| Android support | पूर्ण native Android runtime अभी नहीं |

यह codebase एक research baseline है; यह अपने आप में ChatGPT-जैसा, production-quality assistant सिद्ध नहीं करता।

## 5. CI और वास्तविक गुणवत्ता का अंतर

जाँचा गया अंतिम main run:

- [Q-Mini CI run #145](https://github.com/lion36508/Q-mini-light-0.7M/actions/runs/44347686739)
- commit: `da4df686b8d1073c43b26ad9e6e02c0278ac622c`
- परिणाम: **286 passed, 5 skipped**
- skips: CPU runner पर CUDA stress tests, NEON backend test और pyarrow-dependent tests

उसी run में controlled smoke/evaluation artifact के task pass rates शून्य दिखे। यह छोटे smoke/evaluation सेट की चेतावनी है, किसी वास्तविक Drive checkpoint का comprehensive benchmark नहीं। Green CI केवल code/tests की स्थिति बताता है; इससे assistant की गुणवत्ता सिद्ध नहीं होती।

**इस ऑडिट में Google Drive के निजी training logs/checkpoints उपलब्ध नहीं थे।** इसलिए real training loss, actual optimizer steps, trained checkpoint का NLL या वास्तविक model replies के बारे में कोई नया दावा नहीं किया गया।

## 6. 12–19 नवम्बर 2026: बदलावों का इतिहास

| तारीख | मुख्य काम | मिली गलती / नतीजा |
|---|---|---|
| 12 नवम्बर | बड़े Q-Mini source से Light 0.7M profile अलग किया; config, paths और package defaults सेट किए | कई पुराने build/runner notebook defaults आपस में असंगत थे |
| 13 नवम्बर | one-click notebook और build-only flow | notebook quoting, empty cells, repository URL और package-path मुद्दे मिले; tests जोड़े गए |
| 14 नवम्बर | English training, Hindi variants, CPU/GPU setup, tokenizer और mixed precision | T4 पर BF16/dtype mismatch, tokenizer SQLite thread migration और dependency/runtime setup के मुद्दे सामने आए |
| 15 नवम्बर | T4 tuning, resume और checkpoint flow | physical batch बदलने पर resume contract mismatch; checkpoint migration और Drive timeout safeguards जोड़े गए |
| 16 नवम्बर | Drive lookup/mount, CPU-only options, English SFT और assistant end-token supervision | कई notebooks में newline/escape/Drive fallback glitches; SFT data reload/resume और end-token supervision सुधारना पड़ा |
| 17 नवम्बर | DailyDialog + OASST1 conversation data की ओर बदलाव; V11/V12 CPU resume | CLI/train indentation errors, notebook JSON quoting errors और dataset switch logic टूटे; कई failing CI runs के बाद सुधार हुए |
| 18 नवम्बर | DailyDialog train/test SHA-256 pins ठीक किए; V12 source pin | main में hashes सही हुए, लेकिन V12 पुरानी commit pin करती रही; इस branch में pin अपडेट किया गया |
| 19 नवम्बर | data-quality audit और Add/Sub v1 का अलग contract | OASST1 synthetic/review/rank filters की कमी मिली; अलग Add/Sub architecture line तय की गई |

यह इतिहास Git commit और CI evidence पर आधारित है; Drive में क्या-क्या training सचमुच चली, उसकी निजी logs के बिना वह हिस्सा अप्रमाणित रहेगा।

## 7. Project में क्या रखना है और क्या अलग करना है

### पुराने Q-Mini Light में सुरक्षित रखना

- `qmini/config.py`, `model.py`, `tokenizer.py`, `data.py`, `train.py`, `generate.py`
- पुराने Q2-v2 export/runtime और उसके checkpoints
- पुराने tokenizer, dataset manifests और verified Drive artifacts
- पुराने model के tests और docs

### Development-only रखना

- synthetic learning proof
- synthetic-token benchmark
- architecture preflight
- quantization ablation
- test fixtures

इनको product inference या production training से अलग label करें; बिना उपयोग जाँचे delete न करें।

### Legacy/experimental notebooks

`colab/` में अब केवल दो model lines के तीन सक्रिय entry points रखे गए हैं: legacy Q-Mini का V12 CPU trainer और existing-Drive CPU chat demo, तथा Add/Sub v1 का from-scratch notebook। बाकी 21 historical notebooks को उनके मूल contents/blob IDs बदले बिना `colab/archive/legacy-experiments/` में स्थानांतरित किया गया है। इससे पुराने experiments और Git history सुरक्षित हैं, पर नए users को duplicate training routes नहीं दिखते। Archive index में हर moved notebook सूचीबद्ध है।

### वास्तविक training scope से बाहर

उस समय `qmini/distributed.py` resource planning देता था; यह FSDP/ZeRO trainer नहीं था। वह planning module बाद में Light scope से हटाया गया। Android ARM64 cross-compile पूर्ण Android inference app के बराबर नहीं है।

## 8. Add/Sub v1 का अलग नियम

नया model `qmini_addsub/` में है; पुराना `qmini/` package अपने पुराने format में सुरक्षित है। नया Add/Sub model अलग config, code, tokenizer, data manifest, training checkpoints और Drive folder इस्तेमाल करता है। यह पुराने checkpoint को स्वीकार नहीं करता।

वर्तमान Add/Sub v1 training architecture: RMSNorm, dilated causal depthwise convolution (dilation 1/2/4/8), input-dependent gated diagonal state-space recurrence with parallel scan और sparse Top-1 experts. Default profile में vocab 512, width 96, चार blocks, दो experts, hidden 256 और context 512 है; instantiated parameter count 711,528 है।

नया weight alphabet:

`{-1.5, -0.5, 0, +0.5, +1.5}`

पाँच अलग symbols को fixed-width storage में कम-से-कम **3 bits per weight** चाहिए; 2 bits में पाँच states encode नहीं हो सकते। नए inference path में कोई multiplication/matmul/dot-product/convolution या runtime scale multiplication नहीं होगा। Contribution lookup के बाद numerical accumulation केवल addition/subtraction से होगी। Nonlinear operations lookup tables से होंगी। इस branch में अभी सिर्फ़ additive linear-projection reference runtime तैयार है; full convolution/SSM/MoE/LUT/export runtime अधूरा है। इसलिए इसे तैयार multiply-free deployment runtime नहीं कहा जाएगा।

## 9. इस branch में किए गए परिवर्तन

1. OASST1 preparation में synthetic, review, deletion, ready-state और top-rank filters जोड़े।
2. Regression tests में synthetic/unreviewed/non-ready/rank-1 rows को drop करने का test जोड़ा।
3. V12 notebook को correct dataset-filter source revision पर pin किया और test में pin को lock किया।
4. पुराने Q-Mini और नए Add/Sub v1 के लिए अलग documentation boundary बनाई।
5. स्वतंत्र Add/Sub v1 training model, 3-bit packer, additive projection primitive, tokenizer/data scripts, random-init trainer, response-masked SFT और training-off Colab notebook जोड़े।

## 10. अभी बाकी और acceptance gate

- इस branch के GitHub Actions में सभी tests green होने चाहिए; कोई skipped pyarrow test OASST actual filter test को substitute नहीं करता।
- Real dataset preparation output में हर dropped-category count save करना चाहिए।
- OASST1/ DailyDialog output manifests, exact hashes, record counts और train/validation tree isolation verify होनी चाहिए।
- पुराने Drive checkpoints untouched रहें; कोई notebook स्वतः training शुरू न करे जब तक user अलग से Run न करे।
- Add/Sub v1 में source-level/static audit के साथ compiled runtime audit भी चाहिए कि MUL/DIV/MATMUL instruction path नहीं है।
- पुराने model को नए model के checkpoints/artifacts load करने की अनुमति नहीं होगी।

**सुरक्षा नियम:** पुराने model, उसकी training history या checkpoints को delete/overwrite नहीं किया जाएगा। नई architecture का काम अलग branch/package और अलग artifacts पर ही होगा।


## 11. 20 नवम्बर 2026 — दो-model cleanup और Add/Sub text hygiene

- सक्रिय model implementations को स्पष्ट रूप से दो रखा गया है: legacy `qmini/` और independent `qmini_addsub/`.
- Colab root में तीन उपयोगी entry points बचे हैं; 21 पुराने notebooks को delete करने के बजाय archive किया गया है.
- CI अब active और archived notebooks को recursively syntax-check करता है; पुरानी notebooks को active training recommendations नहीं माना जाता.
- Add/Sub cleaner Unicode NFC, line endings, horizontal whitespace और control characters normalize करता है; email, credential-like secret, स्पष्ट phone-contact PII, empty/short और overlong text reject करता है.
- OASST1 filter review, deletion, synthetic flag, tree state, assistant rank और non-empty tree/message IDs को fail-closed validate करता है.
- Dolly cleaner हर field rejection का अलग counter लिखता है; SFT loader approved sources के records को दोबारा hygiene-check करता है.
- कोई model weight, user checkpoint, dataset cache या existing Drive artifact delete नहीं किया गया.

यह cleanup code/data hygiene को बेहतर बनाता है; Add/Sub v1 का full multiply-free inference graph, export/loader और वास्तविक trained checkpoint अभी अलग acceptance gates हैं.


## 12. 23 नवम्बर 2026 — repository cleanup और notebook/image audit

- पूरी Git tree (recursive, untruncated) में कोई standalone image asset नहीं मिला: `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif`, `.svg`, bitmap या icon फ़ाइलें tracked नहीं थीं। जाँचे गए notebooks में embedded image outputs/attachments भी नहीं मिले। इसलिए इस cleanup में कोई उपयोगी image asset delete नहीं किया गया; एक हटाए गए पुराने notebook में केवल बाहरी Google Colab badge URL था।
- 15 पुराने, duplicate या project direction से बाहर notebooks हटाए गए: चार Hindi variants, पुराने duplicate English trainers, T4/GPU V6–V8 experiments, पुराने optimized trainers, distillation और obsolete personal-assistant experiment। इससे लगभग 1,018,970 bytes (0.97 MiB) tracked notebook content हटता है।
- छह पुराने notebooks जिन्हें अभी regression tests और notebook-validation workflow पढ़ते हैं, उन्हें `colab/archive/` से `tests/fixtures/legacy_colab/` में स्थानांतरित किया गया। उनके content/blob IDs सुरक्षित हैं; वे अब active Colab entry points नहीं हैं।
- `colab/` में केवल तीन supported notebooks रखे गए: Add/Sub from-scratch, English CPU resume और existing-Drive CPU chat।
- 23 नवम्बर audit के समय model implementation और configs सुरक्षित रखे गए थे। यह historical decision 25 नवम्बर के focused cleanup से supersede हुआ: अब `configs/` में केवल Light profile और CI smoke profile रखे गए हैं।
- Regression tests और GitHub Actions notebook validator को नए fixture path पर अपडेट किया गया ताकि सफ़ाई के बाद CI coverage बनी रहे।

यह notebook/structure cleanup है; model architecture, training data या model weights में कोई बदलाव नहीं किया गया।
