# Abstractive_Summarization

## T5
T5是Text-To-Text Transfer Transformer的簡稱，資料先進行預訓練，在使用預訓練的模型參數對真正目標領域進行微調(fine-tuning)。T5適用在許多NLP相關的工作，像是翻譯、分類、回歸（例如，預測兩個句子的相似程度，相似度分數在 1 到 5 之間），其他sequence to sequence任務，如摘要、生成文章。

T5 在預訓練過程中使用C4 (Colossal Clean Crawled Corpus)的資料，C4是透過將網頁文章爬取下來後，刪除重複數據、不完整的句子使資料庫足夠乾淨，預訓練時把C4資料集以corrupted spans方式進行

與GPT2不同的是，T5包含Encoder 和 Decoder，而GPT2只有Decoder。

T5能做的事
- 翻譯
- 問答
- 分類
- 摘要
- 回歸

怎麼做到摘要

decoder被訓練來預測給定前一個單詞的序列中的下一個單詞。

以下是解碼測試序列的步驟：

1. 編碼整個輸入序列並使用編碼器的內部狀態初始化解碼器
2. 將< start> 標記作為輸入傳遞給解碼器
3. 使用內部狀態運行解碼器一個時間步長
4. 輸出將是下一個單詞的概率。將選擇概率最大的單詞
5. 在下一個時間步中將採樣的單詞作為輸入傳遞給解碼器，並使用當前時間步更新內部狀態
6. 重複步驟 3 - 5，直到我們生成 < end> 標記或達到目標序列的最大長度


## mT5

Note: mT5 was only pre-trained on mC4 excluding any supervised training. Therefore, this model has to be fine-tuned before it is useable on a downstream task, unlike the original T5 model. Since mT5 was pre-trained unsupervisedly, there’s no real advantage to using a task prefix during single-task fine-tuning. If you are doing multi-task fine-tuning, you should use a prefix.

mT5 在mC4語料庫上進行了預訓練，涵蓋 101 種語言

mT5 在single-task fine-tuning 有無使用前綴差別不大。
multi-task fine-tuning，你應該使用前綴。

### 資料清理流程
1. 清理前綴字詞
2. 清理英文、website、html tag
3. 清理標題小於5個字、內容小於100字的文章
4. 清理重複文章
5. 清理空白、換行
6. 中文字的比例>50%


### 資料簡介

- 台灣新聞資料

資料 | title | content | summary(抽取式)
------|:-----:|------:|------:
min | 2 | 100| 8
mean| 26 | 849 | 176
median| 24 | 621 | 154
max | 229 | 145770 | 11013
std | 10.65 | 1216.55 | 212.39

- 紐約時報中文網（簡中翻譯繁中）

資料 | title | content | summary | predict
------|:-----:|------:|------:|------:
min | 2 | 1 | 14 | 26
mean| 16 | 2351 |72 | 70
median| 16 | 2199 |75 |  68
max | 35 | 38882 | 156| 234
std | 4 | 1561.49| 7.35 | 14.6

### 測試資料

- 2萬筆資料

模型 | mt5-small | mt5-base | XLSum-base
------|:-----:|------:|------:
TRAIN MEM | 10.90 | 19.52 | 19.68
PREDICT MEM| 4.89 | 5.5 | 5.5
訓練時間 | 2.6 hr | 3.8 hr | 3.7 hr
train_samples_per_second | 4.942 | 3.494 | 3.489
predict_samples_per_second | 6.243 | 5.126 | 5.079
loss | 3.2005 | 2.2277 | 1.9857
eval_loss | 2.9733 | 2.4192 | 2.4770
eval_rouge1 | 15.1976 | 20.4043 | 19.472
predict_rouge1 | 15.8083 | 20.5865 | 18.6896
Model size | 1.1GB | 2.2GB | 2.2GB

- 5萬筆資料
- 模型參數資料(0423_small/0423_base/0423_XLSum)
    - data: 5萬筆
    - evaluation_strategy："steps"
    - learning_rate: 0.0001
    - gradient_accumulation_steps: 1

模型 | mt5-small | mt5-base | XLSum-base
------|:-----:|------:|------:
訓練時間 | 15.5 hr | 19.3 hr | 18.9 hr
train_samples_per_second | 2.067 | 1.668 | 1.706
predict_samples_per_second | 5.625 | 4.747 | 4.803
loss | 2.8489 | 2.0759 | 1.822
eval_loss | 2.6341 | 2.1617 | 2.1492
eval_rouge1 | 18.7681 | 23.9472 | 23.7812
predict_rouge1 | 18.549 | 23.5467 | 23.2484
Model size | 1.1GB | 2.2GB | 2.2GB

- 清完資料
- 模型參數資料(0505_10/0505_)
    - data: 6萬4千筆
    - evaluation_strategy："epoch"
    - learning_rate: 0.001
    - lr_scheduler_type: constant
    - gradient_accumulation_steps: 16

    類別 | title | summary
    ------|:-----:|------:
    訓練時間 | 1.9 hr |  hr
    loss | 2.6019 | 
    eval_loss | 2.5358 | 
    eval_rouge1 | 16.623 | 

- 比較長短target(使用標題和抽取是摘要)
    - data: 10萬筆
    - model: base
    - gradient_accumulation_steps: 8
    - learning rate: 0.0001
    
    target | title | summary(抽取式)
    ------|:-----:|------:
    訓練時間 | 19.53 hr | 8.1625
    epoch | 10 | 3
    train_samples_per_second | 14.381 | 10.308
    predict_samples_per_second | 6.557 | 1.307
    loss | 1.3335 | 0.1525
    eval_loss | 1.6193 | 0.1396
    eval_rouge1 | 29.0476 | 63.7029


- 比較 gradient_accumulation_steps (使用mt5-small)
    - learning_rate: 0.0001
    - 5000筆測資
    - 訓練標題

    gradient_steps | 1 | 2 | 4 | 8| 16
    ------|:-----:|------:|------:|------:|------:
    訓練時間 | 27.3 min | 16.8 min | 11.5 min | 8.9 min| 8.5 min
    loss| 3.582 | 3.7439 | 3.9583 | 4.1808 | 4.5563
    eval_loss| 3.2737 | 3.3445 | 3.4715 | 3.515 | 3.6509
    eval_rouge1 | 13.5184 | 14.0531 | 12.3505 | 11.8577 | 11.1772

- 比較 learning rate (使用mt5-small)
    - gradient_accumulation_steps: 16
    - 5000筆測資
    - 訓練標題

    learning rate | 1e-3 | 1e-4 | 5e-5 
    ------|:-----:|------:|------:
    訓練時間 | 16.5 min | 8.5 min | 18.8 min 
    loss| 3.4117 | 4.5563 |  5.2583
    eval_loss| 3.2354 | 3.6509 | 3.7338
    eval_rouge1 | 13.7175 | 11.1772 | 8.0482

- 比較 learning rate (使用紐約時報資料)
    - mt5-small
    - gradient_accumulation_steps: 16
    - 18000筆測資
    - 訓練summary

    learning rate | 1e-3 | 1e-3 (constant) | 1e-4 
    ------|:-----:|------:|------:
    訓練時間 | 56 min |  77 min | 37 min 
    loss| 2.7852 | 2.7665 |  2.7691
    eval_loss| 2.5925 | 2.6024 | 2.7267
    eval_rouge1 | 7.4122 | 7.6442 | 7.3426

- 紐約時報資料(small vs base)
    - 18000筆測資
    - 訓練summary

    模型 | small(1e-4) | base(1e-3)
    ------|:-----:|------:
    訓練時間 | 37 min | 88 min
    loss| 2.7691 | 2.165
    eval_loss| 2.7267 | 2.3809
    eval_rouge1 | 7.3426 | 8.2238

## 變數解釋

- DataTrainingArguments
    - `text_column`:文章欄位名稱
    - `summary_column`:摘要欄位名稱 (沒有設定，第一個會視為文章、第二個為摘要)
    - `preprocessing_num_workers`: 加快資料預處理的速度
    - `max_source_length`:文章採取長度（可以無限長）
    - `max_target_length`:摘要採取長度
    - `num_beams`:beams 搜尋法 （用在evaluate and predict）
    - `source_prefix`:任務類別（用在t5訓練）

- TrainingArguments
    - `load_best_model_at_end`: 設定為true時，一定要設定`evaluation_strategy`
    - `evaluation_strategy`:可以是steps、no、epoch，如果不是no，do_eval會自動為true
    - `gradient_accumulation_steps`:累積梯度，「變相」擴大`batch size`。
    - `fp16`:mt5 不能使用

## generate text

- Predict時使用(transformers pipeline)
    ```python
    from transformers import pipeline
    import torch
    torch.manual_seed(32)  # 固定random seed
    summarizer = pipeline("summarization",model="./model")
    summarizer(text,args*)

    ```
    - `max_length`: 生產最長字數限制
    - `min_length`: 生產最短字數限制
    - `repetition_penalty`:疑似t5無法使用
    - `no_repeat_ngram_size`:重複字詞限制
    - `num_beams`:beams 搜尋法
    - `early_stopping`: 當所有beams 找到 EOS token.
    - `do_sample`: 隨機抽樣
    - `top_k`: (defaults to 50) 抽樣限制，先透過機率排序，選出機率最大的K個字，再來分機率，最後抽樣
    - `top_p`:(defaults to 1.0)抽樣限制，先透過機率排序，選出累績機率和等於P的字，再來分機率，最後抽樣

- generate 演算法介紹
    - Greedy search
        - `do_sample = False`
        - `num_beams = 1`
        - 走機率最大的
        - 最簡單的演算法，但生產出來的內容，受限於訓練資料。
    - Random sampling
        - `do_sample = True`
        - `num_beams = 1`
        - 依照字的機率，隨機抽一個
        - 可以搭配`temperature`使用，`temperature` 會增加機率大的字的機率，減少機率小的字的機率. 
        - `temperature = 1` -> Random sampling
        - `0 < temperature < 1` -> 字的機率會做調整，越小效果越大
        - `temperature -> 0` -> Greedy search
        - 可以搭配`top_k`、`top_p`使用
        - `top_p` 可以解決使用`top_k`每次都會有一定數量的候選字
    - Beams search
        - `do_sample = False`
        - `num_beams > 1`
        - 會保留前幾名，直到結束，可以用來產生多個結果。
        - 適用在每次輸出長度都差不多的情況
        - 嚴重受到重複生成的影響
        - 隨機性不足，與人類日常不同

    - Beams sampling
        - `do_sample = True`
        - `num_beams > 1`
        - 將兩種方法結合使用
        - mt5使用的過程中，如果有使用topk,temperature容易會有其他語言出現

- 演算法測試
    - 紐約時報資料
    - 預測500筆測資

        演算法 | greedy | beams | sampling | beams sp
        ------|:-----:|------:|------:|------:
        eval_rouge1 |  | 
        eval_rouge2 |  | 
        eval_rougeL |  | 
    
    - beams

        演算法 | n = 2 | n = 5 | n = 10 | n = 20
        ------|:-----:|------:|------:|------:
        time |  | 
        eval_rouge1 |  | 
        eval_rouge2 |  | 
        eval_rougeL |  | 

## 注意事項
1. 訓練基礎因使用超過3000筆資料，資料太少預測結果會出現 `<extra_id_0>`
2. 使用mt5時，不能使用`fp16`，會造成訓練問題，導致預測結果不良
3. 因為記憶體問題，`batch size`無法調太大，但在一定條件下，`batch size`越大，模型越穩定。此時可以調`gradient_accumulation_steps`，來解決顯卡儲存空間的問題，如果`gradient_accumulation_steps`為8，則`batch size`「變相」擴大8倍。

## 參考資料
1. [T5參考影片 -> Colin Raffel](https://www.youtube.com/watch?v=eKqWC577WlI&list=UUEqgmyWChwvt6MFGGlmUQCQ&index=5)
2. [淺談神經機器翻譯 & 用 Transformer 與 TensorFlow 2](https://leemeng.tw/neural-machine-translation-with-transformer-and-tensorflow2.html?fbclid=IwAR2eHxhPxyg96A3mbtveRHd5zFKscSLA-u8jdoDueUC9Dl1g3Vrv-61Y84g)
3. [Decoding Strategies that You Need to Know for Response Generation](https://towardsdatascience.com/decoding-strategies-that-you-need-to-know-for-response-generation-ba95ee0faadc)

