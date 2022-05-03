# Abstractive_Summarization

# mT5、T5 簡介

## T5
Text-to-text transfer Transformer

使用預訓練的模型參數對真正目標領域進行微調(fine-tuning)

使用預訓練資料Colossal Clean Crawled Corpus (C4)

預訓練時把C4資料集以corrupted spans方式進行

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


### 測試資料

- 2萬筆資料

模型 | mt5-small | mt5-base | XLSum-base
------|:-----:|------:|------:
TRAIN MEM | 10.90 | 19.52 | 19.68
PREDICT MEM| 4.89 | 5.5 | 5.5
訓練時間 | 2.6 hr | 3.8 hr | 3.7 hr
train_samples_per_second | 4.942 | 3.494 | 3.489
predict_samples_per_second | 6.243 | 5.126 | 5.079
eval_rouge1 | 15.1976 | 20.4043 | 19.472
predict_rouge1 | 15.8083 | 20.5865 | 18.6896
Model size | 1.1GB | 2.2GB | 2.2GB

- 5萬筆資料

模型 | mt5-small | mt5-base | XLSum-base
------|:-----:|------:|------:
訓練時間 | 15.5 hr | 19.3 hr | 18.9 hr
train_samples_per_second | 2.067 | 1.668 | 1.706
predict_samples_per_second | 5.625 | 4.747 | 4.803
eval_rouge1 | 18.7681 | 23.9472 | 23.7812
predict_rouge1 | 18.549 | 23.5467 | 23.2484
Model size | 1.1GB | 2.2GB | 2.2GB

- 模型參數資料(0423_small/0423_base/0423_XLSum)
    - data: 5萬筆
    - evaluation_strategy："steps"
    - learning_rate: 0.0001
    - gradient_accumulation_steps: 1

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

## 注意事項
1. 訓練基礎因使用超過3000筆資料，資料太少預測結果會出現 `<extra_id_0>`
2. 使用mt5時，不能使用`fp16`，會造成訓練問題，導致預測結果不良
3. 因為記憶體問題，`batch size`無法調太大，但在一定條件下，`batch size`越大，模型越穩定。此時可以調大`gradient_accumulation_steps`，來解決顯卡儲存空間的問題，如果`gradient_accumulation_steps`為8，則`batch size`「變相」擴大8倍。
