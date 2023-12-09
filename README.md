# ResNetXt

## Setup

Setup is the same as gavel project.

## folder 

請在與 `train.py` 相同的目錄底下新增 `dataset`、`json` 資料夾，這兩個資料夾分別是放入 csv 檔 (dataset)、json 檔 (json)。

當 `process.py` 跑完後，處理過後的 `npz` 檔會存在 `data` 資料夾裡面。

當 `train.py` 結束，也就是訓練完成後，模型權重會被儲存在 `weights` 資料夾裡面。

## file

- utils.preprocess : 帶通濾波器放置處
- model.py : ResNetXt 架構
- process.py : 將 CSV 檔清洗成訓練用的格式，請務必先跑一遍
- train.py : 訓練用程式，有參數可調
- validate.py : 測試用，可填入測試資料集路徑進行測試
- train_xgboost.py : 訓練成功後，請加入訓練好的模型權重 (路徑)，再訓練 Xgboost
- validate_xgboost.py : 測試整套程式流程準確率