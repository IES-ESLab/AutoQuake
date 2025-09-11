# GitHub Copilot Context for AutoQuake

## 🚨 重要開發規則 - 請先閱讀

### ❌ 絕對禁止事項
- **絕對不要** 在根目錄創建新檔案 → 使用適當的模組結構
- **絕對不要** 直接將輸出檔案寫入根目錄 → 使用指定的輸出資料夾
- **絕對不要** 創建重複檔案 (manager_v2.py, enhanced_xyz.py等) → 總是擴展現有檔案
- **絕對不要** 對同一概念創建多個實現 → 保持單一真實來源
- **絕對不要** 複製貼上代碼塊 → 提取到共享工具/函數中
- **絕對不要** 硬編碼應該可配置的值 → 使用配置檔案/環境變數
- **絕對不要** 使用 enhanced_, improved_, new_, v2_ 等命名 → 直接擴展原始檔案

### 📝 強制要求
- **編輯前先讀檔案** - 編輯工具如果沒有先讀取檔案會失敗
- **債務預防** - 創建新檔案前，檢查是否有類似功能可以擴展
- **單一真實來源** - 每個功能/概念只有一個權威實現
- **模組化設計** - 每個處理步驟都是獨立組件
- **路徑使用絕對路徑** - 配置中的所有路徑都應為絕對路徑

### 🔍 任務前檢查清單
**步驟1: 任務分析**
- [ ] 這會在根目錄創建檔案嗎？ → 如果是，使用適當的模組結構
- [ ] 這會花費超過30秒嗎？ → 如果是，考慮分解任務
- [ ] 這是3+步驟的複雜任務嗎？ → 如果是，先分解任務

**步驟2: 技術債務預防**
- [ ] **先搜索**: 檢查是否已有類似實現
- [ ] **檢查現有**: 閱讀找到的檔案了解當前功能
- [ ] 是否已存在類似功能？ → 如果是，擴展現有代碼
- [ ] 我是否要創建重複的類/管理器？ → 如果是，改為整合
- [ ] 我是否要複製貼上代碼？ → 提取到共享工具

## 專案概述

AutoQuake是一個自動化的全方位地震目錄生成解決方案，整合多個地震學分析步驟到一個統一的pipeline中，支援傳統地震儀數據(SAC格式)和分散式聲學感測(DAS)數據(HDF5格式)。

## 核心架構

### 主要處理管線組件
位於 `autoquake/` 目錄下：

1. **PhaseNet** (`picker.py`) - 使用深度學習進行地震震相撿拾
2. **GaMMA** (`associator.py`) - 使用高斯混合模型進行地震檢測和定位
3. **H3DD** (`relocator.py`) - 使用雙差分方法進行3D地震重定位
4. **Magnitude** (`magnitude.py`) - 區域規模計算
5. **DitingMotion** (`polarity.py`) - 初動極性判定
6. **GAfocal** (`focal.py`) - 使用遺傳演算法進行震源機制解

### 配置系統
- **`ParamConfig/config_model.py`** - 使用Pydantic的型別安全配置模型
- **`ParamConfig/params.json`** - 包含所有參數的範例配置檔案
- **`predict.py`** - 腳本式pipeline執行
- **`autoquake/scenarios.py`** - 函數式pipeline (`run_autoquake()`)

### 數據流程
1. 原始地震數據(SAC/HDF5) → PhaseNet → 震相撿拾
2. 震相撿拾 → GaMMA → 地震事件和精化撿拾
3. 事件 + 撿拾 → H3DD → 重定位地震(執行兩次精化)
4. 重定位事件 → Magnitude → 區域規模
5. 原始撿拾 → DitingMotion → 初動極性
6. 組合數據 → GAfocal → 震源機制解

## 開發環境設置

### 環境安裝
```bash
# 使用Conda(推薦)
conda env create -f env.yml
conda activate AutoQuake_v0

# 或使用pip
pip install -r requirements.txt
```

### 初始化Submodules
```bash
./init_submodules.sh
```

## 數據結構要求

### 地震儀數據(SAC)
```
/dataset_parent_dir/
├── *YYYYMMDD*/
│   └── waveform.SAC
```

### DAS數據(HDF5)
```
/dataset_parent_dir/
├── *YYYYMMDD*/
│   └── MiDAS_20240402_86100_86400.h5
```

## 外部依賴

### Fortran組件
- **GAfocal/**: 遺傳演算法震源機制解(需要gfortran)
- **H3DD/**: 雙差分重定位代碼(需要gfortran)

建置命令位於各自的`Makefile`中(`GAfocal/src/`和`H3DD/src/`)。

### Git Submodules
- `autoquake/GaMMA` - 高斯混合模型關聯
- `autoquake/EQNet` - PhaseNet實現

## 開發指南

### 代碼品質
- 使用ruff進行代碼檢查和格式化
- 目標Python 3.10
- 行長度：88字符
- 引號風格：單引號

### 檔案和目錄管理
- **模組結構**: 新檔案應放在適當的模組目錄中，不要在根目錄創建
- **輸出管理**: 輸出檔案應寫入指定的輸出資料夾，不要直接寫入根目錄
- **命名規範**: 避免使用版本後綴(_v2, _new, _enhanced)，直接擴展原始檔案
- **單一來源**: 每個概念只維護一個權威實現，避免重複代碼

### 路徑處理
- 配置中的所有路徑應為絕對路徑
- 代碼庫一致使用`pathlib.Path`以確保跨平台兼容性

### 類型檢查
- 使用Pydantic進行配置驗證
- 目前未配置mypy或其他靜態類型檢查

### 工作流程最佳實踐
1. **編輯前先讀檔案**: 使用read_file了解現有代碼結構
2. **搜索現有實現**: 在創建新功能前，先搜索是否已有類似實現
3. **擴展而非重寫**: 優先擴展現有檔案而非創建新檔案
4. **提取共享邏輯**: 將重複的代碼提取到工具函數中
5. **使用配置檔案**: 避免硬編碼，使用params.json或環境變數

### Git工作流程
- 每完成一個任務階段後提交
- 保持提交訊息清楚明確
- 確保submodules正確初始化和更新

```

## 開發工具和命令

### 代碼品質檢查
```bash
# 檢查和格式化代碼
ruff check .
ruff format .

# 自動修復問題
ruff check --fix .
```

### 環境管理
```bash
# 使用Conda設置環境
conda env create -f env.yml
conda activate AutoQuake_v0

# 安裝開發工具
pip install ruff
```

### 建置Fortran組件
```bash
# 建置GAfocal
cd GAfocal/src
make

# 建置H3DD
cd H3DD/src
make
```

### 測試
- 目前沒有配置正式的測試框架
- 透過範例腳本和notebooks進行測試(在.gitignore中排除)

## 常見工作流程

### 執行完整Pipeline
```python
from autoquake.scenarios import run_autoquake
from autoquake import PhaseNet, GaMMA, H3DD

# 配置和執行個別組件
picker = PhaseNet(...)
associator = GaMMA(...)
relocator = H3DD(...)

run_autoquake(picker=picker, associator=associator, relocator=relocator)
```

### 腳本式執行
```bash
python predict.py  # 使用ParamConfig/params.json
```

## 重要注意事項

### 核心設計原則
1. **模組化設計**: 每個處理步驟都是獨立的組件
2. **配置管理**: 使用Pydantic確保配置的型別安全
3. **數據格式**: 支援SAC和HDF5兩種主要格式
4. **Fortran依賴**: 部分組件需要編譯Fortran代碼
5. **Git Submodules**: 確保正確初始化子模組

### 代碼品質原則
6. **單一真實來源**: 避免重複實現，每個功能只有一個權威版本
7. **擴展優於重寫**: 優先擴展現有檔案而非創建新檔案
8. **配置化**: 避免硬編碼，使用配置檔案和環境變數
9. **適當的檔案組織**: 遵循模組結構，不在根目錄創建檔案
10. **路徑一致性**: 統一使用絕對路徑和`pathlib.Path`

### 開發流程原則
11. **先讀後寫**: 編輯檔案前必須先讀取了解現有結構
12. **搜索再創建**: 創建新功能前先搜索是否已有類似實現
13. **提取共享邏輯**: 將重複代碼提取到共享工具函數
14. **階段性提交**: 每完成一個任務階段後進行git提交

這個專案適合地震學研究人員和地球物理學家使用，提供完整的地震數據處理pipeline。

## 快速開始指引

1. **環境設置**: `conda env create -f env.yml && conda activate AutoQuake_v0`
2. **初始化submodules**: `./init_submodules.sh` (已完成)
3. **編譯Fortran組件**: 進入GAfocal/src和H3DD/src執行make
4. **配置參數**: 編輯ParamConfig/params.json
5. **執行pipeline**: `python predict.py` 或使用`autoquake.scenarios.run_autoquake()`
