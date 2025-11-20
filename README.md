# chromegrammarly

英文语法高亮工具，包含：
- 后端：基于 FastAPI 的接口，可选择两套实现：
  - `backend/spacyback`：使用 spaCy + benepar 进行成分句法分析。
  - `backend/nltkback`：使用 NLTK 进行启发式词性/短语标注。
- 前端：Chrome 插件（`extension/`），可将文本发送到后端并在页面上高亮语法结构。

## 运行与部署

### 本地快速启动（默认 spaCy 版）
```bash
pip install -r requirements.txt
cd backend/spacyback
uvicorn mainspacy:app --host 0.0.0.0 --port 12012 --reload
```

### 使用 NLTK 版本
```bash
pip install -r requirements.txt
cd backend/nltkback
uvicorn mainnltk:app --host 0.0.0.0 --port 12012 --reload
```

> benepar 与 spaCy 英文模型会在运行时尝试自动下载，若下载失败，请手动执行：
> ```bash
> python -m spacy download en_core_web_sm
> python -m benepar.download benepar_en3
> ```

### Chrome 插件
1. 打开 Chrome 扩展管理（`chrome://extensions`），开启“开发者模式”。
2. 选择“加载已解压的扩展程序”，指向 `extension/` 目录。
3. 在插件配置中填写后端接口地址（默认 `http://localhost:12012/analyze`），即可发送文本并查看高亮。

<img width="600" height="335" alt="image" src="https://github.com/user-attachments/assets/85f57808-d8dd-4215-b56c-86acbbd20b9a" />

## 推荐内容
本项目非大模型应用，但是利用Codex+gpt5进行了辅助的vibe coding。
感兴趣的用户可以点击以下链接，了解大模型聚合平台服务，优惠力度很大。
https://yunwu.ai/register?aff=SEOm
