const PANEL_ID = 'grammar-panel';
const PANEL_STYLE_ID = 'grammar-panel-style';
const PARAGRAPH_SELECTORS = 'p, article p, section p, div';

let panelEl = null;

function ensurePanelStyle() {
  if (document.getElementById(PANEL_STYLE_ID)) return;
  const style = document.createElement('style');
  style.id = PANEL_STYLE_ID;
  style.textContent = `
    .grammar-panel {
      position: fixed;
      right: 16px;
      bottom: 16px;
      width: 220px;
      padding: 12px;
      border-radius: 10px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.18);
      background: #fff;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      z-index: 2147483647;
      border: 1px solid rgba(15, 23, 42, 0.08);
    }
    .grammar-panel__header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 10px;
    }
    .grammar-panel__title {
      font-size: 11px;
      width: 200px;
      font-weight: 600;
      margin: 0;
      color: #0f172a;
    }
    .button.grammar-panel__close {
      background: transparent;
      border: none;
      width: 5px;
      height: 3px;
      pandding-left: 12px;
      font-size: 11px;
      cursor: pointer;
      color: #475569;
    }
    .grammar-panel button {
      width: 30%;
      font-size: 11px;
      padding: 2px;
      margin-bottom: 3px;
      border: none;
      border-radius: 3px;
      cursor: pointer;
      background: #2563eb;
      color: white;
      transition: background 0.2s ease;
    }
    .grammar-panel button:hover {
      background: #1d4ed8;
    }
    .grammar-panel__status {
      min-height: 16px;
      font-size: 11px;
      color: #475569;
      margin-top: 4px;
    }
    .grammar-panel__status.error {
      color: #dc2626;
    }
    .grammar-panel__status.success {
      color: #0a8754;
    }
  `;
  document.head.appendChild(style);
}

function callBackend(payload) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(
      { type: 'GRAMMAR_API_REQUEST', ...payload },
      (response) => {
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
          return;
        }
        if (!response) {
          reject(new Error('后台未返回数据。'));
          return;
        }
        if (response.error) {
          reject(new Error(response.error));
          return;
        }
        resolve(response);
      }
    );
  });
}

function isParagraphElement(node) {
  return (
    node &&
    node.nodeType === Node.ELEMENT_NODE &&
    node.matches &&
    node.matches(PARAGRAPH_SELECTORS)
  );
}

function collectParagraphNodes(container) {
  if (!container || container.nodeType !== Node.ELEMENT_NODE) return [];
  const paragraphs = Array.from(container.querySelectorAll('p')).filter(
    (el) => el.innerText.trim()
  );
  if (paragraphs.length > 0) return paragraphs;
  return [container];
}

function buildParagraphTarget(nodes) {
  const usableNodes = nodes.filter(
    (node) => node && node.innerText && node.innerText.trim()
  );
  if (usableNodes.length === 0) {
    throw new Error('未找到可用段落文本。');
  }

  const originals = usableNodes.map((node) => node.innerHTML);
  const paragraphs = usableNodes.map((node) => node.innerText.trim());

  return {
    payload: { paragraphs },
    apply(htmlList) {
      const list = Array.isArray(htmlList) ? htmlList : [];
      usableNodes.forEach((node, idx) => {
        const html = list[idx];
        if (typeof html === 'string' && html.trim()) {
          node.innerHTML = html;
        }
      });
    },
    restore() {
      usableNodes.forEach((node, idx) => {
        node.innerHTML = originals[idx];
      });
    }
  };
}

function getSelectionTarget() {
  const selection = window.getSelection();
  if (!selection || selection.rangeCount === 0) {
    throw new Error('请先在页面中选择一段文本。');
  }
  const range = selection.getRangeAt(0);
  if (range.collapsed) {
    throw new Error('所选文本为空。');
  }

  const root =
    range.commonAncestorContainer.nodeType === Node.ELEMENT_NODE
      ? range.commonAncestorContainer
      : range.commonAncestorContainer.parentElement || document.body;

  const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT, {
    acceptNode(node) {
      if (!(node instanceof HTMLElement)) return NodeFilter.FILTER_SKIP;
      return isParagraphElement(node) && range.intersectsNode(node)
        ? NodeFilter.FILTER_ACCEPT
        : NodeFilter.FILTER_SKIP;
    }
  });

  const nodes = [];
  let current = walker.nextNode();
  while (current) {
    nodes.push(current);
    current = walker.nextNode();
  }

  if (nodes.length === 0) {
    const fallback =
      (range.commonAncestorContainer.nodeType === Node.ELEMENT_NODE
        ? range.commonAncestorContainer
        : range.commonAncestorContainer.parentElement) || document.body;
    nodes.push(fallback.closest(PARAGRAPH_SELECTORS) || fallback);
  }

  return buildParagraphTarget(nodes);
}

function getParagraphTarget() {
  const selection = window.getSelection();
  let node =
    selection && selection.rangeCount > 0
      ? selection.getRangeAt(0).commonAncestorContainer
      : document.activeElement;

  if (!node) {
    node = document.body;
  }

  if (node.nodeType === Node.TEXT_NODE) {
    node = node.parentElement;
  }

  const paragraph =
    node.closest(PARAGRAPH_SELECTORS) || document.querySelector('p') || document.body;

  return buildParagraphTarget([paragraph]);
}

function getArticleTarget() {
  const container =
    document.querySelector('article') ||
    document.querySelector('main') ||
    document.body;

  const nodes = collectParagraphNodes(container).filter(
    (node) => node.innerText.trim()
  );

  return buildParagraphTarget(nodes);
}

async function handleAnalyze(mode) {
  let targetGetter;
  switch (mode) {
    case 'selection':
      targetGetter = getSelectionTarget;
      break;
    case 'paragraph':
      targetGetter = getParagraphTarget;
      break;
    case 'article':
      targetGetter = getArticleTarget;
      break;
    default:
      throw new Error('未知的分析模式。');
  }

  const target = targetGetter();
  try {
    const response = await callBackend(target.payload);
    const htmlList =
      response.highlightedHtmls ||
      (response.highlightedHtml ? [response.highlightedHtml] : []);
    if (!htmlList || htmlList.length === 0) {
      throw new Error('后台未返回数据。');
    }
    target.apply(htmlList);
    return {};
  } catch (error) {
    target.restore?.();
    throw error;
  }
}

function setPanelStatus(message, type = '') {
  if (!panelEl) return;
  const status = panelEl.querySelector('.grammar-panel__status');
  if (!status) return;
  status.textContent = message || '';
  status.className = `grammar-panel__status ${type}`;
}

function setPanelDisabled(disabled) {
  if (!panelEl) return;
  panelEl.querySelectorAll('button[data-mode]').forEach((btn) => {
    btn.disabled = disabled;
  });
}

function createPanel() {
  ensurePanelStyle();
  if (panelEl) return panelEl;

  const panel = document.createElement('div');
  panel.id = PANEL_ID;
  panel.className = 'grammar-panel';
  panel.innerHTML = `
    <div class="grammar-panel__header">
      <p class="grammar-panel__title">GTools</p>
      <button class="grammar-panel__close" title="关闭">x</button>
    </div>
    <button data-mode="selection">selection</button>
    <button data-mode="paragraph">paragraph</button>
    <button data-mode="article">article</button>
    <div class="grammar-panel__status"></div>
  `;

  panel.querySelector('.grammar-panel__close')?.addEventListener('click', () => {
    panel.remove();
    panelEl = null;
  });

  panel.querySelectorAll('button[data-mode]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const mode = btn.dataset.mode;
      setPanelStatus('处理中...', '');
      setPanelDisabled(true);
      try {
        await handleAnalyze(mode);
        setPanelStatus('已完成高亮。', 'success');
      } catch (error) {
        setPanelStatus(error.message || '未知错误', 'error');
      } finally {
        setPanelDisabled(false);
      }
    });
  });

  document.body.appendChild(panel);
  panelEl = panel;
  return panel;
}

function togglePanel() {
  if (panelEl) {
    panelEl.remove();
    panelEl = null;
    return;
  }
  createPanel();
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === 'GRAMMAR_ANALYZE') {
    handleAnalyze(message.mode)
      .then(() => sendResponse({ success: true }))
      .catch((error) => {
        sendResponse({ error: error.message || '未知错误' });
      });
    return true;
  }

  if (message?.type === 'GRAMMAR_TOGGLE_PANEL') {
    togglePanel();
  }
});
