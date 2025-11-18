const API_BASE_URL = 'http://0.0.0.0:12012';

async function fetchHighlights(text) {
  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || '后台服务调用失败。');
  }

  const data = await response.json();
  return data.highlighted_html;
}

async function fetchParagraphHighlights(paragraphs) {
  const results = [];
  for (const paragraph of paragraphs) {
    // Process sequentially to avoid overwhelming the backend.
    const highlighted = await fetchHighlights(paragraph);
    results.push(highlighted);
  }
  return results;
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type !== 'GRAMMAR_API_REQUEST') {
    return;
  }

  const { text, paragraphs } = message;

  (Array.isArray(paragraphs) && paragraphs.length > 0
    ? fetchParagraphHighlights(paragraphs).then((highlightedHtmls) => ({
        highlightedHtmls
      }))
    : fetchHighlights(text).then((highlightedHtml) => ({ highlightedHtml })))
    .then((payload) => sendResponse(payload))
    .catch((error) => {
      sendResponse({ error: error.message || '后台服务调用失败。' });
    });

  return true; // keep the message channel open for async response
});

chrome.action.onClicked.addListener((tab) => {
  if (!tab.id) return;
  chrome.tabs.sendMessage(tab.id, { type: 'GRAMMAR_TOGGLE_PANEL' }).catch(() => {});
});
