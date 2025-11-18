const statusEl = document.getElementById('status');

function setStatus(message, type = '') {
  statusEl.textContent = message;
  statusEl.className = type;
}

async function sendAction(mode) {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  return new Promise((resolve) => {
    chrome.tabs.sendMessage(
      tab.id,
      { type: 'GRAMMAR_ANALYZE', mode },
      (response) => {
        if (chrome.runtime.lastError) {
          setStatus('内容脚本不可用，请刷新页面。', 'error');
          resolve();
          return;
        }
        if (!response) {
          setStatus('没有收到任何响应。', 'error');
          resolve();
          return;
        }
        if (response.error) {
          setStatus(response.error, 'error');
        } else {
          setStatus('已完成高亮。', 'success');
        }
        resolve();
      }
    );
  });
}

for (const button of document.querySelectorAll('button[data-mode]')) {
  button.addEventListener('click', async () => {
    const mode = button.dataset.mode;
    setStatus('处理中...', '');
    await sendAction(mode);
  });
}
