const textInput = document.getElementById('text-input');
const analyzeBtn = document.getElementById('analyze-btn');
const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');
const readoutLabel = document.getElementById('readout-label');
const readoutConf = document.getElementById('readout-conf');
const needle = document.getElementById('meter-needle');
const probBars = document.getElementById('prob-bars');

async function analyze() {
  const text = textInput.value.trim();
  if (!text) {
    statusEl.textContent = 'Enter some text first.';
    return;
  }

  analyzeBtn.disabled = true;
  statusEl.textContent = 'Running inference…';

  try {
    const res = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();

    if (!res.ok) {
      statusEl.textContent = data.error || 'Something went wrong.';
      analyzeBtn.disabled = false;
      return;
    }

    renderResult(data);
    statusEl.textContent = '';
  } catch (err) {
    statusEl.textContent = 'Could not reach the server.';
  } finally {
    analyzeBtn.disabled = false;
  }
}

function renderResult(data) {
  resultEl.classList.remove('hidden');

  readoutLabel.textContent = data.label;
  readoutLabel.className = 'readout-label ' + (data.label.toLowerCase().includes('pos') ? 'pos' : 'neg');
  readoutConf.textContent = `${(data.confidence * 100).toFixed(1)}% confidence`;

  // Needle position: assumes a binary negative/positive scale.
  // Uses the positive-class probability if present, otherwise falls back to confidence.
  const posEntry = data.probabilities.find(p => p.label.toLowerCase().includes('pos'));
  const posProb = posEntry ? posEntry.value : data.confidence;
  needle.style.left = `${posProb * 100}%`;

  probBars.innerHTML = '';
  data.probabilities.forEach(p => {
    const row = document.createElement('div');
    row.className = 'prob-row';
    row.innerHTML = `
      <span class="prob-name">${p.label}</span>
      <span class="prob-track"><span class="prob-bar" style="width:${(p.value * 100).toFixed(1)}%"></span></span>
      <span class="prob-val">${(p.value * 100).toFixed(1)}%</span>
    `;
    probBars.appendChild(row);
  });
}

analyzeBtn.addEventListener('click', analyze);
textInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) analyze();
});
