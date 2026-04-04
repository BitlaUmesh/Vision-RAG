/**
 * script.js — Vision RAG Frontend Logic
 * =======================================
 * Handles:
 *   - PDF upload with drag-and-drop, progress feedback
 *   - Chat message sending and display
 *   - Markdown rendering of AI responses
 *   - Loading indicator while backend processes
 *   - RAG metadata badges on each AI message
 *   - Memory clear functionality
 *   - Sidebar toggle for mobile
 *
 * Architecture: Plain ES2022, no build step, no framework.
 * Communicates with FastAPI backend at BASE_URL.
 */

'use strict';

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/** API base URL - empty for same-origin (frontend served by backend) */
const BASE_URL = '';

// Configure marked.js for safe Markdown → HTML rendering
marked.setOptions({
  breaks: true,
  gfm: true,
  headerIds: false,
  mangle: false,
});

// ---------------------------------------------------------------------------
// DOM References
// ---------------------------------------------------------------------------

const messagesWindow   = document.getElementById('messages-window');
const welcomeMessage   = document.getElementById('welcome-message');
const chatInput        = document.getElementById('chat-input');
const sendBtn          = document.getElementById('send-btn');
const clearBtn         = document.getElementById('clear-btn');

const dropZone         = document.getElementById('drop-zone');
const fileInput        = document.getElementById('pdf-file-input');
const uploadStatus     = document.getElementById('upload-status');
const uploadBarFill    = document.getElementById('upload-bar-fill');
const uploadStatusText = document.getElementById('upload-status-text');
const docCard          = document.getElementById('doc-card');
const docCardName      = document.getElementById('doc-card-name');
const docCardMeta      = document.getElementById('doc-card-meta');
const headerMeta       = document.getElementById('header-meta');
const statusDot        = document.getElementById('status-dot');
const sidebarToggle    = document.getElementById('sidebar-toggle');
const sidebar          = document.getElementById('sidebar');

// Overlay elements
const uploadOverlay      = document.getElementById('upload-overlay');
const overlayTitle       = document.getElementById('overlay-title');
const overlayFilename    = document.getElementById('overlay-filename');
const overlayProgressFill = document.getElementById('overlay-progress-fill');
const overlayStatusText  = document.getElementById('overlay-status-text');
const stepUpload         = document.getElementById('step-upload');
const stepOcr            = document.getElementById('step-ocr');
const stepVisual         = document.getElementById('step-visual');
const stepIndex          = document.getElementById('step-index');
const stepConnectors     = document.querySelectorAll('.step-connector');

// ---------------------------------------------------------------------------
// Application State
// ---------------------------------------------------------------------------

let isLoading = false;
let currentDocumentName = '';

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

function autoGrowTextarea() {
  chatInput.style.height = 'auto';
  chatInput.style.height = chatInput.scrollHeight + 'px';
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    messagesWindow.scrollTop = messagesWindow.scrollHeight;
  });
}

function hideWelcome() {
  if (welcomeMessage) {
    welcomeMessage.style.display = 'none';
  }
}

function formatPages(pages) {
  if (!pages || pages.length === 0) return 'no pages';
  return 'pages ' + pages.join(', ');
}

function setInputEnabled(enabled) {
  chatInput.disabled = !enabled;
  sendBtn.disabled   = !enabled;
  isLoading          = !enabled;
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(str));
  return div.innerHTML;
}

function setStatusActive(active) {
  if (statusDot) {
    statusDot.classList.toggle('active', active);
  }
}

// ---------------------------------------------------------------------------
// Message Rendering
// ---------------------------------------------------------------------------

function appendUserMessage(text) {
  hideWelcome();

  const wrapper = document.createElement('div');
  wrapper.className = 'message user';
  wrapper.innerHTML = `
    <div class="message-avatar">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/>
        <circle cx="12" cy="7" r="4"/>
      </svg>
    </div>
    <div class="message-content">
      <div class="message-bubble">${escapeHtml(text)}</div>
    </div>
  `;
  messagesWindow.appendChild(wrapper);
  scrollToBottom();
}

function appendAiMessage(markdown, needsRag, pages, reason) {
  const renderedHtml = marked.parse(markdown);

  const ragTagClass = needsRag ? 'rag-active' : 'no-rag';
  const ragTagText  = needsRag
    ? `📄 RAG · ${formatPages(pages)}`
    : '💬 Chat-only';

  // Show visual tag if RAG was used with pages (multimodal call)
  const visualTag = (needsRag && pages && pages.length > 0)
    ? `<span class="meta-tag visual-tag">🖼️ Visual</span>`
    : '';

  const wrapper = document.createElement('div');
  wrapper.className = 'message ai';
  wrapper.innerHTML = `
    <div class="message-avatar">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2z"/>
        <path d="M12 16v-4"/>
        <path d="M12 8h.01"/>
      </svg>
    </div>
    <div class="message-content">
      <div class="message-bubble">${renderedHtml}</div>
      <div class="message-meta">
        <span class="meta-tag ${ragTagClass}">${ragTagText}</span>
        ${visualTag}
        <span class="meta-tag" title="${escapeHtml(reason)}">🔀 ${escapeHtml(reason.split('—')[0].trim())}</span>
      </div>
    </div>
  `;
  messagesWindow.appendChild(wrapper);
  scrollToBottom();
}

function showLoadingIndicator() {
  const el = document.createElement('div');
  el.className = 'loading-message';
  el.id = 'loading-indicator';
  el.innerHTML = `
    <div class="message-avatar" style="background: var(--surface-container-high);">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2z"/>
        <path d="M12 16v-4"/>
        <path d="M12 8h.01"/>
      </svg>
    </div>
    <div class="loading-bubble">
      <div class="loading-dot"></div>
      <div class="loading-dot"></div>
      <div class="loading-dot"></div>
    </div>
  `;
  messagesWindow.appendChild(el);
  scrollToBottom();
  return el;
}

function removeLoadingIndicator() {
  const el = document.getElementById('loading-indicator');
  if (el) el.remove();
}

function appendErrorMessage(errorText) {
  const wrapper = document.createElement('div');
  wrapper.className = 'message ai';
  wrapper.innerHTML = `
    <div class="message-avatar" style="background: rgba(255, 110, 132, 0.15);">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--danger)" stroke-width="2">
        <circle cx="12" cy="12" r="10"/>
        <line x1="15" y1="9" x2="9" y2="15"/>
        <line x1="9" y1="9" x2="15" y2="15"/>
      </svg>
    </div>
    <div class="message-content">
      <div class="message-bubble" style="border-top: 1px solid rgba(255, 110, 132, 0.3);">
        <strong style="color: var(--danger);">Error:</strong> ${escapeHtml(errorText)}
      </div>
    </div>
  `;
  messagesWindow.appendChild(wrapper);
  scrollToBottom();
}

// ---------------------------------------------------------------------------
// Chat Logic
// ---------------------------------------------------------------------------

async function sendMessage() {
  const message = chatInput.value.trim();

  if (!message || isLoading) return;

  chatInput.value = '';
  autoGrowTextarea();

  appendUserMessage(message);

  showLoadingIndicator();
  setInputEnabled(false);

  try {
    const response = await fetch(`${BASE_URL}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });

    removeLoadingIndicator();

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(errBody.detail || `HTTP ${response.status}`);
    }

    const data = await response.json();

    appendAiMessage(
      data.response,
      data.needs_rag,
      data.retrieved_pages,
      data.routing_reason,
    );

  } catch (err) {
    removeLoadingIndicator();
    appendErrorMessage(err.message || 'Unknown error. Is the backend running?');
    console.error('[VisionRAG] Chat error:', err);
  } finally {
    setInputEnabled(true);
    chatInput.focus();
  }
}

// ---------------------------------------------------------------------------
// Upload Logic
// ---------------------------------------------------------------------------

async function uploadPdf(file) {
  if (!file || !file.name.toLowerCase().endsWith('.pdf')) {
    alert('Please select a valid PDF file.');
    return;
  }

  // Show full-screen overlay
  showOverlay(file.name);
  docCard.classList.remove('visible');

  const formData = new FormData();
  formData.append('file', file);

  try {
    // Step 1: Upload
    setOverlayStep('upload', 15, 'Uploading document...');

    // Step 2: Simulate extraction start
    setTimeout(() => {
      setOverlayStep('ocr', 40, 'Extracting text & handwriting...');
    }, 800);
    setTimeout(() => {
      setOverlayStep('visual', 65, 'Analyzing diagrams, tables & images...');
    }, 1600);
    setTimeout(() => {
      setOverlayStep('index', 80, 'Building search index...');
    }, 2500);

    const response = await fetch(`${BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(errBody.detail || `HTTP ${response.status}`);
    }

    const data = await response.json();

    // Success!
    setOverlayStep('done', 100, 'Document ready!');
    uploadOverlay.classList.add('success');
    overlayTitle.textContent = '✅ Document Ready!';

    const ocrInfo = data.ocr_pages > 0
      ? ` · ✍️ ${data.ocr_pages} handwritten page(s) detected`
      : '';
    const visualInfo = data.visual_pages > 0
      ? ` · 🖼️ ${data.visual_pages} page(s) with visual content`
      : '';

    // Update sidebar
    uploadStatus.style.display = 'flex';
    setUploadStatus(`✓ ${data.index_status}`, 100, 'success');
    currentDocumentName = data.filename;
    docCardName.textContent = data.filename;
    docCardMeta.textContent = data.index_status;
    docCard.classList.add('visible');
    headerMeta.textContent = `${data.filename}${ocrInfo}${visualInfo}`;
    setStatusActive(true);

    messagesWindow.innerHTML = '';
    renderWelcomeBack(data.filename, data.ocr_pages || 0, data.visual_pages || 0);

    console.info(`[VisionRAG] Uploaded & indexed: ${data.filename} (OCR pages: ${data.ocr_pages || 0})`);

    // Auto-dismiss overlay after 1.5s
    setTimeout(() => hideOverlay(), 1500);

  } catch (err) {
    overlayTitle.textContent = '❌ Upload Failed';
    overlayStatusText.textContent = err.message;
    overlayProgressFill.style.width = '0%';
    overlayStatusText.style.color = 'var(--danger)';
    uploadStatus.style.display = 'flex';
    setUploadStatus(`✗ Upload failed: ${err.message}`, 0, 'error');
    console.error('[VisionRAG] Upload error:', err);
    setTimeout(() => hideOverlay(), 3000);
  }
}

// ── Overlay helpers ──

function showOverlay(filename) {
  overlayTitle.textContent = 'Processing Document';
  overlayFilename.textContent = filename;
  overlayProgressFill.style.width = '0%';
  overlayStatusText.textContent = 'Preparing upload...';
  overlayStatusText.style.color = '';
  uploadOverlay.classList.remove('success');

  // Reset all steps
  [stepUpload, stepOcr, stepVisual, stepIndex].forEach(s => {
    s.classList.remove('active', 'done');
  });
  stepConnectors.forEach(c => c.classList.remove('active'));

  uploadOverlay.classList.add('visible');
}

function hideOverlay() {
  uploadOverlay.classList.remove('visible', 'success');
}

function setOverlayStep(stepName, progress, statusText) {
  const steps = [stepUpload, stepOcr, stepVisual, stepIndex];
  const names = ['upload', 'ocr', 'visual', 'index', 'done'];
  const idx = names.indexOf(stepName);

  steps.forEach((s, i) => {
    s.classList.remove('active', 'done');
    if (i < idx) s.classList.add('done');
    if (i === idx && stepName !== 'done') s.classList.add('active');
    if (stepName === 'done') s.classList.add('done');
  });

  // Activate connectors up to current step
  stepConnectors.forEach((c, i) => {
    if (i < idx) {
      c.classList.add('active');
    } else {
      c.classList.remove('active');
    }
  });

  overlayProgressFill.style.width = progress + '%';
  overlayStatusText.textContent = statusText;
}

function setUploadStatus(text, barPercent, cssClass) {
  uploadStatusText.textContent = text;
  uploadStatusText.className = 'upload-status-text ' + cssClass;
  uploadBarFill.style.width = barPercent + '%';
}

function animateBar(target) {
  uploadBarFill.style.width = target + '%';
}

function renderWelcomeBack(filename, ocrPages = 0, visualPages = 0) {
  const div = document.createElement('div');
  div.className = 'welcome-message';
  div.style.flex = '1';

  const ocrNote = ocrPages > 0
    ? `<p style="margin-top:8px;opacity:0.7;font-size:13px;">✍️ Detected <strong>${ocrPages}</strong> handwritten page(s) — AI Vision was used to extract text.</p>`
    : '';

  const visualNote = visualPages > 0
    ? `<p style="margin-top:4px;opacity:0.7;font-size:13px;">🖼️ Analyzed <strong>${visualPages}</strong> page(s) for diagrams, tables, charts & images.</p>`
    : '';

  div.innerHTML = `
    <div class="welcome-glow"></div>
    <div class="welcome-icon" style="font-size:48px;">✅</div>
    <h3>${escapeHtml(filename)} is ready!</h3>
    <p>The document has been indexed with full visual content analysis. Ask me anything about the text, diagrams, tables, or images.</p>
    ${ocrNote}
    ${visualNote}
  `;
  messagesWindow.appendChild(div);
}

// ---------------------------------------------------------------------------
// Memory Clear
// ---------------------------------------------------------------------------

async function clearMemory() {
  if (!confirm('Clear the entire conversation history?')) return;

  try {
    await fetch(`${BASE_URL}/memory`, { method: 'DELETE' });
    messagesWindow.innerHTML = '';
    if (currentDocumentName) {
      renderWelcomeBack(currentDocumentName);
    } else {
      const welcome = document.createElement('div');
      welcome.id = 'welcome-message';
      welcome.className = 'welcome-message';
      welcome.innerHTML = `
        <div class="welcome-glow"></div>
        <div class="welcome-icon" style="font-size:48px;">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--primary)" stroke-width="1.5">
            <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2z"/>
            <path d="M12 16v-4"/>
            <path d="M12 8h.01"/>
          </svg>
        </div>
        <h3>Hello, Umesh!</h3>
        <p>Upload a PDF in the sidebar and ask me anything about it.</p>
      `;
      messagesWindow.appendChild(welcome);
    }
  } catch (err) {
    console.error('[VisionRAG] Clear memory error:', err);
    alert('Failed to clear memory. Is the backend running?');
  }
}

// ---------------------------------------------------------------------------
// Sidebar Toggle (Mobile)
// ---------------------------------------------------------------------------

function toggleSidebar() {
  sidebar.classList.toggle('open');
}

// ---------------------------------------------------------------------------
// Event Listeners
// ---------------------------------------------------------------------------

sendBtn.addEventListener('click', sendMessage);

chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

chatInput.addEventListener('input', autoGrowTextarea);

clearBtn.addEventListener('click', clearMemory);

if (sidebarToggle) {
  sidebarToggle.addEventListener('click', toggleSidebar);
}

dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) {
    uploadPdf(file);
    fileInput.value = '';
  }
});

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) uploadPdf(file);
});

// Close sidebar on outside click (mobile)
document.addEventListener('click', (e) => {
  if (sidebar.classList.contains('open') &&
      !sidebar.contains(e.target) &&
      !sidebarToggle.contains(e.target)) {
    sidebar.classList.remove('open');
  }
});

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

async function init() {
  try {
    const res = await fetch(`${BASE_URL}/health`);
    if (!res.ok) return;
    const data = await res.json();
    if (data.index_ready && data.current_pdf) {
      const name = data.current_pdf.split(/[/\\]/).pop();
      currentDocumentName = name;
      docCardName.textContent = name;
      docCardMeta.textContent = `Already indexed · ${data.conversation_turns} turns in memory`;
      docCard.classList.add('visible');
      headerMeta.textContent = name;
      setStatusActive(true);
      uploadStatus.style.display = 'flex';
      setUploadStatus('✓ Existing index loaded from disk', 100, 'success');
    }
  } catch {
    console.warn('[VisionRAG] Backend not reachable on init. Start the server.');
  }
}

init();
