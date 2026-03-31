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
  breaks: true,       // Convert \n to <br> inside paragraphs
  gfm: true,          // GitHub Flavored Markdown (tables, strikethrough, etc.)
  headerIds: false,   // Don't add id attributes to headers (cleaner HTML)
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

// ---------------------------------------------------------------------------
// Application State
// ---------------------------------------------------------------------------

let isLoading = false;         // Prevents double sends while waiting
let currentDocumentName = '';  // Filename of the currently indexed PDF

// ---------------------------------------------------------------------------
// Utility Functions
// ---------------------------------------------------------------------------

/**
 * Auto-grow the textarea as the user types, up to max-height defined in CSS.
 */
function autoGrowTextarea() {
  chatInput.style.height = 'auto';
  chatInput.style.height = chatInput.scrollHeight + 'px';
}

/**
 * Scroll the messages window to the very bottom.
 * Uses requestAnimationFrame to ensure DOM has painted before scrolling.
 */
function scrollToBottom() {
  requestAnimationFrame(() => {
    messagesWindow.scrollTop = messagesWindow.scrollHeight;
  });
}

/**
 * Hide the welcome screen once the first message appears.
 */
function hideWelcome() {
  if (welcomeMessage) {
    welcomeMessage.style.display = 'none';
  }
}

/**
 * Format a page list as a human-readable string.
 * e.g. [1, 3, 7] → "pages 1, 3, 7"
 */
function formatPages(pages) {
  if (!pages || pages.length === 0) return 'no pages';
  return 'pages ' + pages.join(', ');
}

/**
 * Set the enabled/disabled state of the send button and input.
 */
function setInputEnabled(enabled) {
  chatInput.disabled = !enabled;
  sendBtn.disabled   = !enabled;
  isLoading          = !enabled;
}

// ---------------------------------------------------------------------------
// Message Rendering
// ---------------------------------------------------------------------------

/**
 * Append a user message bubble to the chat window.
 * @param {string} text - The raw user message.
 */
function appendUserMessage(text) {
  hideWelcome();

  const wrapper = document.createElement('div');
  wrapper.className = 'message user';
  wrapper.innerHTML = `
    <div class="message-avatar">👤</div>
    <div class="message-content">
      <div class="message-bubble">${escapeHtml(text)}</div>
    </div>
  `;
  messagesWindow.appendChild(wrapper);
  scrollToBottom();
}

/**
 * Append an AI response bubble with Markdown rendering and metadata tags.
 * @param {string} markdown   - Raw Markdown string from the backend.
 * @param {boolean} needsRag  - Whether retrieval was used.
 * @param {number[]} pages    - Retrieved page numbers.
 * @param {string} reason     - Routing reason string.
 */
function appendAiMessage(markdown, needsRag, pages, reason) {
  const renderedHtml = marked.parse(markdown);

  const ragTagClass = needsRag ? 'rag-active' : 'no-rag';
  const ragTagText  = needsRag
    ? `📄 RAG · ${formatPages(pages)}`
    : '💬 Chat-only (no retrieval)';

  const wrapper = document.createElement('div');
  wrapper.className = 'message ai';
  wrapper.innerHTML = `
    <div class="message-avatar">🤖</div>
    <div class="message-content">
      <div class="message-bubble">${renderedHtml}</div>
      <div class="message-meta">
        <span class="meta-tag ${ragTagClass}">${ragTagText}</span>
        <span class="meta-tag" title="${escapeHtml(reason)}">🔀 ${escapeHtml(reason.split('—')[0].trim())}</span>
      </div>
    </div>
  `;
  messagesWindow.appendChild(wrapper);
  scrollToBottom();
}

/**
 * Show the animated "thinking" loading indicator.
 * Returns the element so we can remove it later.
 */
function showLoadingIndicator() {
  const el = document.createElement('div');
  el.className = 'loading-message';
  el.id = 'loading-indicator';
  el.innerHTML = `
    <div class="message-avatar">🤖</div>
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

/**
 * Remove the loading indicator from the DOM.
 */
function removeLoadingIndicator() {
  const el = document.getElementById('loading-indicator');
  if (el) el.remove();
}

/**
 * Append an error bubble to inform the user of a failure.
 */
function appendErrorMessage(errorText) {
  const wrapper = document.createElement('div');
  wrapper.className = 'message ai';
  wrapper.innerHTML = `
    <div class="message-avatar">⚠️</div>
    <div class="message-content">
      <div class="message-bubble" style="border-color: var(--danger); color: var(--danger);">
        <strong>Error:</strong> ${escapeHtml(errorText)}
      </div>
    </div>
  `;
  messagesWindow.appendChild(wrapper);
  scrollToBottom();
}

/**
 * Escape HTML special characters to prevent XSS when inserting user text.
 */
function escapeHtml(str) {
  const div = document.createElement('div');
  div.appendChild(document.createTextNode(str));
  return div.innerHTML;
}

// ---------------------------------------------------------------------------
// Chat Logic
// ---------------------------------------------------------------------------

/**
 * Send the user's message to the /chat endpoint and display the response.
 */
async function sendMessage() {
  const message = chatInput.value.trim();

  if (!message || isLoading) return;

  // Clear the input immediately for better UX
  chatInput.value = '';
  autoGrowTextarea();

  // Render user bubble
  appendUserMessage(message);

  // Show loading indicator and lock input
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
      // Try to parse error detail from FastAPI
      const errBody = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(errBody.detail || `HTTP ${response.status}`);
    }

    const data = await response.json();

    // Render AI response with metadata
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

/**
 * Handle the PDF file upload: POST to /upload, show progress, update UI.
 * @param {File} file - The PDF File object chosen by the user.
 */
async function uploadPdf(file) {
  if (!file || !file.name.toLowerCase().endsWith('.pdf')) {
    alert('Please select a valid PDF file.');
    return;
  }

  // Show progress bar
  uploadStatus.style.display = 'flex';
  docCard.classList.remove('visible');
  setUploadStatus('Uploading...', 0, '');
  animateBar(30); // Fake early progress

  const formData = new FormData();
  formData.append('file', file);

  try {
    animateBar(60); // Fake mid-progress (indexing is slow)
    setUploadStatus('Indexing pages — this may take a minute...', 60, '');

    const response = await fetch(`${BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(errBody.detail || `HTTP ${response.status}`);
    }

    const data = await response.json();

    animateBar(100);
    setUploadStatus(`✓ ${data.index_status}`, 100, 'success');

    // Update document card
    currentDocumentName = data.filename;
    docCardName.textContent = data.filename;
    docCardMeta.textContent = data.index_status;
    docCard.classList.add('visible');
    headerMeta.textContent = `Indexed: ${data.filename}`;

    // Clear conversation since this is a new document
    messagesWindow.innerHTML = '';
    renderWelcomeBack(data.filename);

    console.info(`[VisionRAG] Uploaded & indexed: ${data.filename}`);

  } catch (err) {
    animateBar(0);
    setUploadStatus(`✗ Upload failed: ${err.message}`, 0, 'error');
    console.error('[VisionRAG] Upload error:', err);
  }
}

function setUploadStatus(text, barPercent, cssClass) {
  uploadStatusText.textContent = text;
  uploadStatusText.className = 'upload-status-text ' + cssClass;
  uploadBarFill.style.width = barPercent + '%';
}

function animateBar(target) {
  uploadBarFill.style.width = target + '%';
}

/**
 * Show a contextual welcome message after a document is indexed.
 */
function renderWelcomeBack(filename) {
  const div = document.createElement('div');
  div.className = 'welcome-message';
  div.style.flex = '1';
  div.innerHTML = `
    <div class="welcome-icon">✅</div>
    <h3>${escapeHtml(filename)} is ready!</h3>
    <p>The document has been indexed. Ask me anything about it below.</p>
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
      // Restore original welcome
      const welcome = document.createElement('div');
      welcome.id = 'welcome-message';
      welcome.className = 'welcome-message';
      welcome.innerHTML = `
        <div class="welcome-icon">🤖</div>
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
// Event Listeners
// ---------------------------------------------------------------------------

// Send on button click
sendBtn.addEventListener('click', sendMessage);

// Send on Enter; Shift+Enter inserts newline
chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Auto-grow textarea
chatInput.addEventListener('input', autoGrowTextarea);

// Clear memory button
clearBtn.addEventListener('click', clearMemory);

// Click on drop zone → open file picker
dropZone.addEventListener('click', () => fileInput.click());

// File picker selection
fileInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) {
    uploadPdf(file);
    fileInput.value = ''; // Reset so same file can be re-uploaded
  }
});

// Drag and drop support
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

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

/**
 * On page load, ping the health endpoint to show current index status.
 */
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
      headerMeta.textContent = `Indexed: ${name}`;
      uploadStatus.style.display = 'flex';
      setUploadStatus('✓ Existing index loaded from disk', 100, 'success');
    }
  } catch {
    // Backend may not be running yet; fail silently on init
    console.warn('[VisionRAG] Backend not reachable on init. Start the server.');
  }
}

init();
