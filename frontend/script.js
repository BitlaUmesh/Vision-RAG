/**
 * script.js — Vision RAG Frontend Logic (v2 — Streaming + PDF Viewer)
 * =====================================================================
 * Handles:
 *   - PDF upload with drag-and-drop, progress feedback
 *   - **SSE streaming** chat — tokens rendered in real-time
 *   - Markdown rendering of AI responses (incremental during stream)
 *   - **PDF.js viewer** — renders uploaded PDF in split pane
 *   - **Clickable page citations** — scroll PDF to cited page
 *   - **Drag-to-resize** split pane
 *   - RAG metadata badges on each AI message
 *   - Memory clear functionality
 *   - Sidebar toggle for mobile
 *
 * Architecture: ES2022 modules, no build step, no framework.
 * Communicates with FastAPI backend at BASE_URL.
 */

'use strict';

// ---------------------------------------------------------------------------
// PDF.js Setup (ESM import)
// ---------------------------------------------------------------------------
import * as pdfjsLib from 'https://cdn.jsdelivr.net/npm/pdfjs-dist@4.5.136/build/pdf.min.mjs';
pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@4.5.136/build/pdf.worker.min.mjs';

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

// Split pane & PDF viewer elements
const splitPane          = document.getElementById('split-pane');
const chatPanel          = document.getElementById('chat-panel');
const pdfPanel           = document.getElementById('pdf-panel');
const resizeHandle       = document.getElementById('resize-handle');
const pdfToggleBtn       = document.getElementById('pdf-toggle-btn');
const pdfToggleLabel     = document.getElementById('pdf-toggle-label');
const pdfToolbarTitle    = document.getElementById('pdf-toolbar-title');
const pdfZoomLevel       = document.getElementById('pdf-zoom-level');
const pdfPageInfo        = document.getElementById('pdf-page-info');
const pdfCanvasContainer = document.getElementById('pdf-canvas-container');
const pdfEmptyState      = document.getElementById('pdf-empty-state');
const pdfCloseBtn        = document.getElementById('pdf-close-btn');
const pdfZoomIn          = document.getElementById('pdf-zoom-in');
const pdfZoomOut         = document.getElementById('pdf-zoom-out');
const pdfPrevPage        = document.getElementById('pdf-prev-page');
const pdfNextPage        = document.getElementById('pdf-next-page');

// ---------------------------------------------------------------------------
// Application State
// ---------------------------------------------------------------------------

let isLoading = false;
let currentDocumentName = '';

// PDF viewer state
let pdfDoc = null;
let pdfCurrentPage = 1;
let pdfTotalPages = 0;
let pdfScale = 1.0;
let pdfPanelVisible = false;
let pdfRenderedPages = new Map(); // page_num -> canvas element
let highlightedPage = null;

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

/**
 * Creates a streaming AI message container and returns an object
 * with methods to append tokens and finalize the message.
 */
function createStreamingMessage() {
  hideWelcome();

  const wrapper = document.createElement('div');
  wrapper.className = 'message ai';

  const bubble = document.createElement('div');
  bubble.className = 'message-bubble';

  const cursor = document.createElement('span');
  cursor.className = 'streaming-cursor';

  bubble.appendChild(cursor);

  wrapper.innerHTML = `
    <div class="message-avatar">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2z"/>
        <path d="M12 16v-4"/>
        <path d="M12 8h.01"/>
      </svg>
    </div>
    <div class="message-content"></div>
  `;

  wrapper.querySelector('.message-content').appendChild(bubble);
  messagesWindow.appendChild(wrapper);

  let accumulated = '';
  let renderTimer = null;

  function renderMarkdown() {
    const html = marked.parse(accumulated);
    bubble.innerHTML = html;
    bubble.appendChild(cursor);
    scrollToBottom();
  }

  return {
    appendToken(token) {
      accumulated += token;
      // Throttle markdown re-renders to every 60ms for performance
      if (!renderTimer) {
        renderTimer = setTimeout(() => {
          renderMarkdown();
          renderTimer = null;
        }, 60);
      }
    },

    finalize(needsRag, pages, reason) {
      // Final render
      if (renderTimer) {
        clearTimeout(renderTimer);
        renderTimer = null;
      }

      const html = marked.parse(accumulated);
      bubble.innerHTML = html;

      // Remove cursor
      const existingCursor = bubble.querySelector('.streaming-cursor');
      if (existingCursor) existingCursor.remove();

      // Add metadata tags
      const metaDiv = document.createElement('div');
      metaDiv.className = 'message-meta';

      const ragTagClass = needsRag ? 'rag-active' : 'no-rag';
      const ragTagText = needsRag
        ? `📄 RAG · ${formatPages(pages)}`
        : '💬 Chat-only';

      metaDiv.innerHTML = `
        <span class="meta-tag ${ragTagClass}">${ragTagText}</span>
        ${needsRag && pages && pages.length > 0 ? `<span class="meta-tag visual-tag">🖼️ Visual</span>` : ''}
        <span class="meta-tag" title="${escapeHtml(reason)}">🔀 ${escapeHtml(reason.split('—')[0].trim())}</span>
      `;

      // Add clickable page citation badges
      if (pages && pages.length > 0) {
        pages.forEach(pageNum => {
          const citationTag = document.createElement('span');
          citationTag.className = 'meta-tag page-citation';
          citationTag.textContent = `📑 Page ${pageNum}`;
          citationTag.title = `View page ${pageNum} in PDF viewer`;
          citationTag.addEventListener('click', () => {
            showPdfPanel();
            scrollPdfToPage(pageNum);
          });
          metaDiv.appendChild(citationTag);
        });
      }

      wrapper.querySelector('.message-content').appendChild(metaDiv);
      scrollToBottom();
    },

    getAccumulated() {
      return accumulated;
    }
  };
}

function appendAiMessage(markdown, needsRag, pages, reason) {
  const renderedHtml = marked.parse(markdown);

  const ragTagClass = needsRag ? 'rag-active' : 'no-rag';
  const ragTagText  = needsRag
    ? `📄 RAG · ${formatPages(pages)}`
    : '💬 Chat-only';

  const visualTag = (needsRag && pages && pages.length > 0)
    ? `<span class="meta-tag visual-tag">🖼️ Visual</span>`
    : '';

  const wrapper = document.createElement('div');
  wrapper.className = 'message ai';

  let citationTags = '';
  if (pages && pages.length > 0) {
    citationTags = pages.map(p =>
      `<span class="meta-tag page-citation" data-page="${p}" title="View page ${p} in PDF viewer">📑 Page ${p}</span>`
    ).join('');
  }

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
        ${citationTags}
      </div>
    </div>
  `;

  // Bind click handlers on citation tags
  wrapper.querySelectorAll('.page-citation').forEach(tag => {
    tag.addEventListener('click', () => {
      const pageNum = parseInt(tag.dataset.page, 10);
      showPdfPanel();
      scrollPdfToPage(pageNum);
    });
  });

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
// Chat Logic — SSE Streaming
// ---------------------------------------------------------------------------

async function sendMessage() {
  const message = chatInput.value.trim();

  if (!message || isLoading) return;

  chatInput.value = '';
  autoGrowTextarea();

  appendUserMessage(message);
  setInputEnabled(false);

  // Create streaming message container
  const streamMsg = createStreamingMessage();

  try {
    const response = await fetch(`${BASE_URL}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(errBody.detail || `HTTP ${response.status}`);
    }

    // Read the SSE stream
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Process complete SSE events (delimited by double newlines)
      const events = buffer.split('\n\n');
      buffer = events.pop(); // Keep incomplete chunk in buffer

      for (const event of events) {
        const line = event.trim();
        if (!line.startsWith('data: ')) continue;

        try {
          const data = JSON.parse(line.slice(6));

          if (data.type === 'token') {
            streamMsg.appendToken(data.content);
          } else if (data.type === 'done') {
            streamMsg.finalize(
              data.needs_rag,
              data.retrieved_pages,
              data.routing_reason,
            );
          }
        } catch (parseErr) {
          console.warn('[VisionRAG] SSE parse error:', parseErr);
        }
      }
    }

    // Process any remaining buffer
    if (buffer.trim()) {
      const line = buffer.trim();
      if (line.startsWith('data: ')) {
        try {
          const data = JSON.parse(line.slice(6));
          if (data.type === 'token') {
            streamMsg.appendToken(data.content);
          } else if (data.type === 'done') {
            streamMsg.finalize(
              data.needs_rag,
              data.retrieved_pages,
              data.routing_reason,
            );
          }
        } catch (_) {}
      }
    }

  } catch (err) {
    // If streaming fails, try fallback to non-streaming endpoint
    console.warn('[VisionRAG] Streaming failed, trying fallback:', err.message);
    try {
      const fallbackResp = await fetch(`${BASE_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
      });

      if (!fallbackResp.ok) throw new Error(`HTTP ${fallbackResp.status}`);

      const data = await fallbackResp.json();
      // Remove the empty streaming message
      const lastMsg = messagesWindow.querySelector('.message.ai:last-child');
      if (lastMsg) lastMsg.remove();

      appendAiMessage(data.response, data.needs_rag, data.retrieved_pages, data.routing_reason);
    } catch (fallbackErr) {
      // Remove the empty streaming message
      const lastMsg = messagesWindow.querySelector('.message.ai:last-child');
      if (lastMsg) lastMsg.remove();
      appendErrorMessage(err.message || 'Unknown error. Is the backend running?');
    }
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
    setOverlayStep('upload', 10, 'Uploading document to server...');

    const response = await fetch(`${BASE_URL}/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({ detail: response.statusText }));
      throw new Error(errBody.detail || `HTTP ${response.status}`);
    }

    const data = await response.json();
    const jobId = data.job_id;

    // Polling loop
    const checkStatus = async () => {
      try {
        const res = await fetch(`${BASE_URL}/indexing/status/${jobId}`);
        if (!res.ok) throw new Error('Failed to get indexing status');
        const statusData = await res.json();

        // Error from background job
        if (statusData.status === 'error') {
          throw new Error(statusData.message || 'Unknown background indexing error');
        }

        // Update UI step based on status returned from backend
        let stepName = statusData.status;
        if (stepName === 'text') stepName = 'ocr';
        setOverlayStep(stepName, statusData.progress, statusData.message);

        // Completion logic
        if (statusData.status === 'completed') {
          uploadOverlay.classList.add('success');
          overlayTitle.textContent = '✅ Document Ready!';

          const ocrPages = statusData.ocr_pages || 0;
          const visualPages = statusData.visual_pages || 0;
          const ocrInfo = ocrPages > 0 ? ` · ✍️ ${ocrPages} handwritten page(s) detected` : '';
          const visualInfo = visualPages > 0 ? ` · 🖼️ ${visualPages} page(s) with visual content` : '';

          // Update sidebar
          uploadStatus.style.display = 'flex';
          const indexMsg = statusData.message; // already constructed in backend "Indexed X pages..."
          setUploadStatus(`✓ ${indexMsg}`, 100, 'success');
          currentDocumentName = data.filename;
          docCardName.textContent = data.filename;
          docCardMeta.textContent = indexMsg;
          docCard.classList.add('visible');
          headerMeta.textContent = `${data.filename}${ocrInfo}${visualInfo}`;
          setStatusActive(true);

          messagesWindow.innerHTML = '';
          renderWelcomeBack(data.filename, ocrPages, visualPages);

          console.info(`[VisionRAG] Uploaded & indexed: ${data.filename} (OCR pages: ${ocrPages})`);

          // Load PDF into viewer
          await loadPdfInViewer();

          // Auto-show PDF panel after upload
          showPdfPanel();

          // Auto-dismiss overlay after 1.5s
          setTimeout(() => hideOverlay(), 1500);
        } else {
          // Keep polling if not done
          setTimeout(checkStatus, 500);
        }
      } catch (pollErr) {
        overlayTitle.textContent = '❌ Indexing Failed';
        overlayStatusText.textContent = pollErr.message;
        overlayProgressFill.style.width = '0%';
        overlayStatusText.style.color = 'var(--danger)';
        uploadStatus.style.display = 'flex';
        setUploadStatus(`✗ Indexing failed: ${pollErr.message}`, 0, 'error');
        console.error('[VisionRAG] Indexing error:', pollErr);
        setTimeout(() => hideOverlay(), 4000);
      }
    };

    // Kick off polling
    checkStatus();

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
// PDF Viewer Logic
// ---------------------------------------------------------------------------

async function loadPdfInViewer() {
  try {
    pdfDoc = await pdfjsLib.getDocument(`${BASE_URL}/pdf/current`).promise;
    pdfTotalPages = pdfDoc.numPages;
    pdfCurrentPage = 1;

    pdfToolbarTitle.textContent = currentDocumentName || 'Document';
    updatePdfPageInfo();

    // Clear previous renders
    pdfCanvasContainer.innerHTML = '';
    pdfRenderedPages.clear();
    if (pdfEmptyState) pdfEmptyState.remove();

    // Render all pages
    for (let i = 1; i <= pdfTotalPages; i++) {
      await renderPdfPage(i);
    }

    console.info(`[VisionRAG] PDF loaded in viewer: ${pdfTotalPages} pages`);
  } catch (err) {
    console.error('[VisionRAG] Failed to load PDF in viewer:', err);
    pdfToolbarTitle.textContent = 'Failed to load PDF';
  }
}

async function renderPdfPage(pageNum) {
  try {
    const page = await pdfDoc.getPage(pageNum);
    const viewport = page.getViewport({ scale: pdfScale });

    const wrapper = document.createElement('div');
    wrapper.className = 'pdf-page-wrapper';
    wrapper.id = `pdf-page-${pageNum}`;
    wrapper.dataset.page = pageNum;

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Use device pixel ratio for crisp rendering
    const dpr = window.devicePixelRatio || 1;
    canvas.width = viewport.width * dpr;
    canvas.height = viewport.height * dpr;
    canvas.style.width = viewport.width + 'px';
    canvas.style.height = viewport.height + 'px';
    ctx.scale(dpr, dpr);

    const label = document.createElement('div');
    label.className = 'pdf-page-label';
    label.textContent = `Page ${pageNum}`;

    wrapper.appendChild(canvas);
    wrapper.appendChild(label);
    pdfCanvasContainer.appendChild(wrapper);
    pdfRenderedPages.set(pageNum, wrapper);

    await page.render({
      canvasContext: ctx,
      viewport: viewport,
    }).promise;

  } catch (err) {
    console.warn(`[VisionRAG] Failed to render page ${pageNum}:`, err);
  }
}

async function reRenderAllPages() {
  if (!pdfDoc) return;

  pdfCanvasContainer.innerHTML = '';
  pdfRenderedPages.clear();

  for (let i = 1; i <= pdfTotalPages; i++) {
    await renderPdfPage(i);
  }

  updatePdfPageInfo();
}

function scrollPdfToPage(pageNum) {
  const wrapper = pdfRenderedPages.get(pageNum);
  if (!wrapper) return;

  // Remove previous highlight
  if (highlightedPage !== null) {
    const prev = pdfRenderedPages.get(highlightedPage);
    if (prev) prev.classList.remove('highlighted');
  }

  // Highlight and scroll
  wrapper.classList.add('highlighted');
  highlightedPage = pageNum;
  wrapper.scrollIntoView({ behavior: 'smooth', block: 'center' });
  pdfCurrentPage = pageNum;
  updatePdfPageInfo();

  // Remove highlight after 3 seconds
  setTimeout(() => {
    wrapper.classList.remove('highlighted');
    if (highlightedPage === pageNum) highlightedPage = null;
  }, 3000);
}

function updatePdfPageInfo() {
  pdfPageInfo.textContent = `${pdfCurrentPage} / ${pdfTotalPages || '—'}`;
  pdfZoomLevel.textContent = `${Math.round(pdfScale * 100)}%`;
}

// PDF panel visibility
function showPdfPanel() {
  if (pdfPanelVisible) return;
  pdfPanelVisible = true;
  splitPane.classList.add('pdf-visible');
  pdfToggleBtn.classList.add('active');
  pdfToggleLabel.textContent = 'Hide PDF';
}

function hidePdfPanel() {
  pdfPanelVisible = false;
  splitPane.classList.remove('pdf-visible');
  pdfToggleBtn.classList.remove('active');
  pdfToggleLabel.textContent = 'Show PDF';
}

function togglePdfPanel() {
  if (pdfPanelVisible) {
    hidePdfPanel();
  } else {
    showPdfPanel();
  }
}

// PDF zoom controls
function zoomIn() {
  if (pdfScale >= 3.0) return;
  pdfScale = Math.min(3.0, pdfScale + 0.25);
  reRenderAllPages();
}

function zoomOut() {
  if (pdfScale <= 0.5) return;
  pdfScale = Math.max(0.5, pdfScale - 0.25);
  reRenderAllPages();
}

// PDF page navigation
function goToPrevPage() {
  if (pdfCurrentPage <= 1) return;
  pdfCurrentPage--;
  scrollPdfToPage(pdfCurrentPage);
}

function goToNextPage() {
  if (pdfCurrentPage >= pdfTotalPages) return;
  pdfCurrentPage++;
  scrollPdfToPage(pdfCurrentPage);
}

// Track current page from scroll position
function updateCurrentPageFromScroll() {
  if (!pdfTotalPages) return;

  const containerRect = pdfCanvasContainer.getBoundingClientRect();
  const containerMid = containerRect.top + containerRect.height / 2;
  let closestPage = 1;
  let closestDist = Infinity;

  pdfRenderedPages.forEach((wrapper, pageNum) => {
    const rect = wrapper.getBoundingClientRect();
    const pageMid = rect.top + rect.height / 2;
    const dist = Math.abs(pageMid - containerMid);
    if (dist < closestDist) {
      closestDist = dist;
      closestPage = pageNum;
    }
  });

  if (closestPage !== pdfCurrentPage) {
    pdfCurrentPage = closestPage;
    updatePdfPageInfo();
  }
}

// ---------------------------------------------------------------------------
// Drag-to-Resize Split Pane
// ---------------------------------------------------------------------------

let isResizing = false;

function startResize(e) {
  if (!pdfPanelVisible) return;
  isResizing = true;
  resizeHandle.classList.add('dragging');
  document.body.style.cursor = 'col-resize';
  document.body.style.userSelect = 'none';
  e.preventDefault();
}

function doResize(e) {
  if (!isResizing) return;

  const containerRect = splitPane.getBoundingClientRect();
  const mouseX = e.clientX - containerRect.left;
  const totalWidth = containerRect.width;

  // Constrain: min 30% for chat, min 20% for PDF
  const chatWidth = Math.max(totalWidth * 0.3, Math.min(totalWidth * 0.8, mouseX));
  const pdfWidth = totalWidth - chatWidth - 6; // 6px for handle

  chatPanel.style.flex = 'none';
  chatPanel.style.width = chatWidth + 'px';
  pdfPanel.style.width = pdfWidth + 'px';
  pdfPanel.style.minWidth = '0';
}

function stopResize() {
  if (!isResizing) return;
  isResizing = false;
  resizeHandle.classList.remove('dragging');
  document.body.style.cursor = '';
  document.body.style.userSelect = '';
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

// Chat
sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
chatInput.addEventListener('input', autoGrowTextarea);

// Clear memory
clearBtn.addEventListener('click', clearMemory);

// Sidebar
if (sidebarToggle) {
  sidebarToggle.addEventListener('click', toggleSidebar);
}

// File upload
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

// PDF viewer controls
pdfToggleBtn.addEventListener('click', togglePdfPanel);
pdfCloseBtn.addEventListener('click', hidePdfPanel);
pdfZoomIn.addEventListener('click', zoomIn);
pdfZoomOut.addEventListener('click', zoomOut);
pdfPrevPage.addEventListener('click', goToPrevPage);
pdfNextPage.addEventListener('click', goToNextPage);

// Resize handle
resizeHandle.addEventListener('mousedown', startResize);
document.addEventListener('mousemove', doResize);
document.addEventListener('mouseup', stopResize);

// Track PDF page from scroll
pdfCanvasContainer.addEventListener('scroll', () => {
  requestAnimationFrame(updateCurrentPageFromScroll);
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

      // Load PDF into viewer in background
      loadPdfInViewer();
    }
  } catch {
    console.warn('[VisionRAG] Backend not reachable on init. Start the server.');
  }
}

init();
