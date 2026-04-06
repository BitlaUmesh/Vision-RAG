"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import { ChevronLeft, ChevronRight, ZoomIn, ZoomOut, Maximize } from "lucide-react";
import { Button } from "@/components/ui/button";

import 'react-pdf/dist/Page/AnnotationLayer.css';
import 'react-pdf/dist/Page/TextLayer.css';

// Configure the worker for react-pdf
pdfjs.GlobalWorkerOptions.workerSrc = `//unpkg.com/pdfjs-dist@${pdfjs.version}/build/pdf.worker.min.mjs`;

export default function PdfViewer({ currentPdf }: { currentPdf: string | null }) {
  const [numPages, setNumPages] = useState<number>(0);
  const [pageNumber, setPageNumber] = useState<number>(1);
  const [scale, setScale] = useState<number>(1.2);
  const [containerWidth, setContainerWidth] = useState<number>(800);
  const [highlightQuote, setHighlightQuote] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleHighlight = (e: CustomEvent<{ page: number; quote: string }>) => {
      setPageNumber(e.detail.page);
      setHighlightQuote(e.detail.quote);
      // Optional: clear highlight after 10 seconds
      setTimeout(() => setHighlightQuote(null), 10000);
    };

    window.addEventListener("highlight-pdf", handleHighlight as EventListener);
    return () => window.removeEventListener("highlight-pdf", handleHighlight as EventListener);
  }, []);

  useEffect(() => {
    // Simple resize observer to make the PDF responsive to the container
    const observer = new ResizeObserver((entries) => {
      window.requestAnimationFrame(() => {
        if (!entries || !entries.length) return;
        const entry = entries[0];
        if (entry.contentRect.width > 0) {
          // Debounce width changes to prevent ResizeObserver loop limit exceeded
          setContainerWidth((prev) => {
            const newWidth = entry.contentRect.width - 40; // 40px padding
            return Math.abs(prev - newWidth) > 5 ? newWidth : prev;
          });
        }
      });
    });

    if (containerRef.current) {
      observer.observe(containerRef.current);
    }
    
    return () => observer.disconnect();
  }, []);

  function onDocumentLoadSuccess({ numPages }: { numPages: number }) {
    setNumPages(numPages);
    setPageNumber(1);
    setHighlightQuote(null);
  }

  const highlightRenderer = useCallback(
    (textItem: { str: string; itemIndex: number }) => {
      if (!highlightQuote) return textItem.str;
      
      const text = textItem.str;
      const lowerText = text.toLowerCase();
      const lowerQuote = highlightQuote.toLowerCase();
      
      // Look for any overlap or partial match in this text span
      // PDF text spans are fragmented, so this handles partial highlights
      if (lowerText.includes(lowerQuote)) {
        const index = lowerText.indexOf(lowerQuote);
        return (
          text.substring(0, index) +
          `<mark class="bg-yellow-300 dark:bg-yellow-500/60 rounded-sm text-inherit font-inherit px-[2px] shadow-sm animate-pulse">${text.substring(index, index + lowerQuote.length)}</mark>` +
          text.substring(index + lowerQuote.length)
        );
      }
      
      // Fallback: if the quote is very long and spans multiple `textItem.str` elements,
      // a simpler heuristic is to check if this span string is fully inside the quote
      if (lowerQuote.includes(lowerText.trim()) && lowerText.trim().length > 4) {
         return `<mark class="bg-yellow-300 dark:bg-yellow-500/60 rounded-sm text-inherit font-inherit px-[2px] shadow-sm animate-pulse">${text}</mark>`;
      }
      
      return text;
    },
    [highlightQuote]
  );

  if (!currentPdf) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground flex-col gap-4 bg-muted/10">
        <div className="w-24 h-32 border-2 border-dashed border-border rounded-lg flex items-center justify-center bg-background/50 shadow-sm">
          <span className="text-4xl text-border">PDF</span>
        </div>
        <p className="font-medium text-sm">Select a document from the sidebar or upload a new one</p>
      </div>
    );
  }

  const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
  const pdfUrl = `${API}/pdf/${currentPdf}`;

  return (
    <div className="h-full flex flex-col relative" ref={containerRef}>
      {/* Top Toolbar */}
      <div className="h-14 border-b border-border/50 bg-background/80 backdrop-blur-md flex items-center justify-between px-4 z-10 shadow-sm absolute top-0 w-full">
        <h3 className="font-medium text-sm truncate max-w-sm text-foreground/80">{currentPdf}</h3>
        
        <div className="flex items-center gap-1 bg-muted/60 p-1 rounded-md border border-border/50">
          <Button 
            variant="ghost" 
            size="icon" 
            className="h-7 w-7 rounded-sm" 
            onClick={() => setPageNumber(p => Math.max(1, p - 1))}
            disabled={pageNumber <= 1}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <span className="text-xs font-medium px-2 w-20 text-center">
            {pageNumber} / {numPages || '-'}
          </span>
          <Button 
            variant="ghost" 
            size="icon" 
            className="h-7 w-7 rounded-sm" 
            onClick={() => setPageNumber(p => Math.min(numPages, p + 1))}
            disabled={pageNumber >= numPages}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>

        <div className="flex items-center gap-1">
          <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground" onClick={() => setScale(s => Math.max(0.5, s - 0.2))}>
            <ZoomOut className="h-4 w-4" />
          </Button>
          <span className="text-xs font-mono w-12 text-center text-muted-foreground">{Math.round(scale * 100)}%</span>
          <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground" onClick={() => setScale(s => Math.min(3, s + 0.2))}>
            <ZoomIn className="h-4 w-4" />
          </Button>
          <div className="w-px h-4 bg-border mx-1" />
          <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground" onClick={() => setScale(1.0)}>
            <Maximize className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* PDF Viewport */}
      <div className="flex-1 overflow-auto mt-14 bg-muted/30 flex flex-col">
        <div className="w-max m-auto py-8 px-4 relative">
          <Document
            file={pdfUrl}
            onLoadSuccess={onDocumentLoadSuccess}
            loading={
              <div className="w-[600px] h-[800px] bg-background animate-pulse rounded-lg shadow-sm border border-border/50 flex flex-col items-center justify-center text-muted-foreground text-sm">
                Loading Document...
              </div>
            }
            error={
              <div className="text-destructive bg-destructive/10 p-4 rounded-md border border-destructive/20 text-sm">
                Failed to load PDF file.
              </div>
            }
          >
            <div className="bg-background shadow-xl border border-border/30 transition-transform origin-top z-0 relative group">
              <Page 
                pageNumber={pageNumber} 
                scale={scale}
                width={containerWidth ? containerWidth * 0.9 : undefined}
                className="overflow-hidden"
                renderTextLayer={true}
                renderAnnotationLayer={true}
                customTextRenderer={highlightRenderer}
              />
            </div>
          </Document>
        </div>
      </div>
    </div>
  );
}
