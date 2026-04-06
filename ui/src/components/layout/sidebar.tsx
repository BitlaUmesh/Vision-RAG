"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { createPortal } from "react-dom";
import { useRouter } from "next/navigation";
import { createClient } from "@/utils/supabase/client";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import {
  FileText,
  UploadCloud,
  Trash2,
  MessageSquareDiff,
  CheckCircle2,
  XCircle,
  Loader2,
  LogOut,
} from "lucide-react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

// ── Types ─────────────────────────────────────────────────────────────────
type IndexingStep = "upload" | "ocr" | "visual" | "index" | "completed" | "error";

interface JobStatus {
  status: IndexingStep;
  progress: number;
  message: string;
  filename?: string;
}

const STEP_ORDER: IndexingStep[] = ["upload", "ocr", "visual", "index", "completed"];

const STEP_LABELS: Record<string, string> = {
  upload: "📤 Uploading",
  ocr: "🔍 Extracting Text",
  visual: "🖼️ Analyzing Visuals",
  index: "⚡ Building Index",
  completed: "✅ Done",
};

// ── Indexing Overlay ───────────────────────────────────────────────────────
function IndexingOverlay({ job, onClose }: { job: JobStatus; onClose: () => void }) {
  const isDone = job.status === "completed";
  const isError = job.status === "error";
  const currentStepIdx = STEP_ORDER.indexOf(job.status);

  const content = (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 backdrop-blur-sm animate-in fade-in-0 duration-300">
      <div className="bg-card border border-border/60 rounded-2xl shadow-2xl p-8 w-[420px] max-w-[90vw] flex flex-col items-center gap-6">
        
        {/* Spinner / Done icon */}
        <div className="relative w-20 h-20 flex items-center justify-center">
          {isError ? (
            <XCircle className="w-14 h-14 text-destructive" />
          ) : isDone ? (
            <CheckCircle2 className="w-14 h-14 text-green-500 animate-in zoom-in-50 duration-300" />
          ) : (
            <>
              <div className="absolute inset-0 rounded-full border-4 border-primary/10" />
              <div className="absolute inset-0 rounded-full border-4 border-t-primary animate-spin" />
              <FileText className="w-8 h-8 text-primary/60" />
            </>
          )}
        </div>

        {/* Title + filename */}
        <div className="text-center">
          <h2 className="text-lg font-semibold tracking-tight">
            {isError ? "Processing Failed" : isDone ? "Ready!" : "Processing Document"}
          </h2>
          <p className="text-sm text-muted-foreground mt-1 truncate max-w-xs">
            {job.filename ?? "document.pdf"}
          </p>
        </div>

        {/* Step indicators */}
        {!isError && (
          <div className="flex items-center gap-1 w-full justify-center">
            {STEP_ORDER.filter((s) => s !== "completed").map((step, i) => {
              const isActive = currentStepIdx === i;
              const isDoneStep = currentStepIdx > i || isDone;
              return (
                <div key={step} className="flex items-center gap-1">
                  <div
                    className={`flex flex-col items-center gap-1 transition-all duration-300 ${
                      isActive ? "opacity-100 scale-105" : isDoneStep ? "opacity-100" : "opacity-40"
                    }`}
                  >
                    <div
                      className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold border-2 transition-colors duration-300 ${
                        isDoneStep
                          ? "bg-primary border-primary text-primary-foreground"
                          : isActive
                          ? "bg-primary/20 border-primary text-primary animate-pulse"
                          : "bg-muted border-border text-muted-foreground"
                      }`}
                    >
                      {isDoneStep ? "✓" : i + 1}
                    </div>
                    <span className="text-[10px] text-muted-foreground text-center leading-tight w-16">
                      {STEP_LABELS[step].split(" ").slice(1).join(" ")}
                    </span>
                  </div>
                  {i < 3 && (
                    <div
                      className={`w-6 h-px mb-4 transition-colors duration-300 ${
                        currentStepIdx > i || isDone ? "bg-primary" : "bg-border"
                      }`}
                    />
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Progress bar */}
        <div className="w-full bg-muted rounded-full h-2 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              isError ? "bg-destructive" : "bg-primary"
            }`}
            style={{ width: `${job.progress}%` }}
          />
        </div>

        {/* Status message */}
        <p className="text-sm text-muted-foreground text-center leading-snug px-2">
          {job.message}
        </p>

        {/* Close button — only shown when done or error */}
        {(isDone || isError) && (
          <Button onClick={onClose} className="w-full rounded-full" variant={isError ? "destructive" : "default"}>
            {isError ? "Dismiss" : "Start Chatting"}
          </Button>
        )}
      </div>
    </div>
  );

  if (typeof document === "undefined") return null;
  return createPortal(content, document.body);
}

// ── Main Sidebar ───────────────────────────────────────────────────────────
export default function Sidebar({
  currentPdf,
  onSelectPdf,
}: {
  currentPdf: string | null;
  onSelectPdf: (filename: string | null) => void;
}) {
  const router = useRouter();
  const supabase = createClient();
  const [documents, setDocuments] = useState<string[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [activeJob, setActiveJob] = useState<JobStatus | null>(null);
  const [clearingMemory, setClearingMemory] = useState(false);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchDocs = useCallback(async () => {
    try {
      const res = await fetch(`${API}/documents`);
      if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
      const data = await res.json();
      if (data && Array.isArray(data.documents)) {
        setDocuments(data.documents);
      }
    } catch (err) {
      console.error("Failed to fetch documents", err);
    }
  }, []);

  useEffect(() => {
    fetchDocs();
  }, [fetchDocs]);

  // Poll indexing job status
  const startPolling = (jobId: string, filename: string) => {
    if (pollingRef.current) clearInterval(pollingRef.current);
    pollingRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API}/indexing/status/${jobId}`);
        if (!res.ok) return;
        const data: JobStatus = await res.json();
        setActiveJob({ ...data, filename });

        if (data.status === "completed" || data.status === "error") {
          clearInterval(pollingRef.current!);
          pollingRef.current = null;
          if (data.status === "completed") {
            await fetchDocs();
          }
        }
      } catch {
        // ignore polling errors
      }
    }, 1500);
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    setActiveJob({
      status: "upload",
      progress: 5,
      message: "Uploading file to server...",
      filename: file.name,
    });

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(`${API}/upload`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`Upload failed: ${errText}`);
      }
      const data = await response.json();
      // Start polling for indexing progress
      startPolling(data.job_id, file.name);
    } catch (err) {
      console.error(err);
      setActiveJob({
        status: "error",
        progress: 0,
        message: `Upload failed: ${err instanceof Error ? err.message : "Unknown error"}`,
        filename: file.name,
      });
    } finally {
      setIsUploading(false);
      e.target.value = "";
    }
  };

  const handleDelete = async (filename: string) => {
    try {
      await fetch(`${API}/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
      if (currentPdf === filename) onSelectPdf(null);
      await fetchDocs();
    } catch (err) {
      console.error("Failed to delete document", err);
    }
  };

  const handleClearMemory = async () => {
    setClearingMemory(true);
    try {
      await fetch(`${API}/memory`, { method: "DELETE" });
    } catch (err) {
      console.error("Failed to clear memory", err);
    } finally {
      setClearingMemory(false);
    }
  };

  const handleOverlayClose = () => {
    if (activeJob?.status === "completed" && activeJob.filename) {
      onSelectPdf(activeJob.filename);
    }
    setActiveJob(null);
  };

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    router.push("/login");
    router.refresh();
  };

  return (
    <>
      {/* Indexing Overlay */}
      {activeJob && <IndexingOverlay job={activeJob} onClose={handleOverlayClose} />}

      <div className="w-64 border-r border-border bg-sidebar text-sidebar-foreground flex flex-col h-full shadow-sm relative z-20">
        {/* Header */}
        <div className="p-4 flex items-center justify-between border-b border-border/50">
          <h2 className="text-lg font-semibold tracking-tight text-primary flex items-center gap-2">
            <div className="w-4 h-4 rounded-sm bg-primary" />
            Vision RAG
          </h2>
        </div>

        <ScrollArea className="flex-1 px-3 py-4">
          <div className="space-y-4">
            {/* Documents Section */}
            <div>
              <div className="flex items-center justify-between px-2 mb-2">
                <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  Documents
                </h3>
                <label
                  className={`cursor-pointer text-muted-foreground hover:text-foreground transition-colors ${
                    isUploading ? "opacity-50 pointer-events-none" : ""
                  }`}
                  title="Upload Document"
                >
                  {isUploading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <UploadCloud className="h-4 w-4" />
                  )}
                  <input
                    type="file"
                    className="hidden"
                    accept=".pdf"
                    onChange={handleUpload}
                    disabled={isUploading}
                  />
                </label>
              </div>

              {/* Upload drop zone hint */}
              {documents.length === 0 && !isUploading && (
                <label className="mx-2 flex flex-col items-center gap-2 px-3 py-4 border border-dashed border-border rounded-lg cursor-pointer hover:border-primary/50 hover:bg-primary/5 transition-all duration-200 group">
                  <UploadCloud className="h-6 w-6 text-muted-foreground group-hover:text-primary transition-colors" />
                  <span className="text-xs text-muted-foreground text-center group-hover:text-foreground transition-colors">
                    Drop a PDF here or click to upload
                  </span>
                  <input
                    type="file"
                    className="hidden"
                    accept=".pdf"
                    onChange={handleUpload}
                    disabled={isUploading}
                  />
                </label>
              )}

              <div className="space-y-1 mt-1">
                {documents.map((doc) => (
                  <div key={doc} className="group flex items-center gap-1">
                    <Button
                      variant={currentPdf === doc ? "secondary" : "ghost"}
                      className={`flex-1 justify-start overflow-hidden text-sm truncate ${
                        currentPdf === doc
                          ? "font-medium bg-sidebar-accent"
                          : "font-normal text-muted-foreground"
                      }`}
                      onClick={() => onSelectPdf(doc)}
                    >
                      <FileText className="mr-2 h-4 w-4 shrink-0" />
                      <span className="truncate">{doc}</span>
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-8 w-8 opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
                      onClick={() => handleDelete(doc)}
                      title={`Delete ${doc}`}
                    >
                      <Trash2 className="h-4 w-4 text-destructive/70 hover:text-destructive" />
                    </Button>
                  </div>
                ))}
              </div>
            </div>

            <Separator className="bg-border/50" />

            {/* Clear Conversation */}
            <div className="px-1">
              <Button
                variant="ghost"
                className="w-full justify-start text-sm text-muted-foreground hover:text-foreground"
                onClick={handleClearMemory}
                disabled={clearingMemory}
              >
                {clearingMemory ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : (
                  <MessageSquareDiff className="mr-2 h-4 w-4" />
                )}
                Clear Conversation
              </Button>
            </div>

            <div className="px-1">
              <Button
                variant="ghost"
                className="w-full justify-start text-sm text-destructive hover:text-destructive hover:bg-destructive/10"
                onClick={handleSignOut}
              >
                <LogOut className="mr-2 h-4 w-4" />
                Sign Out
              </Button>
            </div>
          </div>
        </ScrollArea>

        {/* Footer status */}
        <div className="p-3 border-t border-border/50">
          <p className="text-[10px] text-muted-foreground text-center">
            {documents.length === 0
              ? "No documents loaded"
              : `${documents.length} document${documents.length > 1 ? "s" : ""} loaded`}
          </p>
        </div>
      </div>
    </>
  );
}
