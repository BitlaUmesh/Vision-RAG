"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Send, Bot, Loader2, FileSearch } from "lucide-react";

type Message = {
  role: "user" | "assistant";
  content: string;
};

// Transform [[Page X: "quote"]] or [[Page Filename_X: "quote"]] into [Page X](#citation:X:quote)
function processCitations(text: string) {
  return text.replace(/\[\[Page\s+([^:]+):\s*"([^"]+)"\]\]/gi, (match, pageRef, quote) => {
    const pageMatch = pageRef.match(/(\d+)$/);
    const pageNum = pageMatch ? pageMatch[1] : "1";
    // URL encode the quote to make it safe for the href
    return `[Pg ${pageNum}](#citation:${pageNum}:${encodeURIComponent(quote)})`;
  });
}

export default function ChatInterface({ currentPdf }: { currentPdf: string | null }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !currentPdf) return;

    const userMessage = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setIsLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/chat/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage }),
      });

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let assistantMsg = "";
      
      setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n");
        
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const dataStr = line.slice(6);
            if (!dataStr) continue;
            try {
              const data = JSON.parse(dataStr);
              if (data.type === "token") {
                assistantMsg += data.content;
                setMessages((prev) => {
                  const newMsgs = [...prev];
                  newMsgs[newMsgs.length - 1].content = assistantMsg;
                  return newMsgs;
                });
              }
            } catch {
              // ignore parse error on incomplete chunks
            }
          }
        }
      }
    } catch (err) {
      console.error(err);
      setMessages((prev) => [...prev, { role: "assistant", content: "**Error connecting to server.**" }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full bg-background relative border-l border-border/50">
      <div className="p-4 border-b border-border/50 flex items-center justify-between bg-card text-card-foreground">
        <h3 className="font-semibold tracking-tight">Copilot</h3>
      </div>
      
      <div className="flex-1 p-4 overflow-y-auto">
        <div className="space-y-6 pb-4">
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-muted-foreground mt-20 space-y-4">
              <Bot className="w-12 h-12 opacity-20" />
              <p className="text-sm">Ask questions about your documents</p>
            </div>
          )}
          
          {messages.map((msg, idx) => (
            <div 
              key={idx} 
              className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'} animate-in fade-in slide-in-from-bottom-2 duration-300`}
            >
              {msg.role === 'assistant' && (
                <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center shrink-0 mt-1 border border-primary/20">
                  <Bot className="w-4 h-4 text-primary" />
                </div>
              )}
              <div 
                className={`max-w-[85%] text-sm px-4 py-3 rounded-2xl ${
                  msg.role === 'user' 
                    ? 'bg-primary text-primary-foreground rounded-tr-sm' 
                    : 'bg-muted/50 text-foreground rounded-tl-sm border border-border/50 shadow-sm'
                }`}
              >
                {msg.role === 'user' ? (
                  msg.content
                ) : (
                  <div className="prose prose-sm dark:prose-invert max-w-none text-[15px] leading-relaxed break-words">
                    <ReactMarkdown 
                      remarkPlugins={[remarkGfm]}
                      components={{
                        a: ({ node, href, children, ...props }) => {
                          if (href?.startsWith("#citation:")) {
                            const [, page, quoteEncoded] = href.split(":");
                            const quote = decodeURIComponent(quoteEncoded || "");
                            return (
                              <button
                                onClick={() => {
                                  window.dispatchEvent(
                                    new CustomEvent("highlight-pdf", {
                                      detail: { page: parseInt(page, 10), quote },
                                    })
                                  );
                                }}
                                className="inline-flex items-center gap-1 mx-1 px-2 py-0.5 rounded-full text-xs font-semibold bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 border border-blue-200 dark:border-blue-800/50 hover:bg-blue-200 dark:hover:bg-blue-800/60 transition-colors shadow-sm cursor-pointer whitespace-nowrap active:scale-95"
                                title={`Quote: "${quote}"`}
                                type="button"
                              >
                                <FileSearch className="w-3 h-3" />
                                {children}
                              </button>
                            );
                          }
                          return <a href={href} className="text-primary hover:underline font-medium" target="_blank" rel="noreferrer" {...props}>{children}</a>;
                        }
                      }}
                    >
                      {processCitations(msg.content)}
                    </ReactMarkdown>
                  </div>
                )}
              </div>
            </div>
          ))}
          <div ref={scrollRef} />
        </div>
      </div>

      <div className="p-4 bg-background border-t border-border/50">
        <form onSubmit={handleSubmit} className="flex gap-2">
          <Input 
            value={input} 
            onChange={(e) => setInput(e.target.value)}
            placeholder={currentPdf ? `Ask about ${currentPdf}...` : "Select a document first..."}
            disabled={!currentPdf || isLoading}
            className="rounded-full bg-muted/50 border-border/50 focus-visible:ring-primary shadow-inner"
          />
          <Button 
            type="submit" 
            disabled={!currentPdf || isLoading || !input.trim()} 
            size="icon"
            className="rounded-full shadow-md shrink-0 transition-transform active:scale-95"
          >
            {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          </Button>
        </form>
      </div>
    </div>
  );
}
