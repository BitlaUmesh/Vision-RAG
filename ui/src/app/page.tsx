"use client";

import { useState } from "react";
import Sidebar from "@/components/layout/sidebar";
import ChatInterface from "@/components/chat/chat-interface";
import dynamic from "next/dynamic";
import { Menu } from "lucide-react";
import { Button } from "@/components/ui/button";

const PdfViewer = dynamic(() => import("@/components/pdf/pdf-viewer"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center bg-muted/10 text-muted-foreground animate-pulse text-sm">
      Loading Document Environment...
    </div>
  ),
});

export default function Home() {
  const [currentPdf, setCurrentPdf] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="flex h-screen w-full bg-background overflow-hidden relative">
      {/* Mobile Sidebar Overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 md:hidden transition-opacity" 
          onClick={() => setSidebarOpen(false)} 
        />
      )}
      
      {/* Sidebar */}
      <div className={`
        fixed inset-y-0 left-0 z-50 transform transition-transform duration-300 ease-in-out 
        md:relative md:translate-x-0 h-full shadow-2xl md:shadow-none
        ${sidebarOpen ? "translate-x-0" : "-translate-x-full"}
      `}>
        <Sidebar
          currentPdf={currentPdf}
          onSelectPdf={(pdf) => {
            setCurrentPdf(pdf);
            setSidebarOpen(false);
          }}
        />
      </div>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col lg:flex-row min-w-0 h-full overflow-hidden relative">
        {/* Mobile Header */}
        <div className="md:hidden flex items-center justify-between p-3 border-b border-border/50 bg-card text-card-foreground shrink-0 z-10 shadow-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-sm bg-primary" />
            <span className="font-semibold text-primary tracking-tight">Vision RAG</span>
          </div>
          <Button variant="ghost" size="icon" onClick={() => setSidebarOpen(true)} className="h-8 w-8">
            <Menu className="w-5 h-5 text-foreground" />
          </Button>
        </div>

        {/* Middle Area: PDF Viewer */}
        <div className="flex-[1.2] lg:flex-1 relative border-b lg:border-b-0 lg:border-r border-border/50 bg-muted/10 overflow-hidden flex flex-col min-h-0">
          <PdfViewer currentPdf={currentPdf} />
        </div>

        {/* Right Area: Chat Interface */}
        <div className="flex-1 lg:flex-none w-full lg:w-[400px] xl:w-[450px] relative bg-background shadow-2xl z-20 flex flex-col min-h-0">
          <ChatInterface currentPdf={currentPdf} />
        </div>
      </main>
    </div>
  );
}
