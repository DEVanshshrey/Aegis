"use client";

import { useState, useEffect } from "react";
import {
  ChevronLeft,
  ChevronRight,
  ZoomIn,
  ZoomOut,
  RotateCw,
} from "lucide-react";
import { Button } from "./ui/button";

interface AnalysisResult {
  document: string;
  analysis: {
    clause: string;
    severity: string;
    risk_level: "low" | "medium" | "high";
    explanation?: string;
  }[];
}

interface PDFViewerProps {
  file: File;
  analysisResult: AnalysisResult;
}

export function PDFViewer({ file, analysisResult }: PDFViewerProps) {
  const [currentPage, setCurrentPage] = useState(1);
  const [numPages, setNumPages] = useState(1);
  const [scale, setScale] = useState(1.0);
  const [pdfUrl, setPdfUrl] = useState<string>("");

  useEffect(() => {
    if (file) {
      const url = URL.createObjectURL(file);
      setPdfUrl(url);

      return () => {
        URL.revokeObjectURL(url);
      };
    }
  }, [file]);

  const handleZoomIn = () => {
    setScale((prev) => Math.min(prev + 0.25, 3.0));
  };

  const handleZoomOut = () => {
    setScale((prev) => Math.max(prev - 0.25, 0.5));
  };

  const handlePrevPage = () => {
    setCurrentPage((prev) => Math.max(prev - 1, 1));
  };

  const handleNextPage = () => {
    setCurrentPage((prev) => Math.min(prev + 1, numPages));
  };

  return (
    <div className="flex h-full flex-col">
      {/* PDF Controls */}
      <div className="mb-4 flex items-center justify-between rounded-lg bg-gray-100 p-2">
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handlePrevPage}
            disabled={currentPage <= 1}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <span className="text-sm text-gray-600">
            Page {currentPage} of {numPages}
          </span>
          <Button
            variant="outline"
            size="sm"
            onClick={handleNextPage}
            disabled={currentPage >= numPages}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>

        <div className="flex items-center space-x-2">
          <Button variant="outline" size="sm" onClick={handleZoomOut}>
            <ZoomOut className="h-4 w-4" />
          </Button>
          <span className="min-w-[60px] text-center text-sm text-gray-600">
            {Math.round(scale * 100)}%
          </span>
          <Button variant="outline" size="sm" onClick={handleZoomIn}>
            <ZoomIn className="h-4 w-4" />
          </Button>
        </div>
      </div>

      {/* PDF Display */}
      <div className="flex-1 overflow-auto rounded-lg bg-gray-100 p-4">
        {pdfUrl ? (
          <div className="flex justify-center">
            <iframe
              src={`${pdfUrl}#page=${currentPage}&zoom=${scale * 100}`}
              className="h-full min-h-[600px] w-full border-0 bg-white shadow-lg"
              style={{
                transform: `scale(${scale})`,
                transformOrigin: "top center",
              }}
            />
          </div>
        ) : (
          <div className="flex h-full items-center justify-center">
            <div className="text-center text-gray-500">
              <div className="mx-auto mb-4 h-8 w-8 animate-spin rounded-full border-b-2 border-gray-900"></div>
              <p>Loading PDF...</p>
            </div>
          </div>
        )}
      </div>

      {/* Risk Indicators */}
      <div className="mt-4 rounded-lg border border-yellow-200 bg-yellow-50 p-3">
        <h4 className="mb-2 font-semibold text-yellow-800">
          Document Highlights
        </h4>
        <div className="flex flex-wrap gap-2">
          <div className="flex items-center space-x-1 text-xs">
            <div className="h-3 w-3 rounded-full bg-red-500"></div>
            <span className="text-red-700">High Risk Clauses</span>
          </div>
          <div className="flex items-center space-x-1 text-xs">
            <div className="h-3 w-3 rounded-full bg-yellow-500"></div>
            <span className="text-yellow-700">Medium Risk Clauses</span>
          </div>
          <div className="flex items-center space-x-1 text-xs">
            <div className="h-3 w-3 rounded-full bg-green-500"></div>
            <span className="text-green-700">Low Risk Clauses</span>
          </div>
        </div>
      </div>
    </div>
  );
}
