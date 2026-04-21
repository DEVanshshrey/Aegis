'use client';

import { useCallback, useState } from 'react';
import { Upload, FileText, Loader2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';

interface FileUploadProps {
  onFileUpload: (file: File) => void;
  isLoading: boolean;
}

export function FileUpload({ onFileUpload, isLoading }: FileUploadProps) {
  const [isDragActive, setIsDragActive] = useState(false);

  const onDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
  }, []);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);

    const files = Array.from(e.dataTransfer.files);
    const pdfFile = files.find(file => file.type === 'application/pdf');
    
    if (pdfFile) {
      onFileUpload(pdfFile);
    }
  }, [onFileUpload]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type === 'application/pdf') {
      onFileUpload(file);
    }
  };

  if (isLoading) {
    return (
      <Card className="max-w-2xl mx-auto p-12 text-center">
        <div className="flex flex-col items-center space-y-4">
          <Loader2 className="w-12 h-12 text-blue-600 animate-spin" />
          <h3 className="text-xl font-semibold text-gray-900">Analyzing Document</h3>
          <p className="text-gray-600">
            Our AI is carefully reviewing your document for risks and key terms...
          </p>
        </div>
      </Card>
    );
  }

  return (
    <Card 
      className={`max-w-2xl mx-auto p-12 border-2 border-dashed transition-all duration-200 cursor-pointer hover:border-blue-400 ${
        isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
      }`}
      onDragEnter={onDragEnter}
      onDragLeave={onDragLeave}
      onDragOver={onDragOver}
      onDrop={onDrop}
    >
      <div className="text-center space-y-6">
        <div className="flex justify-center">
          <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center">
            <Upload className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        
        <div>
          <h3 className="text-xl font-semibold text-gray-900 mb-2">
            Upload Your Legal Document
          </h3>
          <p className="text-gray-600 mb-6">
            Drag and drop your PDF rental agreement, contract, or legal document here
          </p>
        </div>

        <div className="space-y-4">
          <input
            type="file"
            accept=".pdf"
            onChange={handleFileSelect}
            className="hidden"
            id="file-upload"
          />
          <label htmlFor="file-upload">
            <Button asChild className="w-full max-w-xs mx-auto">
              <span className="cursor-pointer">
                <FileText className="w-4 h-4 mr-2" />
                Select PDF File
              </span>
            </Button>
          </label>
          
          <p className="text-sm text-gray-500">
            Supports PDF files up to 10MB
          </p>
        </div>
      </div>
    </Card>
  );
}