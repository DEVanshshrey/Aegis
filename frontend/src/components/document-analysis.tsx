'use client';

import { useState } from 'react';
import { 
  FileText, 
  AlertTriangle, 
  CheckCircle, 
  AlertCircle,
  ArrowLeft,
  Download,
  Eye,
  ChevronDown,
  ChevronRight
} from 'lucide-react';
import { Button } from './ui/button';
import { Card } from './ui/card';
import { Badge } from './ui/badge';
import { PDFViewer } from './pdf-viwer';

interface AnalysisResult {
  document: string;
  analysis: {
    clause: string;
    severity: string;
    risk_level: 'low' | 'medium' | 'high';
    explanation?: string;
  }[];
  summary?: {
    rent_amount?: string;
    security_deposit?: string;
    lease_term?: string;
    key_terms?: string[];
  };
}

interface DocumentAnalysisProps {
  analysisResult: AnalysisResult;
  uploadedFile: File | null;
  onReset: () => void;
}

export function DocumentAnalysis({ analysisResult, uploadedFile, onReset }: DocumentAnalysisProps) {
  const [expandedItems, setExpandedItems] = useState<Set<number>>(new Set());
  const [activeView, setActiveView] = useState<'analysis' | 'document'>('analysis');

  const toggleExpanded = (index: number) => {
    const newExpanded = new Set(expandedItems);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedItems(newExpanded);
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'high': return 'text-red-600 bg-red-50';
      case 'medium': return 'text-yellow-600 bg-yellow-50';
      case 'low': return 'text-green-600 bg-green-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const getRiskIcon = (riskLevel: string) => {
    switch (riskLevel) {
      case 'high': return <AlertTriangle className="w-4 h-4" />;
      case 'medium': return <AlertCircle className="w-4 h-4" />;
      case 'low': return <CheckCircle className="w-4 h-4" />;
      default: return <AlertCircle className="w-4 h-4" />;
    }
  };

  const highRiskItems = analysisResult.analysis.filter(item => item.risk_level === 'high');
  const mediumRiskItems = analysisResult.analysis.filter(item => item.risk_level === 'medium');
  const lowRiskItems = analysisResult.analysis.filter(item => item.risk_level === 'low');

  return (
    <div className="container mx-auto px-4 py-6 max-w-7xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-4">
          <Button variant="ghost" onClick={onReset} className="flex items-center space-x-2">
            <ArrowLeft className="w-4 h-4" />
            <span>New Analysis</span>
          </Button>
          <div>
            <h2 className="text-2xl font-bold text-gray-900">Document Analysis</h2>
            <p className="text-gray-600">{analysisResult.document}</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <Button variant="outline" size="sm">
            <Download className="w-4 h-4 mr-2" />
            Export Report
          </Button>
          <div className="flex bg-gray-100 rounded-lg p-1">
            <Button
              variant={activeView === 'analysis' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setActiveView('analysis')}
            >
              Analysis
            </Button>
            <Button
              variant={activeView === 'document' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setActiveView('document')}
            >
              <Eye className="w-4 h-4 mr-1" />
              Document
            </Button>
          </div>
        </div>
      </div>

      <div className="grid lg:grid-cols-12 gap-6">
        {/* Analysis Panel */}
        <div className={`lg:col-span-5 space-y-6 ${activeView === 'document' ? 'hidden lg:block' : ''}`}>
          {/* Summary Card */}
          {analysisResult.summary && (
            <Card className="p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                <FileText className="w-5 h-5 mr-2" />
                Key Details
              </h3>
              <div className="grid grid-cols-2 gap-4 mb-4">
                {analysisResult.summary.rent_amount && (
                  <div>
                    <p className="text-sm text-gray-600">Monthly Rent</p>
                    <p className="text-lg font-semibold text-gray-900">{analysisResult.summary.rent_amount}</p>
                  </div>
                )}
                {analysisResult.summary.security_deposit && (
                  <div>
                    <p className="text-sm text-gray-600">Security Deposit</p>
                    <p className="text-lg font-semibold text-gray-900">{analysisResult.summary.security_deposit}</p>
                  </div>
                )}
                {analysisResult.summary.lease_term && (
                  <div>
                    <p className="text-sm text-gray-600">Lease Term</p>
                    <p className="text-lg font-semibold text-gray-900">{analysisResult.summary.lease_term}</p>
                  </div>
                )}
              </div>
              {analysisResult.summary.key_terms && (
                <div>
                  <p className="text-sm text-gray-600 mb-2">Additional Terms</p>
                  <div className="flex flex-wrap gap-2">
                    {analysisResult.summary.key_terms.map((term, index) => (
                      <Badge key={index} variant="secondary" className="text-xs">
                        {term}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </Card>
          )}

          {/* Risk Overview */}
          <Card className="p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Risk Assessment</h3>
            <div className="grid grid-cols-3 gap-4">
              <div className="text-center p-3 bg-red-50 rounded-lg">
                <div className="text-2xl font-bold text-red-600">{highRiskItems.length}</div>
                <div className="text-sm text-red-700">High Risk</div>
              </div>
              <div className="text-center p-3 bg-yellow-50 rounded-lg">
                <div className="text-2xl font-bold text-yellow-600">{mediumRiskItems.length}</div>
                <div className="text-sm text-yellow-700">Medium Risk</div>
              </div>
              <div className="text-center p-3 bg-green-50 rounded-lg">
                <div className="text-2xl font-bold text-green-600">{lowRiskItems.length}</div>
                <div className="text-sm text-green-700">Low Risk</div>
              </div>
            </div>
          </Card>

          {/* Detailed Analysis */}
          <Card className="p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Detailed Analysis</h3>
            <div className="space-y-4">
              {analysisResult.analysis.map((item, index) => (
                <div key={index} className="border rounded-lg">
                  <div 
                    className="p-4 cursor-pointer hover:bg-gray-50 transition-colors"
                    onClick={() => toggleExpanded(index)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-3 mb-2">
                          <div className={`flex items-center space-x-1 px-2 py-1 rounded-full text-xs font-medium ${getRiskColor(item.risk_level)}`}>
                            {getRiskIcon(item.risk_level)}
                            <span className="capitalize">{item.risk_level} Risk</span>
                          </div>
                          <span className="text-lg">{item.severity}</span>
                        </div>
                        <p className="text-gray-900 font-medium text-sm leading-relaxed">
                          {item.clause}
                        </p>
                      </div>
                      <div className="flex-shrink-0 ml-4">
                        {expandedItems.has(index) ? (
                          <ChevronDown className="w-4 h-4 text-gray-400" />
                        ) : (
                          <ChevronRight className="w-4 h-4 text-gray-400" />
                        )}
                      </div>
                    </div>
                  </div>
                  
                  {expandedItems.has(index) && item.explanation && (
                    <div className="px-4 pb-4 border-t bg-gray-50">
                      <p className="text-gray-700 text-sm leading-relaxed pt-3">
                        {item.explanation}
                      </p>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </Card>
        </div>

        {/* Document Viewer */}
        <div className={`lg:col-span-7 ${activeView === 'analysis' ? 'hidden lg:block' : ''}`}>
          <Card className="h-[800px]">
            <div className="h-full flex flex-col">
              <div className="p-4 border-b bg-gray-50 rounded-t-lg">
                <h3 className="font-semibold text-gray-900">Document Preview</h3>
                <p className="text-sm text-gray-600">{analysisResult.document}</p>
              </div>
              <div className="flex-1 p-4">
                {uploadedFile ? (
                  <PDFViewer file={uploadedFile} analysisResult={analysisResult} />
                ) : (
                  <div className="h-full flex items-center justify-center text-gray-500">
                    <div className="text-center">
                      <FileText className="w-16 h-16 mx-auto mb-4 text-gray-300" />
                      <p>Document preview not available</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}