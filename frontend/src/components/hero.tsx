import {  Zap, Shield, CheckCircle } from 'lucide-react';

export function Hero() {
  return (
    <div className="container mx-auto px-4 py-16 text-center">
      <div className="max-w-4xl mx-auto">
        <h2 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">
          Transform Complex Legal Documents into
          <span className="text-blue-600"> Clear Insights</span>
        </h2>
        
        <p className="text-xl text-gray-600 mb-12 max-w-3xl mx-auto">
          Upload your rental agreement, contract, or legal document and get instant analysis 
          with risk assessments, plain-language explanations, and actionable recommendations.
        </p>

        <div className="grid md:grid-cols-3 gap-8 mb-12">
          <div className="flex flex-col items-center p-6 bg-white rounded-xl shadow-sm">
            <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
              <Zap className="w-6 h-6 text-blue-600" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">Instant Analysis</h3>
            <p className="text-gray-600 text-sm">
              Get comprehensive document analysis in seconds using advanced AI
            </p>
          </div>

          <div className="flex flex-col items-center p-6 bg-white rounded-xl shadow-sm">
            <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mb-4">
              <Shield className="w-6 h-6 text-green-600" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">Risk Detection</h3>
            <p className="text-gray-600 text-sm">
              Identify problematic clauses and potential legal risks automatically
            </p>
          </div>

          <div className="flex flex-col items-center p-6 bg-white rounded-xl shadow-sm">
            <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mb-4">
              <CheckCircle className="w-6 h-6 text-purple-600" />
            </div>
            <h3 className="font-semibold text-gray-900 mb-2">Plain Language</h3>
            <p className="text-gray-600 text-sm">
              Complex legal jargon translated into clear, understandable terms
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}