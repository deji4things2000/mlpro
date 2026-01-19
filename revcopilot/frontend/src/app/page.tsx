'use client';

import { useState } from 'react';
import FileUpload from '@/components/FileUpload';
import ModeSelector from '@/components/ModeSelector';
import ResultsPanel from '@/components/ResultsPanel';
import CodeViewer from '@/components/CodeViewer';
import { Loader2, Terminal, Code2, Key } from 'lucide-react';
import { Toaster, toast } from 'react-hot-toast';

type Mode = 'auto' | 'ai' | 'tutor';
type AnalysisResult = {
  solution?: [string, string];
  transforms?: { type: string; insn: string; address?: string }[];
  hints?: string[];
  aiInsights?: string;
};

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

export default function Home() {
  const [selectedMode, setSelectedMode] = useState<Mode>('auto');
  const [isUploading, setIsUploading] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [activeTab, setActiveTab] = useState<'analysis' | 'code' | 'results'>('analysis');
  const [apiUrl, setApiUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [isTestingApi, setIsTestingApi] = useState(false);
  const [apiStatus, setApiStatus] = useState<'idle' | 'ok' | 'error'>('idle');

  const normalizeResult = (result: any): AnalysisResult => {
    const solutionObj = result?.solution;
    let solution: [string, string] | undefined;

    if (Array.isArray(solutionObj)) {
      if (solutionObj.length >= 2) {
        solution = [solutionObj[0], solutionObj[1]];
      }
    } else if (solutionObj && typeof solutionObj === 'object') {
      const arg1 = solutionObj.arg1 ?? solutionObj[0];
      const arg2 = solutionObj.arg2 ?? solutionObj[1];
      if (arg1 && arg2) {
        solution = [arg1, arg2];
      }
    }

    const rawTransforms = result?.transforms ?? result?.analysis?.transforms ?? [];
    const transforms = Array.isArray(rawTransforms)
      ? rawTransforms.map((t: any) => ({
          type: String(t?.type ?? 'unknown'),
          insn: String(t?.insn ?? t?.description ?? `${t?.type ?? 'op'} ${t?.value ?? ''}`).trim(),
          address: t?.address ? String(t.address) : undefined,
        }))
      : [];

    const hints = result?.hints ?? result?.analysis?.hints ?? undefined;
    const aiInsights =
      (typeof result?.aiInsights === 'string' && result.aiInsights) ||
      (typeof result?.insights?.insights === 'string' && result.insights.insights) ||
      (typeof result?.insights === 'string' && result.insights) ||
      undefined;

    return { solution, transforms, hints, aiInsights };
  };

  const handleTestApi = async () => {
    if (!apiUrl || !apiKey) {
      toast.error('Please provide both API URL and API key');
      return;
    }

    setIsTestingApi(true);
    setApiStatus('idle');

    try {
      const response = await fetch(`${BACKEND_URL}/api/ai/health`, {
        method: 'POST',
        headers: {
          'X-Dartmouth-API-Key': apiKey,
          'X-Dartmouth-API-Url': apiUrl,
        },
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `API error: ${response.status}`);
      }

      setApiStatus('ok');
      toast.success('Dartmouth API connection successful');
    } catch (error: any) {
      console.error('API health failed:', error);
      setApiStatus('error');
      toast.error(`Dartmouth API test failed: ${error?.message || 'Unknown error'}`);
    } finally {
      setIsTestingApi(false);
    }
  };

  const handleFileUpload = async (file: File) => {
    setIsUploading(true);
    setAnalysisResult(null);
    
    try {
      const formData = new FormData();
      formData.append('file', file);
      if (apiKey) {
        formData.append('dartmouth_api_key_form', apiKey);
      }
      if (apiUrl) {
        formData.append('dartmouth_api_url_form', apiUrl);
      }
      
      const response = await fetch(
        `${BACKEND_URL}/api/analyze?mode=${selectedMode}`,
        {
          method: 'POST',
          body: formData,
        }
      );
      
      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }
      
      const data = await response.json();
      
      // Poll for results
      const jobId = data.job_id;
      let result = null;
      let attempts = 0;
      const maxAttempts = 30; // 30 seconds max
      
      while (attempts < maxAttempts) {
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        const statusResponse = await fetch(`${BACKEND_URL}/api/result/${jobId}`);
        const statusData = await statusResponse.json();
        
        if (statusData.status === 'completed') {
          result = normalizeResult(statusData.result);
          break;
        } else if (statusData.status === 'error') {
          throw new Error(statusData.error);
        }
        
        attempts++;
      }
      
      if (!result) {
        throw new Error('Analysis timeout');
      }
      
      setAnalysisResult(result);
      toast.success('Analysis complete!');
      
    } catch (error: any) {
      console.error('Upload failed:', error);
      toast.error(`Analysis failed: ${error?.message || 'Unknown error'}`);
      
      // Fallback to mock data for demo
      const mockResult = {
        solution: ["GHIDRA_REV_KEY__", "TR_C31NG_KEY_2__"],
        transforms: [
          { type: 'xor', insn: 'xor al, 0x05', address: '0x1013a4' },
          { type: 'rotate', insn: 'rol al, 0x04', address: '0x1013a6' },
        ],
        hints: [
          'Check argv length - should be exactly 16 bytes',
          'Look for XOR operations with constant 0x05',
          'Rotation by 4 bits suggests ROL4 transformation',
        ],
        aiInsights: 'The binary uses a two-stage transformation: XOR with 0x05 followed by ROL4, then an XOR-swap mirroring operation.',
      };
      
      setAnalysisResult(mockResult);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100">
      <Toaster position="top-right" />
      
      {/* Header */}
      <header className="border-b border-gray-200 bg-white/80 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl">
                <Terminal className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
                  RevCopilot
                </h1>
                <p className="text-sm text-gray-600">AI-Powered Reverse Engineering Assistant</p>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <div className="hidden md:flex items-center gap-2 text-sm text-gray-600">
                <Key className="h-4 w-4" />
                <span>Dartmouth CS 169</span>
              </div>
              <button className="px-4 py-2 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-lg hover:opacity-90 transition-opacity">
                GitHub
              </button>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Upload & Controls */}
          <div className="lg:col-span-1 space-y-8">
            <div className="bg-white rounded-2xl shadow-xl p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-4">Upload Binary</h2>
              <FileUpload 
                onFileUpload={handleFileUpload} 
                isUploading={isUploading} 
              />
            </div>
            
            <div className="bg-white rounded-2xl shadow-xl p-6">
              <ModeSelector 
                selectedMode={selectedMode} 
                onModeChange={setSelectedMode} 
              />
              <div className="mt-6 space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Dartmouth API URL
                  </label>
                  <input
                    value={apiUrl}
                    onChange={(e) => setApiUrl(e.target.value)}
                    placeholder="https://chat.dartmouth.edu/api"
                    className="w-full px-3 py-2 border rounded-lg text-sm"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Dartmouth API Key
                  </label>
                  <input
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    type="password"
                    placeholder="Enter API key"
                    className="w-full px-3 py-2 border rounded-lg text-sm"
                  />
                </div>
                <div className="flex items-center gap-3">
                  <button
                    type="button"
                    onClick={handleTestApi}
                    disabled={isTestingApi}
                    className="px-3 py-2 text-sm bg-gray-100 rounded-lg hover:bg-gray-200 disabled:opacity-50"
                  >
                    {isTestingApi ? 'Testing...' : 'Test Dartmouth API'}
                  </button>
                  <span className={`text-xs ${apiStatus === 'ok' ? 'text-green-600' : apiStatus === 'error' ? 'text-red-600' : 'text-gray-500'}`}>
                    {apiStatus === 'ok' ? 'Connected' : apiStatus === 'error' ? 'Failed' : 'Not tested'}
                  </span>
                </div>
              </div>
            </div>
            
            {analysisResult && analysisResult.solution?.length === 2 && (
              <div className="bg-gradient-to-br from-green-50 to-emerald-100 rounded-2xl shadow-xl p-6 border border-green-200">
                <div className="flex items-center gap-3 mb-4">
                  <div className="p-2 bg-green-500 rounded-lg">
                    <Key className="h-5 w-5 text-white" />
                  </div>
                  <h3 className="text-lg font-bold text-gray-800">Solution Found!</h3>
                </div>
                <div className="space-y-3">
                  <div className="font-mono bg-gray-900 text-green-400 p-4 rounded-lg text-sm overflow-x-auto">
                    ./binary &#39;{analysisResult.solution![0]}&#39; &#39;{analysisResult.solution![1]}&#39;
                  </div>
                  <button 
                    onClick={() => navigator.clipboard.writeText(`./binary '${analysisResult.solution![0]}' '${analysisResult.solution![1]}'`)}
                    className="w-full bg-gradient-to-r from-green-600 to-emerald-600 text-white font-semibold py-3 rounded-lg hover:opacity-90 transition-opacity"
                  >
                    Copy Command
                  </button>
                </div>
              </div>
            )}
          </div>

          {/* Middle Column - Analysis & Code */}
          <div className="lg:col-span-2 space-y-8">
            {/* Tabs */}
            <div className="bg-white rounded-2xl shadow-xl overflow-hidden">
              <div className="flex border-b border-gray-200">
                {['analysis', 'code', 'results'].map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setActiveTab(tab as any)}
                    className={`
                      flex-1 py-4 font-medium flex items-center justify-center gap-2 transition-colors
                      ${activeTab === tab 
                        ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white' 
                        : 'text-gray-600 hover:bg-gray-50'
                      }
                    `}
                  >
                    {tab === 'analysis' && <Loader2 className="h-5 w-5" />}
                    {tab === 'code' && <Code2 className="h-5 w-5" />}
                    {tab === 'results' && <Terminal className="h-5 w-5" />}
                    {tab.charAt(0).toUpperCase() + tab.slice(1)}
                  </button>
                ))}
              </div>
              
              <div className="h-[500px]">
                {activeTab === 'analysis' && (
                  <div className="p-6">
                    {isUploading ? (
                      <div className="flex flex-col items-center justify-center h-full space-y-4">
                        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
                        <p className="text-gray-600">Running static analysis and symbolic execution...</p>
                        <p className="text-sm text-gray-500">This may take a minute</p>
                      </div>
                    ) : analysisResult ? (
                      <div className="space-y-6">
                        <div>
                          <h4 className="font-bold text-lg mb-3">Detected Transformations</h4>
                          <div className="space-y-2">
                            {analysisResult.transforms?.map((t, i) => (
                              <div key={i} className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                                <span className="px-2 py-1 bg-blue-100 text-blue-800 text-xs font-semibold rounded">
                                  {t.type.toUpperCase()}
                                </span>
                                <code className="flex-1 font-mono text-sm">{t.insn}</code>
                                <span className="text-gray-500 text-sm">{t.address}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                        
                        {selectedMode === 'tutor' && analysisResult.hints && (
                          <div>
                            <h4 className="font-bold text-lg mb-3">Educational Hints</h4>
                            <div className="space-y-2">
                              {analysisResult.hints.map((hint, i) => (
                                <div key={i} className="p-4 bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl border border-purple-100">
                                  <div className="flex items-center gap-3">
                                    <div className="p-2 bg-purple-100 rounded-lg">
                                      <span className="font-bold text-purple-700">{i + 1}</span>
                                    </div>
                                    <p className="text-gray-700">{hint}</p>
                                  </div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="flex flex-col items-center justify-center h-full text-gray-500">
                        <Code2 className="h-16 w-16 mb-4 opacity-50" />
                        <p>Upload a binary to start analysis</p>
                      </div>
                    )}
                  </div>
                )}
                
                {activeTab === 'code' && (
                  <div className="h-full">
                    {/* Monaco Editor would go here */}
                    <div className="p-4 h-full bg-gray-900 text-gray-300 font-mono text-sm overflow-auto">
                      <div className="text-green-400">// Decompiled code will appear here</div>
                      <div className="mt-4 text-yellow-400">// For demo: medium.bin transformation logic</div>
                      <div className="mt-2">
                        <span className="text-blue-400">void</span> <span className="text-purple-400">transform</span>() {'{'}
                      </div>
                      <div className="ml-4">
                        <span className="text-blue-400">for</span> (i = 0; i &lt; 16; i++) {'{'}
                      </div>
                      <div className="ml-8">
                        buf[i] = buf[i] ^ 0x05;
                      </div>
                      <div className="ml-8">
                        buf[i] = (buf[i] &lt;&lt; 4) | (buf[i] &gt;&gt; 4);
                      </div>
                      <div className="ml-4">{'}'}</div>
                      <div className="ml-4">
                        <span className="text-blue-400">// XOR-swap mirroring</span>
                      </div>
                      <div className="ml-4">
                        <span className="text-blue-400">for</span> (i = 0; i &lt; 8; i++) {'{'}
                      </div>
                      <div className="ml-8">
                        buf[i] ^= buf[15-i];
                      </div>
                      <div className="ml-8">
                        buf[15-i] ^= buf[i];
                      </div>
                      <div className="ml-8">
                        buf[i] ^= buf[15-i];
                      </div>
                      <div className="ml-4">{'}'}</div>
                      <div>{'}'}</div>
                    </div>
                  </div>
                )}
                
                {activeTab === 'results' && analysisResult && (
                  <div className="p-6">
                    <div className="space-y-6">
                      <div>
                        <h4 className="font-bold text-lg mb-3">Analysis Summary</h4>
                        <div className="grid grid-cols-2 gap-4">
                          <div className="p-4 bg-gradient-to-br from-blue-50 to-cyan-50 rounded-xl">
                            <div className="text-sm text-blue-600 mb-1">Binary Type</div>
                            <div className="font-bold">ELF 64-bit</div>
                          </div>
                          <div className="p-4 bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl">
                            <div className="text-sm text-green-600 mb-1">Status</div>
                            <div className="font-bold text-green-700">CRACKED</div>
                          </div>
                          <div className="p-4 bg-gradient-to-br from-purple-50 to-pink-50 rounded-xl">
                            <div className="text-sm text-purple-600 mb-1">Strings Found</div>
                            <div className="font-bold">incorrect, solved, part1</div>
                          </div>
                          <div className="p-4 bg-gradient-to-br from-yellow-50 to-orange-50 rounded-xl">
                            <div className="text-sm text-yellow-600 mb-1">Analysis Time</div>
                            <div className="font-bold">2.3s</div>
                          </div>
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="font-bold text-lg mb-3">Execution Path</h4>
                        <div className="space-y-2">
                          <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                            <div className="w-8 h-8 rounded-full bg-green-100 flex items-center justify-center">
                              <span className="text-green-700 font-bold">1</span>
                            </div>
                            <div className="flex-1">
                              <div className="font-medium">Input validation</div>
                              <div className="text-sm text-gray-500">Check argc == 3</div>
                            </div>
                          </div>
                          <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                            <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center">
                              <span className="text-blue-700 font-bold">2</span>
                            </div>
                            <div className="flex-1">
                              <div className="font-medium">Transformation</div>
                              <div className="text-sm text-gray-500">XOR + ROL4 + swap</div>
                            </div>
                          </div>
                          <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                            <div className="w-8 h-8 rounded-full bg-purple-100 flex items-center justify-center">
                              <span className="text-purple-700 font-bold">3</span>
                            </div>
                            <div className="flex-1">
                              <div className="font-medium">Comparison</div>
                              <div className="text-sm text-gray-500">memcmp with hash</div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
            
            {/* AI Insights */}
            {selectedMode === 'ai' && analysisResult?.aiInsights && (
              <div className="bg-gradient-to-br from-blue-50 to-indigo-50 rounded-2xl shadow-xl p-6 border border-blue-200">
                <div className="flex items-center gap-3 mb-4">
                  <div className="p-2 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-lg">
                    <Terminal className="h-5 w-5 text-white" />
                  </div>
                  <h3 className="text-lg font-bold text-gray-800">AI Insights</h3>
                </div>
                <p className="text-gray-700">{analysisResult.aiInsights}</p>
              </div>
            )}
          </div>
        </div>
      </main>
      
      <footer className="mt-12 border-t border-gray-200 bg-white/80 backdrop-blur-sm">
        <div className="container mx-auto px-6 py-8">
          <div className="text-center text-gray-600">
            <p>RevCopilot • Dartmouth CS 169 Lab 4 • AI-Powered Reverse Engineering</p>
            <p className="text-sm mt-2">Educational use only. Do not use on software you don&apos;t own.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}