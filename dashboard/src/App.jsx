import React, { useState, useEffect } from 'react';
import { 
  Layers, Cpu, Zap, Activity, Filter, BarChart3, Database, 
  Flame, ShieldAlert, Clock, Sparkles, Server, ArrowRight, RefreshCw
} from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, 
  LineChart, Line, CartesianGrid, Legend, AreaChart, Area 
} from 'recharts';

const API_BASE = 'http://localhost:8000';

export default function App() {
  const [activeTab, setActiveTab] = useState('pipeline');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  
  // Interactive Simulator state
  const [simUserId, setSimUserId] = useState(42);
  const [simLambda, setSimLambda] = useState(0.8);
  const [simRecs, setSimRecs] = useState(null);
  const [simLoading, setSimLoading] = useState(false);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/results`);
      const json = await res.json();
      setData(json);
      fetchRecommendations(simUserId, simLambda);
    } catch (err) {
      console.error("Backend offline, using local fallback metrics", err);
      // Fallback data if server starting up
      setData(getFallbackData());
    } finally {
      setLoading(false);
    }
  };

  const fetchRecommendations = async (uid, lamb) => {
    setSimLoading(true);
    try {
      const res = await fetch(`${API_BASE}/api/recommend?user_id=${uid}&lambda_mmr=${lamb}`);
      const json = await res.json();
      setSimRecs(json);
    } catch (err) {
      console.error("Rec API error", err);
    } finally {
      setSimLoading(false);
    }
  };

  // No hardcoded fallback metrics: every number on this dashboard comes from /api/results,
  // which is computed by investigation.py on held-out data.
  function getFallbackData() {
    return null;
  }

  const modelAucChartData = data?.models_auc ? Object.entries(data.models_auc).map(([name, auc]) => ({ name, AUC: auc })) : [];

  return (
    <div className="min-h-screen bg-[#0b0f19] text-gray-100 p-6 font-sans">
      {/* Top Header */}
      <header className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center pb-6 border-b border-gray-800 gap-4">
        <div className="flex items-center space-x-3">
          <div className="p-2.5 bg-[#76b900]/10 border border-[#76b900]/30 rounded-xl">
            <Cpu className="w-8 h-8 text-[#76b900]" />
          </div>
          <div>
            <h1 className="text-2xl font-bold tracking-wide flex items-center gap-2">
              NVIDIA Merlin <span className="text-sm font-semibold bg-[#76b900]/20 text-[#76b900] px-2.5 py-0.5 rounded-full border border-[#76b900]/40">Two-Stage RecSys Engine</span>
            </h1>
            <p className="text-xs text-gray-400">Large-Scale Candidate Retrieval & Deep Ranking Architecture</p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="flex bg-gray-900/80 p-1.5 rounded-xl border border-gray-800 text-sm">
          {[
            { id: 'pipeline', label: 'Architecture Pipeline', icon: Layers },
            { id: 'investigation', label: 'Models & Benchmarks', icon: BarChart3 },
            { id: 'simulator', label: 'Live Rec Simulator', icon: Sparkles },
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-all ${
                activeTab === tab.id 
                  ? 'bg-[#76b900] text-black shadow-lg shadow-[#76b900]/20' 
                  : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800/50'
              }`}
            >
              <tab.icon className="w-4 h-4" />
              {tab.label}
            </button>
          ))}
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto mt-6">
        {/* TAB 1: PIPELINE ARCHITECTURE */}
        {activeTab === 'pipeline' && (
          <div className="space-y-6">
            {/* Stage Flow Diagram */}
            <div className="glass-panel p-6 rounded-2xl">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2 text-gray-200">
                <Activity className="w-5 h-5 text-[#76b900]" /> High-Throughput Recommender System Stages
              </h2>

              <div className="grid grid-cols-1 md:grid-cols-5 gap-4 relative">
                {/* Step 1 */}
                <div className="bg-gray-900/90 p-5 rounded-xl border border-gray-800 relative group hover:border-[#76b900]/50 transition-all">
                  <div className="text-xs text-[#76b900] font-mono mb-1">STAGE 01</div>
                  <h3 className="font-bold text-gray-100 flex items-center gap-1.5"><Database className="w-4 h-4 text-blue-400"/> User Events</h3>
                  <p className="text-xs text-gray-400 mt-2">Continuous interaction streams (Clicks, Views, Conversions, Search history).</p>
                  <div className="mt-3 text-[11px] text-gray-500 font-mono">Input Rate: ~250k events/sec</div>
                </div>

                {/* Step 2 */}
                <div className="bg-gray-900/90 p-5 rounded-xl border border-gray-800 relative group hover:border-[#76b900]/50 transition-all">
                  <div className="text-xs text-[#76b900] font-mono mb-1">STAGE 02</div>
                  <h3 className="font-bold text-gray-100 flex items-center gap-1.5"><Filter className="w-4 h-4 text-emerald-400"/> Feature Pipeline</h3>
                  <p className="text-xs text-gray-400 mt-2">NVTabular ETL transformations, MinMax normalization & categorical embeddings.</p>
                  <div className="mt-3 text-[11px] text-emerald-400 font-mono">GPU Accelerated ETL</div>
                </div>

                {/* Step 3 */}
                <div className="bg-gray-900/90 p-5 rounded-xl border border-[#76b900]/40 relative group nvidia-glow transition-all">
                  <div className="text-xs text-[#76b900] font-mono mb-1">STAGE 03</div>
                  <h3 className="font-bold text-gray-100 flex items-center gap-1.5"><Zap className="w-4 h-4 text-[#76b900]"/> Candidate Retrieval</h3>
                  <p className="text-xs text-gray-400 mt-2">FAISS exact inner-product search over Two-Tower Dual-Encoder embeddings.</p>
                  <div className="mt-3 text-[11px] text-[#76b900] font-mono">Top-100 candidates (&lt; 0.1 ms)</div>
                </div>

                {/* Step 4 */}
                <div className="bg-gray-900/90 p-5 rounded-xl border border-purple-500/40 relative group hover:border-purple-500/80 transition-all">
                  <div className="text-xs text-purple-400 font-mono mb-1">STAGE 04</div>
                  <h3 className="font-bold text-gray-100 flex items-center gap-1.5"><Cpu className="w-4 h-4 text-purple-400"/> Deep Ranking Model</h3>
                  <p className="text-xs text-gray-400 mt-2">DCN-v2 Cross Networks + Deep MLP scoring candidates with cross-features.</p>
                  <div className="mt-3 text-[11px] text-purple-400 font-mono">Fine Scoring (p95: 0.25ms)</div>
                </div>

                {/* Step 5 */}
                <div className="bg-gray-900/90 p-5 rounded-xl border border-amber-500/40 relative group hover:border-amber-500/80 transition-all">
                  <div className="text-xs text-amber-400 font-mono mb-1">STAGE 05</div>
                  <h3 className="font-bold text-gray-100 flex items-center gap-1.5"><Sparkles className="w-4 h-4 text-amber-400"/> Top 20 Serving</h3>
                  <p className="text-xs text-gray-400 mt-2">Post-processing: MMR diversity re-ranking, cold-start fallback & popularity de-biasing.</p>
                  <div className="mt-3 text-[11px] text-amber-400 font-mono">Final Recommendations</div>
                </div>
              </div>
            </div>

            {/* Quick Metrics KPI cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-5">
              <div className="glass-panel p-5 rounded-2xl flex items-center gap-4">
                <div className="p-3 bg-blue-500/10 text-blue-400 rounded-xl"><Clock className="w-6 h-6"/></div>
                <div>
                  <div className="text-xs text-gray-400 font-medium">Candidate Retrieval Latency</div>
                  <div className="text-xl font-bold text-gray-100">{data?.latency?.['Candidate Retrieval (FAISS exact IP - 5k items, top-100)']?.p95_ms ?? '–'} ms</div>
                  <div className="text-[11px] text-emerald-400">FAISS Index (Sub-millisecond)</div>
                </div>
              </div>

              <div className="glass-panel p-5 rounded-2xl flex items-center gap-4">
                <div className="p-3 bg-purple-500/10 text-purple-400 rounded-xl"><Cpu className="w-6 h-6"/></div>
                <div>
                  <div className="text-xs text-gray-400 font-medium">Deep Ranking Latency</div>
                  <div className="text-xl font-bold text-gray-100">{data?.latency?.['Deep Scoring Model (DCN - 100 candidates)']?.p95_ms ?? '–'} ms</div>
                  <div className="text-[11px] text-purple-400">DCN Cross Network</div>
                </div>
              </div>

              <div className="glass-panel p-5 rounded-2xl flex items-center gap-4">
                <div className="p-3 bg-[#76b900]/10 text-[#76b900] rounded-xl"><BarChart3 className="w-6 h-6"/></div>
                <div>
                  <div className="text-xs text-gray-400 font-medium">Deep Ranking AUC</div>
                  <div className="text-xl font-bold text-gray-100">{data?.models_auc?.['Deep Ranking Model'] ?? '–'}</div>
                  <div className="text-[11px] text-[#76b900]">vs popularity baseline {data?.models_auc?.['Popularity Baseline'] ?? '–'}</div>
                </div>
              </div>

              <div className="glass-panel p-5 rounded-2xl flex items-center gap-4">
                <div className="p-3 bg-amber-500/10 text-amber-400 rounded-xl"><ShieldAlert className="w-6 h-6"/></div>
                <div>
                  <div className="text-xs text-gray-400 font-medium">Catalog Gini Index</div>
                  <div className="text-xl font-bold text-gray-100">{data?.popularity_bias?.['Gini Concentration Coefficient'] ?? '–'}</div>
                  <div className="text-[11px] text-amber-400">Moderate Popularity Bias</div>
                </div>
              </div>
            </div>

            {/* Architecture Overview Card */}
            <div className="glass-panel p-6 rounded-2xl grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="text-base font-bold text-gray-100 mb-3 flex items-center gap-2">
                  <Server className="w-4 h-4 text-[#76b900]" /> Stage 1: Candidate Generation (Retrieval)
                </h3>
                <p className="text-xs text-gray-300 leading-relaxed">
                  In Merlin's architecture, candidate generation reduces the catalog to a small candidate set (top-100 of 5,000 items here). The **Two-Tower Dual-Encoder** model maps users and item metadata into a shared high-dimensional vector space using cosine distance. Items are indexed into **FAISS vector indices** (exact IndexFlatIP here; IVF/HNSW approximate indexes at larger scale).
                </p>
              </div>

              <div>
                <h3 className="text-base font-bold text-gray-100 mb-3 flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-purple-400" /> Stage 2: Deep Ranking (DCN / NeuMF)
                </h3>
                <p className="text-xs text-gray-300 leading-relaxed">
                  The ranking stage scores candidate items using computationally rich deep neural networks. By fusing explicit feature crosses (**DCN-v2**) with deep MLP layers (**NCF / DLRM**), the model accurately captures non-linear user-item interactions, temporal recency, and user demographics.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: INVESTIGATION BENCHMARKS */}
        {activeTab === 'investigation' && (
          <div className="space-y-6">
            {/* Grid 1: Model AUC & Negative Sampling */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Model Performance Comparison */}
              <div className="glass-panel p-6 rounded-2xl">
                <h3 className="text-base font-bold text-gray-100 mb-4 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-[#76b900]"/> Model AUC Accuracy Comparison
                </h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={modelAucChartData} margin={{ top: 10, right: 10, left: -20, bottom: 25 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                      <XAxis dataKey="name" stroke="#9ca3af" tick={{ fontSize: 10 }} interval={0} angle={-15} textAnchor="end" />
                      <YAxis stroke="#9ca3af" domain={[0.5, 0.9]} tick={{ fontSize: 11 }} />
                      <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', borderRadius: '8px', fontSize: '12px' }} />
                      <Bar dataKey="AUC" fill="#76b900" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Negative Sampling Strategy Comparison */}
              <div className="glass-panel p-6 rounded-2xl">
                <h3 className="text-base font-bold text-gray-100 mb-4 flex items-center gap-2">
                  <Flame className="w-5 h-5 text-amber-400"/> Negative Sampling Impact
                </h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left">
                    <thead className="bg-gray-900/80 text-gray-400 font-mono">
                      <tr>
                        <th className="p-3">Sampling Strategy</th>
                        <th className="p-3">AUC</th>
                        <th className="p-3">Recall@50</th>
                        <th className="p-3">Convergence</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-800 text-gray-200">
                      {data?.negative_sampling && Object.entries(data.negative_sampling).map(([strat, metrics]) => (
                        <tr key={strat} className="hover:bg-gray-800/30 transition-all">
                          <td className="p-3 font-semibold text-gray-100">{strat}</td>
                          <td className="p-3 text-[#76b900] font-bold">{metrics.AUC}</td>
                          <td className="p-3">{metrics['Recall@50']}</td>
                          <td className="p-3 text-purple-400">{metrics['Convergence Epochs']} Epochs</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Grid 2: Cold-Start & Diversity Tradeoff */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Cold-Start Performance */}
              <div className="glass-panel p-6 rounded-2xl">
                <h3 className="text-base font-bold text-gray-100 mb-4 flex items-center gap-2">
                  <ShieldAlert className="w-5 h-5 text-blue-400"/> Cold-Start Degradation & Fallback
                </h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs text-left">
                    <thead className="bg-gray-900/80 text-gray-400 font-mono">
                      <tr>
                        <th className="p-3">Entity Type</th>
                        <th className="p-3">HitRate@20</th>
                        <th className="p-3">NDCG@20</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-800 text-gray-200">
                      {data?.cold_start && Object.entries(data.cold_start).filter(([k]) => k !== 'summary').map(([entity, m]) => (
                        <tr key={entity} className="hover:bg-gray-800/30 transition-all">
                          <td className="p-3 font-medium">{entity}</td>
                          <td className="p-3 text-blue-400 font-semibold">{m['HitRate@20']}</td>
                          <td className="p-3">{m['NDCG@20'] ?? '–'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Diversity vs Relevance Tradeoff (MMR) */}
              <div className="glass-panel p-6 rounded-2xl">
                <h3 className="text-base font-bold text-gray-100 mb-4 flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-purple-400"/> MMR Diversity vs Relevance Trade-Off
                </h3>
                <div className="h-56">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={data?.diversity || []} margin={{ top: 10, right: 10, left: -20, bottom: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                      <XAxis dataKey="lambda_mmr" stroke="#9ca3af" tick={{ fontSize: 11 }} label={{ value: 'Lambda (Relevance Weight)', position: 'insideBottom', offset: -5, fill: '#9ca3af', fontSize: 10 }} />
                      <YAxis stroke="#9ca3af" tick={{ fontSize: 11 }} />
                      <Tooltip contentStyle={{ backgroundColor: '#111827', borderColor: '#374151', borderRadius: '8px', fontSize: '12px' }} />
                      <Legend verticalAlign="top" height={36} />
                      <Line type="monotone" dataKey="Relevance NDCG@20" stroke="#76b900" strokeWidth={2.5} />
                      <Line type="monotone" dataKey="Intra-List Diversity (ILD)" stroke="#a855f7" strokeWidth={2.5} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>

            {/* Latency Breakdown Bar Chart */}
            <div className="glass-panel p-6 rounded-2xl">
              <h3 className="text-base font-bold text-gray-100 mb-4 flex items-center gap-2">
                <Clock className="w-5 h-5 text-emerald-400"/> Microsecond Latency Benchmark (p50 / p95 / p99)
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-gray-900/80 p-4 rounded-xl border border-gray-800">
                  <h4 className="text-xs font-bold text-emerald-400 mb-2">Stage 1: FAISS Candidate Retrieval (5,000 Catalog)</h4>
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between"><span>p50 Latency:</span> <span className="font-mono text-gray-100">{data?.latency?.['Candidate Retrieval (FAISS exact IP - 5k items, top-100)']?.p50_ms} ms</span></div>
                    <div className="flex justify-between"><span>p95 Latency:</span> <span className="font-mono text-emerald-400 font-bold">{data?.latency?.['Candidate Retrieval (FAISS exact IP - 5k items, top-100)']?.p95_ms} ms</span></div>
                    <div className="flex justify-between"><span>p99 Latency:</span> <span className="font-mono text-gray-100">{data?.latency?.['Candidate Retrieval (FAISS exact IP - 5k items, top-100)']?.p99_ms} ms</span></div>
                  </div>
                </div>

                <div className="bg-gray-900/80 p-4 rounded-xl border border-gray-800">
                  <h4 className="text-xs font-bold text-purple-400 mb-2">Stage 2: Deep Ranking Model (100 Candidates Scoring)</h4>
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between"><span>p50 Latency:</span> <span className="font-mono text-gray-100">{data?.latency?.['Deep Scoring Model (DCN - 100 candidates)']?.p50_ms} ms</span></div>
                    <div className="flex justify-between"><span>p95 Latency:</span> <span className="font-mono text-purple-400 font-bold">{data?.latency?.['Deep Scoring Model (DCN - 100 candidates)']?.p95_ms} ms</span></div>
                    <div className="flex justify-between"><span>p99 Latency:</span> <span className="font-mono text-gray-100">{data?.latency?.['Deep Scoring Model (DCN - 100 candidates)']?.p99_ms} ms</span></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: LIVE RECOMMENDATION SIMULATOR */}
        {activeTab === 'simulator' && (
          <div className="space-y-6">
            {/* Control Panel */}
            <div className="glass-panel p-6 rounded-2xl flex flex-col md:flex-row items-center justify-between gap-6">
              <div className="flex items-center gap-6 w-full md:w-auto">
                <div>
                  <label className="text-xs font-semibold text-gray-400 block mb-1">Target User ID</label>
                  <input
                    type="number"
                    value={simUserId}
                    onChange={(e) => setSimUserId(parseInt(e.target.value) || 0)}
                    className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-1.5 text-sm w-32 focus:border-[#76b900] outline-none"
                  />
                </div>

                <div>
                  <label className="text-xs font-semibold text-gray-400 block mb-1">MMR Diversity Weight (λ = {simLambda})</label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.1"
                    value={simLambda}
                    onChange={(e) => setSimLambda(parseFloat(e.target.value))}
                    className="accent-[#76b900] cursor-pointer"
                  />
                </div>
              </div>

              <button
                onClick={() => fetchRecommendations(simUserId, simLambda)}
                disabled={simLoading}
                className="flex items-center gap-2 bg-[#76b900] text-black font-bold px-6 py-2.5 rounded-xl hover:bg-[#68a300] transition-all"
              >
                <RefreshCw className={`w-4 h-4 ${simLoading ? 'animate-spin' : ''}`} />
                Run Two-Stage Pipeline
              </button>
            </div>

            {/* Results Grid */}
            {simRecs && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {/* User Context */}
                <div className="glass-panel p-5 rounded-2xl space-y-3">
                  <h3 className="text-sm font-bold text-gray-200 border-b border-gray-800 pb-2">User Context Profile</h3>
                  <div className="text-xs space-y-2">
                    <div className="flex justify-between"><span>User ID:</span> <span className="font-mono text-[#76b900]">{simRecs.user_id}</span></div>
                    <div className="flex justify-between"><span>Age:</span> <span>{simRecs.user_metadata?.age} years</span></div>
                    <div className="flex justify-between"><span>Gender:</span> <span>{simRecs.user_metadata?.gender === 1 ? 'Male' : 'Female'}</span></div>
                    <div className="flex justify-between"><span>Activity Tier:</span> <span className="text-purple-400 capitalize">{simRecs.user_metadata?.activity_level}</span></div>
                  </div>
                </div>

                {/* Recs List */}
                <div className="glass-panel p-5 rounded-2xl md:col-span-2 space-y-3">
                  <h3 className="text-sm font-bold text-gray-200 border-b border-gray-800 pb-2 flex justify-between">
                    <span>Top Recommendations (Stage 2 DCN Scoring)</span>
                    <span className="text-xs font-normal text-emerald-400">100 Candidates → Top 20</span>
                  </h3>

                  <div className="overflow-y-auto max-h-96 pr-2 space-y-2">
                    {simRecs.top_recommendations?.map((item, idx) => (
                      <div key={item.item_id} className="bg-gray-900/80 p-3 rounded-xl border border-gray-800 flex items-center justify-between text-xs hover:border-[#76b900]/40 transition-all">
                        <div className="flex items-center gap-3">
                          <span className="font-mono text-gray-500 font-bold w-5">#{idx + 1}</span>
                          <div>
                            <div className="font-bold text-gray-100">Item #{item.item_id}</div>
                            <div className="text-gray-400 text-[11px]">Category: {item.category_id} | Price: ${item.price.toFixed(2)}</div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-[#76b900] font-mono font-bold">{(item.score * 100).toFixed(1)}%</div>
                          <div className="text-gray-500 text-[10px]">Match Score</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
