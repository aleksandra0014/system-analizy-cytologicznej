import { Eye, Brain } from "lucide-react";

export default function Home({ onGoAdd, onGoOld }: { onGoAdd: () => void; onGoOld: () => void }) {
  return (
    <div className="max-w-5xl w-full mx-auto px-4">
      <div className="text-center mb-12">
        <h2 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent mb-4">Welcome</h2>
        <p className="text-xl text-gray-600 max-w-2xl mx-auto">Advanced AI-powered cervical cell analysis and classification system</p>
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        <button onClick={onGoOld} className="group rounded-3xl p-8 bg-gradient-to-br from-white/90 to-blue-50/50 hover:from-white hover:to-blue-50 shadow-lg hover:shadow-xl border border-blue-100 transition-all duration-300 text-left transform hover:-translate-y-1">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center">
              <Eye className="w-8 h-8 text-white" />
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-800 mb-1">View Results</div>
              <span className="inline-flex px-2 py-1 text-xs font-medium rounded-md bg-blue-100 text-blue-800 border border-blue-200">Historical Data</span>
            </div>
          </div>
          <p className="text-gray-600 mb-6">Browse previously analyzed slides. Select patients and view detailed results including Grad-CAM visualizations and LIME explanations.</p>
          <div className="flex items-center text-blue-700 font-semibold group-hover:translate-x-2 transition-transform">Explore Archive <span className="ml-2">→</span></div>
        </button>

        <button onClick={onGoAdd} className="group rounded-3xl p-8 bg-gradient-to-br from-white/90 to-green-50/50 hover:from-white hover:to-green-50 shadow-lg hover:shadow-xl border border-green-100 transition-all duration-300 text-left transform hover:-translate-y-1">
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-800 mb-1">New Analysis</div>
              <span className="inline-flex px-2 py-1 text-xs font-medium rounded-md bg-green-100 text-green-800 border border-green-200">AI Processing</span>
            </div>
          </div>
          <p className="text-gray-600 mb-6">Upload a new cervical slide image for AI-powered cell classification and analysis. Get instant results with detailed explanations.</p>
          <div className="flex items-center text-green-700 font-semibold group-hover:translate-x-2 transition-transform">Start Analysis <span className="ml-2">→</span></div>
        </button>
      </div>
    </div>
  );
}