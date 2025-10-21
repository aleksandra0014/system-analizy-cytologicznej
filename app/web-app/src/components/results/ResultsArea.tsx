import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Activity, Brain } from "lucide-react";
import type { Results } from "@/types";
import { getClassColor } from "@/lib/constants";

interface ResultsAreaProps {
  results: Results | null;
  imageUrl: string | null;
  overallClass: string;
  totalCells: number;
  classCounts: Record<string, number>;
  addInfoDraft: string;
  setAddInfoDraft: (v: string) => void;
  savingAddInfo: boolean;
  saveAddInfoMsg: string | null;
  onSaveAddInfo: () => void;
  loading: boolean;
  showDetails: boolean;
  setShowDetails: (v: boolean) => void;
}

export default function ResultsArea({ results, imageUrl, overallClass, totalCells, classCounts, addInfoDraft, setAddInfoDraft, savingAddInfo, saveAddInfoMsg, onSaveAddInfo, loading, showDetails, setShowDetails, }: ResultsAreaProps) {
  if (!results || loading) return null;

  return (
    <div className="w-full max-w-7xl mx-auto px-4 py-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
        <div className="lg:col-span-2 flex flex-col gap-6">
          <div className="rounded-2xl border border-blue-100 bg-gradient-to-br from-white to-blue-50/30 p-4 shadow-lg">
            <div className="relative w-full h-full min-h-[320px] flex items-center justify-center">
              {imageUrl ? (
                <>
                  <img src={imageUrl} alt="Detected cells with bounding boxes" className="w-full h-full object-contain rounded-xl" />
                  <div className="absolute top-2 right-2">
                    <span className={`inline-flex px-3 py-1 text-sm font-semibold rounded-full ${getClassColor(overallClass)}`}>
                      {overallClass}
                    </span>
                  </div>
                </>
              ) : (
                <div className="text-center text-gray-500">
                  <Activity className="w-12 h-12 mx-auto mb-2 opacity-50" />
                  <p>No image available</p>
                </div>
              )}
            </div>
          </div>

          <div className="bg-white/95 backdrop-blur-lg border border-blue-100 rounded-xl p-6 shadow-md">
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0">
                <div className="w-10 h-10 bg-blue-100 rounded-xl flex items-center justify-center">
                  <Activity className="w-5 h-5 text-blue-600" />
                </div>
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="text-lg font-semibold text-gray-800 mb-3">Analysis Summary</h3>
                {results.slide_summary_text || results.slide_summary?.explanation ? (
                  <p className="text-sm text-gray-700 leading-relaxed mb-4">{results.slide_summary_text || results.slide_summary?.explanation}</p>
                ) : (
                  <p className="text-sm text-gray-600 leading-relaxed mb-4">Automated cell detection and classification results with bounding boxes highlighting detected cells. Each cell is analyzed and classified based on morphological features and staining patterns.</p>
                )}
                {totalCells > 0 && (
                  <div className="flex items-center gap-6 text-sm text-gray-600">
                    <span className="flex items-center gap-2">Detected: <strong className="text-blue-600 text-lg">{totalCells}</strong> cells</span>
                    <span className="flex items-center gap-2">Primary class: <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getClassColor(overallClass)}`}>{overallClass}</span></span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        <div className="lg:col-start-3 flex flex-col gap-6">
          <Card className="flex flex-col min-h-0 backdrop-blur-lg bg-white/95 border-blue-100 shadow-lg">
            <CardHeader className="pb-3 shrink-0">
              <CardTitle className="text-lg font-semibold text-gray-800 flex items-center gap-2">
                <Activity className="w-5 h-5 text-blue-600" />
                Analysis Summary
              </CardTitle>
            </CardHeader>
            <CardContent className="flex-1 overflow-y-auto space-y-4">
              <div className="grid grid-cols-1 gap-4">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <p className="text-sm text-gray-500 font-medium">Slide ID</p>
                    <p className="text-sm font-mono break-all">{results.slide_uid ?? "—"}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500 font-medium">Patient ID</p>
                    <p className="text-sm font-mono break-all">{results.pacjent_uid ?? "—"}</p>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <p className="text-sm text-gray-500 font-medium">Classification</p>
                    <span className={`inline-flex px-3 py-1 text-sm font-semibold rounded-full ${getClassColor(overallClass)}`}>{overallClass}</span>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500 font-medium">Total Cells</p>
                    <p className="text-xl font-bold text-blue-600">{totalCells}</p>
                  </div>
                </div>
              </div>

              {totalCells > 0 && (
                <div>
                  <p className="font-semibold mb-3 text-gray-700">Cell Distribution</p>
                  <div className="grid grid-cols-3 gap-2">
                    {Object.entries(classCounts).map(([cls, count]) => (
                      <div key={cls} className="text-center p-2 rounded-lg bg-gray-50">
                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full mb-1 ${getClassColor(cls)}`}>{cls}</span>
                        <p className="text-lg font-bold text-gray-700">{count}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="shrink-0 backdrop-blur-lg bg-white/95 border-blue-100 shadow-md">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-semibold text-gray-800 flex items-center gap-2">
                Notes
              </CardTitle>
            </CardHeader>
            <CardContent>
              <textarea rows={2} className="w-full border border-gray-200 rounded-lg px-3 py-2 bg-white resize-none text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent" value={addInfoDraft} onChange={(e) => setAddInfoDraft(e.target.value)} placeholder="Add notes..." />
              <div className="mt-2 flex items-center gap-3">
                <button disabled={savingAddInfo} onClick={onSaveAddInfo} className="text-sm px-3 py-1 rounded-md border bg-blue-600 text-white disabled:opacity-60">{savingAddInfo ? "Saving..." : "Save"}</button>
                {saveAddInfoMsg && (
                  <span className={`text-xs ${saveAddInfoMsg.includes("Saved") ? "text-green-600" : "text-gray-600"}`}>{saveAddInfoMsg}</span>
                )}
              </div>
            </CardContent>
          </Card>

          <Button variant={showDetails ? "secondary" : "default"} disabled={!results || loading} onClick={() => setShowDetails(!showDetails)} className="w-full shrink-0 flex items-center justify-center gap-2 text-sm px-6 h-10">
            <Brain className="w-5 h-5" />
            {showDetails ? "Hide Details" : "Show Details"}
          </Button>
        </div>
      </div>
    </div>
  );
}