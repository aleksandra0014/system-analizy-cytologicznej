import { useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Brain, Activity, Eye, Zap } from "lucide-react";
import type { Results } from "@/types";
import { getClassColor, mapClass } from "@/lib/constants";

export default function CellsDetails({ results, showDetails, limeLoadingId, limeErrorById, openGradcam, openLime, onCorrectClass, }: { results: Results | null; showDetails: boolean; limeLoadingId: string | null; limeErrorById: Record<string, string>; openGradcam: (id: string) => void; openLime: (id: string) => Promise<void>; onCorrectClass: (cellId: string, newClass: string) => Promise<void>; }) {
  const [editingCell, setEditingCell] = useState<string | null>(null);
  const [correctedClasses, setCorrectedClasses] = useState<Record<string, string>>({});

  if (!results || !showDetails) return null;

  const handleCorrectClass = async (cellId: string, newClass: string) => {
    await onCorrectClass(cellId, newClass);
    setCorrectedClasses((prev) => ({ ...prev, [cellId]: newClass }));
    setEditingCell(null);
  };

  return (
    <div className="w-full max-w-7xl backdrop-blur-lg bg-white/95 rounded-2xl p-6 shadow-lg mx-4">
      <div className="flex items-center gap-3 mb-6">
        <Brain className="w-7 h-7 text-blue-600" />
        <h2 className="text-3xl font-bold text-gray-800">Individual Cell Analysis</h2>
      </div>

      {results.crop_public_urls ? (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
          {Object.entries(results.crop_public_urls).map(([id, url]) => {
            const rawCls = results.predict_fused?.[id] ?? "—";
            const cls = correctedClasses[id] || mapClass(rawCls);
            const probs = results.probs?.[id]?.fused ?? {};
            const features = results.features_list?.[id] ?? {};
            const raw = results.cells_explanations?.[id];
            const explanation = typeof raw === "string" ? raw : (raw?.explanation ?? "");

            const probEntries = Object.entries(probs).sort(([a], [b]) => a.localeCompare(b));
            const featureEntries = Object.entries(features).sort(([a], [b]) => a.localeCompare(b));

            const limeBusy = limeLoadingId === id;
            const limeErr = limeErrorById[id];
            const isEditing = editingCell === id;

            return (
              <Card key={id} className="overflow-hidden shadow-lg border-blue-100">
                <CardHeader className="pb-4">
                  <CardTitle className="flex items-center justify-between">
                    <span className="text-lg">Cell #{id}</span>
                    {isEditing ? (
                      <div className="flex gap-1 relative z-10 pointer-events-auto" onClick={(e) => e.stopPropagation()} onMouseDown={(e) => e.stopPropagation()}>
                        {["HSIL", "LSIL", "NSIL"].map((classOption) => (
                          <button key={classOption} type="button" onClick={() => handleCorrectClass(id, classOption)} className={`px-2 py-1 text-xs font-semibold rounded-md border-2 hover:scale-105 transition-transform ${getClassColor(classOption)}`}>
                            {classOption}
                          </button>
                        ))}
                        <button type="button" onClick={() => setEditingCell(null)} className="px-2 py-1 text-xs text-gray-600 hover:text-gray-800">✕</button>
                      </div>
                    ) : (
                      <button type="button" onClick={(e) => { e.stopPropagation(); setEditingCell(id); }} className={`inline-flex px-2 py-1 text-sm font-semibold rounded-md ${getClassColor(cls)} hover:ring-2 hover:ring-blue-300 transition-all cursor-pointer`} title="Click to correct classification">
                        {cls}
                        {correctedClasses[id] && <span className="ml-1 text-xs">✓</span>}
                      </button>
                    )}
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4 p-4">
                  <div className="w-full h-48 border border-gray-200 rounded-lg flex items-center justify-center bg-gradient-to-br from-gray-50 to-gray-100">
                    <img src={url} alt={`cell-${id}`} className="max-h-44 object-contain rounded" />
                  </div>

                  {explanation && (
                    <div className="bg-grey-50 border border-blue-200 rounded-lg p-3">
                      <p className="text-xs font-semibold text-grey-200 mb-1 flex items-center gap-1">
                        <Brain className="w-3 h-3" />
                        AI Explanation
                      </p>
                      <p className="text-sm text-gray-700 leading-relaxed">{explanation}</p>
                    </div>
                  )}

                  <div className="flex flex-wrap gap-2">
                    <Button variant="outline" size="sm" onClick={() => openGradcam(id)} className="flex-1 hover:bg-blue-50">
                      <Eye className="w-4 h-4 mr-1" />
                      Grad-CAM
                    </Button>
                    <Button variant="outline" size="sm" onClick={() => openLime(id)} disabled={limeBusy || (!results?.features_list?.[id] && !results?.slide_uid)} className="flex-1 hover:bg-green-50" title={!results?.features_list?.[id] && !results?.slide_uid ? "No features found and no slide UID" : ""}>
                      {limeBusy ? (<><div className="w-3 h-3 border border-gray-400 border-t-gray-600 rounded-full animate-spin mr-1" />LIME...</>) : (<><Brain className="w-4 h-4 mr-1" />LIME</>)}
                    </Button>
                  </div>
                  {limeErr && <p className="text-xs text-red-600 text-center">{limeErr}</p>}

                  {probEntries.length > 0 && (
                    <div>
                      <p className="font-semibold mb-2 text-gray-700 flex items-center gap-2">
                        <Activity className="w-4 h-4" />
                        Probabilities
                      </p>
                      <div className="space-y-1">
                        {probEntries.map(([k, v]) => {
                          const percentage = Number.isFinite(v as number) ? (Number(v) * 100).toFixed(1) : "0";
                          const mappedClass = mapClass(k);
                          return (
                            <div key={k} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                              <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-md border ${getClassColor(mappedClass)}`}>{mappedClass}</span>
                              <div className="flex items-center gap-2">
                                <div className="w-16 bg-gray-200 rounded-full h-2">
                                  <div className="bg-blue-600 h-2 rounded-full transition-all" style={{ width: `${percentage}%` }} />
                                </div>
                                <span className="text-sm font-mono w-12 text-right">{percentage}%</span>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {featureEntries.length > 0 && (
                    <div>
                      <p className="font-semibold mb-2 text-gray-700 flex items-center gap-2">
                        <Zap className="w-4 h-4" />
                        Features ({featureEntries.length})
                      </p>
                      <div className="max-h-32 overflow-auto border border-gray-200 rounded-lg">
                        <div className="divide-y divide-gray-100">
                          {featureEntries.map(([k, v]) => (
                            <div key={k} className="flex justify-between items-center p-2 text-sm">
                              <span className="text-gray-600 truncate flex-1 mr-2">{k}</span>
                              <span className="font-mono text-gray-800">{Number.isFinite(v as number) ? Number(v).toFixed(3) : "—"}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>
      ) : (
        <div className="text-center py-12">
          <Activity className="w-16 h-16 mx-auto mb-4 text-gray-400" />
          <p className="text-gray-500 text-lg">No cell crops available for analysis.</p>
        </div>
      )}
    </div>
  );
}