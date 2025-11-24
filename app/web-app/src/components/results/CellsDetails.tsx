import { useMemo, useState } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Brain, Activity, Eye, Zap, Trash, Filter, Info } from "lucide-react";
import type { Results } from "@/types";
import { getClassColor, mapClass } from "@/lib/constants";

export default function CellsDetails(props: {
  results: Results | null;
  showDetails: boolean;
  limeLoadingId: string | null;
  limeErrorById: Record<string, string>;
  openGradcam: (id: string) => void;
  openLime: (id: string) => Promise<void>;
  onCorrectClass: (cellId: string, newClass: string) => Promise<void>;
}) {
  const {
    results,
    showDetails,
    limeLoadingId,
    limeErrorById,
    openGradcam,
    openLime,
    onCorrectClass,
  } = props;

  // --- Hooki (stała kolejność) ---
  const [editingCell, setEditingCell] = useState<string | null>(null);
  const [correctedClasses, setCorrectedClasses] = useState<Record<string, string>>({});
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deletedIds, setDeletedIds] = useState<Record<string, true>>({});
  const BASE_API_URL = import.meta.env.VITE_API_URL;
  // Filtry: tylko HSIL / LSIL / NSIL
  type CellClass = "HSIL" | "LSIL" | "NSIL";
  type FilterKey = CellClass | "HSIL/LSIL_group";

  const [activeClasses, setActiveClasses] = useState<Record<FilterKey, boolean>>({
    HSIL: true,
    LSIL: true,
    NSIL: true,
    "HSIL/LSIL_group": true, // nowy filtr-grupa (domyślnie off)
  });

  // --- Handlery ---
  const handleCorrectClass = async (cellId: string, newClass: string) => {
    await onCorrectClass(cellId, newClass);
    setCorrectedClasses((prev) => ({ ...prev, [cellId]: newClass }));
    setEditingCell(null);
  };

  const handleDeleteCell = async (cellId: string) => {
    const slideUid = results?.slide_uid || (results as any)?.slajd_uid;
    if (!slideUid) {
      alert("Brak identyfikatora slajdu – nie mogę usunąć komórki.");
      return;
    }
    if (!window.confirm("Are you sure you want to delete cell?")) return;

    try {
      setDeletingId(cellId);
      const res = await fetch(
        `${BASE_API_URL}/slide/${encodeURIComponent(slideUid)}/cell/${encodeURIComponent(cellId)}`,
        { method: "DELETE", credentials: "include" }
      );
      if (!(res.ok || res.status === 204)) {
        const msg = await res.text().catch(() => "");
        throw new Error(msg || `Błąd usuwania (HTTP ${res.status})`);
      }
      setDeletedIds((prev) => ({ ...prev, [cellId]: true }));
      alert("Usunięto komórkę.");
    } catch (e: any) {
      alert(`Nie udało się usunąć komórki: ${e?.message ?? e}`);
    } finally {
      setDeletingId(null);
    }
  };

  // --- Dane po filtrach ---
  const items = useMemo(() => {
    if (!results?.crop_public_urls) return [];
    return Object.entries(results.crop_public_urls)
      .filter(([id]) => !deletedIds[id])
      .filter(([id]) => {
        const rawCls = results.predict_fused?.[id] ?? "—";
        const cls = (correctedClasses[id] || mapClass(rawCls)) as "HSIL" | "LSIL" | "NSIL" | "—";
        if (cls === "—") return false; // nie pokazujemy nieokreślonych
        return !!activeClasses[cls];
      });
  }, [results, deletedIds, correctedClasses, activeClasses]);

  const totalCount = Object.keys(results?.crop_public_urls ?? {}).length;
  const shownCount = items.length;

  // Jeśli nie chcemy nic pokazywać — po hookach możemy bezpiecznie wyjść
  if (!showDetails || !results?.crop_public_urls) return null;

  // --- Render ---
  return (
    <div className="w-full max-w-7xl backdrop-blur-lg bg-white/95 rounded-2xl p-6 shadow-lg mx-4">
      {/* Nagłówek + Filtry z objaśnieniem */}
      <div className="flex flex-col gap-3 mb-6">
        <div className="flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <Brain className="w-7 h-7 text-blue-600" />
            <h2 className="text-3xl font-bold text-gray-800">Individual Cell Analysis</h2>
          </div>
          <div className="text-sm text-gray-600">
            Showing <span className="font-semibold">{shownCount}</span> / {totalCount}
          </div>
        </div>

        <div className="flex flex-col gap-2 p-3 border rounded-xl bg-white/70">
          {/* Linia z filtrami */}
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1 text-gray-700 text-sm font-medium">
              <Filter className="w-4 h-4" /> Class:
            </span>

            {(["HSIL", "LSIL", "NSIL", "HSIL/LSIL_group"] as const).map((c) => {
              const active = activeClasses[c];
              return (
                <button
                  key={c}
                  type="button"
                  onClick={() => setActiveClasses((prev) => ({ ...prev, [c]: !prev[c] }))}
                  className={`px-2 py-1 text-xs font-semibold rounded-md border transition-all ${
                    active
                      ? (c === "HSIL/LSIL_group"
                          ? "bg-purple-100 text-purple-800 border-purple-300 ring-2 ring-purple-200"
                          : getClassColor(c as CellClass) + " ring-2 ring-blue-200")
                      : "bg-gray-50 text-gray-600 hover:bg-gray-100"
                  }`}
                  title={`Toggle ${c}`}
                >
                  {c}
                </button>
              );
            })}

            <button
              type="button"
              onClick={() => setActiveClasses({ HSIL: true, LSIL: true, NSIL: true, "HSIL/LSIL_group": false })}
              className="ml-2 px-2 py-1 text-xs rounded-md border text-gray-700 hover:bg-gray-50"
            >
              All
            </button>
            <button
              type="button"
              onClick={() => setActiveClasses({ HSIL: false, LSIL: false, NSIL: false, "HSIL/LSIL_group": false })}
              className="px-2 py-1 text-xs rounded-md border text-gray-700 hover:bg-gray-50"
            >
              None
            </button>
          </div>
        </div>
      </div>

      {/* Karty komórek */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
        {items.map(([id, url]) => {
          const rawCls = results.predict_fused?.[id] ?? "—";
          const cls = correctedClasses[id] || mapClass(rawCls);
          const probs = results.probs?.[id].fused ?? {};
          const features = results.features_list?.[id] ?? {};
          const raw = results.cells_explanations?.[id];
          const explanation = typeof raw === "string" ? raw : raw?.explanation ?? "";

          const probEntries = Object.entries(probs).sort(([a], [b]) => a.localeCompare(b));
          const featureEntries = Object.entries(features).sort(([a], [b]) => a.localeCompare(b));

          const limeBusy = limeLoadingId === id;
          const limeErr = limeErrorById[id];
          const isEditing = editingCell === id;
          const isDeleting = deletingId === id;

          return (
            <Card key={id} className="overflow-hidden shadow-lg border-blue-100">
              <CardHeader className="pb-4">
                <CardTitle className="flex items-center justify-between">
                  <span className="text-lg">Cell #{id}</span>

                  <div className="flex items-center gap-2">
                    {/* Korekta klasy */}
                    {isEditing ? (
                      <div
                        className="flex gap-1 relative z-10 pointer-events-auto"
                        onClick={(e) => e.stopPropagation()}
                        onMouseDown={(e) => e.stopPropagation()}
                      >
                        {["HSIL", "LSIL", "NSIL"].map((classOption) => (
                          <button
                            key={classOption}
                            type="button"
                            onClick={() => handleCorrectClass(id, classOption)}
                            className={`px-2 py-1 text-xs font-semibold rounded-md border-2 hover:scale-105 transition-transform ${getClassColor(
                              classOption
                            )}`}
                          >
                            {classOption}
                          </button>
                        ))}
                        <button
                          type="button"
                          onClick={() => setEditingCell(null)}
                          className="px-2 py-1 text-xs text-gray-600 hover:text-gray-800"
                        >
                          ✕
                        </button>
                      </div>
                    ) : (
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          setEditingCell(id);
                        }}
                        className={`inline-flex px-2 py-1 text-sm font-semibold rounded-md ${getClassColor(
                          cls
                        )} hover:ring-2 hover:ring-blue-300 transition-all cursor-pointer`}
                        title="Click to correct classification"
                      >
                        {cls}
                        {correctedClasses[id] && <span className="ml-1 text-xs">✓</span>}
                      </button>
                    )}

                    {/* Kosz */}
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteCell(id);
                      }}
                      disabled={isDeleting}
                      title="Usuń komórkę"
                      className="inline-flex items-center justify-center p-2 rounded-md border bg-white hover:bg-red-50 hover:text-red-600 transition-colors disabled:opacity-60"
                    >
                      {isDeleting ? (
                        <div className="w-4 h-4 border border-gray-400 border-t-gray-600 rounded-full animate-spin" />
                      ) : (
                        <Trash className="w-4 h-4" />
                      )}
                    </button>
                  </div>
                </CardTitle>
              </CardHeader>

              <CardContent className="space-y-4 p-4">
                <div className="w-full h-48 border border-gray-200 rounded-lg flex items-center justify-center bg-gradient-to-br from-gray-50 to-gray-100">
                  <img src={url} alt={`cell-${id}`} className="max-h-44 object-contain rounded" />
                </div>

                {explanation && (
                  <div className="bg-grey-50 border border-blue-200 rounded-lg p-3">
                    <p className="text-xs font-semibold text-grey-200 mb-1">AI Explanation</p>
                    <p className="text-sm text-gray-700 leading-relaxed">{explanation}</p>
                  </div>
                )}

                <div className="flex flex-wrap gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => openGradcam(id)}
                    className="flex-1 hover:bg-blue-50"
                  >
                    <Eye className="w-4 h-4 mr-1" />
                    Grad-CAM
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => openLime(id)}
                    disabled={limeBusy || (!results?.features_list?.[id] && !results?.slide_uid)}
                    className="flex-1 hover:bg-green-50"
                    title={!results?.features_list?.[id] && !results?.slide_uid ? "No features found and no slide UID" : ""}
                  >
                    {limeBusy ? (
                      <>
                        <div className="w-3 h-3 border border-gray-400 border-t-gray-600 rounded-full animate-spin mr-1" />
                        LIME...
                      </>
                    ) : (
                      <>LIME</>
                    )}
                  </Button>
                </div>
                {limeErr && <p className="text-xs text-red-600 text-center">{limeErr}</p>}

                {!!Object.entries(probs).length && (
                  <div>
                    <p className="font-semibold mb-2 text-gray-700 flex items-center gap-2">
                      Probabilities
                    </p>
                    <div className="space-y-1">
                      {Object.entries(probs)
                        .sort(([a], [b]) => a.localeCompare(b))
                        .map(([k, v]) => {
                          const percentage = Number.isFinite(v as number) ? (Number(v) * 100).toFixed(1) : "0";
                          const mappedClass = mapClass(k);
                          return (
                            <div key={k} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                              <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-md border ${getClassColor(mappedClass)}`}>
                                {mappedClass}
                              </span>
                              <div className="flex items-center gap-2">
                                <div className="w-16 bg-gray-200 rounded-full h-2">
                                  <div className="h-2 rounded-full transition-all bg-blue-600" style={{ width: `${percentage}%` }} />
                                </div>
                                <span className="text-sm font-mono w-12 text-right">{percentage}%</span>
                              </div>
                            </div>
                          );
                        })}
                    </div>
                  </div>
                )}

                {!!Object.entries(features).length && (
                  <div>
                    <p className="font-semibold mb-2 text-gray-700 flex items-center gap-2">
                      Features ({Object.keys(features).length})
                    </p>
                    <div className="max-h-32 overflow-auto border border-gray-200 rounded-lg">
                      <div className="divide-y divide-gray-100">
                        {Object.entries(features)
                          .sort(([a], [b]) => a.localeCompare(b))
                          .map(([k, v]) => (
                            <div key={k} className="flex justify-between items-center p-2 text-sm">
                              <span className="text-gray-600 truncate flex-1 mr-2">{k}</span>
                              <span className="font-mono text-gray-800">
                                {Number.isFinite(v as number) ? Number(v).toFixed(3) : "—"}
                              </span>
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
    </div>
  );
}
