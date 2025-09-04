import { useMemo, useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import FileUpload from "./components/FileUpload";

type Results = {
  request_id?: string;
  predict_fused?: Record<string, string | number>;
  bbox_public_url?: string;
  crop_public_urls?: Record<string, string>;
  probs?: Record<string, { fused?: Record<string, number> }>;
  features_list?: Record<string, Record<string, number>>;
  slide_summary?: {
    overall_class?: string | number;
    explanation?: string;
    confidence?: number;
  };
  slide_summary_text?: string;
  overall_class?: string | number;
};

type GradcamResp = {
  overlay_url: string;
  heatmap_url: string;
  activation_url: string;
  predicted_class: string;
};

/** 0→HSIL, 1→LSIL, 2→NSIL (i przepuszczamy już-nazwane wartości) */
const CLASS_NAME_MAP: Record<string, string> = {
  "0": "HSIL",
  "1": "LSIL",
  "2": "NSIL",
  "HSIL": "HSIL",
  "LSIL": "LSIL",
  "NSIL": "NSIL",
};
const mapClass = (v: unknown) => CLASS_NAME_MAP[String(v)] ?? String(v);

export default function App() {
  const [patientId, setPatientId] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [results, setResults] = useState<Results | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState(false);

  // Grad-CAM modal state
  const [gradcamOpen, setGradcamOpen] = useState(false);
  const [gradcamForId, setGradcamForId] = useState<string | null>(null);
  const [gradcamData, setGradcamData] = useState<GradcamResp | null>(null);
  const [gradcamLoading, setGradcamLoading] = useState(false);
  const [gradcamError, setGradcamError] = useState<string | null>(null);

  // LIME — per-card loading/error
  const [limeLoadingId, setLimeLoadingId] = useState<string | null>(null);
  const [limeErrorById, setLimeErrorById] = useState<Record<string, string>>({});

  const handleProcess = async (file: File) => {
    try {
      setErrorMsg(null);
      setLoading(true);
      setResults(null);
      setImageUrl(null);
      setShowDetails(false);

      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("http://localhost:8000/process-image/", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Request failed with ${response.status}`);
      }

      const data: Results = await response.json();
      setResults(data);
      if (data.bbox_public_url) setImageUrl(data.bbox_public_url);
    } catch (err: any) {
      setErrorMsg(err?.message || "Unexpected error");
    } finally {
      setLoading(false);
    }
  };

  const totalCells = results?.predict_fused
    ? Object.keys(results.predict_fused).length
    : 0;

  const classCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    if (results?.predict_fused) {
      for (const raw of Object.values(results.predict_fused)) {
        const label = mapClass(raw);
        counts[label] = (counts[label] || 0) + 1;
      }
    }
    return counts;
  }, [results?.predict_fused]);

  const openGradcam = async (cellId: string) => {
    const cropUrl = results?.crop_public_urls?.[cellId];
    if (!cropUrl) return;

    setGradcamOpen(true);
    setGradcamForId(cellId);
    setGradcamData(null);
    setGradcamError(null);
    setGradcamLoading(true);

    try {
      const resp = await fetch("http://localhost:8000/gradcam/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_url: cropUrl }),
      });
      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(txt || `Grad-CAM failed with ${resp.status}`);
      }
      const data: GradcamResp = await resp.json();
      setGradcamData(data);
    } catch (e: any) {
      setGradcamError(e?.message || "Grad-CAM error");
      console.error("Grad-CAM error:", e);
    } finally {
      setGradcamLoading(false);
    }
  };

  const openLime = async (cellId: string) => {
    if (!results?.features_list?.[cellId]) {
      setLimeErrorById((p) => ({ ...p, [cellId]: "No features for this cell (run Process again)" }));
      return;
    }
    setLimeLoadingId(cellId);
    setLimeErrorById((p) => {
      const q = { ...p };
      delete q[cellId];
      return q;
    });

    const payload = {
      cell_id: cellId,
      features: results.features_list[cellId], // <--- KLUCZOWE: wysyłamy cechy tej komórki
    };

    try {
      const resp = await fetch("http://localhost:8000/lime/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(txt || `LIME failed with ${resp.status}`);
      }
      const data: { html_url: string } = await resp.json();
      window.open(data.html_url, "_blank", "noopener,noreferrer");
    } catch (e: any) {
      const msg: string =
        e?.message ||
        "LIME error. Tip: ensure backend hasn't reloaded or provide features.";
      setLimeErrorById((p) => ({ ...p, [cellId]: msg }));
      console.error("LIME error:", e);
    } finally {
      setLimeLoadingId(null);
    }
  };

  const overallClass =
    results?.overall_class != null
      ? mapClass(results.overall_class)
      : results?.slide_summary?.overall_class != null
      ? mapClass(results.slide_summary.overall_class)
      : "—";

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-10 gap-8">
      <h1 className="text-5xl font-bold text-blue-900">LBC SLIDES ANALYSIS</h1>

      <div className="flex flex-col gap-4 w-full max-w-xl items-center">
        <Input
          placeholder="Patient ID"
          value={patientId}
          onChange={(e) => setPatientId(e.target.value)}
        />

        <FileUpload onFileSelect={(file) => setSelectedFile(file)} />

        <div className="flex gap-2">
          <Button
            disabled={!patientId || !selectedFile || loading}
            onClick={() => selectedFile && handleProcess(selectedFile)}
          >
            {loading ? "Processing..." : "Process"}
          </Button>
        </div>

        {errorMsg && (
          <p className="text-sm text-red-600 text-center max-w-xl">{errorMsg}</p>
        )}
      </div>

      {results && !loading && (
        <div className="w-full max-w-7xl grid grid-cols-1 md:grid-cols-3 gap-6 items-start">
          {/* Image */}
          <div className="md:col-span-2 h-[640px] rounded border bg-white p-2 flex items-center justify-center">
            {imageUrl ? (
              <img
                src={imageUrl}
                alt="Detected cells with bounding boxes"
                className="w-full h-full object-contain rounded"
              />
            ) : (
              <div className="text-gray-500">No image available</div>
            )}
          </div>

          {/* Sidebar */}
          <div className="h-[640px] flex flex-col gap-4">
            <Card className="flex-1 overflow-auto">
              <CardHeader>
                <CardTitle>Summary</CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <p>
                  <strong>Overall class:</strong> {overallClass}
                </p>
                <p>
                  <strong>Total cells:</strong> {totalCells}
                </p>

                {totalCells > 0 && (
                  <div className="mt-2">
                    <p className="font-semibold mb-1">Cells per class:</p>
                    <table className="text-sm w-full border">
                      <thead>
                        <tr className="bg-gray-100">
                          <th className="border px-2 py-1 text-left">Class</th>
                          <th className="border px-2 py-1 text-right">Count</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(classCounts).map(([cls, count]) => (
                          <tr key={cls}>
                            <td className="border px-2 py-1">{cls}</td>
                            <td className="border px-2 py-1 text-right">{count}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}

                <p className="text-sm text-gray-600 mt-3">
                  {results.slide_summary_text ??
                    results.slide_summary?.explanation ??
                    ""}
                </p>
              </CardContent>
            </Card>

            <Card className="flex-1 overflow-auto">
              <CardHeader>
                <CardTitle>Predictions</CardTitle>
              </CardHeader>
              <CardContent>
                {results.predict_fused ? (
                  <div className="max-h-[220px] overflow-auto">
                    <table className="text-sm w-full border">
                      <thead>
                        <tr className="bg-gray-100">
                          <th className="border px-2 py-1">Cell ID</th>
                          <th className="border px-2 py-1">Class</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(results.predict_fused).map(([id, raw]) => {
                          const label = mapClass(raw);
                          return (
                            <tr key={id}>
                              <td className="border px-2 py-1">{id}</td>
                              <td className="border px-2 py-1">{label}</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="text-gray-500">No predictions</p>
                )}
              </CardContent>
            </Card>
            <Button
            variant="secondary"
            disabled={!results || loading}
            onClick={() => setShowDetails((s) => !s)}
          >
            {showDetails ? "Hide details" : "Details"}
          </Button>
          </div>
        </div>
      )}

      {/* Details grid */}
      {results && showDetails && (
        <div className="w-full max-w-7xl bg-white rounded p-4 shadow">
          <h2 className="text-2xl font-semibold mb-4">Cells — details</h2>

          {results.crop_public_urls ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.entries(results.crop_public_urls).map(([id, url]) => {
                const rawCls = results.predict_fused?.[id] ?? "—";
                const cls = mapClass(rawCls);
                const probs = results.probs?.[id]?.fused ?? {};
                const features = results.features_list?.[id] ?? {};

                const probEntries = Object.entries(probs).sort(([a], [b]) => a.localeCompare(b));
                const featureEntries = Object.entries(features).sort(([a], [b]) => a.localeCompare(b));

                const limeBusy = limeLoadingId === id;
                const limeErr = limeErrorById[id];

                return (
                  <Card key={id} className="overflow-hidden">
                    <CardHeader className="pb-2">
                      <CardTitle className="flex items-center justify-between">
                        <span>Cell #{id}</span>
                        <span className="text-sm px-2 py-1 rounded bg-blue-50 text-blue-800">
                          {cls}
                        </span>
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="w-full h-48 border rounded flex items-center justify-center bg-gray-50">
                        <img src={url} alt={`cell-${id}`} className="max-h-40 object-contain" />
                      </div>

                      <div className="flex flex-wrap gap-2 items-center">
                        <Button variant="outline" size="sm" onClick={() => openGradcam(id)}>
                          Grad-CAM
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => openLime(id)}
                          disabled={limeBusy || !results?.features_list?.[id]}
                          title={!results?.features_list?.[id] ? "No features found for this cell" : ""}
                        >
                          {limeBusy ? "LIME…" : "LIME"}
                        </Button>
                        {limeErr && (
                          <span className="text-xs text-red-600">
                            {limeErr}
                          </span>
                        )}
                      </div>

                      {/* Probabilities */}
                      {probEntries.length > 0 && (
                        <div>
                          <p className="font-semibold mb-1">Probabilities</p>
                          <table className="text-sm w-full border">
                            <thead>
                              <tr className="bg-gray-100">
                                <th className="border px-2 py-1 text-left">Class</th>
                                <th className="border px-2 py-1 text-right">Prob</th>
                              </tr>
                            </thead>
                            <tbody>
                              {probEntries.map(([k, v]) => (
                                <tr key={k}>
                                  <td className="border px-2 py-1">{mapClass(k)}</td>
                                  <td className="border px-2 py-1 text-right">
                                    {Number.isFinite(v) ? Number(v).toFixed(3) : "—"}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      )}

                      {/* Features */}
                      {featureEntries.length > 0 && (
                        <div>
                          <p className="font-semibold mb-1">Features</p>
                          <div className="max-h-40 overflow-auto border rounded">
                            <table className="text-sm w-full">
                              <thead>
                                <tr className="bg-gray-100">
                                  <th className="border px-2 py-1 text-left">Feature</th>
                                  <th className="border px-2 py-1 text-right">Value</th>
                                </tr>
                              </thead>
                              <tbody>
                                {featureEntries.map(([k, v]) => (
                                  <tr key={k}>
                                    <td className="border px-2 py-1">{k}</td>
                                    <td className="border px-2 py-1 text-right">
                                      {Number.isFinite(v) ? Number(v).toFixed(3) : "—"}
                                    </td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          ) : (
            <p className="text-gray-500">No crops available.</p>
          )}
        </div>
      )}

      {/* Grad-CAM Dialog */}
      <Dialog open={gradcamOpen} onOpenChange={setGradcamOpen}>
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <DialogTitle>Grad-CAM {gradcamForId ? `for Cell #${gradcamForId}` : ""}</DialogTitle>
          </DialogHeader>

          {gradcamLoading && (
            <div className="py-6 text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
              <p className="mt-4 text-blue-700">Generating Grad-CAM…</p>
            </div>
          )}

          {gradcamError && <p className="text-sm text-red-600">{gradcamError}</p>}

          {gradcamData && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="border rounded p-2">
                <p className="font-semibold mb-1">Overlay</p>
                <img src={gradcamData.overlay_url} className="w-full h-auto" />
              </div>
              <div className="border rounded p-2">
                <p className="font-semibold mb-1">Grad-CAM</p>
                <img src={gradcamData.heatmap_url} className="w-full h-auto" />
              </div>
              <div className="border rounded p-2">
                <p className="font-semibold mb-1">Avg Activation</p>
                <img src={gradcamData.activation_url} className="w-full h-auto" />
              </div>
              <div className="md:col-span-3">
                <p className="text-sm text-gray-700">
                  Predicted class (for CAM): <strong>{gradcamData.predicted_class}</strong>
                </p>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {loading && (
        <div className="flex flex-col items-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600"></div>
          <p className="mt-4 text-blue-700">Processing...</p>
        </div>
      )}
    </div>
  );
}
