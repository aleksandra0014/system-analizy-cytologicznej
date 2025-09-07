import { useEffect, useMemo, useRef, useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import FileUpload from "./components/FileUpload";

type Results = {
  slide_uid?: string;
  pacjent_uid?: string;
  slide_id?: string;           // legacy
  pacjent_db_id?: string;      // legacy
  predict_fused?: Record<string, string | number>;
  bbox_public_url?: string;
  crop_public_urls?: Record<string, string>;
  crop_gridfs_names?: Record<string, string>;
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

type Patient = { pacjent_uid: string; created_at?: string | null };
type SlideItem = {
  slajd_uid: string;
  status?: string | null;
  overall_class?: string | number | null;
  created_at?: string | null;
};

/** 0→HSIL, 1→LSIL, 2→NSIL */
const CLASS_NAME_MAP: Record<string, string> = {
  "0": "HSIL",
  "1": "LSIL",
  "2": "NSIL",
  HSIL: "HSIL",
  LSIL: "LSIL",
  NSIL: "NSIL",
};
const mapClass = (v: unknown) => CLASS_NAME_MAP[String(v)] ?? String(v);

// Strony aplikacji
type Mode = "home" | "add" | "old";

export default function App() {
  const [mode, setMode] = useState<Mode>("home");

  // Wspólne stany
  const [patientId, setPatientId] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const [results, setResults] = useState<Results | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState(false);

  // Grad-CAM modal
  const [gradcamOpen, setGradcamOpen] = useState(false);
  const [gradcamForId, setGradcamForId] = useState<string | null>(null);
  const [gradcamData, setGradcamData] = useState<GradcamResp | null>(null);
  const [gradcamLoading, setGradcamLoading] = useState(false);
  const [gradcamError, setGradcamError] = useState<string | null>(null);

  // LIME per-card
  const [limeLoadingId, setLimeLoadingId] = useState<string | null>(null);
  const [limeErrorById, setLimeErrorById] = useState<Record<string, string>>({});

  // Progress
  const [progress, setProgress] = useState(0);
  const progressTimerRef = useRef<number | null>(null);

  // --- Stany dla "Show old data" (dropdowny) ---
  const [patients, setPatients] = useState<Patient[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<string>("");
  const [slides, setSlides] = useState<SlideItem[]>([]);
  const [selectedSlide, setSelectedSlide] = useState<string>("");

  const resetState = () => {
    setResults(null);
    setImageUrl(null);
    setShowDetails(false);
    setErrorMsg(null);
    setSelectedFile(null);
    setGradcamOpen(false);
    setGradcamData(null);
    setLimeLoadingId(null);
    setLimeErrorById({});
    setProgress(0);

    // old-data
    setSlides([]);
    setSelectedSlide("");
  };

  const goHome = () => {
    resetState();
    setMode("home");
    setSelectedPatient("");
    setPatients([]);
  };

  const startIndeterminateProgress = () => {
    setProgress(0);
    progressTimerRef.current = window.setInterval(() => {
      setProgress((p) => (p < 90 ? p + 1 : p));
    }, 60);
  };
  const stopIndeterminateProgress = () => {
    if (progressTimerRef.current != null) {
      window.clearInterval(progressTimerRef.current);
      progressTimerRef.current = null;
    }
  };

  // ========== ADD NEW SLIDE ==========
  const handleProcess = async (file: File) => {
    try {
      setErrorMsg(null);
      setLoading(true);
      resetState();
      startIndeterminateProgress();

      const formData = new FormData();
      formData.append("file", file);

      const url = `http://localhost:8000/process-image/?pacjent_id=${encodeURIComponent(
        patientId || "UNKNOWN"
      )}`;

      const response = await fetch(url, { method: "POST", body: formData });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Request failed with ${response.status}`);
      }

      const data: Results = await response.json();
      setResults(data);
      setImageUrl(data.bbox_public_url || null);
      setShowDetails(true);
      setProgress(100);
    } catch (err: any) {
      setErrorMsg(err?.message || "Unexpected error");
    } finally {
      stopIndeterminateProgress();
      setLoading(false);
      setTimeout(() => setProgress(0), 400);
    }
  };

  // ========== SHOW OLD DATA ==========
  // 1) Po wejściu w tryb "old" pobierz listę pacjentów
  useEffect(() => {
    if (mode !== "old") return;
    (async () => {
      try {
        setErrorMsg(null);
        setPatients([]);
        setSlides([]);
        setSelectedPatient("");
        setSelectedSlide("");
        const resp = await fetch("http://localhost:8000/patients");
        if (!resp.ok) {
          const txt = await resp.text();
          throw new Error(txt || `Failed to load patients (${resp.status})`);
        }
        const data: { patients: Patient[] } = await resp.json();
        setPatients(data.patients || []);
      } catch (e: any) {
        setErrorMsg(e?.message || "Failed to load patients");
      }
    })();
  }, [mode]);

  // 2) Po wyborze pacjenta – pobierz jego slajdy
  const onSelectPatient = async (uid: string) => {
    setSelectedPatient(uid);
    setSlides([]);
    setSelectedSlide("");
    setResults(null);
    setImageUrl(null);
    setShowDetails(false);

    if (!uid) return;
    try {
      setErrorMsg(null);
      const resp = await fetch(`http://localhost:8000/patient/${encodeURIComponent(uid)}/slides`);
      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(txt || `Failed to load slides (${resp.status})`);
      }
      const data: { slides: SlideItem[] } = await resp.json();
      setSlides(data.slides || []);
    } catch (e: any) {
      setErrorMsg(e?.message || "Failed to load slides");
    }
  };

  // 3) Po wyborze slajdu – tylko ustawiamy ID (pobranie po kliknięciu Show)
  const onSelectSlide = (sid: string) => {
    setSelectedSlide(sid);
  };

  // 4) Pobierz i pokaż slajd po wybranym ID
  const handleShowSlide = async () => {
    if (!selectedSlide.trim()) return;
    try {
      setErrorMsg(null);
      setLoading(true);
      setResults(null);
      setImageUrl(null);
      setShowDetails(true);

      const resp = await fetch(`http://localhost:8000/slide/${encodeURIComponent(selectedSlide.trim())}`);
      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(txt || `Failed to load slide (${resp.status})`);
      }
      const data: Results = await resp.json();
      setResults(data);
      setImageUrl(data.bbox_public_url || null);
    } catch (e: any) {
      setErrorMsg(e?.message || "Failed to load slide");
    } finally {
      setLoading(false);
    }
  };

  // Statystyki
  const totalCells = results?.predict_fused ? Object.keys(results.predict_fused).length : 0;
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

  // Grad-CAM
  const openGradcam = async (cellId: string) => {
    const cropUrl = results?.crop_public_urls?.[cellId];
    const cropGridName = results?.crop_gridfs_names?.[cellId];

    if (!cropUrl && !cropGridName) return;

    setGradcamOpen(true);
    setGradcamForId(cellId);
    setGradcamData(null);
    setGradcamError(null);
    setGradcamLoading(true);

    try {
      const body = cropGridName ? { crop_gridfs_name: cropGridName } : { image_url: cropUrl };
      const resp = await fetch("http://localhost:8000/gradcam/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(txt || `Grad-CAM failed with ${resp.status}`);
      }
      const data: GradcamResp = await resp.json();
      setGradcamData(data);
    } catch (e: any) {
      const msg = e?.message || "Grad-CAM error";
      setGradcamError(msg);
      console.error("Grad-CAM error:", e);
    } finally {
      setGradcamLoading(false);
    }
  };

  // LIME (działa dla fresh i old – fallback do komorka_uid)
  const openLime = async (cellId: string) => {
    const komorkaUid =
      results?.slide_uid ? `${results.slide_uid}:${cellId}` :
      selectedSlide ? `${selectedSlide}:${cellId}` : null;

    const haveFeatures = !!results?.features_list?.[cellId];

    const payload: any = {};
    if (haveFeatures) {
      payload.features = results!.features_list![cellId];
    } else if (komorkaUid) {
      payload.komorka_uid = komorkaUid;
    } else {
      setLimeErrorById((p) => ({ ...p, [cellId]: "No features or komorka_uid" }));
      return;
    }

    setLimeLoadingId(cellId);
    setLimeErrorById((p) => {
      const q = { ...p };
      delete q[cellId];
      return q;
    });

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
      const msg: string = e?.message || "LIME error.";
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

  // ----- UI -----

  const TopBar = () => (
    <div className="w-full sticky top-0 z-40 backdrop-blur bg-white/50 border-b border-white/20">
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
        <button
          onClick={goHome}
          className="text-sm px-3 py-1.5 rounded-full bg-white/60 hover:bg-white shadow transition"
          title="Back to Home"
        >
          Home
        </button>
        <h1 className="text-xl md:text-2xl font-bold text-blue-900 tracking-tight">
          LBC Slides Analysis
        </h1>
        <div className="opacity-0 pointer-events-none">.</div>
      </div>
    </div>
  );

  const Home = () => (
    <div className="max-w-4xl w-full mx-auto px-4">
      <div className="text-center mb-10">
        <h2 className="text-4xl md:text-5xl font-extrabold text-blue-900 drop-shadow-sm">
          Welcome
        </h2>
        <p className="mt-3 text-gray-600">
          Choose what you want to do – add a new slide or view saved data.
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <button
          onClick={() => { resetState(); setMode("old"); }}
          className="group rounded-2xl p-6 bg-white/70 hover:bg-white shadow-lg hover:shadow-xl border border-white/40 transition text-left"
        >
          <div className="text-3xl mb-3">📚</div>
          <div className="text-2xl font-semibold text-blue-900 mb-1">Show old data</div>
          <p className="text-gray-600">
            First, select the patient, then the slide. View results (Grad-CAM, LIME).
          </p>
          <div className="mt-4 text-blue-700 font-medium group-hover:translate-x-1 transition">
            Open →
          </div>
        </button>

        <button
          onClick={() => { resetState(); setMode("add"); }}
          className="group rounded-2xl p-6 bg-white/70 hover:bg-white shadow-lg hover:shadow-xl border border-white/40 transition text-left"
        >
          <div className="text-3xl mb-3">➕</div>
          <div className="text-2xl font-semibold text-blue-900 mb-1">Add new slide</div>
          <p className="text-gray-600">
            Upload image, provide <span className="font-medium">Patient ID</span> and run the analysis.
          </p>
          <div className="mt-4 text-blue-700 font-medium group-hover:translate-x-1 transition">
            Add →
          </div>
        </button>
      </div>
    </div>
  );

  // Formularz dla dwóch trybów
  const ModeForms = () => {
    if (mode === "add") {
      return (
        <div className="max-w-3xl w-full mx-auto px-4">
          <Card className="backdrop-blur bg-white/70 border-white/50 shadow-xl">
            <CardHeader>
              <CardTitle className="text-sky-900">Add new slide</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <Input
                placeholder="Patient ID"
                value={patientId}
                onChange={(e) => setPatientId(e.target.value)}
              />
              <FileUpload 
                selectedFile={selectedFile}
                onFileSelect={(file) => setSelectedFile(file)} 
              />
              <div className="flex gap-2">
                <Button
                  disabled={!patientId || !selectedFile || loading}
                  onClick={() => selectedFile && handleProcess(selectedFile)}
                >
                  {loading ? "Processing..." : "Process"}
                </Button>
                <Button variant="ghost" onClick={goHome}>
                  Cancel
                </Button>
              </div>
              {errorMsg && <p className="text-sm text-red-600">{errorMsg}</p>}
            </CardContent>
          </Card>
        </div>
      );
    }

    if (mode === "old") {
      return (
        <div className="max-w-3xl w-full mx-auto px-4">
          <Card className="backdrop-blur bg-white/70 border-white/50 shadow-xl">
            <CardHeader>
              <CardTitle className="text-sky-900">Show old data</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Dropdown: Patient */}
              <div className="grid md:grid-cols-[1fr_auto] gap-2">
                <select
                  className="w-full border rounded px-3 py-2 bg-white"
                  value={selectedPatient}
                  onChange={(e) => onSelectPatient(e.target.value)}
                >
                  <option value="">Select patient</option>
                  {patients.map((p) => (
                    <option key={p.pacjent_uid} value={p.pacjent_uid}>
                      {p.pacjent_uid}
                    </option>
                  ))}
                </select>
              </div>

              {/* Dropdown: Slide (tylko dla wybranego pacjenta) */}
              <div className="grid md:grid-cols-[1fr_auto] gap-2">
                <select
                  className="w-full border rounded px-3 py-2 bg-white"
                  value={selectedSlide}
                  onChange={(e) => onSelectSlide(e.target.value)}
                  disabled={!selectedPatient || slides.length === 0}
                >
                  <option value="">
                    {selectedPatient ? (slides.length ? "Select slide" : "No slides") : "Select patient first"}
                  </option>
                  {slides.map((s) => (
                    <option key={s.slajd_uid} value={s.slajd_uid}>
                      {s.slajd_uid}
                      {s.overall_class ? ` • ${mapClass(s.overall_class)}` : ""}
                      {s.status ? ` • ${s.status}` : ""}
                    </option>
                  ))}
                </select>

                <Button
                  variant="outline"
                  disabled={!selectedSlide || loading}
                  onClick={handleShowSlide}
                >
                  Show
                </Button>
              </div>

              <div className="flex gap-2">
                <Button 
                onClick={goHome}>
                  Back
                </Button>

                <Button
                  // variant="ghost"
                  onClick={() => onSelectPatient(selectedPatient)} // odśwież slajdy
                  disabled={!selectedPatient}
                >
                  Refresh slides
                </Button>
              </div>
              {errorMsg && <p className="text-sm text-red-600">{errorMsg}</p>}
            </CardContent>
          </Card>
        </div>
      );
    }
    return null;
  };

  const ResultsArea = () =>
    results && !loading ? (
      <div className="w-full max-w-7xl grid grid-cols-1 md:grid-cols-3 gap-6 items-start px-4">
        {/* Obraz */}
        <div className="md:col-span-2 h-[640px] rounded-2xl border bg-white/70 p-2 flex items-center justify-center shadow">
          {imageUrl ? (
            <img
              src={imageUrl}
              alt="Detected cells with bounding boxes"
              className="w-full h-full object-contain rounded-xl"
            />
          ) : (
            <div className="text-gray-500">No image available</div>
          )}
        </div>

        {/* Prawy panel */}
        <div className="h-[640px] flex flex-col gap-4">
          <Card className="flex-1 overflow-auto backdrop-blur bg-white/70 border-white/50 shadow">
            <CardHeader>
              <CardTitle className="text-sky-900">Summary</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              <p>
                <strong>Slide UID:</strong>{" "}
                {results.slide_uid ?? selectedSlide ?? "—"}
              </p>
              <p>
                <strong>Patient ID:</strong>{" "}
                {results.pacjent_uid ?? selectedPatient ?? patientId ?? "—"}
              </p>
              <p>
                <strong>Overall class:</strong> {overallClass}
              </p>
              <p>
                <strong>Total cells:</strong> {totalCells}
              </p>

              {totalCells > 0 && (
                <div className="mt-2">
                  <p className="font-semibold mb-1">Cells per class:</p>
                  <table className="text-sm w-full border rounded">
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

          <Card className="flex-1 overflow-auto backdrop-blur bg-white/70 border-white/50 shadow">
            <CardHeader>
              <CardTitle className="text-sky-900">Predictions</CardTitle>
            </CardHeader>
            <CardContent>
              {results.predict_fused ? (
                <div className="max-h-[220px] overflow-auto">
                  <table className="text-sm w-full border rounded">
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
    ) : null;

  const CellsDetails = () =>
    results && showDetails ? (
      <div className="w-full max-w-7xl bg-white/70 rounded-2xl p-4 shadow mx-4">
        <h2 className="text-2xl font-semibold mb-4 text-sky-900">Cells — details</h2>

        {results.crop_public_urls ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(results.crop_public_urls).map(([id, url]) => {
              const rawCls = results.predict_fused?.[id] ?? "—";
              const cls = mapClass(rawCls);
              const probs = results.probs?.[id]?.fused ?? {};
              const features = results.features_list?.[id] ?? {};

              const probEntries = Object.entries(probs).sort(([a], [b]) =>
                a.localeCompare(b)
              );
              const featureEntries = Object.entries(features).sort(([a], [b]) =>
                a.localeCompare(b)
              );

              const limeBusy = limeLoadingId === id;
              const limeErr = limeErrorById[id];

              return (
                <Card key={id} className="overflow-hidden shadow">
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
                        disabled={limeBusy || (!results?.features_list?.[id] && !results?.slide_uid && !selectedSlide)}
                        title={
                          !results?.features_list?.[id] && !results?.slide_uid && !selectedSlide
                            ? "No features found and no slide UID"
                            : ""
                        }
                      >
                        {limeBusy ? "LIME…" : "LIME"}
                      </Button>
                      {limeErr && <span className="text-xs text-red-600">{limeErr}</span>}
                    </div>

                    {/* Probabilities */}
                    {probEntries.length > 0 && (
                      <div>
                        <p className="font-semibold mb-1">Probabilities</p>
                        <table className="text-sm w-full border rounded">
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
                                  {Number.isFinite(v as number) ? Number(v).toFixed(3) : "—"}
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
                                    {Number.isFinite(v as number) ? Number(v).toFixed(3) : "—"}
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
    ) : null;

  return (
    <div className="min-h-screen relative">
      {/* Tło */}
      <div className="absolute inset-0 bg-gradient-to-br from-sky-50 via-indigo-50 to-white" />
      <div className="absolute -top-20 -right-20 w-[40rem] h-[40rem] rounded-full bg-sky-200/20 blur-3xl" />
      <div className="absolute -bottom-20 -left-20 w-[36rem] h-[36rem] rounded-full bg-indigo-200/20 blur-3xl" />

      <div className="relative">
        <TopBar />

        {/* Home / Forms */}
        <div className="py-10 flex flex-col items-center gap-8">
          {mode === "home" ? <Home /> : <ModeForms />}

          {/* Wyniki + Detale (po obu trybach) */}
          <ResultsArea />
          <CellsDetails />
        </div>

        {/* Grad-CAM Dialog */}
        <Dialog open={gradcamOpen} onOpenChange={setGradcamOpen}>
          <DialogContent className="max-w-4xl">
            <DialogHeader>
              <DialogTitle>
                Grad-CAM {gradcamForId ? `for Cell #${gradcamForId}` : ""}
              </DialogTitle>
            </DialogHeader>

            {gradcamLoading && (
              <div className="py-6 text-center">
                <div className="animate-spin rounded-full h-12 w-12 border-4 border-gray-200 border-t-blue-600 mx-auto"></div>
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

        {/* Processing Dialog */}
        <Dialog open={loading} onOpenChange={() => { /* block manual close while loading */ }}>
          <DialogContent
            className="max-w-sm"
            onInteractOutside={(e) => e.preventDefault()}
            onEscapeKeyDown={(e) => e.preventDefault()}
          >
            <DialogHeader>
              <DialogTitle>Processing…</DialogTitle>
            </DialogHeader>

            <div className="py-4 text-center space-y-4">
              <div className="animate-spin rounded-full h-12 w-12 border-4 border-gray-200 border-t-blue-600 mx-auto" />
              <div className="w-full h-2 bg-gray-200 rounded overflow-hidden">
                <div
                  className="h-full bg-blue-600 transition-[width] duration-200"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="text-sm text-gray-700">
                I am analysing patient slide
                {patientId ? ` (${patientId})` : ""} — please wait…
              </p>
            </div>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}
