import { useEffect, useMemo, useRef, useState } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Activity, Brain, Eye } from "lucide-react";
import TopBar from "@/components/TopBar";
import Home from "@/components/home/Home";
import LoginView from "@/components/auth/LoginView";
import AddForm from "@/components/add/AddForm";
import OldForm from "@/components/old/OldForm";
import ResultsArea from "@/components/results/ResultsArea";
import CellsDetails from "@/components/results/CellsDetails";
import { api } from "@/lib/api";
import { mapClass } from "@/lib/constants";
import type { Mode, User, Results, Patient, SlideItem, GradcamResp } from "@/types";

export default function App() {
  const [mode, setMode] = useState<Mode>("home");
  const [user, setUser] = useState<User | null>(null);

  // auth
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [authErr, setAuthErr] = useState<string | null>(null);

  // shared UI states
  const [patientId, setPatientId] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const [results, setResults] = useState<Results | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [showDetails, setShowDetails] = useState(false);

  // gradcam
  const [gradcamOpen, setGradcamOpen] = useState(false);
  const [gradcamForId, setGradcamForId] = useState<string | null>(null);
  const [gradcamData, setGradcamData] = useState<GradcamResp | null>(null);
  const [gradcamLoading, setGradcamLoading] = useState(false);
  const [gradcamError, setGradcamError] = useState<string | null>(null);

  // lime
  const [limeLoadingId, setLimeLoadingId] = useState<string | null>(null);
  const [limeErrorById, setLimeErrorById] = useState<Record<string, string>>({});

  // progress
  const [progress, setProgress] = useState(0);
  const progressTimerRef = useRef<number | null>(null);

  // old-data
  const [patients, setPatients] = useState<Patient[]>([]);
  const [selectedPatient, setSelectedPatient] = useState<string>("");
  const [slides, setSlides] = useState<SlideItem[]>([]);
  const [selectedSlide, setSelectedSlide] = useState<string>("");

  // add info
  const [addInfoDraft, setAddInfoDraft] = useState<string>("");
  const [savingAddInfo, setSavingAddInfo] = useState(false);
  const [saveAddInfoMsg, setSaveAddInfoMsg] = useState<string | null>(null);

  const BASE_API_URL = "http://localhost:8000"
  // = import.meta.env.VITE_API_URL;

  // auth helpers
  const checkMe = async () => {
    try {
      const r = await fetch(`${BASE_API_URL}/auth/me`, { credentials: "include" });;
      if (r.ok) {
        setUser(await r.json());
        setMode("home");
      } else {
        setUser(null);
        setMode("login");
      }
    } catch {
      setUser(null);
      setMode("login");
    }
  };
  useEffect(() => { checkMe(); }, []);

  const doLogin = async () => {
    setAuthErr(null);
    const r = await fetch(`${BASE_API_URL}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ email, password: password }),
    });
    if (!r.ok) {
      const t = await r.text();
      setAuthErr(t || "Invalid credentials");
      return;
    }
    await checkMe();
  };

  const doLogout = async () => {
    await api(`${BASE_API_URL}/auth/logout`, { method: "POST" });
    setUser(null);
    setMode("login");
  };

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
    setSlides([]);
    setSelectedSlide("");
    setAddInfoDraft("");
    setSavingAddInfo(false);
    setSaveAddInfoMsg(null);
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

      const url = `${BASE_API_URL}/process-image/?patient_uid=${encodeURIComponent(patientId || "UNKNOWN")}`;

      const response = await fetch(url, { method: "POST", body: formData, credentials: "include" });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Request failed with ${response.status}`);
      }

      const data: Results = await response.json();
      console.log("Processing results:", data);
      setResults(data);
      setImageUrl(data.bbox_public_url || null);
      setShowDetails(false);
      setAddInfoDraft(data.add_info ?? "");
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
  useEffect(() => {
    if (mode !== "old") return;
    (async () => {
      try {
        setErrorMsg(null);
        setPatients([]);
        setSlides([]);
        setSelectedPatient("");
        setSelectedSlide("");
        const resp = await api(`${BASE_API_URL}/patients`);
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

  const onRefreshSlides = async (uid: string) => {
    if (!uid) return;
    try {
      setErrorMsg(null);
      const resp = await api(`${BASE_API_URL}/patient/${encodeURIComponent(uid)}/slides`);
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

  const handleShowSlide = async () => {
    if (!selectedSlide.trim()) return;
    try {
      setErrorMsg(null);
      setLoading(true);
      setResults(null);
      setImageUrl(null);
      setShowDetails(false);
      setAddInfoDraft("");

      // Przykład użycia w Twoim pliku API
      // const BASE_API_URL = import.meta.env.VITE_API_URL;

      // Zrób zapytanie, używając zmiennej środowiskowej
      const resp = await api(`${BASE_API_URL}/slide/${encodeURIComponent(selectedSlide.trim())}`);

      // const resp = await api(`http://localhost:8000/slide/${encodeURIComponent(selectedSlide.trim())}`);
      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(txt || `Failed to load slide (${resp.status})`);
      }
      const data: Results = await resp.json();
      console.log("Loaded slide results:", data);
      setResults(data);
      setImageUrl(data.bbox_public_url || null);
      setAddInfoDraft(data.add_info ?? "");
    } catch (e: any) {
      setErrorMsg(e?.message || "Failed to load slide");
    } finally {
      setLoading(false);
    }
  };

  const saveAddInfo = async () => {
    const sid = results?.slide_uid ?? selectedSlide;
    if (!sid) return;
    try {
      setSaveAddInfoMsg(null);
      setSavingAddInfo(true);
      const prev = results?.add_info ?? "";
      setResults((p) => (p ? { ...p, add_info: addInfoDraft } : p));
      const r = await fetch(`${BASE_API_URL}/slide/${encodeURIComponent(sid)}/add-info`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ add_info: addInfoDraft }),
      });
      if (!r.ok) {
        setResults((p) => (p ? { ...p, add_info: prev } : p));
        const t = await r.text();
        throw new Error(t || `Save failed (${r.status})`);
      }
      setSaveAddInfoMsg("Saved successfully!");
      setTimeout(() => setSaveAddInfoMsg(null), 2000);
    } catch (e: any) {
      setSaveAddInfoMsg(e?.message || "Save failed");
    } finally {
      setSavingAddInfo(false);
    }
  };

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

  const overallClass =
    results?.overall_class != null
      ? mapClass(results.overall_class)
      : results?.slide_summary?.overall_class != null
      ? mapClass(results.slide_summary.overall_class)
      : "—";

  const openGradcam = async (cellId: string) => {
    const cropUrl = results?.crop_public_urls?.[cellId];
    const cropGridName = results?.crop_gridfs_names?.[cellId];
    if (!cropUrl && !cropGridName) return;

    const cellUid = results?.slide_uid ? `${results.slide_uid}:${cellId}` : selectedSlide ? `${selectedSlide}:${cellId}` : null;

    setGradcamOpen(true);
    setGradcamForId(cellId);
    setGradcamData(null);
    setGradcamError(null);
    setGradcamLoading(true);

    try {
      const body: Record<string, any> = cropGridName ? { crop_gridfs_name: cropGridName } : { image_url: cropUrl };
      if (cellUid) body.cell_uid = cellUid;
      const resp = await fetch(`${BASE_API_URL}/gradcam/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify(body),
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
    const cellUid = results?.slide_uid ? `${results.slide_uid}:${cellId}` : selectedSlide ? `${selectedSlide}:${cellId}` : null; 

  const haveFeatures = !!results?.features_list?.[cellId];
  const payload: Record<string, any> = {}; 
  if (haveFeatures) payload.features = results!.features_list![cellId];
  
  if (cellUid) payload.cell_uid = cellUid; 
    
  if (!haveFeatures && !cellUid) {
   setLimeErrorById((p) => ({ ...p, [cellId]: "No features or cell_uid available" })); 
   return;
  }
    setLimeLoadingId(cellId);
    setLimeErrorById((p) => { const q = { ...p }; delete q[cellId]; return q; });

    try {
      const resp = await fetch(`${BASE_API_URL}/lime/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify(payload),
      });
      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(txt || `LIME failed with ${resp.status}`);
      }
      const data: { html_url: string } = await resp.json();
      window.open(data.html_url, "_blank", "noopener,noreferrer");
    } catch (e: any) {
      setLimeErrorById((p) => ({ ...p, [cellId]: e?.message || "LIME error." }));
      console.error("LIME error:", e);
    } finally {
      setLimeLoadingId(null);
    }
  };

  const correctCellClass = async (cellId: string, newClass: "HSIL" | "LSIL" | "NSIL") => {
    const komorkaUid = results?.slide_uid ? `${results.slide_uid}:${cellId}` : selectedSlide ? `${selectedSlide}:${cellId}` : null;
    if (!komorkaUid) { console.error("Brak slide_uid do korekcji klasy"); return; }
    const res = await fetch(`${BASE_API_URL}/cell/${encodeURIComponent(komorkaUid)}/correct-class`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ class_corrected: newClass }),
    });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(txt || `Korekcja nie powiodła się (${res.status})`);
    }
  };

  return (
    <div className="min-h-screen relative overflow-x-hidden font-sans" style={{ fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif" }}>
      <div className="fixed inset-0 bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50" />
      <div className="fixed -top-40 -right-40 w-96 h-96 rounded-full bg-gradient-to-br from-blue-200/40 to-indigo-300/40 blur-3xl animate-pulse" />
      <div className="fixed -bottom-40 -left-40 w-96 h-96 rounded-full bg-gradient-to-br from-indigo-200/40 to-purple-300/40 blur-3xl animate-pulse" style={{ animationDelay: '2s' }} />
      <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-gradient-to-br from-blue-100/20 to-indigo-100/20 blur-3xl" />

      <div className="relative z-10">
        <TopBar user={user} goHome={goHome} doLogout={doLogout} />
        <div className="py-8 flex flex-col items-center gap-8">
          {!user ? (
            <LoginView email={email} setEmail={setEmail} password={password} setPassword={setPassword} doLogin={doLogin} authErr={authErr} />
          ) : (
            <>
              {mode === "home" && (<Home onGoAdd={() => { resetState(); setMode("add"); }} onGoOld={() => { resetState(); setMode("old"); }} />)}
              {mode === "add" && (<AddForm patientId={patientId} setPatientId={setPatientId} selectedFile={selectedFile} setSelectedFile={setSelectedFile} loading={loading} errorMsg={errorMsg} onProcess={handleProcess} onCancel={goHome} />)}
              {mode === "old" && (<OldForm patients={patients} selectedPatient={selectedPatient} setSelectedPatient={(v) => { setSelectedPatient(v); }} slides={slides} selectedSlide={selectedSlide} setSelectedSlide={setSelectedSlide} loading={loading} errorMsg={errorMsg} onRefreshSlides={onRefreshSlides} onShowSlide={handleShowSlide} onBack={goHome} />)}

              <ResultsArea results={results} imageUrl={imageUrl} overallClass={overallClass} totalCells={totalCells} classCounts={classCounts} addInfoDraft={addInfoDraft} setAddInfoDraft={setAddInfoDraft} savingAddInfo={savingAddInfo} saveAddInfoMsg={saveAddInfoMsg} onSaveAddInfo={saveAddInfo} loading={loading} showDetails={showDetails} setShowDetails={setShowDetails} />

              <CellsDetails results={results} showDetails={showDetails} limeLoadingId={limeLoadingId} limeErrorById={limeErrorById} openGradcam={openGradcam} openLime={openLime} onCorrectClass={correctCellClass} />
            </>
          )}
        </div>

        <Dialog open={gradcamOpen} onOpenChange={setGradcamOpen}>
          <DialogContent className="max-w-6xl">
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2 text-xl">
                <Eye className="w-6 h-6 text-blue-600" />
                Grad-CAM Analysis {gradcamForId && `• Cell #${gradcamForId}`}
              </DialogTitle>
            </DialogHeader>

            {gradcamLoading && (
              <div className="py-12 text-center">
                <div className="relative mx-auto w-16 h-16 mb-6">
                  <div className="absolute inset-0 border-4 border-blue-200 rounded-full" />
                  <div className="absolute inset-0 border-4 border-blue-600 border-t-transparent rounded-full animate-spin" />
                </div>
                <p className="text-lg text-blue-700 font-medium">Generating Grad-CAM visualization...</p>
                <p className="text-sm text-gray-600 mt-2">This may take a few moments</p>
              </div>
            )}

            {gradcamError && (
              <div className="text-center py-8">
                <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                  <Activity className="w-8 h-8 text-red-600" />
                </div>
                <p className="text-red-600 font-medium">{gradcamError}</p>
              </div>
            )}

            {gradcamData && (
              <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                  <div className="overflow-hidden border rounded">
                    <div className="p-3 text-sm font-medium text-gray-600">Original</div>
                    <img src={gradcamData.overlay_url} className="w-full h-auto" alt="Overlay" />
                  </div>
                  <div className="overflow-hidden border rounded">
                    <div className="p-3 text-sm font-medium text-gray-600">Grad-CAM</div>
                    <img src={gradcamData.heatmap_url} className="w-full h-auto" alt="Heatmap" />
                  </div>
                  <div className="overflow-hidden border rounded">
                    <div className="p-3 text-sm font-medium text-gray-600">Activation Map</div>
                    <img src={gradcamData.activation_url} className="w-full h-auto" alt="Activation" />
                  </div>
                </div>
                <div className="flex items-center justify-center p-4 bg-blue-50 rounded-lg">
                  <p className="text-gray-700">Model prediction for this visualization: <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-md ml-2 bg-blue-100 text-blue-800 border border-blue-200">{gradcamData.predicted_class}</span></p>
                </div>
              </div>
            )}
          </DialogContent>
        </Dialog>

        <Dialog open={loading} onOpenChange={() => {}}>
          <DialogContent className="max-w-md" onInteractOutside={(e) => e.preventDefault()} onEscapeKeyDown={(e) => e.preventDefault()}>
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2 text-xl">
                <Brain className="w-6 h-6 text-blue-600" />
                AI Analysis in Progress
              </DialogTitle>
            </DialogHeader>
            <div className="py-6 text-center space-y-6">
              <div className="relative mx-auto w-16 h-16">
                <div className="absolute inset-0 border-4 border-blue-200 rounded-full" />
                <div className="absolute inset-0 border-4 border-blue-600 border-t-transparent rounded-full animate-spin" />
              </div>
              <div className="space-y-3">
                <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden">
                  <div className="h-full bg-gradient-to-r from-blue-500 to-indigo-600 transition-all duration-300 ease-out" style={{ width: `${progress}%` }} />
                </div>
                <p className="text-sm font-medium text-gray-700">{progress}% Complete</p>
              </div>
              <div className="space-y-2">
                <p className="text-lg font-medium text-gray-800">Analyzing slide{patientId && ` for patient ${patientId}`}</p>
                <p className="text-sm text-gray-600">Our AI is examining the cellular structures and classifying each cell. Please wait...</p>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}