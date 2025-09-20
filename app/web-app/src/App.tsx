import { useEffect, useMemo, useRef, useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import FileUpload from "./components/FileUpload";

/* ====================== TYPES ====================== */
type Results = {
  slide_uid?: string;
  pacjent_uid?: string;
  bbox_public_url?: string;
  crop_public_urls?: Record<string, string>;
  crop_gridfs_names?: Record<string, string>;
  predict_fused?: Record<string, string | number>;
  probs?: Record<string, { fused?: Record<string, number> }>;
  features_list?: Record<string, Record<string, number>>;
  slide_summary?: { overall_class?: string | number; explanation?: string; confidence?: number };
  slide_summary_text?: string;
  overall_class?: string | number;
  add_info?: string | null;
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
  add_info?: string | null;
};

type User = { email: string; imie?: string; nazwisko?: string; rola?: string };
type Mode = "home" | "add" | "old" | "login";

/* ====================== HELPERS ====================== */
const CLASS_NAME_MAP: Record<string, string> = {
  "0": "HSIL",
  "1": "LSIL",
  "2": "NSIL",
  HSIL: "HSIL",
  LSIL: "LSIL",
  NSIL: "NSIL",
};
const mapClass = (v: unknown) => CLASS_NAME_MAP[String(v)] ?? String(v);
const api = (url: string, init?: RequestInit) => fetch(url, { ...init, credentials: "include" });

/* ====================== EXTRACTED COMPONENTS ====================== */

// --- TOP BAR ---
function TopBar({ user, goHome, doLogout }: { user: User | null; goHome: () => void; doLogout: () => void }) {
  return (
    <div className="w-full sticky top-0 z-40 backdrop-blur bg-white/50 border-b border-white/20">
      <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
        <button
          onClick={goHome}
          className="text-sm px-3 py-1.5 rounded-full bg-white/60 hover:bg-white shadow transition"
          title="Back to Home"
        >
          Home
        </button>
        <h1 className="text-xl md:text-2xl font-bold text-blue-900 tracking-tight">LBC Slides Analysis</h1>
        <div className="flex items-center gap-3">
          {user ? (
            <>
              <span className="text-sm text-gray-700">
                {user.imie} {user.nazwisko} ({user.email})
              </span>
              <button
                onClick={doLogout}
                className="text-sm px-3 py-1.5 rounded-full bg-white/60 hover:bg-white shadow transition"
              >
                Logout
              </button>
            </>
          ) : (
            <div className="opacity-0">.</div>
          )}
        </div>
      </div>
    </div>
  );
}

// --- LOGIN VIEW ---
function LoginView({
  email,
  setEmail,
  password,
  setPassword,
  doLogin,
  authErr,
}: {
  email: string;
  setEmail: (v: string) => void;
  password: string;
  setPassword: (v: string) => void;
  doLogin: () => void;
  authErr: string | null;
}) {
  return (
    <div className="max-w-md w-full mx-auto px-4">
      <Card className="backdrop-blur bg-white/70 border-white/50 shadow-xl">
        <CardHeader>
          <CardTitle className="text-sky-900">Log in</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />
          <Input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          <Button onClick={doLogin} className="w-full">
            Login
          </Button>
          {authErr && <p className="text-sm text-red-600">{authErr}</p>}
        </CardContent>
      </Card>
    </div>
  );
}

// --- HOME ---
function Home({ onGoAdd, onGoOld }: { onGoAdd: () => void; onGoOld: () => void }) {
  return (
    <div className="max-w-4xl w-full mx-auto px-4">
      <div className="text-center mb-10">
        <h2 className="text-4xl md:text-5xl font-extrabold text-blue-900 drop-shadow-sm">Welcome</h2>
        <p className="mt-3 text-gray-600">Choose what you want to do – add a new slide or view saved data.</p>
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <button
          onClick={onGoOld}
          className="group rounded-2xl p-6 bg-white/70 hover:bg-white shadow-lg hover:shadow-xl border border-white/40 transition text-left"
        >
          <div className="text-3xl mb-3">📚</div>
          <div className="text-2xl font-semibold text-blue-900 mb-1">Show old data</div>
          <p className="text-gray-600">First, select the patient, then the slide. View results (Grad-CAM, LIME).</p>
          <div className="mt-4 text-blue-700 font-medium group-hover:translate-x-1 transition">Open →</div>
        </button>

        <button
          onClick={onGoAdd}
          className="group rounded-2xl p-6 bg-white/70 hover:bg-white shadow-lg hover:shadow-xl border border-white/40 transition text-left"
        >
          <div className="text-3xl mb-3">➕</div>
          <div className="text-2xl font-semibold text-blue-900 mb-1">Add new slide</div>
          <p className="text-gray-600">
            Upload image, provide <span className="font-medium">Patient ID</span> and run the analysis.
          </p>
          <div className="mt-4 text-blue-700 font-medium group-hover:translate-x-1 transition">Add →</div>
        </button>
      </div>
    </div>
  );
}

// --- ADD FORM ---
function AddForm({
  patientId,
  setPatientId,
  selectedFile,
  setSelectedFile,
  loading,
  errorMsg,
  onProcess,
  onCancel,
}: {
  patientId: string;
  setPatientId: (v: string) => void;
  selectedFile: File | null;
  setSelectedFile: (f: File | null) => void;
  loading: boolean;
  errorMsg: string | null;
  onProcess: (f: File) => void;
  onCancel: () => void;
}) {
  return (
    <div className="max-w-3xl w-full mx-auto px-4">
      <Card className="backdrop-blur bg-white/70 border-white/50 shadow-xl">
        <CardHeader>
          <CardTitle className="text-sky-900">Add new slide</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <Input placeholder="Patient ID" value={patientId} onChange={(e) => setPatientId(e.target.value)} />
          <FileUpload selectedFile={selectedFile} onFileSelect={(file) => setSelectedFile(file)} />
          <div className="flex gap-2">
            <Button disabled={!patientId || !selectedFile || loading} onClick={() => selectedFile && onProcess(selectedFile)}>
              {loading ? "Processing..." : "Process"}
            </Button>
            <Button variant="ghost" onClick={onCancel}>Cancel</Button>
          </div>
          {errorMsg && <p className="text-sm text-red-600">{errorMsg}</p>}
        </CardContent>
      </Card>
    </div>
  );
}

// --- OLD FORM (bez tabeli pod selectem) ---
function OldForm({
  patients,
  selectedPatient,
  setSelectedPatient,
  slides,
  selectedSlide,
  setSelectedSlide,
  loading,
  errorMsg,
  onRefreshSlides,
  onShowSlide,
  onBack,
}: {
  patients: Patient[];
  selectedPatient: string;
  setSelectedPatient: (v: string) => void;
  slides: SlideItem[];
  selectedSlide: string;
  setSelectedSlide: (v: string) => void;
  loading: boolean;
  errorMsg: string | null;
  onRefreshSlides: (uid: string) => void;
  onShowSlide: () => void;
  onBack: () => void;
}) {
  return (
    <div className="max-w-3xl w-full mx-auto px-4">
      <Card className="backdrop-blur bg-white/70 border-white/50 shadow-xl">
        <CardHeader>
          <CardTitle className="text-sky-900">Show old data</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Patient */}
          <select
            className="w-full border rounded px-3 py-2 bg-white"
            value={selectedPatient}
            onChange={(e) => {
              const v = e.target.value;
              setSelectedPatient(v);
              onRefreshSlides(v);
            }}
          >
            <option value="">Select patient</option>
            {patients.map((p) => (
              <option key={p.pacjent_uid} value={p.pacjent_uid}>{p.pacjent_uid}</option>
            ))}
          </select>

          {/* Slide (no table) */}
          <div className="grid md:grid-cols-[1fr_auto] gap-2">
            <select
              className="w-full border rounded px-3 py-2 bg-white"
              value={selectedSlide}
              onChange={(e) => setSelectedSlide(e.target.value)}
              disabled={!selectedPatient || slides.length === 0}
            >
              <option value="">
                {selectedPatient ? (slides.length ? "Select slide" : "No slides") : "Select patient first"}
              </option>
              {slides.map((s) => (
                <option key={s.slajd_uid} value={s.slajd_uid}>
                  {s.slajd_uid}{s.overall_class ? ` • ${mapClass(s.overall_class)}` : ""}{s.status ? ` • ${s.status}` : ""}
                </option>
              ))}
            </select>

            <Button variant="outline" disabled={!selectedSlide || loading} onClick={onShowSlide}>Show</Button>
          </div>

          <div className="flex gap-2">
            <Button onClick={onBack}>Back</Button>
            <Button onClick={() => onRefreshSlides(selectedPatient)} disabled={!selectedPatient}>Refresh slides</Button>
          </div>

          {errorMsg && <p className="text-sm text-red-600">{errorMsg}</p>}
        </CardContent>
      </Card>
    </div>
  );
}

// --- ADD INFO CARD (krótka) ---
function AddInfoCard({
  value,
  onChange,
  onSave,
  saving,
  message,
}: {
  value: string;
  onChange: (v: string) => void;
  onSave: () => void;
  saving: boolean;
  message: string | null;
}) {
  return (
    <Card className="overflow-hidden backdrop-blur bg-white/70 border-white/50 shadow">
      <CardHeader className="pb-2">
        <CardTitle className="text-sky-900">Add info</CardTitle>
      </CardHeader>
      <CardContent>
        <textarea
          rows={2}
          className="w-full border rounded p-2 bg-white resize-y min-h-[40px] max-h-[90px]"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Short note for this slide…"
        />
        <div className="mt-2 flex items-center gap-2">
          <Button size="sm" onClick={onSave} disabled={saving}>
            {saving ? "Saving…" : "Save"}
          </Button>
          {message && <span className="text-sm text-gray-700">{message}</span>}
        </div>
      </CardContent>
    </Card>
  );
}

// --- RESULTS AREA (z AddInfo pod Predictions) ---
function ResultsArea({
  results,
  imageUrl,
  overallClass,
  totalCells,
  classCounts,
  addInfoDraft,
  setAddInfoDraft,
  savingAddInfo,
  saveAddInfoMsg,
  onSaveAddInfo,
  loading,
  showDetails,
  setShowDetails,
}: {
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
}) {
  if (!results || loading) return null;

  return (
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
        {/* Summary */}
        <Card className="flex-1 overflow-auto backdrop-blur bg-white/70 border-white/50 shadow">
          <CardHeader>
            <CardTitle className="text-sky-900">Summary</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <p><strong>Slide UID:</strong> {results.slide_uid ?? "—"}</p>
            <p><strong>Patient ID:</strong> {results.pacjent_uid ?? "—"}</p>
            <p><strong>Overall class:</strong> {overallClass}</p>
            <p><strong>Total cells:</strong> {totalCells}</p>

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
              {results.slide_summary_text ?? results.slide_summary?.explanation ?? ""}
            </p>
          </CardContent>
        </Card>

        {/* Predictions */}
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

        {/* Add info – krótka karta pod Predictions */}
        <AddInfoCard
          value={addInfoDraft}
          onChange={setAddInfoDraft}
          onSave={onSaveAddInfo}
          saving={savingAddInfo}
          message={saveAddInfoMsg}
        />

        <Button variant="secondary" disabled={!results || loading} onClick={() => setShowDetails(!showDetails)}>
          {showDetails ? "Hide details" : "Details"}
        </Button>
      </div>
    </div>
  );
}

// --- CELLS DETAILS ---
function CellsDetails({
  results,
  showDetails,
  limeLoadingId,
  limeErrorById,
  openGradcam,
  openLime,
}: {
  results: Results | null;
  showDetails: boolean;
  limeLoadingId: string | null;
  limeErrorById: Record<string, string>;
  openGradcam: (id: string) => void;
  openLime: (id: string) => Promise<void>;
}) {
  if (!results || !showDetails) return null;

  return (
    <div className="w-full max-w-7xl bg-white/70 rounded-2xl p-4 shadow mx-4">
      <h2 className="text-2xl font-semibold mb-4 text-sky-900">Cells — details</h2>

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
              <Card key={id} className="overflow-hidden shadow">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center justify-between">
                    <span>Cell #{id}</span>
                    <span className="text-sm px-2 py-1 rounded bg-blue-50 text-blue-800">{cls}</span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="w-full h-48 border rounded flex items-center justify-center bg-gray-50">
                    <img src={url} alt={`cell-${id}`} className="max-h-40 object-contain" />
                  </div>

                  <div className="flex flex-wrap gap-2 items-center">
                    <Button variant="outline" size="sm" onClick={() => openGradcam(id)}>Grad-CAM</Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => openLime(id)}
                      disabled={
                        limeBusy || (!results?.features_list?.[id] && !results?.slide_uid)
                      }
                      title={
                        !results?.features_list?.[id] && !results?.slide_uid
                          ? "No features found and no slide UID"
                          : ""
                      }
                    >
                      {limeBusy ? "LIME…" : "LIME"}
                    </Button>
                    {limeErr && <span className="text-xs text-red-600">{limeErr}</span>}
                  </div>

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
  );
}

/* ====================== MAIN APP ====================== */
export default function App() {
  const [mode, setMode] = useState<Mode>("login");
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

  // auth helpers
  const checkMe = async () => {
    try {
      const r = await api("http://localhost:8000/auth/me");
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
    const r = await fetch("http://localhost:8000/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ email, haslo: password }),
    });
    if (!r.ok) {
      const t = await r.text();
      setAuthErr(t || "Invalid credentials");
      return;
    }
    await checkMe();
  };

  const doLogout = async () => {
    await api("http://localhost:8000/auth/logout", { method: "POST" });
    setUser(null);
    setMode("login");
  };

  // helpers
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

      const url = `http://localhost:8000/process-image/?pacjent_id=${encodeURIComponent(
        patientId || "UNKNOWN"
      )}`;

      const response = await fetch(url, { method: "POST", body: formData, credentials: "include" });
      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Request failed with ${response.status}`);
      }

      const data: Results = await response.json();
      setResults(data);
      setImageUrl(data.bbox_public_url || null);
      setShowDetails(true);
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
        const resp = await api("http://localhost:8000/patients");
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
      const resp = await api(`http://localhost:8000/patient/${encodeURIComponent(uid)}/slides`);
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
      setShowDetails(true);
      setAddInfoDraft("");

      const resp = await api(`http://localhost:8000/slide/${encodeURIComponent(selectedSlide.trim())}`);
      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(txt || `Failed to load slide (${resp.status})`);
      }
      const data: Results = await resp.json();
      setResults(data);
      setImageUrl(data.bbox_public_url || null);
      setAddInfoDraft(data.add_info ?? "");
    } catch (e: any) {
      setErrorMsg(e?.message || "Failed to load slide");
    } finally {
      setLoading(false);
    }
  };

  // save add_info – optymistycznie, bez odświeżania
  const saveAddInfo = async () => {
    const sid = results?.slide_uid ?? selectedSlide;
    if (!sid) return;
    try {
      setSaveAddInfoMsg(null);
      setSavingAddInfo(true);
      const prev = results?.add_info ?? "";

      // optimistic
      setResults((p) => (p ? { ...p, add_info: addInfoDraft } : p));

      const r = await fetch(`http://localhost:8000/slide/${encodeURIComponent(sid)}/add-info`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        credentials: "include",
        body: JSON.stringify({ add_info: addInfoDraft }),
      });
      if (!r.ok) {
        // rollback
        setResults((p) => (p ? { ...p, add_info: prev } : p));
        const t = await r.text();
        throw new Error(t || `Save failed (${r.status})`);
      }
      setSaveAddInfoMsg("Saved.");
      setTimeout(() => setSaveAddInfoMsg(null), 1500);
    } catch (e: any) {
      setSaveAddInfoMsg(e?.message || "Save failed");
    } finally {
      setSavingAddInfo(false);
    }
  };

  // stats & helpers for ResultsArea
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

  // Grad-CAM handler
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

  // LIME handler
  const openLime = async (cellId: string) => {
    const komorkaUid =
      results?.slide_uid ? `${results.slide_uid}:${cellId}` :
      selectedSlide ? `${selectedSlide}:${cellId}` : null;

    const haveFeatures = !!results?.features_list?.[cellId];
    const payload: any = {};
    if (haveFeatures) payload.features = results!.features_list![cellId];
    else if (komorkaUid) payload.komorka_uid = komorkaUid;
    else {
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

  return (
    <div className="min-h-screen relative">
      {/* Tło */}
      <div className="absolute inset-0 bg-gradient-to-br from-sky-50 via-indigo-50 to-white" />
      <div className="absolute -top-20 -right-20 w-[40rem] h-[40rem] rounded-full bg-sky-200/20 blur-3xl" />
      <div className="absolute -bottom-20 -left-20 w-[36rem] h-[36rem] rounded-full bg-indigo-200/20 blur-3xl" />

      <div className="relative">
        <TopBar user={user} goHome={goHome} doLogout={doLogout} />

        <div className="py-10 flex flex-col items-center gap-8">
          {!user ? (
            <LoginView
              email={email}
              setEmail={setEmail}
              password={password}
              setPassword={setPassword}
              doLogin={doLogin}
              authErr={authErr}
            />
          ) : (
            <>
              {mode === "home" && (
                <Home
                  onGoAdd={() => { resetState(); setMode("add"); }}
                  onGoOld={() => { resetState(); setMode("old"); }}
                />
              )}

              {mode === "add" && (
                <AddForm
                  patientId={patientId}
                  setPatientId={setPatientId}
                  selectedFile={selectedFile}
                  setSelectedFile={setSelectedFile}
                  loading={loading}
                  errorMsg={errorMsg}
                  onProcess={handleProcess}
                  onCancel={goHome}
                />
              )}

              {mode === "old" && (
                <OldForm
                  patients={patients}
                  selectedPatient={selectedPatient}
                  setSelectedPatient={(v) => { setSelectedPatient(v); }}
                  slides={slides}
                  selectedSlide={selectedSlide}
                  setSelectedSlide={setSelectedSlide}
                  loading={loading}
                  errorMsg={errorMsg}
                  onRefreshSlides={onRefreshSlides}
                  onShowSlide={handleShowSlide}
                  onBack={goHome}
                />
              )}

              {/* Wyniki + detale */}
              <ResultsArea
                results={results}
                imageUrl={imageUrl}
                overallClass={overallClass}
                totalCells={totalCells}
                classCounts={classCounts}
                addInfoDraft={addInfoDraft}
                setAddInfoDraft={setAddInfoDraft}
                savingAddInfo={savingAddInfo}
                saveAddInfoMsg={saveAddInfoMsg}
                onSaveAddInfo={saveAddInfo}
                loading={loading}
                showDetails={showDetails}
                setShowDetails={setShowDetails}
              />
              <CellsDetails
                results={results}
                showDetails={showDetails}
                limeLoadingId={limeLoadingId}
                limeErrorById={limeErrorById}
                openGradcam={openGradcam}
                openLime={openLime}
              />
            </>
          )}
        </div>

        {/* Grad-CAM Dialog */}
        <Dialog open={gradcamOpen} onOpenChange={setGradcamOpen}>
          <DialogContent className="max-w-4xl">
            <DialogHeader>
              <DialogTitle>Grad-CAM {gradcamForId ? `for Cell #${gradcamForId}` : ""}</DialogTitle>
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
        <Dialog open={loading} onOpenChange={() => {}}>
          <DialogContent
            className="max-w-sm"
            onInteractOutside={(e) => e.preventDefault()}
            onEscapeKeyDown={(e) => e.preventDefault()}
          >
            <DialogHeader><DialogTitle>Processing…</DialogTitle></DialogHeader>
            <div className="py-4 text-center space-y-4">
              <div className="animate-spin rounded-full h-12 w-12 border-4 border-gray-200 border-t-blue-600 mx-auto" />
              <div className="w-full h-2 bg-gray-200 rounded overflow-hidden">
                <div className="h-full bg-blue-600 transition-[width] duration-200" style={{ width: `${progress}%` }} />
              </div>
              <p className="text-sm text-gray-700">
                I am analysing patient slide {patientId ? `(${patientId})` : ""} — please wait…
              </p>
            </div>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}
