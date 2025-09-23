import { useEffect, useMemo, useRef, useState } from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

import { User, Home as HomeIcon, LogOut, Upload, Eye, Brain, Zap, Activity } from "lucide-react";

// Placeholder for FileUpload component since it's imported
function FileUpload({ selectedFile, onFileSelect }: { selectedFile: File | null; onFileSelect: (file: File | null) => void }) {
  return (
    <div className="border-2 border-dashed border-blue-200 rounded-xl p-6 text-center hover:border-blue-300 transition-colors">
      <input
        type="file"
        accept="image/*"
        className="hidden"
        id="file-upload"
        onChange={(e) => onFileSelect(e.target.files?.[0] || null)}
      />
      <label htmlFor="file-upload" className="cursor-pointer">
        <Upload className="w-8 h-8 mx-auto mb-2 text-blue-500" />
        <p className="text-sm text-gray-600">
          {selectedFile ? selectedFile.name : "Click to upload slide image"}
        </p>
      </label>
    </div>
  );
}

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

const CLASS_COLORS: Record<string, string> = {
  HSIL: "bg-red-100 text-red-800 border-red-200",
  LSIL: "bg-yellow-100 text-yellow-800 border-yellow-200",
  NSIL: "bg-green-100 text-green-800 border-green-200",
};

const mapClass = (v: unknown) => CLASS_NAME_MAP[String(v)] ?? String(v);
const api = (url: string, init?: RequestInit) => fetch(url, { ...init, credentials: "include" });

/* ====================== EXTRACTED COMPONENTS ====================== */

// --- TOP BAR ---
function TopBar({ user, goHome, doLogout }: { user: User | null; goHome: () => void; doLogout: () => void }) {
  return (
    <div className="w-full sticky top-0 z-40 backdrop-blur-lg bg-white/80 border-b border-blue-100 shadow-sm">
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <Button
          onClick={goHome}
          variant="ghost"
          size="sm"
          className="flex items-center gap-2 hover:bg-blue-50"
        >
          <HomeIcon className="w-4 h-4" />
          Home
        </Button>
        <div className="text-center">
          <h1 className="text-2xl md:text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
            LBC Slides Analysis
          </h1>
          <p className="text-sm text-gray-500 hidden md:block">Advanced Cervical Cell Classification</p>
        </div>
        <div className="flex items-center gap-3">
          {user ? (
            <>
              <div className="hidden md:flex items-center gap-2 px-3 py-2 bg-blue-50 rounded-full">
                <User className="w-4 h-4 text-blue-600" />
                <span className="text-sm text-gray-700">
                  {user.imie} {user.nazwisko}
                </span>
              </div>
              <Button
                onClick={doLogout}
                variant="ghost"
                size="sm"
                className="flex items-center gap-2 hover:bg-red-50 hover:text-red-600"
              >
                <LogOut className="w-4 h-4" />
                Logout
              </Button>
            </>
          ) : (
            <div className="w-20"></div>
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
      <Card className="backdrop-blur-lg bg-white/90 border-blue-100 shadow-xl">
        <CardHeader className="text-center pb-4">
          <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full flex items-center justify-center mx-auto mb-4">
            <Activity className="w-8 h-8 text-white" />
          </div>
          <CardTitle className="text-2xl text-gray-800">Welcome Back</CardTitle>
          <p className="text-gray-600">Sign in to continue to LBC Analysis</p>
        </CardHeader>
        <CardContent className="space-y-4">
          <Input
            type="email"
            placeholder="Email address"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="h-12"
          />
          <Input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="h-12"
          />
          <Button onClick={doLogin} className="w-full h-12 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700">
            Sign In
          </Button>
          {authErr && <p className="text-sm text-red-600 text-center">{authErr}</p>}
        </CardContent>
      </Card>
    </div>
  );
}

// --- HOME ---
function Home({ onGoAdd, onGoOld }: { onGoAdd: () => void; onGoOld: () => void }) {
  return (
    <div className="max-w-5xl w-full mx-auto px-4">
      <div className="text-center mb-12">
        <h2 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent mb-4">
          Welcome
        </h2>
        <p className="text-xl text-gray-600 max-w-2xl mx-auto">
          Advanced AI-powered cervical cell analysis and classification system
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-8">
        <button
          onClick={onGoOld}
          className="group rounded-3xl p-8 bg-gradient-to-br from-white/90 to-blue-50/50 hover:from-white hover:to-blue-50 shadow-lg hover:shadow-xl border border-blue-100 transition-all duration-300 text-left transform hover:-translate-y-1"
        >
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center">
              <Eye className="w-8 h-8 text-white" />
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-800 mb-1">View Results</div>
              <span className="inline-flex px-2 py-1 text-xs font-medium rounded-md bg-blue-100 text-blue-800 border border-blue-200">Historical Data</span>
            </div>
          </div>
          <p className="text-gray-600 mb-6">
            Browse previously analyzed slides. Select patients and view detailed results including 
            Grad-CAM visualizations and LIME explanations.
          </p>
          <div className="flex items-center text-blue-700 font-semibold group-hover:translate-x-2 transition-transform">
            Explore Archive <span className="ml-2">→</span>
          </div>
        </button>

        <button
          onClick={onGoAdd}
          className="group rounded-3xl p-8 bg-gradient-to-br from-white/90 to-green-50/50 hover:from-white hover:to-green-50 shadow-lg hover:shadow-xl border border-green-100 transition-all duration-300 text-left transform hover:-translate-y-1"
        >
          <div className="flex items-center gap-4 mb-4">
            <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-800 mb-1">New Analysis</div>
              <span className="inline-flex px-2 py-1 text-xs font-medium rounded-md bg-green-100 text-green-800 border border-green-200">AI Processing</span>
            </div>
          </div>
          <p className="text-gray-600 mb-6">
            Upload a new cervical slide image for AI-powered cell classification and analysis.
            Get instant results with detailed explanations.
          </p>
          <div className="flex items-center text-green-700 font-semibold group-hover:translate-x-2 transition-transform">
            Start Analysis <span className="ml-2">→</span>
          </div>
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
    <div className="max-w-4xl w-full mx-auto px-4">
      <Card className="backdrop-blur-lg bg-white/90 border-blue-100 shadow-xl">
        <CardHeader className="text-center">
          <div className="w-16 h-16 bg-gradient-to-br from-green-500 to-emerald-600 rounded-full flex items-center justify-center mx-auto mb-4">
            <Brain className="w-8 h-8 text-white" />
          </div>
          <CardTitle className="text-2xl text-gray-800">New Slide Analysis</CardTitle>
          <p className="text-gray-600">Upload and analyze cervical cell slides with AI</p>
        </CardHeader>
        <CardContent className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Patient ID</label>
            <Input 
              placeholder="Enter patient identifier" 
              value={patientId} 
              onChange={(e) => setPatientId(e.target.value)}
              className="h-12"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Slide Image</label>
            <FileUpload selectedFile={selectedFile} onFileSelect={(file) => setSelectedFile(file)} />
          </div>
          
          <div className="flex gap-3 pt-4">
            <Button 
              disabled={!patientId || !selectedFile || loading} 
              onClick={() => selectedFile && onProcess(selectedFile)}
              className="flex-1 h-12 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin mr-2" />
                  Processing...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4 mr-2" />
                  Start Analysis
                </>
              )}
            </Button>
            <Button variant="outline" onClick={onCancel} className="h-12">
              Cancel
            </Button>
          </div>
          {errorMsg && <p className="text-sm text-red-600 text-center">{errorMsg}</p>}
        </CardContent>
      </Card>
    </div>
  );
}

// --- OLD FORM ---
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
    <div className="max-w-4xl w-full mx-auto px-4">
      <Card className="backdrop-blur-lg bg-white/90 border-blue-100 shadow-xl">
        <CardHeader className="text-center">
          <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full flex items-center justify-center mx-auto mb-4">
            <Eye className="w-8 h-8 text-white" />
          </div>
          <CardTitle className="text-2xl text-gray-800">View Previous Results</CardTitle>
          <p className="text-gray-600">Browse and analyze historical slide data</p>
        </CardHeader>
        <CardContent className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Select Patient</label>
            <select
              className="w-full border border-gray-200 rounded-lg px-4 py-3 bg-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              value={selectedPatient}
              onChange={(e) => {
                const v = e.target.value;
                setSelectedPatient(v);
                onRefreshSlides(v);
              }}
            >
              <option value="">Choose a patient...</option>
              {patients.map((p) => (
                <option key={p.pacjent_uid} value={p.pacjent_uid}>{p.pacjent_uid}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Select Slide</label>
            <div className="flex gap-3">
              <select
                className="flex-1 border border-gray-200 rounded-lg px-4 py-3 bg-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                value={selectedSlide}
                onChange={(e) => setSelectedSlide(e.target.value)}
                disabled={!selectedPatient || slides.length === 0}
              >
                <option value="">
                  {selectedPatient ? (slides.length ? "Choose a slide..." : "No slides available") : "Select patient first"}
                </option>
                {slides.map((s) => {
                  const classLabel = s.overall_class ? mapClass(s.overall_class) : null;
                  return (
                    <option key={s.slajd_uid} value={s.slajd_uid}>
                      {s.slajd_uid}
                      {classLabel && ` • ${classLabel}`}
                      {s.status && ` • ${s.status}`}
                    </option>
                  );
                })}
              </select>

              <Button 
                variant="default" 
                disabled={!selectedSlide || loading} 
                onClick={onShowSlide}
                className="h-[52px] bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700"
              >
                <Eye className="w-4 h-4 mr-2" />
                View Results
              </Button>
            </div>
          </div>

          <div className="flex gap-3 pt-4">
            <Button onClick={onBack} variant="outline" className="h-12">
              Back to Home
            </Button>
            <Button 
              onClick={() => onRefreshSlides(selectedPatient)} 
              disabled={!selectedPatient}
              variant="ghost"
              className="h-12"
            >
              Refresh Slides
            </Button>
          </div>

          {errorMsg && <p className="text-sm text-red-600 text-center">{errorMsg}</p>}
        </CardContent>
      </Card>
    </div>
  );
}

// --- ADD INFO CARD ---
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
    <Card className="backdrop-blur-lg bg-white/95 border-blue-100 shadow-md">
      <CardHeader className="pb-3">
        <CardTitle className="text-lg text-gray-800 flex items-center gap-2">
          <User className="w-5 h-5" />
          Additional Notes
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <textarea
          rows={3}
          className="w-full border border-gray-200 rounded-lg p-3 bg-white resize-y min-h-[60px] max-h-[120px] focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Add your observations or notes about this slide..."
        />
        <div className="flex items-center justify-between">
          <Button size="sm" onClick={onSave} disabled={saving} className="bg-blue-600 hover:bg-blue-700">
            {saving ? "Saving..." : "Save Notes"}
          </Button>
          {message && (
            <span className={`text-sm ${message.includes('Saved') ? 'text-green-600' : 'text-gray-600'}`}>
              {message}
            </span>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

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

export function ResultsArea({
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
}: ResultsAreaProps) {
  if (!results || loading) return null;

  const getClassColor = (className: string) =>
    CLASS_COLORS[className] || "bg-gray-100 text-gray-800 border-gray-200";

  return (
    <div className="w-full max-w-7xl mx-auto px-4 py-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
        {/* LEWA KOLUMNA */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          {/* Obraz */}
          <div className="rounded-2xl border border-blue-100 bg-gradient-to-br from-white to-blue-50/30 p-4 shadow-lg">
            <div className="relative w-full h-full min-h-[320px] flex items-center justify-center">
              {imageUrl ? (
                <>
                  <img
                    src={imageUrl}
                    alt="Detected cells with bounding boxes"
                    className="w-full h-full object-contain rounded-xl"
                  />
                  <div className="absolute top-2 right-2">
                    <span
                      className={`inline-flex px-3 py-1 text-sm font-semibold rounded-full ${getClassColor(
                        overallClass
                      )}`}
                    >
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

          {/* Podsumowanie analizy */}
          <div className="bg-white/95 backdrop-blur-lg border border-blue-100 rounded-xl p-6 shadow-md">
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0">
                <div className="w-10 h-10 bg-blue-100 rounded-xl flex items-center justify-center">
                  <Activity className="w-5 h-5 text-blue-600" />
                </div>
              </div>
              <div className="flex-1 min-w-0">
                <h3 className="text-lg font-semibold text-gray-800 mb-3">
                  Analysis Summary
                </h3>
                {results.slide_summary_text ||
                results.slide_summary?.explanation ? (
                  <p className="text-sm text-gray-700 leading-relaxed mb-4">
                    {results.slide_summary_text ||
                      results.slide_summary?.explanation}
                  </p>
                ) : (
                  <p className="text-sm text-gray-600 leading-relaxed mb-4">
                    Automated cell detection and classification results with
                    bounding boxes highlighting detected cells. Each cell is
                    analyzed and classified based on morphological features
                    and staining patterns.
                  </p>
                )}
                {totalCells > 0 && (
                  <div className="flex items-center gap-6 text-sm text-gray-600">
                    <span className="flex items-center gap-2">
                      Detected:{" "}
                      <strong className="text-blue-600 text-lg">
                        {totalCells}
                      </strong>{" "}
                      cells
                    </span>
                    <span className="flex items-center gap-2">
                      Primary class:{" "}
                      <span
                        className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getClassColor(
                          overallClass
                        )}`}
                      >
                        {overallClass}
                      </span>
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* PRAWA KOLUMNA */}
        <div className="lg:col-start-3 flex flex-col gap-6">
          {/* GÓRA: Analysis Summary */}
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
                    <p className="text-sm font-mono break-all">
                      {results.slide_uid ?? "—"}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500 font-medium">
                      Patient ID
                    </p>
                    <p className="text-sm font-mono break-all">
                      {results.pacjent_uid ?? "—"}
                    </p>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <p className="text-sm text-gray-500 font-medium">
                      Classification
                    </p>
                    <span
                      className={`inline-flex px-3 py-1 text-sm font-semibold rounded-full ${getClassColor(
                        overallClass
                      )}`}
                    >
                      {overallClass}
                    </span>
                  </div>
                  <div>
                    <p className="text-sm text-gray-500 font-medium">
                      Total Cells
                    </p>
                    <p className="text-xl font-bold text-blue-600">
                      {totalCells}
                    </p>
                  </div>
                </div>
              </div>

              {totalCells > 0 && (
                <div>
                  <p className="font-semibold mb-3 text-gray-700">
                    Cell Distribution
                  </p>
                  <div className="grid grid-cols-3 gap-2">
                    {Object.entries(classCounts).map(([cls, count]) => (
                      <div
                        key={cls}
                        className="text-center p-2 rounded-lg bg-gray-50"
                      >
                        <span
                          className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full mb-1 ${getClassColor(
                            cls
                          )}`}
                        >
                          {cls}
                        </span>
                        <p className="text-lg font-bold text-gray-700">
                          {count}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* DÓŁ: Individual Cell Predictions */}
          <Card className="flex flex-col flex-1 min-h-0 backdrop-blur-lg bg-white/95 border-blue-100 shadow-lg">
            <CardHeader className="pb-3 shrink-0">
              <CardTitle className="text-lg font-semibold text-gray-800 flex items-center gap-2">
                <Activity className="w-5 h-5 text-blue-600" />
                Individual Cell Predictions
              </CardTitle>
            </CardHeader>
            <CardContent className="flex-1 p-0 overflow-y-hidden">
              <div className="h-64 overflow-y-auto border-t border-gray-200">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 sticky top-0 z-10">
                    <tr>
                      <th className="px-4 py-3 text-left font-medium text-gray-600 border-b">
                        Cell ID
                      </th>
                      <th className="px-4 py-3 text-left font-medium text-gray-600 border-b">
                        Class
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.predict_fused &&
                      Object.entries(results.predict_fused).map(
                        ([cellId, rawClass], index) => {
                          const className = mapClass(rawClass);
                          return (
                            <tr
                              key={cellId}
                              className={`border-b border-gray-100 hover:bg-gray-50 ${
                                index % 2 === 0
                                  ? "bg-white"
                                  : "bg-gray-50/50"
                              }`}
                            >
                              <td
                                className="px-4 py-3 font-mono text-gray-800 text-xs"
                                title={cellId}
                              >
                                {cellId}
                              </td>
                              <td className="px-4 py-3">
                                <span
                                  className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getClassColor(
                                    className
                                  )}`}
                                >
                                  {className}
                                </span>
                              </td>
                            </tr>
                          );
                        }
                      )}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>

          {/* Notatki */}
          <Card className="shrink-0 backdrop-blur-lg bg-white/95 border-blue-100 shadow-md">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-semibold text-gray-800 flex items-center gap-2">
                <User className="w-5 h-5 text-blue-600" />
                Notes
              </CardTitle>
            </CardHeader>
            <CardContent>
              <textarea
                rows={2}
                className="w-full border border-gray-200 rounded-lg px-3 py-2 bg-white resize-none text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                value={addInfoDraft}
                onChange={(e) => setAddInfoDraft(e.target.value)}
                placeholder="Add notes..."
              />
              {saveAddInfoMsg && (
                <span
                  className={`block mt-2 text-xs ${
                    saveAddInfoMsg.includes("Saved")
                      ? "text-green-600"
                      : "text-gray-600"
                  }`}
                >
                  {saveAddInfoMsg}
                </span>
              )}
            </CardContent>
          </Card>

          {/* Przycisk Show Details */}
          <Button
            variant={showDetails ? "secondary" : "default"}
            disabled={!results || loading}
            onClick={() => setShowDetails(!showDetails)}
            className="w-full shrink-0 flex items-center justify-center gap-2 text-sm px-6 h-10"
          >
            <Brain className="w-5 h-5" />
            {showDetails ? "Hide Details" : "Show Details"}
          </Button>
        </div>
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
    <div className="w-full max-w-7xl backdrop-blur-lg bg-white/95 rounded-2xl p-6 shadow-lg mx-4">
      <div className="flex items-center gap-3 mb-6">
        <Brain className="w-7 h-7 text-blue-600" />
        <h2 className="text-3xl font-bold text-gray-800">Individual Cell Analysis</h2>
      </div>

      {results.crop_public_urls ? (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
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
              <Card key={id} className="overflow-hidden shadow-lg border-blue-100">
                <CardHeader className="pb-4">
                  <CardTitle className="flex items-center justify-between">
                    <span className="text-lg">Cell #{id}</span>
                    <span className={`inline-flex px-2 py-1 text-sm font-semibold rounded-md ${getClassColor(cls)}`}>
                      {cls}
                    </span>
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4 p-4">
                  <div className="w-full h-48 border border-gray-200 rounded-lg flex items-center justify-center bg-gradient-to-br from-gray-50 to-gray-100">
                    <img src={url} alt={`cell-${id}`} className="max-h-44 object-contain rounded" />
                  </div>

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
                      disabled={
                        limeBusy || (!results?.features_list?.[id] && !results?.slide_uid)
                      }
                      className="flex-1 hover:bg-green-50"
                      title={
                        !results?.features_list?.[id] && !results?.slide_uid
                          ? "No features found and no slide UID"
                          : ""
                      }
                    >
                      {limeBusy ? (
                        <>
                          <div className="w-3 h-3 border border-gray-400 border-t-gray-600 rounded-full animate-spin mr-1" />
                          LIME...
                        </>
                      ) : (
                        <>
                          <Brain className="w-4 h-4 mr-1" />
                          LIME
                        </>
                      )}
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
                              <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-md border ${getClassColor(mappedClass)}`}>
                                {mappedClass}
                              </span>
                              <div className="flex items-center gap-2">
                                <div className="w-16 bg-gray-200 rounded-full h-2">
                                  <div 
                                    className="bg-blue-600 h-2 rounded-full transition-all" 
                                    style={{ width: `${percentage}%` }}
                                  />
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
                          {featureEntries.slice(0, featureEntries.length).map(([k, v]) => (
                            <div key={k} className="flex justify-between items-center p-2 text-sm">
                              <span className="text-gray-600 truncate flex-1 mr-2">{k}</span>
                              <span className="font-mono text-gray-800">
                                {Number.isFinite(v as number) ? Number(v).toFixed(3) : "—"}
                              </span>
                            </div>
                          ))}
                          {/* {featureEntries.length > 5 && (
                            <div className="p-2 text-xs text-gray-500 text-center">
                              ... and {featureEntries.length - 5} more features
                            </div>
                          )} */}
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

// Helper function to get class colors
const getClassColor = (className: string) => {
  return CLASS_COLORS[className] || "bg-gray-100 text-gray-800 border-gray-200";
};

/* ====================== MAIN APP ====================== */
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
      setShowDetails(false); // Start with details closed
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
      setShowDetails(false);
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

  // save add_info
  const saveAddInfo = async () => {
    const sid = results?.slide_uid ?? selectedSlide;
    if (!sid) return;
    try {
      setSaveAddInfoMsg(null);
      setSavingAddInfo(true);
      const prev = results?.add_info ?? "";

      // optimistic update
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
      setSaveAddInfoMsg("Saved successfully!");
      setTimeout(() => setSaveAddInfoMsg(null), 2000);
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
    <div className="min-h-screen relative overflow-x-hidden font-sans" style={{ fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif" }}>
      {/* Enhanced Background */}
      <div className="fixed inset-0 bg-gradient-to-br from-blue-50 via-indigo-50 to-purple-50" />
      <div className="fixed -top-40 -right-40 w-96 h-96 rounded-full bg-gradient-to-br from-blue-200/40 to-indigo-300/40 blur-3xl animate-pulse" />
      <div className="fixed -bottom-40 -left-40 w-96 h-96 rounded-full bg-gradient-to-br from-indigo-200/40 to-purple-300/40 blur-3xl animate-pulse" style={{ animationDelay: '2s' }} />
      <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-gradient-to-br from-blue-100/20 to-indigo-100/20 blur-3xl" />

      <div className="relative z-10">
        <TopBar user={user} goHome={goHome} doLogout={doLogout} />

        <div className="py-8 flex flex-col items-center gap-8">
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

              {/* Results and Details */}
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

        {/* Enhanced Grad-CAM Dialog */}
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
                  <div className="absolute inset-0 border-4 border-blue-200 rounded-full"></div>
                  <div className="absolute inset-0 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
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
                  <Card className="overflow-hidden">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-medium text-gray-600">Original + Overlay</CardTitle>
                    </CardHeader>
                    <CardContent className="p-0">
                      <img src={gradcamData.overlay_url} className="w-full h-auto" alt="Overlay" />
                    </CardContent>
                  </Card>
                  
                  <Card className="overflow-hidden">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-medium text-gray-600">Grad-CAM Heatmap</CardTitle>
                    </CardHeader>
                    <CardContent className="p-0">
                      <img src={gradcamData.heatmap_url} className="w-full h-auto" alt="Heatmap" />
                    </CardContent>
                  </Card>
                  
                  <Card className="overflow-hidden">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-medium text-gray-600">Activation Map</CardTitle>
                    </CardHeader>
                    <CardContent className="p-0">
                      <img src={gradcamData.activation_url} className="w-full h-auto" alt="Activation" />
                    </CardContent>
                  </Card>
                </div>
                
                <div className="flex items-center justify-center p-4 bg-blue-50 rounded-lg">
                  <p className="text-gray-700">
                    Model prediction for this visualization: 
                    <span className="inline-flex px-2 py-1 text-xs font-semibold rounded-md ml-2 bg-blue-100 text-blue-800 border border-blue-200">
                      {gradcamData.predicted_class}
                    </span>
                  </p>
                </div>
              </div>
            )}
          </DialogContent>
        </Dialog>

        {/* Enhanced Processing Dialog */}
        <Dialog open={loading} onOpenChange={() => {}}>
          <DialogContent
            className="max-w-md"
            onInteractOutside={(e) => e.preventDefault()}
            onEscapeKeyDown={(e) => e.preventDefault()}
          >
            <DialogHeader>
              <DialogTitle className="flex items-center gap-2 text-xl">
                <Brain className="w-6 h-6 text-blue-600" />
                AI Analysis in Progress
              </DialogTitle>
            </DialogHeader>
            <div className="py-6 text-center space-y-6">
              <div className="relative mx-auto w-16 h-16">
                <div className="absolute inset-0 border-4 border-blue-200 rounded-full"></div>
                <div className="absolute inset-0 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
              </div>
              
              <div className="space-y-3">
                <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-gradient-to-r from-blue-500 to-indigo-600 transition-all duration-300 ease-out" 
                    style={{ width: `${progress}%` }} 
                  />
                </div>
                <p className="text-sm font-medium text-gray-700">
                  {progress}% Complete
                </p>
              </div>
              
              <div className="space-y-2">
                <p className="text-lg font-medium text-gray-800">
                  Analyzing slide{patientId && ` for ${patientId}`}
                </p>
                <p className="text-sm text-gray-600">
                  Our AI is examining the cellular structures and classifying each cell. Please wait...
                </p>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  );
}