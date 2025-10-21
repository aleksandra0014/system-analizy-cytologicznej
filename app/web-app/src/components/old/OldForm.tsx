import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Eye } from "lucide-react";
import type { Patient, SlideItem } from "@/types";
import { mapClass } from "@/lib/constants";

export default function OldForm({ patients, selectedPatient, setSelectedPatient, slides, selectedSlide, setSelectedSlide, loading, errorMsg, onRefreshSlides, onShowSlide, onBack, }: { patients: Patient[]; selectedPatient: string; setSelectedPatient: (v: string) => void; slides: SlideItem[]; selectedSlide: string; setSelectedSlide: (v: string) => void; loading: boolean; errorMsg: string | null; onRefreshSlides: (uid: string) => void; onShowSlide: () => void; onBack: () => void; }) {
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
            <select className="w-full border border-gray-200 rounded-lg px-4 py-3 bg-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500" value={selectedPatient} onChange={(e) => { const v = e.target.value; setSelectedPatient(v); onRefreshSlides(v); }}>
              <option value="">Choose a patient...</option>
              {patients.map((p) => (
                <option key={p.pacjent_uid} value={p.pacjent_uid}>{p.pacjent_uid}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Select Slide</label>
            <div className="flex gap-3">
              <select className="flex-1 border border-gray-200 rounded-lg px-4 py-3 bg-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500" value={selectedSlide} onChange={(e) => setSelectedSlide(e.target.value)} disabled={!selectedPatient || slides.length === 0}>
                <option value="">{selectedPatient ? (slides.length ? "Choose a slide..." : "No slides available") : "Select patient first"}</option>
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
              <Button variant="default" disabled={!selectedSlide || loading} onClick={onShowSlide} className="h-[52px] bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700">
                <Eye className="w-4 h-4 mr-2" />
                View Results
              </Button>
            </div>
          </div>
          <div className="flex gap-3 pt-4">
            <Button onClick={onBack} variant="outline" className="h-12">Back to Home</Button>
            <Button onClick={() => onRefreshSlides(selectedPatient)} disabled={!selectedPatient} variant="ghost" className="h-12">Refresh Slides</Button>
          </div>
          {errorMsg && <p className="text-sm text-red-600 text-center">{errorMsg}</p>}
        </CardContent>
      </Card>
    </div>
  );
}