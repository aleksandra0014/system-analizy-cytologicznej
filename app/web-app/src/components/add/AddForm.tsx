import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Brain, Zap } from "lucide-react";
import FileUpload from "@/components/common/FileUpload";

export default function AddForm({ patientId, setPatientId, selectedFile, setSelectedFile, loading, errorMsg, onProcess, onCancel, }: { patientId: string; setPatientId: (v: string) => void; selectedFile: File | null; setSelectedFile: (f: File | null) => void; loading: boolean; errorMsg: string | null; onProcess: (f: File) => void; onCancel: () => void; }) {
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
            <Input placeholder="Enter patient identifier" value={patientId} onChange={(e) => setPatientId(e.target.value)} className="h-12" />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Slide Image</label>
            <FileUpload selectedFile={selectedFile} onFileSelect={(file) => setSelectedFile(file)} />
          </div>
          <div className="flex gap-3 pt-4">
            <Button disabled={!patientId || !selectedFile || loading} onClick={() => selectedFile && onProcess(selectedFile)} className="flex-1 h-12 bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700">
              {loading ? (<><div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin mr-2" />Processing...</>) : (<><Zap className="w-4 h-4 mr-2" />Start Analysis</>)}
            </Button>
            <Button variant="outline" onClick={onCancel} className="h-12">Cancel</Button>
          </div>
          {errorMsg && <p className="text-sm text-red-600 text-center">{errorMsg}</p>}
        </CardContent>
      </Card>
    </div>
  );
}