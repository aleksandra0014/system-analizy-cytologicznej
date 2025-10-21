import { Upload } from "lucide-react";

export default function FileUpload({ selectedFile, onFileSelect }: { selectedFile: File | null; onFileSelect: (file: File | null) => void }) {
  return (
    <div className="border-2 border-dashed border-blue-200 rounded-xl p-6 text-center hover:border-blue-300 transition-colors">
      <input type="file" accept="image/*" className="hidden" id="file-upload" onChange={(e) => onFileSelect(e.target.files?.[0] || null)} />
      <label htmlFor="file-upload" className="cursor-pointer">
        <Upload className="w-8 h-8 mx-auto mb-2 text-blue-500" />
        <p className="text-sm text-gray-600">{selectedFile ? selectedFile.name : "Click to upload slide image"}</p>
      </label>
    </div>
  );
}