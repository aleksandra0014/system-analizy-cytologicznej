import { useState } from "react";

export default function FileUpload({
  onFileSelect,
  height = 160, // you can tweak the height if you like
}: {
  onFileSelect: (file: File) => void;
  height?: number;
}) {
  const [isDragging, setIsDragging] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setFileName(file.name);
      onFileSelect(file);
    }
  };

  return (
    <div className="flex flex-col items-center gap-2 w-full">
      <div
        className={`w-full border-2 border-dashed rounded-lg flex items-center justify-center cursor-pointer transition
        ${isDragging ? "border-blue-500 bg-blue-50" : "border-gray-300 bg-white"}`}
        style={{ height }}
        onDragOver={(e) => {
          e.preventDefault();
          setIsDragging(true);
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={handleDrop}
        onClick={() => document.getElementById("fileInput")?.click()}
      >
        <input
          id="fileInput"
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            if (e.target.files && e.target.files[0]) {
              const file = e.target.files[0];
              setFileName(file.name);
              onFileSelect(file);
            }
          }}
        />
        <p className="text-gray-500 text-center px-4">
          {isDragging
            ? "Drop the file here 📂"
            : "Drag & drop a file here or click to select"}
        </p>
      </div>

      {fileName && (
        <p className="text-sm text-gray-600 w-full truncate">
          📄 <span title={fileName}>{fileName}</span>
        </p>
      )}
    </div>
  );
}
