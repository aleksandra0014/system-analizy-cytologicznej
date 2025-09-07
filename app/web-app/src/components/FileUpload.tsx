import { useRef, useState, useEffect } from "react";

export default function FileUpload({
  onFileSelect,
  height = 160,
  selectedFile = null, // Dodajemy prop do synchronizacji
}: {
  onFileSelect: (file: File | null) => void;
  height?: number;
  selectedFile?: File | null; // Nowy prop
}) {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Synchronizacja z zewnętrznym stanem
  useEffect(() => {
    setFile(selectedFile);
  }, [selectedFile]);

  // mapowanie MIME → rozszerzenia
  const EXT_BY_MIME: Record<string, string> = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/bmp": "bmp",
  };

  // nazwa do wyświetlenia
  const displayName = (f: File) => {
    const raw = (f.name || "").trim();
    if (raw) return raw;
    const ext = EXT_BY_MIME[f.type] || "bin";
    return `plik-${Date.now()}.${ext}`;
  };

  const pickFile = (f: File | null) => {
    if (!f) {
      setFile(null);
      onFileSelect(null);
      return;
    }
    // sprawdzenie rozszerzenia/MIME
    const okMime = Object.keys(EXT_BY_MIME).includes(f.type);
    const okExt = /\.(jpe?g|png|bmp)$/i.test(f.name || "");
    if (!okMime && !okExt) {
      alert("Dozwolone pliki: JPG, JPEG, PNG, BMP");
      return;
    }
    setFile(f);
    onFileSelect(f);
  };

  const hasFileOrDrag = !!file || isDragging;
  const boxClass = [
    "relative w-full rounded-lg flex items-center justify-center cursor-pointer transition p-4 bg-white text-center",
    "border",
    hasFileOrDrag
      ? "border-2 border-dashed border-yellow-600 bg-orange-200"
      : "border-2 border-dashed border-gray-300 hover:border-gray-400",
  ].join(" ");

  // Debug info
  console.log("FileUpload Debug:");
  console.log("- Internal file state:", file);
  console.log("- External selectedFile prop:", selectedFile);
  console.log("- File name:", file?.name);
  console.log("- Display name:", file ? displayName(file) : "no file");

  return (
    <div className="flex flex-col items-center gap-2 w-full">
      <div
        className={boxClass}
        style={{ height }}
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => { 
          e.preventDefault(); 
          setIsDragging(true); 
        }}
        onDragLeave={() => setIsDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setIsDragging(false);
          pickFile(e.dataTransfer.files?.[0] ?? null);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".jpg,.jpeg,.png,.bmp"
          className="hidden"
          onChange={(e) => pickFile(e.target.files?.[0] ?? null)}
        />

        {!file ? (
          <div className="flex flex-col items-center justify-center text-zinc-500 px-4">
            <div className="text-4xl mb-2">📁</div>
            <p>
              Drag and drop file or click to select
            </p>
            <p className="text-xs mt-1">
              (JPG, JPEG, PNG, BMP)
            </p>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center w-full h-full">
            <div className="font-medium text-yellow-800 text-center break-words max-w-full px-2">
              Added file: {displayName(file)}
            </div>
            <div className="text-sm text-yellow-600 mt-1">
              {(file.size / 1024 / 1024).toFixed(2)} MB
            </div>
          </div>
        )}
      </div>
      
    </div>
  );
}