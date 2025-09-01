import { useState } from "react";
// import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

const App = () => {
  type Results = {
  df_preds?: unknown;
  features_list?: unknown;
  probs?: unknown;
  bbox_public_url?: string; // <---
  [key: string]: unknown;
  };
  const [results, setResults] = useState<Results | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleProcess = async (file: File) => {
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("http://localhost:8000/process-image/", {
      method: "POST",
      body: formData,
    });
    const data: Results = await response.json();
    setResults(data);
    setLoading(false);

    if (data.bbox_public_url) {
      // backend już zwrócił poprawny absolutny URL
      setImageUrl(data.bbox_public_url);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center py-10">
      <Card className="w-full max-w-xl shadow-lg">
        <CardHeader>
          <CardTitle>Przetwarzanie obrazu cytologicznego</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-4">
            <Input
              type="file"
              accept="image/*"
              onChange={e => {
                if (e.target.files && e.target.files[0]) {
                  handleProcess(e.target.files[0]);
                }
              }}
              className="file:mr-4 file:py-2 file:px-4 file:rounded file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
            />
            {loading && <div className="text-blue-600">Przetwarzanie...</div>}
            {imageUrl && (
              <img
                src={imageUrl}
                alt="Wynik z bboxami"
                className="rounded border max-w-full mx-auto my-4"
                style={{ maxHeight: 400 }}
              />
            )}
            {results && (
              <div className="space-y-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Predykcje</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <pre className="whitespace-pre-wrap text-xs">{JSON.stringify(results.predict_fused, null, 2)}</pre>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle>Features</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <pre className="whitespace-pre-wrap text-xs">{JSON.stringify(results.features_list, null, 2)}</pre>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle>Probs</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <pre className="whitespace-pre-wrap text-xs">{JSON.stringify(results.probs, null, 2)}</pre>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default App;