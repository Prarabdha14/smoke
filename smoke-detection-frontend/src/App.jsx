import React, { useState } from "react";
import "./index.css";

const Button = ({ children, ...props }) => (
  <button
    className="bg-gradient-to-r from-blue-500 to-indigo-600 text-white px-6 py-3 rounded-xl hover:from-blue-600 hover:to-indigo-700 disabled:opacity-50 w-full transition-all"
    {...props}
  >
    {children}
  </button>
);

const Card = ({ children }) => (
  <div className="bg-white/80 backdrop-blur-md shadow-2xl rounded-3xl p-10 w-full max-w-md mx-auto border border-gray-200">
    {children}
  </div>
);

const CardContent = ({ children }) => <div>{children}</div>;

function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState("");

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setResult("");

    if (selectedFile) {
      setPreviewUrl(URL.createObjectURL(selectedFile));
    } else {
      setPreviewUrl(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setLoading(true);
    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await fetch("http://127.0.0.1:5000/detect", {
        method: "POST",
        body: formData,
        credentials: "include",
      });

      const data = await response.json();
      if (data.error) {
        setResult("Error: " + data.error);
      } else {
        setResult(`${data.label} (${data.confidence})`);
      }
    } catch (error) {
      setResult("Error: Unable to connect to server.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="flex items-center justify-center min-h-screen bg-gradient-to-br from-gray-100 to-blue-100 px-4">
      <Card>
        <CardContent>
          <h1 className="text-3xl font-extrabold mb-6 text-center text-gray-800 tracking-tight">
            🚀 Smoke Detection AI
          </h1>

          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            className="block w-full mb-4 text-sm text-gray-600 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          />

          {previewUrl && (
            <div className="mb-4">
              <img
                src={previewUrl}
                alt="Preview"
                className="w-full h-48 object-cover rounded-lg shadow"
              />
            </div>
          )}

          <Button onClick={handleUpload} disabled={loading}>
            {loading ? "Analyzing..." : "Upload & Analyze"}
          </Button>

          {result && (
            <p className="mt-6 text-center text-lg font-semibold text-indigo-700 animate-fade-in">
              Result: {result}
            </p>
          )}
        </CardContent>
      </Card>
      {/*<Card>
        <CardContent>
        <h2 className="text-xl font-bold mt-10 mb-4 text-center text-gray-700">🌀 Enhance Image with CycleGAN</h2>

        <input
          type="file"
          accept="image/*"
          onChange={(e) => {
            const selected = e.target.files[0];
            setFile(selected);
            if (selected) setPreviewUrl(URL.createObjectURL(selected));
          }}
          className="block w-full mb-4 text-sm text-gray-600 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-purple-50 file:text-purple-700 hover:file:bg-purple-100"
        />

        <Button onClick={async () => {
          if (!file) return;
          setLoading(true);
          const formData = new FormData();
          formData.append("image", file);

          try {
            const response = await fetch("http://127.0.0.1:5000/generate", {
              method: "POST",
              body: formData,
            });
            const blob = await response.blob();
            const url = URL.createObjectURL(blob);
            setResult(url);
          } catch (err) {
            console.error(err);
          } finally {
            setLoading(false);
          }
        }} disabled={loading}>
          {loading ? "Generating..." : "Generate Enhanced Image"}
        </Button>

        {result && (
          <div className="mt-6 text-center">
            <p className="text-md text-gray-700 font-medium mb-2">Enhanced Output:</p>
            <img src={result} alt="Enhanced" className="w-full h-48 object-cover rounded-lg shadow" />
          </div>
        )}
        </CardContent>
      </Card>*/}
    </main>
  );
}

export default App;
