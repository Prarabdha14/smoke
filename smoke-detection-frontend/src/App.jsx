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
  // Smoke Detection State
  const [smokeFile, setSmokeFile] = useState(null);
  const [smokePreview, setSmokePreview] = useState(null);
  const [smokeLoading, setSmokeLoading] = useState(false);
  const [smokeResult, setSmokeResult] = useState("");

  // CycleGAN State
  const [ganFile, setGanFile] = useState(null);
  const [ganPreview, setGanPreview] = useState(null);
  const [ganLoading, setGanLoading] = useState(false);
  const [ganResult, setGanResult] = useState(null);
  const [ganError, setGanError] = useState(null); // Added this line

  // Smoke Detection Handlers
  const handleSmokeFileChange = (e) => {
    const file = e.target.files[0];
    setSmokeFile(file);
    setSmokeResult("");
    
    if (file) {
      setSmokePreview(URL.createObjectURL(file));
    } else {
      setSmokePreview(null);
    }
  };

  const handleSmokeUpload = async () => {
    if (!smokeFile) return;
    
    setSmokeLoading(true);
    const formData = new FormData();
    formData.append("image", smokeFile);

    try {
      const response = await fetch("http://127.0.0.1:5000/detect", {
        method: "POST",
        body: formData,
        credentials: "include",
      });

      const data = await response.json();
      if (data.error) {
        setSmokeResult("Error: " + data.error);
      } else {
        setSmokeResult(`${data.label} (${data.confidence})`);
      }
    } catch (error) {
      setSmokeResult("Error: Unable to connect to server.");
    } finally {
      setSmokeLoading(false);
    }
  };

  // CycleGAN Handlers
  const handleGanFileChange = (e) => {
    const file = e.target.files[0];
    setGanFile(file);
    setGanResult(null);
    setGanError(null); // Clear previous errors when selecting new file
    
    if (file) {
      setGanPreview(URL.createObjectURL(file));
    } else {
      setGanPreview(null);
    }
  };

const handleGanGenerate = async () => {
  if (!ganFile) return;
  
  setGanLoading(true);
  setGanResult(null);
  setGanError(null);
  
  const formData = new FormData();
  formData.append("image", ganFile);

  try {
    const response = await fetch("http://127.0.0.1:5000/generate", {
      method: "POST",
      body: formData,
    });
    
    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(
        errorText.includes('{') 
          ? JSON.parse(errorText).error 
          : `Server error: ${response.status}`
      );
    }
    
    const blob = await response.blob();
    setGanResult(URL.createObjectURL(blob));
  } catch (error) {
    console.error("Generation error:", error);
    setGanError(
      error.message.includes("shape") 
        ? "Model input shape mismatch. Please check image dimensions."
        : error.message
    );
  } finally {
    setGanLoading(false);
  }
};

  return (
    <main className="flex flex-col md:flex-row items-center justify-center min-h-screen bg-gradient-to-br from-gray-100 to-blue-100 px-4 gap-8 py-8">
      {/* Smoke Detection Card */}
      <Card>
        <CardContent>
          <h1 className="text-3xl font-extrabold mb-6 text-center text-gray-800 tracking-tight">
            🚀 Smoke Detection AI
          </h1>

          <input
            type="file"
            accept="image/*"
            onChange={handleSmokeFileChange}
            className="block w-full mb-4 text-sm text-gray-600 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          />

          {smokePreview && (
            <div className="mb-4">
              <img
                src={smokePreview}
                alt="Preview"
                className="w-full h-48 object-cover rounded-lg shadow"
              />
            </div>
          )}

          <Button onClick={handleSmokeUpload} disabled={smokeLoading}>
            {smokeLoading ? "Analyzing..." : "Upload & Analyze"}
          </Button>

          {smokeResult && (
            <p className="mt-6 text-center text-lg font-semibold text-indigo-700 animate-fade-in">
              Result: {smokeResult}
            </p>
          )}
        </CardContent>
      </Card>

      {/* CycleGAN Card */}
      <Card>
        <CardContent>
          <h2 className="text-3xl font-extrabold mb-6 text-center text-gray-800 tracking-tight">
            🌀 Image Enhancement (CycleGAN)
          </h2>

          <input
            type="file"
            accept="image/*"
            onChange={handleGanFileChange}
            className="block w-full mb-4 text-sm text-gray-600 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-purple-50 file:text-purple-700 hover:file:bg-purple-100"
          />

          {ganPreview && (
            <div className="mb-4">
              <img
                src={ganPreview}
                alt="Preview"
                className="w-full h-48 object-cover rounded-lg shadow"
              />
            </div>
          )}

          <Button onClick={handleGanGenerate} disabled={ganLoading}>
            {ganLoading ? "Generating..." : "Generate Enhanced Image"}
          </Button>

          {/* Error Display */}
          {ganError && (
            <p className="mt-4 text-center text-red-600">
              Error: {ganError}
            </p>
          )}

          {/* Result Display */}
          {ganResult && ganResult.startsWith('blob:') && (
            <div className="mt-6">
              <p className="text-center text-lg font-semibold text-purple-700 mb-2">
                Enhanced Image:
              </p>
              <img
                src={ganResult}
                alt="Enhanced"
                className="w-full h-48 object-cover rounded-lg shadow"
              />
            </div>
          )}
        </CardContent>
      </Card>
    </main>
  );
}

export default App;