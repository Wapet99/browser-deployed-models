import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

const CLASSES = [
  "0","1","2","3","4","5","6","7","8","9",
  "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
  "a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"
];

export default function CharacterRecognition() {
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [loading, setLoading] = useState(true);
  const [output, setOutput] = useState<any>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [processedPreviewUrl, setProcessedPreviewUrl] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const drawing = useRef(false);

  // -----------------------------
  // Load ONNX model
  // -----------------------------
  useEffect(() => {
    async function loadModel() {
      try {
        const modelPath = `${import.meta.env.BASE_URL}models/character_recognition.onnx`;
        console.log("Loading model from:", modelPath);

        const s = await ort.InferenceSession.create(modelPath, {
          executionProviders: ["wasm"],
        });

        setSession(s);
      } catch (err) {
        console.error("Failed to load ONNX model:", err);
      } finally {
        setLoading(false);
      }
    }

    loadModel();
  }, []);

  // -----------------------------
  // Canvas drawing handlers
  // -----------------------------
  function startDraw(e: React.MouseEvent) {
    drawing.current = true;
    draw(e);
  }

  function endDraw() {
    drawing.current = false;
  }

  function draw(e: React.MouseEvent) {
    if (!drawing.current || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d")!;
    const rect = canvas.getBoundingClientRect();

    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.fillStyle = "white";
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fill();
  }

  function clearCanvas() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  // -----------------------------
  // Preprocess image (upload or canvas)
  // -----------------------------
  async function preprocessImageFromUrl(url: string, onDebugImage?: (url: string) => void): Promise<ort.Tensor> {
    return new Promise((resolve) => {
      const img = new Image();
      img.src = url;

      img.onload = () => {
        // Step 1: Draw original image to a working canvas
        const srcCanvas = document.createElement("canvas");
        srcCanvas.width = 200;
        srcCanvas.height = 200;
        const srcCtx = srcCanvas.getContext("2d")!;
        srcCtx.drawImage(img, 0, 0, 200, 200);

        const srcData = srcCtx.getImageData(0, 0, 200, 200);
        const pixels = srcData.data;

        // Step 2: Convert to grayscale
        const gray = new Float32Array(200 * 200);
        for (let i = 0; i < 200 * 200; i++) {
          const r = pixels[i * 4];
          const g = pixels[i * 4 + 1];
          const b = pixels[i * 4 + 2];
          const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
          gray[i] = luminance / 255;
        }

        // Step 3: Find bounding box of non‑zero pixels
        let minX = 200, minY = 200, maxX = 0, maxY = 0;
        for (let y = 0; y < 200; y++) {
          for (let x = 0; x < 200; x++) {
            const v = gray[y * 200 + x];
            if (v > 0.05) { // threshold
              if (x < minX) minX = x;
              if (x > maxX) maxX = x;
              if (y < minY) minY = y;
              if (y > maxY) maxY = y;
            }
          }
        }

        // Handle blank canvas
        if (minX > maxX || minY > maxY) {
          const blank = new Float32Array(28 * 28);
          resolve(new ort.Tensor("float32", blank, [1, 1, 28, 28]));
          return;
        }

        const width = maxX - minX + 1;
        const height = maxY - minY + 1;

        // Step 4: Crop to bounding box
        const cropCanvas = document.createElement("canvas");
        cropCanvas.width = width;
        cropCanvas.height = height;
        const cropCtx = cropCanvas.getContext("2d")!;
        const cropImageData = cropCtx.createImageData(width, height);

        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const v = gray[(y + minY) * 200 + (x + minX)] * 255;
            const idx = (y * width + x) * 4;
            cropImageData.data[idx] = v;
            cropImageData.data[idx + 1] = v;
            cropImageData.data[idx + 2] = v;
            cropImageData.data[idx + 3] = 255;
          }
        }
        cropCtx.putImageData(cropImageData, 0, 0);

        // Step 5: Resize to 20×20 (preserving aspect ratio)
        const targetCanvas = document.createElement("canvas");
        targetCanvas.width = 20;
        targetCanvas.height = 20;
        const targetCtx = targetCanvas.getContext("2d")!;
        targetCtx.drawImage(cropCanvas, 0, 0, 20, 20);

        // Step 6: Pad to 28×28
        const finalCanvas = document.createElement("canvas");
        finalCanvas.width = 28;
        finalCanvas.height = 28;
        const finalCtx = finalCanvas.getContext("2d")!;

        finalCtx.fillStyle = "black";
        finalCtx.fillRect(0, 0, 28, 28);

        const padX = Math.floor((28 - 20) / 2);
        const padY = Math.floor((28 - 20) / 2);
        finalCtx.drawImage(targetCanvas, padX, padY);

        // Step 7: Center using center of mass
        const finalData = finalCtx.getImageData(0, 0, 28, 28);
        const f = finalData.data;

        let sum = 0, sumX = 0, sumY = 0;
        for (let y = 0; y < 28; y++) {
          for (let x = 0; x < 28; x++) {
            const v = f[(y * 28 + x) * 4] / 255;
            sum += v;
            sumX += x * v;
            sumY += y * v;
          }
        }

        if (sum > 0) {
          const cx = sumX / sum;
          const cy = sumY / sum;
          const shiftX = Math.round(14 - cx);
          const shiftY = Math.round(14 - cy);

          const shiftedCanvas = document.createElement("canvas");
          shiftedCanvas.width = 28;
          shiftedCanvas.height = 28;
          const shiftedCtx = shiftedCanvas.getContext("2d")!;
          shiftedCtx.fillStyle = "black";
          shiftedCtx.fillRect(0, 0, 28, 28);
          shiftedCtx.drawImage(finalCanvas, shiftX, shiftY);

          finalCtx.drawImage(shiftedCanvas, 0, 0);
        }

        // Step 8: Convert to tensor
        const out = finalCtx.getImageData(0, 0, 28, 28);
        const outData = new Float32Array(28 * 28);

        for (let i = 0; i < 28 * 28; i++) {
          outData[i] = out.data[i * 4] / 255; // already inverted
        }

        const inputTensor = new ort.Tensor("float32", outData, [1, 1, 28, 28]);
        if (onDebugImage) {
          onDebugImage(finalCanvas.toDataURL("image/png"));
        }

        resolve(inputTensor);
      };
    });
  }

  // -----------------------------
  // Inference
  // -----------------------------
  async function runInference(url: string) {
    if (!session) return;

    const inputTensor = await preprocessImageFromUrl(url, (debugUrl) => {setProcessedPreviewUrl(debugUrl)});

    const feeds: Record<string, ort.Tensor> = {};
    feeds[session.inputNames[0]] = inputTensor;

    const start = performance.now();
    const results = await session.run(feeds);
    const end = performance.now();

    const outputTensor = results[session.outputNames[0]];
    const logits = Array.from(outputTensor.data as Float32Array);

    const maxLogit = Math.max(...logits);
    const exps = logits.map((x) => Math.exp(x - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b, 0);
    const softmax = exps.map((x) => x / sumExps);

    const topIndex = softmax.indexOf(Math.max(...softmax));

    setOutput({
      logits,
      softmax,
      topIndex,
      topLabel: CLASSES[topIndex],
      latencyMs: (end - start).toFixed(2),
    });
  }

  // -----------------------------
  // Upload handler
  // -----------------------------
  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    runInference(url);
  }

  // -----------------------------
  // Canvas inference
  // -----------------------------
  async function handleCanvasPredict() {
    if (!canvasRef.current) return;

    const url = canvasRef.current.toDataURL("image/png");
    setPreviewUrl(url);
    runInference(url);
  }

  // -----------------------------
  // UI
  // -----------------------------
  return (
    <div style={{ padding: 20 }}>
      <h1>Character Recognition</h1>

      {loading && <p>Loading model…</p>}

      {!loading && !session && (
        <p style={{ color: "red" }}>Failed to load model.</p>
      )}

      {!loading && session && (
        <>
          <h3>Upload an Image</h3>
          <input type="file" accept="image/*" onChange={handleUpload} />

          <h3 style={{ marginTop: 30 }}>Or Draw a Character</h3>
          <canvas
            ref={canvasRef}
            width={200}
            height={200}
            style={{
              border: "1px solid #ccc",
              background: "black",
              borderRadius: 8,
              cursor: "crosshair",
            }}
            onMouseDown={startDraw}
            onMouseUp={endDraw}
            onMouseMove={draw}
          />

          <div style={{ marginTop: 10 }}>
            <button onClick={clearCanvas} style={{ marginRight: 10 }}>
              Clear
            </button>
            <button onClick={handleCanvasPredict}>Predict</button>
          </div>

          {previewUrl && (
            <div style={{ marginTop: 20 }}>
              <h3>Input Preview</h3>
              <div style={{ display: "flex", justifyContent: "center", gap: 20 }}>
                <div>
                  <div style={{ fontSize: 14, marginBottom: 4 }}>Original</div>
                  <img
                    src={previewUrl}
                    alt="preview"
                    style={{ 
                      width: 200, 
                      background: "black",
                      borderRadius: 8,
                    }}
                  />
                </div>

                {processedPreviewUrl && (
                  <div>
                    <div style={{ fontSize: 14, marginBottom: 4 }}>Processed (28×28)</div>
                    <img
                      src={processedPreviewUrl}
                      alt="processed"
                      style={{
                        width: 200,
                        imageRendering: "pixelated",
                        background: "black",
                        borderRadius: 8,
                      }}
                    />
                  </div>
                )}
              </div>
            </div>
          )}

          {output && (
            <div style={{ marginTop: 20 }}>
              <h3>Prediction</h3>
              <p>
                <strong>Top class:</strong> {output.topLabel}
              </p>
              <p>
                <strong>Latency:</strong> {output.latencyMs} ms
              </p>

              <h3>Confidence</h3>
              {output.softmax.map((conf: number, i: number) => (
                <div key={i} style={{ marginBottom: 8 }}>
                  <div style={{ fontSize: 14, marginBottom: 4 }}>
                    {CLASSES[i]} — {(conf * 100).toFixed(1)}%
                  </div>
                  <div
                    style={{
                      height: 10,
                      background: "#ddd",
                      borderRadius: 4,
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        width: `${conf * 100}%`,
                        height: "100%",
                        background:
                          i === output.topIndex ? "#4cbc50" : "#2173e6",
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
