import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

const CLASSES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".split("");

export default function CharacterRecognition() {
  const [session, setSession] = useState<ort.InferenceSession | null>(null);
  const [loading, setLoading] = useState(true);
  const [output, setOutput] = useState<any>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const drawing = useRef(false);

  // -----------------------------
  // Load ONNX model
  // -----------------------------
  useEffect(() => {
    async function loadModel() {
      try {
        const modelPath = `${import.meta.env.BASE_URL}models/character_recognition.onnx`;

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

    ctx.fillStyle = "black";
    ctx.beginPath();
    ctx.arc(x, y, 8, 0, Math.PI * 2);
    ctx.fill();
  }

  function clearCanvas() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  // -----------------------------
  // Preprocess image (upload or canvas)
  // -----------------------------
  async function preprocessImageFromUrl(url: string): Promise<ort.Tensor> {
    return new Promise((resolve) => {
      const img = new Image();
      img.src = url;

      img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = 28;
        canvas.height = 28;

        const ctx = canvas.getContext("2d")!;
        ctx.drawImage(img, 0, 0, 28, 28);

        const { data } = ctx.getImageData(0, 0, 28, 28);

        const floatData = new Float32Array(1 * 28 * 28);

        for (let i = 0; i < 28 * 28; i++) {
          const pixel = data[i * 4]; // grayscale
          floatData[i] = pixel / 255;
        }

        const tensor = new ort.Tensor("float32", floatData, [1, 1, 28, 28]);
        resolve(tensor);
      };
    });
  }

  // -----------------------------
  // Inference
  // -----------------------------
  async function runInference(url: string) {
    if (!session) return;

    const inputTensor = await preprocessImageFromUrl(url);

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
              background: "white",
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
              <img
                src={previewUrl}
                alt="preview"
                style={{ width: 200, borderRadius: 8 }}
              />
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
