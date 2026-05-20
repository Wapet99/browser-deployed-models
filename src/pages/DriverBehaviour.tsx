import { useEffect, useState } from "react";
import * as ort from "onnxruntime-web";

const DRIVER_CLASSES = [
  "safe driving",
  "texting (right)",
  "talking on phone (right)",
  "texting (left)",
  "talking on phone (left)",
  "operating the radio",
  "drinking",
  "reaching behind",
  "hair / makeup",
  "talking to passenger",
];


export default function DriverBehavior() {
    const [session, setSession] = useState<ort.InferenceSession | null>(null);
    const [loading, setLoading] = useState(true);
    const [output, setOutput] = useState<any>(null);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);

    useEffect(() => {
        async function loadModel() {
            try {
                const modelPath = `${import.meta.env.BASE_URL}models/minicnn_int8.onnx`;
                console.log("Loading model from:", modelPath);

                const s = await ort.InferenceSession.create(
                    modelPath,
                    { executionProviders: ["wasm"] }
                );
                setSession(s);
                } catch (err) {
                    console.error("Failed to load ONNX model:", err);
                } finally {
                    setLoading(false);
                    console.log("Model loaded successfully")
                }
        }

        loadModel();
    }, []);

    // Image preprocessing
    async function preprocessImage(file: File): Promise<ort.Tensor> {
        return new Promise((resolve) => {
            const img = new Image();
            img.src = URL.createObjectURL(file);

            img.onload = () => {
                const canvas = document.createElement("canvas");
                canvas.width = 224;
                canvas.height = 224;

                const ctx = canvas.getContext("2d")!;
                ctx.drawImage(img, 0, 0, 224, 224);

                const imageData = ctx.getImageData(0, 0, 224, 224);
                const { data } = imageData;

                //ImageNet normalisation
                const mean = [0.485, 0.456, 0.406];
                const std = [0.229, 0.224, 0.225];

                const floatData = new Float32Array(3 * 224 * 224);

                for (let i = 0; i < 224 * 224; i++) {
                    const r = data[i * 4] / 255;
                    const g = data[i * 4 + 1] / 255;
                    const b = data[i * 4 + 2] /255;

                    floatData[i] = (r - mean[0]) / std[0];
                    floatData[i + 224 * 224] = (g - mean[1]) / std[1];
                    floatData[i + 2 * 224 * 224] = (b - mean[2]) / std[2];
                }

                const tensor = new ort.Tensor("float32", floatData, [1, 3, 224, 224]);
                resolve(tensor);
            };
        });
    }

    // Run inference
    async function handleImageUpload(e: React.ChangeEvent<HTMLInputElement>) {
        if (!session) return;

        const file = e.target.files?.[0];
        if (!file) return;

        setPreviewUrl(URL.createObjectURL(file));

        const inputTensor = await preprocessImage(file);

        const feeds: Record<string, ort.Tensor> = {};
        feeds[session.inputNames[0]] = inputTensor;

        const start = performance.now();
        const results = await session.run(feeds);
        const end = performance.now();

        const latency = end - start;

        const outputTensor = results[session.outputNames[0]];
        const logits = Array.from(outputTensor.data as Float32Array);

        // Softmax
        const maxLogit = Math.max(...logits);
        const exps = logits.map((x) => Math.exp(x - maxLogit));
        const sumExps = exps.reduce((a, b) => a + b, 0);
        const softmax = exps.map((x) => x / sumExps);

        // Top class
        const topIndex = softmax.indexOf(Math.max(...softmax));
        const topLabel = DRIVER_CLASSES[topIndex];

        setOutput({
            logits,
            softmax,
            topIndex,
            topLabel,
            latencyMs: latency.toFixed(2),
        });
    }

    return (
        <div style={{ padding: 20 }}>
            <h1>Driver Behaviour Prediction</h1>

            {loading && <p>Loading model…</p>}

            {!loading && !session && (
                <p style={{ color: "red" }}>Failed to load model.</p>
            )}

            {!loading && session && (
                <>
                    <input type="file" accept="image/*" onChange={handleImageUpload} />
                    {previewUrl && (
                        <div style={{ marginTop: 20 }}>
                            <h3>Image Preview</h3>
                            <img src={previewUrl} alt="preview" style={{ width: "100%", borderRadius: 8 }} />
                        </div>
                    )}

                    {output && (
                        <div style={{ marginTop: 20 }}>
                            <h3>Prediction</h3>
                            <p><strong>Top class:</strong> c{output.topIndex} - {output.topLabel}</p>
                            <p><strong>Latency:</strong> {output.latencyMs} ms</p>

                            <h3>Confidence</h3>
                            {output.softmax.map((conf: number, i: number) => (
                                <div key={i} style={{ marginBottom: 8 }}>
                                    <div style={{ fontSize: 14, marginBottom: 4 }}>
                                        {i}: {DRIVER_CLASSES[i]} - {(conf * 100).toFixed(1)}%
                                    </div>
                                    <div style={{ height: 10, background: "#ddd", borderRadius: 4, overflow: "hidden" }}>
                                        <div style={{ width: `${conf * 100}%`, height: "100%", background: i === output.topIndex ? "#4cbc50" : "#2173e6" }} />
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
