import { Link } from "react-router-dom";

export default function Home() {
  return (
    <div style={{ padding: 20 }}>
      <h1>Browser‑Deployed Models</h1>

      <button>
        <Link to="/driver">Driver Behaviour Prediction</Link>
      </button>
    </div>
  );
}
