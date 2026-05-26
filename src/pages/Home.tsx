import ModelLinkButton from "../components/ModelLinkButton";

export default function Home() {
  return (
    <div style={{ padding: 20 }}>
      <h1>Browser‑Deployed Models</h1>

      <ModelLinkButton
        to="/driver"
        label="Driver Behaviour Prediction"
      />

      <ModelLinkButton
        to="/character-recognition"
        label="Character Recognition"
      />
    </div>
  );
}
