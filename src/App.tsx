import { BrowserRouter, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import DriverBehavior from "./pages/DriverBehaviour";
import CharacterRecognition from "./pages/CharacterRecognition";
import "./styles/buttons.css";

export default function App() {
  return (
    <BrowserRouter basename="/browser-deployed-models">
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/driver" element={<DriverBehavior />} />
        <Route path="/character-recognition" element={<CharacterRecognition />} />
      </Routes>
    </BrowserRouter>
  );
}

