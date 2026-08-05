import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { AttributionsPage } from "./components/AttributionsPage";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <AttributionsPage />
  </StrictMode>,
);
