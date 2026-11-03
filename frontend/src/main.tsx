import { createRoot } from "react-dom/client";
const root = createRoot(document.getElementById("root")!);
if (import.meta.env.MODE === "demo") {
  void import("./Demo").then(({ default: Demo }) => root.render(<Demo />));
} else {
  void import("./App").then(({ default: App }) => root.render(<App />));
}
