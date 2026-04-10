import { Suspense } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { LocationsOverview } from "@/pages/LocationsOverview";
import { Dashboard } from "@/pages/Dashboard";

function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={null}>
        <Routes>
          <Route path="/" element={<LocationsOverview />} />
          <Route path="/dashboard/hamburg" element={<Dashboard />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}

export default App;
