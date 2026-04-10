import { Suspense } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { LocationsOverview } from "@/pages/LocationsOverview";
import { HamburgDashboard } from "@/pages/HamburgDashboard";
import { LoadingSpinner } from "@/components/LoadingSpinner";

function App() {
  return (
    <BrowserRouter>
      <Suspense fallback={<LoadingSpinner fullscreen />}>
        <Routes>
          <Route path="/" element={<LocationsOverview />} />
          <Route path="/dashboard/hamburg" element={<HamburgDashboard />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}

export default App;
