import { useState } from "react";
import { Sidebar } from "./components/Sidebar";
import { Dashboard } from "./pages/Dashboard";
import { RecoveryCase } from "./pages/RecoveryCase";
import { DEMO_PAYMENT_CASES } from "./data/demoCases";
import { DemoPaymentCase } from "./types/recovery";

export function App() {
  const [activeTab, setActiveTab] = useState<"dashboard" | "cases">("dashboard");
  const [selectedCase, setSelectedCase] = useState<DemoPaymentCase>(DEMO_PAYMENT_CASES[0]);

  const handleSelectCase = (c: DemoPaymentCase) => {
    setSelectedCase(c);
    setActiveTab("cases");
  };

  return (
    <div className="flex min-h-screen bg-slate-950 text-slate-100 selection:bg-emerald-500/20 selection:text-emerald-300">
      {/* Fixed Left Navigation Sidebar */}
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />

      {/* Main Content Area */}
      <main className="flex-1 min-w-0 overflow-y-auto px-4 sm:px-8 py-8">
        <div className="max-w-7xl mx-auto">
          {activeTab === "dashboard" && (
            <Dashboard onSelectCase={handleSelectCase} />
          )}

          {activeTab === "cases" && (
            <RecoveryCase
              currentCase={selectedCase}
              onSelectCase={setSelectedCase}
            />
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
