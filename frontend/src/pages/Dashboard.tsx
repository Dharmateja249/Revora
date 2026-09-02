import React from "react";
import {
  TrendingUp,
  AlertTriangle,
  CheckCircle2,
  Activity,
  ArrowRight,
  ShieldCheck,
  Zap,
} from "lucide-react";
import { MetricCard } from "../components/MetricCard";
import { PaymentCaseCard } from "../components/PaymentCaseCard";
import { RecentCases } from "../components/RecentCases";
import { DEMO_PAYMENT_CASES } from "../data/demoCases";
import { DemoPaymentCase } from "../types/recovery";

interface DashboardProps {
  onSelectCase: (c: DemoPaymentCase) => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ onSelectCase }) => {
  const featuredCase = DEMO_PAYMENT_CASES[0]; // 2FA Card failure case

  return (
    <div className="space-y-8 pb-12 animate-fade-in">
      {/* Top Header & Context Banner */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight">
            Revenue Recovery Command Center
          </h1>
          <p className="text-sm text-slate-400 mt-1">
            Real-time telemetry, agent intelligence, and policy-governed payment recovery
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="px-3.5 py-1.5 rounded-full bg-emerald-950/60 text-emerald-400 border border-emerald-800/60 text-xs font-semibold flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
            <span>Autonomous Recovery Agent Active</span>
          </div>
        </div>
      </div>

      {/* Safety Philosophy Strip */}
      <div className="p-4 rounded-2xl bg-gradient-to-r from-slate-900 via-slate-900/90 to-slate-950 border border-slate-800 flex flex-wrap items-center justify-between gap-4 text-xs">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400">
            <ShieldCheck className="w-4 h-4" />
          </div>
          <div>
            <span className="font-bold text-white text-sm">Strict Safety Invariant:</span>{" "}
            <span className="text-slate-300">
              AI recommends optimal strategy · PolicyValidator enforces Razorpay 2FA/limits · Approved actions execute.
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2 text-slate-400 font-mono text-[11px]">
          <Zap className="w-3.5 h-3.5 text-amber-400" />
          <span>Opt-in execution model</span>
        </div>
      </div>

      {/* Main KPI Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Revenue at Risk"
          value="₹1,00,648"
          subtext="5 active failed payments"
          change="+12.4% today"
          isPositive={false}
          icon={AlertTriangle}
          accentColor="amber"
        />
        <MetricCard
          label="Autonomous Recovery Rate"
          value="74.2%"
          subtext="Empirical benchmark"
          change="+4.8% vs baseline"
          isPositive={true}
          icon={TrendingUp}
          accentColor="emerald"
        />
        <MetricCard
          label="Net Revenue Recovered"
          value="₹74,680"
          subtext="Saved through policy actions"
          change="+₹18,500 today"
          isPositive={true}
          icon={CheckCircle2}
          accentColor="blue"
        />
        <MetricCard
          label="Total Interventions"
          value="48 Cases"
          subtext="99.8% policy compliance"
          change="0 violations"
          isPositive={true}
          icon={Activity}
          accentColor="indigo"
        />
      </div>

      {/* Featured Failed Payment Section */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-amber-400 animate-ping" />
            <h2 className="text-base font-bold text-white uppercase tracking-wider">
              High Priority Intercept Opportunity
            </h2>
          </div>
          <button
            onClick={() => onSelectCase(featuredCase)}
            className="text-xs font-semibold text-emerald-400 hover:text-emerald-300 flex items-center gap-1 transition-colors cursor-pointer"
          >
            <span>Open in Recovery Console</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <PaymentCaseCard
          paymentCase={featuredCase}
          isHero={true}
          onAnalyze={() => onSelectCase(featuredCase)}
        />
      </div>

      {/* Recent Recovery Cases */}
      <RecentCases
        cases={DEMO_PAYMENT_CASES}
        onSelectCase={(selectedCase) => onSelectCase(selectedCase)}
      />
    </div>
  );
};
