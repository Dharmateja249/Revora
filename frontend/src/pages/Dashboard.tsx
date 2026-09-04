import React, { useEffect, useState } from "react";
import {
  TrendingUp,
  AlertTriangle,
  CheckCircle2,
  Activity,
  ArrowRight,
  ShieldCheck,
  Zap,
  RefreshCw,
  Database,
  Lock,
  Cpu,
  HelpCircle,
} from "lucide-react";
import { fetchDashboardMetrics } from "../api/dashboard";
import { evaluateRecoveryDecision } from "../api/recovery";
import { MetricCard } from "../components/MetricCard";
import { PaymentCaseCard } from "../components/PaymentCaseCard";
import { RecentCases } from "../components/RecentCases";
import { DEMO_PAYMENT_CASES } from "../data/demoCases";
import { DashboardMetrics, DemoPaymentCase } from "../types/recovery";

interface DashboardProps {
  onSelectCase: (c: DemoPaymentCase) => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ onSelectCase }) => {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [featuredCase, setFeaturedCase] = useState<DemoPaymentCase>(DEMO_PAYMENT_CASES[0]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const loadMetrics = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchDashboardMetrics();
      setMetrics(data);
    } catch (err: unknown) {
      setError(
        err instanceof Error
          ? err.message
          : "Failed to load live dashboard telemetry."
      );
    } finally {
      setLoading(false);
    }

    // Non-intrusive hydration for featured payment case
    evaluateRecoveryDecision({
      ...DEMO_PAYMENT_CASES[0].requestPayload,
      execute_action: false,
    })
      .then((res) => {
        setFeaturedCase((prev) => ({
          ...prev,
          attempt_count: res.attempt_count ?? res.previous_attempts?.length,
          requestPayload: {
            ...prev.requestPayload,
            payment_id: res.payment_id || prev.requestPayload.payment_id,
            opportunity_status: res.opportunity_status || prev.requestPayload.opportunity_status,
            previous_attempts: res.previous_attempts || prev.requestPayload.previous_attempts,
            customer: res.customer
              ? {
                  customer_id: res.customer.customer_id || prev.requestPayload.customer?.customer_id,
                  total_payments: res.customer.total_payments,
                  successful_payments: res.customer.successful_payments,
                  failed_payments: res.customer.failed_payments,
                  historical_success_rate: res.customer.historical_success_rate,
                }
              : prev.requestPayload.customer,
          },
        }));
      })
      .catch(() => {
        // Non-fatal hydration fallback
      });
  };

  useEffect(() => {
    loadMetrics();
  }, []);

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
          <button
            onClick={loadMetrics}
            disabled={loading}
            className="px-3 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700/80 text-slate-300 text-xs font-semibold flex items-center gap-1.5 transition-colors border border-slate-700/60 cursor-pointer disabled:opacity-50"
            title="Refresh dashboard metrics"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
            <span>Refresh</span>
          </button>
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

      {/* Error Alert */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-950/60 border border-rose-800/80 text-rose-300 text-xs flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-rose-400 shrink-0" />
            <span>{error}</span>
          </div>
          <button
            onClick={loadMetrics}
            className="font-bold underline hover:text-rose-100 cursor-pointer"
          >
            Retry
          </button>
        </div>
      )}

      {/* Primary KPI Metric Cards (Calculated from Real Database State) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Recovery Rate"
          value={loading ? "..." : `${metrics ? metrics.recovery_rate.toFixed(1) : "0.0"}%`}
          subtext={
            metrics
              ? `${metrics.recovered_cases} of ${metrics.total_cases} cases recovered`
              : "0 of 0 cases recovered"
          }
          isPositive={metrics ? metrics.recovery_rate >= 50 : true}
          icon={TrendingUp}
          accentColor="emerald"
        />
        <MetricCard
          label="Net Revenue Recovered"
          value={
            loading
              ? "..."
              : `₹${metrics ? metrics.amount_recovered.toLocaleString("en-IN") : "0"}`
          }
          subtext="Saved through policy actions"
          isPositive={true}
          icon={CheckCircle2}
          accentColor="blue"
        />
        <MetricCard
          label="Recovered Cases"
          value={loading ? "..." : `${metrics ? metrics.recovered_cases : "0"}`}
          subtext="Successfully resolved"
          isPositive={true}
          icon={Activity}
          accentColor="indigo"
        />
        <MetricCard
          label="Failed Cases"
          value={loading ? "..." : `${metrics ? metrics.failed_cases : "0"}`}
          subtext={
            metrics && metrics.pending_cases > 0
              ? `${metrics.pending_cases} active in recovery`
              : "Terminal / max attempts"
          }
          isPositive={metrics ? metrics.failed_cases === 0 : true}
          icon={AlertTriangle}
          accentColor={metrics && metrics.failed_cases > 0 ? "rose" : "amber"}
        />
      </div>

      {/* Secondary AI & Operational Telemetry Cards */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
            <Cpu className="w-3.5 h-3.5 text-emerald-400" />
            <span>AI Operational & Gateway Telemetry</span>
          </h2>
          <span className="text-[11px] text-slate-500 font-mono">
            Live database state & runtime VectorIndex
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <div className="flex items-center justify-between text-slate-400 text-xs">
              <span>Execution Success Rate</span>
              <Activity className="w-3.5 h-3.5 text-teal-400" />
            </div>
            <p className="text-xl font-bold text-white">
              {loading
                ? "..."
                : `${metrics ? metrics.execution_success_rate.toFixed(1) : "0.0"}%`}
            </p>
            <p className="text-[11px] text-slate-500">
              {metrics
                ? `${metrics.successful_executions}/${metrics.total_executions} gateway dispatches`
                : "0 executions"}
            </p>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <div className="flex items-center justify-between text-slate-400 text-xs">
              <span>Policy Overrides</span>
              <Lock className="w-3.5 h-3.5 text-amber-400" />
            </div>
            <p className="text-xl font-bold text-white">
              {loading ? "..." : `${metrics ? metrics.policy_overrides : "0"}`}
            </p>
            <p className="text-[11px] text-slate-500">Enforced by PolicyValidator</p>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <div className="flex items-center justify-between text-slate-400 text-xs">
              <span>RAG Precedents</span>
              <Database className="w-3.5 h-3.5 text-blue-400" />
            </div>
            <p className="text-xl font-bold text-white">
              {loading ? "..." : `${metrics ? metrics.rag_precedents : "0"}`}
            </p>
            <p className="text-[11px] text-slate-500">Runtime VectorIndex documents</p>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <div className="flex items-center justify-between text-slate-400 text-xs">
              <span>Average AI Confidence</span>
              <TrendingUp className="w-3.5 h-3.5 text-emerald-400" />
            </div>
            <p className="text-xl font-bold text-white">
              {loading
                ? "..."
                : `${metrics ? (metrics.average_confidence * 100).toFixed(0) : "0"}%`}
            </p>
            <p className="text-[11px] text-slate-500">Recommendation confidence</p>
          </div>

          <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 space-y-1">
            <div className="flex items-center justify-between text-slate-400 text-xs">
              <span>Deterministic Fallbacks</span>
              <HelpCircle className="w-3.5 h-3.5 text-slate-400" />
            </div>
            <p className="text-xl font-bold text-white">
              {loading ? "..." : `${metrics ? metrics.fallback_decisions : "0"}`}
            </p>
            <p className="text-[11px] text-slate-500">Provider safety fallbacks</p>
          </div>
        </div>
      </div>

      {/* Recovery Outcomes & Execution Breakdown Section */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Recovery Outcomes */}
        <div className="p-5 rounded-2xl bg-slate-900/70 border border-slate-800 space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800">
            <h3 className="text-sm font-bold text-white">Recovery Case Outcomes</h3>
            <span className="text-xs font-mono text-slate-400">
              Total: {metrics ? metrics.total_cases : 0}
            </span>
          </div>
          <div className="grid grid-cols-3 gap-3 text-center">
            <div className="p-3 rounded-xl bg-emerald-950/40 border border-emerald-800/40">
              <span className="text-[11px] text-emerald-400 font-semibold uppercase">Recovered</span>
              <p className="text-lg font-extrabold text-white mt-1">
                {metrics ? metrics.recovered_cases : 0}
              </p>
            </div>
            <div className="p-3 rounded-xl bg-rose-950/40 border border-rose-800/40">
              <span className="text-[11px] text-rose-400 font-semibold uppercase">Failed</span>
              <p className="text-lg font-extrabold text-white mt-1">
                {metrics ? metrics.failed_cases : 0}
              </p>
            </div>
            <div className="p-3 rounded-xl bg-amber-950/40 border border-amber-800/40">
              <span className="text-[11px] text-amber-400 font-semibold uppercase">Pending</span>
              <p className="text-lg font-extrabold text-white mt-1">
                {metrics ? metrics.pending_cases : 0}
              </p>
            </div>
          </div>
        </div>

        {/* Gateway Execution Status */}
        <div className="p-5 rounded-2xl bg-slate-900/70 border border-slate-800 space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800">
            <h3 className="text-sm font-bold text-white">Gateway Execution Status</h3>
            <span className="text-xs font-mono text-slate-400">
              Total: {metrics ? metrics.total_executions : 0}
            </span>
          </div>
          <div className="grid grid-cols-3 gap-3 text-center">
            <div className="p-3 rounded-xl bg-teal-950/40 border border-teal-800/40">
              <span className="text-[11px] text-teal-400 font-semibold uppercase">Successful</span>
              <p className="text-lg font-extrabold text-white mt-1">
                {metrics ? metrics.successful_executions : 0}
              </p>
            </div>
            <div className="p-3 rounded-xl bg-rose-950/40 border border-rose-800/40">
              <span className="text-[11px] text-rose-400 font-semibold uppercase">Failed</span>
              <p className="text-lg font-extrabold text-white mt-1">
                {metrics ? metrics.failed_executions : 0}
              </p>
            </div>
            <div className="p-3 rounded-xl bg-slate-800/60 border border-slate-700/60">
              <span className="text-[11px] text-slate-400 font-semibold uppercase">In Progress</span>
              <p className="text-lg font-extrabold text-white mt-1">
                {metrics ? metrics.pending_executions : 0}
              </p>
            </div>
          </div>
        </div>
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
