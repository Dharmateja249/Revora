import React, { useEffect, useState } from "react";
import {
  LayoutDashboard,
  Zap,
  BarChart3,
  Settings,
  ShieldCheck,
  CheckCircle2,
  AlertCircle,
  Cpu,
  RefreshCw,
} from "lucide-react";
import { checkBackendHealth } from "../api/recovery";

interface SidebarProps {
  activeTab: "dashboard" | "cases";
  onTabChange: (tab: "dashboard" | "cases") => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, onTabChange }) => {
  const [backendStatus, setBackendStatus] = useState<"checking" | "online" | "offline">("checking");
  const [backendVersion, setBackendVersion] = useState<string>("0.1.0");

  useEffect(() => {
    let mounted = true;
    const verifyHealth = async () => {
      try {
        const health = await checkBackendHealth();
        if (mounted) {
          setBackendStatus("online");
          if (health.version) setBackendVersion(health.version);
        }
      } catch {
        if (mounted) setBackendStatus("offline");
      }
    };

    verifyHealth();
    const interval = setInterval(verifyHealth, 15000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <aside className="w-64 bg-slate-950 border-r border-slate-800/80 flex flex-col justify-between h-screen sticky top-0 select-none">
      {/* Brand & System Status */}
      <div className="p-5 space-y-6">
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-emerald-400 to-teal-600 flex items-center justify-center shadow-lg shadow-emerald-500/20 text-slate-950 font-black text-xl tracking-tighter">
              R
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-lg tracking-tight text-white">Revora</span>
                <span className="text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded bg-emerald-950/80 text-emerald-400 border border-emerald-800/60">
                  Agent
                </span>
              </div>
              <p className="text-[11px] font-medium text-slate-400">AI Revenue Recovery</p>
            </div>
          </div>
        </div>

        {/* Live System Health Badge */}
        <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800/80 space-y-1.5">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400 font-medium flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-slate-400" />
              Decision Engine
            </span>
            {backendStatus === "checking" && (
              <span className="text-slate-500 text-[11px] flex items-center gap-1">
                <RefreshCw className="w-3 h-3 animate-spin" /> Checking
              </span>
            )}
            {backendStatus === "online" && (
              <span className="text-emerald-400 text-[11px] font-semibold flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
                Live v{backendVersion}
              </span>
            )}
            {backendStatus === "offline" && (
              <span className="text-rose-400 text-[11px] font-semibold flex items-center gap-1">
                <AlertCircle className="w-3 h-3" /> Offline (Port 8000)
              </span>
            )}
          </div>

          <div className="flex items-center justify-between text-[11px] pt-1 border-t border-slate-800/60 text-slate-400">
            <span>Gateway Mode</span>
            <span className="text-amber-400 font-medium bg-amber-950/40 px-1.5 py-0.2 rounded border border-amber-900/50">
              Razorpay Dry-Run
            </span>
          </div>
        </div>

        {/* Navigation Menu */}
        <nav className="space-y-1">
          <button
            onClick={() => onTabChange("dashboard")}
            className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all ${
              activeTab === "dashboard"
                ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 shadow-sm"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/60"
            }`}
          >
            <LayoutDashboard className="w-4 h-4" />
            <span>Dashboard</span>
          </button>

          <button
            onClick={() => onTabChange("cases")}
            className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-xl text-sm font-medium transition-all ${
              activeTab === "cases"
                ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 shadow-sm"
                : "text-slate-400 hover:text-slate-200 hover:bg-slate-900/60"
            }`}
          >
            <Zap className="w-4 h-4" />
            <span>Recovery Console</span>
          </button>

          <div className="pt-2">
            <div className="px-3 pb-1 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
              Operations
            </div>

            <div className="flex items-center justify-between px-3.5 py-2 rounded-xl text-sm font-medium text-slate-500 cursor-not-allowed">
              <span className="flex items-center gap-3">
                <BarChart3 className="w-4 h-4" />
                <span>Analytics</span>
              </span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-900 text-slate-500 border border-slate-800">
                Soon
              </span>
            </div>

            <div className="flex items-center justify-between px-3.5 py-2 rounded-xl text-sm font-medium text-slate-500 cursor-not-allowed">
              <span className="flex items-center gap-3">
                <Settings className="w-4 h-4" />
                <span>Policies & Rules</span>
              </span>
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-slate-900 text-slate-500 border border-slate-800">
                Soon
              </span>
            </div>
          </div>
        </nav>
      </div>

      {/* Safety & Invariants Footer */}
      <div className="p-4 border-t border-slate-900 bg-slate-950/80 space-y-3">
        <div className="p-3 rounded-xl bg-slate-900/90 border border-slate-800/80 space-y-1.5">
          <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-200">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            <span>Safety Architecture</span>
          </div>
          <p className="text-[11px] leading-relaxed text-slate-400">
            <strong className="text-slate-300">AI recommends.</strong><br />
            <strong className="text-emerald-400">Policy validates.</strong><br />
            Approved actions execute explicitly.
          </p>
          <div className="pt-1 flex items-center gap-1.5 text-[10px] text-slate-400 font-medium">
            <CheckCircle2 className="w-3 h-3 text-emerald-500" />
            <span>Opt-in execution required</span>
          </div>
        </div>
      </div>
    </aside>
  );
};
