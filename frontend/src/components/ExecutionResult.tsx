import React, { useState } from "react";
import {
  CheckCircle,
  Copy,
  Check,
  ExternalLink,
  ShieldCheck,
  Zap,
  Info,
  AlertCircle,
} from "lucide-react";
import { ActionExecutionResultDTO } from "../types/recovery";

interface ExecutionResultProps {
  execution: ActionExecutionResultDTO;
}

export const ExecutionResult: React.FC<ExecutionResultProps> = ({ execution }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    if (execution.resource_url) {
      navigator.clipboard.writeText(execution.resource_url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const isSimulated = execution.status === "simulated" || execution.reference_id?.startsWith("plink_sim_");

  return (
    <div className="rounded-2xl bg-gradient-to-b from-slate-900 to-slate-950 border border-emerald-500/30 p-6 shadow-2xl space-y-5 animate-slide-up">
      {/* Header Banner */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-800">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
            <Zap className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-extrabold text-lg text-white flex items-center gap-2">
              <span>Action Executed via Gateway Adapter</span>
              <CheckCircle className="w-5 h-5 text-emerald-400" />
            </h3>
            <p className="text-xs text-slate-400">
              Dispatched through ActionExecutor → RazorpayAdapter
            </p>
          </div>
        </div>

        {/* Mode indicator */}
        {isSimulated ? (
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-950/80 text-amber-300 border border-amber-800/80 text-xs font-semibold">
            <Info className="w-3.5 h-3.5 text-amber-400" />
            <span>Demo Mode · Razorpay execution simulated</span>
          </div>
        ) : (
          <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-950/80 text-emerald-300 border border-emerald-800/80 text-xs font-semibold">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            <span>Live Razorpay Execution</span>
          </div>
        )}
      </div>

      {/* Outcome Message */}
      <div className="p-4 rounded-xl bg-slate-900/90 border border-slate-800 space-y-1">
        <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
          Execution Outcome
        </div>
        <div className="text-sm font-medium text-slate-100">{execution.message}</div>
      </div>

      {/* Payment Link Card (if payment link was generated) */}
      {execution.resource_url && (
        <div className="p-5 rounded-xl bg-emerald-950/20 border border-emerald-500/30 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-emerald-300 uppercase tracking-wider">
              Customer Payment Recovery Link
            </span>
            <span className="text-[11px] font-mono text-emerald-400 bg-emerald-950/80 px-2 py-0.5 rounded border border-emerald-800">
              Interactive 2FA Ready
            </span>
          </div>

          <div className="flex items-center gap-2">
            <input
              type="text"
              readOnly
              value={execution.resource_url}
              className="flex-1 bg-slate-950 border border-slate-700/80 text-slate-200 text-xs font-mono px-3 py-2.5 rounded-lg focus:outline-none"
            />
            <button
              onClick={handleCopy}
              className="px-3.5 py-2.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold transition-all flex items-center gap-1.5 cursor-pointer border border-slate-700"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
              <span>{copied ? "Copied" : "Copy"}</span>
            </button>
            <a
              href={execution.resource_url}
              target="_blank"
              rel="noopener noreferrer"
              className="px-3.5 py-2.5 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-slate-950 text-xs font-bold transition-all flex items-center gap-1.5 shadow-md shadow-emerald-500/20"
            >
              <span>Open</span>
              <ExternalLink className="w-3.5 h-3.5" />
            </a>
          </div>
        </div>
      )}

      {/* Metadata Telemetry */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
        <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
          <span className="text-slate-400 font-medium block">Gateway Reference ID</span>
          <span className="font-mono font-semibold text-slate-200 mt-1 block truncate">
            {execution.reference_id || "None"}
          </span>
        </div>

        <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
          <span className="text-slate-400 font-medium block">Action Status</span>
          <span className="font-semibold text-emerald-400 capitalize mt-1 block">
            {execution.status}
          </span>
        </div>

        <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
          <span className="text-slate-400 font-medium block">External Attempt</span>
          <span className="font-semibold text-slate-200 mt-1 block">
            {execution.attempted ? "Attempted via Adapter" : "Skipped (Safety Protected)"}
          </span>
        </div>
      </div>

      {/* Diagnostics / Error (if failed) */}
      {execution.error && (
        <div className="p-3 rounded-xl bg-rose-950/30 border border-rose-900/50 text-xs text-rose-300 flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
          <span>{execution.error}</span>
        </div>
      )}
    </div>
  );
};
