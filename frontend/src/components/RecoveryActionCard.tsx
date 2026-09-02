import React from "react";
import {
  ShieldCheck,
  Sparkles,
  Link,
  RotateCw,
  RefreshCw,
  AlertTriangle,
  FileCheck,
  CheckCircle2,
  Lock,
  ArrowRight,
  ExternalLink,
} from "lucide-react";
import { RecoveryDecisionResponse } from "../types/recovery";

interface RecoveryActionCardProps {
  decision: RecoveryDecisionResponse;
  onExecute?: () => void;
  isExecuting?: boolean;
}

export const RecoveryActionCard: React.FC<RecoveryActionCardProps> = ({
  decision,
  onExecute,
  isExecuting = false,
}) => {
  const confidencePercent = Math.round(decision.confidence * 100);

  const getActionTheme = (action: string) => {
    switch (action.toLowerCase()) {
      case "payment_link":
        return {
          label: "PAYMENT LINK",
          icon: Link,
          color: "text-emerald-400 bg-emerald-950/60 border-emerald-500/30",
          badge: "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
          executable: true,
        };
      case "change_payment_method":
        return {
          label: "CHANGE PAYMENT METHOD",
          icon: RefreshCw,
          color: "text-teal-400 bg-teal-950/60 border-teal-500/30",
          badge: "bg-teal-500/20 text-teal-300 border-teal-500/40",
          executable: true,
        };
      case "retry_payment":
        return {
          label: "RETRY PAYMENT",
          icon: RotateCw,
          color: "text-blue-400 bg-blue-950/60 border-blue-500/30",
          badge: "bg-blue-500/20 text-blue-300 border-blue-500/40",
          executable: true,
        };
      case "wait_and_retry":
        return {
          label: "WAIT AND RETRY",
          icon: RotateCw,
          color: "text-indigo-400 bg-indigo-950/60 border-indigo-500/30",
          badge: "bg-indigo-500/20 text-indigo-300 border-indigo-500/40",
          executable: false,
        };
      case "escalate_support":
        return {
          label: "ESCALATE TO SUPPORT",
          icon: AlertTriangle,
          color: "text-amber-400 bg-amber-950/60 border-amber-500/30",
          badge: "bg-amber-500/20 text-amber-300 border-amber-500/40",
          executable: false,
        };
      default:
        return {
          label: decision.recommended_action.toUpperCase(),
          icon: ShieldCheck,
          color: "text-slate-400 bg-slate-900 border-slate-700",
          badge: "bg-slate-800 text-slate-300 border-slate-700",
          executable: false,
        };
    }
  };

  const actionTheme = getActionTheme(decision.recommended_action);
  const ActionIcon = actionTheme.icon;

  return (
    <div className="rounded-2xl bg-slate-900/80 border border-slate-800 p-6 shadow-2xl space-y-6">
      {/* Header Banner: Policy Verification & AI Invariant */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-400">
            <Sparkles className="w-4 h-4" />
          </div>
          <div>
            <h3 className="font-bold text-base text-white">Revora Agent Intelligence</h3>
            <p className="text-xs text-slate-400">
              {decision.agent_used ? "LLM synthesis grounded in empirical RAG context" : "Deterministic baseline fallback"}
            </p>
          </div>
        </div>

        {/* Policy Status Badge */}
        <div className="flex items-center gap-2">
          {decision.policy_overridden ? (
            <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-950/80 text-amber-300 border border-amber-800/80 text-xs font-semibold">
              <Lock className="w-3.5 h-3.5" />
              Policy Overridden for Safety
            </span>
          ) : (
            <span className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-950/80 text-emerald-300 border border-emerald-800/80 text-xs font-semibold">
              <ShieldCheck className="w-3.5 h-3.5" />
              PolicyValidator Approved
            </span>
          )}
        </div>
      </div>

      {/* Hero Decision Section */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {/* Recommended Action Tile */}
        <div className={`col-span-2 rounded-xl p-5 border ${actionTheme.color} flex flex-col justify-between`}>
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold uppercase tracking-wider text-slate-300">
                Recommended Recovery Action
              </span>
              <span className={`text-[11px] font-extrabold uppercase px-2 py-0.5 rounded border ${actionTheme.badge}`}>
                {actionTheme.label}
              </span>
            </div>
            <div className="flex items-center gap-3 pt-1">
              <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800">
                <ActionIcon className="w-6 h-6" />
              </div>
              <div>
                <h2 className="text-2xl font-black text-white tracking-tight">
                  {actionTheme.label}
                </h2>
                <p className="text-xs text-slate-300 mt-0.5">
                  Automated recovery pathway validated through compliance rules
                </p>
              </div>
            </div>
          </div>

          {/* Policy Override Explanation Alert */}
          {decision.policy_overridden && (
            <div className="mt-4 p-3 rounded-lg bg-amber-950/50 border border-amber-800/60 text-xs text-amber-200 flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <div>
                <strong className="font-semibold">Safety Override Active:</strong> Direct automated retry was prohibited by PolicyValidator (e.g. 2FA customer presence required). Safe interactive action was enforced.
              </div>
            </div>
          )}

          {/* Fallback explanation if triggered */}
          {decision.is_fallback && (
            <div className="mt-4 p-3 rounded-lg bg-slate-950/60 border border-slate-800 text-xs text-slate-300 flex items-start gap-2">
              <AlertTriangle className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
              <div>
                <strong className="font-semibold">Deterministic Fallback:</strong> {decision.fallback_reason || "Rule-based baseline applied."}
              </div>
            </div>
          )}
        </div>

        {/* Confidence Gauge */}
        <div className="rounded-xl p-5 bg-slate-950/60 border border-slate-800 flex flex-col justify-between">
          <div>
            <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
              Decision Confidence
            </div>
            <div className="text-3xl font-black text-emerald-400 mt-2">
              {confidencePercent}%
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Empirical recovery probability derived from RAG vectors and domain models.
            </p>
          </div>

          <div className="space-y-1.5 pt-4">
            <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-emerald-500 to-teal-400 rounded-full transition-all duration-500"
                style={{ width: `${confidencePercent}%` }}
              />
            </div>
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>0% Baseline</span>
              <span>100% High Certainty</span>
            </div>
          </div>
        </div>
      </div>

      {/* Prominent LLM Reasoning Section */}
      <div className="p-5 rounded-xl bg-slate-950/70 border border-slate-800 space-y-2.5">
        <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-slate-300">
          <FileCheck className="w-4 h-4 text-emerald-400" />
          <span>Synthesized Agent Reasoning</span>
        </div>
        <p className="text-sm text-slate-200 leading-relaxed font-sans bg-slate-900/60 p-3.5 rounded-lg border border-slate-800/80">
          "{decision.reasoning}"
        </p>
      </div>

      {/* Key Factors & Historical Case IDs */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Key Signals */}
        <div className="p-4 rounded-xl bg-slate-950/50 border border-slate-800 space-y-2">
          <div className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
            Key Diagnostic Signals
          </div>
          <div className="flex flex-wrap gap-2 pt-1">
            {decision.key_factors.length > 0 ? (
              decision.key_factors.map((factor, idx) => (
                <span
                  key={idx}
                  className="text-xs font-mono px-2.5 py-1 rounded-md bg-slate-900 text-emerald-300 border border-slate-800"
                >
                  • {factor}
                </span>
              ))
            ) : (
              <span className="text-xs text-slate-500">Standard failure telemetry</span>
            )}
          </div>
        </div>

        {/* Historical References */}
        <div className="p-4 rounded-xl bg-slate-950/50 border border-slate-800 space-y-2">
          <div className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
            Referenced Historical Cases
          </div>
          <div className="flex flex-wrap gap-2 pt-1">
            {decision.referenced_case_ids.length > 0 ? (
              decision.referenced_case_ids.map((caseId, idx) => (
                <span
                  key={idx}
                  className="text-xs font-mono px-2.5 py-1 rounded-md bg-slate-900 text-blue-300 border border-slate-800 flex items-center gap-1"
                >
                  <ExternalLink className="w-3 h-3" />
                  {caseId}
                </span>
              ))
            ) : (
              <span className="text-xs text-slate-500">No relevant historical precedents</span>
            )}
          </div>
        </div>
      </div>

      {/* Execution Call to Action: Explicit opt-in execution */}
      {onExecute && (
        <div className="pt-4 border-t border-slate-800 flex flex-wrap items-center justify-between gap-4">
          <div className="text-xs text-slate-400 flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            <span>
              Decision is validated. Execution requires explicit confirmation (safety invariant).
            </span>
          </div>

          <button
            onClick={onExecute}
            disabled={isExecuting}
            className={`flex items-center gap-2 px-6 py-3 rounded-xl font-bold text-sm shadow-xl transition-all cursor-pointer ${
              isExecuting
                ? "bg-slate-800 text-slate-400 cursor-not-allowed"
                : "bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 text-slate-950 shadow-emerald-500/20 hover:shadow-emerald-500/30"
            }`}
          >
            {isExecuting ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Dispatching to Razorpay...</span>
              </>
            ) : (
              <>
                <span>Execute Recovery</span>
                <ArrowRight className="w-4 h-4" />
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
};
