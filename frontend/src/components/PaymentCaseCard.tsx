import React from "react";
import { CreditCard, Smartphone, ShieldAlert, History, AlertTriangle, ArrowRight, DollarSign } from "lucide-react";
import { DemoPaymentCase } from "../types/recovery";

interface PaymentCaseCardProps {
  paymentCase: DemoPaymentCase;
  onAnalyze?: () => void;
  isHero?: boolean;
}

export const PaymentCaseCard: React.FC<PaymentCaseCardProps> = ({
  paymentCase,
  onAnalyze,
  isHero = false,
}) => {
  const req = paymentCase.requestPayload;
  const successRate = Math.round((req.customer?.historical_success_rate || 0) * 100);

  const getMethodIcon = (method: string) => {
    switch (method.toLowerCase()) {
      case "upi":
        return <Smartphone className="w-4 h-4 text-emerald-400" />;
      case "card":
        return <CreditCard className="w-4 h-4 text-blue-400" />;
      default:
        return <DollarSign className="w-4 h-4 text-indigo-400" />;
    }
  };

  return (
    <div
      className={`rounded-2xl border transition-all ${
        isHero
          ? "bg-slate-900/90 border-slate-700/80 shadow-2xl shadow-emerald-500/5 p-6"
          : "bg-slate-900/60 border-slate-800 p-5 hover:border-slate-700"
      }`}
    >
      {/* Top Bar: Customer & Badges */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 flex-wrap">
            <h3 className={`font-bold text-white ${isHero ? "text-lg" : "text-base"}`}>
              {paymentCase.customerName}
            </h3>
            <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
              {paymentCase.customerTier}
            </span>
            <span className="text-[11px] font-medium text-slate-400">
              {paymentCase.timestamp}
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">{paymentCase.customerEmail}</p>
        </div>

        <div className="text-right">
          <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Revenue at Risk
          </div>
          <div className="text-2xl font-black text-white tracking-tight">
            ₹{req.amount.toLocaleString("en-IN", { minimumFractionDigits: 2 })}
          </div>
        </div>
      </div>

      {/* Description */}
      <p className="text-xs text-slate-300 leading-relaxed mt-3.5 pt-3 border-t border-slate-800/80">
        {paymentCase.description}
      </p>

      {/* Diagnostic Metadata Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-4">
        <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80">
          <div className="text-[11px] text-slate-400 font-medium">Payment Method</div>
          <div className="flex items-center gap-1.5 mt-1 font-semibold text-xs text-slate-200 uppercase">
            {getMethodIcon(req.payment_method)}
            <span>{req.payment_method}</span>
          </div>
        </div>

        <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80">
          <div className="text-[11px] text-slate-400 font-medium">Failure Reason</div>
          <div className="flex items-center gap-1.5 mt-1 text-xs font-mono font-medium text-amber-300 truncate" title={req.failure_reason}>
            <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
            <span className="truncate">{req.failure_reason}</span>
          </div>
        </div>

        <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80">
          <div className="text-[11px] text-slate-400 font-medium">Customer Trust Score</div>
          <div className="flex items-center gap-2 mt-1">
            <span className="text-xs font-bold text-emerald-400">{successRate}%</span>
            <div className="flex-1 h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-emerald-400 rounded-full"
                style={{ width: `${successRate}%` }}
              />
            </div>
          </div>
        </div>

        <div className="p-3 rounded-xl bg-slate-950/60 border border-slate-800/80">
          <div className="text-[11px] text-slate-400 font-medium">Attempt Budget</div>
          <div className="flex items-center gap-1.5 mt-1 text-xs font-semibold text-slate-200">
            <History className="w-3.5 h-3.5 text-slate-400" />
            <span>
              {req.previous_attempts?.length || 0} / {req.max_attempts || 3} used
            </span>
          </div>
        </div>
      </div>

      {/* Prior Attempts List (if any) */}
      {req.previous_attempts && req.previous_attempts.length > 0 && (
        <div className="mt-3.5 p-3 rounded-xl bg-rose-950/20 border border-rose-900/40 text-xs text-rose-300 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ShieldAlert className="w-4 h-4 text-rose-400 shrink-0" />
            <span>
              Prior Attempt Failed: <strong className="font-mono">{req.previous_attempts[0].action}</strong> ({req.previous_attempts[0].error_code || "FAILED"})
            </span>
          </div>
          <span className="text-[11px] font-semibold text-rose-400">Escalation Ready</span>
        </div>
      )}

      {/* Call to Action Button */}
      {onAnalyze && (
        <div className="mt-5 flex items-center justify-end">
          <button
            onClick={onAnalyze}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-400 hover:to-teal-500 text-slate-950 font-bold text-sm shadow-lg shadow-emerald-500/20 hover:shadow-emerald-500/30 transition-all cursor-pointer group"
          >
            <span>Analyze with Revora</span>
            <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
          </button>
        </div>
      )}
    </div>
  );
};
