import React from "react";
import { ArrowRight, Smartphone, CreditCard, DollarSign } from "lucide-react";
import { DemoPaymentCase } from "../types/recovery";

interface RecentCasesProps {
  cases: DemoPaymentCase[];
  onSelectCase: (paymentCase: DemoPaymentCase) => void;
}

export const RecentCases: React.FC<RecentCasesProps> = ({ cases, onSelectCase }) => {
  const getMethodIcon = (method: string) => {
    switch (method.toLowerCase()) {
      case "upi":
        return <Smartphone className="w-3.5 h-3.5 text-emerald-400" />;
      case "card":
        return <CreditCard className="w-3.5 h-3.5 text-blue-400" />;
      default:
        return <DollarSign className="w-3.5 h-3.5 text-indigo-400" />;
    }
  };

  return (
    <div className="rounded-2xl bg-slate-900/70 border border-slate-800 p-6 shadow-xl space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="font-bold text-base text-white">Recent Payment Failure Events</h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Incoming failures streamed for autonomous agent triage and policy-validated recovery
          </p>
        </div>
        <span className="text-xs font-medium text-slate-400">
          Showing {cases.length} active scenarios
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-slate-800 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
              <th className="py-3 px-3">Customer</th>
              <th className="py-3 px-3">Amount</th>
              <th className="py-3 px-3">Method</th>
              <th className="py-3 px-3">Failure Reason</th>
              <th className="py-3 px-3">Age</th>
              <th className="py-3 px-3 text-right">Action</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60 text-xs">
            {cases.map((c) => {
              const req = c.requestPayload;
              return (
                <tr
                  key={c.id}
                  onClick={() => onSelectCase(c)}
                  className="hover:bg-slate-850/50 transition-colors cursor-pointer group"
                >
                  <td className="py-3.5 px-3">
                    <div className="font-semibold text-slate-100 group-hover:text-emerald-400 transition-colors">
                      {c.customerName}
                    </div>
                    <div className="text-[11px] text-slate-400 font-mono">{c.customerEmail}</div>
                  </td>
                  <td className="py-3.5 px-3 font-mono font-bold text-slate-100">
                    ₹{req.amount.toLocaleString("en-IN", { minimumFractionDigits: 2 })}
                  </td>
                  <td className="py-3.5 px-3">
                    <div className="flex items-center gap-1.5 uppercase font-medium text-slate-300">
                      {getMethodIcon(req.payment_method)}
                      <span>{req.payment_method}</span>
                    </div>
                  </td>
                  <td className="py-3.5 px-3">
                    <span className="font-mono text-[11px] px-2 py-0.5 rounded bg-slate-950 text-amber-300 border border-slate-800">
                      {req.failure_reason}
                    </span>
                  </td>
                  <td className="py-3.5 px-3 text-slate-400 font-medium whitespace-nowrap">
                    {c.timestamp}
                  </td>
                  <td className="py-3.5 px-3 text-right">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectCase(c);
                      }}
                      className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 text-xs font-semibold transition-all cursor-pointer"
                    >
                      <span>Analyze</span>
                      <ArrowRight className="w-3 h-3 group-hover:translate-x-0.5 transition-transform" />
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
