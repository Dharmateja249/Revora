import React from "react";
import { LucideIcon } from "lucide-react";

interface MetricCardProps {
  label: string;
  value: string;
  subtext?: string;
  change?: string;
  isPositive?: boolean;
  icon: LucideIcon;
  accentColor?: "emerald" | "blue" | "indigo" | "amber" | "rose";
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  subtext,
  change,
  isPositive = true,
  icon: Icon,
  accentColor = "emerald",
}) => {
  const colorMap = {
    emerald: "from-emerald-500/10 to-transparent border-emerald-500/20 text-emerald-400 bg-emerald-500/10",
    blue: "from-blue-500/10 to-transparent border-blue-500/20 text-blue-400 bg-blue-500/10",
    indigo: "from-indigo-500/10 to-transparent border-indigo-500/20 text-indigo-400 bg-indigo-500/10",
    amber: "from-amber-500/10 to-transparent border-amber-500/20 text-amber-400 bg-amber-500/10",
    rose: "from-rose-500/10 to-transparent border-rose-500/20 text-rose-400 bg-rose-500/10",
  };

  return (
    <div className="relative overflow-hidden rounded-2xl bg-slate-900/70 border border-slate-800 p-5 shadow-lg shadow-black/20 hover:border-slate-700/80 transition-all group">
      {/* Subtle top gradient accent */}
      <div className={`absolute inset-x-0 top-0 h-1 bg-gradient-to-r ${colorMap[accentColor]}`} />

      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{label}</p>
          <p className="text-2xl font-extrabold text-white tracking-tight">{value}</p>
        </div>
        <div className={`p-2.5 rounded-xl border border-slate-800 ${colorMap[accentColor]} flex items-center justify-center`}>
          <Icon className="w-5 h-5" />
        </div>
      </div>

      {(subtext || change) && (
        <div className="mt-3.5 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs">
          {change && (
            <span
              className={`font-semibold flex items-center gap-1 ${
                isPositive ? "text-emerald-400" : "text-rose-400"
              }`}
            >
              {isPositive ? "↑" : "↓"} {change}
            </span>
          )}
          {subtext && <span className="text-slate-400 font-medium">{subtext}</span>}
        </div>
      )}
    </div>
  );
};
