import React from "react";
import {
  AlertOctagon,
  Database,
  Search,
  BrainCircuit,
  ShieldCheck,
  CheckCircle,
  Zap,
  ArrowRight,
  Check,
  Clock,
  Loader2,
  XCircle,
} from "lucide-react";

export type PipelineStageStatus = "pending" | "processing" | "completed" | "failed" | "skipped";

export interface PipelineStage {
  id: string;
  name: string;
  subtext: string;
  status: PipelineStageStatus;
  detail?: string;
  icon: React.ElementType;
}

interface DecisionPipelineProps {
  stages: PipelineStage[];
}

export const DecisionPipeline: React.FC<DecisionPipelineProps> = ({ stages }) => {
  const getStatusIcon = (status: PipelineStageStatus) => {
    switch (status) {
      case "completed":
        return <Check className="w-3.5 h-3.5 text-emerald-400" />;
      case "processing":
        return <Loader2 className="w-3.5 h-3.5 text-blue-400 animate-spin" />;
      case "failed":
        return <XCircle className="w-3.5 h-3.5 text-rose-400" />;
      case "skipped":
        return <Clock className="w-3.5 h-3.5 text-slate-500" />;
      default:
        return <span className="w-1.5 h-1.5 rounded-full bg-slate-600" />;
    }
  };

  const getStageBorder = (status: PipelineStageStatus) => {
    switch (status) {
      case "completed":
        return "border-emerald-500/40 bg-emerald-950/20 text-emerald-300";
      case "processing":
        return "border-blue-500/50 bg-blue-950/30 text-blue-300 shadow-lg shadow-blue-500/10 animate-pulse-subtle";
      case "failed":
        return "border-rose-500/40 bg-rose-950/20 text-rose-300";
      case "skipped":
        return "border-slate-800/60 bg-slate-900/30 text-slate-500 opacity-60";
      default:
        return "border-slate-800/60 bg-slate-950/40 text-slate-500 opacity-50";
    }
  };

  return (
    <div className="rounded-2xl bg-slate-900/70 border border-slate-800 p-6 shadow-xl space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h4 className="text-sm font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <BrainCircuit className="w-4 h-4 text-emerald-400" />
            <span>Autonomous Recovery Pipeline</span>
          </h4>
          <p className="text-xs text-slate-400 mt-0.5">
            Real-time visual telemetry through Revora's agent & policy validator
          </p>
        </div>
        <div className="text-[11px] font-mono px-2.5 py-1 rounded-full bg-slate-800 text-slate-300 border border-slate-700 flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping" />
          <span>8-Stage Invariant Chain</span>
        </div>
      </div>

      {/* Pipeline Stages Progression */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 pt-2">
        {stages.map((stage, idx) => {
          const Icon = stage.icon;
          return (
            <div
              key={stage.id}
              className={`relative rounded-xl border p-3.5 transition-all flex flex-col justify-between min-h-[92px] ${getStageBorder(
                stage.status
              )}`}
            >
              {/* Step indicator header */}
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2">
                  <div className="p-1.5 rounded-lg bg-slate-900/80 border border-slate-800/80">
                    <Icon className="w-4 h-4" />
                  </div>
                  <div>
                    <div className="text-xs font-bold text-slate-100">{stage.name}</div>
                    <div className="text-[10px] text-slate-400 font-medium">{stage.subtext}</div>
                  </div>
                </div>

                {/* Status chip */}
                <div className="w-5 h-5 rounded-full bg-slate-900/90 border border-slate-800 flex items-center justify-center shrink-0">
                  {getStatusIcon(stage.status)}
                </div>
              </div>

              {/* Dynamic stage detail text */}
              {stage.detail && (
                <div className="mt-2 text-[11px] font-mono text-slate-300 bg-slate-950/60 p-1.5 rounded border border-slate-800/60 truncate" title={stage.detail}>
                  {stage.detail}
                </div>
              )}

              {/* Connector arrow for visual chain (except last) */}
              {idx < stages.length - 1 && (
                <div className="hidden lg:block absolute -right-2 top-1/2 -translate-y-1/2 z-10">
                  <ArrowRight className="w-3.5 h-3.5 text-slate-700" />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export const createInitialPipelineStages = (): PipelineStage[] => [
  {
    id: "step_1",
    name: "1. Payment Failed",
    subtext: "Ingest failure signal",
    status: "pending",
    icon: AlertOctagon,
  },
  {
    id: "step_2",
    name: "2. Context Retrieved",
    subtext: "Profile & attempts",
    status: "pending",
    icon: Database,
  },
  {
    id: "step_3",
    name: "3. RAG Retrieval",
    subtext: "Vector evidence match",
    status: "pending",
    icon: Search,
  },
  {
    id: "step_4",
    name: "4. AI Recommendation",
    subtext: "Structured LLM reason",
    status: "pending",
    icon: BrainCircuit,
  },
  {
    id: "step_5",
    name: "5. Policy Validation",
    subtext: "Mandatory safety rules",
    status: "pending",
    icon: ShieldCheck,
  },
  {
    id: "step_6",
    name: "6. Recovery Action",
    subtext: "Safe approved action",
    status: "pending",
    icon: CheckCircle,
  },
  {
    id: "step_7",
    name: "7. Gateway Execution",
    subtext: "ActionExecutor adapter",
    status: "pending",
    icon: Zap,
  },
  {
    id: "step_8",
    name: "8. Recovery Result",
    subtext: "Telemetry & audit",
    status: "pending",
    icon: Check,
  },
];
