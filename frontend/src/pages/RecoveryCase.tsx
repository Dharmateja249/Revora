import React, { useState, useEffect } from "react";
import {
  BrainCircuit,
  RotateCcw,
  Zap,
  AlertCircle,
  ShieldCheck,
  Cpu,
  Layers,
  CheckCircle2,
} from "lucide-react";
import { PaymentCaseCard } from "../components/PaymentCaseCard";
import {
  DecisionPipeline,
  createInitialPipelineStages,
  PipelineStage,
} from "../components/DecisionPipeline";
import { RecoveryActionCard } from "../components/RecoveryActionCard";
import { ExecutionResult } from "../components/ExecutionResult";
import { evaluateRecoveryDecision, ApiError } from "../api/recovery";
import {
  DemoPaymentCase,
  RecoveryDecisionResponse,
  ActionExecutionResultDTO,
} from "../types/recovery";
import { DEMO_PAYMENT_CASES } from "../data/demoCases";

interface RecoveryCaseProps {
  currentCase: DemoPaymentCase;
  onSelectCase: (c: DemoPaymentCase) => void;
}

export const RecoveryCase: React.FC<RecoveryCaseProps> = ({
  currentCase,
  onSelectCase,
}) => {
  const [activeCase, setActiveCase] = useState<DemoPaymentCase>(currentCase);
  const [pipelineStages, setPipelineStages] = useState<PipelineStage[]>(
    createInitialPipelineStages()
  );
  const [decision, setDecision] = useState<RecoveryDecisionResponse | null>(null);
  const [execution, setExecution] = useState<ActionExecutionResultDTO | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [isExecuting, setIsExecuting] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // Sync and hydrate case state from authoritative backend when switching cases or on initial mount
  useEffect(() => {
    setActiveCase(currentCase);
    setPipelineStages(createInitialPipelineStages());
    setDecision(null);
    setExecution(null);
    setErrorMessage(null);

    // Non-intrusive hydration: retrieve authoritative database state (attempts, customer counters) for stable payment_id
    let isSubscribed = true;
    evaluateRecoveryDecision({
      ...currentCase.requestPayload,
      execute_action: false,
    })
      .then((res) => {
        if (!isSubscribed) return;
        setActiveCase((prev) => {
          const updated: DemoPaymentCase = {
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
          };
          onSelectCase(updated);
          return updated;
        });
      })
      .catch((_err) => {
        // Hydration failure (e.g. backend not reachable yet) is non-fatal; user can still analyze
      });

    return () => {
      isSubscribed = false;
    };
  }, [currentCase.id]);

  /**
   * Helper to update specific stage status in the 8-step pipeline
   */
  const updateStage = (
    id: string,
    status: PipelineStage["status"],
    detail?: string
  ) => {
    setPipelineStages((prev) =>
      prev.map((s) => (s.id === id ? { ...s, status, detail: detail ?? s.detail } : s))
    );
  };

  /**
   * Phase 1: Analyze with Revora (execute_action: false)
   * Evaluates failed payment through AgentOrchestrator, RAG, and PolicyValidator
   */
  const handleAnalyze = async () => {
    setIsAnalyzing(true);
    setErrorMessage(null);
    setDecision(null);
    setExecution(null);

    // Initial pipeline animation sequence
    updateStage("step_1", "completed", "Failure detected");
    updateStage("step_2", "processing", "Loading profile...");

    try {
      // Simulate micro-progression to make agent steps clearly visible
      await new Promise((r) => setTimeout(r, 200));
      updateStage("step_2", "completed", "Trust score parsed");
      updateStage("step_3", "processing", "Vector RAG matching...");

      await new Promise((r) => setTimeout(r, 200));
      updateStage("step_4", "processing", "LLM synthesizing...");

      // Call REAL backend API with execute_action: false
      const result = await evaluateRecoveryDecision({
        ...activeCase.requestPayload,
        execute_action: false,
      });

      // Synchronize authoritative state from response
      const updatedAnalyzed: DemoPaymentCase = {
        ...activeCase,
        attempt_count: result.attempt_count ?? result.previous_attempts?.length,
        requestPayload: {
          ...activeCase.requestPayload,
          payment_id: result.payment_id || activeCase.requestPayload.payment_id,
          opportunity_status: result.opportunity_status || activeCase.requestPayload.opportunity_status,
          previous_attempts: result.previous_attempts || activeCase.requestPayload.previous_attempts,
          customer: result.customer
            ? {
                customer_id: result.customer.customer_id || activeCase.requestPayload.customer?.customer_id,
                total_payments: result.customer.total_payments,
                successful_payments: result.customer.successful_payments,
                failed_payments: result.customer.failed_payments,
                historical_success_rate: result.customer.historical_success_rate,
              }
            : activeCase.requestPayload.customer,
        },
      };
      setActiveCase(updatedAnalyzed);
      onSelectCase(updatedAnalyzed);

      // Stage 3: Derived dynamically from actual RAG retrieval results
      const precedentCount = result.referenced_case_ids?.length || 0;
      if (precedentCount > 0) {
        updateStage(
          "step_3",
          "completed",
          `${precedentCount} precedent${precedentCount > 1 ? "s" : ""} cited`
        );
      } else {
        updateStage("step_3", "completed", "No relevant historical precedents");
      }

      // Stage 4: AI Recommendation
      updateStage("step_4", "completed", `${Math.round(result.confidence * 100)}% confidence`);

      // Stage 5: Policy Validation
      if (result.policy_overridden) {
        updateStage("step_5", "completed", "Overridden (Safe)");
      } else {
        updateStage("step_5", "completed", "Policy approved");
      }

      // Stage 6: Recovery Action Finalized
      updateStage("step_6", "completed", result.recommended_action);

      setDecision(result);
    } catch (err: unknown) {
      const msg = err instanceof ApiError ? err.detail : "Failed to communicate with Revora decision API.";
      setErrorMessage(msg);
      updateStage("step_3", "failed", "RAG check incomplete");
      updateStage("step_4", "failed", "Analysis failed");
    } finally {
      setIsAnalyzing(false);
    }
  };

  /**
   * Phase 2: Execute Recovery (execute_action: true)
   * Dispatches the policy-approved action via ActionExecutor and RazorpayAdapter
   */
  const handleExecute = async () => {
    if (!decision) return;

    setIsExecuting(true);
    setErrorMessage(null);

    updateStage("step_7", "processing", "Calling RazorpayAdapter...");

    try {
      await new Promise((r) => setTimeout(r, 300));

      // Call REAL backend API with execute_action: true
      const result = await evaluateRecoveryDecision({
        ...activeCase.requestPayload,
        execute_action: true,
      });

      setDecision(result);

      // Synchronize authoritative state from response
      const updatedExecuted: DemoPaymentCase = {
        ...activeCase,
        attempt_count: result.attempt_count ?? result.previous_attempts?.length,
        requestPayload: {
          ...activeCase.requestPayload,
          payment_id: result.payment_id || activeCase.requestPayload.payment_id,
          opportunity_status: result.opportunity_status || activeCase.requestPayload.opportunity_status,
          previous_attempts: result.previous_attempts || activeCase.requestPayload.previous_attempts,
          customer: result.customer
            ? {
                customer_id: result.customer.customer_id || activeCase.requestPayload.customer?.customer_id,
                total_payments: result.customer.total_payments,
                successful_payments: result.customer.successful_payments,
                failed_payments: result.customer.failed_payments,
                historical_success_rate: result.customer.historical_success_rate,
              }
            : activeCase.requestPayload.customer,
        },
      };
      setActiveCase(updatedExecuted);
      onSelectCase(updatedExecuted);

      if (result.execution) {
        setExecution(result.execution);

        if (result.execution.success || result.execution.status === "simulated") {
          updateStage("step_7", "completed", result.execution.status);
          updateStage("step_8", "completed", "Link created & logged");
        } else {
          updateStage("step_7", "failed", result.execution.status);
          updateStage("step_8", "failed", "Execution aborted");
        }
      }
    } catch (err: unknown) {
      const msg = err instanceof ApiError ? err.detail : "Failed to execute recovery action.";
      setErrorMessage(msg);
      updateStage("step_7", "failed", "Adapter error");
    } finally {
      setIsExecuting(false);
    }
  };

  const handleReset = () => {
    setPipelineStages(createInitialPipelineStages());
    setDecision(null);
    setExecution(null);
    setErrorMessage(null);
  };

  return (
    <div className="space-y-8 pb-16 animate-fade-in">
      {/* Top Header & Scenario Switcher */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono font-bold uppercase tracking-wider text-emerald-400 bg-emerald-950/60 px-2 py-0.5 rounded border border-emerald-800/60">
              Active Case Investigation
            </span>
            <span className="text-xs text-slate-400 font-mono">ID: {currentCase.id}</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight mt-1">
            {currentCase.customerName}
          </h1>
          <p className="text-xs text-slate-400 mt-0.5">{currentCase.description}</p>
        </div>

        {/* Quick Scenario Picker Buttons */}
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs text-slate-400 font-medium mr-1 flex items-center gap-1">
            <Layers className="w-3.5 h-3.5" />
            Switch Scenario:
          </span>
          {DEMO_PAYMENT_CASES.map((c) => (
            <button
              key={c.id}
              onClick={() => onSelectCase(c)}
              className={`text-xs font-medium px-3 py-1.5 rounded-lg border transition-all cursor-pointer ${
                c.id === currentCase.id
                  ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/50 shadow-sm"
                  : "bg-slate-900/80 text-slate-400 border-slate-800 hover:text-slate-200 hover:border-slate-700"
              }`}
            >
              {c.requestPayload.payment_method.toUpperCase()} ({c.customerTier})
            </button>
          ))}
        </div>
      </div>

      {/* Error Alert (if API call fails) */}
      {errorMessage && (
        <div className="p-4 rounded-xl bg-rose-950/40 border border-rose-800/80 text-rose-200 text-sm flex items-start gap-3 shadow-lg">
          <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <strong className="font-semibold block">Backend Connectivity / Validation Error:</strong>
            <p className="text-xs leading-relaxed font-mono">{errorMessage}</p>
            <p className="text-[11px] text-rose-300 pt-1">
              Ensure FastAPI backend is running (<code className="bg-slate-950 px-1.5 py-0.5 rounded">uvicorn app.main:app --port 8000</code>).
            </p>
          </div>
        </div>
      )}

      {/* Payment Case Details Header */}
      <PaymentCaseCard paymentCase={activeCase} />

      {/* Autonomous Recovery Pipeline Visualizer */}
      <DecisionPipeline stages={pipelineStages} />

      {/* Primary Action Trigger: Analyze Button */}
      {!decision && (
        <div className="p-8 rounded-2xl bg-gradient-to-b from-slate-900/90 to-slate-950 border border-slate-800 text-center space-y-4 shadow-xl">
          <div className="w-14 h-14 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 flex items-center justify-center mx-auto shadow-lg shadow-emerald-500/10">
            <BrainCircuit className="w-7 h-7" />
          </div>
          <div className="max-w-md mx-auto space-y-1">
            <h3 className="text-lg font-extrabold text-white">
              Initialize Autonomous Recovery Analysis
            </h3>
            <p className="text-xs text-slate-400 leading-relaxed">
              Invokes the Revora Agent: synthesizes vector RAG evidence, queries LLM orchestrator, and validates constraints via PolicyValidator.
            </p>
          </div>

          <div className="pt-2 flex items-center justify-center gap-3">
            <button
              onClick={handleAnalyze}
              disabled={isAnalyzing}
              className={`flex items-center gap-2.5 px-7 py-3.5 rounded-xl font-bold text-sm shadow-xl transition-all cursor-pointer ${
                isAnalyzing
                  ? "bg-slate-800 text-slate-400 cursor-not-allowed"
                  : "bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 text-slate-950 shadow-emerald-500/20 hover:shadow-emerald-500/30"
              }`}
            >
              {isAnalyzing ? (
                <>
                  <Cpu className="w-4 h-4 animate-spin" />
                  <span>Synthesizing Agent Context...</span>
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4 fill-current" />
                  <span>Analyze with Revora (execute_action: false)</span>
                </>
              )}
            </button>
          </div>

          <div className="pt-2 flex items-center justify-center gap-2 text-[11px] text-slate-400 font-medium">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-400" />
            <span>Safety Invariant: Read-only evaluation. No money movement or gateway actions.</span>
          </div>
        </div>
      )}

      {/* Decision Result Card */}
      {decision && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h3 className="text-base font-bold text-white uppercase tracking-wider flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-400" />
              <span>Evaluated Decision Telemetry</span>
            </h3>
            <button
              onClick={handleReset}
              className="text-xs font-semibold text-slate-400 hover:text-slate-200 flex items-center gap-1.5 transition-colors cursor-pointer"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Reset Analysis</span>
            </button>
          </div>

          <RecoveryActionCard
            decision={decision}
            onExecute={!execution ? handleExecute : undefined}
            isExecuting={isExecuting}
          />
        </div>
      )}

      {/* External Action Execution Result */}
      {execution && (
        <div className="space-y-4">
          <h3 className="text-base font-bold text-white uppercase tracking-wider flex items-center gap-2">
            <Zap className="w-4 h-4 text-emerald-400" />
            <span>Gateway Execution Outcome</span>
          </h3>
          <ExecutionResult execution={execution} />
        </div>
      )}
    </div>
  );
};
