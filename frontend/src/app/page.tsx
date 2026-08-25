export default function Home() {
  return (
    <main className="min-h-screen flex items-center justify-center bg-slate-950 text-slate-100 px-4">
      <div className="text-center space-y-5 max-w-lg p-8 rounded-2xl bg-slate-900/60 border border-slate-800 backdrop-blur-md shadow-2xl">
        <div className="inline-flex items-center gap-2 px-3 py-1 text-xs font-semibold uppercase tracking-wider text-emerald-400 bg-emerald-950/50 border border-emerald-800/60 rounded-full">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          System Online
        </div>
        <h1 className="text-5xl font-extrabold tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
          Revora
        </h1>
        <p className="text-lg font-medium text-slate-200">
          Adaptive AI Revenue Recovery Agent
        </p>
        <div className="pt-2">
          <p className="text-xs font-semibold tracking-wider text-slate-400 uppercase bg-slate-800/50 py-2 px-4 rounded-lg border border-slate-700/50 inline-block">
            Detect → Decide → Recover → Learn
          </p>
        </div>
      </div>
    </main>
  );
}
