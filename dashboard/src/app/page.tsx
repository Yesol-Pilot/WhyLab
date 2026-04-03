"use client";

import { motion } from "framer-motion";
import { ArrowRight, BarChart3, BrainCircuit, GitFork, CreditCard, Megaphone, FlaskConical, Play } from "lucide-react";
import Link from "next/link";
import React from "react";

export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-4 md:p-24 relative overflow-hidden">

      {/* Background Glow */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[600px] h-[600px] bg-brand-500/20 rounded-full blur-3xl opacity-30 pointer-events-none" />
      <div className="absolute bottom-0 right-1/4 w-[400px] h-[400px] bg-accent-cyan/10 rounded-full blur-3xl opacity-20 pointer-events-none" />

      {/* Hero Section */}
      <section className="z-10 w-full max-w-5xl text-center space-y-8">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="space-y-4"
        >
          <div className="inline-block px-4 py-1.5 rounded-full border border-brand-500/30 bg-brand-500/10 text-brand-300 text-sm font-medium mb-4 backdrop-blur-md">
            <FlaskConical className="w-3.5 h-3.5 inline mr-1.5 -mt-0.5" />
            Research Prototype · Ground Truth Validated
          </div>
          <h1 className="text-5xl md:text-7xl font-bold tracking-tight text-white drop-shadow-lg">
            AI-Powered <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-brand-400 to-accent-cyan text-glow">
              Causal Inference
            </span>
          </h1>
          <p className="text-lg md:text-xl text-slate-300 max-w-2xl mx-auto leading-relaxed">
            Automatically separate correlation from causation. <span className="text-brand-300 font-semibold">AutoML + Double Machine Learning</span> uncovers the real causes hidden in your data and provides actionable insights.
          </p>
        </motion.div>

        {/* Scenario Selection Cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, duration: 0.8 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-w-5xl mx-auto mt-10"
        >
          <ScenarioCard
            title="Scenario A: Credit Limit"
            description="Optimize credit limits to minimize default rates."
            features={['Credit Limit', 'Default Rate', 'Income']}
            href="/dashboard?scenario=A"
            icon={<CreditCard className="w-6 h-6" />}
            color="brand"
            delay={0.5}
          />
          <ScenarioCard
            title="Scenario B: Medical Treatment"
            description="Healthcare outcome analysis with confounding variables"
            features={['Treatment Effect', 'Patient History', 'Cure Rate']}
            href="/dashboard?scenario=B"
            icon={<Megaphone className="w-6 h-6" />}
            color="cyan"
            delay={0.6}
          />
          <ScenarioCard
            title="Scenario C: Personal Career"
            description="Analyze causal impact of your skills on market value based on local knowledge base."
            features={['Skill Level', 'Project Completion', 'Market Value']}
            href="/dashboard?scenario=C"
            icon={<BarChart3 className="w-6 h-6" />}
            color="brand"
            delay={0.7}
            badge="REAL DATA"
          />
        </motion.div>

        {/* Links */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.7, duration: 0.8 }}
          className="flex flex-col sm:flex-row gap-4 justify-center items-center mt-6"
        >
          <Link
            href="/dashboard"
            className="group px-8 py-4 rounded-xl bg-brand-600 hover:bg-brand-500 text-white font-bold text-lg transition-all shadow-lg shadow-brand-500/20 flex items-center gap-2"
          >
            Start Analysis
            <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </Link>
          <Link
            href="https://github.com/Yesol-Pilot/WhyLab"
            target="_blank"
            className="px-8 py-4 rounded-xl border border-white/10 hover:bg-white/5 text-slate-300 transition-all backdrop-blur-sm"
          >
            GitHub Repository
          </Link>
        </motion.div>
      </section>

      {/* Feature Cards */}
      <section className="z-10 grid grid-cols-1 md:grid-cols-3 gap-6 mt-20 w-full max-w-6xl">
        <FeatureCard
          icon={<BrainCircuit className="w-8 h-8 text-brand-400" />}
          title="AI AutoML Causal Inference"
          desc="AI auto-selects the optimal model between LinearDML and CausalForest. RMSE 0.028–0.609, Correlation 0.977–0.996."
          delay={0.8}
        />
        <FeatureCard
          icon={<GitFork className="w-8 h-8 text-accent-cyan" />}
          title="Causal Graph (DAG)"
          desc="Automatically discover causal structures with the PC Algorithm. Visually identify confounders, treatment, and outcome variables."
          delay={0.9}
        />
        <FeatureCard
          icon={<BarChart3 className="w-8 h-8 text-accent-pink" />}
          title="What-If Decision Making"
          desc="'Will raising the credit limit reduce defaults?' — Ground truth validated counterfactual simulation."
          delay={1.0}
        />
      </section>

      {/* Footer */}
      <footer className="absolute bottom-6 text-slate-500 text-sm">
        © 2026 WhyLab Project. Built with Next.js & EconML.
      </footer>
    </main>
  );
}

/* ── Scenario Selection Card ── */
interface ScenarioCardProps {
  title: string;
  description: string;
  features: string[];
  href: string;
  icon: React.ReactNode;
  color: "brand" | "cyan";
  delay: number;
  badge?: string;
}

function ScenarioCard({ title, description, features, href, icon, color, delay, badge }: ScenarioCardProps) {
  const borderColor = color === "brand" ? "border-brand-500/30 hover:border-brand-500/60" : "border-accent-cyan/30 hover:border-accent-cyan/60";
  const bgColor = color === "brand" ? "bg-brand-500/5 hover:bg-brand-500/10" : "bg-accent-cyan/5 hover:bg-accent-cyan/10";
  const iconBg = color === "brand" ? "text-brand-400" : "text-accent-cyan";

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5 }}
    >
      <Link
        href={href}
        className={`group relative block p-5 rounded-xl border ${borderColor} ${bgColor} backdrop-blur-md transition-all duration-300 hover:translate-y-[-3px] hover:shadow-lg h-full`}
      >
        {badge && (
          <div className="absolute top-4 right-4 px-2 py-0.5 bg-brand-500/20 text-brand-300 text-[10px] font-bold rounded uppercase border border-brand-500/30">
            {badge}
          </div>
        )}
        <div className="flex items-center gap-3 mb-3">
          <div className={`${iconBg}`}>{icon}</div>
          <div>
            <h3 className="text-lg font-bold text-white">{title}</h3>
          </div>
          <ArrowRight className="w-4 h-4 text-slate-500 ml-auto group-hover:translate-x-1 transition-transform" />
        </div>
        <p className="text-slate-400 text-sm mb-4 leading-relaxed h-10">
          {description}
        </p>
        <div className="space-y-1">
          {features.map((feature, i) => (
            <div key={i} className="flex items-center gap-2 text-xs text-slate-500">
              <div className="w-1 h-1 rounded-full bg-brand-500/50" />
              {feature}
            </div>
          ))}
        </div>
      </Link>
    </motion.div>
  );
}

/* ── Feature Card ── */
function FeatureCard({ icon, title, desc, delay }: { icon: React.ReactNode; title: string; desc: string; delay: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: delay, duration: 0.6 }}
      className="glass-card flex flex-col items-start gap-4 hover:translate-y-[-5px]"
    >
      <div className="p-3 rounded-lg bg-white/5 border border-white/10">
        {icon}
      </div>
      <h3 className="text-xl font-bold text-white">{title}</h3>
      <p className="text-slate-400 leading-relaxed text-sm">{desc}</p>
    </motion.div>
  );
}
