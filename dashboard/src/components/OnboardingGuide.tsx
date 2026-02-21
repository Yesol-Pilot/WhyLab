"use client";

import { useState } from "react";
import { Info, X, ChevronRight, Lightbulb } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

export default function OnboardingGuide() {
    const [isVisible, setIsVisible] = useState(true);
    const [step, setStep] = useState(0);

    const steps = [
        {
            title: "Welcome to WhyLab",
            desc: "This is a space for analyzing causal relationships in data to help you make better decisions. Go beyond observing phenomena — explore why things happen.",
            icon: <Lightbulb className="w-6 h-6 text-yellow-400" />
        },
        {
            title: "Scenario Selection",
            desc: "Use the 'Scenario A' and 'Scenario B' buttons at the top to switch between analysis topics. Currently, A is 'Credit Limit Adjustment' and B is 'Medical Treatment'.",
            icon: <ChevronRight className="w-6 h-6 text-brand-400" />
        },
        {
            title: "Causal Effect (ATE)",
            desc: "The most important number is 'Total ATE' at the top. This represents the pure causal impact of a policy (treatment) on the outcome.",
            icon: <Info className="w-6 h-6 text-blue-400" />
        },
        {
            title: "What-If Simulation",
            desc: "Try the simulator on the right. It predicts in real time how outcomes would change if you increase or decrease the treatment intensity.",
            icon: <Info className="w-6 h-6 text-green-400" />
        }
    ];

    if (!isVisible) return (
        <button
            onClick={() => setIsVisible(true)}
            className="fixed bottom-4 right-4 bg-brand-600 text-white p-3 rounded-full shadow-lg hover:bg-brand-500 transition-colors z-50"
        >
            <Info className="w-6 h-6" />
        </button>
    );

    return (
        <AnimatePresence>
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.9 }}
                className="fixed bottom-6 right-6 w-80 glass-card border border-brand-500/30 shadow-2xl z-50 overflow-hidden"
            >
                <div className="bg-brand-500/10 p-4 border-b border-white/5 flex justify-between items-start">
                    <div className="flex gap-3">
                        <div className="mt-1">{steps[step].icon}</div>
                        <div>
                            <h3 className="font-bold text-white text-sm">{steps[step].title}</h3>
                            <div className="flex gap-1 mt-1">
                                {steps.map((_, i) => (
                                    <div key={i} className={`h-1 rounded-full transition-all ${i === step ? "w-4 bg-brand-400" : "w-1 bg-slate-600"}`} />
                                ))}
                            </div>
                        </div>
                    </div>
                    <button onClick={() => setIsVisible(false)} className="text-slate-400 hover:text-white">
                        <X className="w-4 h-4" />
                    </button>
                </div>

                <div className="p-4">
                    <p className="text-sm text-slate-300 leading-relaxed min-h-[60px]">
                        {steps[step].desc}
                    </p>
                </div>

                <div className="p-4 pt-0 flex justify-between items-center">
                    <button
                        onClick={() => setStep(Math.max(0, step - 1))}
                        disabled={step === 0}
                        className="text-xs text-slate-500 hover:text-white disabled:opacity-30"
                    >
                        Prev
                    </button>
                    <button
                        onClick={() => {
                            if (step < steps.length - 1) setStep(step + 1);
                            else setIsVisible(false);
                        }}
                        className="bg-brand-500 hover:bg-brand-400 text-white px-4 py-1.5 rounded-md text-xs font-medium transition-colors"
                    >
                        {step === steps.length - 1 ? "Start Exploring" : "Next"}
                    </button>
                </div>
            </motion.div>
        </AnimatePresence>
    );
}
