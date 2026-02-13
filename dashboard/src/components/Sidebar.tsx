"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard, GitFork, BarChart3, Settings, HelpCircle, LogOut } from "lucide-react";
import { clsx } from "clsx";

const menuItems = [
    { href: "/dashboard", icon: LayoutDashboard, label: "Overview" },
    { href: "/dashboard/causal-graph", icon: GitFork, label: "Causal Graph" },
    { href: "/dashboard/simulator", icon: BarChart3, label: "Simulation" },
    { href: "/dashboard/settings", icon: Settings, label: "Settings" },
];

export default function Sidebar() {
    const pathname = usePathname();

    return (
        <aside className="fixed left-0 top-0 h-screen w-64 bg-slate-900/50 backdrop-blur-xl border-r border-white/5 flex flex-col items-center py-6">
            {/* Logo */}
            <div className="mb-10 w-full px-6">
                <Link href="/" className="flex items-center gap-2 group">
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-500 to-accent-pink flex items-center justify-center text-white font-bold text-xl shadow-lg shadow-brand-500/20 group-hover:scale-105 transition-transform">
                        W
                    </div>
                    <span className="text-xl font-bold text-white tracking-tight group-hover:text-glow transition-all">WhyLab</span>
                </Link>
            </div>

            {/* Menu */}
            <nav className="flex-1 w-full px-4 space-y-2">
                {menuItems.map((item) => {
                    const isActive = pathname === item.href;
                    return (
                        <Link
                            key={item.href}
                            href={item.href}
                            className={clsx(
                                "flex items-center gap-3 px-4 py-3 rounded-xl transition-all duration-200",
                                isActive
                                    ? "bg-brand-500/20 text-brand-300 border border-brand-500/20 shadow-[0_0_15px_rgba(139,92,246,0.1)]"
                                    : "text-slate-400 hover:text-white hover:bg-white/5"
                            )}
                        >
                            <item.icon className={clsx("w-5 h-5", isActive ? "text-brand-400" : "text-slate-500")} />
                            <span className="font-medium text-sm">{item.label}</span>
                        </Link>
                    );
                })}
            </nav>

            {/* Footer Actions */}
            <div className="w-full px-4 space-y-2 mt-auto">
                <button className="w-full flex items-center gap-3 px-4 py-3 text-slate-400 hover:text-white hover:bg-white/5 rounded-xl transition-colors">
                    <HelpCircle className="w-5 h-5" />
                    <span className="font-medium text-sm">Documentation</span>
                </button>
                <div className="pt-4 border-t border-white/5">
                    <Link href="/" className="flex items-center gap-3 px-4 py-3 text-slate-400 hover:text-red-400 hover:bg-red-500/10 rounded-xl transition-colors">
                        <LogOut className="w-5 h-5" />
                        <span className="font-medium text-sm">Exit Demo</span>
                    </Link>
                </div>
            </div>
        </aside>
    );
}
