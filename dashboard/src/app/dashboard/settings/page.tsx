"use client";

import { motion } from "framer-motion";
import { Settings, Bell, Database, Shield, Palette } from "lucide-react";

const SECTIONS = [
    {
        icon: Database,
        title: "데이터 소스",
        desc: "CSV, SQL, BigQuery 등 외부 데이터 연결 관리",
        status: "데모 모드",
    },
    {
        icon: Bell,
        title: "모니터링 & 알림",
        desc: "인과 드리프트 감지 주기 및 Slack 알림 설정",
        status: "비활성",
    },
    {
        icon: Shield,
        title: "견고성 검증",
        desc: "Placebo, Bootstrap, E-value 임계값 설정",
        status: "기본값",
    },
    {
        icon: Palette,
        title: "대시보드 테마",
        desc: "컬러 팔레트, 차트 스타일, 레이아웃 설정",
        status: "다크",
    },
];

export default function SettingsPage() {
    return (
        <div className="space-y-8">
            <div>
                <h1 className="text-2xl font-bold text-white flex items-center gap-3">
                    <Settings className="w-7 h-7 text-brand-400" />
                    설정
                </h1>
                <p className="text-sm text-slate-400 mt-1">
                    WhyLab 파이프라인 및 대시보드 설정을 관리합니다.
                </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {SECTIONS.map((s, i) => (
                    <motion.div
                        key={s.title}
                        initial={{ opacity: 0, y: 12 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.1 }}
                        className="glass-card flex items-start gap-4"
                    >
                        <div className="p-2.5 rounded-xl bg-brand-500/10 text-brand-400">
                            <s.icon className="w-5 h-5" />
                        </div>
                        <div className="flex-1">
                            <div className="flex items-center justify-between">
                                <h3 className="font-semibold text-white text-sm">{s.title}</h3>
                                <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-700/60 text-slate-400 border border-white/5">
                                    {s.status}
                                </span>
                            </div>
                            <p className="text-xs text-slate-500 mt-1">{s.desc}</p>
                        </div>
                    </motion.div>
                ))}
            </div>

            {/* 데모 안내 */}
            <div className="glass-card border-brand-500/20 text-center py-8">
                <p className="text-slate-400 text-sm">
                    🧪 현재 <span className="text-brand-400 font-medium">데모 모드</span>입니다.
                </p>
                <p className="text-slate-500 text-xs mt-1">
                    실제 파이프라인 연동 시 설정이 활성화됩니다.
                </p>
            </div>
        </div>
    );
}
