import Sidebar from "@/components/Sidebar";

export default function DashboardLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <div className="flex min-h-screen bg-dots-pattern">
            <Sidebar />
            <div className="flex-1 ml-64 p-8 overflow-y-auto">
                {/* Top Bar Area (Placeholder for Breadcrumbs / User Profile) */}
                <header className="flex justify-between items-center mb-8">
                    <div className="text-sm breadcrumbs text-slate-400">
                        Dashboard <span className="mx-2">/</span> Overview
                    </div>
                    <div className="flex items-center gap-4">
                        <div className="w-8 h-8 rounded-full bg-slate-700 flex items-center justify-center text-xs text-white">Guest</div>
                    </div>
                </header>
                <main>
                    {children}
                </main>
            </div>
        </div>
    );
}
