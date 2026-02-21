import { CausalAnalysisResult } from '@/types';
import { scenarioA_Mock, scenarioB_Mock, scenarioC_Mock } from '@/lib/mockData';

/**
 * 인과분석 결과 데이터를 로딩합니다.
 *
 * 우선순위:
 *  1. public/data/latest.json (시나리오 A) 또는 scenario_b.json (시나리오 B)
 *  2. Mock 데이터 (Fallback)
 */
export async function getCausalData(
    scenario: string = 'A',
): Promise<CausalAnalysisResult> {
    const jsonFile = scenario === 'B' ? 'scenario_b.json' : 'latest.json';

    try {
        const basePath =
            typeof window !== 'undefined'
                ? window.location.origin
                : '';

        // Scenario C: Synthetic Mode (No API Call needed)
        if (scenario === 'C') {
            console.info("[DataLoader] Scenario C detected: Switching to Synthetic Mode.");
            return scenarioC_Mock;
        }

        const res = await fetch(`${basePath}/data/${jsonFile}`);
        if (res.ok) {
            return (await res.json()) as CausalAnalysisResult;
        }
    } catch (error) {
        console.warn(
            `[DataLoader] Failed to load ${jsonFile}. Falling back to mock data.`,
            error,
        );
    }

    // Fallback: Mock Data
    console.info(`[DataLoader] Using Mock Data for Scenario ${scenario}`);
    if (scenario === 'C') return scenarioC_Mock;
    return scenario === 'B' ? scenarioB_Mock : scenarioA_Mock;
}
