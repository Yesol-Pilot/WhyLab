import { CausalAnalysisResult } from '@/types';
import { scenarioA_Mock, scenarioB_Mock } from '@/lib/mockData';

/**
 * 인과분석 결과 데이터를 로딩합니다.
 *
 * 우선순위:
 *  1. public/data/latest.json (엔진 파이프라인 산출물)
 *  2. Mock 데이터 (Fallback)
 */
export async function getCausalData(
    scenario: string = 'A',
): Promise<CausalAnalysisResult> {
    try {
        // 정적 export에서는 fetch로 public 폴더 접근
        const basePath =
            typeof window !== 'undefined'
                ? window.location.origin
                : '';
        const res = await fetch(`${basePath}/data/latest.json`);
        if (res.ok) {
            return (await res.json()) as CausalAnalysisResult;
        }
    } catch (error) {
        console.warn(
            '[DataLoader] latest.json 로드 실패. Mock 데이터를 사용합니다.',
            error,
        );
    }

    // Fallback: Mock Data
    console.info(`[DataLoader] Using Mock Data for Scenario ${scenario}`);
    return scenario === 'B' ? scenarioB_Mock : scenarioA_Mock;
}
