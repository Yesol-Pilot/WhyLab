import type { NextConfig } from "next";

// GitHub Pages 빌드 시에만 static export + basePath 적용
// 로컬 dev / Vercel에서는 rewrites 등 서버 기능 사용
const isStaticExport = process.env.GITHUB_PAGES === "1";

const nextConfig: NextConfig = {
  ...(!isStaticExport && {
    async rewrites() {
      return [
        {
          source: '/api/:path*',
          destination: `${process.env.NEXT_PUBLIC_API_URL || "http://localhost:4001"}/:path*`,
        },
      ]
    },
  }),
  // GitHub Pages 배포 시에만 static export + basePath 적용
  ...(isStaticExport ? { output: "export", basePath: "/WhyLab" } : {}),
  images: { unoptimized: true },
};

export default nextConfig;
