import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  basePath: "/ai-tech-blog",
  images: {
    unoptimized: true,
  },
  trailingSlash: true,
};

export default nextConfig;
