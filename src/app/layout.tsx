import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Tech Insights - Latest AI News, Tutorials & Guides",
  description:
    "Stay ahead with the latest AI technology insights, tutorials, tools reviews, and practical guides for developers and tech enthusiasts.",
  keywords: [
    "AI",
    "artificial intelligence",
    "machine learning",
    "LLM",
    "ChatGPT",
    "AI tools",
    "AI tutorials",
    "tech blog",
  ],
  authors: [{ name: "AI Tech Insights" }],
  openGraph: {
    type: "website",
    locale: "en_US",
    siteName: "AI Tech Insights",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="antialiased">
        <nav className="border-b border-gray-200 bg-white/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="max-w-4xl mx-auto px-6 py-4 flex items-center justify-between">
            <a href="/" className="text-xl font-bold text-gray-900">
              🤖 AI Tech Insights
            </a>
            <div className="flex gap-6 text-sm font-medium text-gray-600">
              <a href="/" className="hover:text-gray-900">
                Home
              </a>
              <a href="/blog" className="hover:text-gray-900">
                Blog
              </a>
              <a href="/about" className="hover:text-gray-900">
                About
              </a>
            </div>
          </div>
        </nav>
        <main className="max-w-4xl mx-auto px-6 py-8">{children}</main>
        <footer className="border-t border-gray-200 mt-16">
          <div className="max-w-4xl mx-auto px-6 py-8 text-center text-sm text-gray-500">
            © {new Date().getFullYear()} AI Tech Insights. Powered by AI.
            <br />
            <a
              href="mailto:contact@aitechinsights.dev"
              className="hover:text-gray-700"
            >
              Contact
            </a>
          </div>
        </footer>
      </body>
    </html>
  );
}
