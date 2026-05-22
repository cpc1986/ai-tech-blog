import type { Metadata } from "next";
import Link from "next/link";
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
      <body className="antialiased min-h-screen flex flex-col">
        {/* Navigation */}
        <nav className="sticky top-0 z-50 bg-white/70 backdrop-blur-xl border-b border-gray-100">
          <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
            <Link href="/" className="flex items-center gap-2.5 group">
              <span className="text-2xl">🤖</span>
              <span className="text-lg font-bold text-gray-900 tracking-tight group-hover:text-indigo-600 transition-colors">
                AI Tech Insights
              </span>
            </Link>
            <div className="flex gap-1">
              {[
                { href: "/", label: "Home" },
                { href: "/blog", label: "Blog" },
                { href: "/about", label: "About" },
              ].map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  className="px-4 py-2 text-sm font-medium text-gray-600 hover:text-indigo-600 hover:bg-indigo-50 rounded-lg transition-all"
                >
                  {link.label}
                </Link>
              ))}
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="flex-1 max-w-5xl mx-auto px-6 py-10 w-full">
          {children}
        </main>

        {/* Footer */}
        <footer className="border-t border-gray-100 bg-white/50">
          <div className="max-w-5xl mx-auto px-6 py-8 flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-gray-500">
            <span>© {new Date().getFullYear()} AI Tech Insights</span>
            <div className="flex gap-6">
              <Link
                href="/blog"
                className="hover:text-indigo-600 transition-colors"
              >
                Blog
              </Link>
              <Link
                href="/about"
                className="hover:text-indigo-600 transition-colors"
              >
                About
              </Link>
              <a
                href="mailto:contact@aitechinsights.dev"
                className="hover:text-indigo-600 transition-colors"
              >
                Contact
              </a>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
