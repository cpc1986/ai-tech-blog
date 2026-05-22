import Link from "next/link";
import { getAllPosts } from "@/lib/posts";

export default function Home() {
  const posts = getAllPosts();
  const featured = posts[0];
  const rest = posts.slice(1, 9);

  return (
    <div>
      {/* Hero */}
      <section className="text-center py-20 mb-12">
        <div className="inline-block mb-4 text-xs font-semibold tracking-wider uppercase text-indigo-600 bg-indigo-50 px-4 py-1.5 rounded-full">
          AI-Powered Knowledge Hub
        </div>
        <h1 className="text-5xl md:text-6xl font-extrabold mb-5 tracking-tight">
          <span className="bg-gradient-to-r from-indigo-600 via-purple-500 to-pink-500 bg-clip-text text-transparent">
            AI Tech Insights
          </span>
        </h1>
        <p className="text-lg text-gray-500 max-w-xl mx-auto leading-relaxed">
          Your daily source for AI news, tutorials, tools, and practical guides.
          Stay ahead of the curve.
        </p>
      </section>

      {/* Featured Article */}
      {featured && (
        <section className="mb-14">
          <Link
            href={`/blog/${featured.slug}`}
            className="group block relative overflow-hidden rounded-2xl border border-gray-200 bg-white p-8 md:p-10 hover:border-indigo-200 transition-all duration-300 hover:shadow-lg hover:shadow-indigo-100/50"
          >
            <div className="flex items-center gap-3 mb-4">
              <span className="text-xs font-semibold text-indigo-600 bg-indigo-50 px-3 py-1 rounded-full">
                {featured.category}
              </span>
              <span className="text-xs font-medium text-indigo-500 bg-gradient-to-r from-indigo-50 to-purple-50 px-3 py-1 rounded-full">
                ⭐ Featured
              </span>
            </div>
            <h2 className="text-2xl md:text-3xl font-bold text-gray-900 mb-3 group-hover:text-indigo-600 transition-colors">
              {featured.title}
            </h2>
            <p className="text-gray-500 mb-4 leading-relaxed">
              {featured.excerpt}
            </p>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">
                {featured.date} · {featured.readingTime}
              </span>
              <span className="text-sm font-medium text-indigo-600 opacity-0 group-hover:opacity-100 transition-opacity">
                Read more →
              </span>
            </div>
          </Link>
        </section>
      )}

      {/* Recent Posts */}
      <section>
        <div className="flex items-center justify-between mb-8">
          <h2 className="text-2xl font-bold text-gray-900">Latest Articles</h2>
          <Link
            href="/blog"
            className="text-sm font-medium text-indigo-600 hover:text-indigo-700 transition-colors"
          >
            View all →
          </Link>
        </div>
        {rest.length === 0 && (
          <p className="text-gray-400 text-center py-12">
            More articles coming soon...
          </p>
        )}
        <div className="grid gap-5 md:grid-cols-2">
          {rest.map((post) => (
            <Link
              key={post.slug}
              href={`/blog/${post.slug}`}
              className="group block rounded-xl border border-gray-200 bg-white p-6 hover:border-indigo-200 hover:shadow-md hover:shadow-indigo-50 transition-all duration-300"
            >
              <span className="text-xs font-medium text-purple-600 bg-purple-50 px-2.5 py-1 rounded-full">
                {post.category}
              </span>
              <h3 className="text-lg font-semibold mt-3 mb-2 text-gray-900 group-hover:text-indigo-600 transition-colors">
                {post.title}
              </h3>
              <p className="text-sm text-gray-500 line-clamp-2 leading-relaxed">
                {post.excerpt}
              </p>
              <span className="text-xs text-gray-400 mt-3 block">
                {post.date} · {post.readingTime}
              </span>
            </Link>
          ))}
        </div>
      </section>

      {/* CTA */}
      <section className="mt-16 relative overflow-hidden rounded-2xl bg-gradient-to-br from-indigo-600 via-purple-600 to-pink-500 p-10 text-center text-white">
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHZpZXdCb3g9IjAgMCA2MCA2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZyBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPjxnIGZpbGw9IiNmZmYiIGZpbGwtb3BhY2l0eT0iMC4wNSI+PHBhdGggZD0iTTM2IDM0djItSDI0di0yaDEyem0wLTRWMjhIMjR2Mmgxem0tMTItNmgydi0ySDI0djJ6Ii8+PC9nPjwvZz48L3N2Zz4=')] opacity-30" />
        <div className="relative z-10">
          <h2 className="text-3xl font-bold mb-3">Stay Updated</h2>
          <p className="mb-6 text-indigo-100 max-w-md mx-auto">
            Bookmark this page and check back daily for new AI insights.
          </p>
          <Link
            href="/blog"
            className="inline-block bg-white text-indigo-600 font-semibold px-8 py-3 rounded-full hover:bg-indigo-50 transition-all shadow-lg shadow-indigo-900/20 hover:shadow-xl"
          >
            Browse All Articles →
          </Link>
        </div>
      </section>
    </div>
  );
}
