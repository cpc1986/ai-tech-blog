import Link from "next/link";
import { getAllPosts } from "@/lib/posts";

export default function Home() {
  const posts = getAllPosts();
  const featured = posts[0];
  const rest = posts.slice(1, 9);

  return (
    <div>
      {/* Hero */}
      <section className="text-center py-16 mb-8">
        <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          AI Tech Insights
        </h1>
        <p className="text-xl text-gray-600 max-w-2xl mx-auto">
          Your daily source for AI news, tutorials, tools, and practical guides.
          Stay ahead of the curve.
        </p>
      </section>

      {/* Featured Article */}
      {featured && (
        <section className="mb-12">
          <Link
            href={`/blog/${featured.slug}`}
            className="block bg-white rounded-2xl shadow-sm border border-gray-200 p-8 hover:shadow-md transition-shadow"
          >
            <span className="text-xs font-medium text-blue-600 bg-blue-50 px-3 py-1 rounded-full">
              {featured.category} · Featured
            </span>
            <h2 className="text-2xl font-bold mt-3 mb-2">{featured.title}</h2>
            <p className="text-gray-600 mb-3">{featured.excerpt}</p>
            <span className="text-sm text-gray-400">
              {featured.date} · {featured.readingTime}
            </span>
          </Link>
        </section>
      )}

      {/* Recent Posts */}
      <section>
        <h2 className="text-2xl font-bold mb-6">Latest Articles</h2>
        {rest.length === 0 && (
          <p className="text-gray-500">More articles coming soon...</p>
        )}
        <div className="grid gap-6 md:grid-cols-2">
          {rest.map((post) => (
            <Link
              key={post.slug}
              href={`/blog/${post.slug}`}
              className="block bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow"
            >
              <span className="text-xs font-medium text-purple-600 bg-purple-50 px-2 py-0.5 rounded-full">
                {post.category}
              </span>
              <h3 className="text-lg font-semibold mt-2 mb-1">{post.title}</h3>
              <p className="text-sm text-gray-600 line-clamp-2">
                {post.excerpt}
              </p>
              <span className="text-xs text-gray-400 mt-2 block">
                {post.date} · {post.readingTime}
              </span>
            </Link>
          ))}
        </div>
      </section>

      {/* Newsletter CTA */}
      <section className="mt-16 bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl p-8 text-center text-white">
        <h2 className="text-2xl font-bold mb-2">Stay Updated</h2>
        <p className="mb-4 text-blue-100">
          Bookmark this page and check back daily for new AI insights.
        </p>
        <a
          href="/blog"
          className="inline-block bg-white text-blue-600 font-semibold px-6 py-2 rounded-full hover:bg-blue-50 transition"
        >
          Browse All Articles →
        </a>
      </section>
    </div>
  );
}
