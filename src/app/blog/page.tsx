import Link from "next/link";
import { getAllPosts } from "@/lib/posts";

export default function BlogPage() {
  const posts = getAllPosts();

  return (
    <div>
      {/* Header */}
      <div className="mb-10">
        <h1 className="text-3xl font-extrabold text-gray-900 mb-2">All Articles</h1>
        <p className="text-gray-500">
          {posts.length} article{posts.length !== 1 ? "s" : ""} about AI, machine learning, and technology
        </p>
      </div>

      {/* Posts List */}
      {posts.length === 0 ? (
        <div className="text-center py-20">
          <p className="text-gray-400 text-lg">Articles are being generated. Check back soon!</p>
        </div>
      ) : (
        <div className="flex flex-col gap-4">
          {posts.map((post) => (
            <Link
              key={post.slug}
              href={`/blog/${post.slug}`}
              className="group block rounded-xl border border-gray-200 bg-white p-6 hover:border-indigo-200 hover:shadow-md hover:shadow-indigo-50 transition-all duration-300"
            >
              <div className="flex items-center gap-3 mb-2">
                <span className="text-xs font-semibold text-indigo-600 bg-indigo-50 px-2.5 py-1 rounded-full">
                  {post.category}
                </span>
                <span className="text-xs text-gray-400">
                  {post.date} · {post.readingTime}
                </span>
              </div>
              <h2 className="text-xl font-semibold text-gray-900 mb-1.5 group-hover:text-indigo-600 transition-colors">
                {post.title}
              </h2>
              <p className="text-sm text-gray-500 line-clamp-2 leading-relaxed">
                {post.excerpt}
              </p>
              {post.tags.length > 0 && (
                <div className="flex gap-2 mt-3">
                  {post.tags.slice(0, 4).map((tag) => (
                    <span
                      key={tag}
                      className="text-xs text-gray-400 bg-gray-50 px-2 py-0.5 rounded"
                    >
                      #{tag}
                    </span>
                  ))}
                </div>
              )}
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}
