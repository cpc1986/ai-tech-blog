import { notFound } from "next/navigation";
import Link from "next/link";
import { getAllPosts, getPostBySlug } from "@/lib/posts";
import { remark } from "remark";
import html from "remark-html";

export async function generateStaticParams() {
  const posts = getAllPosts();
  return posts.map((post) => ({ slug: post.slug }));
}

async function markdownToHtml(markdown: string): Promise<string> {
  const result = await remark().use(html).process(markdown);
  return result.toString();
}

export default async function BlogPost({
  params,
}: {
  params: Promise<{ slug: string }>;
}) {
  const { slug } = await params;
  const post = getPostBySlug(slug);

  if (!post) {
    notFound();
  }

  const contentHtml = await markdownToHtml(post.content);

  return (
    <article className="max-w-3xl mx-auto">
      {/* Back link */}
      <Link
        href="/blog"
        className="inline-flex items-center gap-1.5 text-sm font-medium text-indigo-600 hover:text-indigo-700 mb-8 group"
      >
        <span className="group-hover:-translate-x-1 transition-transform">←</span>
        Back to Blog
      </Link>

      {/* Header */}
      <header className="mb-10">
        <div className="flex items-center gap-3 mb-4">
          <span className="text-xs font-semibold text-indigo-600 bg-indigo-50 px-3 py-1 rounded-full">
            {post.category}
          </span>
          <span className="text-sm text-gray-400">
            {post.date} · {post.readingTime}
          </span>
        </div>
        <h1 className="text-3xl md:text-4xl font-extrabold text-gray-900 leading-tight tracking-tight">
          {post.title}
        </h1>
      </header>

      {/* Article Body */}
      <div
        className="prose max-w-none"
        dangerouslySetInnerHTML={{ __html: contentHtml }}
      />

      {/* Tags */}
      {post.tags.length > 0 && (
        <div className="mt-12 pt-8 border-t border-gray-200">
          <div className="flex gap-2 flex-wrap">
            {post.tags.map((tag) => (
              <span
                key={tag}
                className="text-sm text-gray-500 bg-gray-50 border border-gray-200 px-3 py-1 rounded-full"
              >
                #{tag}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Bottom navigation */}
      <div className="mt-10 pt-8 border-t border-gray-200 flex justify-between items-center">
        <Link
          href="/blog"
          className="text-sm font-medium text-indigo-600 hover:text-indigo-700 transition-colors"
        >
          ← All Articles
        </Link>
        <Link
          href="/"
          className="text-sm font-medium text-gray-500 hover:text-gray-700 transition-colors"
        >
          Home →
        </Link>
      </div>
    </article>
  );
}
