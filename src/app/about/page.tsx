import Link from "next/link";

export default function About() {
  return (
    <div className="max-w-2xl">
      <h1 className="text-3xl font-bold mb-6">About AI Tech Insights</h1>
      
      <div className="prose max-w-none">
        <p>
          <strong>AI Tech Insights</strong> is an AI-powered technology blog
          dedicated to bringing you the latest news, tutorials, and practical
          guides about artificial intelligence, machine learning, and emerging
          tech.
        </p>

        <h2>What We Cover</h2>
        <ul>
          <li><strong>AI News & Trends</strong> — Latest developments in the AI industry</li>
          <li><strong>Tutorials & How-Tos</strong> — Step-by-step guides for AI tools and frameworks</li>
          <li><strong>Tool Reviews</strong> — Honest reviews of AI products and services</li>
          <li><strong>Industry Analysis</strong> — Deep dives into AI market trends and opportunities</li>
          <li><strong>Developer Resources</strong> — Code snippets, APIs, and technical references</li>
        </ul>

        <h2>Our Mission</h2>
        <p>
          We believe AI knowledge should be accessible to everyone. Our content
          is crafted to help developers, entrepreneurs, and tech enthusiasts
          stay informed and make the most of AI technology.
        </p>

        <h2>Contact</h2>
        <p>
          Have questions or suggestions? Reach out to us at{" "}
          <a href="mailto:contact@aitechinsights.dev">
            contact@aitechinsights.dev
          </a>
        </p>
      </div>

      <div className="mt-8">
        <Link
          href="/blog"
          className="inline-block bg-blue-600 text-white px-6 py-2 rounded-full hover:bg-blue-700 transition"
        >
          Read Our Articles →
        </Link>
      </div>
    </div>
  );
}
