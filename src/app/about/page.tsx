export default function About() {
  return (
    <div className="max-w-2xl">
      <h1 className="text-3xl font-extrabold text-gray-900 mb-6">
        About AI Tech Insights
      </h1>

      <div className="prose max-w-none">
        <p>
          <strong>AI Tech Insights</strong> is an AI-powered technology blog
          dedicated to bringing you the latest news, tutorials, and practical
          guides about artificial intelligence, machine learning, and emerging
          tech.
        </p>

        <h2>What We Cover</h2>
        <ul>
          <li>
            <strong>AI News & Trends</strong> — Latest developments in the AI
            industry
          </li>
          <li>
            <strong>Tutorials & How-Tos</strong> — Step-by-step guides for AI
            tools and frameworks
          </li>
          <li>
            <strong>Tool Reviews</strong> — Honest reviews of AI products and
            services
          </li>
          <li>
            <strong>Industry Analysis</strong> — Deep dives into AI market
            trends and opportunities
          </li>
          <li>
            <strong>Developer Resources</strong> — Code snippets, APIs, and
            technical references
          </li>
        </ul>

        <h2>Our Mission</h2>
        <p>
          We believe AI knowledge should be accessible to everyone. Our content
          is crafted to help developers, entrepreneurs, and tech enthusiasts stay
          informed and make the most of AI technology.
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
        <a
          href="/blog"
          className="inline-flex items-center gap-2 bg-indigo-600 text-white px-6 py-2.5 rounded-full hover:bg-indigo-700 transition-colors font-medium shadow-sm"
        >
          Read Our Articles
          <span>→</span>
        </a>
      </div>
    </div>
  );
}
