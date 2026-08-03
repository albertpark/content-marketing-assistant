"""Per-platform export formatting (Markdown, HTML, plain text, PDF, images)."""

from __future__ import annotations


def blog_to_markdown(blog_post: dict) -> str:
    """Renders a blog_post dict as a single Markdown document: H1 title, an
    italic meta-description line, then the body."""
    title = blog_post.get("title", "")
    meta_description = blog_post.get("meta_description", "")
    body = blog_post.get("body_markdown", "")

    parts = []
    if title:
        parts.append(f"# {title}")
    if meta_description:
        parts.append(f"*{meta_description}*")
    if body:
        parts.append(body)
    return "\n\n".join(parts)


def blog_export_filename(blog_post: dict) -> str:
    slug = blog_post.get("slug") or "blog-post"
    return f"{slug}.md"


def linkedin_post_to_text(linkedin_post: dict) -> str:
    """Renders a linkedin_post dict as plain text ready to paste into LinkedIn:
    the post body followed by a hashtag line."""
    text = linkedin_post.get("text", "")
    hashtags = " ".join(linkedin_post.get("hashtags", []))
    parts = [part for part in (text, hashtags) if part]
    return "\n\n".join(parts)
