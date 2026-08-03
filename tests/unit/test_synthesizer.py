from src.workflow.synthesizer import synthesize


def test_synthesize_assembles_package():
    state = {
        "blog_post": {
            "title": "My Post",
            "slug": "my-post",
            "body_markdown": "# Hello",
            "meta_description": "A post.",
        },
        "linkedin_post": {"text": "Check out my post!", "hashtags": ["#AI"]},
        "image_assets": [
            {"url": None, "path": None, "prompt": "p", "provider_used": "stub", "alt_text": "My Post"}
        ],
        "research_findings": [
            {"source": "serpapi", "title": "Source A", "url": "https://a.example", "snippet": "..."},
            {"source": "serpapi", "title": "No URL", "url": "", "snippet": "..."},
        ],
    }
    result = synthesize(state)
    package = result["content_package"]

    assert package["blog"]["title"] == "My Post"
    assert package["blog"]["slug"] == "my-post"
    assert package["linkedin"]["text"] == "Check out my post!"
    assert len(package["images"]) == 1
    assert package["citations"] == [{"title": "Source A", "url": "https://a.example"}]


def test_synthesize_handles_missing_fields():
    result = synthesize({})
    package = result["content_package"]

    assert package["blog"]["title"] == ""
    assert package["linkedin"]["text"] == ""
    assert package["images"] == []
    assert package["citations"] == []
