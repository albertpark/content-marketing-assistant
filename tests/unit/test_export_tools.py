from src.utils.export_tools import blog_export_filename, blog_to_markdown, linkedin_post_to_text


def test_blog_to_markdown_includes_title_meta_and_body():
    blog_post = {
        "title": "My Post",
        "meta_description": "A short description.",
        "body_markdown": "## Section\n\nSome content.",
    }
    result = blog_to_markdown(blog_post)
    assert result == "# My Post\n\n*A short description.*\n\n## Section\n\nSome content."


def test_blog_to_markdown_handles_missing_fields():
    assert blog_to_markdown({}) == ""
    assert blog_to_markdown({"title": "Only Title"}) == "# Only Title"


def test_blog_export_filename_uses_slug():
    assert blog_export_filename({"slug": "my-post"}) == "my-post.md"


def test_blog_export_filename_falls_back_when_no_slug():
    assert blog_export_filename({}) == "blog-post.md"


def test_linkedin_post_to_text_combines_text_and_hashtags():
    linkedin_post = {"text": "Check this out!", "hashtags": ["#AI", "#ContentMarketing"]}
    assert linkedin_post_to_text(linkedin_post) == "Check this out!\n\n#AI #ContentMarketing"


def test_linkedin_post_to_text_handles_missing_hashtags():
    assert linkedin_post_to_text({"text": "Just text"}) == "Just text"


def test_linkedin_post_to_text_handles_empty_dict():
    assert linkedin_post_to_text({}) == ""
