"""
crawl.py
--------
Command-line utility to crawl any URL using Crawl4AI, detect content type (sitemap, .txt, or regular page),
use the appropriate crawl method, and save the resulting Markdown files to the output directory.

Usage:
    python crawl.py <URL> [--output-dir ...] [--max-depth ...] [--max-concurrent ...]
"""

import argparse
import asyncio
import os
import re
import sys
from typing import Any, Dict, List
from urllib.parse import urldefrag, urlparse
from xml.etree import ElementTree

import requests
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    DefaultMarkdownGenerator,
    MemoryAdaptiveDispatcher,
)
from crawl4ai.content_filter_strategy import PruningContentFilter

prune_filter = PruningContentFilter(
    threshold=0.5,
    threshold_type="fixed",  # or "dynamic"
    min_word_threshold=50,
)


MD_GENERATOR = DefaultMarkdownGenerator(
    content_source="cleaned_html",  # default; another option is fit_html
    content_filter=prune_filter,
    options={
        "ignore_links": True,
        "ignore_images": True,
        "skip_internal_links": True,
    },
)


def url_to_filename(url: str) -> str:
    """Converts URL to safe filename preserving domain/path structure.

    Example:
        https://example.com/page/subpage -> example_com_page_subpage.md
        https://example.com/page?query=1 -> example_com_page_query_1.md
    """
    parsed = urlparse(url)

    # Extract domain and replace dots with underscores
    domain = parsed.netloc.replace(".", "_").replace(":", "_")

    # Extract path and clean it up
    path = parsed.path.strip("/")
    if path:
        # Replace slashes with underscores and clean special chars
        path = path.replace("/", "_").replace("\\", "_")
        # Remove or replace invalid filename characters
        path = re.sub(r'[<>:"|?*]', "_", path)
        filename = f"{domain}_{path}"
    else:
        filename = domain

    # Add query string if present (simplified)
    if parsed.query:
        query_clean = re.sub(r'[<>:"|?*&=]', "_", parsed.query)
        filename = f"{filename}_{query_clean}"

    # Add fragment if present (simplified)
    if parsed.fragment:
        fragment_clean = re.sub(r'[<>:"|?*#]', "_", parsed.fragment)
        filename = f"{filename}_{fragment_clean}"

    # Ensure filename isn't too long (max 255 chars for most filesystems)
    if len(filename) > 200:
        filename = filename[:200]

    # Remove trailing underscores and ensure it ends with .md
    filename = filename.strip("_")
    if not filename:
        filename = "index"

    return f"{filename}.md"


def save_markdown_file(url: str, markdown: str, output_dir: str) -> str:
    """Saves markdown content to a file with YAML frontmatter containing source URL.

    Args:
        url: Original source URL
        markdown: Markdown content to save
        output_dir: Directory to save the file

    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Generate filename from URL
    filename = url_to_filename(url)
    filepath = os.path.join(output_dir, filename)

    # Handle filename collisions by appending a number
    base_filepath = filepath
    counter = 1
    while os.path.exists(filepath):
        base_name = os.path.splitext(base_filepath)[0]
        filepath = f"{base_name}_{counter}.md"
        counter += 1

    # Write markdown with YAML frontmatter
    content = f"---\nsource_url: {url}\n---\n\n{markdown}"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    return filepath


def is_sitemap(url: str) -> bool:
    return url.endswith("sitemap.xml") or "sitemap" in urlparse(url).path


def is_txt(url: str) -> bool:
    return url.endswith(".txt")


async def crawl_recursive_internal_links(
    start_urls, max_depth=3, max_concurrent=5
) -> List[Dict[str, Any]]:
    """Recursive crawl using logic from 5-crawl_recursive_internal_links.py. Returns list of dicts with url and markdown."""
    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS, stream=False, markdown_generator=MD_GENERATOR
    )
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent,
    )

    visited = set()

    def normalize_url(url):
        return urldefrag(url)[0]

    current_urls = set([normalize_url(u) for u in start_urls])
    results_all = []

    # Get domain of the start URL
    start_domain = urlparse(next(iter(current_urls))).netloc

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for depth in range(max_depth):
            urls_to_crawl = [
                normalize_url(url)
                for url in current_urls
                if normalize_url(url) not in visited
            ]
            if not urls_to_crawl:
                break

            # Clean up some URLs that are not relevant
            urls_to_crawl = [
                url
                for url in urls_to_crawl
                if 
                start_domain in url
                and ".action" not in url
                and "$" not in url
                and "~" not in url
                and "login" not in url
                and "export" not in url
                and "exportword" not in url
                and "category" not in url
                and "label" not in url
                and "site" not in url
                and "display" not in url
            ]

            results = await crawler.arun_many(
                urls=urls_to_crawl, config=run_config, dispatcher=dispatcher
            )
            next_level_urls = set()

            for result in results:
                norm_url = normalize_url(result.url)
                visited.add(norm_url)

                if result.success and result.markdown:
                    results_all.append({"url": result.url, "markdown": result.markdown})
                    for link in result.links.get("internal", []):
                        next_url = normalize_url(link["href"])
                        if next_url not in visited:
                            next_level_urls.add(next_url)

            current_urls = next_level_urls

    return results_all


async def crawl_markdown_file(url: str) -> List[Dict[str, Any]]:
    """Crawl a .txt or markdown file using logic from 4-crawl_and_chunk_markdown.py."""
    browser_config = BrowserConfig(headless=True)
    crawl_config = CrawlerRunConfig(markdown_generator=MD_GENERATOR)

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=crawl_config)
        if result.success and result.markdown:
            return [{"url": url, "markdown": result.markdown}]
        else:
            print(f"Failed to crawl {url}: {result.error_message}")
            return []


def parse_sitemap(sitemap_url: str) -> List[str]:
    resp = requests.get(sitemap_url)
    urls = []

    if resp.status_code == 200:
        try:
            tree = ElementTree.fromstring(resp.content)
            urls = [loc.text for loc in tree.findall(".//{*}loc")]
        except Exception as e:
            print(f"Error parsing sitemap XML: {e}")

    return urls


async def crawl_batch(
    urls: List[str], max_concurrent: int = 10
) -> List[Dict[str, Any]]:
    """Batch crawl using logic from 3-crawl_sitemap_in_parallel.py."""
    browser_config = BrowserConfig(headless=True, verbose=False)
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS, stream=False, markdown_generator=MD_GENERATOR
    )
    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        results = await crawler.arun_many(
            urls=urls, config=crawl_config, dispatcher=dispatcher
        )
        return [
            {"url": r.url, "markdown": r.markdown}
            for r in results
            if r.success and r.markdown
        ]


def main():
    parser = argparse.ArgumentParser(
        description="Crawl websites and save markdown files"
    )
    parser.add_argument("url", help="URL to crawl (regular, .txt, or sitemap)")
    parser.add_argument(
        "--output-dir", default="./output", help="Output directory for markdown files"
    )
    parser.add_argument(
        "--max-depth", type=int, default=3, help="Recursion depth for regular URLs"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=3, help="Max parallel browser sessions"
    )
    args = parser.parse_args()

    # Detect URL type and crawl
    url = args.url
    if is_txt(url):
        print(f"Detected .txt/markdown file: {url}")
        crawl_results = asyncio.run(crawl_markdown_file(url))
    elif is_sitemap(url):
        print(f"Detected sitemap: {url}")
        sitemap_urls = parse_sitemap(url)
        if not sitemap_urls:
            print("No URLs found in sitemap.")
            sys.exit(1)
        print(f"Found {len(sitemap_urls)} URLs in sitemap. Crawling...")
        crawl_results = asyncio.run(
            crawl_batch(sitemap_urls, max_concurrent=args.max_concurrent)
        )
    else:
        print(f"Detected regular URL: {url}")
        crawl_results = asyncio.run(
            crawl_recursive_internal_links(
                [url], max_depth=args.max_depth, max_concurrent=args.max_concurrent
            )
        )

    if not crawl_results:
        print("No documents found to save.")
        sys.exit(1)

    # Save each crawled result to a markdown file
    saved_files = []
    for doc in crawl_results:
        url = doc["url"]
        markdown = doc["markdown"]
        if markdown:
            filepath = save_markdown_file(url, markdown, args.output_dir)
            saved_files.append(filepath)
            print(f"Saved: {filepath}")

    print(
        f"\nSuccessfully saved {len(saved_files)} markdown files to '{args.output_dir}'."
    )


if __name__ == "__main__":
    main()
