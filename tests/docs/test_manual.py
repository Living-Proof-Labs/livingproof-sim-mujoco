# SPDX-License-Identifier: Apache-2.0
"""Exercise the delivered HTML over HTTP, including its actual assets."""

from functools import partial
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import threading
from urllib.parse import unquote, urlsplit

import pytest
from playwright.sync_api import sync_playwright

SITE = Path("artifacts/manual/html").resolve()


class Links(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.links = []
        self.ids = set()
        self.feed(path.read_text())

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for key in ("href", "src"):
            if key in attrs:
                self.links.append(attrs[key])


def test_all_local_pages_assets_and_fragments():
    pages = {p: Links(p) for p in SITE.rglob("*.html")}
    assert len(pages) >= 7
    for page, parsed in pages.items():
        for link in parsed.links:
            url = urlsplit(link)
            if url.scheme or url.netloc or link.startswith("data:"):
                continue
            local = unquote(url.path)
            if local.startswith("/"):
                assert local.startswith("/livingproof-sim-mujoco/"), (page, link)
                target = SITE / local.removeprefix("/livingproof-sim-mujoco/")
            else:
                target = (page.parent / local).resolve() if local else page
            if target.is_dir():
                target /= "index.html"
            assert target.is_relative_to(SITE), (page, link)
            assert target.exists(), (page, link)
            if url.fragment and target in pages:
                assert unquote(url.fragment) in pages[target].ids, (page, link)


def test_canonical_sources_and_installed_identity():
    import hashlib

    record = json.loads((SITE / "provenance.json").read_text())
    assert len(record["revision"]) == 40
    assert record["identity"]["provenance"] == "recorded-build"
    for name, digest in record["source_hashes"].items():
        assert hashlib.sha256((Path("docs") / name).read_bytes()).hexdigest() == digest
    assert (
        "https://docs.livingprooflabs.com/livingproof-sim-mujoco/"
        in (SITE / "index.html").read_text()
    )
    scenarios = (SITE / "features.html").read_text()
    assert "test_falling_body_advances" in scenarios
    assert "PASSED" in scenarios


def test_run_context_in_delivered_manual():
    record = json.loads((SITE / "provenance.json").read_text())
    if os.environ.get("GITHUB_ACTIONS") == "true":
        expected = (
            f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
            f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        )
    else:
        expected = "local; hosted publication unverified"
    assert record["ci_run"] == expected
    assert expected in (SITE / "context.html").read_text()


@pytest.fixture(scope="module")
def browser():
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=SITE)
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--no-sandbox"])
        yield browser, f"http://127.0.0.1:{server.server_port}"
        browser.close()
    server.shutdown()
    thread.join()


def test_navigation_search_and_loaded_assets(browser):
    engine, url = browser
    page = engine.new_page()
    failures = []
    page.on("requestfailed", lambda r: failures.append(r.url))
    page.on("response", lambda r: failures.append(r.url) if r.status >= 400 else None)
    page.set_default_timeout(8000)
    page.goto(url)
    page.locator(".md-nav--primary").get_by_text("Reference", exact=True).first.click()
    page.get_by_role("link", name="Commands", exact=True).first.click()
    page.wait_for_url("**/command.html")
    assert "--headless" in page.locator("main").inner_text()
    page.get_by_role("button", name="Search", exact=False).click()
    search = page.get_by_placeholder("Search", exact=True)
    search.fill("timestep")
    result = page.locator('dialog a[href*="features.html"]').first
    result.wait_for()
    result.click()
    page.wait_for_url("**/features.html*")
    assert not failures, failures
    page.screenshot(path="artifacts/manual/search.png", full_page=True)
    page.close()


def test_mobile_navigation_and_themes(browser):
    engine, url = browser
    page = engine.new_page(viewport={"width": 390, "height": 844}, color_scheme="dark")
    page.set_default_timeout(8000)
    page.goto(url)
    page.wait_for_function("document.body.getAttribute('data-md-color-scheme') === 'slate'")
    page.locator('[data-md-component="palette"] label:visible').first.click()
    page.wait_for_function("document.body.getAttribute('data-md-color-scheme') === 'default'")
    page.locator('[data-md-component="palette"] label:visible').first.click()
    page.wait_for_function("document.body.getAttribute('data-md-color-scheme') === 'slate'")
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    page.locator('label[for="__drawer"]:visible').first.click()
    page.get_by_role("link", name="Scenarios and results", exact=True).first.click()
    page.wait_for_url("**/features.html")
    page.screenshot(path="artifacts/manual/mobile-dark.png", full_page=True)
    page.close()


def test_published_paths_and_heading_fragments_survive():
    # Snapshot from reviewed preview of 9b10ec5, subsequently merged as 5ecc8d5.
    published = json.loads(Path("tests/docs/published-links.json").read_text())
    for filename, anchors in published.items():
        page = SITE / filename
        assert page.exists(), filename
        assert set(anchors) <= Links(page).ids, filename


@pytest.mark.parametrize("inherited", [False, True], ids=["host-env", "container-env"])
def test_native_task_reference_is_current(inherited):
    import subprocess

    env = dict(os.environ)
    if inherited:
        env["UV_PYTHON_DOWNLOADS"] = "never"
    else:
        env.pop("UV_PYTHON_DOWNLOADS", None)
    subprocess.run(["python", "tools/task_reference.py", "--check"], check=True, env=env)
    html = (SITE / "tasks.html").read_text()
    assert "task -t Taskfile.runtime.yml setup" in html
    assert "docker compose run" in html
    assert "VERSION=0.0.0" in html


def test_shared_navigation_and_identity_on_every_page():
    record = json.loads((SITE / "provenance.json").read_text())
    for path in SITE.glob("*.html"):
        html = path.read_text()
        assert 'aria-label="Documentation links"' in html
        assert "https://docs.livingprooflabs.com/" in Links(path).links
        assert (
            "https://github.com/Living-Proof-Labs/livingproof-sim-mujoco/tree/" + record["revision"]
        ) in Links(path).links
        assert "Help / report an issue" in html
        assert "Docs: development" in html
    assert "Package release v1.0.0" in (SITE / "release.html").read_text()


@pytest.mark.parametrize("width", [1280, 390], ids=["desktop", "mobile"])
def test_keyboard_links_and_deep_reload_without_scripts(browser, width):
    engine, url = browser
    context = engine.new_context(
        java_script_enabled=False, viewport={"width": width, "height": 844}
    )
    page = context.new_page()
    page.goto(url + "/tasks.html#prepared-environment")
    page.reload()
    assert page.locator("#prepared-environment").is_visible()
    footer = page.get_by_role("navigation", name="Documentation links")
    home = footer.get_by_role("link", name="Documentation home", exact=True)
    assert home.get_attribute("href") == "https://docs.livingprooflabs.com/"
    home.focus()
    page.keyboard.press("Tab")
    assert page.evaluate("document.activeElement.textContent.trim()") == "Source"
    license_link = footer.get_by_role("link", name="License", exact=True)
    license_link.focus()
    page.keyboard.press("Enter")
    page.wait_for_url("**/license.html")
    assert "Apache-2.0" in page.locator("main").inner_text()
    context.close()
