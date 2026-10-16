# verify_layout.py  —  run after: python -m playwright install chromium
from playwright.sync_api import sync_playwright

URLS = [
    "http://localhost:8501/?theme=dark",
    "http://localhost:8501/?theme=light",
]
VIEWPORTS = [(1920, 1080), (1536, 730), (400, 900)]
PAGES = ["Overview", "Rain Prediction", "Weather Analytics",
         "Model Performance", "Batch Predictions", "About the Project"]

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    for url in URLS:
        for w, h in VIEWPORTS:
            ctx = browser.new_context(viewport={"width": w, "height": h})
            pg = ctx.new_page()
            errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.goto(url, wait_until="networkidle")
            for name in PAGES:
                pg.get_by_text(name, exact=True).first.click()
                pg.wait_for_timeout(600)
                # 1) Title must be topmost at its own center
                title = pg.locator(".brand-title").first
                box = title.bounding_box()
                cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
                top = pg.evaluate(
                    "([x,y]) => { const el = document.elementFromPoint(x,y);"
                    " return el ? (el.closest('.brand-title') ? 'ok' : el.tagName + '.' + el.className) : 'none'; }",
                    [cx, cy],
                )
                assert top == "ok", f"{name} @ {w}x{h} occluded by {top}"
                # 2) Scroll down and back, recheck
                pg.mouse.wheel(0, 2000); pg.wait_for_timeout(300)
                pg.evaluate("window.scrollTo(0,0)"); pg.wait_for_timeout(400)
                assert title.is_visible(), f"{name} title not visible after scroll"
                # 3) No horizontal overflow
                ow = pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
                assert ow <= 1, f"{name} @ {w}x{h} horizontal overflow {ow}px"
            assert not errs, f"JS errors at {url} {w}x{h}: {errs}"
            ctx.close()
    browser.close()
print("All layout checks passed.")