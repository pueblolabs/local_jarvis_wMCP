# src/ui/utils.py  (or tuck it inside MainWindow)
import html, re

URL_RE = re.compile(r'(https?://[^\s<>"]+)')
MD_LINK_RE = re.compile(r'\[([^\]]+)]\((https?://[^\s)]+)\)')

def linkify(text: str, shorten: bool = True) -> str:
    """Convert bare URLs & markdown links into HTML anchor tags."""
    # 1️⃣ markdown-style  [title](url) → <a href="url">title</a>
    def md_sub(match):
        title, url = match.groups()
        return f'<a href="{html.escape(url)}">{html.escape(title)}</a>'
    text = MD_LINK_RE.sub(md_sub, text)

    # 2️⃣ bare URLs → <a href="url">display</a>
    def url_sub(match):
        url = match.group(0)
        display = url
        if shorten and len(url) > 45:
            display = url[:25] + "…" + url[-12:]          # ▸ trim long slugs
        return f'<a href="{html.escape(url)}">{html.escape(display)}</a>'
    return URL_RE.sub(url_sub, text)