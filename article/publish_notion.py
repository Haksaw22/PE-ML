"""Publish article/draft.md to Notion as a child page of Blog Base Page.

- Markdown subset: #/##/### headings, paragraphs, - bullets, --- dividers,
  $$block$$ and $inline$ math, **bold**, *italic*, `code`, [links](url), ![img](path).
- Local images are uploaded via the Notion File Upload API.
- Relative/placeholder links render as anchor text + (repo path) in code style until
  the GitHub repo is public; set REPO_BASE env to emit real links.
- Idempotent: archives any existing child page with the same title first.
Run: .venv/Scripts/python.exe article/publish_notion.py
"""
import json
import os
import re
import time
import urllib.request

TOKEN = open(os.path.expanduser("~/.notion_token")).read().strip()
PARENT = "24235890-b98b-8064-b6eb-d55f469400ba"  # Blog Base Page
NV = "2022-06-28"
REPO_BASE = os.environ.get("REPO_BASE", "")  # e.g. https://github.com/user/repo/blob/main
SRC = "article/draft.md"
TITLE = "Persistency of Excitation: The Geometry of Asking Good Questions"


def api(method, path, payload=None, headers_extra=None, raw=None):
    url = f"https://api.notion.com/v1/{path}"
    headers = {"Authorization": f"Bearer {TOKEN}", "Notion-Version": NV}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode()
    if raw is not None:
        data = raw
    if headers_extra:
        headers.update(headers_extra)
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def upload_file(path):
    """Single-part Notion file upload; returns file_upload id."""
    name = os.path.basename(path)
    ctype = "image/gif" if name.endswith(".gif") else "image/png"
    fu = api("POST", "file_uploads", {"filename": name, "content_type": ctype})
    boundary = "----mdboundary7429"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
        f"filename=\"{name}\"\r\nContent-Type: {ctype}\r\n\r\n"
    ).encode() + open(path, "rb").read() + f"\r\n--{boundary}--\r\n".encode()
    api("POST", f"file_uploads/{fu['id']}/send", raw=body,
        headers_extra={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return fu["id"]


# ------------------------- inline rich-text parser ---------------------------
def rich(text, bold=False, italic=False):
    """Parse inline markdown into Notion rich_text objects."""
    out = []
    # tokenize by inline code, inline math, links, bold, italic — in that priority
    pattern = re.compile(
        r"(`[^`]+`)|(\$[^$]+\$)|(\[[^\]]+\]\([^)]+\))|(\*\*[^*]+\*\*)|(\*[^*]+\*)")
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            out.append(_txt(text[pos:m.start()], bold, italic))
        tok = m.group(0)
        if tok.startswith("`"):
            out.append(_txt(tok[1:-1], bold, italic, code=True))
        elif tok.startswith("$"):
            out.append({"type": "equation", "equation": {"expression": tok[1:-1]},
                        "annotations": _ann(bold, italic, False)})
        elif tok.startswith("["):
            lm = re.match(r"\[([^\]]+)\]\(([^)]+)\)", tok)
            label, url = lm.group(1), lm.group(2)
            if url.startswith("http") and "【" not in url:
                out.append({"type": "text", "text": {"content": label, "link": {"url": url}},
                            "annotations": _ann(bold, italic, False)})
            elif REPO_BASE and not url.startswith("http"):
                full = REPO_BASE.rstrip("/") + "/" + url.lstrip("./").replace("../", "")
                out.append({"type": "text", "text": {"content": label, "link": {"url": full}},
                            "annotations": _ann(bold, italic, False)})
            else:
                out.append(_txt(label, bold, italic))
                out.append(_txt(f" ({url.replace('../', '').replace('【repo-url】', 'repo link coming')})",
                                bold, italic, code=True))
        elif tok.startswith("**"):
            out.extend(rich(tok[2:-2], bold=True, italic=italic))
        else:
            out.extend(rich(tok[1:-1], bold=bold, italic=True))
        pos = m.end()
    if pos < len(text):
        out.append(_txt(text[pos:], bold, italic))
    return [r for r in out if r.get("type") != "text" or r["text"]["content"]]


def _ann(b, i, c):
    return {"bold": b, "italic": i, "code": c, "strikethrough": False,
            "underline": False, "color": "default"}


def _txt(content, b=False, i=False, code=False):
    return {"type": "text", "text": {"content": content}, "annotations": _ann(b, i, code)}


# ------------------------------ block parser ---------------------------------
def blocks_from_markdown(md):
    lines = md.split("\n")
    blocks, i = [], 0
    para = []

    def flush():
        nonlocal para
        if para:
            text = " ".join(para).strip()
            if text:
                blocks.append({"object": "block", "type": "paragraph",
                               "paragraph": {"rich_text": rich(text)}})
            para = []

    while i < len(lines):
        ln = lines[i]
        s = ln.strip()
        if s.startswith("$$"):
            flush()
            expr = s.strip("$").strip()
            if not expr:  # multiline $$ ... $$
                j = i + 1
                acc = []
                while j < len(lines) and not lines[j].strip().startswith("$$"):
                    acc.append(lines[j]); j += 1
                expr = " ".join(a.strip() for a in acc)
                i = j
            blocks.append({"object": "block", "type": "equation",
                           "equation": {"expression": expr}})
        elif s.startswith("### "):
            flush(); blocks.append({"object": "block", "type": "heading_3",
                                    "heading_3": {"rich_text": rich(s[4:])}})
        elif s.startswith("## "):
            flush(); blocks.append({"object": "block", "type": "heading_2",
                                    "heading_2": {"rich_text": rich(s[3:])}})
        elif s.startswith("# "):
            flush()  # page title handled separately; skip H1
        elif s == "---":
            flush(); blocks.append({"object": "block", "type": "divider", "divider": {}})
        elif s.startswith("- "):
            flush()
            blocks.append({"object": "block", "type": "bulleted_list_item",
                           "bulleted_list_item": {"rich_text": rich(s[2:])}})
        elif s.startswith("!["):
            flush()
            m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", s)
            path = os.path.join("article", m.group(2))
            try:
                fid = upload_file(path)
                blocks.append({"object": "block", "type": "image",
                               "image": {"type": "file_upload",
                                         "file_upload": {"id": fid},
                                         "caption": rich(m.group(1))}})
                print(f"  uploaded {path}")
            except Exception as e:
                print(f"  IMAGE SKIPPED {path}: {e}")
        elif s == "":
            flush()
        else:
            para.append(s)
        i += 1
    flush()
    return blocks


def main():
    md = open(SRC, encoding="utf-8").read()
    # strip the internal draft-note preamble (italic line + following divider)
    md = re.sub(r"\*Draft v2[^\n]*\n[^\n]*\n\n---\n", "", md, count=1)

    # archive prior version if present
    kids = api("GET", f"blocks/{PARENT}/children?page_size=100")
    for k in kids.get("results", []):
        if k.get("type") == "child_page" and k["child_page"]["title"] == TITLE:
            api("PATCH", f"blocks/{k['id']}", {"archived": True})
            print("archived previous version")

    blocks = blocks_from_markdown(md)
    print(f"built {len(blocks)} blocks")
    page = api("POST", "pages", {
        "parent": {"page_id": PARENT},
        "properties": {"title": {"title": [_txt(TITLE)]}},
        "children": blocks[:100],
    })
    for start in range(100, len(blocks), 100):
        api("PATCH", f"blocks/{page['id']}/children", {"children": blocks[start:start + 100]})
        time.sleep(0.4)
    print("PUBLISHED:", page.get("url"))


if __name__ == "__main__":
    main()
