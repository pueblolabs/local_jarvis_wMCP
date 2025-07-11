#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RegNavigator Lite Core Logic – AI briefing generator for US federal dockets
(Refactored from Streamlit app to be a callable library)
"""
import asyncio, html, io, json, os, re, textwrap
from collections import Counter
from datetime import datetime
from enum import Enum
from pathlib import Path

import httpx
import pandas as pd
import pdfplumber
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from tenacity import retry, wait_exponential, stop_after_attempt

import google.generativeai as genai
from google.generativeai.types import (
    GenerationConfig,
    HarmCategory,
    HarmBlockThreshold,
)
from pydantic import BaseModel

REGS_API_KEY = os.getenv("REGS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

REGS_BASE = "https://api.regulations.gov/v4"
HTTP_TIMEOUT = 30.0
API_DELAY = 0.6

REGS_SEM = asyncio.Semaphore(10)
GEMINI_SEM = asyncio.Semaphore(10)
PDF_SEM = asyncio.Semaphore(4)

class Stance(str, Enum):
    SUPPORT = "SUPPORT"
    OPPOSE = "OPPOSE"
    NEUTRAL = "NEUTRAL"

class CommentAnalysis(BaseModel):
    organization: str
    summary: str
    stance: Stance | None = None 
    keyphrases: list[str] | None = None

@retry(wait=wait_exponential(1, 2, 10), stop=stop_after_attempt(4))
async def _api_get(endpoint: str, params: dict | None = None) -> dict:
    if not REGS_API_KEY:
        raise ValueError("REGS_API_KEY is not set.")
    async with REGS_SEM:
        async with httpx.AsyncClient(
            base_url=REGS_BASE,
            headers={"X-Api-Key": REGS_API_KEY},
            timeout=HTTP_TIMEOUT,
        ) as c:
            await asyncio.sleep(API_DELAY)
            r = await c.get(endpoint, params=params or {})
            r.raise_for_status()
            return r.json()

async def _download_pdf(url: str) -> bytes | None:
    async with PDF_SEM, httpx.AsyncClient(timeout=60) as c:
        try:
            r = await c.get(url)
            r.raise_for_status()
            return r.content
        except Exception:
            return None

async def extract_pdf_text(url: str) -> str | None:
    data = await _download_pdf(url)
    if not data:
        return None
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "".join(p.extract_text() or "" for p in pdf.pages)
    except Exception:
        return None

def clean_inline(txt: str | None) -> str | None:
    if not txt or re.match(r"^\s*(see|please\s+see)\s+attached", txt, re.I):
        return None
    return html.unescape(txt.strip())

SAFE_NONE = {
    c: HarmBlockThreshold.BLOCK_NONE
    for c in [
        HarmCategory.HARM_CATEGORY_HARASSMENT,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
    ]
}
MODEL = genai.GenerativeModel("gemini-2.0-flash-lite") if GEMINI_API_KEY else None

async def gemini_json(prompt: str, schema) -> dict | None:
    """
    Call Gemini and parse JSON. Removes any 'default' keys from the Pydantic
    schema because the Gemini Responses API rejects them.

    Args:
        prompt:   prompt text
        schema:   a pydantic BaseModel class (e.g. CommentAnalysis)

    Returns:
        Parsed dict or None on failure.
    """

    def _strip_disallowed(obj: dict | list) -> None:
        """Recursively delete keys Gemini doesn't accept ('default', '$defs', 'title', 'description', 'anyOf', 'oneOf', 'allOf')."""
        if isinstance(obj, dict):
            for bad in ("default", "$defs", "title", "description", "anyOf", "oneOf", "allOf"):
                obj.pop(bad, None)
            for v in obj.values():
                _strip_disallowed(v)
        elif isinstance(obj, list):
            for item in obj:
                _strip_disallowed(item)

    if not MODEL:
        raise ConnectionError("Gemini model not initialized. Check API Key.")

    # Build schema dict without 'default' entries
    schema_dict = schema.model_json_schema()
    _strip_disallowed(schema_dict)

    async with GEMINI_SEM:
        try:
            r = await MODEL.generate_content_async(
                prompt,
                generation_config=GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=schema_dict,
                ),
                safety_settings=SAFE_NONE,
            )
            return json.loads(r.text)
        except json.JSONDecodeError as bad_json:
            print(f"Gemini JSON error: {bad_json}")
            return None
        except Exception as e:
            print(f"Gemini JSON error (other): {e}")
            return None

async def gemini_bullets(prompt: str) -> str | None:
    if not MODEL: raise ConnectionError("Gemini model not initialized. Check API Key.")
    async with GEMINI_SEM:
        try:
            r = await MODEL.generate_content_async(
                prompt,
                generation_config=GenerationConfig(max_output_tokens=250),
                safety_settings=SAFE_NONE,
            )
            return r.text.strip()
        except Exception as e:
            print(f"Gemini bullets error: {e}")
            return None

async def process_comment(cid: str) -> dict | None:
    try:
        det = await _api_get("/comments/" + cid, {"include": "attachments"})
        attr = det["data"]["attributes"]
        attachments = det.get("included", [])
        text = None
        for att in attachments:
            for f in att.get("attributes", {}).get("fileFormats", []):
                if f.get("format", "").lower() == "pdf" and f.get("fileUrl"):
                    text = await extract_pdf_text(f["fileUrl"])
                    break
            if text:
                break
        if not text: text = clean_inline(attr.get("comment"))
        if not text: return None
        payload = await gemini_json(
            "Extract JSON: organization, ≤30-word summary, stance (SUPPORT/OPPOSE/NEUTRAL), 2-5 keyphrases.\n###\n"
            + text[:25000],
            CommentAnalysis,
        )
        if not payload: return None
        return {"id": cid, "title": attr.get("title") or "N/A", **CommentAnalysis(**payload).model_dump()}
    except Exception as e:
        print(f"⚠️ Comment processing failed for {cid}: {e}")
        return None

async def build_brief(summaries: list[str], k: dict) -> str:
    mood = "supportive" if k["sentiment"] > 0.25 else "opposed" if k["sentiment"] < -0.25 else "mixed"
    prompt = textwrap.dedent(f'Produce 3-5 Markdown bullet points (≤ 20 words each) summarising overall {mood} sentiment, top reasons, and any notable outlier.\n---\n{" ".join(summaries[:200])}').strip()
    return await gemini_bullets(prompt) or "- Brief unavailable"

def build_pdf_data(df: pd.DataFrame, k: dict, docket: str, title: str, brief_md: str) -> bytes:
    bullets = [ln.lstrip("-• ").strip() for ln in brief_md.splitlines() if ln.strip().startswith(("-", "•"))]
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    w, h = letter
    m = 50
    y = h - m
    c.setFont("Helvetica-Bold", 16)
    c.drawString(m, y, f"Docket {docket}")
    y -= 20
    c.setFont("Helvetica", 12)
    c.drawString(m, y, title[:95])
    y -= 24
    if bullets:
        c.setFont("Helvetica-Bold", 11)
        c.drawString(m, y, "Key takeaways:")
        y -= 14
        c.setFont("Helvetica-Oblique", 9)
        for ln in bullets:
            c.drawString(m + 10, y, f"• {ln[:110]}")
            y -= 12
        y -= 4
    c.setFont("Helvetica-Bold", 11)
    c.drawString(m, y, f"Comments {k['total']} | Orgs {k['unique_orgs']} | S/O/N {k['support']}/{k['oppose']}/{k['neutral']} | Sentiment {k['sentiment']}")
    y -= 18
    c.line(m, y, w - m, y)
    y -= 14
    c.setFont("Helvetica", 9)
    for _, row in df.iterrows():
        record_lines = [f"Organization: {row['organization']}", f"Stance: {row['stance']}"]
        wrapped = textwrap.wrap(str(row["summary"]), width=95)
        record_lines.append(f"Summary: {wrapped[0] if wrapped else ''}")
        record_lines.extend([" " * 9 + ln for ln in wrapped[1:]])
        for ln in record_lines:
            if y < m + 40:
                c.showPage()
                y = h - m
                c.setFont("Helvetica", 9)
            c.drawString(m, y, ln)
            y -= 12
        y -= 6
    c.setFont("Helvetica-Oblique", 8)
    c.drawString(m, m - 10, f"© {datetime.now():%Y-%m-%d %H:%M}")
    c.save()
    buf.seek(0)
    return buf.getvalue()

async def analyse(docket_id: str, target: int = 25): # Limit to 25 comments for demo speed
    if not REGS_API_KEY or not GEMINI_API_KEY:
        raise ValueError("API keys for Regulations.gov or Gemini are missing.")

    docs = await _api_get("/documents", {"filter[docketId]": docket_id, "page[size]": 250})
    if not docs["data"]: raise FileNotFoundError(f"No documents found for docket ID {docket_id}")

    primary = next((d for d in docs["data"] if d["attributes"]["documentType"] == "Proposed Rule"), docs["data"][0])
    title = primary["attributes"]["title"]
    obj_id = primary["attributes"]["objectId"]

    ids = []
    page_params = {"filter[commentOnId]": obj_id, "page[size]": min(250, target * 2), "page[number]": 1}
    while len(ids) < target:
        batch = await _api_get("/comments", page_params)
        if not batch.get("data"): break
        ids.extend(c["id"] for c in batch["data"])
        if not batch["meta"]["hasNextPage"]: break
        page_params["page[number]"] += 1
    ids = ids[:min(250, target * 2)]

    tasks = [process_comment(cid) for cid in ids]
    results_raw = await asyncio.gather(*tasks)
    results = [r for r in results_raw if r]

    if not results: raise ValueError("Could not process any comments to generate a summary.")
    if len(results) < target: print(f"Warning: Only {len(results)} of the targeted {target} comments had usable text.")
    
    df = pd.DataFrame(results[:target])
    sent_map = {"SUPPORT": 1, "NEUTRAL": 0, "OPPOSE": -1}
    # Compute KPI counts and sentiment (tolerates missing stance)
    stance_scores = df["stance"].map(sent_map).fillna(0)
    kpis = {
        "total": len(df),
        "unique_orgs": df["organization"].nunique(),
        "support": (df["stance"] == "SUPPORT").sum(),
        "oppose": (df["stance"] == "OPPOSE").sum(),
        "neutral": (df["stance"] == "NEUTRAL").sum(),
        "sentiment": round(stance_scores.mean() if len(df) else 0, 2),
    }
    brief = await build_brief(df["summary"].tolist(), kpis) if len(df) else "- No comments summarised"
    
    pdf_data = build_pdf_data(df, kpis, docket_id, title, brief)
    return brief, pdf_data

async def generate_briefing(docket_id: str, download_folder: str) -> tuple[str, str]:
    """Top-level function to run the analysis and save the PDF."""
    print(f"Starting analysis for docket: {docket_id}")
    summary_bullets, pdf_data = await analyse(docket_id)
    
    pdf_filename = f"{docket_id.replace('/', '_')}_briefing.pdf"
    pdf_filepath = os.path.join(download_folder, pdf_filename)
    
    with open(pdf_filepath, "wb") as f: f.write(pdf_data)
    print(f"PDF briefing saved to: {pdf_filepath}")
    return summary_bullets, pdf_filepath