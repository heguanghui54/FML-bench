"""Primary-source literature retrieval for the executable research DAG."""

from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any


class LiteratureProviderError(RuntimeError):
    pass


@dataclass
class OpenAlexArxivLiteratureProvider:
    timeout_seconds: int = 30
    user_agent: str = "fml-self-evolving-scientist/1.0"

    def _json(self, url: str) -> dict[str, Any]:
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise LiteratureProviderError(f"Primary-source metadata request failed: {exc}") from exc

    def _text(self, url: str) -> str:
        request = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read().decode("utf-8")
        except Exception as exc:
            raise LiteratureProviderError(f"arXiv metadata request failed: {exc}") from exc

    @staticmethod
    def _openalex_abstract(inverted: dict[str, list[int]] | None) -> str:
        if not inverted:
            return ""
        positions = sorted((position, token) for token, rows in inverted.items() for position in rows)
        return " ".join(token for _, token in positions)

    def search(self, query: str, *, limit: int = 8) -> list[dict[str, Any]]:
        if not query.strip():
            raise LiteratureProviderError("Literature query must not be empty")
        encoded = urllib.parse.quote(query)
        openalex_url = (
            "https://api.openalex.org/works?"
            f"search={encoded}&per-page={max(1, min(limit, 25))}&select=id,doi,title,publication_year,"
            "primary_location,authorships,abstract_inverted_index"
        )
        openalex = self._json(openalex_url)
        rows: list[dict[str, Any]] = []
        for work in openalex.get("results", []):
            location = work.get("primary_location") or {}
            source = location.get("source") or {}
            doi = work.get("doi")
            landing = location.get("landing_page_url") or doi or work.get("id")
            if not landing or not work.get("title"):
                continue
            record = {
                "provider": "OpenAlex",
                "provider_id": work.get("id"),
                "title": work["title"],
                "year": work.get("publication_year"),
                "doi": doi,
                "url": landing,
                "venue": source.get("display_name"),
                "authors": [
                    ((row.get("author") or {}).get("display_name"))
                    for row in work.get("authorships", [])
                    if (row.get("author") or {}).get("display_name")
                ],
                "abstract": self._openalex_abstract(work.get("abstract_inverted_index")),
            }
            record["record_sha256"] = hashlib.sha256(
                json.dumps(record, sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            rows.append(record)

        # arXiv is an independent primary metadata route and also covers recent
        # work that may not yet have complete DOI metadata.
        arxiv_url = (
            "https://export.arxiv.org/api/query?"
            f"search_query=all:{encoded}&start=0&max_results={max(1, min(limit, 10))}"
        )
        try:
            root = ET.fromstring(self._text(arxiv_url))
            ns = {"a": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("a:entry", ns):
                url = (entry.findtext("a:id", default="", namespaces=ns) or "").strip()
                title = " ".join((entry.findtext("a:title", default="", namespaces=ns) or "").split())
                if not url or not title:
                    continue
                record = {
                    "provider": "arXiv",
                    "provider_id": url.rsplit("/", 1)[-1],
                    "title": title,
                    "year": (entry.findtext("a:published", default="", namespaces=ns) or "")[:4] or None,
                    "doi": None,
                    "url": url,
                    "venue": "arXiv",
                    "authors": [
                        author.findtext("a:name", default="", namespaces=ns)
                        for author in entry.findall("a:author", ns)
                    ],
                    "abstract": " ".join(
                        (entry.findtext("a:summary", default="", namespaces=ns) or "").split()
                    ),
                }
                record["record_sha256"] = hashlib.sha256(
                    json.dumps(record, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest()
                rows.append(record)
        except LiteratureProviderError:
            if not rows:
                raise

        deduplicated: dict[str, dict[str, Any]] = {}
        for row in rows:
            key = str(row.get("doi") or row.get("title", "")).lower()
            deduplicated.setdefault(key, row)
        result = list(deduplicated.values())[:limit]
        if not result:
            raise LiteratureProviderError("No source-verifiable literature records were returned")
        return result
