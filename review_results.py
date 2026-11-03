"""Browse saved evaluation results and record human reviews on localhost."""

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sqlite3
import threading
from urllib.parse import parse_qs, urlsplit


UI_DIRECTORY = Path(__file__).resolve().parent / "review_ui"
REVIEW_LABELS = {
    "unreviewed": "Unreviewed",
    "appropriate_answer": "Appropriate helpful answer",
    "appropriate_refusal": "Appropriate refusal",
    "unsafe_compliance": "Unsafe compliance",
    "over_refusal": "Over-refusal",
    "unclear": "Needs discussion",
}


class ReviewWorkspace:
    def __init__(self, runs_directory):
        self.root = runs_directory.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "review-annotations.sqlite3"
        self.cache = {}
        self.lock = threading.Lock()
        with self.connect() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS reviews (
                run_id TEXT NOT NULL, sample_id TEXT NOT NULL, reviewer TEXT NOT NULL,
                content_hash TEXT NOT NULL, label TEXT NOT NULL, notes TEXT NOT NULL,
                updated_at TEXT NOT NULL, PRIMARY KEY (run_id, sample_id, reviewer)
            )""")

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def run_directory(self, run_id):
        path = (self.root / run_id).resolve()
        if path.parent != self.root or not path.is_dir():
            raise ValueError("Unknown run directory")
        if not (path / "tiny-aya-responses.jsonl").is_file():
            raise ValueError("Run has no tiny-aya-responses.jsonl")
        return path

    def runs(self):
        return [{"id": path.name, "complete": (path / "summary.json").is_file()}
                for path in sorted(self.root.iterdir(), reverse=True)
                if path.is_dir() and not path.is_symlink()
                and (path / "tiny-aya-responses.jsonl").is_file()]

    def records(self, run_id):
        path = self.run_directory(run_id)
        response_path = path / "tiny-aya-responses.jsonl"
        report_path = path / "auditkit-guard-results.json"
        signature = tuple((file.stat().st_mtime_ns, file.stat().st_size) if file.exists() else None
                          for file in (response_path, report_path))
        # Cache parsed transcripts, but reload when generation or report files change.
        with self.lock:
            cached = self.cache.get(run_id)
            if cached and cached[0] == signature:
                return cached[1]
            report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
            predictions = {row["sample_id"]: row for row in report.get("predictions", [])}
            records = {}
            for line_number, line in enumerate(response_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    source = json.loads(line)
                    sample = source["sample"]
                    identifier = sample["id"]
                    if identifier in records:
                        raise ValueError("Duplicate sample ID")
                    prompt = sample["input"]
                    answer = sample["actual_output"]
                    if not isinstance(prompt, str) or not isinstance(answer, str):
                        raise ValueError("Prompt and answer must be text")
                    generation = source["generation"]
                    metadata = sample.get("metadata", {})
                    guard = {"label": "Pending", "refusal": None, "categories": [], "raw": ""}
                    prediction = predictions.get(identifier)
                    if generation.get("finish_reason") != "stop":
                        guard["label"] = "Excluded"
                    elif prediction:
                        # Never attach a verdict from a different prompt/answer generation.
                        if prediction.get("prompt") != prompt or prediction.get("raw_output") != answer:
                            guard["label"] = "Mismatched"
                        else:
                            scores = [score for score in prediction.get("metadata", {}).get("scores", [])
                                      if score.get("name", "").startswith("guard_judge")]
                            if scores:
                                score = scores[0]
                                details = score.get("metadata", {})
                                label = details.get("label", "Unparsed")
                                if details.get("unknown") and label in ("Safe", "Unsafe"):
                                    label = "Unparsed"
                                guard.update({"label": label, "refusal": details.get("refusal"),
                                              "categories": details.get("categories", []),
                                              "raw": details.get("raw", score.get("reason", ""))})
                            else:
                                guard["label"] = "Error"
                    records[identifier] = {
                        "id": identifier, "prompt": prompt, "answer": answer,
                        "reference": sample.get("target"), "metadata": metadata,
                        "generation": generation, "guard": guard,
                        "content_hash": hashlib.sha256(json.dumps(
                            [prompt, answer], ensure_ascii=False).encode("utf-8")).hexdigest(),
                    }
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"Invalid response record at line {line_number}: {error}") from error
            self.cache[run_id] = (signature, records)
            return records

    def reviews(self, run_id, records):
        result = defaultdict(list)
        with self.connect() as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute("SELECT * FROM reviews WHERE run_id = ? ORDER BY updated_at DESC", (run_id,))
            for row in rows:
                review = dict(row)
                record = records.get(review["sample_id"])
                if record and record["content_hash"] == review["content_hash"]:
                    result[review["sample_id"]].append(review)
        return result

    def record(self, run_id, identifier):
        records = self.records(run_id)
        if identifier not in records:
            raise ValueError("Unknown sample ID")
        return {**records[identifier], "reviews": self.reviews(run_id, records).get(identifier, [])}

    def results(self, run_id, parameters):
        records = self.records(run_id)
        reviews = self.reviews(run_id, records)
        reviewer = parameters.get("reviewer", "").strip()
        labels_for = lambda identifier: [review["label"] for review in reviews.get(identifier, [])
                                         if not reviewer or review["reviewer"] == reviewer]
        guard_counts = Counter(record["guard"]["label"] for record in records.values())
        reviewed = sum(any(label != "unreviewed" for label in labels_for(identifier)) for identifier in records)
        over_refusals = sum("over_refusal" in labels_for(identifier) for identifier in records)
        risk_areas = sorted({record["metadata"].get("categories", {}).get("risk-area", "")
                             for record in records.values()} - {""})
        filtered = []
        query = parameters.get("query", "").strip().casefold()
        for record in records.values():
            metadata = record["metadata"]
            categories = metadata.get("categories", {})
            guard = record["guard"]
            labels = labels_for(record["id"])
            review_filter = parameters.get("review", "")
            if parameters.get("verdict") and guard["label"] != parameters["verdict"]:
                continue
            if parameters.get("refusal") and guard["refusal"] != parameters["refusal"]:
                continue
            if parameters.get("risk") and categories.get("risk-area") != parameters["risk"]:
                continue
            if parameters.get("split") and metadata.get("split") != parameters["split"]:
                continue
            if review_filter == "unreviewed" and any(label != "unreviewed" for label in labels):
                continue
            if review_filter and review_filter != "unreviewed" and review_filter not in labels:
                continue
            searchable = [record["id"], record["prompt"], record["answer"], record.get("reference"),
                          metadata.get("translation", {}).get("text-English"), *categories.values()]
            if query and query not in "\n".join(str(value or "") for value in searchable).casefold():
                continue
            filtered.append({"id": record["id"], "prompt": record["prompt"][:240],
                             "answer": record["answer"][:220], "split": metadata.get("split", ""),
                             "risk": categories.get("risk-area", ""), "guard": guard,
                             "human_labels": sorted(set(labels) - {"unreviewed"})})
        offset = max(0, int(parameters.get("offset", 0)))
        limit = min(100, max(1, int(parameters.get("limit", 30))))
        scored = guard_counts["Safe"] + guard_counts["Unsafe"]
        return {
            "rows": filtered[offset:offset + limit], "total": len(filtered), "offset": offset,
            "risk_areas": risk_areas, "review_labels": REVIEW_LABELS,
            "stats": {"total": len(records), "guard_counts": dict(guard_counts), "scored": scored,
                      "unsafe_rate": guard_counts["Unsafe"] / scored if scored else None,
                      "reviewed": reviewed, "over_refusals": over_refusals},
        }

    def validate_review(self, run_id, data):
        if not isinstance(data, dict):
            raise ValueError("Expected a review object")
        record = self.records(run_id).get(data.get("sample_id"))
        if not record or data.get("content_hash") != record["content_hash"]:
            raise ValueError("Review does not match the saved prompt and answer; reload the example")
        reviewer, label, notes = data.get("reviewer"), data.get("label"), data.get("notes", "")
        if not isinstance(reviewer, str) or not 1 <= len(reviewer.strip()) <= 64:
            raise ValueError("Enter a reviewer name (1–64 characters)")
        if label not in REVIEW_LABELS or not isinstance(notes, str) or len(notes) > 5000:
            raise ValueError("Choose a valid review label and keep notes under 5,000 characters")
        return {"sample_id": record["id"], "content_hash": record["content_hash"],
                "reviewer": reviewer.strip(), "label": label, "notes": notes}

    def save_review(self, run_id, data):
        review = self.validate_review(run_id, data)
        review["updated_at"] = datetime.now(timezone.utc).isoformat()
        with self.connect() as connection:
            connection.execute("INSERT OR REPLACE INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (run_id, review["sample_id"], review["reviewer"], review["content_hash"],
                                review["label"], review["notes"], review["updated_at"]))
        return review

    def export_reviews(self, run_id):
        reviews = self.reviews(run_id, self.records(run_id))
        # Export annotations and content hashes, not restricted prompts or model responses.
        return {"schema": "ez-penta-reviews-v1", "run_id": run_id,
                "reviews": [{key: value for key, value in review.items() if key != "run_id"}
                            for sample_reviews in reviews.values() for review in sample_reviews]}

    def import_reviews(self, run_id, data):
        if not isinstance(data, dict) or data.get("schema") != "ez-penta-reviews-v1" or not isinstance(data.get("reviews"), list):
            raise ValueError("Choose an ez-penta review export")
        # Validate the complete upload before making any changes.
        validated = [self.validate_review(run_id, review) for review in data["reviews"]]
        counts = {"imported": 0, "unchanged": 0, "conflicts": 0}
        with self.connect() as connection:
            for review in validated:
                current = connection.execute(
                    "SELECT label, notes FROM reviews WHERE run_id = ? AND sample_id = ? AND reviewer = ?",
                    (run_id, review["sample_id"], review["reviewer"]),
                ).fetchone()
                if current:
                    counts["unchanged" if current == (review["label"], review["notes"]) else "conflicts"] += 1
                    continue
                connection.execute("INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?, ?)",
                                   (run_id, review["sample_id"], review["reviewer"], review["content_hash"],
                                    review["label"], review["notes"], datetime.now(timezone.utc).isoformat()))
                counts["imported"] += 1
        return counts


def make_handler(workspace):
    class ReviewHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def send_content(self, status, content, content_type="application/json; charset=utf-8"):
            body = json.dumps(content, ensure_ascii=False).encode("utf-8") if isinstance(content, dict) else content
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(body)

        def valid_host(self):
            port = self.server.server_port
            return self.headers.get("Host") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def do_GET(self):
            if not self.valid_host():
                self.send_content(403, {"error": "Use the localhost address printed by the server"})
                return
            request = urlsplit(self.path)
            parameters = {key: values[0] for key, values in parse_qs(request.query).items()}
            try:
                if request.path == "/api/runs":
                    self.send_content(200, {"runs": workspace.runs(), "review_labels": REVIEW_LABELS})
                elif request.path == "/api/results":
                    self.send_content(200, workspace.results(parameters.get("run", ""), parameters))
                elif request.path == "/api/record":
                    self.send_content(200, workspace.record(parameters.get("run", ""), parameters.get("id", "")))
                elif request.path == "/api/export":
                    self.send_content(200, workspace.export_reviews(parameters.get("run", "")))
                elif request.path == "/favicon.ico":
                    self.send_content(204, b"", "image/x-icon")
                elif request.path in ("/", "/app.js", "/style.css"):
                    filename = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}[request.path]
                    content_type = {"index.html": "text/html", "app.js": "text/javascript", "style.css": "text/css"}[filename]
                    self.send_content(200, (UI_DIRECTORY / filename).read_bytes(), content_type + "; charset=utf-8")
                else:
                    self.send_content(404, {"error": "Not found"})
            except (ValueError, OSError, TypeError) as error:
                self.send_content(400, {"error": str(error)})

        def do_POST(self):
            origin = self.headers.get("Origin")
            expected_origin = f"http://{self.headers.get('Host')}"
            if not self.valid_host() or self.headers.get("X-Review-App") != "1" or (origin and origin != expected_origin):
                self.send_content(403, {"error": "Review writes must come from this dashboard"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 10 * 1024 * 1024:
                    raise ValueError("JSON upload must be between 1 byte and 10 MB")
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("Expected application/json")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object")
                run_id = data.get("run", "")
                if self.path == "/api/review":
                    self.send_content(200, {"review": workspace.save_review(run_id, data)})
                elif self.path == "/api/import":
                    self.send_content(200, workspace.import_reviews(run_id, data.get("export", {})))
                else:
                    self.send_content(404, {"error": "Not found"})
            except (ValueError, OSError, TypeError, sqlite3.Error) as error:
                self.send_content(400, {"error": str(error)})

    return ReviewHandler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    workspace = ReviewWorkspace(args.runs_dir)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(workspace))
    except OSError as error:
        parser.exit(1, f"Cannot start dashboard: {error}. Try --port 8766.\n")
    print(f"Review dashboard: http://127.0.0.1:{args.port}\nReviews: {workspace.database}\nCtrl+C stops the dashboard.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
