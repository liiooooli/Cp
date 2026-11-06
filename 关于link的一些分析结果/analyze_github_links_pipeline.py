#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import html
import json
import math
import re
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import tldextract


# =========================
# 路径配置
# =========================
INPUT_JSONL = Path("/home/lrq/dockerhub-data/platforms/analysis/matched_metadata.jsonl").expanduser()
# INPUT_JSONL = Path("./github_link_1266.jsonl").expanduser()
OUT_DIR = Path("./link_analysis_1266").expanduser()

EXPANDED_CSV = OUT_DIR / "1repo_link_expanded.csv"
BASELINE_TXT = OUT_DIR / "1baseline_stats.txt"

DOMAIN_SUMMARY_CSV = OUT_DIR / "2domain_summary.csv"
REGISTERED_DOMAIN_SUMMARY_CSV = OUT_DIR / "2registered_domain_summary.csv"
TOP_REGISTERED_DOMAINS_CSV = OUT_DIR / "2top_registered_domains.csv"
REPO_SUMMARY_CSV = OUT_DIR / "2repo_summary.csv"
HIGH_PRIORITY_CSV = OUT_DIR / "2high_priority_repos.csv"

DOMAIN_COOCCURRENCE_CSV = OUT_DIR / "3domain_cooccurrence.csv"
REPO_SIMILARITY_CSV = OUT_DIR / "3repo_similarity.csv"

SOURCE_FIELDS = ["link_desc", "link_readme", "link_html"]


# =========================
# 通用配置
# =========================
DOMAIN_RE = re.compile(
    r"^(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?::\d+)?(?:/.*)?$"
)

MAINSTREAM_REGISTERED_DOMAINS = {
    "github.com",
    "githubusercontent.com",
    "google.com",
    "youtube.com",
    "youtu.be",
    "twitter.com",
    "x.com",
    "telegram.org",
    "t.me",
    "discord.com",
    "discord.gg",
    "facebook.com",
    "vk.com",
    "medium.com",
    "substack.com",
    "notion.so",
    "notion.site",
    "speakerdeck.com",
    "huggingface.co",
    "kaggle.com",
    "dropbox.com",
    "mega.nz",
    "mediafire.com",
    "archive.org",
    "paypal.com",
    "patreon.com",
    "buymeacoffee.com",
    "ko-fi.com",
    "opencollective.com",
    "docker.com",
    "docker.io",
    "pypi.org",
    "npmjs.com",
    "anaconda.org",
    "rubygems.org",
    "gitlab.com",
    "bitbucket.org",
    "slack.com",
    "matrix.to",
}

MIN_FREQ_FOR_PATTERN = 2
MAX_DOMAIN_FREQ_FOR_COOCCUR = 300
MAX_REGDOMAIN_FREQ_FOR_SIM = 200
MIN_SHARED_REPOS_FOR_COOCCUR = 2
MIN_SHARED_REGDOMAINS_FOR_SIM = 2
MIN_JACCARD_FOR_SIM = 0.35

DOMAIN_STOPLIST_FOR_COOCCUR = {
    "github.com",
    "raw.githubusercontent.com",
    "avatars.githubusercontent.com",
}
REGDOMAIN_STOPLIST_FOR_SIM = {
    "github.com",
    "githubusercontent.com",
}


# =========================
# 基础函数
# =========================
def ensure_out_dir():
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def strip_nuls(s):
    if s is None:
        return s
    if isinstance(s, str):
        return s.replace("\x00", "")
    return s


def clean_row_strings(row: dict):
    cleaned = {}
    for k, v in row.items():
        if isinstance(v, str):
            cleaned[k] = v.replace("\x00", "")
        else:
            cleaned[k] = v
    return cleaned


def clean_csv_text_iter(path: Path, encoding="utf-8-sig"):
    """
    按行读取 CSV，并去掉 NUL 字符，避免 csv.DictReader 报:
    _csv.Error: line contains NUL
    """
    with path.open("r", encoding=encoding, newline="") as f:
        for line in f:
            if "\x00" in line:
                line = line.replace("\x00", "")
            yield line


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            raw = line.strip()
            if not raw:
                continue
            if "\x00" in raw:
                raw = raw.replace("\x00", "")
            # 容忍行首混入脏前缀
            if not raw.startswith("{"):
                pos = raw.find("{")
                if pos != -1:
                    raw = raw[pos:]
            try:
                yield lineno, json.loads(raw)
            except Exception as e:
                print(f"[WARN] {path} line {lineno} JSON解析失败: {e}")


def percentile(values, p):
    if not values:
        return 0
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return vals[int(k)]
    d0 = vals[f] * (c - k)
    d1 = vals[c] * (k - f)
    return d0 + d1


def counter_to_json(counter_obj):
    items = sorted(counter_obj.items(), key=lambda x: (-x[1], x[0]))
    return json.dumps(dict(items), ensure_ascii=False)


def list_to_json(x):
    return json.dumps(x, ensure_ascii=False)


def looks_like_domain_without_scheme(s: str) -> bool:
    return bool(DOMAIN_RE.match(s))


def normalize_list_field(v):
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return [v]


def safe_urlsplit(url: str):
    try:
        return urlsplit(url)
    except Exception:
        return None


def strip_outer_quotes_and_space(s: str) -> str:
    s = html.unescape(str(s)).strip()
    s = s.strip('"').strip("'").strip()
    return s


def normalize_url(raw, source_field=None):
    """
    只保留 http/https 链接。
    homepage 若无 scheme，且长得像域名，则自动补 https://
    """
    if raw is None:
        return None

    s = strip_nuls(strip_outer_quotes_and_space(raw))
    if not s:
        return None

    if s.startswith("//"):
        s = "https:" + s

    scheme_like = re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", s)

    if source_field == "homepage" and not scheme_like:
        if looks_like_domain_without_scheme(s):
            s = "https://" + s
        else:
            return None
    elif not scheme_like:
        return None

    parsed = safe_urlsplit(s)
    if parsed is None:
        return None

    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        return None

    host = parsed.hostname
    if not host:
        return None
    host = host.lower()

    try:
        port = parsed.port
    except Exception:
        return None

    netloc = host
    if port and not ((scheme == "http" and port == 80) or (scheme == "https" and port == 443)):
        netloc = f"{host}:{port}"

    path = parsed.path or ""
    query = parsed.query or ""

    if path == "/" and not query:
        path = ""

    normalized = urlunsplit((scheme, netloc, path, query, ""))
    return strip_nuls(normalized)


def extract_domain_info(url: str):
    parsed = safe_urlsplit(url)
    if parsed is None or not parsed.hostname:
        return None, None

    domain = parsed.hostname.lower()

    ext = tldextract.extract(domain)
    if ext.domain and ext.suffix:
        registered_domain = f"{ext.domain}.{ext.suffix}"
    else:
        registered_domain = domain

    return strip_nuls(domain), strip_nuls(registered_domain)


def classify_url(domain: str, registered_domain: str, url: str):
    """
    返回:
      platform_type, function_type
    """
    parsed = safe_urlsplit(url)
    path = (parsed.path or "").lower() if parsed else ""

    if registered_domain in {"twitter.com", "x.com"}:
        return "twitter_x", "social_media"
    if domain in {"t.me", "telegram.me"} or registered_domain == "telegram.org":
        return "telegram", "social_media"
    if registered_domain == "facebook.com":
        return "facebook", "social_media"
    if registered_domain == "vk.com":
        return "vk", "social_media"
    if registered_domain in {"youtube.com", "youtu.be"}:
        return "youtube", "video_media"
    if registered_domain == "bilibili.com":
        return "bilibili", "video_media"

    if registered_domain in {"discord.com", "discord.gg"}:
        return "discord", "community"
    if registered_domain == "slack.com":
        return "slack", "community"
    if registered_domain == "matrix.to":
        return "matrix", "community"

    if domain == "docs.google.com":
        if path.startswith("/presentation"):
            return "google_slides", "docs_presentation"
        if path.startswith("/document"):
            return "google_docs", "docs_presentation"
        if path.startswith("/spreadsheets"):
            return "google_sheets", "docs_data"
        return "google_docs", "docs_presentation"

    if registered_domain in {"speakerdeck.com", "notion.so", "notion.site", "medium.com", "substack.com"}:
        return registered_domain.replace(".", "_"), "docs_presentation"

    if registered_domain == "github.com":
        return "github", "code_collaboration"
    if registered_domain == "gitlab.com":
        return "gitlab", "code_collaboration"
    if registered_domain == "bitbucket.org":
        return "bitbucket", "code_collaboration"

    if registered_domain == "huggingface.co":
        return "huggingface", "model_data_hosting"
    if registered_domain == "kaggle.com":
        return "kaggle", "model_data_hosting"
    if domain == "colab.research.google.com":
        return "google_colab", "model_data_hosting"

    if domain == "drive.google.com":
        return "google_drive", "cloud_storage"
    if registered_domain == "dropbox.com":
        return "dropbox", "cloud_storage"
    if registered_domain == "mega.nz":
        return "mega", "cloud_storage"
    if registered_domain == "mediafire.com":
        return "mediafire", "cloud_storage"
    if registered_domain == "archive.org":
        return "archive", "cloud_storage"

    if registered_domain == "paypal.com":
        return "paypal", "payment_fundraising"
    if registered_domain == "patreon.com":
        return "patreon", "payment_fundraising"
    if registered_domain == "buymeacoffee.com":
        return "buymeacoffee", "payment_fundraising"
    if registered_domain == "ko-fi.com":
        return "ko_fi", "payment_fundraising"
    if registered_domain == "opencollective.com":
        return "opencollective", "payment_fundraising"

    if domain == "hub.docker.com" or registered_domain in {"docker.com", "docker.io"}:
        return "docker", "download_deployment"
    if registered_domain == "pypi.org":
        return "pypi", "package_registry"
    if registered_domain == "npmjs.com":
        return "npm", "package_registry"
    if registered_domain == "anaconda.org":
        return "anaconda", "package_registry"
    if registered_domain == "rubygems.org":
        return "rubygems", "package_registry"

    return "custom_site", "website_landing"


# =========================
# Step 1: 归一化展开
# =========================
def step1_expand():
    ensure_out_dir()

    fieldnames = [
        "repo_name",
        "source_field",
        "url",
        "domain",
        "registered_domain",
        "platform_type",
        "function_type",
        "is_mainstream_registered_domain",
    ]

    total_repos = 0
    total_rows = 0

    unique_urls = set()
    unique_domains = set()
    unique_registered_domains = set()

    occurrence_by_registered_domain = Counter()
    links_per_repo = Counter()
    unique_domains_per_repo = defaultdict(set)
    unique_registered_domains_per_repo = defaultdict(set)

    with EXPANDED_CSV.open("w", encoding="utf-8-sig", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for lineno, obj in iter_jsonl(INPUT_JSONL):
            name = strip_nuls(obj.get("name"))
            if not name:
                print(f"[WARN] {INPUT_JSONL} line {lineno} 缺少 name，跳过")
                continue

            total_repos += 1
            local_seen = set()

            for source_field in SOURCE_FIELDS:
                for raw_url in normalize_list_field(obj.get(source_field, [])):
                    norm_url = normalize_url(raw_url, source_field=source_field)
                    if not norm_url:
                        continue

                    key = (source_field, norm_url)
                    if key in local_seen:
                        continue
                    local_seen.add(key)

                    domain, reg_domain = extract_domain_info(norm_url)
                    if not domain or not reg_domain:
                        continue

                    platform_type, function_type = classify_url(domain, reg_domain, norm_url)
                    mainstream_flag = "1" if reg_domain in MAINSTREAM_REGISTERED_DOMAINS else "0"

                    row = {
                        "repo_name": name,
                        "source_field": source_field,
                        "url": norm_url,
                        "domain": domain,
                        "registered_domain": reg_domain,
                        "platform_type": platform_type,
                        "function_type": function_type,
                        "is_mainstream_registered_domain": mainstream_flag,
                    }
                    writer.writerow(clean_row_strings(row))

                    total_rows += 1
                    unique_urls.add(norm_url)
                    unique_domains.add(domain)
                    unique_registered_domains.add(reg_domain)
                    occurrence_by_registered_domain[reg_domain] += 1
                    links_per_repo[name] += 1
                    unique_domains_per_repo[name].add(domain)
                    unique_registered_domains_per_repo[name].add(reg_domain)

            home = normalize_url(obj.get("homepage"), source_field="homepage")
            if home:
                key = ("homepage", home)
                if key not in local_seen:
                    domain, reg_domain = extract_domain_info(home)
                    if domain and reg_domain:
                        platform_type, function_type = classify_url(domain, reg_domain, home)
                        mainstream_flag = "1" if reg_domain in MAINSTREAM_REGISTERED_DOMAINS else "0"

                        row = {
                            "repo_name": name,
                            "source_field": "homepage",
                            "url": home,
                            "domain": domain,
                            "registered_domain": reg_domain,
                            "platform_type": platform_type,
                            "function_type": function_type,
                            "is_mainstream_registered_domain": mainstream_flag,
                        }
                        writer.writerow(clean_row_strings(row))

                        total_rows += 1
                        unique_urls.add(home)
                        unique_domains.add(domain)
                        unique_registered_domains.add(reg_domain)
                        occurrence_by_registered_domain[reg_domain] += 1
                        links_per_repo[name] += 1
                        unique_domains_per_repo[name].add(domain)
                        unique_registered_domains_per_repo[name].add(reg_domain)

    links_values = list(links_per_repo.values())
    unique_domain_values = [len(v) for v in unique_domains_per_repo.values()]
    unique_reg_values = [len(v) for v in unique_registered_domains_per_repo.values()]

    with BASELINE_TXT.open("w", encoding="utf-8") as f:
        f.write(f"INPUT_JSONL: {INPUT_JSONL}\n")
        f.write(f"EXPANDED_CSV: {EXPANDED_CSV}\n\n")

        f.write("=== Baseline Stats ===\n")
        f.write(f"total_repos = {total_repos}\n")
        f.write(f"total_link_rows = {total_rows}\n")
        f.write(f"unique_urls = {len(unique_urls)}\n")
        f.write(f"unique_domains = {len(unique_domains)}\n")
        f.write(f"unique_registered_domains = {len(unique_registered_domains)}\n\n")

        f.write("=== Links per Repo ===\n")
        f.write(f"p50 = {percentile(links_values, 50):.2f}\n")
        f.write(f"p90 = {percentile(links_values, 90):.2f}\n")
        f.write(f"p95 = {percentile(links_values, 95):.2f}\n")
        f.write(f"p99 = {percentile(links_values, 99):.2f}\n\n")

        f.write("=== Unique Domains per Repo ===\n")
        f.write(f"p50 = {percentile(unique_domain_values, 50):.2f}\n")
        f.write(f"p90 = {percentile(unique_domain_values, 90):.2f}\n")
        f.write(f"p95 = {percentile(unique_domain_values, 95):.2f}\n")
        f.write(f"p99 = {percentile(unique_domain_values, 99):.2f}\n\n")

        f.write("=== Unique Registered Domains per Repo ===\n")
        f.write(f"p50 = {percentile(unique_reg_values, 50):.2f}\n")
        f.write(f"p90 = {percentile(unique_reg_values, 90):.2f}\n")
        f.write(f"p95 = {percentile(unique_reg_values, 95):.2f}\n")
        f.write(f"p99 = {percentile(unique_reg_values, 99):.2f}\n\n")

        f.write("=== Top 100 Registered Domains by Occurrence ===\n")
        for reg_domain, cnt in occurrence_by_registered_domain.most_common(100):
            f.write(f"{reg_domain}\t{cnt}\n")

    print(f"[OK] Step 1 完成: {EXPANDED_CSV}")
    print(f"[OK] Baseline 报告: {BASELINE_TXT}")


# =========================
# Step 2: 汇总 + 风险优先级
# =========================
def compute_repo_score(
    repo_name,
    repo_metrics,
    reg_repo_count_map,
    p95_links,
    p99_links,
):
    function_counter = repo_metrics["function_counter"]
    registered_domains = repo_metrics["registered_domains"]

    social_count = (
        function_counter["social_media"]
        + function_counter["video_media"]
        + function_counter["community"]
    )
    docs_count = function_counter["docs_presentation"] + function_counter["docs_data"]
    storage_count = (
        function_counter["cloud_storage"]
        + function_counter["model_data_hosting"]
        + function_counter["download_deployment"]
        + function_counter["package_registry"]
    )
    payment_count = function_counter["payment_fundraising"]
    website_count = function_counter["website_landing"]

    custom_registered_domains = [
        d for d in registered_domains if d not in MAINSTREAM_REGISTERED_DOMAINS
    ]

    rare_registered_domains = [
        d
        for d in registered_domains
        if d not in MAINSTREAM_REGISTERED_DOMAINS
        and 2 <= reg_repo_count_map.get(d, 0) <= 5
    ]

    unique_urls = len(repo_metrics["urls"])
    function_diversity = sum(1 for _, c in function_counter.items() if c > 0)

    score = 0.0
    reasons = []

    if social_count >= 1:
        score += 2.0
        reasons.append("有社交/传播入口(+2)")
    if social_count >= 2:
        score += 1.0
        reasons.append("社交/传播平台不止一个(+1)")
    if docs_count >= 1:
        score += 1.0
        reasons.append("有文档/叙事载体(+1)")
    if storage_count >= 1:
        score += 2.0
        reasons.append("有托管/下载/部署链路(+2)")
    if payment_count >= 1:
        score += 3.0
        reasons.append("有支付/募资入口(+3)")
    if len(custom_registered_domains) >= 1:
        add = min(3.0, 0.6 * len(custom_registered_domains))
        score += add
        reasons.append(f"存在自定义/小众域名(+{add:.1f})")
    if len(rare_registered_domains) >= 1:
        add = min(4.0, 1.0 * len(rare_registered_domains))
        score += add
        reasons.append(f"存在稀有但复用域名(+{add:.1f})")
    if unique_urls >= p95_links:
        score += 2.0
        reasons.append("外链数量高于P95(+2)")
    if unique_urls >= p99_links:
        score += 1.0
        reasons.append("外链数量高于P99(+1)")
    if function_diversity >= 4:
        score += 2.0
        reasons.append("功能类型较复杂(+2)")
    if social_count >= 1 and (docs_count >= 1 or storage_count >= 1 or payment_count >= 1 or website_count >= 1):
        score += 2.0
        reasons.append("存在“传播+载体/托管/支付/官网”链条(+2)")

    active_functions = {k for k, v in function_counter.items() if v > 0}
    if active_functions.issubset({"code_collaboration", "website_landing"}) and len(registered_domains) <= 2:
        score -= 2.0
        reasons.append("更像常规项目主页/代码协作(-2)")
    if active_functions.issubset({"code_collaboration", "docs_presentation"}) and len(registered_domains) <= 2:
        score -= 1.0
        reasons.append("更像常规文档型项目(-1)")

    if score >= 8:
        band = "high"
    elif score >= 5:
        band = "medium"
    else:
        band = "low"

    return score, band, custom_registered_domains, rare_registered_domains, reasons


def step2_summarize():
    ensure_out_dir()

    if not EXPANDED_CSV.exists():
        raise FileNotFoundError(f"请先跑 Step 1，缺少文件: {EXPANDED_CSV}")

    domain_data = defaultdict(lambda: {
        "registered_domain": "",
        "repos": set(),
        "occurrences": 0,
        "platform_counter": Counter(),
        "function_counter": Counter(),
    })

    regdomain_data = defaultdict(lambda: {
        "repos": set(),
        "occurrences": 0,
        "domain_set": set(),
        "platform_counter": Counter(),
        "function_counter": Counter(),
    })

    repo_data = defaultdict(lambda: {
        "urls": set(),
        "domains": set(),
        "registered_domains": set(),
        "platform_counter": Counter(),
        "function_counter": Counter(),
    })

    reader = csv.DictReader(clean_csv_text_iter(EXPANDED_CSV))
    for row in reader:
        repo = strip_nuls(row.get("repo_name", ""))
        url = strip_nuls(row.get("url", ""))
        domain = strip_nuls(row.get("domain", ""))
        reg_domain = strip_nuls(row.get("registered_domain", ""))
        platform_type = strip_nuls(row.get("platform_type", ""))
        function_type = strip_nuls(row.get("function_type", ""))

        if not repo or not url or not domain or not reg_domain:
            continue

        domain_data[domain]["registered_domain"] = reg_domain
        domain_data[domain]["repos"].add(repo)
        domain_data[domain]["occurrences"] += 1
        domain_data[domain]["platform_counter"][platform_type] += 1
        domain_data[domain]["function_counter"][function_type] += 1

        regdomain_data[reg_domain]["repos"].add(repo)
        regdomain_data[reg_domain]["occurrences"] += 1
        regdomain_data[reg_domain]["domain_set"].add(domain)
        regdomain_data[reg_domain]["platform_counter"][platform_type] += 1
        regdomain_data[reg_domain]["function_counter"][function_type] += 1

        repo_data[repo]["urls"].add(url)
        repo_data[repo]["domains"].add(domain)
        repo_data[repo]["registered_domains"].add(reg_domain)
        repo_data[repo]["platform_counter"][platform_type] += 1
        repo_data[repo]["function_counter"][function_type] += 1

    with DOMAIN_SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as fout:
        fieldnames = [
            "domain",
            "registered_domain",
            "repo_count",
            "total_occurrences",
            "top_platform_type",
            "top_function_type",
            "example_repos",
        ]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        rows = []
        for domain, info in domain_data.items():
            top_platform = info["platform_counter"].most_common(1)[0][0] if info["platform_counter"] else ""
            top_function = info["function_counter"].most_common(1)[0][0] if info["function_counter"] else ""
            rows.append({
                "domain": domain,
                "registered_domain": info["registered_domain"],
                "repo_count": len(info["repos"]),
                "total_occurrences": info["occurrences"],
                "top_platform_type": top_platform,
                "top_function_type": top_function,
                "example_repos": "|".join(sorted(info["repos"])[:5]),
            })

        rows.sort(key=lambda x: (-int(x["repo_count"]), -int(x["total_occurrences"]), x["domain"]))
        writer.writerows([clean_row_strings(r) for r in rows])

    with REGISTERED_DOMAIN_SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as fout:
        fieldnames = [
            "registered_domain",
            "repo_count",
            "total_occurrences",
            "subdomain_count",
            "top_platform_type",
            "top_function_type",
            "example_repos",
        ]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        rows = []
        for reg_domain, info in regdomain_data.items():
            top_platform = info["platform_counter"].most_common(1)[0][0] if info["platform_counter"] else ""
            top_function = info["function_counter"].most_common(1)[0][0] if info["function_counter"] else ""
            rows.append({
                "registered_domain": reg_domain,
                "repo_count": len(info["repos"]),
                "total_occurrences": info["occurrences"],
                "subdomain_count": len(info["domain_set"]),
                "top_platform_type": top_platform,
                "top_function_type": top_function,
                "example_repos": "|".join(sorted(info["repos"])[:5]),
            })

        rows.sort(key=lambda x: (-int(x["repo_count"]), -int(x["total_occurrences"]), x["registered_domain"]))
        writer.writerows([clean_row_strings(r) for r in rows])

    with TOP_REGISTERED_DOMAINS_CSV.open("w", encoding="utf-8-sig", newline="") as fout:
        fieldnames = [
            "registered_domain",
            "repo_count",
            "total_occurrences",
            "top_function_type",
        ]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        rows = []
        for reg_domain, info in regdomain_data.items():
            top_function = info["function_counter"].most_common(1)[0][0] if info["function_counter"] else ""
            rows.append({
                "registered_domain": reg_domain,
                "repo_count": len(info["repos"]),
                "total_occurrences": info["occurrences"],
                "top_function_type": top_function,
            })

        rows.sort(key=lambda x: (-int(x["repo_count"]), -int(x["total_occurrences"]), x["registered_domain"]))
        writer.writerows([clean_row_strings(r) for r in rows[:200]])

    reg_repo_count_map = {k: len(v["repos"]) for k, v in regdomain_data.items()}
    repo_link_counts = [len(v["urls"]) for v in repo_data.values()]
    p95_links = percentile(repo_link_counts, 95)
    p99_links = percentile(repo_link_counts, 99)

    repo_rows = []

    for repo_name, info in repo_data.items():
        score, band, custom_regs, rare_regs, reasons = compute_repo_score(
            repo_name=repo_name,
            repo_metrics=info,
            reg_repo_count_map=reg_repo_count_map,
            p95_links=p95_links,
            p99_links=p99_links,
        )

        row = {
            "repo_name": repo_name,
            "total_links": len(info["urls"]),
            "unique_domains": len(info["domains"]),
            "unique_registered_domains": len(info["registered_domains"]),
            "platform_type_counts": counter_to_json(info["platform_counter"]),
            "function_type_counts": counter_to_json(info["function_counter"]),
            "custom_registered_domain_count": len(custom_regs),
            "rare_registered_domain_count": len(rare_regs),
            "function_diversity": sum(1 for _, c in info["function_counter"].items() if c > 0),
            "registered_domains": list_to_json(sorted(info["registered_domains"])),
            "custom_registered_domains": list_to_json(sorted(custom_regs)),
            "rare_registered_domains": list_to_json(sorted(rare_regs)),
            "suspicious_score": f"{score:.2f}",
            "priority_band": band,
            "score_reasons": " | ".join(reasons),
        }
        repo_rows.append(row)

    repo_rows.sort(
        key=lambda x: (-float(x["suspicious_score"]), -int(x["unique_registered_domains"]), x["repo_name"])
    )

    with REPO_SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as fout:
        fieldnames = [
            "repo_name",
            "total_links",
            "unique_domains",
            "unique_registered_domains",
            "platform_type_counts",
            "function_type_counts",
            "custom_registered_domain_count",
            "rare_registered_domain_count",
            "function_diversity",
            "registered_domains",
            "custom_registered_domains",
            "rare_registered_domains",
            "suspicious_score",
            "priority_band",
            "score_reasons",
        ]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([clean_row_strings(r) for r in repo_rows])

    with HIGH_PRIORITY_CSV.open("w", encoding="utf-8-sig", newline="") as fout:
        fieldnames = [
            "repo_name",
            "suspicious_score",
            "priority_band",
            "total_links",
            "unique_registered_domains",
            "custom_registered_domain_count",
            "rare_registered_domain_count",
            "registered_domains",
            "score_reasons",
        ]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        out_rows = []
        for row in repo_rows:
            if row["priority_band"] in {"high", "medium"}:
                out_rows.append({
                    "repo_name": row["repo_name"],
                    "suspicious_score": row["suspicious_score"],
                    "priority_band": row["priority_band"],
                    "total_links": row["total_links"],
                    "unique_registered_domains": row["unique_registered_domains"],
                    "custom_registered_domain_count": row["custom_registered_domain_count"],
                    "rare_registered_domain_count": row["rare_registered_domain_count"],
                    "registered_domains": row["registered_domains"],
                    "score_reasons": row["score_reasons"],
                })
        writer.writerows([clean_row_strings(r) for r in out_rows])

    print(f"[OK] Step 2 完成:")
    print(f"     {DOMAIN_SUMMARY_CSV}")
    print(f"     {REGISTERED_DOMAIN_SUMMARY_CSV}")
    print(f"     {TOP_REGISTERED_DOMAINS_CSV}")
    print(f"     {REPO_SUMMARY_CSV}")
    print(f"     {HIGH_PRIORITY_CSV}")


# =========================
# Step 3: 域名共现 + repo 相似度
# =========================
def step3_patterns():
    ensure_out_dir()

    if not EXPANDED_CSV.exists():
        raise FileNotFoundError(f"请先跑 Step 1，缺少文件: {EXPANDED_CSV}")

    repo_to_domains = defaultdict(set)
    repo_to_regdomains = defaultdict(set)

    domain_repo_sets = defaultdict(set)
    regdomain_repo_sets = defaultdict(set)

    reader = csv.DictReader(clean_csv_text_iter(EXPANDED_CSV))
    for row in reader:
        repo = strip_nuls(row.get("repo_name", ""))
        domain = strip_nuls(row.get("domain", ""))
        reg_domain = strip_nuls(row.get("registered_domain", ""))

        if not repo or not domain or not reg_domain:
            continue

        repo_to_domains[repo].add(domain)
        repo_to_regdomains[repo].add(reg_domain)

        domain_repo_sets[domain].add(repo)
        regdomain_repo_sets[reg_domain].add(repo)

    domain_repo_count = {d: len(rs) for d, rs in domain_repo_sets.items()}
    regdomain_repo_count = {d: len(rs) for d, rs in regdomain_repo_sets.items()}

    informative_repo_domains = {}
    for repo, domains in repo_to_domains.items():
        informative = {
            d for d in domains
            if domain_repo_count.get(d, 0) >= MIN_FREQ_FOR_PATTERN
            and domain_repo_count.get(d, 0) <= MAX_DOMAIN_FREQ_FOR_COOCCUR
            and d not in DOMAIN_STOPLIST_FOR_COOCCUR
        }
        if len(informative) >= 2:
            informative_repo_domains[repo] = informative

    pair_counter = Counter()
    for repo, domains in informative_repo_domains.items():
        for a, b in combinations(sorted(domains), 2):
            pair_counter[(a, b)] += 1

    with DOMAIN_COOCCURRENCE_CSV.open("w", encoding="utf-8-sig", newline="") as fout:
        fieldnames = ["domain_a", "domain_b", "shared_repo_count"]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        rows = []
        for (a, b), cnt in pair_counter.items():
            if cnt >= MIN_SHARED_REPOS_FOR_COOCCUR:
                rows.append({
                    "domain_a": a,
                    "domain_b": b,
                    "shared_repo_count": cnt,
                })

        rows.sort(key=lambda x: (-int(x["shared_repo_count"]), x["domain_a"], x["domain_b"]))
        writer.writerows([clean_row_strings(r) for r in rows])

    informative_repo_regdomains = {}
    inverted_index = defaultdict(list)

    for repo, regdomains in repo_to_regdomains.items():
        informative = {
            d for d in regdomains
            if regdomain_repo_count.get(d, 0) >= MIN_FREQ_FOR_PATTERN
            and regdomain_repo_count.get(d, 0) <= MAX_REGDOMAIN_FREQ_FOR_SIM
            and d not in REGDOMAIN_STOPLIST_FOR_SIM
        }
        if informative:
            informative_repo_regdomains[repo] = informative
            for d in informative:
                inverted_index[d].append(repo)

    candidate_pair_counter = Counter()
    for reg_domain, repos in inverted_index.items():
        repos = sorted(set(repos))
        if len(repos) > MAX_REGDOMAIN_FREQ_FOR_SIM:
            continue
        for a, b in combinations(repos, 2):
            candidate_pair_counter[(a, b)] += 1

    with REPO_SIMILARITY_CSV.open("w", encoding="utf-8-sig", newline="") as fout:
        fieldnames = [
            "repo_a",
            "repo_b",
            "shared_registered_domains",
            "shared_count",
            "jaccard_score",
        ]
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        rows = []
        for (repo_a, repo_b), shared_count in candidate_pair_counter.items():
            if shared_count < MIN_SHARED_REGDOMAINS_FOR_SIM:
                continue

            set_a = informative_repo_regdomains.get(repo_a, set())
            set_b = informative_repo_regdomains.get(repo_b, set())
            if not set_a or not set_b:
                continue

            inter = sorted(set_a & set_b)
            union_size = len(set_a | set_b)
            if union_size == 0:
                continue

            jaccard = len(inter) / union_size
            if jaccard < MIN_JACCARD_FOR_SIM:
                continue

            rows.append({
                "repo_a": repo_a,
                "repo_b": repo_b,
                "shared_registered_domains": list_to_json(inter),
                "shared_count": len(inter),
                "jaccard_score": f"{jaccard:.4f}",
            })

        rows.sort(key=lambda x: (-int(x["shared_count"]), -float(x["jaccard_score"]), x["repo_a"], x["repo_b"]))
        writer.writerows([clean_row_strings(r) for r in rows])

    print(f"[OK] Step 3 完成:")
    print(f"     {DOMAIN_COOCCURRENCE_CSV}")
    print(f"     {REPO_SIMILARITY_CSV}")


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(description="Analyze GitHub external links pipeline")
    parser.add_argument(
        "--step",
        choices=["1", "2", "3", "all"],
        required=True,
        help="Run step 1 / 2 / 3 / all",
    )
    args = parser.parse_args()

    if args.step == "1":
        step1_expand()
    elif args.step == "2":
        step2_summarize()
    elif args.step == "3":
        step3_patterns()
    elif args.step == "all":
        step1_expand()
        step2_summarize()
        step3_patterns()


if __name__ == "__main__":
    main()
