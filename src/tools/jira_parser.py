import re
from typing import Dict, List, Optional


def _clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    # Strip HTML tags if any simple cases
    s = re.sub(r"<[^>]+>", "\n", s)
    # Normalize whitespace
    s = s.replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def extract_acceptance_criteria(text: str) -> List[str]:
    """Extract bullet points under headings like 'Acceptance Criteria'/'AC' from Jira description.
    Supports simple wiki/markdown bullets (-, *, 1.)
    """
    if not text:
        return []
    lines = text.splitlines()
    ac: List[str] = []
    in_ac = False
    for line in lines:
        l = line.strip()
        # Detect start of AC section
        if re.search(r"^(acceptance\s*criteria|ac)[:\s]*$", l, re.IGNORECASE):
            in_ac = True
            continue
        # Another header ends the section
        if in_ac and re.match(r"^[#>=\-]{2,}|^[A-Za-z].*:$", l) and not re.match(r"^[-*] ", l):
            in_ac = False
        # Collect bullets
        if in_ac and re.match(r"^([-*]|\d+\.)\s+", l):
            ac.append(re.sub(r"^([-*]|\d+\.)\s+", "", l))
    # Fallback: gather bullets anywhere if no explicit AC header
    if not ac:
        for l in lines:
            ls = l.strip()
            if re.match(r"^([-*]|\d+\.)\s+", ls):
                ac.append(re.sub(r"^([-*]|\d+\.)\s+", "", ls))
    # Deduplicate, keep order
    seen = set()
    uniq = []
    for item in ac:
        if item not in seen:
            seen.add(item)
            uniq.append(item)
    return uniq[:15]


def extract_gherkin(text: str) -> List[str]:
    if not text:
        return []
    gherkin = []
    for l in text.splitlines():
        ls = l.strip()
        if re.match(r"^(Given|When|Then|And|But)\b", ls, re.IGNORECASE):
            gherkin.append(ls)
    return gherkin[:50]


def parse_issue(summary: Optional[str], description: Optional[str]) -> Dict:
    """Return structured context for test generation from Jira fields."""
    desc = _clean_text(description)
    ac = extract_acceptance_criteria(desc)
    gherkin = extract_gherkin(desc)
    context = {
        "user_story": summary or "",
        "acceptance_criteria": ac,
        "gherkin": gherkin,
        "raw_description": desc[:4000],
    }
    # Heuristic test goals from AC or gherkin
    goals: List[str] = []
    for item in ac:
        goals.append(item)
    if not goals and gherkin:
        goals.extend(gherkin[:5])
    if not goals and summary:
        goals.append(summary)
    context["test_goals"] = goals[:10]
    return context
