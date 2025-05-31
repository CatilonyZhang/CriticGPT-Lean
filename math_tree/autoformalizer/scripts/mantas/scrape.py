import json
import re

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def parse_handles(file):
    with open(file, "r") as f:
        entries = f.readlines()
    # Clean up newline characters
    entries = [entry.strip().replace('"', "").replace(",", "") for entry in entries]
    entries = list(filter(lambda entry: entry.count("/") == 2, entries))
    return entries


def get_github_link(page_url):
    """Extract the GitHub repository link from the given Lean Reservoir page."""
    try:
        # Fetch the webpage
        response = requests.get(page_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
    except requests.RequestException as e:
        print(f"Failed to fetch {page_url}: {e}")
        return None

    # Find the section containing the word "Repository"
    repo_section = None
    for section in soup.find_all(
        text=lambda text: text and "repository" in text.lower()
    ):
        # Look for the next sibling or associated link in this section
        repo_section = section.find_parent()  # Parent tag of the text node
        break  # Assuming only one such section

    if repo_section:
        # Look for the GitHub logo (could be an <img> with "github" in alt or src)
        print(repo_section)
        # Look for the hyperlink (<a>) next to the GitHub logo
        github_link = repo_section.find_next("a", href=True)
        if github_link and "github.com" in github_link["href"]:
            return github_link["href"]

    # If no GitHub link is found
    return None


def extract_repository_link(page_url):
    """Extract the 'Repository' section with bold, large font from the given Lean Reservoir page."""
    try:
        # Fetch the webpage
        response = requests.get(page_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
    except requests.RequestException as e:
        print(f"Failed to fetch {page_url}: {e}")
        return None

    # Search for headers (e.g., <h2>, <h3>, etc.) with the word 'Repository'
    repo_header = None
    for header in soup.find_all(
        ["h3"], string=lambda text: text and "repository" in text.lower()
    ):
        # Check if the header has bold text and a larger font (we can check CSS or inline styles)
        repo_header = header

    if repo_header:
        # Find the parent section that contains this header (this might include the repository link)
        repo_section = repo_header.find_parent()
        github_link = repo_section.find("a", class_="hard-link")
        return github_link["href"]

    # If no repository section is found
    return "No repository section with bold, large font found."


def extract_commit_hash(page_url):
    """Extract the 'Repository' section with bold, large font from the given Lean Reservoir page."""
    try:
        # Fetch the webpage
        response = requests.get(page_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
    except requests.RequestException as e:
        print(f"Failed to fetch {page_url}: {e}")
        return None

    # Search for headers (e.g., <h2>, <h3>, etc.) with the word 'Repository'
    version_header = None
    for header in soup.find_all(
        ["h3"], string=lambda text: text and "lean" in text.lower()
    ):
        # Check if the header has bold text and a larger font (we can check CSS or inline styles)
        version_header = header

    if version_header:
        # Find the parent section that contains this header (this might include the repository link)
        version_section = version_header.find_parent()
        # Find the <li> element that contains the v4.9.0 - v4.11.0 versions
        version_pattern = r"\bv4\.(9|10|11)\.([0-9]+)(?:-rc[0-9]+)?"
        li_elements = version_section.find_all("li")
        for li in li_elements:
            if re.search(version_pattern, li.get_text()):
                # Find the span with the commit information
                commit_info = li.find(string=re.compile(r"Commit"))
                build_outcome = li.find(class_="outcome-success") is not None
                if commit_info and build_outcome:
                    # Extract commit hash using regex
                    commit_hash = re.search(
                        r"Commit ([a-z0-9]+)", commit_info.get_text()
                    )
                    print(page_url)
                    print(commit_hash)
                    print("____")
                    if commit_hash:
                        return commit_hash.group(1)


def export_data_json(page):
    gh_link = extract_repository_link(page)
    commit_id = extract_commit_hash(page)
    if gh_link and commit_id:
        return {"repository_link": gh_link, "commit_hash": commit_id}


if __name__ == "__main__":
    base_url = "https://reservoir.lean-lang.org"
    output_file = "scripts/mantas/reservoir/scraped_reservoir.jsonl"
    output = []
    pages = parse_handles("scripts/mantas/reservoir/reservoir_handles.txt")
    with tqdm(
        total=len(pages), desc="Extracting GitHub links and commit hashes"
    ) as pbar:
        for page in pages:
            link = f"{base_url}{page}"
            result = export_data_json(link)
            if result:
                output.append(result)
            pbar.update(1)
    # Save the results to a JSON file
    with open(output_file, "w") as f:
        for entry in output:
            json_line = json.dumps(entry)
            f.write(json_line + "\n")
