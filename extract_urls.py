import re

# Read the HTML file
with open('Untitled-1.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Extract all data-clip values
clips = re.findall(r'data-clip="([^"]+)"', content)

# Remove duplicates while preserving order
seen = set()
unique_clips = []
for clip in clips:
    if clip not in seen:
        seen.add(clip)
        unique_clips.append(clip)

# Generate URLs
base_url = "https://pesemne.ro/wp-content/uploads/clips/"

# Write URLs to file
with open('urls.txt', 'w', encoding='utf-8') as f:
    for clip in unique_clips:
        f.write(f"{base_url}{clip}\n")

print(f"Extracted {len(unique_clips)} unique video URLs to urls.txt")
