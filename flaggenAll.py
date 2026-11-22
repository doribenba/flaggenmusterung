# 3 Oct. 2026 - Dorian B.
from PIL import Image
import numpy as np
import os
import colorsys
from itertools import combinations
import math as m
import ast

FOLDER = "Flaggen100"
NUMBER_OF_RESULTS = 10
CANDIDATE_POOL_SIZE = 50
EXCEPTIONAL_ASPECT_RATIO_PENALTY = 10
USE_ABSTRACT_SCORE = False
SYMBOL_PRESENCE_PENALTY = 5
SYMBOL_COLOUR_PENALTY = 3
SYMBOL_LAYOUT_WEIGHT = 0.2
FLAG_TYPE_MISMATCH_PENALTY = 5
STRIPE_WIDTH_MISMATCH_PENALTY = 2
CLUSTER_SIMILARITY_THRESHOLD = 88.0

def parse_color_line(line):
    color_name = line.split(" (")[0]
    pixel_count = int(
        line.split(": ")[1].split(" pixels")[0]
    )
    return color_name, pixel_count

def base_color(color_name):
    parts = color_name.split()
    if parts[0] in ["light", "dark"]:
        return " ".join(parts[1:])
    return color_name

def load_flag(filename):
    path = os.path.join(FOLDER, filename)
    colors = {}
    with open(path, "r") as file:
        for line in file:
            if line.strip() == "Abstract:":
                break
            if not line.strip():
                continue
            color_name, pixel_count = parse_color_line(line)
            color = base_color(color_name)
            if color in colors:
                colors[color] += pixel_count
            else:
                colors[color] = pixel_count
    total_pixels = sum(colors.values())
    for color in colors:
        colors[color] = colors[color] / total_pixels * 100
    return colors

def aspect_ratio_penalty(filename):
    path = os.path.join(FOLDER, filename)
    ratio = None
    non_uniform = False
    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("ratio: "):
                ratio = float(line.split(": ", 1)[1])
            elif line == "nonUniform: yes":
                non_uniform = True

    is_square = ratio is not None and abs(ratio - 1.0) < 0.01
    is_very_wide = ratio is not None and ratio > 2.5
    if non_uniform or is_square or is_very_wide:
        return EXCEPTIONAL_ASPECT_RATIO_PENALTY
    return 0

def load_symbol_metadata(filename):
    """Read the central-symbol metadata written by preprocess.py."""
    detected = False
    comparable = True
    colors = []
    in_symbol_section = False
    with open(os.path.join(FOLDER, filename), "r") as file:
        for line in file:
            line = line.strip()
            if line == "Central symbol:":
                in_symbol_section = True
            elif in_symbol_section and line == "detected: yes":
                detected = True
            elif in_symbol_section and line == "detected: not checked":
                comparable = False
            elif in_symbol_section and line.startswith("new colors: "):
                colors = ast.literal_eval(line.split(": ", 1)[1])
    return {"detected": detected, "comparable": comparable, "colors": set(colors)}

def symbol_colour_penalty(first_symbol, second_symbol):
    if not (
        first_symbol["comparable"] and second_symbol["comparable"]
        and first_symbol["detected"] and second_symbol["detected"]
    ):
        return 0
    all_colors = first_symbol["colors"] | second_symbol["colors"]
    if not all_colors:
        return 0
    common = first_symbol["colors"] & second_symbol["colors"]
    return SYMBOL_COLOUR_PENALTY * (1 - len(common) / len(all_colors))

def load_flag_type(filename):
    """Read the coarse layout type written by preprocessing."""
    category = None
    with open(os.path.join(FOLDER, filename), "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("category: "):
                category = line.split(": ", 1)[1]
                break

    comparable = category is not None and category not in {
        "flag mixed", "other / mixed pattern", "other / transparent edge",
    }
    width_type = None
    if category and category.endswith("(same widths)"):
        width_type = "same widths"
    elif category and category.endswith("(unequal widths)"):
        width_type = "unequal widths"
    family = category.rsplit(" (", 1)[0] if width_type else category
    return {"category": category, "family": family,
            "width_type": width_type, "comparable": comparable}

def flag_type_penalty(first_type, second_type):
    """Return a modest deduction for incompatible coarse flag layouts."""
    if not first_type["comparable"] and not second_type["comparable"]:
        return 0
    if first_type["comparable"] != second_type["comparable"]:
        return FLAG_TYPE_MISMATCH_PENALTY
    if first_type["family"] != second_type["family"]:
        return FLAG_TYPE_MISMATCH_PENALTY
    if first_type["width_type"] != second_type["width_type"]:
        return STRIPE_WIDTH_MISMATCH_PENALTY
    return 0

def load_abstracts(filename):
    """Load the simplified edge-colour sequences saved by preprocess.py."""
    path = os.path.join(FOLDER, filename)
    abstracts = {}
    in_abstract_section = False

    with open(path, "r") as file:
        for line in file:
            line = line.strip()
            if line == "Abstract:":
                in_abstract_section = True
                continue
            if not in_abstract_section or not line.startswith("abstract"):
                continue
            name, values = line.split(": ", 1)
            abstracts[name] = ast.literal_eval(values)

    for axis in ("X", "Y"):
        merged_name = f"abstract{axis}"
        if merged_name in abstracts:
            abstracts[f"abstract{axis}1"] = abstracts[merged_name]
            abstracts[f"abstract{axis}2"] = abstracts[merged_name]
    return abstracts

def abstract_similarity(target, candidate):
    """Compare exact edges and weaker X-to-Y rotation-like matches."""
    score = 0
    maximum_score = 0

    def compare_edges(target_edge, candidate_edge, weight):
        nonlocal score, maximum_score
        target_edge = target_edge[::2]
        candidate_edge = candidate_edge[::2]
        target_length = len(target_edge)
        candidate_length = len(candidate_edge)

        maximum_score += weight
        if target_length == 0 and candidate_length == 0:
            score += weight
        elif target_length and candidate_length:
            score += weight * min(target_length, candidate_length) / max(
                target_length, candidate_length
            )

        maximum_score += weight * max(target_length, candidate_length)
        for target_color, candidate_color in zip(target_edge, candidate_edge):
            if base_color(target_color) == base_color(candidate_color):
                score += weight

    for edge in ("abstractX1", "abstractX2", "abstractY1", "abstractY2"):
        compare_edges(target.get(edge, []), candidate.get(edge, []), weight=1.0)

    for target_edge, candidate_edge in (
        ("abstractX1", "abstractY1"),
        ("abstractX2", "abstractY2"),
        ("abstractY1", "abstractX1"),
        ("abstractY2", "abstractX2"),
    ):
        compare_edges(
            target.get(target_edge, []),
            candidate.get(candidate_edge, []),
            weight=0.5,
        )

    return score / maximum_score * 100 if maximum_score else 0

def similarity(target, candidate):
    """Compare the legacy base-colour percentages."""
    def merge_base_colours(colours):
        merged = {}
        for color, percentage in colours.items():
            base = base_color(color)
            merged[base] = merged.get(base, 0) + percentage
        return merged

    target_base = merge_base_colours(target)
    candidate_base = merge_base_colours(candidate)
    legacy_similarity = 100 - sum(
        abs(target_base.get(color, 0) - candidate_base.get(color, 0))
        for color in set(target_base) | set(candidate_base)
    )

    return max(legacy_similarity, 0)

# --------------------------------------------------
# LAYOUT SIMILARITY
# --------------------------------------------------
def rgb_to_color_name(r, g, b, a):
    if a == 0:
        return "transparent"
    r = r / 255
    g = g / 255
    b = b / 255
    maximum = max(r, g, b)
    minimum = min(r, g, b)
    difference = maximum - minimum
    if difference < 0.08:
        if maximum < 0.15:
            return "black"
        elif maximum > 0.85:
            return "white"
        elif maximum < 0.35:
            return "dark gray"
        elif maximum > 0.65:
            return "light gray"
        else:
            return "gray"
    if maximum == r:
        hue = (60 * ((g - b) / difference)) % 360
    elif maximum == g:
        hue = 60 * ((b - r) / difference) + 120
    else:
        hue = 60 * ((r - g) / difference) + 240
    if hue < 15 or hue >= 345:
        base = "red"
    elif hue < 45:
        base = "orange"
    elif hue < 70:
        base = "yellow"
    elif hue < 165:
        base = "green"
    elif hue < 195:
        base = "cyan"
    elif hue < 255:
        base = "blue"
    elif hue < 290:
        base = "purple"
    else:
        base = "red"
    if maximum < 0.35:
        return f"dark {base}"
    elif maximum > (0.80 if base == "blue" else 0.75):
        return f"light {base}"
    return base

def pixel_color_similarity(color1, color2):
    if color1 == color2:
        return 1.0
    if color1 == "transparent" or color2 == "transparent":
        return 0.0
    parts1 = color1.split()
    parts2 = color2.split()
    modifier1 = parts1[0] if parts1[0] in ["light", "dark"] else None
    base1 = " ".join(parts1[1:]) if modifier1 else color1
    modifier2 = parts2[0] if parts2[0] in ["light", "dark"] else None
    base2 = " ".join(parts2[1:]) if modifier2 else color2
    if base1 != base2:
        return 0.0
    if modifier1 is None or modifier2 is None:
        return 0.5
    if modifier1 != modifier2:
        return 0.25
    return 0.5

def get_color_image(filename):
    path = os.path.join(FOLDER, filename)
    img = np.array(Image.open(path).convert("RGBA"))
    color_image = np.empty(img.shape[:2], dtype=object)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            r, g, b, a = img[y, x]
            color_image[y, x] = rgb_to_color_name(r, g, b, a)
    return color_image

def layout_similarity(target, candidate, similarity_lookup):
    """Compare cached, numeric colour grids without rereading image files."""
    return similarity_lookup[target, candidate].mean() * 100

# Main Process
all_txt_files = [f for f in os.listdir(FOLDER) if f.endswith(".txt")]
all_flags_data = {f: load_flag(f) for f in all_txt_files}
all_abstracts = (
    {f: load_abstracts(f) for f in all_txt_files}
    if USE_ABSTRACT_SCORE else {}
)
all_aspect_penalties = {f: aspect_ratio_penalty(f) for f in all_txt_files}
all_symbols = {f: load_symbol_metadata(f) for f in all_txt_files}
all_flag_types = {f: load_flag_type(f) for f in all_txt_files}

# First stage: reproduce flaggen.py's palette filter for every target flag.
# A pair is retained when either flag places the other in its own top 50.
colour_scores = {}
for flag1, flag2 in combinations(all_txt_files, 2):
    colour_scores[tuple(sorted((flag1, flag2)))] = similarity(
        all_flags_data[flag1], all_flags_data[flag2]
    )

candidate_pairs = set()
for target in all_txt_files:
    candidates = []
    for candidate in all_txt_files:
        if candidate == target:
            continue
        pair = tuple(sorted((target, candidate)))
        candidates.append((colour_scores[pair], candidate))

    candidates.sort(key=lambda item: item[0], reverse=True)
    for _, candidate in candidates[:CANDIDATE_POOL_SIZE]:
        candidate_pairs.add(tuple(sorted((target, candidate))))

candidate_pairs = sorted(candidate_pairs)

# Cache only the grids which survived the palette filter.
candidate_flags = {
    filename for flag1, flag2 in candidate_pairs
    for filename in (flag1, flag2)
}
named_color_images = {
    filename: get_color_image(filename.replace(".txt", ".png"))
    for filename in candidate_flags
}
color_names = sorted({color for image in named_color_images.values() for color in image.flat})
color_ids = {color: index for index, color in enumerate(color_names)}
color_images = {
    filename: np.vectorize(color_ids.__getitem__)(image)
    for filename, image in named_color_images.items()
}
similarity_lookup = np.array([
    [pixel_color_similarity(color1, color2) for color2 in color_names]
    for color1 in color_names
])

pair_results = []

# Second stage: score every pair selected by at least one target's top-50 list.
for flag1, flag2 in candidate_pairs:
    color_score = colour_scores[(flag1, flag2)]

    layout_score = layout_similarity(
        color_images[flag1], color_images[flag2], similarity_lookup
    )
    first_symbol = all_symbols[flag1]
    second_symbol = all_symbols[flag2]
    symbol_layout_score = None
    if (
        first_symbol["comparable"] and second_symbol["comparable"]
        and first_symbol["detected"] and second_symbol["detected"]
    ):
        first_crop = color_images[flag1][25:75, 25:75]
        second_crop = color_images[flag2][25:75, 25:75]
        symbol_layout_score = layout_similarity(
            first_crop, second_crop, similarity_lookup
        )
    abstract_score = (
        abstract_similarity(all_abstracts[flag1], all_abstracts[flag2])
        if USE_ABSTRACT_SCORE else 0
    )

    combined_terms = color_score*color_score + layout_score*layout_score
    combined_weight = 2
    if USE_ABSTRACT_SCORE:
        combined_terms += abstract_score*abstract_score * 2 / 3
        combined_weight += 2 / 3
    if symbol_layout_score is not None:
        combined_terms += symbol_layout_score*symbol_layout_score * SYMBOL_LAYOUT_WEIGHT
        combined_weight += SYMBOL_LAYOUT_WEIGHT
    combined_score = m.sqrt(combined_terms / combined_weight)

    symbol_penalty = symbol_colour_penalty(first_symbol, second_symbol)
    if (
        first_symbol["comparable"] and second_symbol["comparable"]
        and first_symbol["detected"] != second_symbol["detected"]
    ):
        symbol_penalty += SYMBOL_PRESENCE_PENALTY
    type_penalty = flag_type_penalty(
        all_flag_types[flag1], all_flag_types[flag2]
    )
    combined_score = max(
        combined_score - all_aspect_penalties[flag1] - all_aspect_penalties[flag2]
        - symbol_penalty - type_penalty,
        0,
    )
    pair_results.append((flag1, flag2, combined_score))

# Sort by highest combined score
pair_results.sort(key=lambda x: x[2], reverse=True)

print(f"Top {NUMBER_OF_RESULTS} most similar flag pairs:")
print("-" * 40)
for f1, f2, score in pair_results[:NUMBER_OF_RESULTS]:
    label1 = f1.replace(".txt", "")
    label2 = f2.replace(".txt", "")
    if all_aspect_penalties[f1]:
        label1 += " (exceptional ratio)"
    if all_aspect_penalties[f2]:
        label2 += " (exceptional ratio)"
    print(f"{label1} <--> {label2}: {score:.2f}%")

# --------------------------------------------------
# CLUSTERED GROUPS (Complete-linkage above threshold)
# --------------------------------------------------
pair_score_lookup = {
    tuple(sorted((f1, f2))): score
    for f1, f2, score in pair_results
}

clusters = []
for f1, f2, s in pair_results:
    if s < CLUSTER_SIMILARITY_THRESHOLD:
        break
    c1 = next((c for c in clusters if f1 in c), None)
    c2 = next((c for c in clusters if f2 in c), None)

    if c1 is None and c2 is None:
        clusters.append({f1, f2})
    elif c1 is not None and c2 is None:
        if all(pair_score_lookup.get(tuple(sorted((m, f2))), 0) >= CLUSTER_SIMILARITY_THRESHOLD for m in c1):
            c1.add(f2)
    elif c1 is None and c2 is not None:
        if all(pair_score_lookup.get(tuple(sorted((m, f1))), 0) >= CLUSTER_SIMILARITY_THRESHOLD for m in c2):
            c2.add(f1)
    elif c1 is not c2:
        if all(pair_score_lookup.get(tuple(sorted((m1, m2))), 0) >= CLUSTER_SIMILARITY_THRESHOLD for m1 in c1 for m2 in c2):
            c1.update(c2)
            clusters.remove(c2)

print()
print(f"Top {min(NUMBER_OF_RESULTS, len(clusters))} clustered groups of similar flags (complete linkage >= {CLUSTER_SIMILARITY_THRESHOLD:.0f}%):")
print("-" * 75)

for index, cluster in enumerate(clusters[:NUMBER_OF_RESULTS], 1):
    cluster_flags = sorted(list(cluster))
    c_pairs = [
        ((m1, m2), pair_score_lookup[tuple(sorted((m1, m2)))])
        for m1, m2 in combinations(cluster_flags, 2)
    ]
    c_pairs.sort(key=lambda x: x[1], reverse=True)
    avg_score = sum(s for _, s in c_pairs) / len(c_pairs)
    max_score = c_pairs[0][1]

    names_str = ", ".join(m.replace(".txt", "") for m in cluster_flags)
    if len(cluster_flags) == 2:
        header = f"{index}. Gruppe (2 Flaggen | Ähnlichkeit: {max_score:.2f}%): {names_str}"
    else:
        header = f"{index}. Gruppe ({len(cluster_flags)} Flaggen | Max: {max_score:.2f}% | Ø: {avg_score:.2f}%): {names_str}"

    print(header)
    for (m1, m2), s in c_pairs:
        l1 = m1.replace(".txt", "")
        l2 = m2.replace(".txt", "")
        if all_aspect_penalties[m1]:
            l1 += " (exceptional ratio)"
        if all_aspect_penalties[m2]:
            l2 += " (exceptional ratio)"
        print(f"     • {l1} <--> {l2}: {s:.2f}%")
    print()
