# 3 Oct. 2026 - Dorian B.
from PIL import Image
import numpy as np
import os
import colorsys
import math as m
import ast

FOLDER = "Flaggen100"
country = input("Land: ")
TARGET = f"{country}.txt"
NUMBER_OF_RESULTS = 5
CANDIDATE_POOL_SIZE = 50
EXCEPTIONAL_ASPECT_RATIO_PENALTY = 10
USE_ABSTRACT_SCORE = False
SYMBOL_PRESENCE_PENALTY = 5
SYMBOL_COLOUR_PENALTY = 3
SYMBOL_LAYOUT_WEIGHT = 0.2
FLAG_TYPE_MISMATCH_PENALTY = 5
STRIPE_WIDTH_MISMATCH_PENALTY = 2

def parse_color_line(line):

    color_name = line.split(" (")[0]

    pixel_count = int(

        line.split(": ")[1].split(" pixels")[0]

    )

    return color_name, pixel_count

def base_color(color_name):

    """

    Turns:

        light blue -> blue

        dark blue  -> blue

        blue       -> blue

    This means light/dark versions of a color

    are considered related.

    """

    parts = color_name.split()

    if parts[0] in ["light", "dark"]:

        return " ".join(parts[1:])

    return color_name

def load_flag(filename):

    """

    Reads a .txt file and converts pixel counts

    into percentages.

    """

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

    # Convert pixel counts to percentages

    for color in colors:

        colors[color] = colors[color] / total_pixels * 100

    return colors


def aspect_ratio_penalty(filename):
    """Return a fixed penalty for proportions that are hard to compare fairly."""
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
    """Read the descriptive central-symbol result from preprocessing."""
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


def symbol_colour_penalty(target_symbol, candidate_symbol):
    """Return a small penalty proportional to different symbol colours."""
    if not (
        target_symbol["comparable"] and candidate_symbol["comparable"]
        and target_symbol["detected"] and candidate_symbol["detected"]
    ):
        return 0
    all_colors = target_symbol["colors"] | candidate_symbol["colors"]
    if not all_colors:
        return 0
    common = target_symbol["colors"] & candidate_symbol["colors"]
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

    # Removing the width modifier leaves the actual layout category, such as
    # ``3 horizontal stripes`` or ``Nordic cross``.
    family = category.rsplit(" (", 1)[0] if width_type else category
    return {"category": category, "family": family,
            "width_type": width_type, "comparable": comparable}


def flag_type_penalty(target_type, candidate_type):
    """Return a modest deduction for incompatible coarse flag layouts."""
    if not target_type["comparable"] and not candidate_type["comparable"]:
        return 0, None
    if target_type["comparable"] != candidate_type["comparable"]:
        return FLAG_TYPE_MISMATCH_PENALTY, "type mismatch"
    if target_type["family"] != candidate_type["family"]:
        return FLAG_TYPE_MISMATCH_PENALTY, "type mismatch"
    if target_type["width_type"] != candidate_type["width_type"]:
        return STRIPE_WIDTH_MISMATCH_PENALTY, "stripe-width mismatch"
    return 0, None


def load_abstracts(filename):

    """Load the abstract edge color sequences from a flag text file."""

    path = os.path.join(FOLDER, filename)

    abstracts = {}

    in_abstract_section = False

    with open(path, "r") as file:

        for line in file:

            line = line.strip()

            if line == "Abstract:":
                in_abstract_section = True
                continue

            if not in_abstract_section or not line:
                continue

            if not line.startswith("abstract"):
                continue
            name, values = line.split(": ", 1)
            abstracts[name] = ast.literal_eval(values)

    # A merged edge represents the same sequence on both opposing sides.
    for axis in ("X", "Y"):

        merged_name = f"abstract{axis}"

        if merged_name in abstracts:
            abstracts[f"abstract{axis}1"] = abstracts[merged_name]
            abstracts[f"abstract{axis}2"] = abstracts[merged_name]

    return abstracts


def abstract_similarity(target, candidate):

    """Compare edge abstractions, including weaker rotation-like matches."""

    score = 0
    maximum_score = 0

    def compare_edges(target_edge, candidate_edge, weight):
        """Add a weighted score for one pair of abstract flag edges."""
        nonlocal score, maximum_score

        # Abstracts alternate between a colour and its run length.  The
        # optional score currently compares the colour sequence only.
        target_edge = target_edge[::2]
        candidate_edge = candidate_edge[::2]

        target_length = len(target_edge)
        candidate_length = len(candidate_edge)

        # Length is gradual: 3 vs 2 earns 2/3; 3 vs 8 earns 3/8.
        maximum_score += weight
        if target_length == 0 and candidate_length == 0:
            score += weight
        elif target_length and candidate_length:
            score += weight * min(target_length, candidate_length) / max(
                target_length, candidate_length
            )

        # Same-position base colours are still scored independently.
        maximum_score += weight * max(target_length, candidate_length)
        for target_color, candidate_color in zip(target_edge, candidate_edge):
            if base_color(target_color) == base_color(candidate_color):
                score += weight

    exact_edges = ("abstractX1", "abstractX2", "abstractY1", "abstractY2")
    for edge in exact_edges:
        compare_edges(target.get(edge, []), candidate.get(edge, []), weight=1.0)

    # A horizontal layout can be structurally related to a vertical layout.
    # These X-to-Y comparisons together have half of the influence
    # of exact X-to-X / Y-to-Y comparisons, so orientation remains meaningful.
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

# Load target flag

target_colors = load_flag(TARGET)

target_abstracts = load_abstracts(TARGET) if USE_ABSTRACT_SCORE else {}

target_aspect_penalty = aspect_ratio_penalty(TARGET)
target_symbol = load_symbol_metadata(TARGET)
target_flag_type = load_flag_type(TARGET)

results = []

for filename in os.listdir(FOLDER):

    if not filename.endswith(".txt"):

        continue

    if filename == TARGET:

        continue

    candidate_colors = load_flag(filename)

    score = similarity(

        target_colors,

        candidate_colors

    )

    results.append(

        (filename, score, candidate_colors)

    )

# The palette-only ranking is used to select candidates for the more detailed
# comparison below; it is intentionally not printed.
results.sort(key=lambda item: item[1], reverse=True)

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

    # Grayscale

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

    # Hue

    if maximum == r:

        hue = (60 * ((g - b) / difference)) % 360

    elif maximum == g:

        hue = 60 * ((b - r) / difference) + 120

    else:

        hue = 60 * ((r - g) / difference) + 240

    # Base color

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

    # Modifier

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

    modifier1 = None

    modifier2 = None

    if parts1[0] in ["light", "dark"]:

        modifier1 = parts1[0]

        base1 = " ".join(parts1[1:])

    else:

        base1 = color1

    if parts2[0] in ["light", "dark"]:

        modifier2 = parts2[0]

        base2 = " ".join(parts2[1:])

    else:

        base2 = color2

    # Completely different colors

    if base1 != base2:

        return 0.0

    # blue vs light blue

    # blue vs dark blue

    if modifier1 is None or modifier2 is None:

        return 0.5

    # dark blue vs light blue

    if modifier1 != modifier2:

        return 0.25

    return 0.5

def get_color_image(filename):

    path = os.path.join(FOLDER, filename)

    img = np.array(

        Image.open(path).convert("RGBA")

    )

    color_image = np.empty(

        img.shape[:2],

        dtype=object

    )

    for y in range(img.shape[0]):

        for x in range(img.shape[1]):

            r, g, b, a = img[y, x]

            color_image[y, x] = rgb_to_color_name(

                r, g, b, a

            )

    return color_image

def layout_similarity(target_filename, candidate_filename, central_only=False):

    target = get_color_image(target_filename)

    candidate = get_color_image(candidate_filename)

    if central_only:
        height, width = target.shape
        target = target[height // 4:3 * height // 4, width // 4:3 * width // 4]
        candidate = candidate[height // 4:3 * height // 4, width // 4:3 * width // 4]

    total = 0

    for y in range(target.shape[0]):

        for x in range(target.shape[1]):

            total += pixel_color_similarity(

                target[y, x],

                candidate[y, x]

            )

    total_pixels = target.shape[0] * target.shape[1]

    return total / total_pixels * 100

# --------------------------------------------------

# APPLY LAYOUT COMPARISON ONLY TO TOP RESULTS

# --------------------------------------------------

# Use colour composition only as a cheap first-stage filter.  The more useful
# pixel-layout comparison then gets enough candidates to correct that ranking.
top_results = results[:CANDIDATE_POOL_SIZE]

layout_results = []

target_png = TARGET.replace(".txt", ".png")

for filename, color_score, colors in top_results:

    candidate_png = filename.replace(".txt", ".png")

    layout_score = layout_similarity(

        target_png,

        candidate_png

    )

    abstract_score = (
        abstract_similarity(target_abstracts, load_abstracts(filename))
        if USE_ABSTRACT_SCORE else 0
    )
    candidate_symbol = load_symbol_metadata(filename)
    candidate_flag_type = load_flag_type(filename)
    symbol_layout_score = (
        layout_similarity(target_png, candidate_png, central_only=True)
        if (
            target_symbol["comparable"] and candidate_symbol["comparable"]
            and target_symbol["detected"] and candidate_symbol["detected"]
        ) else None
    )

    layout_results.append(

        (

            filename,

            color_score,

            layout_score,

            abstract_score,

            candidate_symbol,

            candidate_flag_type,

            symbol_layout_score,

            colors

        )

    )

# --------------------------------------------------

# COMBINED SCORE

# --------------------------------------------------

for i in range(len(layout_results)):

    (
        filename, color_score, layout_score, abstract_score, candidate_symbol,
        candidate_flag_type, symbol_layout_score, colors
    ) = layout_results[i]

    combined_terms = color_score*color_score + layout_score*layout_score
    combined_weight = 2
    if USE_ABSTRACT_SCORE:
        combined_terms += abstract_score*abstract_score * 2 / 3
        combined_weight += 2 / 3
    if symbol_layout_score is not None:
        combined_terms += symbol_layout_score*symbol_layout_score * SYMBOL_LAYOUT_WEIGHT
        combined_weight += SYMBOL_LAYOUT_WEIGHT
    combined_score = m.sqrt(combined_terms / combined_weight)

    symbol_penalty = symbol_colour_penalty(target_symbol, candidate_symbol)
    if (
        target_symbol["comparable"] and candidate_symbol["comparable"]
        and target_symbol["detected"] != candidate_symbol["detected"]
    ):
        symbol_penalty += SYMBOL_PRESENCE_PENALTY

    type_penalty, type_note = flag_type_penalty(
        target_flag_type, candidate_flag_type
    )

    # Apply the same fixed deduction whether the exceptional flag is the
    # requested target or the candidate being compared with it.
    combined_score = max(
        combined_score - target_aspect_penalty - aspect_ratio_penalty(filename)
        - symbol_penalty - type_penalty,
        0,
    )

    layout_results[i] = (

        filename,

        color_score,

        layout_score,

        abstract_score,

        candidate_symbol,

        candidate_flag_type,

        symbol_layout_score,

        combined_score,

        type_note,

        colors

    )

# Rank by combined score

layout_results.sort(

    key=lambda item: item[7],

    reverse=True

)

print()

target_label = TARGET.replace(".txt", "")
if target_aspect_penalty:
    target_label += " (exceptional ratio)"

print(f"Target: {target_label}")

print()

print(

    f"Top {NUMBER_OF_RESULTS} matches, "

    "ranked by combined score:"

)

print()

for (

    filename,

    color_score,

    layout_score,

    abstract_score,

    candidate_symbol,

    candidate_flag_type,

    symbol_layout_score,

    combined_score,

    type_note,

    colors

) in layout_results[:NUMBER_OF_RESULTS]:

    label = filename.replace(".txt", "")
    if aspect_ratio_penalty(filename):
        label += " (exceptional ratio)"

    score_parts = f"color {color_score:.1f}% | layout {layout_score:.1f}%"
    if USE_ABSTRACT_SCORE:
        score_parts += f" | abstract {abstract_score:.1f}%"
    if symbol_layout_score is not None:
        score_parts += f" | symbol {symbol_layout_score:.1f}%"
    elif (
        target_symbol["comparable"] and candidate_symbol["comparable"]
        and target_symbol["detected"] != candidate_symbol["detected"]
    ):
        score_parts += " | symbol mismatch"
    if type_note:
        score_parts += f" | {type_note}"
    print(f"{label}: {score_parts} | combined {combined_score:.1f}%")
