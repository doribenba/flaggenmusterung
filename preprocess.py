# 3 Oct. 2026 - Dorian B.
from PIL import Image
import numpy as np
import os

INPUT_FOLDER = "Flaggen"
OUTPUT_FOLDER = "Flaggen100"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Metadata measured from the original image, before scaling to 100×100.
flag_metadata = {}

for filename in os.listdir(INPUT_FOLDER):
    if not filename.lower().endswith(".png"):
        continue
    input_path = os.path.join(INPUT_FOLDER, filename)
    img = np.array(Image.open(input_path).convert("RGBA"))

    # Non-uniform: transparent pixels inside the bounding box (e.g. Nepal).
    visible = img[:, :, 3] != 0
    visible_y, visible_x = np.where(visible)
    if len(visible_x) == 0:
        raise ValueError(f"{filename} contains no non-transparent pixels")

    left, right = visible_x.min(), visible_x.max()
    top, bottom = visible_y.min(), visible_y.max()
    visible_width = right - left + 1
    visible_height = bottom - top + 1
    non_uniform = not visible[top:bottom + 1, left:right + 1].all()
    flag_metadata[filename] = (
        visible_width / visible_height,
        non_uniform,
    )

    # Use the bounding box for the horizontal crop (avoids hardcoded row indices).
    startX = int(left)
    endX   = int(right)

    # Create 100 × 100 image
    output = np.zeros((100, 100, 4), dtype=np.uint8)
    # Squish flag into 100 × 100
    for newX in range(100):
        oldX = startX + round(
            newX * (endX - startX) / 99
        )
        for newY in range(100):
            oldY = round(
                newY * (img.shape[0] - 1) / 99
            )
            output[newY, newX] = img[oldY, oldX]

    output_path = os.path.join(OUTPUT_FOLDER, filename)

    Image.fromarray(output).save(output_path)

    print(f"{filename}: {startX} → {endX}")

print("Done.")

COLOR_CUTOFF = 80
ABSTRACT_TOLERANCE = 1
MIN_ABSTRACT_RUN_LENGTH = 3  # ignore 1-2px runs (anti-aliasing / thin borders)
EDGE_INSET = 2


def is_different(pixel1, pixel2):
    """Return whether two RGBA pixels differ beyond the abstraction tolerance."""
    return np.linalg.norm(pixel1.astype(int) - pixel2.astype(int)) > ABSTRACT_TOLERANCE


def abstract_edge(pixels):
    """Reduce an edge to alternating colour names and run lengths."""
    runs = []
    for pixel in pixels:
        if runs and not is_different(runs[-1][0], pixel):
            runs[-1][1] += 1
        else:
            runs.append([pixel.copy(), 1])

    named_runs = []
    for pixel, length in runs:
        if pixel[3] == 0 or length < MIN_ABSTRACT_RUN_LENGTH:  # skip transparent / short
            continue

        name = color_name(*pixel[:3])
        # Merge adjacent runs of the same colour split by a discarded short run.
        if named_runs and named_runs[-1][0] == name:
            named_runs[-1][1] += length
        else:
            named_runs.append([name, length])

    abstract = []
    for name, length in named_runs:
        abstract.extend((name, length))

    return abstract


def base_color_name(color):
    """Ignore light/dark prefixes when recognizing coarse flag patterns."""
    parts = color.split()
    return " ".join(parts[1:]) if parts[0] in ("light", "dark") else color


def split_abstract(abstract):
    """Turn [colour, length, ...] into separate colour and length lists."""
    return abstract[::2], abstract[1::2]


def same_widths(lengths):
    """Allow a two-pixel rounding difference after scaling to 100 pixels."""
    return max(lengths) - min(lengths) <= 2


def dominant_edge_colour(colours, lengths):
    """Return a nearly solid edge's main colour, ignoring tiny end remnants."""
    total = sum(lengths)
    if not total:
        return None
    longest = max(range(len(lengths)), key=lengths.__getitem__)
    if lengths[longest] / total >= 0.85:
        return colours[longest]
    return None


def classify_flag(abstract_y1, abstract_y2, abstract_x1, abstract_x2):
    """Identify common flag layouts from their four edge abstractions.
    Uses base colour names so anti-aliasing cannot corrupt the result.
    """
    (y1, y1_lengths), (y2, y2_lengths), (x1, x1_lengths), (x2, x2_lengths) = (
        (
            [base_color_name(color) for color in colors],
            lengths,
        )
        for colors, lengths in (
            split_abstract(edge)
            for edge in (abstract_y1, abstract_y2, abstract_x1, abstract_x2)
        )
    )
    edges = (y1, y2, x1, x2)

    if not all(edges):
        return "other / transparent edge", []

    edge_colours = {color for edge in edges for color in edge}
    if len(edge_colours) == 1:
        return "one-colour background", [next(iter(edge_colours))]

    # Canton: top and hoist edges start with canton colour then background;
    # fly and bottom edges are background only (e.g. Tonga, Taiwan, Samoa).
    if (
        len(y1) == len(x1) == 2
        and len(y2) == len(x2) == 1
        and y1[0] == x1[0]
        and y1[1] == x1[1] == y2[0] == x2[0]
    ):
        return "upper-left rectangle", [y1[0], y1[1]]

    # Nordic cross: same palindromic sequence on all four edges.
    if (
        x1 == x2 == y1 == y2
        and len(y1) >= 3
        and len(y1) % 2 == 1
        and y1 == list(reversed(y1))
    ):
        return "Nordic cross", y1[:len(y1) // 2 + 1]

    # UAE-style: single hoist bar + horizontal stripes to its right.
    if (
        len(y1) == 1
        and len(y2) >= 2
        and len(x1) == len(x2) == 2
        and x1[0] == x2[0] == y1[0]
        and x1[1] == y2[0]
        and x2[1] == y2[-1]
    ):
        return "left vertical stripe with horizontal stripes", [y1[0]] + y2

    # Vertical stripes: left/right edges are the outer stripes; top/bottom list all.
    if (
        x1 == x2
        and len(x1) >= 2
        and len(y1) == len(y2) == 1
        and x1[0] == y1[0]
        and x1[-1] == y2[0]
    ):
        count = "four or more" if len(x1) >= 4 else str(len(x1))
        widths = "same widths" if same_widths(x1_lengths) else "unequal widths"
        return f"{count} vertical stripes ({widths})", x1

    # Triangle on hoist: checked before horizontal stripes because the triangle
    # can leave tiny stripe-colour remnants at the hoist edge.
    triangle_colour = dominant_edge_colour(y1, y1_lengths)
    if (
        triangle_colour is not None
        and len(y2) >= 2
        and len(x1) == len(x2) == 1
        and y2[0] == x1[0]
        and y2[-1] == x2[0]
        and triangle_colour not in y2
    ):
        return "triangle on the left", [triangle_colour] + y2

    # Horizontal stripes are the same construction with X and Y exchanged.
    if (
        len(y1) == len(y2) >= 2
        and len(x1) == len(x2) == 1
        and y1[0] == y2[0] == x1[0]
        and (y1[-1] == x2[0] or y2[-1] == x2[0])
    ):
        count = "four or more" if len(y1) >= 4 else str(len(y1))
        widths = "same widths" if same_widths(y1_lengths) else "unequal widths"
        return f"{count} horizontal stripes ({widths})", y1

    return "other / mixed pattern", sorted(edge_colours)


def detect_central_symbol(img, flag_type, layout_colours):
    """Detect extra colours in the central 50% region.
    Cannot find monochrome symbols that reuse existing flag colours.
    """
    if flag_type.startswith("other /") or flag_type == "flag mixed":
        return "not checked", [], []

    height, width = img.shape[:2]
    crop = img[height // 4:3 * height // 4, width // 4:3 * width // 4]
    colours = {}
    for r, g, b, alpha in crop.reshape(-1, 4):
        if alpha == 0:
            continue
        name = base_color_name(color_name(r, g, b))
        colours[name] = colours.get(name, 0) + 1

    total = sum(colours.values())
    if not total:
        return "not checked", [], []

    # Colours below 3% are assumed to be anti-aliasing.
    significant = {
        name: count / total * 100
        for name, count in colours.items()
        if count / total >= 0.03
    }
    new_colours = sorted(set(significant) - set(layout_colours))
    is_richer = len(significant) >= len(set(layout_colours)) + 2
    detected = "yes" if new_colours or is_richer else "no"
    palette = [
        f"{name} {percentage:.1f}%"
        for name, percentage in sorted(
            significant.items(), key=lambda item: item[1], reverse=True
        )
    ]
    return detected, new_colours, palette


def abstracts_match(first, second):
    """Compare abstract colour runs while allowing minor resize rounding."""
    first_colours, first_lengths = split_abstract(first)
    second_colours, second_lengths = split_abstract(second)
    if len(first_colours) != len(second_colours):
        return False
    return all(
        base_color_name(a) == base_color_name(b)
        and abs(length_a - length_b) <= 3
        for a, length_a, b, length_b in zip(
            first_colours, first_lengths, second_colours, second_lengths
        )
    )


def central_stripe_symbol(img, flag_type, abstract_y1, abstract_y2,
                          abstract_x1, abstract_x2):
    """Detect same-colour symbols on stripe flags via the central scan line.
    A central line matching the edge layout means no symbol; a deviation does.
    """
    # Only plain stripe patterns; excludes UAE-style left-bar flags.
    is_vertical = "vertical stripes (" in flag_type
    is_horizontal = "horizontal stripes (" in flag_type
    if not (is_vertical or is_horizontal):
        return False, None, None

    middle_y = img.shape[0] // 2
    middle_x = img.shape[1] // 2
    central_y = abstract_edge(img[:, middle_x])
    central_x = abstract_edge(img[middle_y, :])
    if is_vertical:
        matches_layout = (
            abstracts_match(central_x, abstract_x1)
            or abstracts_match(central_x, abstract_x2)
        )
    else:
        matches_layout = (
            abstracts_match(central_y, abstract_y1)
            or abstracts_match(central_y, abstract_y2)
        )

    return not matches_layout, central_y, central_x


def write_abstracts(file, img, non_uniform=False):
    """Write edge colour sequences for all four sides.
    Opposing edges that are identical are merged under a shared name.
    """
    # Read slightly inward to skip accidental 1-2px border stripes (e.g. Poland).
    abstract_y1 = abstract_edge(img[:, EDGE_INSET])
    abstract_y2 = abstract_edge(img[:, -1 - EDGE_INSET])
    abstract_x1 = abstract_edge(img[EDGE_INSET, :])
    abstract_x2 = abstract_edge(img[-1 - EDGE_INSET, :])

    file.write("\nAbstract:\n")

    if abstract_y1 == abstract_y2:
        file.write(f"abstractY: {abstract_y1}\n")
    else:
        file.write(f"abstractY1: {abstract_y1}\n")
        file.write(f"abstractY2: {abstract_y2}\n")

    if abstract_x1 == abstract_x2:
        file.write(f"abstractX: {abstract_x1}\n")
    else:
        file.write(f"abstractX1: {abstract_x1}\n")
        file.write(f"abstractX2: {abstract_x2}\n")

    flag_type, colors = classify_flag(
        abstract_y1, abstract_y2, abstract_x1, abstract_x2
    )
    # Non-rectangular flags are distorted by the square grid; skip layout inference.
    if non_uniform:
        flag_type, colors = "flag mixed", []
    file.write("\nFlag type:\n")
    file.write(f"category: {flag_type}\n")
    file.write(f"colors: {colors}\n")

    symbol, new_colours, central_palette = detect_central_symbol(
        img, flag_type, colors
    )
    central_y = central_x = None
    symbol_reason = "new central colours" if symbol == "yes" else None
    if symbol == "no":
        (
            abstract_symbol, central_y, central_x,
        ) = central_stripe_symbol(
            img, flag_type, abstract_y1, abstract_y2, abstract_x1, abstract_x2
        )
        if abstract_symbol:
            symbol = "yes"
            symbol_reason = "central abstract differs from stripe layout"

    file.write("\nCentral symbol:\n")
    file.write(f"detected: {symbol}\n")
    if symbol != "not checked":
        file.write(f"new colors: {new_colours}\n")
        file.write(f"central palette: {central_palette}\n")
        if symbol_reason:
            file.write(f"reason: {symbol_reason}\n")
    if central_y is not None:
        file.write(f"abstractYCenter: {central_y}\n")
        file.write(f"abstractXCenter: {central_x}\n")

def color_name(r, g, b):

    # Normalize to 0–1

    r_norm = r / 255

    g_norm = g / 255

    b_norm = b / 255

    # Find brightness and saturation

    max_value = max(r_norm, g_norm, b_norm)

    min_value = min(r_norm, g_norm, b_norm)

    difference = max_value - min_value

    # Black / dark gray / white / gray

    if max_value < 0.15:

        return "black"

    if difference < 0.08:

        if max_value > 0.85:

            return "white"

        elif max_value < 0.35:

            return "dark gray"

        elif max_value > 0.65:

            return "light gray"

        else:

            return "gray"

    # Calculate hue

    if difference == 0:

        hue = 0

    elif max_value == r_norm:

        hue = (60 * ((g_norm - b_norm) / difference)) % 360

    elif max_value == g_norm:

        hue = 60 * ((b_norm - r_norm) / difference) + 120

    else:

        hue = 60 * ((r_norm - g_norm) / difference) + 240

    # Clamp hue into broad color categories

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

    # Brightness modifier

    if max_value < 0.35:

        return f"dark {base}"

    # Only blue uses the higher cutoff: Nicaragua's saturated RGB(0, 103,
    # 198) is blue, while the established thresholds stay unchanged for all
    # other colours.
    elif max_value > (0.80 if base == "blue" else 0.75):

        return f"light {base}"

    else:

        return base

for filename in os.listdir(OUTPUT_FOLDER):

    if not filename.lower().endswith(".png"):

        continue

    input_path = os.path.join(OUTPUT_FOLDER, filename)

    img = np.array(Image.open(input_path).convert("RGBA"))

    colors = {}

    for x in range(img.shape[1]):

        for y in range(img.shape[0]):

            pixel = tuple(img[y, x])

            # Transparent background is not a flag colour.  In particular,
            # non-rectangular flags must not gain a large artificial black
            # area just because transparent RGBA pixels contain zeroes.
            if pixel[3] == 0:
                continue

            if pixel in colors:

                colors[pixel] += 1

            else:

                colors[pixel] = 1

    output_path = os.path.join(

        OUTPUT_FOLDER,

        filename.replace(".png", ".txt")

    )

    with open(output_path, "w") as file:

        # Ignore tiny anti-aliasing colours, then scale the retained colours
        # to a common 10,000-pixel palette.  Largest-remainder rounding makes
        # the written counts add up to exactly 10,000.
        retained = [
            (color, count) for color, count in colors.items()
            if count >= COLOR_CUTOFF
        ]
        retained_total = sum(count for _, count in retained)
        scaled = []
        for color, count in retained:
            exact_count = count / retained_total * 10_000
            scaled.append([color, int(exact_count), exact_count % 1])

        remaining = 10_000 - sum(count for _, count, _ in scaled)
        for index in sorted(
            range(len(scaled)), key=lambda index: scaled[index][2], reverse=True
        )[:remaining]:
            scaled[index][1] += 1

        for (r, g, b, a), count, _ in sorted(
            scaled, key=lambda item: item[1], reverse=True
        ):
            name = color_name(r, g, b)
            file.write(f"{name} ({r}, {g}, {b}, {a}): {count} pixels\n")

        aspect_ratio, non_uniform = flag_metadata[filename]
        write_abstracts(file, img, non_uniform)

        file.write("\nAspect ratio:\n")
        file.write(f"ratio: {aspect_ratio:.4f}\n")
        file.write(f"nonUniform: {'yes' if non_uniform else 'no'}\n")

