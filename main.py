import argparse
import math
import sys
from dataclasses import dataclass
from typing import List

import numpy as np
import svgwrite
from PIL import Image
from tqdm import tqdm
import numba

def map_value(value, from_min, from_max, to_min, to_max):
    if from_max == from_min:
        return (to_min + to_max) / 2
    return (value - from_min) * (to_max - to_min) / (from_max - from_min) + to_min

# Takes start and end of a line and returns a np array with the pixels that the line passes over most appropriately
# Exact implementation as found on wikipedia
@numba.jit(nopython=True)
def _bresenham(x0, y0, x1, y1, width, height):
    pixels = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        px, py = int(x0), int(y0)
        if 0 <= px < width and 0 <= py < height:
            pixels.append((np.int16(py), np.int16(px)))
        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err

        if e2 > -dy:
            err -= dy
            x0 += sx

        if e2 < dx:
            err += dx
            y0 += sy

    return np.array(pixels, dtype=np.int16)

# Main line finding engine. Runs in parallel across all CPU cores, calculates a loss for each of the lines
# and finds the one which most effectively contributes to the output image.
#
# Simple greedy algorithm just made parallel.
@numba.jit(nopython=True, parallel=True)
def find_line(
    num_nails,
    current,
    cache,
    lookup,
    color,
    fade_factor,
    src_img,
    curr_img,
    prev_idxs
):
    loss = np.full(num_nails, np.inf, dtype=np.float32)

    for i in numba.prange(num_nails):
        if i == current:
            continue

        is_previous = False
        for prev_idx in prev_idxs:
            if i == prev_idx:
                is_previous = True
                break

        if is_previous:
            continue

        line_index = lookup[current, i]
        if line_index == -1:
            continue

        pixel_arr = cache[line_index]
        if pixel_arr.shape[0] == 0:
            continue

        total_diff = 0.0
        for p_idx in range(pixel_arr.shape[0]):
            y, x = pixel_arr[p_idx, 0], pixel_arr[p_idx, 1]
            orig_pixel = src_img[y, x]
            curr_pixel = curr_img[y, x]

            new_pixel = color * fade_factor + curr_pixel * (1 - fade_factor)

            e0 = np.sum(np.abs(orig_pixel - curr_pixel))
            e1 = np.sum(np.abs(orig_pixel - new_pixel))
            pixel_diff = e1 - e0

            if pixel_diff > 0:
                pixel_diff *= 0.2

            total_diff += pixel_diff

        score = total_diff / len(pixel_arr)
        loss[i] = score**3 if score < 0 else score

    min_err = np.min(loss)
    best_nail_idx = np.argmin(loss)

    return best_nail_idx, min_err


# Bleeds the color of existing threads into the new thread added so it looks good with so many threads and not cluttered
# Can play around with the fade_factor to find which one looks the best. Higher fade_factor implies more fading
# obviously.
@numba.jit(nopython=True)
def _draw_line(pixel_arr, curr_img, color, fade_factor):
    for i in range(pixel_arr.shape[0]):
        y, x = pixel_arr[i, 0], pixel_arr[i, 1]
        curr_pixel = curr_img[y, x]
        new_pixel = color * fade_factor + curr_pixel * (1 - fade_factor)
        for j in range(4):
            if new_pixel[j] < 0:
                new_pixel[j] = 0
            elif new_pixel[j] > 255:
                new_pixel[j] = 255

        curr_img[y, x] = new_pixel

@dataclass
class Color:
    r: int
    g: int
    b: int
    a: int = 255

@dataclass
class Point:
    x: float
    y: float


# Main class
class StringArtGenerator:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.width = 30.0
        self.height = 30.0
        self.radius = 10.0
        self.thread_diam = 0.01
        self.fade_factor = 1 / (self.downscale_factor * 1.8)
        self.nails_pos: List[Point] = []
        self.threads: List['Thread'] = []
        self.src_img_data: np.ndarray = None
        self.curr_img_data: np.ndarray = None
        self.dwg = svgwrite.Drawing(
            self.output_svg,
            viewBox=f'{-self.width / 2} {-self.height / 2} {self.width} {self.height}'
        )

        # The cache stores the line data, and the lookup serves as a 2D lookup table for any pixel.
        self.cache = None
        self.lookup = None


    # Initialize the nails, calculates the x, y position of the nails
    def _init_nails(self):
        print("Initializing square grid...")
        self.num_nails = self.nails_side*self.nails_side

        for i in range(self.nails_side):
            for j in range(self.nails_side):
                x = map_value(j, 0, self.nails_side - 1, -self.radius, self.radius)
                y = map_value(i, 0, self.nails_side - 1, -self.radius, self.radius)
                self.nails_pos.append(Point(x, y))

    # Converts from SVG vector space to pixel coordinate space.
    def _get_image_point(self, svg_point: Point) -> Point:
        x = math.floor(map_value(svg_point.x, -self.radius, self.radius, 0, self.width - 1))
        y = math.floor(map_value(svg_point.y, -self.radius, self.radius, 0, self.height - 1))
        return Point(x, y)


    # Loads the image and resizes it.
    # TODO: Make this thing consistent as in a fixed height and width so that the line cache
    # can be stored permanently and be used for any image.
    #
    # Might have to go for upscaling or downscaling as required not a big deal, likely 256x256
    # will be the standard way to go size.
    def _load_img(self):
        try:
            img = Image.open(self.input_image).convert("RGBA")
        except FileNotFoundError:
            print(f"Error: Input image not found at '{self.input_image}'")
            sys.exit(1)

        max_res = (self.radius * 2 / self.thread_diam) / self.downscale_factor
        img_ar = img.width / img.height
        self.width = int(max_res)
        self.height = int(max_res / img_ar) if img_ar > 0 else int(max_res)
        resized_img = img.resize((self.width, self.height), Image.LANCZOS)
        self.src_img_data = np.array(resized_img, dtype=np.float32)
        self.curr_img_data = np.full_like(self.src_img_data, [128, 128, 128, 255], dtype=np.float32)

    # Initializes the cache by computing all the possible lines, and storing the pixels they
    # mpass through before hand.
    #
    # Memory heavy but saves a ton of computation time.
    #
    # Uses the bresenham line algorithm to find the pixels the line actually passes through1
    def _init_cache(self):
        self.lookup = np.full((self.num_nails, self.num_nails), -1, dtype=np.int32)

        all_lines_pixels = []
        line_counter = 0

        image_points = [self._get_image_point(p) for p in self.nails_pos]

        for i in tqdm(range(self.num_nails), desc="Caching lines"):
            for j in range(i + 1, self.num_nails):
                p1 = image_points[i]
                p2 = image_points[j]
                pixels = _bresenham(p1.x, p1.y, p2.x, p2.y, self.width, self.height)
                all_lines_pixels.append(pixels)
                self.lookup[i, j] = self.lookup[j, i] = line_counter
                line_counter += 1

        self.cache = numba.typed.List(all_lines_pixels)
        print("Cached.")

    def run(self):
        self._init_nails()
        self._load_img()
        self._init_cache()

        self.threads = [Thread(self, 0, Color(0, 0, 0)), Thread(self, 0, Color(255, 255, 255))]
        self.dwg.add(
            self.dwg.rect(
                insert=(-self.width / 2, -self.height / 2),
                size=(self.width, self.height),
                fill='grey'
            )
        )

        string_group = self.dwg.g(class_='strings')
        self.dwg.add(string_group)

        for _ in tqdm(range(self.max_lines), desc="Processing lines", unit="line"):
            weights = [t.next_weight() for t in self.threads]
            best_thread_index = np.argmin(weights)

            if weights[best_thread_index] == np.inf:
                print("\nNo more beneficial strings.")
                break

            best_thread = self.threads[best_thread_index]
            start_nail_idx = best_thread.current
            end_nail_idx = best_thread.next_nail()

            if end_nail_idx is None: break

            start_pos, end_pos, color = self.nails_pos[start_nail_idx], self.nails_pos[end_nail_idx], best_thread.color
            string_group.add(
                self.dwg.line(
                    start=(start_pos.x, start_pos.y),
                    end=(end_pos.x, end_pos.y),
                    stroke=f'rgb({color.r},{color.g},{color.b})',
                    stroke_opacity=self.thread_opacity,
                    stroke_width=self.thread_diam
                )
            )

        self._finalize_and_save()

    # Simply save the svg file.
    def _finalize_and_save(self):
        print(f"Saving SVG to '{self.output_svg}'...")
        self.dwg.save()
        print("Done.")

class Thread:
    # Initialization obviously.
    def __init__(self, generator: StringArtGenerator, start_nail: int, color: Color):
        self.generator = generator
        self.current = start_nail
        self.color = color
        self.color_np = np.array([color.r, color.g, color.b, color.a], dtype=np.float32)
        self.nail_order = [start_nail]
        self.prev_connections = {}
        self.next_nail = -1

    # The weight represents how good a line is. Higher weight means better line.
    # This figures out the best line by calling the find_line method, mostly a
    # wrapper function.
    def next_weight(self) -> float:
        prev_indices = np.array(list(self.prev_connections.get(self.current, {}).keys()), dtype=np.int32)

        best_nail, min_err = find_line(
            self.generator.num_nails,
            self.current,
            self.generator.cache,
            self.generator.lookup,
            self.color_np,
            self.generator.fade_factor,
            self.generator.src_img_data,
            self.generator.curr_img_data,
            prev_indices
        )

        self.next_nail = best_nail

        return min_err if min_err < 0 else np.inf

    # Updates the thread state, looks up the pixel data, draws it on canvs, and update the
    # nail_order and prev_connections.
    def next_nail(self) -> int:
        if self.next_nail != -1:
            line_index = self.generator.lookup[self.current, self.next_nail]
            line_pixels = self.generator.cache[line_index]

            _draw_line(line_pixels, self.generator.curr_img_data, self.color_np, self.generator.fade_factor)

            self.prev_connections.setdefault(self.current, {})[self.next_nail] = True
            self.prev_connections.setdefault(self.next_nail, {})[self.current] = True

            self.current = self.next_nail
            self.nail_order.append(self.current)
            return self.current

        return None

def main():
    parser = argparse.ArgumentParser(description="String Art Generator")
    parser.add_argument('--input_image', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--nails_side', type=int, default=17, help="Number of nails per side.")
    parser.add_argument('--max_lines', type=int, default=7500, help="Maximum number of strings to generate.")
    parser.add_argument('--output_svg', type=str, default='output.svg', help="Path for the output SVG file.")
    parser.add_argument('--downscale_factor', type=int, default=6, help="")
    parser.add_argument('--thread_opacity', type=float, default=1.0, help="")
    args = parser.parse_args()

    StringArtGenerator(**vars(args)).run()

if __name__ == "__main__":
    main()
