# Greedy String Art Generator
This script creates colorful string art from any image using a greedy algorithm to find the best possible line at each step.

<img width="637" height="635" alt="image" src="https://github.com/user-attachments/assets/9d470d01-4a24-4b24-9d51-8c3555381612" />
## Working
### Algortihm
The script creates multiple threads each with a color from a palette defined by the user (or the default one), with each thread starting from a nail. At each step an exhaustive search for the best possible line is done to any other nail. For each of these lines an error score is generated that measures how much would the specific line contribute to the overall resemblance of the art to the target image.

To be more precise, the pixels in the path of the line are first decided by using the Bresenham's Line Algorithm. For each pixel on the path, the difference (say $e1$) between the color on the canvas and the color in the source image is calculated and summed. Similarly the error (say $e2$) is calculated for each of the lines that can be drawn from the present nail. Then the improvement score $e1 - e0$ is determined.

A negative score implies that the line drawn improves the image, and a positive score implies otherwise. A large negative number qualifies as a good move. The line which reduces the error the most is chosen and drawn subsequently, and the whole process repeats for the next line, till no better moves are possible or the maximum number of lines specified by the user have been drawn on the canvas.

### Color and Blending
The greedy algorithm checks which color line improves the art the most in each step. The fade-factor decides how a new string blends with the threads underneath it. A low fade-factor makes the thread very transparent whereas a high fade factor makes the new thread very opaque. The following logic has been implemented to calculate colors while keeping in mind the fade-factor

```python
new_pixel = (new_thread_color * fade_factor) + (current_canvas_color * (1 - fade_factor))
```

## Running 
You must first create a virtual environment to run the script, after which you should install the following required python packages using the following command,
```
pip install numpy Pillow svgwrite tqdm numba
```
A sample command to run the script is as follows
```
python color.py --input_image input.png --output_svg output.svg --max_lines 20000 --nails_side 40
```
You can experiment with the max_lines and nails_side arguments. The script scales linearly with the number of colors supplied, and the number of threads to be drawn. As for the growth with respect to the number of nails, for the cache initialization, if we have $n$ nails then $^n_2 C$ total possible pairs of nails, we would have had, which is approximately $n^2$. But here num_nails is the number of nails per side, so the growth is quartic, i.e. $n^4$ since number of nails = $n*n$. As for the main loop, it scales quadratically, i.e. with the total number of nails on the canvas.

The script accepts the following arguments, only the first one (`input_image`) is required, rest are all optional. 
- `--input_image`: Path to the image file you want to process.
- `--output_svg`: The filename for the final vector art.
- `--nails_side`: The number of nails per side of the square grid.
- `--max_lines`: The total number of string lines to draw.
- `--colors`: A comma-separated list of hex color codes (e.g., FF0000,00FF00,0000FF) to use for the threads.
