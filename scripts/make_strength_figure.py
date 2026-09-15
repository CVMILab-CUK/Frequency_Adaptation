#!/usr/bin/env python3
"""Figure: the same test images under adapter strength and under the cutoff.

Tiles are the saved generations, untouched except for downscaling, taken from
the runs whose numbers Table tab:strength prints, for the first test images in
generation order (the rows of Figure fig:dial):
  reference | fixed-r=0.1 adapter at scale 1.00, 0.75, 0.50, 0.25 (E16)
            | randomised-r adapter at r=0.1, scale 0.50, 0.25 (E12)
            | randomised-r adapter at full strength, r=0.1, 0.3 (E10b)
"""
import os
from PIL import Image

T = 256
IDS = ["00000", "00001"]
COLS = [("results/E10b/reference", "{}.png")] + \
       [(f"results/E16_spec_scale{s}/gen_r0.1", "{}.png") for s in ["1.0", "0.75", "0.5", "0.25"]] + \
       [(f"results/E12_scale{s}/gen_r0.1", "{}.png") for s in ["0.5", "0.25"]] + \
       [(f"results/E10b/gen_r{r}", "{}.png") for r in ["0.1", "0.3"]]
sheet = Image.new("RGB", (T * len(COLS), T * len(IDS)), "white")
for y, i in enumerate(IDS):
    for x, (d, pat) in enumerate(COLS):
        im = Image.open(os.path.join(d, pat.format(i))).convert("RGB").resize((T, T), Image.LANCZOS)
        sheet.paste(im, (x * T, y * T))
sheet.save("papers/figures/strength.png", optimize=True)
print("papers/figures/strength.png", sheet.size)
