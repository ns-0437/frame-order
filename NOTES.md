# Notes

## Approach

Each frame is decoded, converted to grayscale, and downsampled to a fixed
48x32 thumbnail (size-invariant, fast to compare). I compute the full
pairwise mean-absolute-pixel-difference matrix between all frame
thumbnails, then treat "find the original order" as a shortest
Hamiltonian *path* problem (open-ended TSP): the correct order should be
the ordering that minimizes the sum of frame-to-frame differences along
the path, since consecutive real video frames tend to be visually similar.

Concretely:
1. Build a nearest-neighbor path from a handful of candidate start frames
   (all frames if there are few, otherwise a small sample), keep the
   cheapest one.
2. Improve it with a time-boxed 2-opt local search (vectorized with numpy)
   until no improving swap is found or a ~45s budget runs out.
3. Print the resulting filename order.

This matches the 5-frame `examples/digits` sequence exactly (1,2,3,4,5),
even though that toy example isn't really a "smoothly changing" video —
the nearest-neighbor + 2-opt search still found the one ordering with the
lowest total pairwise difference.

## Where this still goes wrong

- **Direction ambiguity**: minimizing total path length is symmetric —
  reversing the entire output has identical cost. Nothing in the pixel
  data alone tells you which end is the start vs. the end of the clip.
  I don't currently have a reliable general heuristic for this (e.g. a
  consistent motion-direction signal would require optical flow, which
  itself can be noisy and camera/scene dependent), so the output may be
  exactly backwards.
- **Near-static video / low motion**: if most frames are nearly identical
  (e.g. a mostly-still camera), the pairwise distance matrix is close to
  flat and the nearest-neighbor/2-opt search has very little signal to
  work with, so ordering quality degrades toward random.
- **Hard scene cuts or repeated/looping content**: if a video has multiple
  visually similar segments (e.g. it loops, or cuts between two similar
  shots), the global "minimize total difference" objective can connect
  the wrong pair of segments together, producing a locally plausible but
  globally wrong order.
- **Compression/lighting artifacts**: flicker, sensor noise, or
  auto-exposure changes between frames add noise to the pixel-difference
  signal independent of actual motion, which can throw off ordering on
  otherwise static scenes.
- **Runtime budget**: pairwise distances are O(n^2) and 2-opt passes are
  also O(n^2) per full sweep; for very long videos (many hundreds/
  thousands of frames) the 45s 2-opt budget may cut off before it
  converges, leaving some suboptimal local ordering uncorrected.
