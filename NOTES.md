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
   until no improving swap is found or a 4-minute wall-clock budget
   (measured from program start) runs out.
3. Print the resulting filename order.

### Large-video fallback

The README doesn't state a maximum frame count, and the full O(n^2)
distance matrix + 2-opt pipeline above gets too slow to finish in time
once there are many thousands of frames. Above `MAX_FULL_MATRIX_FRAMES`
(1500) frames, `order.py` switches to `fast_greedy_path`: a greedy
nearest-neighbor chain built with a KD-tree (`scipy.spatial.cKDTree`)
instead of a full pairwise matrix, using lazy deletion (a visited mask
plus periodic tree rebuilds) to stay close to O(n log n) instead of
O(n^2). No 2-opt refinement is attempted at this scale. Tested against a
synthetic 1700-frame video with continuous motion: 99.9% of consecutive
output steps were within +-2 of the true original index, and a 12,000
frame synthetic stress test completed in ~32s, well under the time
budget.

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
  thousands of frames) the 4-minute total budget may still cut off before
  2-opt converges, leaving some suboptimal local ordering uncorrected
  (though the nearest-neighbor pass always completes first, so output is
  never lost, only less refined).
