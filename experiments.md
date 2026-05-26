# SAM 3 PCS Experiments Log

## Successful Strategies
- **Multi-Class Competition:** Passing all prompts in a single `forward_grounding` pass to force the transformer to decide the best-matching class. This prevents teams with similar visual features from merging.
- **Weighted Spatial Grounding:** Requiring the bottom 25% (foot region) of a person's mask to overlap with the `wooden basketball court floor` mask. This effectively filters out audience and bench players.
- **Visual Descriptors:** Using "white jersey" and "blue jersey" instead of abstract labels like "Team 1".
- **BFloat16 Autocast:** Essential for preventing dtype mismatch errors in fused operations.

## Experiments

| Date | Strategy | Result | Note |
| :--- | :--- | :--- | :--- |
| 2026-05-26 | Concept list (serial) | ❌ Poor | Mixed teams, many spectators detected. |
| 2026-05-26 | Competition Pass | ✅ Good | Teams separated, but some bench players included. |
| 2026-05-26 | Foot-point Filter | ✅ Clean | Very precise on-court detections, but missed jumping players. |
| 2026-05-26 | Inclusive Mask Filter | ❌ Overkill | Recovered all players but brought back half the audience. |
| 2026-05-26 | Weighted Grounding | ✅ Good | Clean separation, but relies heavily on strong Floor detection. |
| 2026-05-26 | Robust ROI Mask | 🏆 **Best** | Combined Floor + Landmarks to prevent missing players when floor mask is patchy. |

## Failed Prompts
- `"players of team 1 on the court"`: Too abstract, low confidence scores.
- `"players in white jerseys, players in blue jerseys, referees"` as a single string: Causes class confusion.
- `"basketball player in white jersey"`: Surprisingly less effective than `"basketball player wearing a white jersey uniform"`.
- `"referee in black pants"`: Poorly identified compared to `"referee in black and white vertical striped shirt"`.

## Technical Learnings
- **ROI Dependency:** If the `court_mask` is built ONLY from the floor, any player whose mask doesn't touch the floor (jumping, or floor mask has holes) is discarded.
- **Dilation fix:** A small dilation (5-10px) on the court mask significantly improves player recall near the boundaries.


## Known Issues
- **Dtype Mismatch:** SAM 3's fused layers force `bfloat16`, causing crashes if the rest of the model is in `float32`. Fixed via `torch.autocast`.
- **Batch Scaling:** `_get_dummy_prompt` must match the number of text prompts to avoid `grid_sample` batch size errors.
