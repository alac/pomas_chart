# PomasChart
Automatically generates "usage" charts, from a collection of "results" screenshots from Pokemon Masters EX.  A time-saver for a handful of people in Pokemon Masters EX communities.

Currently, it works well with some limitations. The image recognition accuracy is 100% on _identical_ Sync Pairs, so no false positives. But, realistically, it needs to match "**mechanically** identical Sync Pairs" (e.g. EX-upgraded vs not, costume/pose variations). Brainstorming ways to avoid manually tagging groups.

Progress:
- [X] Extract trainer faces
- [X] Group identical Sync Pairs
- [X] Generate charts
- [ ] Group variations of the same Sync Pairs (E.g. pose variations, EX vs non-EX)

## Installation

1. Install python 1.12
2. Open the command line and run this to install

```
git clone [repo url]
cd pomas_chart
pip install -r requirements
```

## Usage

1. Add images to the `in` folder
2. Go to the `pomas_chart` folder and run this command `python main.py`
3. The chart should be in `out\chart.png`

`out` has intermediate outputs. zip them up (along with the `in` folder) and send them along if you have a bug to report.

If you want to tweak the chart, you can also modify `out\groups`; each group should contain the collected screenshots of Sync Pair. You can regenerate the chart after by running `python generate_chart.py`.
