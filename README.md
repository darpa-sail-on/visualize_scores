# Visualize results for 1 Level of Novelty

```
Usage: vis.py [OPTIONS]

  Quick visualization script

Options:
  -r, --result_path DIRECTORY  Filepath to a folder of json results (or a
                               folder of folders of json results)

  -o, --output_path DIRECTORY  Filepath where visualizations should be stored
  --help                       Show this message and exit.
```

Example run:
`python vis.py -r /data/benjamin.pikus/TA1-activity-recognition-self-eval_results/results -o output_res`

# Visualize results for multiple levels of Novelty

```
Usage: vis_across_novelty.py [OPTIONS]

Options:
  -h, --help            show this help message and exit
  --result-root-path RESULT_ROOT_PATH
                        Path to root directory where results are stored
  --algorithms ALGORITHMS [ALGORITHMS ...]
                        Name of the algorithm that would be included in the
                        plots
  --novelty-levels NOVELTY_LEVELS [NOVELTY_LEVELS ...]
                        Novelty levels used in plots
  --timestamps TIMESTAMPS [TIMESTAMPS ...]
                        Timestamps used in plots
  --metrics METRICS [METRICS ...]
                        Metrics used in plots
  --row-subplot ROW_SUBPLOT
                        Row for subplots
  --col-subplot COL_SUBPLOT
                        Column for subplots
  --in-plot IN_PLOT [IN_PLOT ...]
                        Values combined in a plot
  --output-path OUTPUT_PATH
                        Filepath where visualizations should be stored
```

Example run
```
python vis_across_novelty.py --result-root-path /data/SAIL-ON/TA1-activity-recognition-self-eval/activity-recognition/results/raw_results/ \
                             --algorithms gae_kl_nd fixed_x3d adaptive_x3d \
                             --novelty-levels class temporal \
                             --timestamps early in-middle late \
                             --metrics m1 m2 m2.1 no_cdt --row-subplot metrics \
                             --col-subplot novelty_levels \
                             --in-plot algorithms timestamps --output-path two_levels.png
```
