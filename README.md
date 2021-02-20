# AdvantageMemManagement
When dealing with streams of data, classification systems must take into account changes in data or concept drift. For example, predicting the top products on amazon in real time may drift as season change. One mthod of dealing with such change is to build a model for each similar concept. These models can be stored and reused when similar conditions reappear, for example a `summer` model may be stored and reused every summer. 

Managing how these models are stored is an important open research problem. To store a model repository in memory, it becomes neccessary to delete or merge models. The decision of which models to delete or merge can have a large impact on the performance and explainibility of a system. In this project, we propose a new `Advantage` based policy to manage memory. This policy optimizes both performance and explainibility.

For futher details, please see the [journal paper](https://link.springer.com/article/10.1007/s10618-021-00736-w).
![Advantage of a model](https://github.com/BenHals/AdvantageMemManagement/blob/main/readme_img.png?raw=true)
# Implementation

The basic idea of advantage is to measure the benefit of storing and reusing a model compared to rebuilding a replacement model.
A finite state machine is proposed as an efficient method of storing the information needed to compute this measure online.

## Instructions to Run
1. Install packages in requirements.txt
2. Create a data directory. This will contain the data set to run classification systems on. 
3. Create data sets. The system expects a nested folder structure, with the directory names indicating parameters. The bottom level data sets are expected to be in ARFF format. The script `GenerateDatastreamFiles\generate_dataset.py` can be used to create the synthetic data sets used in the paper. The data directory to create files in should be passed using the `-d` commandline argument. Other arguments modify the data sets created, such as `-st` for the type of stream, `-nc` for the total number of concepts, `-hd` and `-ed` for the difficulty levels to use (e.g. the depth of the tree used to label observations in the TREE data set). These are set to the defaults used in the paper.
4. To run classifiers on datasets, use the `FSMAdaptiveClassifier\run_datastream_classifier.py` script. Set commandline arguments. Most are set to reasonable defaults. 
- Set the data directory to use using the `-d` command.
- `-mm` can be used to specify the memory management policy to use. Our proposed variants are: `rA, auc` and `score`.
- The system will detect all data set `ARFF` files and run the specified system on each, creating result files in the same directory following the naming scheme.
5. To compute result measures, run the `SysMemResultsGen\run_csv_results.py` script. This will 1) produce a .txt file alongside each result file containing relevant measures, but also consolidate all results into a `SysMemResultsGen\Mem_Manage_Results\save-name.pickle` file where `save-name` is specified using the `-sn` argument. 
- This pickle file can be plotted using the relevant `SysMemResultsGen\mml_plot_bw.py` or `SysMemResultsGen\noise_plot_bw.py` scripts, passing `save-name.pickle` using the `-f` command.

# Citation
This work is published in the Journal of Data Mining and Knowledge Discovery. Please cite the following paper:
`Halstead, Ben, Yun Sing Koh, Patricia Riddle, Russel Pears, Mykola Pechenizkiy, and Albert Bifet. "Recurring concept memory management in data streams: exploiting data stream concept evolution to improve performance and transparency." Data Mining and Knowledge Discovery (2021): 1-41.`

Or as Bibtex:

`@article{halstead2021recurring,
  title={Recurring concept memory management in data streams: exploiting data stream concept evolution to improve performance and transparency},
  author={Halstead, Ben and Koh, Yun Sing and Riddle, Patricia and Pears, Russel and Pechenizkiy, Mykola and Bifet, Albert},
  journal={Data Mining and Knowledge Discovery},
  pages={1--41},
  year={2021},
  publisher={Springer}
}`