# longitudinal_interpolation


This repository contains the analysis code for: "Interpolation of Microbiome Composition in Longitudinal Datasets" (Peleg & Borenstein).


## Scripts and files overview

* "Interpolation - Analysis.ipynb": Includes all the analysis and figures that appear in the paper.
* "all interpolation results.zip" - results of interpolation results for each individual and interpolation method. Also include a folder for the results of the interpolationin the Monte-Carlo analysis


### Code:

* MethodAnalysis.py, ToolsForTemporalAnalysis.py - utility functions for interpolation resut analysis
* PlotTemporalFiguresAnalysis.py - code for basic figures based on the interpolation results
* gLV_interpolation.py - code for interpolation using the gLV based methods.
* interpolation_methods.py - code for interpolation using non-gLV methods
* parsing_data.py - code for reading and parsing datasets
