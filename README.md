# Outline
* Our Streamlit Demo! (To be updated soon)
* [Tables and Figures]()
* [Code Details]()
* [Data Details]()

# Tables and Figures

| Table | Link |
| -------------| -----|
| Table 1 | [compare_augmented_models.ipynb](compare_augmented_models.ipynb) |
| Table 2 | [compare_augmented_models.ipynb](compare_augmented_models.ipynb) |
| Table 3 | [compare_models_fine_tune_india.ipynb](compare_models_fine_tune_india.ipynb) |
| Table A | [compare_models.ipynb](compare_models.ipynb) & [compare_augmented_models.ipynb](compare_augmented_models.ipynb) |
| Table B | [compare_models.ipynb](compare_models.ipynb) & [compare_augmented_models.ipynb](compare_augmented_models.ipynb) |

| Figure | Link |
| -------------| -----|
| Figure 1 | Not Applicable |
| Figure 2       | [Brick kilns in Delhi-NCR](figure2.ipynb) |
| Figure 4 | [model_gradcam.ipynb](model_gradcam.ipynb) |
| Figure 5       | [Brick kilns identified by model](figure5.ipynb) |
| Figure 6 | [year_wise_gain.ipynb](year_wise_gain.ipynb) |

# Code Details

We have the follwoing code files:
1. `compare_models.ipynb` : Model trained on Bangladesh train dataset and tested on Bangaldesh test and Indian test dataset.
2. `compare_augmented_models.ipynb` : Model trained on Augmented Bangladesh train dataset and tested on Bangaldesh test and Indian test dataset. We can generate `Table 1`, `Table 2` and  from this notebook.
3. `compare_models_fine_tune_india.ipynb` : Model trained on Indian train dataset and Indian test dataset. We can generate `Table 3 ` from this notebook.
4. `model_gradcam.ipynb` : We have generated heatcam for all the images missclassified by our model. We can generate `Figure 4` from this notebook.
We can generate Appendix `Table A` and `Table B` from `compare_models.ipynb` and `compare_augmented_models.ipynb`.
5. `utils.py`: Contains all the functions and model definations.

# Data Details

To run the scripts, open a terminal and use the following command:

```python <python_file_name> <csv_file_name> <output_folder_name>```

| CSV        | Description                   |
| -------------- | -------------------------------- |
|Indo_gangetic_dataset.csv| [CSV for genrating the 80,000 Indo-gangetic region dataset ](Indo_gangetic_dataset.csv)|
| manually_identified_kilns.csv    | [Manually identified brick kilns ](manually_identified_kilns.csv)|
| model_identified_kilns.csv    | [Model identified brick kilns ](model_identified_kilns.csv)|
| data_download.py  | [Script to generate 80,000 images using 188 manually identified kilns ](data_download.py)|
| data.py    | [Script to download satellite images](data.py)|