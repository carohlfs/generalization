# Generalization in Neural Networks: A Broad Survey

This repository contains Python and R scripts used to generate the results for the paper "Generalization in Neural Networks: A Broad Survey," forthcoming in *Neurocomputing* (Volume 611, Article 128701, January 1, 2025) by Chris Rohlfs. The paper provides a comprehensive overview of how neural networks generalize, including insights into how models can be applied to new data.

## Table of Contents
- [Overview](#overview)
- [Scripts](#scripts)
  - [Sample Generalization](#sample-generalization)
  - [Distribution Generalization](#distribution-generalization)
  - [Scope Generalization](#scope-generalization)
  - [Explanation Methods](#explanation-methods)
- [Citing This Work](#citing-this-work)
- [License](#license)

## Overview
Each script in this repository demonstrates different concepts discussed in the paper, with code for assessing generalization performance in neural networks, handling overfitting, analyzing distribution shifts, and exploring feature attribution methods.

## Scripts

### Sample Generalization

- **overfit_nn.R**: Examines sample generalization by assessing overfitting on the ImageNet and ImageNet-V2 datasets. For the Python code that produces the training, validation, and test performance, consult the src/1_probabilities subfolder within my [imavis repository](https://github.com/carohlfs/imavis). Given these results, the output is saved as `overfit_nn.png`, which appears as Figure 3a in the paper.

- **norm.py**: Calculates the norm of the weights for neural networks, which is used by `overfit_norm.R`.
  
- **overfit_norm.R**: Similar to `overfit_nn.R`, but uses the norm of the weights as a complexity measure. The output is saved as `overfit_norm.png`, which appears as Figure 3b in the paper.

- Please note that Figure 3c follows a similar structure and uses the complexity calculations performed in the src/3_cost folder in my [imavis repository](https://github.com/carohlfs/imavis).

### Distribution Generalization

- **liontest.py**: Used to calculate the predicted classes for the two DALL-E-generated lion images (`lionhabitat.png` and `lioncity.png`) to assess the 12 neural networks' ability to classify in-context and out-of-context images. Results presented in Table 3.

### Scope Generalization

- **detectron.py**: Uses Meta's Detectron to perform object labeling, which is applied to the `julia.png` image to label the items that are seen in the image of Julia Child's kitchen in Figure 12b.

- **gettysburg.py**: Uses a BERT model to predict a masked word in the Gettysburg Address. The script generates Shapley value visualizations, saved as `gettysburg_shapley.png` and predicted probabilities for the top five choices, saved as `gettysburg_probs.png`. These two images are Figures 14a and 14b in the paper.

### Explanation Methods

- **lionlrp.py**: Uses Layer-wise Relevance Propagation (LRP) to analyze the classification of lion images. Output is saved as `lioncity_lrp_heatmap.png` and `lionhabitat_lrp_heatmap.png` and appears in Figures 15a and 15b.

- **liongradcam.py**: Applies Grad-CAM to visualize important regions for a model’s prediction on lion images. Output is saved as `lioncity_cam_heatmap.png` and `lionhabitat_cam_heatmap.png` and appears in Figures 15c and 15d.

- **cam.py**: Contains code for the both Grad-CAM and LRP for the three top choices from the two models, generating relevance heatmaps for lion images. Output is saved as `lionhabitat_cam_output.png` and `lioncity_cam_output.png` and appears in Figures 16a and 16b.


## Citing This Work

If you use this code or the results from the paper in your research, please cite the following paper:

```bibtex
@article{rohlfs2025generalization,
  title={Generalization in Neural Networks: A Broad Survey},
  author={Chris Rohlfs},
  journal={Neurocomputing},
  volume={611},
  pages={128701},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.neucom.2024.07.045}
}
```

## License
This repository is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.