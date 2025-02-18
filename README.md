# Multitask Learning for Improved Late Mechanical Activation Detection of Heart from Cine Dense MRI
![pipeline-new-2-1](https://github.com/user-attachments/assets/24e3b110-cda2-4d61-8245-6ae6358e0b49)

This repository contains the official implementation of the paper: [Multitask Learning for Improved Late Mechanical Activation Detection of Heart from Cine Dense MRI | IEEE Conference Publication](https://ieeexplore.ieee.org/document/10230782/figures#figures). 

The paper introduces a multi-task deep learning framework that simultaneously estimates Late Mechanical Activation (LMA) and classifies scar-free LMA regions based on cine Displacement Encoding with Stimulated Echoes (DENSE) Magnetic Resonance Imaging (MRI). The proposed model demonstrates improved robustness to complex patterns caused by myocardial scars, significantly enhancing the accuracy of LMA detection and scar classification.

## Abstract

The selection of an optimal pacing site, which is ideally scar-free and late activated, is critical to the response of cardiac resynchronization therapy (CRT). Despite the success of current approaches formulating the detection of such late mechanical activation (LMA) regions as a problem of activation time regression, their accuracy remains unsatisfactory, particularly in cases where myocardial scar exists. To address this issue, this paper introduces a multi-task deep learning framework that simultaneously estimates LMA amount and classifies the scar-free LMA regions based on cine displacement encoding with stimulated echoes (DENSE) magnetic resonance imaging (MRI). With a newly introduced auxiliary LMA region classification sub-network, our proposed model shows more robustness to the complex pattern caused by myocardial scar, significantly eliminates their negative effects in LMA detection, and in turn improves the performance of scar classification. To evaluate the effectiveness of our method, we test our model on real cardiac MR images and compare the predicted LMA with the state-of-the-art approaches. It shows that our approach achieves substantially increased accuracy. In addition, we employ the gradient-weighted class activation mapping (Grad-CAM) to visualize the feature maps learned by all methods. Experimental results suggest that our proposed model better recognizes the LMA region pattern.

## Usage

### Training the Model

To train the model, run the training script:

```bash
python train.py
```

The training script will load the configuration file, preprocess the data, and train the model. The trained model will be saved in the `./trained_networks` directory.

### Running Inference

To run inference on the trained model, use the inference script:

```bash
python test.py
```

The inference script will load the trained model, preprocess the input data, and generate predictions. The results will be saved in the `./results` directory.

### Visualizing Results

The repository includes tools for visualizing the results, including:

- **Grad-CAM**: Visualize the feature maps learned by the model.
- **3D Activation Maps**: Generate 3D activation maps to visualize the LMA regions.

To generate visualizations, use the provided plotting functions in the `utils.plot` module.


## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@article{yourcitationkey,
  title={Multitask Learning for Improved Late Mechanical Activation Detection of Heart from Cine Dense MRI},
  author={Your Name and Co-authors},
  journal={Journal Name},
  year={2023},
  publisher={Publisher}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or issues, please open an issue on GitHub or contact the author at [your.email@example.com](mailto:your.email@example.com).
