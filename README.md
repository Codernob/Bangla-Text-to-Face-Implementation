# Official Implementation of the IJACSA paper : Facial Image Generation from Bangla Textual Description using DCGAN and Bangla FastText

This repository contains the official implementation of the IJACSA paper titled "Facial Image Generation from Bangla Textual Description using DCGAN and Bangla FastText" [1].
This implementation is built based on the FGTD [2] github repository.

Overall System:
![download](https://github.com/Codernob/Bangla-Text-to-Face-Implementation/assets/55651740/5c8f1428-9c8e-4932-9fd1-97ad94425d7e)

Qualitative Results:
![generated images and captions grid](https://github.com/Codernob/Bangla-Text-to-Face-Implementation/assets/55651740/1d25d396-6bcb-40b6-9888-4cc3f6c9778a)

Computational Feasibility:
The DCGAN and SAGAN models can run within 8 GB VRAM; while DFGAN requires 12 GB VRAM. Training time is no more than 1 day on a single RTX 3060.

Please read our paper [1] for further details.

Steps for running our code:
1. Adopt the `fgbtd.yml` anaconda environment.
2. To use the Bangla FastText[4] pretrained text encoder model, Collect the `Bangla_FastText_skipgram.pickle` file from https://drive.google.com/uc?id=1ENn6e9wvVNgrVufflmQvascLgPVQFEfp
   and in `scripts/text_encoder/sentence_encoder.py'`, edit the location of `Bangla_FastText_skipgram.pickle` accordingly.
3. Download images of the CelebA dataset[3] and place in `dataset/img_align_celeba`
   https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
4. In `Face-GANs/`, you may edit the absolute paths as required.
5. You are now free to train, pause, resume, evaluate the DCGAN, SAGAN, DFGAN models by using the jupyter notebook files provided in `Face-GANs`. Remember to set input dimension of generator to 300 for Bangla FastText and 768 for sbnltk sentence transformer.
6. Some sample scripts for calculating FID, IS, LPIPS, FSS, FSD are provided in `evaluation`.

If you use ideas from our paper [1], kindly cite it.
`@article{arnob2023facial,
  title={Facial Image Generation from Bangla Textual Description using DCGAN and Bangla FastText},
  author={Arnob, Noor Mairukh Khan and Rahman, Nakiba Nuren and Mahmud, Saiyara and Uddin, Md Nahiyan and Rahman, Rashik and Saha, Aloke Kumar},
  journal={International Journal of Advanced Computer Science and Applications},
  volume={14},
  number={6},
  year={2023},
  publisher={Science and Information (SAI) Organization Limited}
}`

# References
[1] http://dx.doi.org/10.14569/IJACSA.2023.01406134 <br />
[2] https://github.com/kad99kev/FGTD <br />
[3] https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html <br />
[4] https://www.mdpi.com/2076-3417/12/6/2848 <br />
