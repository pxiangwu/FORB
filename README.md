## FORB: A Flat Object Retrieval Benchmark for Universal Image Embedding

[Pengxiang Wu](https://scholar.google.com/citations?user=MXLs7GcAAAAJ&hl=en), [Siman Wang](https://github.com/simanw304), [Kevin Dela Rosa](https://perhaps.ai/) [<img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/twitter.svg" width="14" height="14">](https://twitter.com/kdrwins) [<img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/linkedin-in.svg" width="14" height="14">](https://www.linkedin.com/in/kdrosa/) [<img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/brands/github.svg" width="14" height="14">](https://github.com/kdr) [<img src="https://raw.githubusercontent.com/FortAwesome/Font-Awesome/6.x/svgs/solid/graduation-cap.svg" width="14" height="14">](https://scholar.google.com/citations?user=8Pc5MiUAAAAJ&hl=en), [Derek Hao Hu](https://scholar.google.com/citations?user=Ks81aO0AAAAJ&hl=en)

NeurIPS 2023 Datasets and Benchmarks Track ([Paper Link](https://arxiv.org/abs/2309.16249))

---

![](https://github.com/pxiangwu/FORB/blob/master/teaser/images.png)

We introduce a benchmark for evaluating the retrieval performance on flat objects. Specifically, we consider the following objects:

- Animated card
- Photorealisitc card
- Book cover
- Painting
- Currency
- Logo
- Packaged goods
- Movie poster

In this repository, we provide scripts for downloading the benchmark images, as well as supporting code for evaluating various baselines.

The structure of this repository is:

- `baselines/`: contains code for evaluating the performances of different methods.
- `metadata/`: the metadata of the benchmark.
- `downloader/`: contains code for downloading the benchmark images.
- `metric_helper/`: contains some utility functions to help evaluating image retrieval accuracies.

We also offer this [Google Drive Link](https://drive.google.com/file/d/1Oy7wK7khzJhsop3tf7hM1F2V7zrjtAcH/view) for downloading the benchmark data directly.
