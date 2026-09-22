# Mean-Cov Neural Encoding on Ventral-stream Spiking Dataset

This project studies neural encoding in the THINGS Ventral-stream Spiking Dataset (TVSD), a large-scale macaque electrophysiology resource built on the THINGS image database. THINGS contains 1,854 systematically sampled object concepts and 26,107 naturalistic images, and TVSD provides 1,024 electrodes across V1, V4, and IT in two macaques viewing about 22k THINGS images in the ventral-stream experiment. The present work focuses on image-conditioned prediction of visual responses and on interpretable analysis of the learned encoding model.

The proposed framework is a mean-covariance neural encoding model in which, for an image \(x\), the neural response vector \(r\) is described by an image-dependent mean and an image-dependent covariance:

$$
r \mid x \sim \mathcal{N}(\mu_\theta(x), \Sigma_\phi(x))
$$

The mean term captures stimulus-driven firing-rate structure, whereas the covariance term is intended to capture structured trial-to-trial cofluctuation beyond the mean. This decomposition matters because an image may shape not only the expected amplitude of responses but also the pattern of shared variability across electrodes and regions, which motivates the second stage of the model.

The image encoder transforms each image into a feature representation, and the mean model maps those features to predicted neural activity for individual electrodes or channels:

$$
\hat{\mu}(x) = W f_\psi(x) + b
$$

Here, \(f_\psi(x)\) denotes the image encoder and \(W\) projects encoder features into neural-response space. This stage is designed to explain stimulus-locked variation in the expected response using a standard neural-encoding pipeline in which pretrained visual features are transformed into predicted neural responses.

Model interpretability was assessed with occlusion analysis. In this approach, local image patches are masked one at a time and the resulting drop or increase in predicted neural response is measured (Figure 1).

## Figure 1

The top panel shows a V1 example from electrode 405 at time bin 15; the middle panel shows a V4 example from electrode 987 at time bin 19; and the bottom panel shows an IT example from electrode 548 at time bin 23. These panels are intended to illustrate how the same natural image can engage different predictive structure across V1, V4, and IT, with region-dependent differences in timing, feature preference, and attribution pattern. Red regions indicate image locations whose occlusion reduces the predicted response, meaning that those pixels support the model prediction, whereas blue regions indicate locations whose occlusion increases the predicted response, meaning that those pixels suppress or compete with the preferred evidence under the fitted model.

The broader goal is to extend this mean model into an image-conditioned covariance model. In the planned formulation, the covariance component will include latent structure for within-ROI and between-ROI interactions, and these latent components will also depend on the image. This makes it possible to ask not only how the image is reflected in firing rate, but also how image-dependent covariance may help characterize the progression of information from V1 to V4 and then to IT.
