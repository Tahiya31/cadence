Code for our arXiv pre-print of the paper `Cadence: A Practical Time-series Partitioning Algorithm for Unlabeled IoT Sensor Streams`

Please find the paper here: https://arxiv.org/pdf/2112.03360



**Cadence: A Practical Time-series Partitioning Algorithm for Unlabeled IoT Sensor Streams**
Tahiya Chowdhury, Murtadha Aldeer, Shantanu Laghate, Jorge Ortiz

Timeseries partitioning is an essential step in most machine-learning driven, sensor-based IoT applications. This paper introduces a sample-efficient, robust, time-series segmentation model and algorithm. We show that by learning a representation specifically with the segmentation objective based on maximum mean discrepancy (MMD), our algorithm can robustly detect time-series events across different applications. Our loss function allows us to infer whether consecutive sequences of samples are drawn from the same distribution (null hypothesis) and determines the change-point between pairs that reject the null hypothesis (i.e., come from different distributions). We demonstrate its applicability in a real-world IoT deployment for ambient-sensing based activity recognition. Moreover, while many works on change-point detection exist in the literature, our model is significantly simpler and can be fully trained in 9-93 seconds on average with little variation in hyperparameters for data across different applications. We empirically evaluate Cadence on four popular change point detection (CPD) datasets where Cadence matches or outperforms existing CPD techniques.
