# Image_Deraining_For_Autonomous_Driving
DIP Final Project :


Autonomous driving technology relies heavily on clear and accurate visual data to make split-second decisions. In adverse weather conditions, such as snowy or foggy environments, the visibility of road markings, obstacles, and other key features is significantly reduced, creating challenges for both perception and decision-making systems. Snow, in particular, is a complex atmospheric condition that not only obstructs the field of view with falling snow particles and streaks but also covers road surfaces, markings, and objects in a way that conventional image processing algorithms struggle to handle. Additionally, the veiling effect caused by snow’s reflective properties further reduces image contrast and obscures crucial visual information.


Traditionally, deep learning approaches have been employed to remove such obstructions, with various machine learning models trained to recognize and remove snow artifacts from images. However, learning-based methods require extensive, diverse, and often highly specific datasets to be effective. This dependence on training data poses several challenges:


Data Collection and Diversity: Building a comprehensive dataset that captures all types of snow patterns, intensities, and scenarios encountered on the road is impractical. Real-world snow scenes vary greatly depending on geographical location, season, time of day, and even the unique microclimates around certain regions.
Real-Time Performance: Deep learning models tend to be computationally heavy and may not meet the real-time constraints of autonomous driving hardware. Processing speed is critical to maintaining safe navigation in rapidly changing driving environments, and the latency introduced by complex learning models can be a safety concern.
Adaptability: A learning-based approach trained on one dataset may not generalize well to new or unseen snow conditions, limiting its reliability in real-world deployment.
Given these challenges, our project proposes a non-learning-based desnowing system for autonomous driving. This approach focuses on leveraging classical image processing techniques, frequency domain transformations, and multi-spectral data to identify and remove snow obstructions from images in real time. By avoiding dependency on large training datasets, we ensure that the system remains adaptable to a wide range of snow conditions, regardless of where or when they occur. Additionally, the non-learning nature of our system offers faster and more efficient processing, enabling it to meet the low-latency demands of autonomous driving hardware.
