# Data Description

- **beam_output**  
  - **Shape:** (9638, 8, 32)  
  - **Represents:** Target labels representing beam scores for each sample.  
  - **Example:** For one sample, a portion of the beam scores might look like:  
    ```
    [[1.51e-06, 1.76e-06, 3.01e-06, ...], 
     [1.77e-06, 2.02e-06, 3.37e-06, ...], 
     ...]
    ```  
    Each value is a float (with an imaginary part of zero), indicating the quality of a specific beam pair.

- **coord_input**  
  - **Shape:** (9638, 2)  
  - **Represents:** 2D coordinates associated with each sample (e.g., spatial positions).  
  - **Example:** The first sample might have coordinates similar to:  
    ```
    [748.92, 624.72]
    ```

- **image_input**  
  - **Shape:** (9638, 48, 81, 1)  
  - **Represents:** Grayscale image data where each pixel is an 8-bit unsigned integer.  
  - **Example:** A snippet from the first image might include pixel values such as:  
    ```
    [[[156], [136], [119], ...],
     [[131], [105], [91], ...],
     [[153], [116], [100], ...],
     ...]
    ```

- **lidar_input**  
  - **Shape:** (9638, 20, 200, 10)  
  - **Represents:** LIDAR data formatted as a multi-channel grid (20×200 with 10 channels), encoding spatial features or intensities. This representation indicates that LIDAR data contains spatial information about obstructions, base stations, and the target vehicle, which can be crucial for predicting beamforming paths.
  - **Semantic Meaning of Voxel Values:**  
    - **-2:** Base Station (BS) location  
    - **-1:** Target vehicle (receiver)  
    - **1:** Obstacles (e.g., other vehicles, buildings, pedestrians, trees)  
    - **0:** Empty space (free path for mmWave signals)  
  - **Example:**  
    ```
    [[[0, 0, 0, ..., 0, 0, 0],
      [0, 0, 0, ..., 0, 0, 0],
      ...,
      [-2, -1, 1, ..., 0, 0, 0]],
     ...]
    ```