# satprocess
Project to preprocess satellite data, focusing on CHASE data, in an easy-to-use and scalable manner.

This project requires external libraries - to avoid any conflicts, it is best to use this project in a virtual environment.

To setup the dependencies, use '''pip install -r requirements.txt'''.

For aligning an image, the code will look for a folder with CHASE satellite data, downloaded from https://ssdc.nju.edu.cn/NdchaseSatellite.
Currently, this is setup to look for /TestImages, although this can be reconfigured. Simply change the variable "folder_path" in main.

Once a dataset has been processed, the realigning values are saved in a .csv file.
This means that when the program is rerun, it doesn't need to recalculate the values, and should display results faster.
