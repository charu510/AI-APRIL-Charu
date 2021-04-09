# AI-APRIL-Charu
- This repository contains the three scripts namely dataset, trainer and detector which is used for facial recognition
- The dataset.py script will basically capture the images from your webcam stream and store in a folder. It is also responsible for making the necessary changes in teh sqlite3 dataset.
- The trainer.py script will basically train our module on the basis of the dataset previously formed by making use of the LBPH algorithm.
- The detector.py will finally make use of the getProfile function defined in it to get all the details of the person based on ID. This in turn will be used for displaying the name and gender of the person on the bounding box appearing.
