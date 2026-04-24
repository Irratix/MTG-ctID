# Magic: the Gathering creature type identifier
This project is now live on here https://mengersponge.website/jsprojects/mtgctid/

This is a deep learning project to build a model that attempts to guess creature types of any image.
`get_data.py` collects creature card data from Scryfall such that there is at least 100 cards of each creature type in the dataset and all remaining cards are suitable for play in Vintage. The trainingg process works by transfer learning on of EfficientNet_b2. The current best model has an IoU-score of 0.251.
