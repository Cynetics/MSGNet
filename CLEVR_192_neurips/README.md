First execute setup.sh
to run the VQVAE run train_vqvae.py
to run the pixelsnail run train_pixelsnail.py
to run the layoutpixelsnaiil run train_layoutpixelsnail.py
The models are defined in vqvae.py and pixelsnail.py respectively.

Note that this code can be adapted for the other datasets. The main difference is that the self-attention module is placed in features spaces smaller than 64 in width or height. 
