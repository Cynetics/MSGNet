python train_vqvae.py --batch 32 --lr 1e-3 --iterations 50000
python extract_code.py 
python train_pixelsnail.py --batch 32
