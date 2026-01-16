python predict.py --input datasets/data/000840.png \
                --dataset cityscapes \
                --model deeplabv3plus_mobilenet \
                --ckpt checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth \
                --save_val_results_to test_results

# python predict.py --input datasets/data/0030.jpg \
#                 --dataset voc \
#                 --model deeplabv3plus_mobilenet \
#                 --ckpt checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth \
#                 --save_val_results_to test_results

# python predict.py --input samples/114_image.png \
#                 --dataset voc \
#                 --model deeplabv3plus_mobilenet \
#                 --ckpt checkpoints/best_deeplabv3plus_mobilenet_voc_os16.pth \
#                 --save_val_results_to test_results