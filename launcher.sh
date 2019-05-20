#!sh

python evaluation/retrain.py \
  --image_dir=/home/ruben/Repositories/WSI-analysis/test_level0
  --how_many_training_steps=5000
  --output_graph=
  --print_misclassified_test_images=
  
  #--tfhub_module