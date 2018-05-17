module load hdf5 python_gpu/3.6.4 cuda/9.0.176 cudnn/7.0
pip install --user -r requirements.txt
pip install --user tensorflow_gpu

bsub -W 4:00 -R "rusage[mem=16000, ngpus_excl_p=1]" "python main.py --nthreads 6 &> $HOME/error.log"
bsub -W 4:00 -n 12 -R "rusage[mem=5000]" "./fasttext sent2vec -input final.txt -output model2 -minCount 1 -dim 500 -thread 12 -epoch 30 -wordNgrams 4
 &> $HOME/error.log"
