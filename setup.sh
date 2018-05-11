module load python_gpu/3.6.4 cuda/9.0.176 cudnn/7.0
pip install --user -r requirements.txt
pip install --user tensorflow_gpu

bsub -W 4:00 -R "rusage[mem=6000, ngpus_excl_p=1]" "python main.py --nthreads 6 &> $HOME/error.log"
