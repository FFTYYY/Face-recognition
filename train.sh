#train resnet
python -W ignore train.py --model=resnet --optim=mysgd --n=9 --batch_size=128 --gpus=5 \
    --lr=0.1 --n_epochs=200 --drop_p=0.0 --seed=2333 --log_file=my_resnet.txt

#train multi-dim transformer
python -W ignore train.py --model=transformer --optim=my_adam --batch_size=64 --gpus=0,1,2,3,4,5,6,7  \
--n_epochs=30 --model_save="trained_trans.pkl" --d_model=512 --h=8 --n_persis=512 --drop_p=0.0

