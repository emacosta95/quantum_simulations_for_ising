for i in {48..64}
do
	nohup python parallel_transverse_ising.py --pbc --device=cpu --seed=$i --l=$i --n_dataset=60000 --nbatch=100 --file_name=240123/unet_periodic > output/mkdataset_1nn_1.txt &
done