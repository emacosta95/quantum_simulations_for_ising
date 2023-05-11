for i in {1}
do
	nohup python parallel_transverse_ising.py --pbc --device=cpu --seed=174 --l=20 --n_dataset=1000000 --train    --nbatch=1000 --file_name=081222/test_unet_periodic --h_max=1.8  > output/mkdataset_1nn_3.txt &
done