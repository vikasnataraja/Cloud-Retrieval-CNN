echo "Using anaconda to install packages. This might take a few minutes ..."

# echo "============== Installing numpy =============="
# conda install -c anaconda numpy=1.18.1 -y
echo "============== Installing tensorflow =============="
conda install -c conda-forge tensorflow=2.0.0 -y
echo "============== Installing keras =============="
conda install -c conda-forge keras=2.3.1 -y
echo "============== Installing scikit-learn =============="
conda install -c anaconda scikit-learn=0.22.1 -y
echo "============== Installing openCV =============="
conda install -c conda-forge opencv=4.1.0 -y
# echo "============== Installing albumentations =============="
# conda install -c conda-forge albumentations=0.4.3 -y
echo "============== Installing matplotlib =============="
conda install -c conda-forge matplotlib=3.1.3 -y
echo "============== Installing h5py =============="
conda install -c anaconda h5py=2.10.0 -y

echo "Finished! Use conda list to show packages installed"
