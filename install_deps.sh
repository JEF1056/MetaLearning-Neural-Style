cd /home/
sudo apt install libopenblas-dev 
sudo apt install libopencv-dev 
sudo apt install libboost-dev 
sudo apt install libboost-system-dev 
sudo apt install libboost-filesystem-dev 
sudo apt install libboost-regex-dev 
sudo apt install libboost-thread-dev  
sudo apt install libboost-python-dev 
sudo apt install libprotobuf-dev 
sudo apt install protobuf-compiler 
sudo apt install libgflags-dev 
sudo apt install libgoogle-glog-dev 
sudo apt install python-numpy 
sudo apt install python-opencv 
sudo apt install libmpich-dev
sudo apt-get install git
sudo apt-get update 
sudo apt-get install -y $buildDeps 
sudo apt-get install cuda=10.1.168
wget https://developer.download.nvidia.com/compute/machine-learning/cudnn/secure/v7.5.1/prod/10.1_20190418/Ubuntu18_04-x64/libcudnn7_7.5.1.10-1%2Bcuda10.1_amd64.deb?jUdcv7bzHOnmHCjGvhxuD2THmnKmCmLKrTcZAoZrLFoF5P46z0xY2z0rFS186MpPcSEs8UR6MgLKyRYXsbSXV8KQpMYM4Us80xPcGfvnfNNVQ4CIfUPRepP8Cdho2_amLMVZlceq0JNpvole5IWF1e7RyO-f0m2oEJLx4-94aLUpqcvdErB7dtnE-e2tGWVPqrE9pkjyhL_-U2Ag1Su5snkS4jP7xPCWvfyso2Mr9TZ6ksAaPIb71Q
sudo dpkg -i libcudnn7_7.5.1.10-1+cuda10.1_amd64.deb
wget http://github.com/NVIDIA/nccl/archive/master.zip 
unzip master.zip 
cd nccl-master 
make
make install
cd /home/
rm master.zip
rm -rf nccl
rm -r nccl-master
git clone https://github.com/JEF1056/MetaLearning-Neural-Style.git
mv MetaLearning-Neural-Style styletransfer
rm -r styletransfer/build
cd styletransfer
make 
make pycaffe
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1DxTIo8wVLTjEPLrin00ifgOSbr5HKBwK' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1DxTIo8wVLTjEPLrin00ifgOSbr5HKBwK" -O python/train_8.caffemodel && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1jI07ubQBsudvcDV8hNXM5L8n_Oog0dsc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1jI07ubQBsudvcDV8hNXM5L8n_Oog0dsc" -O python/train_32.caffemodel && rm -rf /tmp/cookies.txt
echo "Install complete (as long as all the errors are #WARNING)! Run the demo.py file in /home/styletransfer/python/"
cd