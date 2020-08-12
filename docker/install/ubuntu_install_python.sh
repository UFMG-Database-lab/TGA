# install python and pip, don't modify this, modify install_python_package.sh
apt-get update -y
apt-get install -y python-dev python3-dev software-properties-common

add-apt-repository ppa:deadsnakes/ppa -y
apt update -y

apt install python3.8 -y
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

apt install --reinstall python3-pip -y
apt install --reinstall python3.8-distutils python3-apt -y
cd /tmp && wget https://bootstrap.pypa.io/get-pip.py
#python2 get-pip.py
python3 get-pip.py


#apt-get update
#apt install python3.8
#update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# install pip
#cd /tmp && wget https://bootstrap.pypa.io/get-pip.py
#python2 get-pip.py
#python3 get-pip.py
