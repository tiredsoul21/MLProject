# MLProject
cd MLProject
docker build -t python-image .
docker run -v ./src:/app/src -v ./data:/app/data python-image

sudo apt install python3-matplotlib
sudo apt install python3-numpy
sudo apt install python3-opencv
sudo apt install python3-pandas
sudo apt install python3-seaborn