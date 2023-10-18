# MLProject
cd MLProject
docker build -t python-image .
docker run -v ./src:/app/src -v ./data:/app/data python-image