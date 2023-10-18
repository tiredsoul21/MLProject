# Use Python Alpine base image
FROM python:3.8-alpine

# Set the working directory
WORKDIR /app

RUN apk add --no-cache --virtual .build-deps build-base linux-headers
RUN apk add --no-cache libstdc++
RUN pip install numpy && \
    apk del .build-deps
RUN apk add --no-cache --virtual .build-deps  build-base linux-headers && \
    pip install matplotlib && \
    apk del .build-deps
RUN apk add --no-cache --virtual .build-deps build-base linux-headers&& \
    pip install pandas && \
    apk del .build-deps
RUN apk add --no-cache --virtual .build-deps build-base linux-headers&& \
    pip install seaborn && \
    apk del .build-deps

# Run your script
CMD ["python", "src/app.py"]