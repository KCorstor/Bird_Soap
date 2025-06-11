#!/bin/bash

# Create a temporary directory for the layer
mkdir -p python/lib/python3.9/site-packages

# Install requirements into the layer directory
pip install -r requirements.txt -t python/lib/python3.9/site-packages

# Create the ZIP file
zip -r lambda_layer.zip python/

# Clean up
rm -rf python/

echo "Lambda layer created as lambda_layer.zip" 