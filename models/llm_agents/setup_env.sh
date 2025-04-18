#!/bin/bash

echo "Setting up environment for Hugging Face API access"
echo "=================================================="
echo ""
echo "You'll need to get your Hugging Face API token from: https://huggingface.co/settings/tokens"
echo "If you don't have one, please visit the link above and create one."
echo ""
echo "Enter your Hugging Face API token:"
read -s token

if [ -z "$token" ]; then
  echo "No token provided. Exiting."
  exit 1
fi

export HUGGINGFACEHUB_API_TOKEN="$token"
echo "Token set as environment variable HUGGINGFACEHUB_API_TOKEN."
echo ""
echo "To make this permanent, add the following line to your ~/.bashrc file:"
echo "export HUGGINGFACEHUB_API_TOKEN=\"your_token_here\""
echo ""
echo "You can now run your script with:"
echo "python main.py" 