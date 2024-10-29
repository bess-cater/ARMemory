# Use the NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:22.03-py3
# Set up work directory
WORKDIR /memory
# Expose the port for Jupyter (if using Jupyter Notebook inside)
EXPOSE 9999
# Entry point to keep container running in interactive mode
CMD ["bash"]