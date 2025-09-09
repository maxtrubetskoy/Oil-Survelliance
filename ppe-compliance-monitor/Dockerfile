# -----------------------------------------------------------------------------
# Dockerfile for the Real-time PPE Compliance Monitor
# -----------------------------------------------------------------------------

# Stage 1: Use an official Python runtime as a parent image.
# Using a "slim" variant is a good practice to keep the image size down.
FROM python:3.10-slim

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files.
# PYTHONUNBUFFERED: Ensures that print statements are sent straight to the
#                   terminal without being buffered, which is useful for logging.
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory inside the container. All subsequent commands
# will be run from this directory.
WORKDIR /app

# Copy the requirements file first. This allows Docker to cache the dependency
# installation layer. This layer will only be rebuilt if requirements.txt changes.
COPY ./ppe-compliance-monitor/requirements.txt .

# Install the Python dependencies.
# --no-cache-dir: Disables the pip cache, which reduces the image size.
# -r requirements.txt: Installs all packages listed in the file.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container's working directory.
# This includes `main.py`, the `src/`, `utils/`, `config/`, and `models/` directories.
COPY ./ppe-compliance-monitor/ .

# Specify the command to run when the container starts.
# This will execute `python main.py` inside the /app directory.
# The user will still need to ensure that the config.yaml points to a valid
# video source accessible from within the container and that the model file
# is present in the models directory before building the image.
ENTRYPOINT ["python", "main.py"]
