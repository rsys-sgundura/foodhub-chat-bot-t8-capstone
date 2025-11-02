# ----------------------------
# STEP 1: Choose Base Image
# ----------------------------
# Start from an official lightweight Python 3.9 image
# This ensures we have Python and pip preinstalled.
FROM python:3.9

# ----------------------------
# STEP 2: Set Working Directory
# ----------------------------
# Create and set the working directory inside the container to /app
WORKDIR /app

# ----------------------------
# STEP 3: Copy Application Code
# ----------------------------
# Copy all files from the project directory on the host
# into the container’s /app directory
COPY . .

# ----------------------------
# STEP 4: Install Dependencies
# ----------------------------
# Install all Python dependencies required by the app,
# listed in the requirements.txt file
RUN pip3 install -r requirements.txt

# ----------------------------
# STEP 5: Create Non-root User
# ----------------------------
# Create a new user with UID 1000 to avoid running as root
RUN useradd -m -u 1000 user
USER user

# ----------------------------
# STEP 6: Set Environment Variables
# ----------------------------
# Define HOME path for the new user and update PATH
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# ----------------------------
# STEP 7: Adjust Working Directory
# ----------------------------
# Move working directory to /home/user/app
WORKDIR $HOME/app

# Copy the code again ensuring the correct ownership
COPY --chown=user . $HOME/app

# ----------------------------
# STEP 8: Define Startup Command
# ----------------------------
# Run the Streamlit app when the container starts
# - Binds to port 8501
# - Accessible on all network interfaces (0.0.0.0)
# - Disables XSRF protection to allow external access
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
