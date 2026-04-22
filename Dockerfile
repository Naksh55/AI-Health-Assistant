# Use a slim Python image for faster builds
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Streamlit uses port 8501 by default, but Hugging Face Spaces requires 7860
EXPOSE 7860

# Command to run your app
CMD ["streamlit", "run", "app.py", "--server.port", "7860", "--server.address", "0.0.0.0"]