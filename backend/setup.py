import subprocess
import sys
import os


def install_requirements():
    """Install required Python packages."""
    requirements = [
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "python-multipart==0.0.6",
        "python-docx==0.8.11",
        "PyPDF2==3.0.1",
        "pydantic==2.5.0",
        "python-dotenv==1.0.0",
        "google-cloud-aiplatform==1.38.0",
        "google-cloud-language==2.12.0",
        "spacy==3.7.2",
        "uuid==1.30"
    ]

    for requirement in requirements:
        subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])

    # Download spaCy model
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])


def setup_environment():
    """Create .env file template."""
    env_template = """# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_APPLICATION_CREDENTIALS=path/to/your/service-account.json
VERTEX_AI_LOCATION=us-central1
VERTEX_AI_MODEL=text-bison@002

# Optional: Custom settings
LOG_LEVEL=INFO
MAX_DOCUMENT_SIZE_MB=10
"""

    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write(env_template)
        print("Created .env file template. Please update with your Google Cloud credentials.")


if __name__ == "__main__":
    print("Setting up Legal lens...")
    install_requirements()
    setup_environment()
    print("Setup complete! Please configure your .env file with Google Cloud credentials.")