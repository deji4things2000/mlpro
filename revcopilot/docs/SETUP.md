# RevCopilot Setup Instructions

## Prerequisites

Before you begin, ensure you have the following installed on your machine:

- **Docker**: [Install Docker](https://docs.docker.com/get-docker/)
- **Docker Compose**: [Install Docker Compose](https://docs.docker.com/compose/install/)
- **Python 3.11**: [Install Python](https://www.python.org/downloads/)
- **Node.js**: [Install Node.js](https://nodejs.org/)

## Clone the Repository

Start by cloning the repository to your local machine:

```bash
git clone https://github.com/yourusername/revcopilot.git
cd revcopilot
```

## Environment Setup

1. **Copy the Environment Variables Template**:

   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file** to include your OpenAI API key and any other necessary environment variables.

## Backend Setup

1. **Navigate to the backend directory**:

   ```bash
   cd backend
   ```

2. **Install Backend Dependencies**:

   You can create a virtual environment and install the dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Frontend Setup

1. **Navigate to the frontend directory**:

   ```bash
   cd ../frontend
   ```

2. **Install Frontend Dependencies**:

   ```bash
   npm install
   ```

## Ghidra Service Setup

1. **Navigate to the Ghidra service directory**:

   ```bash
   cd ../ghidra_service
   ```

2. **Install Ghidra Service Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Navigate back to the root directory**:

   ```bash
   cd ..
   ```

2. **Start the application using Docker Compose**:

   ```bash
   docker-compose up
   ```

3. **Access the Application**:

   Open your web browser and go to `http://localhost:3000` to access the frontend.

## Additional Notes

- Ensure that Docker is running before executing the `docker-compose up` command.
- For any issues, refer to the documentation in the `docs` directory or check the GitHub repository for updates.