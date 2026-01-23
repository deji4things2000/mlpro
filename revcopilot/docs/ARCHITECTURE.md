# RevCopilot Architecture

## Overview

RevCopilot is designed as a microservices architecture that integrates various components to provide an AI-powered reverse engineering assistant. The architecture consists of three main services: the backend, the frontend, and the Ghidra service. Each service is containerized using Docker and orchestrated with Docker Compose.

## Components

### 1. Backend Service

- **Technology Stack**: Python, FastAPI, Uvicorn
- **Responsibilities**:
  - Handle API requests and responses.
  - Perform binary analysis using the `angr` library.
  - Integrate AI functionalities for code explanation and analysis.
  
- **Key Files**:
  - `main.py`: Entry point for the FastAPI application.
  - `local_solver.py`: Implements local analysis algorithms.
  - `ai_module.py`: Manages AI-related tasks.
  - `api/endpoints.py`: Defines API endpoints for client interactions.
  - `api/websockets.py`: Manages real-time communication via WebSockets.

### 2. Frontend Service

- **Technology Stack**: JavaScript, React, Next.js, Tailwind CSS
- **Responsibilities**:
  - Provide a user interface for uploading binaries and viewing results.
  - Display analysis results and AI explanations.
  - Facilitate user interactions through various components.

- **Key Files**:
  - `package.json`: Lists frontend dependencies.
  - `src/pages/index.tsx`: Main landing page.
  - `src/components/`: Contains reusable UI components like file upload, code viewer, and results panel.

### 3. Ghidra Service

- **Technology Stack**: Python
- **Responsibilities**:
  - Provide a headless Ghidra environment for binary decompilation.
  - Expose a REST API for decompilation requests from the backend.

- **Key Files**:
  - `server.py`: Implements the server logic for handling decompilation requests.
  - `requirements.txt`: Lists dependencies specific to the Ghidra service.

## Communication

- The frontend communicates with the backend via REST API calls.
- The backend interacts with the Ghidra service to perform decompilation tasks.
- WebSocket connections are used for real-time updates and notifications between the backend and frontend.

## Deployment

- The entire application is containerized using Docker, allowing for easy deployment and scaling.
- Docker Compose is used to manage the multi-container setup, ensuring that all services can be started and stopped together.

## Conclusion

RevCopilot leverages modern web technologies and AI capabilities to provide a comprehensive solution for reverse engineering tasks. The modular architecture allows for easy maintenance and scalability as new features are added.