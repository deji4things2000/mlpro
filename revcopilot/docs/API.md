# API Documentation for RevCopilot

## Overview

The RevCopilot API provides endpoints for interacting with the backend services of the RevCopilot project. This API allows users to upload binaries for analysis, retrieve results, and interact with the AI-assisted features of the application.

## Base URL

The base URL for the API is:

```
http://localhost:8000/api
```

## Endpoints

### 1. Upload Binary

- **POST** `/upload`
  
  Uploads a binary file for analysis.

  **Request Body:**
  - `file`: The binary file to be uploaded.

  **Response:**
  - `201 Created`: Returns the analysis ID.
  - `400 Bad Request`: If the file is not valid.

### 2. Get Analysis Result

- **GET** `/results/{id}`
  
  Retrieves the analysis results for a given ID.

  **Path Parameters:**
  - `id`: The ID of the analysis.

  **Response:**
  - `200 OK`: Returns the analysis results.
  - `404 Not Found`: If the analysis ID does not exist.

### 3. AI Explanation

- **POST** `/explain`
  
  Requests an AI-generated explanation for a specific code segment.

  **Request Body:**
  - `code`: The code segment to be explained.

  **Response:**
  - `200 OK`: Returns the AI-generated explanation.
  - `400 Bad Request`: If the code is not valid.

### 4. WebSocket Connection

- **WebSocket** `/ws`
  
  Establishes a WebSocket connection for real-time updates.

  **Response:**
  - Sends updates regarding the analysis progress and results.

## Error Handling

All API responses include a standard error format:

```json
{
  "error": {
    "code": "error_code",
    "message": "Error message describing the issue."
  }
}
```

## Authentication

The API may require authentication via API keys or tokens. Ensure to include the necessary credentials in the request headers.

## Rate Limiting

To ensure fair usage, the API enforces rate limits. Exceeding the limit will result in a `429 Too Many Requests` response.

## Conclusion

This API documentation provides a comprehensive overview of the available endpoints and their usage. For further details, refer to the source code and implementation in the backend service.