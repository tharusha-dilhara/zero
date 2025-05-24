# Agentic AI System for Educational Platform

This project implements a multi-agent AI system tailored for a software engineering education platform, utilizing Flask, Groq, Qdrant, Chainlit, and LangGraph.

## System Architecture

The system contains four specialized agents:

1. **Sales Agent**: Promotes and sells educational courses
2. **Help Agent**: Provides prompt support to students
3. **Manage Agent**: Oversees operations and ensures system efficiency
4. **Sales & Marketing Agent**: Analyzes data to enhance marketing efforts

## Technologies Used

- **Flask**: RESTful API backend
- **Qdrant**: Vector database for long-term memory storage
- **Groq**: LLM provider (using Llama3 models)
- **Chainlit**: Interactive chat interfaces
- **LangGraph**: Agent communication framework

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- Groq API key

### Local Development

1. Clone the repository
2. Copy `.env.example` to `.env` and fill in your API keys
3. Run the Docker Compose setup:

```bash
docker-compose up -d
```

4. Access the Chainlit interface at http://localhost:8000
5. Access the API at http://localhost:5000

### Environment Variables

- `GROQ_API_KEY`: Your Groq API key
- `GROQ_MODEL`: The model to use (default: llama3-70b-8192)
- `QDRANT_URL`: URL for the Qdrant vector database

## API Endpoints

- `/chat`: General chat endpoint that routes to the appropriate agent
- `/agent/<agent_id>`: Direct communication with a specific agent

## Agent Communication

Agents communicate using LangGraph's workflow system and the Agent2Agent (A2A) Protocol, allowing them to share information and coordinate responses.

## Deployment

For production deployment:

1. Build the Docker images
2. Deploy to Google Cloud Run or similar service
3. Configure environment variables in the cloud environment

## License

MIT
```
