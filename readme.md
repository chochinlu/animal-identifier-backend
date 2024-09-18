# Animal Identifier Backend

This project is a backend service for identifying animals in images and providing relevant information about them.

## Features

### Image Analysis
- Analyzes uploaded images to identify animals
- Uses AI-powered image recognition
- Handles cases where the image doesn't contain a recognizable animal

### Animal Information Retrieval
- Fetches detailed information about identified animals
- Provides descriptions, Wikipedia URLs, and danger assessments
- Handles cases where information is not available

### API Endpoints
- `/recognize-animal`: Accepts image uploads and returns animal identification results

## Setup

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set the `OPENAI_API_KEY` environment variable

Linux/MacOS:
```
export OPENAI_API_KEY=<your key>
```

Windows:
```
set OPENAI_API_KEY=<your key>
```

## Usage

1. Start the server in dev mode:

   ```
   fastapi dev main.py
   ```
   or start the server in prod mode:
   ```
   fastapi run main.py
   ```
   
2. Access the API at `http://localhost:8000`


## Dependencies

Major dependencies include:
- FastAPI: Web framework
- Transformers: AI model for image recognition
- LlamaIndex: For enhanced information retrieval

For a full list of dependencies, see `requirements.txt`.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here]
