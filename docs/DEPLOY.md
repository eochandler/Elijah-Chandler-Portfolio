# Deployment guide

This project is designed for a free-tier deployment path using Docker-based hosting.

## Hugging Face Spaces

1. Create a new Docker Space on Hugging Face.
2. Upload this repository or connect the GitHub repo.
3. Set `app_port` to `8000` or the container's exposed port.
4. Ensure the Dockerfile exposes the service port and build context is correct.
5. Deploy and test with the sample payload under `examples/sample_request.json`.

## Render

1. Create a new Web Service using the Dockerfile.
2. Set the environment variable `PORT` to `8000`.
3. Push the project and confirm `/health` works.

## GHCR local run

```bash
docker run --rm -p 8000:8000 ghcr.io/<owner>/turbofan-rul:latest
```

Note: free-tier terms change often, so check each provider's current page before publishing. Some providers require a credit card or have free limits.
