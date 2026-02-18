# flask_ml_api

> **WARNING:** This repository may be unstable or non-functional. Use at your own risk.

`flask_ml_api` is a Flask service that loads pre-trained CatBoost models and exposes a prediction endpoint.

## Run With Docker

```bash
docker build -t flask-ml-api .
docker run --rm -p 5000:5000 flask-ml-api
```

## API

- `GET /health` returns service status.
- `POST /predict/<model_name>`
- `Content-Type: application/json`
- Request body: JSON array of numeric features (single sample) or array of samples.

Response example:

```json
{
  "model": "arrhythmia",
  "count": 1,
  "predictions": [1]
}
```

Example:

```bash
curl --location "http://127.0.0.1:5000/predict/arrhythmia" \
  --header "Content-Type: application/json" \
  --data "[51.0,0.0,170.0,82.0,90.0,155.0,382.0,216.0,88.0,9.0]"
```

Error responses are always JSON:

```json
{"error":"..."}
```

## License

See `LICENSE`.
