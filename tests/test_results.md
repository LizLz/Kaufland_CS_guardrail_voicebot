# Test Results

## How to reproduce

```bash
pytest tests/test_cases.py -v                    # full set (requires GROQ_API_KEY)
pytest tests/test_cases.py -m "not integration"  # subset, no API key needed
```

## Results by module
![Test Result](/image/test.png)


