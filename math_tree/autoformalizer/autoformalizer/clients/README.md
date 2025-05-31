### Set up API Key

Please contact Jia for the API key

```bash
export MOONSHOT_LEAN_API_KEY=xxxxx
```

### Test connection

```bash
python autoformalizer/clients/lean4_client.py
```

You should see something like
```
2024-12-10 06:12:30.103 | INFO     | __main__:test_connection:97 - Test connection: {'error': None, 'response': {'env': 0}}
2024-12-10 06:12:30.104 | INFO     | __main__:<module>:204 - Connection successful!
```

### Unit testing

If you are modifying the logic, please test

```base
python -m unittest autoformalizer/clients/tests/test_lean.py
```

### Multi thread example

In practice, we run the client with multi process to speed up. You can adapt from this example.

```bash
python scripts/jia/lean4_client_example.py
```

This should run in one or two minutes, with output:
```
2024-12-10 06:42:13.466 | INFO     | __main__:main:42 - Total proofs to verify: 2000
2024-12-10 06:42:13.467 | INFO     | __main__:main:43 - No error proofs: 561
```

**IMPORTANT: The maximum number of parallel processes is approximately 1000. If you are certain that no one else is using this server, you can set num_proc to 1000 for optimal speed. Tools to check server load and activity will be available in the near future.**