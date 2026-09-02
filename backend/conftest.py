import os

# Isolate test runs from developer .env files to preserve hermetic test assertions
os.environ["REVORA_TESTING"] = "1"
