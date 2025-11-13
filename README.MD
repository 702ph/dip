## Setting-UP Virtual Environment
```
uv venv --python python3.12
source .venv/bin/activate
uv sync
```

```
deactivate
```

## How to execute
when you are in /dip
```
source .venv/bin/activate
uv sync
cd dip2_py
```

when you are in /dip/dip2_py
```
python main.py testimage_small.jpg
python unit_test.py
```