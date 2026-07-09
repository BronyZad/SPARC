# SPARC Codebase

Core implementation and scripts for the SPARC paper. 

## Files
- `server_*.py` & `sparc_core_transport.py`: main backend and transport logic
- `benchmark_*.py`: scripts to run evaluations (LongBench, LEval, etc.)
- `*_judge.py`: auto-scoring scripts
- `requirements.txt`: sparc environment dependencies

## How to run

1. Install dependencies:
`pip install -r requirements.txt`

2. Run the backend:
`python server_decode.py`

3. Execute your chosen benchmark script (e.g., `python benchmark_longbench.py`).
