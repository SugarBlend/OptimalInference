uv venv
uv sync --all-extras
git clone https://github.com/SugarBlend/-DeployAndServe.git
cd "-DeployAndServe"
poetry build
uv pip install dist/deploy2serve-0.3.0-py3-none-any.whl --no-deps
