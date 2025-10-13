git clone https://github.com/SugarBlend/-DeployAndServe.git
cd "-DeployAndServe"
git checkout kandinsky_2
poetry build
pip install dist/deploy2serve-0.3.0-py3-none-any.whl
pip install -r requirements.txt
