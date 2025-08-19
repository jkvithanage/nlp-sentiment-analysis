### Setup
1. Make sure python (3.12 recommended) is installed `python3 --version`
2. Clone the repo and `cd nlp-sentiment-analysis`
3. Create a virtual environment `python3 -m venv venv`
4. Activate venv `source venv/bin/activate`
5. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
6. Download dataset from https://www.cs.jhu.edu/%7Emdredze/datasets/sentiment/index2.html
7. Unzip, rename it to `data` and move to project directory.
8. Run `python train.py` to train both models
9. Run `python launch_ui.py` to launch the Gradio UI

Additional commands:
- Run `python data_loader.py` to export raw data to a csv file
- Run `python preprocess.py` to preprocess the data
