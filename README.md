# Energy QA Model Fine-Tuning

This project aims to fine-tune the Energy QA model for improved performance on energy-related questions.

## Project Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/hmohamed080/energy-qa-finetuning.git
   cd energy-qa-finetuning
   ```

2. **Install dependencies**:
   Make sure you have Python 3.8 or later installed. You can install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**:
   Download the necessary datasets and place them in the appropriate directory. Update the paths in the configuration file as needed.

## Usage Instructions

To fine-tune the model, run the following command:
```bash
python train.py --config config.yaml
```

Make sure to adjust the parameters in `config.yaml` to fit your dataset and training requirements.

## Additional Information

For further details, check the project wiki or the documentation folder.