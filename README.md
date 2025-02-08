# AI Prompt Testing Framework

A lightweight framework for testing and comparing AI model responses to the same prompt.

## Project Structure

- **prompt_testing.py**: Main script for running the prompt tester
- **prompt.txt**: The prompt to be tested
- **configs/<model_name>.json**: Configuration for the models to be used
- **.env**: Environment variables for API keys
- **results/<model_name>.txt**: Output of the model

## Usage

1. Clone the repository
2. Install the required dependencies
3. Configure the models in `configs/<model_name>.json`
4. Run the script with `python prompt_testing.py`

## Dependencies

- `python-dotenv`
- `openai`
- `gemini`
- `requests`
- `json`
