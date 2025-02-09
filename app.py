import dash
from dash import html, dcc, callback, Output, Input, State
import os
import json
from pathlib import Path
import asyncio
from prompt_testing import OpenAIModel, GeminiModel, DeepseekModel, MistralModel
from dotenv import load_dotenv

# Initialize the Dash app with Tailwind CSS
app = dash.Dash(
    __name__,
    external_stylesheets=[
        'https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css'
    ]
)

# Load environment variables
load_dotenv()

# Add model options at the top of the file
MODEL_OPTIONS = {
    'openai': [
        {'label': 'GPT-3.5 Turbo', 'value': 'gpt-3.5-turbo'},
        {'label': 'GPT-4', 'value': 'gpt-4'},
        {'label': 'GPT-4 Turbo', 'value': 'gpt-4-turbo-preview'},
        {'label': 'GPT-4o Latest', 'value': 'gpt-4o-latest'},
        {'label': 'o1-preview', 'value': 'o1-preview'},
        {'label': 'gpt-4o-mini', 'value': 'gpt-4o-mini'},
        {'label': 'gpt-4o-2024-08-06', 'value': 'gpt-4o-2024-08-06'}
    ],
    'gemini': [
        {'label': 'Gemini Pro', 'value': 'gemini-pro'},
        {'label': 'Gemini 1.5 Pro', 'value': 'gemini-1.5-pro'},
        {'label': 'Gemini 1.5 Flash', 'value': 'gemini-1.5-flash'},
        {'label': 'Gemini 1.5 Pro (latest)', 'value': 'gemini-1.5-pro-latest'}
    ],
    'deepseek': [
        {'label': 'Deepseek Chat', 'value': 'deepseek-chat'},
        {'label': 'Deepseek Coder', 'value': 'deepseek-coder'},
        {'label': 'Deepseek Chat v3', 'value': 'deepseek-chat-v3'}
    ],
    'mistral': [
        {'label': 'Mistral Tiny', 'value': 'mistral-tiny'},
        {'label': 'Mistral Small', 'value': 'mistral-small'},
        {'label': 'Mistral Medium', 'value': 'mistral-medium'},
        {'label': 'Mistral Large', 'value': 'mistral-large'}
    ]
}

# Add custom button styles at the top of the file after MODEL_OPTIONS
BUTTON_STYLES = {
    'primary': 'bg-gradient-to-r from-blue-600 to-blue-500 text-white font-medium py-3 px-6 rounded-lg ' +
              'hover:from-blue-700 hover:to-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 ' +
              'focus:ring-offset-2 transition-all duration-200 ease-in-out shadow-md hover:shadow-lg ' +
              'active:shadow-inner active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed',
    'secondary': 'bg-white text-gray-700 font-medium py-2 px-4 rounded-lg border border-gray-300 ' +
                'hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 ' +
                'transition-all duration-200 ease-in-out shadow-sm hover:shadow-md active:bg-gray-100',
    'danger': 'bg-gradient-to-r from-red-600 to-red-500 text-white font-medium py-2 px-4 rounded-lg ' +
              'hover:from-red-700 hover:to-red-600 focus:outline-none focus:ring-2 focus:ring-red-500 ' +
              'focus:ring-offset-2 transition-all duration-200 ease-in-out shadow-md hover:shadow-lg ' +
              'active:shadow-inner active:scale-[0.98]'
}

# Get list of available models from results directory
def get_available_models():
    results_dir = Path("results")
    models = []
    if results_dir.exists():
        for file in results_dir.glob("*.txt"):
            models.append(file.stem)
    return models

# Read model output
def read_model_output(model_name):
    try:
        with open(f"results/{model_name}.txt", 'r') as f:
            return f.read()
    except FileNotFoundError:
        return "No output available for this model"

# Layout
app.layout = html.Div([
    # Navigation Bar
    html.Nav(
        html.Div([
            html.H1("AI Model Output Comparison", 
                    className='text-3xl font-bold text-white'),
            html.Div([
                html.Span("Status: ", className='text-gray-300'),
                html.Span(id="loading-output", className='text-white font-medium')
            ])
        ], className='max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center'),
        className='bg-gray-800 shadow-md mb-8'
    ),
    
    # Main Content
    html.Div([
        # Left Panel - Model Parameters
        html.Div([
            html.H2("Model Parameters", className='text-2xl font-bold text-gray-800 mb-6'),
            
            # Loading Spinner
            dcc.Loading(
                id="loading-spinner",
                type="circle",
                children=[
                    html.Div(id="loading-output-container", className='text-center text-sm text-gray-600 mb-4')
                ],
                className='mb-4'
            ),
            
            # Tabs for different models
            dcc.Tabs([
                # OpenAI Tab
                dcc.Tab(
                    label='OpenAI',
                    children=[
                    html.Div([
                        html.Label("Model:", className='block text-sm font-medium text-gray-700 mt-4'),
                        dcc.Dropdown(
                            id='openai-model-dropdown',
                            options=MODEL_OPTIONS['openai'],
                            value='gpt-3.5-turbo',
                            className='mb-4'
                        ),
                        html.Label("Response Format:", className='block text-sm font-medium text-gray-700'),
                        dcc.Dropdown(
                            id='openai-response-format-dropdown',
                            options=[
                                {'label': 'Text', 'value': 'text'},
                                {'label': 'JSON', 'value': 'json_object'}
                            ],
                            value='text',
                            className='mb-4'
                        ),
                        html.Label("Temperature:", className='block text-sm font-medium text-gray-700'),
                        dcc.Slider(
                            id='openai-temperature-slider',
                            min=0, max=1, step=0.1, value=0.7,
                            marks={i/10: str(i/10) for i in range(11)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className='mb-6'
                        ),
                        html.Label("Max Tokens:", className='block text-sm font-medium text-gray-700'),
                        dcc.Slider(
                            id='openai-max-tokens-slider',
                            min=100, max=2000, step=100, value=1000,
                            marks={i: str(i) for i in range(0, 2001, 500)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className='mb-6'
                        ),
                        html.Label("Top P:", className='block text-sm font-medium text-gray-700'),
                        dcc.Slider(
                            id='openai-top-p-slider',
                            min=0, max=1, step=0.1, value=1.0,
                            marks={i/10: str(i/10) for i in range(11)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className='mb-6'
                        ),
                        html.Label("Frequency Penalty:", className='block text-sm font-medium text-gray-700'),
                        dcc.Slider(
                            id='openai-frequency-penalty-slider',
                            min=0, max=2, step=0.1, value=0.0,
                            marks={i/2: str(i/2) for i in range(5)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className='mb-6'
                        ),
                        html.Label("Presence Penalty:", className='block text-sm font-medium text-gray-700'),
                        dcc.Slider(
                            id='openai-presence-penalty-slider',
                            min=0, max=2, step=0.1, value=0.0,
                            marks={i/2: str(i/2) for i in range(5)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className='mb-4'
                        )
                    ], className='p-4')
                ],
                className='custom-tab',
                selected_className='custom-tab--selected',
                style={'fontSize': '12px', 'textAlign': 'center', 'padding': '8px 16px'}
                ),
                
                # Gemini Tab
                dcc.Tab(
                    label='Gemini',
                    children=[
                    html.Div([
                        html.Label("Model:", className='block text-sm font-medium text-gray-700 mt-4'),
                        dcc.Dropdown(
                            id='gemini-model-dropdown',
                            options=MODEL_OPTIONS['gemini'],
                            value='gemini-1.5-pro',
                            className='mb-4'
                        ),
                        html.Label("Temperature:", className='block text-sm font-medium text-gray-700'),
                        dcc.Slider(
                            id='gemini-temperature-slider',
                            min=0, max=1, step=0.1, value=0.7,
                            marks={i/10: str(i/10) for i in range(11)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className='mb-6'
                        ),
                        html.Label("Max Tokens:", className='block text-sm font-medium text-gray-700'),
                        dcc.Slider(
                            id='gemini-max-tokens-slider',
                            min=100, max=2000, step=100, value=1000,
                            marks={i: str(i) for i in range(0, 2001, 500)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className='mb-4'
                        )
                    ], className='p-4')
                ],
                className='custom-tab',
                selected_className='custom-tab--selected',
                style={'fontSize': '12px', 'textAlign': 'center', 'padding': '8px 16px'}
                ),
                
                # Deepseek Tab
                dcc.Tab(
                    label='Deepseek',
                    children=[
                    html.Div([
                        html.Label("Model:", className='block text-sm font-medium text-gray-700 mt-4'),
                        dcc.Dropdown(
                            id='deepseek-model-dropdown',
                            options=MODEL_OPTIONS['deepseek'],
                            value='deepseek-chat',
                            className='mb-4'
                        ),
                        html.Label("Temperature:", className='block text-sm font-medium text-gray-700'),
                        dcc.Slider(
                            id='deepseek-temperature-slider',
                            min=0, max=1, step=0.1, value=0.7,
                            marks={i/10: str(i/10) for i in range(11)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className='mb-6'
                        ),
                        html.Label("Max Tokens:", className='block text-sm font-medium text-gray-700'),
                        dcc.Slider(
                            id='deepseek-max-tokens-slider',
                            min=100, max=2000, step=100, value=1000,
                            marks={i: str(i) for i in range(0, 2001, 500)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className='mb-4'
                        )
                    ], className='p-4')
                ],
                className='custom-tab',
                selected_className='custom-tab--selected',
                style={'fontSize': '12px', 'textAlign': 'center', 'padding': '8px 16px'}
                ),
                
                # Mistral Tab
                dcc.Tab(
                    label='Mistral',
                    children=[
                    html.Div([
                        html.Label("Model:", className='block text-sm font-medium text-gray-700 mt-4'),
                        dcc.Dropdown(
                            id='mistral-model-dropdown',
                            options=MODEL_OPTIONS['mistral'],
                            value='mistral-medium',
                            className='mb-4'
                        ),
                        html.Label("Temperature:", className='block text-sm font-medium text-gray-700'),
                        dcc.Slider(
                            id='mistral-temperature-slider',
                            min=0, max=1, step=0.1, value=0.7,
                            marks={i/10: str(i/10) for i in range(11)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className='mb-6'
                        ),
                        html.Label("Max Tokens:", className='block text-sm font-medium text-gray-700'),
                        dcc.Slider(
                            id='mistral-max-tokens-slider',
                            min=100, max=2000, step=100, value=1000,
                            marks={i: str(i) for i in range(0, 2001, 500)},
                            tooltip={"placement": "bottom", "always_visible": True},
                            className='mb-4'
                        )
                    ], className='p-4')
                ],
                className='custom-tab',
                selected_className='custom-tab--selected',
                style={'fontSize': '12px', 'textAlign': 'center', 'padding': '8px 16px'}
                )
            ], className='custom-tabs-container', style={'height': '42px'})
        ], className='w-1/4 bg-white rounded-lg shadow-card p-6'),
        
        # Right Panel - Prompt Input and Comparison
        html.Div([
            # Prompt Input Section
            html.Div([
                html.H2("Input Prompt", className='text-2xl font-bold text-gray-800 mb-4'),
                dcc.Textarea(
                    id='prompt-input',
                    value='',
                    placeholder='Enter your prompt here...',
                    className='w-full h-32 p-4 border border-gray-300 rounded-lg focus:ring-primary-500 focus:border-primary-500 mb-4'
                ),
                html.Div([
                    html.Div(id='running-status', className='text-sm text-gray-600 mr-4'),
                    html.Div([
                        html.Button(
                            'Run Models', 
                            id='submit-button', 
                            n_clicks=0,
                            className=BUTTON_STYLES['primary']
                        ),
                        html.Button(
                            'Clear All', 
                            id='clear-button', 
                            n_clicks=0,
                            className=BUTTON_STYLES['secondary'] + ' ml-4'
                        )
                    ])
                ], className='flex justify-between items-center')
            ], className='bg-white rounded-lg shadow-card p-6 mb-6'),
            
            # Model Comparison Section
            html.Div([
                html.Div([
                    html.H2("Model Comparison", className='text-2xl font-bold text-gray-800'),
                    html.Div([
                        html.Button(
                            'Copy Diff', 
                            id='copy-diff-button',
                            className=BUTTON_STYLES['secondary'] + ' text-sm'
                        ),
                        html.Button(
                            'Export Results', 
                            id='export-button',
                            className=BUTTON_STYLES['secondary'] + ' text-sm ml-2'
                        ),
                    ], className='flex items-center space-x-2')
                ], className='flex justify-between items-center mb-4'),
                html.Div([
                    # Left Model
                    html.Div([
                        html.Div([
                            dcc.Dropdown(
                                id='model1-dropdown',
                                options=[{'label': model, 'value': model} for model in get_available_models()],
                                value=get_available_models()[0] if get_available_models() else None,
                                className='w-2/3'
                            ),
                            html.Button(
                                'Copy', 
                                id='copy-model1-button',
                                className=BUTTON_STYLES['secondary'] + ' text-sm ml-2 py-1 w-1/4'
                            )
                        ], className='flex items-center justify-between mb-2 gap-2'),
                        html.Div(
                            id='model1-output',
                            className='h-96 overflow-auto p-4 bg-gray-50 border border-gray-200 rounded-lg font-mono text-sm'
                        )
                    ], className='w-1/2 pr-2'),
                    
                    # Right Model
                    html.Div([
                        html.Div([
                            dcc.Dropdown(
                                id='model2-dropdown',
                                options=[{'label': model, 'value': model} for model in get_available_models()],
                                value=get_available_models()[1] if len(get_available_models()) > 1 else None,
                                className='w-2/3'
                            ),
                            html.Button(
                                'Copy', 
                                id='copy-model2-button',
                                className=BUTTON_STYLES['secondary'] + ' text-sm ml-2 py-1 w-1/4'
                            )
                        ], className='flex items-center justify-between mb-2 gap-2'),
                        html.Div(
                            id='model2-output',
                            className='h-96 overflow-auto p-4 bg-gray-50 border border-gray-200 rounded-lg font-mono text-sm'
                        )
                    ], className='w-1/2 pl-2')
                ], className='flex'),
                
                # Metadata Section with improved styling
                html.Div([
                    html.Div([
                        html.H4("Comparison Metadata", className='text-lg font-semibold text-gray-700'),
                        html.Button(
                            'Copy Metadata', 
                            id='copy-metadata-button',
                            className=BUTTON_STYLES['secondary'] + ' text-sm'
                        )
                    ], className='flex justify-between items-center mb-2'),
                    html.Div(
                        id='metadata',
                        className='p-4 bg-gray-50 rounded-lg text-sm text-gray-600'
                    )
                ], className='mt-6')
            ], className='bg-white rounded-lg shadow-card p-6')
        ], className='w-3/4 pl-6')
    ], className='max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex gap-6')
], className='min-h-screen bg-gray-100 pb-12')

# Callbacks
@callback(
    [Output('loading-output', 'children'),
     Output('loading-output-container', 'children'),
     Output('running-status', 'children'),
     Output('model1-dropdown', 'options'),
     Output('model2-dropdown', 'options'),
     Output('model1-dropdown', 'value'),
     Output('model2-dropdown', 'value'),
     Output('model1-output', 'children'),
     Output('model2-output', 'children'),
     Output('metadata', 'children')],
    [Input('submit-button', 'n_clicks'),
     Input('model1-dropdown', 'value'),
     Input('model2-dropdown', 'value')],
    [State('prompt-input', 'value'),
     # Model selections
     State('openai-model-dropdown', 'value'),
     State('openai-response-format-dropdown', 'value'),
     State('gemini-model-dropdown', 'value'),
     State('deepseek-model-dropdown', 'value'),
     State('mistral-model-dropdown', 'value'),
     # Parameters
     State('openai-temperature-slider', 'value'),
     State('openai-max-tokens-slider', 'value'),
     State('openai-top-p-slider', 'value'),
     State('openai-frequency-penalty-slider', 'value'),
     State('openai-presence-penalty-slider', 'value'),
     State('gemini-temperature-slider', 'value'),
     State('gemini-max-tokens-slider', 'value'),
     State('deepseek-temperature-slider', 'value'),
     State('deepseek-max-tokens-slider', 'value'),
     State('mistral-temperature-slider', 'value'),
     State('mistral-max-tokens-slider', 'value')],
    prevent_initial_call=True
)
def update_all(n_clicks, model1_current, model2_current, prompt,
               openai_model, openai_response_format, gemini_model, deepseek_model, mistral_model,
               openai_temp, openai_max_tokens, openai_top_p, openai_freq_penalty, openai_pres_penalty,
               gemini_temp, gemini_max_tokens,
               deepseek_temp, deepseek_max_tokens,
               mistral_temp, mistral_max_tokens):
    # Get the trigger
    ctx = dash.callback_context
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Initialize default values
    loading_output = "Ready"
    loading_spinner = ""
    running_status = ""
    available_models = get_available_models()
    options = [{'label': model, 'value': model} for model in available_models]
    model1_value = model1_current
    model2_value = model2_current
    model1_output = read_model_output(model1_current) if model1_current else "Please select a model"
    model2_output = read_model_output(model2_current) if model2_current else "Please select a model"
    metadata = html.Div([
        html.Div([
            html.P(
                f"Comparing: ",
                className="font-semibold text-gray-700"
            ),
            html.Div([
                html.Span(
                    model1_current or 'None',
                    className="px-2 py-1 bg-primary-100 text-primary-700 rounded"
                ),
                html.Span(" vs ", className="px-2 text-gray-500"),
                html.Span(
                    model2_current or 'None',
                    className="px-2 py-1 bg-primary-100 text-primary-700 rounded"
                )
            ], className="flex items-center mt-1")
        ], className="mb-4"),
        
        html.H5("Parameters:", className="font-semibold text-gray-700 mb-2"),
        html.Div([
            # OpenAI Parameters
            html.Div([
                html.H6("OpenAI", className="font-medium text-gray-700 mb-1"),
                html.Div([
                    html.Span("Model: ", className="text-gray-500"),
                    html.Span(openai_model, className="font-medium"),
                    html.Br(),
                    html.Span("Response Format: ", className="text-gray-500"),
                    html.Span(openai_response_format.replace('_', ' ').title(), className="font-medium"),
                    html.Br(),
                    html.Span("Temperature: ", className="text-gray-500"),
                    html.Span(f"{openai_temp:.1f}", className="font-medium"),
                    html.Br(),
                    html.Span("Max Tokens: ", className="text-gray-500"),
                    html.Span(str(openai_max_tokens), className="font-medium"),
                    html.Br(),
                    html.Span("Top P: ", className="text-gray-500"),
                    html.Span(f"{openai_top_p:.1f}", className="font-medium"),
                    html.Br(),
                    html.Span("Frequency Penalty: ", className="text-gray-500"),
                    html.Span(f"{openai_freq_penalty:.1f}", className="font-medium"),
                    html.Br(),
                    html.Span("Presence Penalty: ", className="text-gray-500"),
                    html.Span(f"{openai_pres_penalty:.1f}", className="font-medium"),
                ], className="pl-4")
            ], className="mb-3"),
            
            # Gemini Parameters
            html.Div([
                html.H6("Gemini", className="font-medium text-gray-700 mb-1"),
                html.Div([
                    html.Span("Model: ", className="text-gray-500"),
                    html.Span(gemini_model, className="font-medium"),
                    html.Br(),
                    html.Span("Temperature: ", className="text-gray-500"),
                    html.Span(f"{gemini_temp:.1f}", className="font-medium"),
                    html.Br(),
                    html.Span("Max Tokens: ", className="text-gray-500"),
                    html.Span(str(gemini_max_tokens), className="font-medium"),
                ], className="pl-4")
            ], className="mb-3"),
            
            # Deepseek Parameters
            html.Div([
                html.H6("Deepseek", className="font-medium text-gray-700 mb-1"),
                html.Div([
                    html.Span("Model: ", className="text-gray-500"),
                    html.Span(deepseek_model, className="font-medium"),
                    html.Br(),
                    html.Span("Temperature: ", className="text-gray-500"),
                    html.Span(f"{deepseek_temp:.1f}", className="font-medium"),
                    html.Br(),
                    html.Span("Max Tokens: ", className="text-gray-500"),
                    html.Span(str(deepseek_max_tokens), className="font-medium"),
                ], className="pl-4")
            ], className="mb-3"),
            
            # Mistral Parameters
            html.Div([
                html.H6("Mistral", className="font-medium text-gray-700 mb-1"),
                html.Div([
                    html.Span("Model: ", className="text-gray-500"),
                    html.Span(mistral_model, className="font-medium"),
                    html.Br(),
                    html.Span("Temperature: ", className="text-gray-500"),
                    html.Span(f"{mistral_temp:.1f}", className="font-medium"),
                    html.Br(),
                    html.Span("Max Tokens: ", className="text-gray-500"),
                    html.Span(str(mistral_max_tokens), className="font-medium"),
                ], className="pl-4")
            ])
        ], className="grid grid-cols-2 gap-4")
    ])

    # If the trigger was the submit button
    if trigger_id == 'submit-button' and n_clicks and prompt:
        loading_output = "Running models..."
        loading_spinner = "Processing prompt with all models..."
        running_status = html.Div([
            html.Span("Running Models ", className="animate-pulse"),
            html.Div(className="inline-block w-2 h-2 bg-primary-500 rounded-full animate-bounce mx-0.5"),
            html.Div(className="inline-block w-2 h-2 bg-primary-500 rounded-full animate-bounce mx-0.5 delay-100"),
            html.Div(className="inline-block w-2 h-2 bg-primary-500 rounded-full animate-bounce mx-0.5 delay-200")
        ], className="flex items-center")

        # Update model configurations with selected models
        model_configs = {
            'openai_config.json': {
                "model": openai_model,
                "kwargs": {
                    "temperature": openai_temp,
                    "max_tokens": openai_max_tokens,
                    "top_p": openai_top_p,
                    "frequency_penalty": openai_freq_penalty,
                    "presence_penalty": openai_pres_penalty,
                    "stream": False,
                    "response_format": {"type": openai_response_format} if openai_response_format == 'json_object' else None
                }
            },
            'gemini_config.json': {
                "model": gemini_model,
                "kwargs": {
                    # "temperature": gemini_temp,
                    # "max_output_tokens": gemini_max_tokens
                }
            },
            'deepseek_config.json': {
                "model": deepseek_model,
                "kwargs": {
                    "temperature": deepseek_temp,
                    "max_tokens": deepseek_max_tokens,
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "stream": False
                }
            },
            'mistral_config.json': {
                "model": mistral_model,
                "kwargs": {
                    "temperature": mistral_temp,
                    "max_tokens": mistral_max_tokens,
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0,
                    "stream": False
                }
            }
        }

        # Update config files
        for config_file, config_data in model_configs.items():
            with open(f'configs/{config_file}', 'w') as f:
                json.dump(config_data, f, indent=4)

        # Save prompt to file
        with open('prompt.txt', 'w') as f:
            f.write(prompt)
        
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        # Initialize models
        models = [
            OpenAIModel(api_key=os.getenv("OPENAI_API_KEY")),
            GeminiModel(api_key=os.getenv("GEMINI_API_KEY")),
            DeepseekModel(api_key=os.getenv("DEEPSEEK_API_KEY")),
            MistralModel(api_key=os.getenv("MISTRAL_API_KEY"))
        ]
        
        # Run models asynchronously
        async def run_all_models():
            for model in models:
                try:
                    response = await model.generate_response(prompt)
                    with open(f"results/{model.__class__.__name__}.txt", "w") as file:
                        file.write(response)
                except Exception as e:
                    with open(f"results/{model.__class__.__name__}.txt", "w") as file:
                        file.write(f"Error: {str(e)}")
        
        # Run the async function
        asyncio.run(run_all_models())
        
        # After running models
        loading_output = "Models run complete!"
        loading_spinner = "All models have finished processing!"
        running_status = html.Div([
            html.Span("Complete ", className="text-green-600 font-medium"),
            html.Span("✓", className="text-green-500 text-lg")
        ], className="flex items-center")
        
        # Update values after running models
        available_models = get_available_models()
        options = [{'label': model, 'value': model} for model in available_models]
        
        # Keep the same selected models if they exist, otherwise use first two available
        model1_value = model1_current if model1_current in available_models else (available_models[0] if available_models else None)
        model2_value = model2_current if model2_current in available_models else (available_models[1] if len(available_models) > 1 else None)
        
        # Get the new outputs
        model1_output = read_model_output(model1_value) if model1_value else "Please select a model"
        model2_output = read_model_output(model2_value) if model2_value else "Please select a model"

    # If the trigger was one of the dropdowns
    elif trigger_id in ['model1-dropdown', 'model2-dropdown']:
        model1_output = read_model_output(model1_current) if model1_current else "Please select a model"
        model2_output = read_model_output(model2_current) if model2_current else "Please select a model"

    return (
        loading_output,
        loading_spinner,
        running_status,
        options,
        options,
        model1_value,
        model2_value,
        model1_output,
        model2_output,
        metadata
    )

if __name__ == '__main__':
    app.run_server(debug=True) 