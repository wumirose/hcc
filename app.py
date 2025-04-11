import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.subplots as sp
import numpy as np
import re
from scipy.stats import entropy
import base64
import io

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Human Classification App"

# Global Variables
df = pd.DataFrame()
data_store = []
current_index = 0

# Layout
app.layout = html.Div([
    html.H1("Model Predicate Comparison", style={'textAlign': 'center'}),

    # File Upload Component
    html.Div(id="upload-section",
             children=[dcc.Upload(
                           id='upload-data',
                           children=html.Button('Upload CSV File', style={'marginBottom': '10px'}),
                           multiple=False
                       )], style={'textAlign': 'center', 'marginTop': '20px'}),

    # Display Current Row Data
    html.Div(id="current-row-display", style={'textAlign': 'center', 'marginTop': '20px', 'fontSize': '18px'}),

    # Option Box for Human Classification
    html.Div(id='option-box-container', style={'width': '50%', 'margin': 'auto', 'marginTop': '20px'}),

    # Submit Button
    html.Div(html.Button("Submit Classification", id="submit-btn", style={'marginTop': '10px'}, disabled=True),
             style={'textAlign': 'center', 'marginTop': '20px'}),

    # Navigation Controls
    html.Div([
        html.Button("Previous", id="prev-btn", n_clicks=0, style={'marginRight': '10px'}),
        html.Button("Next", id="next-btn", n_clicks=0)
    ], style={'textAlign': 'center', 'marginTop': '20px'}),

    dbc.Progress(id="classification-progress", value=0, max=100, style={'width': '50%', 'marginTop': '10px'}),

    # Autosave Status
    html.Div(id='autosave-status', style={'textAlign': 'center', 'marginTop': '10px', 'color': 'green'}),

    # Download Button
    html.Div(html.Button("Download Updated Data", id="download-btn", style={'marginTop': '20px'}),
             style={'textAlign': 'center', 'marginTop': '20px'}),
    dcc.Download(id="download-dataframe-csv"),

    html.Div([
        html.H2("Analysis"),
        html.Div(id="statistical-analysis"),
        html.Div(id='accuracy-display'),
        dcc.Graph(id='agreement-bar-chart'),
        dcc.Graph(id='relationship-heatmap'),
        dcc.Graph(id='entropy-histogram'),

    ], id="stat-section", style={'display': 'none'}),

    # Hidden Div to store selected option
    dcc.Store(id='selected-option', data='none'),  # Hidden storage for selection
    dcc.Store(id='data-uploaded', data=False),  # Store the upload status
])


@app.callback(
    [
        Output('prev-btn', 'disabled'),
        Output('next-btn', 'disabled'),
        Output('option-box-container', 'style')
    ],
    Input('upload-data', 'contents')
)
def handle_button_visibility( contents ):
    if not contents:
        return True, True, {'display': 'none'}
    return False, False, {'display': 'block'}


@app.callback(
    Output('submit-btn', 'disabled'),
    Input('selected-option', 'data')
)
def toggle_submit_button( selected_option ):
    return selected_option in [None, '']


@app.callback(
    [
        Output('classification-progress', 'value'),
        Output('stat-section', 'style'),
        Output('current-row-display', 'children'),
        Output('option-box-container', 'children'),
        Output('autosave-status', 'children'),
        Output('statistical-analysis', 'children'),
        Output('agreement-bar-chart', 'figure'),
        Output('entropy-histogram', 'figure'),
        Output('accuracy-display', 'children'),
        Output('relationship-heatmap', 'figure'),
    ],
    [
        Input('upload-data', 'contents'),
        Input('prev-btn', 'n_clicks'),
        Input('next-btn', 'n_clicks'),
        Input('submit-btn', 'n_clicks'),
        Input({'type': 'option-box', 'index': dash.ALL}, 'n_clicks')
    ],
    [
        State('upload-data', 'filename'),
        State('selected-option', 'data')
    ]
)
def update_dashboard( contents, prev_clicks, next_clicks, submit_clicks, option_clicks, filename, selected_option ):
    global df, data_store, current_index
    ctx = dash.callback_context

    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    # === Handle File Upload === #
    if trigger == 'upload-data':
        if not contents:
            return 0, {'display': 'none'}, "Upload a CSV file to get started.", [], "", html.Div(
                "Statistics will appear here"), {}, {}, "", {}

        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.BytesIO(decoded))

        if "row_id" in df.columns:
            df.drop("row_id", axis=1, inplace=True)
        df.reset_index(inplace=True)  # this turns the index into a column called "index"
        df.rename(columns={"index": "row_id"}, inplace=True)

        if "human_choice" not in df.columns:
            df["human_choice"] = ""
        df["human_choice"] = df["human_choice"].fillna("")

        df['entropy'] = df.apply(compute_entropy, axis=1)
        df = df.sort_values(by="entropy", ascending=False).reset_index(drop=True)

        data_store = df.to_dict("records")
        option_box = generate_option_box(get_current_options())
        accuracy_table, fig_accuracy, fig_entropy, relation_heatmap = get_statistics()

        progress_value = (df["human_choice"].notna().sum() / len(df)) * 100
        return progress_value, {'display': 'block'}, display_current_row(
            current_index), option_box, "", accuracy_table, fig_accuracy, fig_entropy, "", relation_heatmap

    # === Handle Navigation === #
    if trigger in ["prev-btn", "next-btn"]:
        if trigger == "prev-btn" and current_index > 0:
            current_index -= 1
        elif trigger == "next-btn" and current_index < len(data_store) - 1:
            current_index += 1

        option_box = generate_option_box(get_current_options())
        accuracy_table, fig_accuracy, fig_entropy, relation_heatmap = get_statistics()
        progress_value = (df["human_choice"].notna().sum() / len(df)) * 100
        return progress_value, {'display': 'block'}, display_current_row(
            current_index), option_box, "", accuracy_table, fig_accuracy, fig_entropy, "", relation_heatmap

    # === Handle Classification Submission === #
    if trigger == "submit-btn":
        if len(data_store) > 0 and selected_option:
            chosen_value = selected_option.strip()
            data_store[current_index]["human_choice"] = chosen_value
            row_id = data_store[current_index]["row_id"]
            df.loc[df["row_id"] == row_id, "human_choice"] = chosen_value
            # df.loc[current_index, "human_choice"] = chosen_value
            df.to_csv(filename, index=False)
        option_box = generate_option_box(get_current_options())
        accuracy_table, fig_accuracy, fig_entropy, relation_heatmap = get_statistics()
        progress_value = (df["human_choice"].notna().sum() / len(df)) * 100
        return progress_value, {'display': 'block'}, display_current_row(
            current_index), option_box, "Classifications saved!", accuracy_table, fig_accuracy, fig_entropy, "", relation_heatmap

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


@app.callback(
    Output('selected-option', 'data'),
    Input({'type': 'option-box', 'index': dash.ALL}, 'n_clicks_timestamp'),
    State({'type': 'option-box', 'index': dash.ALL}, 'id'),
    prevent_initial_call=True
)
def update_selection( click_timestamps, option_ids ):
    if not click_timestamps or all(ts is None for ts in click_timestamps):
        return dash.no_update

    click_timestamps = [ts if ts is not None else 0 for ts in click_timestamps]
    latest_index = np.argmax(click_timestamps)
    selected_option = option_ids[latest_index]['index'].replace('option-', '').strip()
    return selected_option


# Callback to update the style of the selected option
@app.callback(
    Output({'type': 'option-box', 'index': dash.ALL}, 'style'),
    Input('selected-option', 'data'),
    State({'type': 'option-box', 'index': dash.ALL}, 'id'),
    prevent_initial_call=True
)
def update_styles( selected_value, option_ids ):
    if not option_ids:
        return dash.no_update
    return [
        {
            'padding': '10px',
            'cursor': 'pointer',
            'textAlign': 'center',
            'borderRadius': '5px',
            'backgroundColor': '#ffeb3b' if option['index'].replace('option-', '') == selected_value else '#f8f9fa',
            'color': 'black' if option['index'].replace('option-', '') != selected_value else 'white',
            'border': '2px solid #0056b3' if option['index'].replace('option-',
                                                                     '') == selected_value else '1px solid #ccc',
            'margin': '5px',
            'fontSize': '16px',
            'fontWeight': 'bold',
            'boxShadow': '2px 2px 10px rgba(0,0,0,0.2)' if option['index'].replace('option-',
                                                                                   '') == selected_value else None
        }
        for option in option_ids
    ]


def display_current_row( index ):
    if len(data_store) == 0:
        return "No data available."

    row = data_store[index]
    # Extract subject and object
    subject = row["subject"]
    obj = row["object"]
    abstract = row["abstract"]
    pred = row['relationship']

    highlighted = highlight_text(abstract, subject, obj)
    # highlighted = highlight_text(highlighted, subject)

    # Render the abstract with highlighted subject and object using Div for HTML rendering
    return html.Div([
        html.P([f"{(df['human_choice'] != '').sum()} / {len(data_store)} Abstracts Completed"]),
        html.Div(children=highlighted),
        html.P(f"Subject: {subject}"),
        html.P(f"Object: {obj}"),
        html.P(f"LLM Predicate: {pred}"),
        html.P(f"My Current Choice: {row.get('human_choice', 'Not classified yet')}", style={'color': 'green'}),
        html.P("Possible Choices")
    ])


def highlight_text( sentence, substring1, substring2 ):
    matches1 = list(re.finditer(re.escape(substring1), sentence, re.IGNORECASE))
    matches2 = list(re.finditer(re.escape(substring2), sentence, re.IGNORECASE))

    matches = []

    for match in matches1:
        matches.append({'start': match.start(), 'end': match.end(), 'substring': sentence[match.start():match.end()],
                        'color': 'yellow'})
    for match in matches2:
        matches.append({'start': match.start(), 'end': match.end(), 'substring': sentence[match.start():match.end()],
                        'color': 'orange'})

    matches.sort(key=lambda match: match['start'])

    if not matches:
        return sentence

    elements = []
    last_index = 0
    for match in matches:
        start = match['start']
        end = match['end']
        color = match['color']
        substring = match['substring']
        elements.append(sentence[last_index:start])
        elements.append(html.Span(substring, style={'background-color': color}))
        last_index = end

    elements.append(sentence[last_index:])
    return elements


def generate_option_box( options ):
    # print([option.values() for option in options])
    return html.Div(
        id='option-box',
        children=[
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                html.Div(
                                    option["label"],
                                    id={'type': 'option-box', 'index': f"option-{option['value']}"},
                                    n_clicks=0,
                                    style={
                                        'padding': '10px',
                                        'cursor': 'pointer',
                                        'textAlign': 'center',
                                        'borderRadius': '5px',
                                        'backgroundColor': option.get('color', '#f8f9fa'),  # Light gray background
                                        'border': '1px solid #ccc',
                                        'margin': '5px',
                                        'fontSize': '16px',
                                        'fontWeight': 'bold'
                                    }
                                )
                            ),
                            style={"width": "100%", "boxShadow": "2px 2px 10px rgba(0,0,0,0.1)"}
                        ),
                        style={"width": "auto"}
                    ) for option in options
                ],
                justify="center"  # Centers the options
            )
        ],
        style={'textAlign': 'center', 'margin': '20px auto', 'width': '50%'}
    )


def get_current_options():
    models = ["llama2", "llama3", "deepseek", "gemma3"]
    row = data_store[current_index]
    vector_choices = row.get("all_vector_candidate", "").split(",")
    models_choices = [row.get(m) for m in models]
    classification_options = []
    for rel in vector_choices:
        rel = rel.strip()
        if rel in models_choices:
            classification_options.append({"label": rel, "value": rel, 'color': '#b5cbdf'})
        else:
            classification_options.append({"label": rel, "value": rel})
    if "none" in models_choices:
        classification_options.append({"label": "none", "value": "none", "color": "#b5cbdf"})
    else:
        classification_options.append({"label": "none", "value": "none"})

    # print("classification options: ", classification_options)
    # print("model choices: ", models_choices)
    return classification_options


def get_all_unique_labels( models ):
    label_set = set()

    for model in models:
        label_set.update(df[model].dropna().str.strip().str.lower().unique())

    label_set.update(df['human_choice'].dropna().str.strip().str.lower().unique())

    return sorted(label_set)


def get_model_performance_by_relationship():
    if df.empty:
        return html.Div("No data available."), {}

    models = ['llama2', 'llama3', 'deepseek', 'gemma3']
    unique_labels = get_all_unique_labels(models)
    total_rows = len(df)

    count_data = []
    heatmap_matrix = []

    for model in models + ['human_choice']:
        row = {'Model': model}
        model_counts = []
        total = 0

        # Temporary dict to calculate raw counts for this model
        model_raw_counts = df[model].str.strip().str.lower().value_counts().to_dict()

        for label in unique_labels:
            raw_count = model_raw_counts.get(label.lower(), 0)
            percentage = (raw_count / total_rows) * 100 if total_rows > 0 else 0
            row[label] = f"{raw_count} ({percentage:.1f}%)"
            model_counts.append(raw_count)
            total += raw_count

        # Add total summary
        total_percent = (total / total_rows) * 100 if total_rows > 0 else 0
        row['Total (Count %)'] = f"{total} ({total_percent:.1f}%)"

        count_data.append(row)
        heatmap_matrix.append(model_counts)

    # Table with counts and percentages
    columns = [{"name": col, "id": col} for col in ['Model'] + unique_labels + ['Total (Count %)']]
    frequency_table = dash_table.DataTable(
        columns=columns,
        data=count_data,
        style_table={'overflowX': 'auto'},
        style_header={'fontWeight': 'bold', 'textAlign': 'center'},
        style_cell={'textAlign': 'center', 'padding': '5px', 'whiteSpace': 'normal'},
    )

    # Reset heatmap matrix
    heatmap_matrix = {model: [] for model in models}

    for model in models:
        heatmap_data = []

        for human_label in unique_labels:
            row = []
            for model_label in unique_labels:
                count = df[
                    (df['human_choice'].str.strip().str.lower() == human_label.lower()) &
                    (df[model].str.strip().str.lower() == model_label.lower())
                    ].shape[0]
                row.append(count)
            heatmap_data.append(row)

        heatmap_matrix[model] = heatmap_data

    fig = sp.make_subplots(
        rows=1, cols=4, shared_yaxes=True,
        subplot_titles=[f"Human's vs {model} " for model in models],
        column_widths=[1, 1, 1, 1]
    )

    # Add each heatmap to the subplot
    for i, model in enumerate(models):
        heatmap_fig = px.imshow(
            np.array(heatmap_matrix[model]),
            x=unique_labels,
            y=unique_labels,
            color_continuous_scale='Blues',
            text_auto=True
        )

        for trace in heatmap_fig.data:
            fig.add_trace(trace, row=1, col=i + 1)

        # Add axis titles
        fig.update_xaxes(title_text="Model's Choice", row=1, col=i + 1)
        if i == 0:  # Only add y-axis label once
            fig.update_yaxes(title_text="My Choice", row=1, col=i + 1)

    # Final layout tweaks
    fig.update_layout(
        height=600, width=1200,
        title_text="Model Predictions vs Human Choice",
        showlegend=False
    )
    fig.update_coloraxes(showscale=False)
    return frequency_table, fig

# === Entropy Calculation ===
def compute_entropy( row ):
    labels = [row["llama2"], row["llama3"], row["deepseek"], row["gemma3"]]
    label_counts = pd.Series(labels).value_counts().values
    return entropy(label_counts)

def get_statistics():
    if df.empty:
        return html.Div("No data available for statistics."), {}, {}

    # === Histogram of Entropy ===
    fig_entropy = px.histogram(
        df, x='entropy', nbins=20,
        title="Uncertainty Counts (Entropy)",
        labels={'entropy': 'Entropy Value'},
        color_discrete_sequence=['purple']
    )

    # == = Accuracy Calculation == =
    individual_model_columns = ['llama2', 'llama3', 'deepseek', 'gemma3']

    individual_accuracy = {}

    # Calculate accuracy for individual models
    for model in individual_model_columns:
        correct_predictions = (df[model].str.strip().str.lower() == df['human_choice'].str.strip().str.lower()).sum()
        total_predictions = len(df[model])
        individual_accuracy[model] = (correct_predictions / total_predictions) * 100

    # Convert accuracy dictionary to a DataFrame for display
    accuracy_df = pd.DataFrame(list(individual_accuracy.items()), columns=['Model', 'Accuracy (%)'])

    # === Table for Accuracy ===
    accuracy_columns = [{"name": col, "id": col} for col in accuracy_df.columns]
    accuracy_table = dash_table.DataTable(
        columns=accuracy_columns,
        data=accuracy_df.to_dict('records'),
        style_table={'overflowX': 'auto'},
        style_header={'fontWeight': 'bold', 'textAlign': 'center'},
        style_cell={'textAlign': 'center'},
    )

    # === Bar Plot for Accuracy ===
    fig_accuracy = px.bar(
        accuracy_df, x='Model', y='Accuracy (%)',
        labels={'x': 'Model', 'y': 'Accuracy (%)'},
        title="Model Accuracy",
        color='Accuracy (%)',
        color_continuous_scale='teal'
    )
    fig_accuracy.update_coloraxes(showscale=False)
    # Get new detailed table + heatmap
    relation_table, relation_heatmap = get_model_performance_by_relationship()

    return html.Div([
        html.H3("Overall Accuracy"),
        accuracy_table,
        html.H3("Model Agreement by Relationship"),
        relation_table,
    ]), fig_accuracy, fig_entropy, relation_heatmap


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("download-btn", "n_clicks"),
    prevent_initial_call=True
)
def download_data( n_clicks ):
    global df
    if "row_id" in df.columns:
        df.drop("row_id", axis=1, inplace=True)
    return dcc.send_data_frame(df.to_csv, "updated_data.csv", index=False)


if __name__ == "__main__":
    app.run(debug=True)
