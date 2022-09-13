"""Load GP model and prediction."""
import math

import pl_utils as plu

import torch


def main():
    # create some input data to be predicted on
    samples = 1000
    freq = 4 * math.pi
    x_domain = 1
    predict_input_data = torch.linspace(0, x_domain, samples).unsqueeze(1)

    # load model
    pt_path = 'gp_regression_example.pt'
    checkpoint = torch.load(pt_path)
    model_state_dict = checkpoint['model_state_dict']
    train_input_data = checkpoint['train_input_data']
    train_output_data = checkpoint['train_output_data']
    model = plu.models.gp.BIMOEGPModel(train_input_data, train_output_data)
    model.load_state_dict(model_state_dict)

    # predict on mode
    model.eval()
    with torch.no_grad():
        predictions = model.likelihood(model(predict_input_data))
        mean = predictions.mean
        lower, upper = predictions.confidence_region()

    # plots
    # pylint: disable=import-outside-toplevel
    import plotly.graph_objects as go

    fig_simp = go.Figure()

    fig_simp.add_trace(
        go.Scatter(
            x=model.hparams.train_input_data[:, 0],
            y=model.hparams.train_output_data[:, 0],
            mode='markers',
            marker={
                'size': 10,
            },
            name='data samples',
            showlegend=True
        )
    )

    fig_simp.add_trace(
        go.Scatter(
            x=predict_input_data[:, 0],
            y=mean[:, 0],
            mode='lines',
            name='posterior mean',
            showlegend=True
        )
    )

    fig_simp.add_trace(
        go.Scatter(
            x=predict_input_data[:, 0],
            y=upper[:, 0],
            mode='lines',
            line={
                'color': 'blue',
            },
            opacity=0,
            name='sin upper',
            showlegend=False
        )
    )

    fig_simp.add_trace(
        go.Scatter(
            x=predict_input_data[:, 0],
            y=lower[:, 0],
            mode='lines',
            line={
                'color': 'blue',
            },
            opacity=0,
            name='sin lower',
            showlegend=False
        )
    )

    fig_simp.add_trace(
        go.Scatter(
            x=torch.cat([predict_input_data[:, 0],
                         torch.flip(predict_input_data[:, 0], [0])]),
            y=torch.cat([upper[:, 0], torch.flip(lower[:, 0], [0])]),
            mode='lines',
            line={
                'color': 'rgba(255, 255, 255, 0)',
            },
            fillcolor='black',
            opacity=0.2,
            fill='toself',
            # name='sin confidence',
            name='confidence region',
            showlegend=True
        )
    )

    fig_simp.update_layout(
        title='Gaussian Process Regression',
        xaxis_title='Input',
        yaxis_title='Observation',
        font={
            'size': 18,
        },
    )

    fig_simp.show()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=predict_input_data[:, 0],
            y=torch.sin(predict_input_data[:, 0] * freq),
            mode='lines',
            line={
                'color': 'blue',
                'dash': 'dot',
            },
            name='sin func',
            showlegend=True
        )
    )

    fig.add_trace(
        go.Scatter(
            x=model.hparams.train_input_data[:, 0],
            y=model.hparams.train_output_data[:, 0],
            mode='markers',
            marker={
                'size': 10,
                'color': 'blue',
            },
            name='sin data',
            showlegend=True
        )
    )

    fig.add_trace(
        go.Scatter(
            x=predict_input_data[:, 0],
            y=mean[:, 0],
            mode='lines',
            line={
                'color': 'blue',
            },
            name='sin mean',
            showlegend=True
        )
    )

    fig.add_trace(
        go.Scatter(
            x=predict_input_data[:, 0],
            y=upper[:, 0],
            mode='lines',
            line={
                'color': 'blue',
            },
            opacity=0,
            name='sin upper',
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=predict_input_data[:, 0],
            y=lower[:, 0],
            mode='lines',
            line={
                'color': 'blue',
            },
            opacity=0,
            name='sin lower',
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=torch.cat([predict_input_data[:, 0],
                         torch.flip(predict_input_data[:, 0], [0])]),
            y=torch.cat([upper[:, 0], torch.flip(lower[:, 0], [0])]),
            mode='lines',
            line={
                'color': 'rgba(255, 255, 255, 0)',
            },
            fillcolor='blue',
            opacity=0.2,
            fill='toself',
            name='sin confidence',
            showlegend=True
        )
    )

    fig.add_trace(
        go.Scatter(
            x=predict_input_data[:, 0],
            y=torch.cos(predict_input_data[:, 0] * freq),
            mode='lines',
            line={
                'color': 'red',
                'dash': 'dot',
            },
            name='cos func',
            showlegend=True
        )
    )

    fig.add_trace(
        go.Scatter(
            x=model.hparams.train_input_data[:, 0],
            y=model.hparams.train_output_data[:, 1],
            mode='markers',
            marker={
                'size': 10,
                'color': 'red',
            },
            name='cos data',
            showlegend=True
        )
    )

    fig.add_trace(
        go.Scatter(
            x=predict_input_data[:, 0],
            y=mean[:, 1],
            mode='lines',
            line={
                'color': 'red',
            },
            name='cos mean',
            showlegend=True
        )
    )

    fig.add_trace(
        go.Scatter(
            x=predict_input_data[:, 0],
            y=upper[:, 1],
            mode='lines',
            line={
                'color': 'red',
            },
            opacity=0,
            name='cos upper',
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=predict_input_data[:, 0],
            y=lower[:, 1],
            mode='lines',
            line={
                'color': 'red',
            },
            opacity=0,
            name='cos lower',
            showlegend=False
        )
    )

    fig.add_trace(
        go.Scatter(
            x=torch.cat([predict_input_data[:, 0],
                         torch.flip(predict_input_data[:, 0], [0])]),
            y=torch.cat([upper[:, 1], torch.flip(lower[:, 1], [0])]),
            mode='lines',
            line={
                'color': 'rgba(255, 255, 255, 0)',
            },
            fillcolor='red',
            opacity=0.2,
            fill='toself',
            name='cos confidence',
            showlegend=True
        )
    )

    fig.show()


if __name__ == '__main__':
    main()
