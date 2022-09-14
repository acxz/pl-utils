"""Evaluate model in a production setting."""
import onnxruntime
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def main():
    """Test model inference."""
    filepath = 'mlp_example.onnx'
    ort_session = onnxruntime.InferenceSession(filepath)
    input_name = ort_session.get_inputs()[0].name
    input_dim = 1
    ort_inputs = {input_name: np.random.rand(1, input_dim).astype(np.float32)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs)

    # predictions
    samples = 100000
    time = np.linspace(1, 5.7, samples)
    data_arr = np.empty((samples, 3))
    predicted_arr = np.empty_like(data_arr)
    for sample_idx in range(samples):
        # Data
        data_arr[sample_idx] = np.array([
            np.cos(2 * np.pi * time[sample_idx]),
            np.sin(2 * np.pi * time[sample_idx]),
            time[sample_idx]])

        # Predicted
        ort_inputs = {input_name: np.array(
            [[time[sample_idx]]]).astype(np.float32)}
        ort_out = ort_session.run(None, ort_inputs)
        predicted_arr[sample_idx] = ort_out[0]

    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=("Dimension 1", "Output Trajectory", "Dimension 2",
                        "Dimension 3"),
        specs=[[{"colspan": 2}, None, {"rowspan": 3, "type": "scatter3d"}],
               [{"colspan": 2}, None, None],
               [{"colspan": 2}, None, None]])

    # data
    fig.add_trace(
        go.Scatter(
            x=time,
            y=data_arr[:, 0],
            mode="lines",
            line={'color': "seagreen"},
            name="Data x",
            legendgroup="Data",
            showlegend=False,
        ), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=time,
            y=data_arr[:, 1],
            mode="lines",
            line={'color': "seagreen"},
            name="Data y",
            legendgroup="Data",
            showlegend=False,
        ), row=2, col=1)
    fig.add_trace(
        go.Scatter(
            x=time,
            y=data_arr[:, 2],
            mode="lines",
            line={'color': "seagreen"},
            name="Data z",
            legendgroup="Data",
            showlegend=False,
        ), row=3, col=1)
    fig.add_trace(
        go.Scatter3d(
            x=data_arr[:, 0],
            y=data_arr[:, 1],
            z=data_arr[:, 2],
            mode="lines",
            line={'color': "seagreen"},
            name="Data",
            legendgroup="Data",
            showlegend=True,
        ), row=1, col=3)

    # predicted
    fig.add_trace(
        go.Scatter(
            x=time,
            y=predicted_arr[:, 0],
            mode="lines",
            line={'color': "plum"},
            name="Predicted x",
            legendgroup="Prediction",
            showlegend=False,
        ), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=time,
            y=predicted_arr[:, 1],
            mode="lines",
            line={'color': "plum"},
            name="Predicted y",
            legendgroup="Prediction",
            showlegend=False,
        ), row=2, col=1)
    fig.add_trace(
        go.Scatter(
            x=time,
            y=predicted_arr[:, 2],
            mode="lines",
            line={'color': "plum"},
            name="Predicted z",
            legendgroup="Prediction",
            showlegend=False,
        ), row=3, col=1)
    fig.add_trace(
        go.Scatter3d(
            x=predicted_arr[:, 0],
            y=predicted_arr[:, 1],
            z=predicted_arr[:, 2],
            mode="lines",
            marker={'color': "plum"},
            name="Prediction",
            legendgroup="Prediction",
            showlegend=True,
        ), row=1, col=3)

    fig.update_xaxes(title_text="Input", row=1, col=1)
    fig.update_xaxes(title_text="Input", row=2, col=1)
    fig.update_xaxes(title_text="Input", row=3, col=1)

    fig.update_yaxes(title_text="Output", row=1, col=1)
    fig.update_yaxes(title_text="Output", row=2, col=1)
    fig.update_yaxes(title_text="Output", row=3, col=1)

    fig.update_layout(
        title='Multi-Layered Perceptron Regression',
        font={
            'size': 18,
        },
    )

    # To update subplot titles need separate command and size is not scaled as
    # in update_layout
    fig.update_annotations(
        font={
            'size': 22,
        },
    )

    fig.show()


if __name__ == '__main__':
    main()
