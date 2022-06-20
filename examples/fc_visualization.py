"""Evaluate model in a production setting."""
import onnxruntime
import numpy as np


def main():
    """Test model inference."""
    filepath = 'fc_example.onnx'
    ort_session = onnxruntime.InferenceSession(filepath)
    input_name = ort_session.get_inputs()[0].name
    input_dim = 1
    ort_inputs = {input_name: np.random.rand(1, input_dim).astype(np.float32)}
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs)


if __name__ == '__main__':
    main()
