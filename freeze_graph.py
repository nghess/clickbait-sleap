import tensorflow as tf
from pathlib import Path
import sleap
import json
import numpy as np
from sleap.nn.inference import (
    CentroidCrop,
    CentroidInferenceModel,
    TopDownInferenceModel,
    FindInstancePeaks,
    TopDownMultiClassFindPeaks,
    TopDownMultiClassInferenceModel,
    SingleInstanceInferenceModel,
    SingleInstanceInferenceLayer
)

import ast
def export_frozen_graph(model, preds, output_path):

    tensors = {}

    for key, val in preds.items():
        dtype = str(val.dtype) if isinstance(val.dtype, np.dtype) else repr(val.dtype)
        tensors[key] = {
            "type": f"{type(val).__name__}",
            "shape": f"{val.shape}",
            "dtype": dtype,
            "device": f"{val.device if hasattr(val, 'device') else 'N/A'}",
        }

    with output_path as d:
        model.export_model(d.as_posix(), tensors=tensors)

        tf.compat.v1.reset_default_graph()
        with tf.compat.v2.io.gfile.GFile(f"{d}/frozen_graph.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def)

        with open(f"{d}/info.json") as json_file:
            info = json.load(json_file)

        for tensor_info in info["frozen_model_inputs"] + info["frozen_model_outputs"]:

            saved_name = (
                tensor_info.split("Tensor(")[1].split(", shape")[0].replace('"', "")
            )
            saved_shape = ast.literal_eval(
                tensor_info.split("shape=", 1)[1].split("), ")[0] + ")"
            )
            saved_dtype = tensor_info.split("dtype=")[1].split(")")[0]

            loaded_shape = tuple(graph.get_tensor_by_name(f"import/{saved_name}").shape)
            loaded_dtype = graph.get_tensor_by_name(f"import/{saved_name}").dtype.name

            assert saved_shape == loaded_shape
            assert saved_dtype == loaded_dtype

#Single instance model (using full picture for now...)

runs_folder = r"A:/clickbait-sleap/models/"

single_instance_model_path = runs_folder + "/" + r"241211_201049.single_instance.n=197/best_model.h5"
single_instance_model = tf.keras.models.load_model(single_instance_model_path, compile = False)

model = SingleInstanceInferenceModel(
    SingleInstanceInferenceLayer(keras_model=single_instance_model)
)

preds = model.predict(np.zeros((1, 976, 448, 1), dtype="uint8"))

export_frozen_graph(model, preds, Path(runs_folder))