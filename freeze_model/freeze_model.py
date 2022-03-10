import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model


# Clear any previous session.
tf.keras.backend.clear_session()
tf.compat.v1.disable_eager_execution()

save_pb_dir = './model'
model_fname = './model/model.h5'
def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='frozen_model.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.compat.v1.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.compat.v1.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen

# This line must be executed before loading Keras model.
tf.keras.backend.set_learning_phase(0) 

model = load_model(model_fname)

session = tf.compat.v1.keras.backend.get_session()


input_names = [t.op.name for t in model.inputs]
output_names = [t.op.name for t in model.outputs]
output_names = ["VarIsInitializedOp"]
print (input_names, output_names)

#Prints input and output nodes names, take notes of them.
#[print(n.name) for n in tf.compat.v1.get_default_graph().as_graph_def().node]

frozen_graph = freeze_graph(session.graph, session, output_names, save_pb_dir=save_pb_dir)

# TODO run this on the jetson, as tensor rt is not compatible with my computer
from tensorflow.python.compiler.tensorrt import trt_convert as trt

trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP16',
    minimum_segment_size=50
)

graph_io.write_graph(trt_graph, "./model/", "trt_graph.pb", as_text=False)

# to see how to load the model, look at the rest of the tutorial
# https://www.dlology.com/blog/how-to-run-keras-model-on-jetson-nano/'''