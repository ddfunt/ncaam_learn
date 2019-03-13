import tensorflow as tf
from tensorflow.core.framework import graph_pb2 as gpb
from google.protobuf import text_format as pbtf

gdef = gpb.GraphDef()

with open('model/graph.pbtxt', 'r') as fh:
    graph_str = fh.read()

pbtf.Parse(graph_str, gdef)

tf.import_graph_def(gdef)