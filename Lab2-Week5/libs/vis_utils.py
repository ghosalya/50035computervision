
""" Code taken from
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/deepdream/deepdream.ipynb
Placed here for convenience only.
"""
from IPython.display import clear_output, Image, display, HTML
# import tensorflow as tf

from math import sqrt, ceil
import numpy as np

def visualize_grid(Xs, ubound=255.0, padding=1):
  """
  Reshape a 4D tensor of image data to a grid for easy visualization.

  Inputs:
  - Xs: Data of shape (N, H, W, C)
  - ubound: Output grid will have values scaled to the range [0, ubound]
  - padding: The number of blank pixels between elements of the grid
  """
  (N, H, W, C) = Xs.shape
  grid_size = int(ceil(sqrt(N)))
  grid_height = H * grid_size + padding * (grid_size - 1)
  grid_width = W * grid_size + padding * (grid_size - 1)
  grid = np.zeros((grid_height, grid_width, C))
  next_idx = 0
  y0, y1 = 0, H
  for y in range(grid_size):
    x0, x1 = 0, W
    for x in range(grid_size):
      if next_idx < N:
        img = Xs[next_idx]
        low, high = np.min(img), np.max(img)
        grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
        # grid[y0:y1, x0:x1] = Xs[next_idx]
        next_idx += 1
      x0 += W + padding
      x1 += W + padding
    y0 += H + padding
    y1 += H + padding
  # grid_max = np.max(grid)
  # grid_min = np.min(grid)
  # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
  return grid

def vis_grid(Xs):
  """ visualize a grid of images """
  (N, H, W, C) = Xs.shape
  A = int(ceil(sqrt(N)))
  G = np.ones((A*H+A, A*W+A, C), Xs.dtype)
  G *= np.min(Xs)
  n = 0
  for y in range(A):
    for x in range(A):
      if n < N:
        G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = Xs[n,:,:,:]
        n += 1
  # normalize to [0,1]
  maxg = G.max()
  ming = G.min()
  G = (G - ming)/(maxg-ming)
  return G
  
def vis_nn(rows):
  """ visualize array of arrays of images """
  N = len(rows)
  D = len(rows[0])
  H,W,C = rows[0][0].shape
  Xs = rows[0][0]
  G = np.ones((N*H+N, D*W+D, C), Xs.dtype)
  for y in range(N):
    for x in range(D):
      G[y*H+y:(y+1)*H+y, x*W+x:(x+1)*W+x, :] = rows[y][x]
  # normalize to [0,1]
  maxg = G.max()
  ming = G.min()
  G = (G - ming)/(maxg-ming)
  return G



# def strip_consts(graph_def, max_const_size=32):
#     """Strip large constant values from graph_def."""
#     strip_def = tf.GraphDef()
#     for n0 in graph_def.node:
#         n = strip_def.node.add()
#         n.MergeFrom(n0)
#         if n.op == 'Const':
#             tensor = n.attr['value'].tensor
#             size = len(tensor.tensor_content)
#             if size > max_const_size:
#                 tensor.tensor_content = "<stripped %d bytes>"%size
#     return strip_def

# def show_graph(graph_def, max_const_size=32):
#     """Visualize TensorFlow graph."""
#     if hasattr(graph_def, 'as_graph_def'):
#         graph_def = graph_def.as_graph_def()
#     strip_def = strip_consts(graph_def, max_const_size=max_const_size)
#     code = """
#         <script>
#           function load() {{
#             document.getElementById("{id}").pbtxt = {data};
#           }}
#         </script>
#         <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
#         <div style="height:600px">
#           <tf-graph-basic id="{id}"></tf-graph-basic>
#         </div>
#     """.format(data=repr(str(strip_def)), id='graph'+str(np.random.rand()))

#     iframe = """
#         <iframe seamless style="width:1200px;height:620px;border:0" srcdoc="{}"></iframe>
#     """.format(code.replace('"', '&quot;'))
#     display(HTML(iframe))
