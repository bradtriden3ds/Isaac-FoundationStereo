# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
import socket
import struct
import os
from PIL import Image
import io

from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *


class SAM6DClient:
  def __init__(self, host='localhost', port=8000):
    self.host = host
    self.port = port
    self.socket = None

  def connect(self):
    """Connect to the SAM-6D server"""
    try:
      self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      self.socket.connect((self.host, self.port))
      logging.info(f"Connected to SAM-6D server at {self.host}:{self.port}")
      return True
    except Exception as e:
      logging.error(f"Failed to connect to server: {str(e)}")
      return False

  def send_data(self, data):
    """Send data to the server"""
    serialized = pickle.dumps(data)
    length = struct.pack('!I', len(serialized))
    self.socket.sendall(length + serialized)

  def receive_data(self):
    """Receive data from the server"""
    # First, receive the length of the message
    length_data = b''
    while len(length_data) < 4:
      chunk = self.socket.recv(4 - len(length_data))
      if not chunk:
        raise ConnectionError("Connection closed")
      length_data += chunk

    length = struct.unpack('!I', length_data)[0]

    # Now receive the actual data
    data = b''
    while len(data) < length:
      chunk = self.socket.recv(length - len(data))
      if not chunk:
        raise ConnectionError("Connection closed")
      data += chunk

    return data

  def disconnect(self):
    """Disconnect from the server"""
    if self.socket:
      self.socket.close()

def receive_images(connection, address):
  print(f"Connection from {address} has been established!")

  image0_data = receive_data(connection)
  img0 = cv2.imdecode(np.frombuffer(image0_data, dtype=np.uint8), cv2.IMREAD_COLOR)

  image1_data = receive_data(connection)
  img1 = cv2.imdecode(np.frombuffer(image1_data, dtype=np.uint8), cv2.IMREAD_COLOR)

  return img0, img1

def receive_data(conn):
    """Receive data over socket with length prefix"""
    # First, receive the length of the message
    length_data = b''
    while len(length_data) < 4:
        chunk = conn.recv(4 - len(length_data))
        if not chunk:
            raise ConnectionError("Connection closed")
        length_data += chunk
    
    length = struct.unpack('!I', length_data)[0]
    logging.info(f"Receiving image of size {length}")
    # Now receive the actual data
    data = b''
    while len(data) < length:
        chunk = conn.recv(length - len(data))
        logging.info(f"Received packet of size {len(data)}.  Total received {len(data)}")
        if not chunk:
            raise ConnectionError("Connection closed")
        data += chunk
    logging.info(f"Receiving image complete")
    return data


def run_inference(img0, img1, baseline, K, model, args):
  img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
  img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
  H, W = img0.shape[:2]

  img0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
  img1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
  padder = InputPadder(img0.shape, divis_by=32, force_square=False)
  img0, img1 = padder.pad(img0, img1)

  with torch.cuda.amp.autocast(True):
    if not args.hiera:
      disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True)
    else:
      disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
  disp = padder.unpad(disp.float())
  disp = disp.data.cpu().numpy().reshape(H, W)

  if args.remove_invisible:
    yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
    us_right = xx - disp
    invalid = us_right < 0
    disp[invalid] = np.inf

  K[:2] *= scale
  depth = K[0, 0] * baseline / disp
  depth_mm = (depth * 1000.0).astype(np.uint16)
  return depth_mm
  #return cv2.cvtColor(depth_mm, cv2.COLOR_RGB2BGR)
  #return Image.fromarray(depth_mm)

def start_server(args, model, host='0.0.0.0', port=12345, clientport=8000):
  server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  server_socket.bind((host, port))
  server_socket.listen(5)
  print(f"Server listening on {host}:{port}")

  with open(args.intrinsic_file, 'r') as f:
    lines = f.readlines()
    K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3, 3)
    baseline = float(lines[1])

  sam6d_client = SAM6DClient(port=clientport)
  sam6d_client.connect()
  
  try:
    running = True
    while running:
      try:
        connection, address = server_socket.accept()
        logging.info(f"connection received from {address}")

        img0, img1 = receive_images(connection, address)
        depth_array = run_inference(img0, img1, baseline, K, model, args)

        logging.info(f"inference complete")
        img0bytes = cv2.imencode('.png', img0)[1].tobytes()
        _, depth_png = cv2.imencode('.png', depth_array)
        depth_bytes = depth_png.tobytes()
        # server_socket send img0b
        # create request compress
        request = {
          'action': 'sam6d_inference',
          'rgb_bytes': img0bytes,
          'depth_bytes': depth_bytes,
          'det_score_thresh': 0.5,
          'visualize': False
        }
        sam6d_client.send_data(request)
        logging.info(f"sent depth + image to SAM6D")
        response = sam6d_client.receive_data()
        logging.info(f"received responsed from SAM6D")

        # server_socket send size
        connection.send(struct.pack('!I', len(response)))
        connection.sendall(response)
        logging.info(f"sent response to isaac")
      except OSError:
        break
      except socket.timeout:
        running=False
        break
  except Exception as e:
    logging.error(f"Server error: {str(e)}")
  finally:
    """Cleanup server resources"""
    logging.info("Shutting down server...")
    if server_socket:
        server_socket.close()
    logging.info("Server shutdown complete")  

if __name__=="__main__":
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/K.txt', type=str, help='camera intrinsic matrix and baseline file')
  parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/11-33-40/model_best_bp2.pth', type=str, help='pretrained model path')
  parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
  parser.add_argument('--hiera', default=0, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
  parser.add_argument('--z_far', default=10, type=float, help='max depth to clip in point cloud')
  parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
  parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
  parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')
  parser.add_argument('--denoise_cloud', type=int, default=1, help='whether to denoise the point cloud')
  parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
  parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
  parser.add_argument('--sam6dport', type=int, default=8000, help='port sam6d is running on')
  parser.add_argument('--isaacsimport', type=int, default=12345, help='port to listen for connection from isaac sim')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)
  #os.makedirs(args.out_dir, exist_ok=True)

  ckpt_dir = args.ckpt_dir
  cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
  if 'vit_size' not in cfg:
    cfg['vit_size'] = 'vitl'
  for k in args.__dict__:
    cfg[k] = args.__dict__[k]
  args = OmegaConf.create(cfg)
  logging.info(f"args:\n{args}")
  logging.info(f"Using pretrained model from {ckpt_dir}")

  model = FoundationStereo(args)

  ckpt = torch.load(ckpt_dir, weights_only=False, map_location='cpu')
  logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
  model.load_state_dict(ckpt['model'])

  model.cuda()
  model.eval()

  scale = args.scale
  assert scale<=1, "scale must be <=1"

  start_server(args, model, port=args.isaacsimport, clientport=args.sam6dport)
