import tensorflow as tf
import xmltodict
import collections


def code_to_class_string(argument):
    switcher = {
                    'n02691156': "airplane",
                    'n02419796': "antelope",
                    'n02131653': "bear",
                    'n02834778': "bicycle",
                    'n01503061': "bird",
                    'n02924116': "bus",
                    'n02958343': "car",
                    'n02402425': "cattle",
                    'n02084071': "dog",
                    'n02121808': "domestic_cat",
                    'n02503517': "elephant",
                    'n02118333': "fox",
                    'n02510455': "giant_panda",
                    'n02342885': "hamster",
                    'n02374451': "horse",
                    'n02129165': "lion",
                    'n01674464': "lizard",
                    'n02484322': "monkey",
                    'n03790512': "motorcycle",
                    'n02324045': "rabbit",
                    'n02509815': "red_panda",
                    'n02411705': "sheep",
                    'n01726692': "snake",
                    'n02355227': "squirrel",
                    'n02129604': "tiger",
                    'n04468005': "train",
                    'n01662784': "turtle",
                    'n04530566': "watercraft",
                    'n02062744': "whale",
                    'n02391049': "zebra"            }
    return switcher.get(argument, "nothing")

def code_to_code_chall(argument):
    switcher = {
                    'n02691156': 1,
                    'n02419796': 2,
                    'n02131653': 3,
                    'n02834778': 4,
                    'n01503061': 5,
                    'n02924116': 6,
                    'n02958343': 7,
                    'n02402425': 8,
                    'n02084071': 9,
                    'n02121808': 10,
                    'n02503517': 11,
                    'n02118333': 12,
                    'n02510455': 13,
                    'n02342885': 14,
                    'n02374451': 15,
                    'n02129165': 16,
                    'n01674464': 17,
                    'n02484322': 18,
                    'n03790512': 19,
                    'n02324045': 20,
                    'n02509815': 21,
                    'n02411705': 22,
                    'n01726692': 23,
                    'n02355227': 24,
                    'n02129604': 25,
                    'n04468005': 26,
                    'n01662784': 27,
                    'n04530566': 28,
                    'n02062744': 29,
                    'n02391049': 30            }
    return switcher.get(argument, None)

class ImageCoder(object):
    """Helper class that provides TensorFlow image coding utilities."""

    def __init__(self, resize_maintain_aspect=None):
        # Create a single Session to run all image coding calls.
        self._sess = tf.Session()

        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)
        self._decode_jpeg_shape = tf.shape(self._decode_jpeg)


        if resize_maintain_aspect is not None:
            size = tf.constant(resize_maintain_aspect[0]*resize_maintain_aspect[1],dtype=tf.float32)
            image_shape = tf.cast(tf.shape(self._decode_jpeg),tf.float32)
            image_size = image_shape[0]*image_shape[1]
            size_diff = tf.sqrt(size/image_size)
            new_size = tf.stack([image_shape[0]*size_diff,image_shape[1]*size_diff])
            resize_image =  tf.image.resize_images(self._decode_jpeg, tf.cast(new_size,tf.int32))
            encoded_image = tf.image.encode_jpeg(tf.cast(resize_image,tf.uint8))
            self._encode_jpeg = tf.cond(size_diff < tf.constant(1.0, tf.float32), lambda: encoded_image, false_fn=lambda: tf.identity(self._decode_jpeg_data))
            self._decode_jpeg_shape = tf.cond(size_diff < tf.constant(1.0, tf.float32), lambda: tf.shape(resize_image), false_fn=lambda: self._decode_jpeg_shape)
        else:
            self._encode_jpeg = self._decode_jpeg_data

    def encode_jpeg(self, image_data):
        image_data,shape = self._sess.run([self._encode_jpeg,self._decode_jpeg_shape],
                               feed_dict={self._decode_jpeg_data: image_data})
        assert len(shape) == 3
        assert shape[2] == 3
        return image_data,shape

def norm_bbox(bbox,height,width):
    xmin = float(bbox["xmin"])/float(width)
    ymin = float(bbox["ymin"]) / float(height)
    xmax = float(bbox["xmax"]) / float(width)
    ymax = float(bbox["ymax"]) / float(height)
    return [ymin,xmin,ymax,xmax]

def parse_object(track_object,height,width):
    id = int(track_object["trackid"])
    bbox = norm_bbox(track_object["bndbox"],height,width)
    occluded = int(track_object["occluded"])
    generated = int(track_object["generated"])
    label = code_to_code_chall(str(track_object["name"]))
    return  {"id":id, "bbox":bbox,"occluded":occluded,"generated":generated, "label":label}


def process_xml(filename):

    # parse an xml file by name
    with open(filename) as fd:
        annotations = xmltodict.parse(fd.read())
    if not "object" in annotations["annotation"]:
        return []
    height = annotations["annotation"]["size"]["height"]
    width = annotations["annotation"]["size"]["width"]

    annotated_objects = []
    parsed_objects = annotations["annotation"]["object"]
    if not isinstance(parsed_objects, list):
        parsed_objects = [parsed_objects]
    for object in parsed_objects:
        annotated_objects.append(parse_object(object,height,width))
    return annotated_objects

def process_image(filename, coder):
    """Process a single image file.

    Args:
      filename: string, path to an image file e.g., '/path/to/example.JPG'.
      coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
      image_buffer: string, JPEG encoding of RGB image.
      height: integer, image height in pixels.
      width: integer, image width in pixels.
    """
    # Read the image file.
    with tf.gfile.FastGFile(filename, 'rb') as f:
        image_data = f.read()

    # Decode the RGB JPEG.
    image_data,shape = coder.encode_jpeg(image_data)
    # Check that image converted to RGB
    assert len(shape) == 3
    height = shape[0]
    width = shape[1]
    assert shape[2] == 3

    return image_data, height, width

