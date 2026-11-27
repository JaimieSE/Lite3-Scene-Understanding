import json
import threading
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO


class SceneNode(Node):
    def __init__(self):
        super().__init__('scene_detector')
        self.declare_parameter('image_topic', '/builtin_camera/image_raw')
        self.declare_parameter('model', str(Path(__file__).parent / 'yolo26n.pt'))
        self.declare_parameter('confidence', 0.4)
        self.bridge = CvBridge()
        self.model = YOLO(self.get_parameter('model').value)
        self.images = self.create_publisher(Image, '/scene/annotated_image', 1)
        self.detections = self.create_publisher(String, '/scene/detections', 1)
        self.lock = threading.Lock()
        self.latest = None
        self.stop = threading.Event()
        self.subscription = self.create_subscription(
            Image, self.get_parameter('image_topic').value,
            self.receive, qos_profile_sensor_data)
        self.worker = threading.Thread(target=self.process, daemon=True)
        self.worker.start()
        self.get_logger().info('Listening for camera frames; detection output uses JSON in std_msgs/String')

    def receive(self, msg):
        with self.lock:
            self.latest = msg

    def process(self):
        while not self.stop.is_set():
            with self.lock:
                msg, self.latest = self.latest, None
            if msg is None:
                self.stop.wait(0.02)
                continue
            try:
                started = time.monotonic()
                frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
                result = self.model.predict(frame, imgsz=640,
                    conf=self.get_parameter('confidence').value,
                    device='cpu', verbose=False)[0]
                annotated = self.bridge.cv2_to_imgmsg(result.plot(), 'bgr8')
                annotated.header = msg.header
                self.images.publish(annotated)
                objects = [dict(label=result.names[int(b.cls.item())],
                    confidence=float(b.conf.item()),
                    xyxy=[float(v) for v in b.xyxy[0].tolist()]) for b in result.boxes]
                payload = dict(stamp=dict(sec=msg.header.stamp.sec,
                    nanosec=msg.header.stamp.nanosec), frame_id=msg.header.frame_id,
                    image_width=msg.width, image_height=msg.height,
                    processing_ms=round((time.monotonic()-started)*1000, 1),
                    objects=objects)
                self.detections.publish(String(data=json.dumps(payload)))
            except Exception as exc:
                self.get_logger().error(str(exc))
                self.stop.wait(1)


def main():
    rclpy.init()
    node = SceneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop.set()
        node.worker.join()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
