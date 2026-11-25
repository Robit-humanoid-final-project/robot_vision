import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_srvs.srv import Trigger

from humanoid_interfaces.msg import VSCommand
from insta360_usb_cam.msg import InstaPanTiltMsgs


class PanTiltNode(Node):
    def __init__(self):
        super().__init__("pan_tilt_node")

        self.declare_parameter("camera_tilt_deg", 0.0)
        self.declare_parameter("camera_kick_tilt_deg", 70.0)
        self.declare_parameter("pan_sweep_interval", 0.03)
        self.declare_parameter("pan_sweep_step_deg", 2.0)

        camera_tilt_deg = self.get_parameter("camera_tilt_deg").value
        camera_kick_tilt_deg = self.get_parameter("camera_kick_tilt_deg").value
        pan_sweep_step_deg = self.get_parameter("pan_sweep_step_deg").value
        self.default_tilt = round(-camera_tilt_deg * 3600.0)
        self.kick_tilt = round(-camera_kick_tilt_deg * 3600.0)
        self.tilt = self.default_tilt
        self.pan_min = -60 * 3600
        self.pan_max = 60 * 3600
        self.pan_step = round(pan_sweep_step_deg * 3600.0)
        self.pan = 0
        self.pan_sweep_phase = "to_left"
        self.sweep_active = False
        self.sweep_continuous = False

        self.publisher = self.create_publisher(
            InstaPanTiltMsgs,
            "/camera1/pan_tilt",
            10,
        )
        self.vision_command_subscription = self.create_subscription(
            VSCommand,
            "vision_command",
            self.vision_command_callback,
            10,
        )
        self.sweep_service = self.create_service(
            Trigger,
            "/start_pan_sweep",
            self.start_pan_sweep,
        )
        self.timer = self.create_timer(
            self.get_parameter("pan_sweep_interval").value,
            self.publish_pan_tilt_command,
        )

    def start_pan_sweep(self, _request, response):
        self.pan = 0
        self.pan_sweep_phase = "to_left"
        self.sweep_active = True
        self.sweep_continuous = False
        response.success = True
        response.message = "Pan sweep started"
        return response

    def vision_command_callback(self, message):
        # if message.vision_mode == 1:
        #     if not self.sweep_continuous or not self.sweep_active:
        #         self.pan = 0
        #         self.pan_sweep_phase = "to_left"
        #     self.tilt = self.default_tilt
        #     self.sweep_active = True
        #     self.sweep_continuous = True
        if message.vision_mode == 2:
            self.pan = 0
            self.tilt = self.kick_tilt
            self.sweep_active = False
            self.sweep_continuous = False
        else:
            self.pan = 0
            self.tilt = self.default_tilt
            self.sweep_active = False
            self.sweep_continuous = False

    def publish_pan_tilt_command(self):
        message = InstaPanTiltMsgs()
        message.pan = self.pan
        message.tilt = self.tilt
        self.publisher.publish(message)

        if not self.sweep_active:
            return

        if self.pan_sweep_phase == "to_left":
            if self.pan <= self.pan_min:
                self.pan_sweep_phase = "to_right"
                self.pan = min(self.pan + self.pan_step, self.pan_max)
            else:
                self.pan = max(self.pan - self.pan_step, self.pan_min)
        elif self.pan_sweep_phase == "to_right":
            if self.pan >= self.pan_max:
                if self.sweep_continuous:
                    self.pan_sweep_phase = "to_left"
                    self.pan = max(self.pan - self.pan_step, self.pan_min)
                else:
                    self.pan_sweep_phase = "to_center"
                    self.pan = max(self.pan - self.pan_step, 0)
            else:
                self.pan = min(self.pan + self.pan_step, self.pan_max)
        else:
            if self.pan <= 0:
                self.pan = 0
                self.sweep_active = False
            else:
                self.pan = max(self.pan - self.pan_step, 0)


def main(args=None):
    rclpy.init(args=args)
    node = PanTiltNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
