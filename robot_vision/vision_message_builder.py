from humanoid_interfaces.msg import Vision2LocalObserv, VisionDetection, VisionObstacle


class VisionMessageBuilder:
    def __init__(self, distance_estimator):
        self.distance_estimator = distance_estimator

    def build(self, result, distance_context):
        message = Vision2LocalObserv()
        message.header.frame_id = "camera"
        message.status = 0

        position_error = False
        detection_id = 0
        for detection in result.detections:
            class_id = detection["class_id"]
            confidence = detection["confidence"]
            detection_id += 1

            if class_id == 0:
                distance = self.distance_estimator.compute_distance_A(
                    bbox=detection["bbox"],
                    width=self.distance_estimator.ball_width,
                    context=distance_context,
                    # height=self.distance_estimator.ball_height,
                )
                if distance[0] is not None:
                    message.ball = [
                        float(distance[0]),
                        float(distance[1]),
                        confidence,
                    ]
                else:
                    position_error = True
            elif class_id == 5:
                distance = self.distance_estimator.compute_distance_B(
                    bbox=detection["bbox"],
                    context=distance_context,
                )
                if distance[0] is None:
                    position_error = True
                    continue
                detection_message = VisionDetection()
                detection_message.type = 3
                detection_message.detection_id = detection_id
                detection_message.x = float(distance[0])
                detection_message.y = float(distance[1])
                detection_message.confidence = confidence
                message.detections.append(detection_message)

        for intersection in result.intersections:
            distance = self.distance_estimator.compute_distance_B(
                point=intersection["point"],
                context=distance_context,
            )
            if distance[0] is None:
                position_error = True
                continue
            detection_id += 1
            detection_message = VisionDetection()
            detection_message.type = intersection["class_id"] - 2
            detection_message.detection_id = detection_id
            detection_message.x = float(distance[0])
            detection_message.y = float(distance[1])
            detection_message.confidence = intersection["confidence"]
            message.detections.append(detection_message)

        for x, y, width, height, obstacle_id in result.obstacle_bboxes:
            distance = self.distance_estimator.compute_distance_A(
                bbox=(x, y, x + width, y + height),
                width=self.distance_estimator.cone_width,
                height=self.distance_estimator.cone_height,
                context=distance_context,
            )
            if distance[0] is None:
                position_error = True
                continue
            obstacle = VisionObstacle()
            obstacle.detection_id = obstacle_id
            obstacle.x = float(distance[0])
            obstacle.y = float(distance[1])
            obstacle.confidence = 1.0
            message.obstacles.append(obstacle)

        if position_error:
            message.status = 4
        elif not message.detections and not any(message.ball) and not message.obstacles:
            message.status = 1

        # ros2 topic echo /vision_packages
        return message
