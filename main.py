import cv2
import numpy as np
import random
import math
from cvzone.HandTrackingModule import HandDetector
import cvzone

# Let's get the camera running
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Time to detect some hands
detector = HandDetector(detectionCon=0.8, maxHands=1)

# The Snake Game
class SnakeGame:
    def __init__(self, food_image_path):
        # Game variables
        self.points = []
        self.lengths = []
        self.current_length = 0
        self.allowed_length = 150
        self.previous_head = 0, 0
        self.food_image = cv2.imread(food_image_path, cv2.IMREAD_UNCHANGED)
        self.food_height, self.food_width, _ = self.food_image.shape
        self.food_point = 0, 0
        self.random_food_location()
        self.score = 0
        self.game_over = False

    def random_food_location(self):
        # Move the food to a random spot
        self.food_point = random.randint(100, 1000), random.randint(100, 600)

    def update(self, img_main, current_head):
        # Update the game state
        if self.game_over:
            cvzone.putTextRect(img_main, "Game Over", [300, 400], scale=7, thickness=5, offset=20)
            cvzone.putTextRect(img_main, f'Your Score: {self.score}', [300, 550], scale=7, thickness=5, offset=20)
        else:
            px, py = self.previous_head
            cx, cy = current_head

            self.points.append([cx, cy])
            distance = math.hypot(cx - px, cy - py)
            self.lengths.append(distance)
            self.current_length += distance
            self.previous_head = cx, cy

            if self.current_length > self.allowed_length:
                while self.current_length > self.allowed_length:
                    self.current_length -= self.lengths.pop(0)
                    self.points.pop(0)

            rx, ry = self.food_point
            if rx - self.food_width // 2 < cx < rx + self.food_width // 2 and \
                    ry - self.food_height // 2 < cy < ry + self.food_height // 2:
                self.random_food_location()
                self.allowed_length += 50
                self.score += 1

            if self.points:
                for i, point in enumerate(self.points):
                    if i != 0:
                        cv2.line(img_main, tuple(self.points[i - 1]), tuple(self.points[i]), (0, 0, 0), 20)
                cv2.circle(img_main, tuple(self.points[-1]), 20, (0, 255, 0), cv2.FILLED)

            img_main = cvzone.overlayPNG(img_main, self.food_image, (rx - self.food_width // 2, ry - self.food_height // 2))

            cvzone.putTextRect(img_main, f'Score: {self.score}', [50, 80], scale=3, thickness=3, offset=10)

            if len(self.points) > 2:
                pts = np.array(self.points[:-2], np.int32)
                pts = pts.reshape((-1, 1, 2))
                min_dist = cv2.pointPolygonTest(pts, (cx, cy), True)
                if -1 <= min_dist <= 1:
                    print("Hit")
                    self.game_over = True

        return img_main

# Let's start the game
game = SnakeGame("image/apple.png")

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=False)

    if hands:
        lm_list = hands[0]['lmList']
        point_index = lm_list[8][0:2]
        img = game.update(img, point_index)

    cv2.imshow("Snake Game", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        game.game_over = False


