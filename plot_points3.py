import random
import numpy as np
import time
from os.path import dirname, abspath
import sys

# Import get_world_name from tester44.py
sys.path.append(dirname(dirname(abspath(__file__))))
from actor import get_world_name


class PositionChecker:
    def __init__(self, world='world_15'):
        # Sanitize the world name by removing prefixes and suffixes
        world = world.replace("BARN/", "").replace(".world", "")
        self.invalid_zones = self.set_invalid_zones(world)

    def set_invalid_zones(self, world):
        if world == 'world_0':
            return [19, 21, 24, 26, 28, 35, 37, 40, 42, 44, 51, 53, 56, 58, 60, 62, 67, 69, 72, 74, 76, 83, 85, 88, 90, 92, 95, 99, 101, 103, 104, 106, 108, 115, 117, 120, 122, 124, 131, 133, 136, 138, 140, 147, 149, 152, 154, 156, 163, 165, 168, 170, 172, 174, 179, 181, 182, 184, 186, 188, 195, 200, 202, 204, 223]
        elif world == 'world_1':
            return [23, 35, 38, 46, 52, 57, 61, 69, 76, 82, 86, 91, 103, 106, 111, 117, 125, 130, 140, 151, 154, 166, 171, 175, 178, 181, 188, 196, 205, 211, 214, 222, 234]
        elif world == 'world_2':
            return [5, 21, 32, 37, 48, 53, 56, 59, 62, 64, 80, 82, 86, 94, 98, 102, 104, 107, 110, 114, 118, 126, 130, 134, 142, 148, 152, 156, 164, 168, 170, 172, 180, 184, 188, 190, 196, 200, 204, 207, 214, 218, 230, 234, 246, 250]
        elif world == 'world_3':
            return [22, 29, 39, 56, 58, 60, 62, 73, 83, 86, 90, 93, 95, 100, 107, 117, 121, 124, 130, 134, 141, 147, 151, 153, 158, 164, 168, 178, 181, 185, 190, 195, 198, 202, 207, 212]
        elif world == 'world_4':
            return [23, 27, 34, 47, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 83, 92, 99, 102, 103, 104, 105, 106, 108, 110, 115, 122, 124, 131, 134, 138, 140, 143, 147, 150, 151, 152, 153, 154, 156, 163, 179, 188, 190, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 207, 229, 233]
        elif world == 'world_5':
            return [35, 38, 42, 46, 51, 54, 55, 58, 67, 70, 72, 74, 76, 79, 83, 86, 90, 99, 100, 102, 106, 108, 116, 119, 124, 126, 132, 135, 140, 148, 151, 154, 156, 164, 167, 169, 172, 180, 183, 188, 191, 196, 199, 204, 212, 215, 220]
        elif world == 'world_6':
            return [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 47, 50, 63, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 79, 92, 95, 99, 108, 111, 124, 127, 131, 132, 133, 134, 135, 136, 137, 138, 140, 143, 154, 156, 159, 165, 168, 170, 172, 175, 186, 188, 191, 194, 196, 202, 204, 207, 216, 218, 220, 223, 229, 236, 239]
        elif world == 'world_7':
            return [29, 36, 41, 45, 50, 71, 84, 91, 95, 111, 119, 135, 138, 141, 148, 154, 157, 162, 164, 180, 184, 207, 211, 214, 220, 223, 236]
        elif world == 'world_8':
            return [35, 40, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 68, 77, 84, 88, 89, 90, 93, 100, 103, 107, 109, 110, 116, 119, 123, 125, 130, 132, 135, 139, 143, 147, 148, 151, 155, 157, 164, 168, 169, 170, 171, 173, 180, 189, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 229, 233, 238]
        elif world == 'world_9':
            return [21, 28, 34, 38, 43, 55, 58, 66, 78, 79, 83, 86, 91, 94, 100, 103, 106, 109, 148, 151, 154, 157, 163, 166, 171, 174, 178, 188, 191, 199, 202, 210, 214, 219, 229, 236]
        elif world == 'world_10':
            return [35, 38, 41, 44, 52, 55, 58, 61, 66, 72, 78, 83, 89, 95, 100, 106, 117, 123, 134, 140, 146, 151, 157, 163, 168, 174, 180, 185, 191, 197, 202, 214, 219, 221, 226, 231, 238]
        elif world == 'world_11':
            return [19, 22, 34, 38, 40, 54, 56, 58, 59, 61, 62, 70, 74, 78, 82, 83, 84, 85, 86, 90, 94, 106, 107, 109, 110, 147, 148, 150, 151, 163, 167, 171, 172, 173, 174, 175, 179, 183, 187, 195, 196, 198, 199, 203, 217, 219, 222, 233, 235]
        elif world == 'world_12':
            return [18, 22, 31, 52, 55, 58, 61, 66, 79, 100, 103, 106, 109, 114, 127, 148, 151, 154, 157, 162, 175, 196, 199, 202, 205, 210, 223, 232, 236]
        elif world == 'world_13':
            return [34, 38, 46, 51, 53, 59, 68, 74, 76, 89, 93, 104, 110, 116, 119, 127, 132, 135, 152, 158, 169, 173, 180, 186, 188, 195, 197, 203, 210, 214, 223]
        elif world == 'world_14':
            return [37, 39, 42, 47, 52, 57, 62, 67, 72, 75, 77, 83, 88, 93, 105, 110, 117, 122, 127, 129, 131, 133, 134, 136, 138, 139, 141, 143, 158, 163, 168, 173, 179, 181, 184, 189, 196, 201, 206, 213, 218, 223, 237]
        elif world == 'world_15':
            return [21, 25, 35, 38, 40, 46, 55, 61, 66, 72, 76, 83, 89, 91, 100, 106, 111, 117, 120, 123, 126, 130, 134, 140, 151, 157, 166, 168, 174, 181, 185, 191, 196, 202, 211, 216, 219, 222, 226, 231, 236]
        else:
            print(f"[WARN] Unknown world: {world}. No invalid zones set.")
            return []




    def is_valid_position(self, x, y, margin=0.5):
        """Check if the position is valid (not in invalid zones or too close to them)."""
        for zone in self.invalid_zones:
            zone_x, zone_y = self.zone_to_coordinates(zone)
            if (zone_x - margin < x < zone_x + 1 + margin) and (zone_y - margin < y < zone_y + 1 + margin):
                return False
        return True

    def zone_to_coordinates(self, zone):
        """Convert a zone number to its corresponding (x, y) coordinates."""
        col = (zone - 1) % 16
        row = 15 - (zone - 1) // 16
        x = col - 8
        y = row - 8
        return x, y

    def generate_random_position(self, size=16):
        """Generate a random position within the scenario dimensions."""
        half_size = size / 2
        while True:
            x = random.uniform(-half_size, half_size)
            y = random.uniform(-half_size, half_size)
            angle = random.uniform(-np.pi, np.pi)
            if self.is_valid_position(x, y):
                return [x, y, angle]

    def distance(self, point1, point2):
        """Calculate the Euclidean distance between two points."""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def main():
    world_name = get_world_name()
    # Sanitize the world name by removing prefixes and suffixes
    world_name = world_name.replace("BARN/", "").replace(".world", "")

    position_checker = PositionChecker(world=world_name)

    try:
        while True:
            start_point = position_checker.generate_random_position()
            goal_point = position_checker.generate_random_position()

            while position_checker.distance(start_point, goal_point) < 8:
                goal_point = position_checker.generate_random_position()

            #print(f"Start Point: {start_point[:2]}")
            #print(f"Goal Point: {goal_point[:2]}")
            time.sleep(1)  # Sleep for 1 second before generating new points

    except KeyboardInterrupt:
        print("Process interrupted")

if __name__ == "__main__":
    main()

