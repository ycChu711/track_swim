import cv2
import numpy as np
from shapely.geometry import Point, Polygon
import os


# extract video frame
def extract_frame(video_path, frame_number):
    '''
    video_path: str, path to the video file
    frame_number: int, frame number to extract
    '''
    cap = cv2.VideoCapture(video_path)
    for _ in range(frame_number):
        _, frame = cap.read()
    cv2.imwrite(f'{frame_number}_frame.png', frame)
    cap.release()

# Define the mouse callback function to capture the coordinates
coordinates = []
named_coordinates = {}

def get_coordinates(event, x, y, flags, param):
    '''
    event: int, type of mouse event
    x: int, x-coordinate of the mouse event
    y: int, y-coordinate of the mouse event
    flags: int, any flags passed with the event
    param: any parameters passed with the event
    '''
    if event == cv2.EVENT_LBUTTONDOWN:  # Check if the left mouse button was clicked
        print(f"Coordinates: ({x}, {y})")
        # Store the coordinates in the global list
        coordinates.append((x, y))

def save_named_coordinates(name):
    '''
    name: str, name of the selected coordinates
    '''
    global named_coordinates
    named_coordinates[name] = coordinates.copy()
    coordinates.clear()  # Clear the current list for new selections


# use extracted frame to manually select coordinates for lane detection
def get_lane_coordinates(video_path, frame_number):
    '''
    video_path: str, path to the video file
    frame_number: int, frame number to extract
    '''
    extract_frame(video_path, frame_number)
    image_path = f'{frame_number}_frame.png'
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error loading image.")
    else:
        # Display the frame
        cv2.imshow("Frame", frame)

        # Set the mouse callback function to capture the coordinates
        cv2.setMouseCallback("Frame", get_coordinates)

        # Main loop
        print("Click on the frame to get coordinates.")
        print("Press 's' to save selected coordinates with a name.")
        print("Press 'q' to exit the selection process.")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Exit the loop
                break
            elif key == ord('s'):  # Save the selected coordinates
                name = input("Enter a name for the selected coordinates: ")
                save_named_coordinates(name)
                print(f"Saved coordinates: {named_coordinates}")

        cv2.destroyAllWindows()

    # Create a file name based on the video name and save the coordinates
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join('lane_identification', f'{video_name}_lane_coordinates.txt')
    
    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        for name, coordinates in named_coordinates.items():
            f.write(f'{name}: {coordinates}\n')
    
    print(f"Lane coordinates saved to {save_path}")


# Convert the selected coordinates in a dictionary to a dictionary of polygons
def convert_coordinates_to_polygons(coordinates_dict):
    '''
    coordinates_dict: dict, dictionary of areas with their coordinates
    '''
    polygons = {}
    for name, coordinates in coordinates_dict.items():
        polygons[name] = Polygon(coordinates)
    return polygons

# Assign objects to areas based on their coordinates
def assign_objects_to_areas(center_x, center_y, areas):
    '''
    center_x: int, x-coordinate of the object
    center_y: int, y-coordinate of the object
    areas: dict, dictionary of areas with their polygons
    '''
    point = Point(center_x, center_y)
    for area_name, polygon in areas.items():
        if polygon.contains(point):
            return area_name
    return None


# Run the lane identification process
if __name__ == '__main__':
    video_path = 'test_video/test_pool_trimmed_2min.mp4'
    frame_number = 100
    get_lane_coordinates(video_path, frame_number)