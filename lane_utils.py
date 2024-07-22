def load_lane_coordinates(path):
    '''
    Description:
    Load the lane coordinates from a file.
    
    Arguments:
        path: str, path to the file containing the lane coordinates

    Returns:
        lane_coordinates: dict, dictionary containing the lane names and coordinates    
    '''
    lane_coordinates = {}
    with open(path, 'r') as f:
        for line in f:
            name, coordinates = line.strip().split(':')
            coordinates = eval(coordinates)
            lane_coordinates[name] = coordinates
    return lane_coordinates