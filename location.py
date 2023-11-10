import numpy as np
import numpy.typing as npt


def solved_location() -> npt.NDArray:
    return np.array([
        [[1, 2],
        [3, 4]],
        [[5, 6],
        [7, 8]],
    ], dtype=np.uint8)


def location_diff() -> npt.NDArray:
    return np.array([
        [0, 1, 1, 2, 1, 2, 2, 3],
        [1, 0, 2, 1, 2, 1, 3, 2],
        [1, 2, 0, 1, 2, 3, 1, 2],
        [2, 1, 1, 0, 3, 2, 2, 1],
        [1, 2, 2, 3, 0, 1, 1, 2],
        [2, 1, 3, 2, 1, 0, 2, 1],
        [2, 3, 1, 2, 1, 2, 0, 1],
        [3, 2, 2, 1, 2, 1, 1, 0],
    ], dtype=np.uint8)


def next_location(location, action):

    location = np.copy(location)

    # left to up
    if action == 1:
        location[:, :, 0] = np.rot90(location[:, :, 0], 1)

    # right to up
    elif action == 2:
        location[:, :, 1] = np.rot90(location[:, :, 1], 1)

    # down to left
    elif action == 3:
        location[:, 1, :] = np.rot90(location[:, 1, :], 1)

    # up to left
    elif action == 4:
        location[:, 0, :] = np.rot90(location[:, 0, :], 1)

    # back to right
    elif action == 5:
        location[1, :, :] = np.rot90(location[1, :, :], -1)

    # front to right
    elif action == 6:
        location[0, :, :] = np.rot90(location[0, :, :], -1)

    # left to down
    if action == 7:
        location[:, :, 0] = np.rot90(location[:, :, 0], -1)

    # right to down
    if action == 8:
        location[:, :, 1] = np.rot90(location[:, :, 1], -1)
        
    # down to right
    elif action == 9:
        location[:, 1, :] = np.rot90(location[:, 1, :], -1)

    # up to right
    elif action == 10:
        location[:, 0, :] = np.rot90(location[:, 0, :], -1)

    # back to left
    elif action == 11:
        location[1, :, :] = np.rot90(location[1, :, :], 1)

    # front to left
    elif action == 12:
        location[0, :, :] = np.rot90(location[0, :, :], 1)

    return location


if __name__ == '__main__':
    initial_location = solved_location()
    print('intial location:')
    print(initial_location)
    print()
    child_location = next_location(initial_location, action=4)
    print('next location:')
    print(child_location)