import matplotlib.pyplot as plt


def draw_hand(landmarks):
    landmarks.squeeze()
    connections = []
    connections.append(landmarks[[0, 1, 2, 3, 4], :])
    connections.append(landmarks[[0, 5, 6, 7, 8], :])
    connections.append(landmarks[[0, 17, 18, 19, 20], :])
    connections.append(landmarks[[9, 10, 11, 12], :])
    connections.append(landmarks[[13, 14, 15, 16], :])
    connections.append(landmarks[[5, 9, 13, 17], :])
    connections = [[t[:, 0], t[:, 1]] for t in connections]
    connections = [x for sub in connections for x in sub]
    return plt.plot(*connections, marker='o')
