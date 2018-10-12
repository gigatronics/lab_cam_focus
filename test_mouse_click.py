from pylab import *
import sys
from numpy import *
from matplotlib import pyplot as plt

class Test:

    def __init__(self, pts_list):
        self.pts = pts_list
        self.fig = plt.get_current_fig_manager().canvas.figure
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

  # def __call__(self,event):
  #   if event.inaxes:
  #     print("Inside drawing area!")
  #     print("x: ", event.x)
  #     print("y: ", event.y)
  #   else:
  #     print("Outside drawing area!")
    def on_click(self, event):
        print("(x, y) = %d, %d" % (event.x, event.y))
        pts_list.append([event.x, event.y])
        print(pts_list)

    def on_key(self, event):
        if event.key =='enter':
            self.fig.canvas.mpl_disconnect(self.on_click)
            return


def onclick(event):

    print("(x, y) = %d, %d" % (event.x, event.y))
    pts_list.append([event.x, event.y])
    print(pts_list)

def handle_close(evt):
    print('Closed Figure!')


if __name__ == '__main__':
# # UNIT TEST: onclick
#     global pts_list
#     pts_list = []
#
#     fig, ax = plt.subplots()
#     ax.plot(np.random.rand(10))
#     cid = fig.canvas.mpl_connect('button_press_event', onclick) # cid = call id
#     plt.show()
#
#
# # UNIT TEST: test handle_close
#     fig = plt.figure()
#     fig.canvas.mpl_connect('close_event', handle_close)
#
#     plt.text(0.35, 0.5, 'Close Me!', dict(size=30))
#     plt.show()
#

# UNIT TEST: Test class
    pts_list = []
    #fig = plt.figure()
    fig, ax = plt.subplots()
    ax.plot(np.random.rand(10))

    test = Test(pts_list)
    plt.show()
