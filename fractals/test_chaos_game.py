'''
Script for playing the Chaos game
'''
import numpy as np
import matplotlib.pylab as plt

n = 3

#generate n points on a unit circle
r = np.arange(0,n)
points = np.exp(2.0*np.pi*1j*r/n)
print(points)

#plot points
res = 100
w = np.arange(0,res)
circle_points = np.exp(2.0*np.pi*1j*w/res)

#starting point 
start = 0.1+0.5j

#play the game
select = np.random.randint(0,n)
print(points[select])

#new point
new_point = points[select] - start
new_point += start

#plot it

#full algorithm
def compute(startloc):
    '''
    compute new position for game
    '''
    randloc = np.random.randint(0,n)
    new_point = (points[randloc] - startloc)/2.0
    new_point += startloc
    return new_point, points[randloc]

p1, rloc = compute(start)

# plt.plot(np.real(circle_points), np.imag(circle_points), "b-")
plt.plot(np.real(points), np.imag(points), "r.")
# plt.plot(np.real(start), np.imag(start), "g.")

#run the game
iterations = 10000

next_point = start
for i in range(iterations):
    next_point, rloc = compute(next_point)
    plt.plot(np.real(next_point), np.imag(next_point), "b.")

plt.show()
