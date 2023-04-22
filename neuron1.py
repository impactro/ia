# see video https://youtube.com/live/zQFjr_0jZsE?feature=share

x = 1
a = 2
c = 3

y = a*x + c
# y = 2*1 + 3 => 2 + 3 => 5
# 5 = a*1 + 3
# a = -5+3 => 2

input = 5
output_desire = 50 # y

weight = 0.5
bias = 0.5
learning_rate = 0.01

n = 0 
l1 = []
l2 = []
while True:
    n += 1
    output = input * weight + bias
    error = output_desire - output

    if round(error,3)==0:
        print("OK Found")
        break

    weight += learning_rate * input * error
    bias += learning_rate * error

    l1.append(weight)
    l2.append(bias)

    print("iteration:", n, 
          "weight:", round(weight, 3), 
          "bias:", round(bias,3),
          "error:", round(error,3))
    

import matplotlib.pyplot as plt
lx = [x for x in range(len(l1))]
plt.plot(lx, l1, label='aX')
plt.plot(lx, l2, label='C(bias)')
plt.title('aX and C (bias)')
plt.legend()
plt.show()


