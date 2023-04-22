import random
import matplotlib.pyplot as plt

# https://youtube.com/live/zQFjr_0jZsE?feature=share

class Persepton:

    def __init__(self, learning_rate=0.1) -> None:
        self.input_weight = random.uniform(-1, 1)
        self.bias_weight = random.uniform(-1, 1)
        self.learning_rate = learning_rate
        self.l1 = []
        self.l2 = []

    def predict(self, inputs):
        return inputs * self.input_weight + self.bias_weight

    def fit(self, inputs, expected_outputs, epoch=100):
        for _ in range(epoch):
            for x, y in zip(inputs, expected_outputs):
                output = x * self.input_weight + self.bias_weight  # Calcula a saída do perceptron
                error = y - output  # Calcula o erro
                self.input_weight += self.learning_rate * error * x  # Atualiza o pesos aX
                self.bias_weight += self.learning_rate * error  # Atualiza o peso do bias
                self.l1.append(self.input_weight)
                self.l2.append(self.bias_weight)


# Define os valores de entrada e saída esperada
inputs = [1, 2, 3, 4, 5]
# y = 2x + 3: o aX = 2
# expected_outputs = [2*1+3, 2*2+3, 2*3+3, 2*4+3, 2*5+3]
# expected_outputs = [5, 7, 9, 11, 13]
dX = 4
dC = 3
expected_outputs = [dX*x+dC for x in inputs]

print("Input", inputs, "Outputs", expected_outputs)

neuronio = Persepton()
neuronio.fit(inputs, expected_outputs)
print()
print("aX:", round(neuronio.input_weight,0),
      "biasC", round(neuronio.bias_weight,0))

# second try
input = 9
output_desire = dX * input + dC
output = neuronio.predict(input)
print("my new input", input,
      "predicted output:", round(output, 3),
      "desired ouput:", output_desire,
      "Acuracy:", 100-round(100*abs(output-output_desire)/output_desire, 1))


lx = [x for x in range(len(neuronio.l1))]
plt.plot(lx, neuronio.l1, label='aX')
plt.plot(lx, neuronio.l2, label='C(bias)')
plt.title('new aX e C (bias)')
plt.legend()
plt.show()

