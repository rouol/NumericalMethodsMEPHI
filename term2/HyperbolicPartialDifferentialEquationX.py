import math, sys
import numpy as np


class HyperbolicPartialDifferentialEquation:

    u = None
    xs = list()
    ts = list()
    is_solved = False

    def __init__(self, f, g, phi, psi,
        left_coefficients: tuple[float], 
        right_coefficients: tuple[float], 
        space_interval: tuple[float], 
        T: float, h: float, tau: float) -> None:

        self.left_coefficients = np.array(left_coefficients)
        self.right_coefficients = np.array(right_coefficients)
        self.space_interval = np.array(space_interval)
        self.T = T
        self.h = h
        self.tau = tau

        # parse function strings
        self.f = f
        self.g = g
        self.phi = phi
        self.psi = psi

        store_h: float = self.h
        store_tau: float = self.tau

        if (abs(2 * self.h * self.left_coefficients[1] \
            - \
            3 * self.left_coefficients[0]) < sys.float_info.epsilon):
            self.h /= 2
        if (abs(2 * self.h * self.right_coefficients[1]
            +
            3 * self.right_coefficients[0]) < sys.float_info.epsilon):
            self.h /= 2

        if (self.h < sys.float_info.epsilon):
            raise(Exception("Very small step"))

        first_x: float = self.space_interval[0] + self.h
        next_first_x: float = self.space_interval[0] + 2*self.h
        previous_first_x: float = self.space_interval[0]
        minimum: float = self.f(first_x) * self.g(first_x)
        maximum: float = math.pow(self.f(first_x) * \
            (self.g(next_first_x) - self.g(previous_first_x) / 4), 2) \
            + \
            math.pow(self.f(first_x) * self.g(first_x), 2)
        
        for x in np.arange(self.space_interval[0] + 2*self.h,
            self.space_interval[1],
            self.h):
            next_x: float = x + self.h
            previous_x: float = x - self.h
            right_expression: float = self.f(x) * self.g(x)
            left_expression: float = math.pow(self.f(x) * \
                (self.g(next_x) - self.g(previous_x) / 4), 2) \
                + \
                math.pow(self.f(x) * self.g(x), 2)
            minimum = right_expression if right_expression < minimum else minimum
            maximum = left_expression if left_expression > maximum else maximum

        if (maximum < sys.float_info.epsilon):
            self.tau = self.h
        else:
            if (self.tau * self.tau >= minimum / maximum * self.h * self.h):
                if (minimum / maximum > 0):
                    self.tau = math.sqrt(minimum / maximum) * self.h / 2
                else:
                    raise Exception("This problem is not solved by this scheme.")


        if (self.tau < sys.float_info.epsilon):
            raise Exception("Very small step")

        self.offset_h = int(store_h / self.h)
        self.offset_tau = int(store_tau / self.tau)

        self.u = np.zeros((math.ceil(self.T / (self.offset_tau * self.tau)) + 1,
            math.ceil((self.space_interval[1] - self.space_interval[0]) / (self.offset_h * self.h)) + 1))

        print(self.u.shape[1], "x", self.u.shape[0])
        if (self.u.shape[1] < 2 or self.u.shape[0] < 2):
            raise Exception("Very big step")

        self.is_solved = False

    def Solution(self) -> np.array:
        if not self.is_solved:
            self.Solve()
        return self.u
    
    def GetXs(self) -> list:
        if not self.is_solved:
            self.Solve()
        return self.xs

    def GetTs(self) -> list:
        if not self.is_solved:
            self.Solve()
        return self.ts
    
    def Solve(self) -> None:
        last_three_layers = np.zeros((3, math.ceil((self.space_interval[1] - self.space_interval[0]) / self.h) + 1))
        x: float = self.space_interval[0]
        next_x: float = self.space_interval[0] + self.h
        previous_x: float = self.space_interval[0] - self.h
        print(last_three_layers.shape[1], "- elements in row")
        print(math.ceil(self.T / self.tau) + 1, "- elements in col")
        for n in range(0, last_three_layers.shape[1] - 1, 1):
            last_three_layers[0][n] = self.phi(x)
            if n % self.offset_h == 0:
                self.u[0][int(int(n / self.offset_h))] = last_three_layers[0][n]
                self.xs.append(x)
            
            last_three_layers[1][n] = self.phi(x) + self.tau * self.psi(x) + \
                self.f(x) * self.tau * self.tau / 2 * \
                ((self.g(next_x) - self.g(previous_x)) / \
                    (2 * self.h) * \
                    (self.phi(next_x) - self.phi(previous_x)) / \
                    (2 * self.h) \
                    + \
                    self.g(x) * (self.phi(next_x) - \
                        2 * self.phi(x) + \
                        self.phi(previous_x)) / (self.h * self.h))
            
            if ((1 % self.offset_tau == 0 or 1 == math.ceil(self.T / self.tau)) and n % self.offset_h == 0):
                self.u[1][int(int(n / self.offset_h))] = last_three_layers[1][n]
            x += self.h
            next_x += self.h
            previous_x += self.h

        x = self.space_interval[1]
        next_x = self.space_interval[1] + self.h
        previous_x = self.space_interval[1] - self.h
        last_three_layers[0][last_three_layers.shape[1] - 1] = self.phi(x)
        self.u[0][self.u.shape[1] - 1] = last_three_layers[0][last_three_layers.shape[1] - 1]
        self.xs.append(x)
        last_three_layers[1][last_three_layers.shape[1] - 1] = self.phi(x) + self.tau * self.psi(x) + \
            self.f(x) * self.tau * self.tau / 2 * ((self.g(next_x) - self.g(previous_x)) / \
                (2 * self.h) * (self.phi(next_x) - self.phi(previous_x)) / \
                (2 * self.h) + self.g(x) * (self.phi(next_x) - 2 * self.phi(x) + \
                self.phi(previous_x)) / (self.h * self.h))
        self.ts.append(0)
        if (1 % self.offset_tau == 0 or 1 == math.ceil(self.T / self.tau)):
            self.ts.append(self.tau)
            self.u[1][self.u.shape[1] - 1] = last_three_layers[1][last_three_layers.shape[1] - 1]

        for m in range(2, math.ceil(self.T / self.tau), 1):
            x = self.space_interval[0] + self.h
            next_x = self.space_interval[0] + 2 * self.h
            previous_x = self.space_interval[0]
            for n in range(1, last_three_layers.shape[1] - 1, 1):
                last_three_layers[2][n] = 2 * last_three_layers[1][n] - last_three_layers[0][n] + \
                        self.tau * self.tau * self.f(x) * ( \
                        (self.g(next_x) - self.g(previous_x)) * \
                        (last_three_layers[1][n + 1] - last_three_layers[1][n - 1]) / \
                        (4 * self.h * self.h) \
                        + \
                        self.g(x) * \
                        (last_three_layers[1][n + 1] - \
                            2 * last_three_layers[1][n] + \
                            last_three_layers[1][n - 1]) / \
                        (self.h * self.h))
                if (m % self.offset_tau == 0 and n % self.offset_h == 0):
                    self.u[int(m / self.offset_tau)][int(n / self.offset_h)] = last_three_layers[2][n]
                x += self.h
                next_x += self.h
                previous_x += self.h
            
            last_three_layers[2][0] = self.left_coefficients[0] * \
                (last_three_layers[2][2] - 4 * last_three_layers[2][1]) + \
                2 * self.h * self.left_coefficients[2] / \
                (2 * self.h * self.left_coefficients[1] - \
                    3 * self.left_coefficients[0])
            last_h: float = self.space_interval[1] - x + self.h
            last_three_layers[2][last_three_layers.shape[1] - 1] = (self.right_coefficients[0] * \
                ((self.h + last_h) / (self.h * last_h) * last_three_layers[2][last_three_layers.shape[1] - 2] - \
                    (last_h / (self.h * (self.h + last_h))) * last_three_layers[2][last_three_layers.shape[1] - 3] + \
                self.right_coefficients[2]) / \
                (self.right_coefficients[1] + \
                    (2 * last_h + self.h) / (last_h * (self.h + last_h)) * self.right_coefficients[0]))
            if m % self.offset_tau == 0:
                self.u[int(m / self.offset_tau)][0] = last_three_layers[2][0]
                self.u[int(m / self.offset_tau)][self.u.shape[1] - 1] = last_three_layers[2][last_three_layers.shape[1] - 1]
                self.ts.append(m * self.tau)

            last_three_layers[0] = last_three_layers[1]
            last_three_layers[1] = last_three_layers[2]
            last_three_layers[2] = np.zeros((math.ceil((self.space_interval[1] - self.space_interval[0]) / self.h) + 1))

        #m == std::ceil(self.T / self.tau) :
        last_tau: float = self.T - (math.ceil(self.T / self.tau) - 1) * self.tau
        x = self.space_interval[0] + self.h
        next_x = self.space_interval[0] + 2 * self.h
        previous_x = self.space_interval[0]
        for n in range(1, last_three_layers.shape[1] - 1, 1):
            last_three_layers[2][n] = (self.tau + last_tau) / self.tau * last_three_layers[1][n] - \
                last_tau / self.tau * last_three_layers[0][n] + \
                last_tau * (last_tau + self.tau) / 2 * self.f(x) * ( \
                    (self.g(next_x) - self.g(previous_x)) * \
                    (last_three_layers[1][n + 1] - last_three_layers[1][n - 1]) / \
                    (4 * self.h * self.h) \
                    + \
                    self.g(x) * \
                    (last_three_layers[1][n + 1] - \
                        2 * last_three_layers[1][n] + \
                        last_three_layers[1][n - 1]) / \
                    (self.h * self.h))
            if n % self.offset_h == 0:
                self.u[self.u.shape[0] - 1][int(n / self.offset_h)] = last_three_layers[2][n]
            x += self.h
            next_x += self.h
            previous_x += self.h

        last_three_layers[2][0] = (self.left_coefficients[0] * \
            (last_three_layers[2][2] - 4 * last_three_layers[2][1]) + \
            2 * self.h * self.left_coefficients)[2] / \
            (2 * self.h * self.left_coefficients[1] - \
                3 * self.left_coefficients[0])
        last_h: float = self.space_interval[1] - x + self.h
        last_three_layers[2][last_three_layers.shape[1] - 1] = (self.right_coefficients[0] * \
            ((self.h + last_h) / (self.h * last_h) * last_three_layers[2][last_three_layers.shape[1] - 2] - \
                (last_h / (self.h * (self.h + last_h))) * last_three_layers[2][last_three_layers.shape[1] - 3]) + \
            self.right_coefficients[2]) / \
            (self.right_coefficients[1] + \
                (2 * last_h + self.h) / (last_h * (self.h + last_h)) * self.right_coefficients[0])
        self.u[self.u.shape[0] - 1][0] = last_three_layers[2][0]
        self.u[self.u.shape[0] - 1][self.u.shape[1] - 1] = last_three_layers[2][last_three_layers.shape[1] - 1]
        self.ts.append(self.T)
        self.is_solved = True