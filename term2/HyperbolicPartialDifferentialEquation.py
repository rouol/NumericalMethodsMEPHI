import math, sys
import numpy as np


class HyperbolicPartialDifferentialEquation:
    # __slots__ = 'u', 'xs', 'ts', 'is_solved', 'offset_h', 'offset_tau', \
    #     'f', 'g', 'q', 'phi', 'psi', \
    #     'left_coefficients', 'right_coefficients', 'space_interval', \
    #     'T', 'h', 'tau'

    def __init__(self, f, g, q, phi, psi,
        left_coefficients: tuple[float], 
        right_coefficients: tuple[float], 
        space_interval: tuple[float], 
        T: float, h: float, tau: float) -> None:

        self.is_solved = False

        self.left_coefficients = np.array(left_coefficients)
        self.right_coefficients = np.array(right_coefficients)
        self.space_interval = np.array(space_interval)
        self.T = T
        self.h = h
        self.tau = tau

        self.f = f
        self.g = g
        self.q = q
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

        x_t = np.array([self.space_interval[0] + self.h, self.tau])
        next_x_t = np.array([self.space_interval[0] + 2 * self.h, self.tau])
        previous_x_t = np.array([self.space_interval[0], self.tau])

        minimum: float = self.f(x_t[0], x_t[1]) * self.g(x_t[0], x_t[1])
        maximum: float = math.pow(self.f(x_t[0], x_t[1]) * \
            (self.g(next_x_t[0], next_x_t[1]) - self.g(previous_x_t[0], previous_x_t[1]) / 4), 2) \
            + \
            math.pow(self.f(x_t[0], x_t[1]) * self.g(x_t[0], x_t[1]), 2)

        while x_t[1] < self.T:
            while x_t[0] < self.space_interval[1]:
                right_expression: float = self.f(x_t[0], x_t[1]) * self.g(x_t[0], x_t[1])
                left_expression: float = math.pow(self.f(x_t[0], x_t[1]) * \
                    (self.g(next_x_t[0], next_x_t[1]) - self.g(previous_x_t[0], previous_x_t[1]) / 4), 2) \
                    + \
                    math.pow(self.f(x_t[0], x_t[1]) * self.g(x_t[0], x_t[1]), 2)
                minimum = right_expression if right_expression < minimum else minimum
                maximum = left_expression if left_expression > maximum else maximum
                x_t[0] += self.h
                next_x_t[0] += self.h
                previous_x_t[0] += self.h
            x_t[1] += self.tau

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
        self.xs = list()
        self.ts = list()

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

        x_t_0 = np.array([self.space_interval[0], 0])
        next_x_t_0 = np.array([self.space_interval[0] + self.h, 0])
        previous_x_t_0 = np.array([self.space_interval[0] - self.h, 0])

        x_t_1 = np.array([self.space_interval[0], self.tau])
        next_x_t_1 = np.array([self.space_interval[0] + self.h, self.tau])
        previous_x_t_1 = np.array([self.space_interval[0] - self.h, self.tau])

        print(last_three_layers.shape[1], "- elements in row")
        print(math.ceil(self.T / self.tau) + 1, "- elements in col")

        for n in range(0, last_three_layers.shape[1] - 1, 1):
            last_three_layers[0][n] = self.phi(x_t_0[0])
            if n % self.offset_h == 0:
                self.u[0][int(n / self.offset_h)] = last_three_layers[0][n]
                self.xs.append(x_t_0[0])

            last_three_layers[1][n] = self.phi(x_t_1[0]) + self.tau * self.psi(x_t_1[0]) + \
                self.f(x_t_1[0], x_t_1[1]) * self.tau * self.tau / 2 * \
                ((self.g(next_x_t_1[0], next_x_t_1[1]) - self.g(previous_x_t_1[0], previous_x_t_1[1])) / \
                    (2 * self.h) * \
                    (self.phi(next_x_t_1[0]) - self.phi(previous_x_t_1[0])) / \
                    (2 * self.h) \
                    + \
                    self.g(x_t_1[0], x_t_1[1]) * (self.phi(next_x_t_1[0]) - \
                        2 * self.phi(x_t_1[0]) + \
                        self.phi(previous_x_t_1[0])) / (self.h * self.h)) + \
                    self.tau * self.tau / 2 * self.q(x_t_1[0], x_t_1[1])
            
            if ((1 % self.offset_tau == 0 or 1 == math.ceil(self.T / self.tau)) and n % self.offset_h == 0):
                self.u[1][int(n / self.offset_h)] = last_three_layers[1][n]
            
            x_t_0[0] += self.h
            next_x_t_0[0] += self.h
            previous_x_t_0[0] += self.h

            x_t_1[0] += self.h
            next_x_t_1[0] += self.h
            previous_x_t_1[0] += self.h

        x_t_0 = np.array([self.space_interval[1], 0])
        next_x_t_0 = np.array([self.space_interval[1] + self.h, 0])
        previous_x_t_0 = np.array([self.space_interval[1] - self.h, 0])

        x_t_1 = np.array([self.space_interval[1], self.tau])
        next_x_t_1 = np.array([self.space_interval[1] + self.h, self.tau])
        previous_x_t_1 = np.array([self.space_interval[1] - self.h, self.tau])

        last_three_layers[0][last_three_layers.shape[1] - 1] = self.phi(x_t_0[0])
        self.u[0][self.u.shape[1] - 1] = last_three_layers[0][last_three_layers.shape[1] - 1]
        self.xs.append(x_t_0[0])

        last_three_layers[1][last_three_layers.shape[1] - 1] = self.phi(x_t_1[0]) + \
            self.tau * self.psi(x_t_1[0]) + \
            self.f(x_t_1[0], x_t_1[1]) * self.tau * self.tau / 2 * \
            ((self.g(next_x_t_1[0], next_x_t_1[1]) - self.g(previous_x_t_1[0], previous_x_t_1[1])) / \
                (2 * self.h) * \
                (self.phi(next_x_t_1[0]) - self.phi(previous_x_t_1[0])) / \
                (2 * self.h) \
                + \
                self.g(x_t_1[0], x_t_1[1]) * (self.phi(next_x_t_1[0]) - \
                    2 * self.phi(x_t_1[0]) + \
                    self.phi(previous_x_t_1[0])) / (self.h * self.h)) + \
                self.tau * self.tau / 2 * self.q(x_t_1[0], x_t_1[1])
        
        self.ts.append(0)
        
        if (1 % self.offset_tau == 0 or 1 == math.ceil(self.T / self.tau)):
            self.ts.append(self.tau)
            self.u[1][self.u.shape[1] - 1] = last_three_layers[1][last_three_layers.shape[1] - 1]

        for m in range(2, math.ceil(self.T / self.tau), 1):
            x_t = np.array([self.space_interval[0] + self.h, m * self.tau])
            next_x_t = np.array([self.space_interval[0] + 2 * self.h, m * self.tau])
            previous_x_t = np.array([self.space_interval[0], m * self.tau])
            for n in range(1, last_three_layers.shape[1] - 1, 1):
                last_three_layers[2][n] = 2 * last_three_layers[1][n] - last_three_layers[0][n] + \
                    self.tau * self.tau * self.f(x_t[0], x_t[1]) * ( \
                        (self.g(next_x_t[0], next_x_t[1]) - self.g(previous_x_t[0], previous_x_t[1])) * \
                        (last_three_layers[1][n + 1] - last_three_layers[1][n - 1]) / \
                        (4 * self.h * self.h) \
                        + \
                        self.g(x_t[0], x_t[1]) * \
                        (last_three_layers[1][n + 1] - \
                            2 * last_three_layers[1][n] + \
                            last_three_layers[1][n - 1]) / \
                        (self.h * self.h)) + \
                        self.tau*self.tau*self.q(x_t[0], x_t[1])

                if (m % self.offset_tau == 0 and n % self.offset_h == 0):
                    self.u[int(m / self.offset_tau)][int(n / self.offset_h)] = last_three_layers[2][n]
                x_t[0] += self.h
                next_x_t[0] += self.h
                previous_x_t[0] += self.h
            
            last_three_layers[2][0] = self.left_coefficients[0] * \
                (last_three_layers[2][2] - 4 * last_three_layers[2][1]) + \
                2 * self.h * self.left_coefficients[2] / \
                (2 * self.h * self.left_coefficients[1] - \
                    3 * self.left_coefficients[0])

            last_h: float = self.space_interval[1] - x_t[0] + self.h

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

        last_tau: float = self.T - (math.ceil(self.T / self.tau) - 1) * self.tau

        x_t = np.array([self.space_interval[0] + self.h, self.T])
        next_x_t = np.array([self.space_interval[0] + 2 * self.h, self.T])
        previous_x_t = np.array([self.space_interval[0], self.T])

        for n in range(1, last_three_layers.shape[1] - 1, 1):
            last_three_layers[2][n] = (self.tau + last_tau) / self.tau * last_three_layers[1][n] - \
                last_tau / self.tau * last_three_layers[0][n] + \
                last_tau * (last_tau + self.tau) / 2 * self.f(x_t[0], x_t[1]) * ( \
                    (self.g(next_x_t[0], next_x_t[1]) - self.g(previous_x_t[0], previous_x_t[1])) * \
                    (last_three_layers[1][n + 1] - last_three_layers[1][n - 1]) / \
                    (4 * self.h * self.h) \
                    + \
                    self.g(x_t[0], x_t[1]) * \
                    (last_three_layers[1][n + 1] - \
                        2 * last_three_layers[1][n] + \
                        last_three_layers[1][n - 1]) / \
                    (self.h * self.h)) + \
                    self.tau * self.tau * self.q(x_t[0], x_t[1])
            if n % self.offset_h == 0:
                self.u[self.u.shape[0] - 1][int(n / self.offset_h)] = last_three_layers[2][n]
            x_t[0] += self.h
            next_x_t[0] += self.h
            previous_x_t[0] += self.h

        last_three_layers[2][0] = (self.left_coefficients[0] * \
            (last_three_layers[2][2] - 4 * last_three_layers[2][1]) + \
            2 * self.h * self.left_coefficients)[2] / \
            (2 * self.h * self.left_coefficients[1] - \
                3 * self.left_coefficients[0])
        
        last_h: float = self.space_interval[1] - x_t[0] + self.h

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