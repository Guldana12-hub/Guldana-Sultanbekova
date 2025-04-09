#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 18:37:28 2025

@author: guldanasultanbekova
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- ФИЗИЧЕСКИЕ КОНСТАНТЫ ---
G = 6.67430e-11            # Гравитационная постоянная (м³·кг⁻¹·с⁻²)
AU = 1.496e11              # Астрономическая единица (м)
day = 86400               # День в секундах
Msun = 1.98847e30         # Масса Солнца (кг)

# --- ПАРАМЕТРЫ СИСТЕМЫ SPICA ---
m1 = 10.25 * Msun         # Масса Spica A
m2 = 6.97 * Msun          # Масса Spica B
a = 0.12 * AU             # Большая полуось (м)
period = 4.0145 * day     # Орбитальный период (с)

# --- ФУНКЦИЯ: УРАВНЕНИЯ ДВУХ ТЕЛ ---
def two_body_system(y, t, m1, m2):
    r1 = y[:3]
    r2 = y[3:6]
    v1 = y[6:9]
    v2 = y[9:12]
    r = r2 - r1
    r_norm = np.linalg.norm(r)
    a1 = G * m2 / r_norm**3 * r
    a2 = -G * m1 / r_norm**3 * r
    return np.concatenate([v1, v2, a1, a2])

# --- НАЧАЛЬНЫЕ УСЛОВИЯ ---
mu = m1 + m2
r1_0 = np.array([-m2/mu * a, 0, 0])
r2_0 = np.array([m1/mu * a, 0, 0])
v_orb = np.sqrt(G * mu / a)
v1_0 = np.array([0, -m2/mu * v_orb, 0])
v2_0 = np.array([0, m1/mu * v_orb, 0])
y0 = np.concatenate([r1_0, r2_0, v1_0, v2_0])

# --- ИНТЕГРИРОВАНИЕ ДВИЖЕНИЯ ---
t_max = 2 * period
Npoints = 1000
t = np.linspace(0, t_max, Npoints)
sol = odeint(two_body_system, y0, t, args=(m1, m2))

# --- ВИЗУАЛИЗАЦИЯ ОРБИТ ---
# Координаты звёзд
x1, y1 = sol[:, 0], sol[:, 1]
x2, y2 = sol[:, 3], sol[:, 4]

plt.figure(figsize=(8, 6))
plt.plot(x1 / AU, y1 / AU, label='Spica A (более массивная звезда)')
plt.plot(x2 / AU, y2 / AU, label='Spica B (менее массивная звезда)')
plt.scatter([x1[0] / AU, x2[0] / AU], [y1[0] / AU, y2[0] / AU],
            color='red', marker='o', label='Начальные положения')
plt.xlabel('x (а.е.)')
plt.ylabel('y (а.е.)')
plt.title('Орбиты звёзд системы Spica (α Vir)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()
