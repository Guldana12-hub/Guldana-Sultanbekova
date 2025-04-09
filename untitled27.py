#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 22:04:08 2025

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
Rsun = 6.957e8            # Радиус Солнца (м)

# --- ПАРАМЕТРЫ СИСТЕМЫ SPICA ---
m1 = 10.25 * Msun         # Масса Spica A
m2 =  6.97 * Msun         # Масса Spica B
a  =  0.12 * AU           # Большая полуось (м)
period = 4.0145 * day     # Орбитальный период (с)

# Радиусы для моделирования затмений (искусственно увеличены)
R1 = 10.0 * Rsun
R2 =  6.0 * Rsun

# Светимости (относительные)
L1 = 1.0
L2 = 0.3

# Наклон орбиты (i = 90° — наблюдаем "с ребра")
inclination_deg = 90
inclination_rad = np.radians(inclination_deg)

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
r2_0 = np.array([ m1/mu * a, 0, 0])
v_orb = np.sqrt(G * mu / a)
v1_0 = np.array([0, -m2/mu * v_orb, 0])
v2_0 = np.array([0,  m1/mu * v_orb, 0])
y0 = np.concatenate([r1_0, r2_0, v1_0, v2_0])

# --- ИНТЕГРИРОВАНИЕ ДВИЖЕНИЯ ---
t_max = 2 * period
Npoints = 1000
t = np.linspace(0, t_max, Npoints)
sol = odeint(two_body_system, y0, t, args=(m1, m2))

# --- ПОВОРОТ ОРБИТЫ ПО НАКЛОНУ ---
def rotate_inclination(sol, inc_rad):
    sol_rot = sol.copy()
    for i_obj in [0, 1]:
        ix, iy, iz = 3 * i_obj, 3 * i_obj + 1, 3 * i_obj + 2
        x, y, z = sol[:, ix], sol[:, iy], sol[:, iz]
        x_new = x
        y_new = y * np.cos(inc_rad) - z * np.sin(inc_rad)
        z_new = y * np.sin(inc_rad) + z * np.cos(inc_rad)
        sol_rot[:, ix] = x_new
        sol_rot[:, iy] = y_new
        sol_rot[:, iz] = z_new
    return sol_rot

sol_rot = sol.copy()
sol_rot[:, :6] = rotate_inclination(sol[:, :6], inclination_rad)

# --- КРИВАЯ БЛЕСКА ---
def calc_light_curve(sol_xyz, R1, R2, L1, L2):
    lc = np.ones(len(sol_xyz)) * (L1 + L2)
    for i in range(len(sol_xyz)):
        xA, yA, zA = sol_xyz[i, :3]
        xB, yB, zB = sol_xyz[i, 3:6]
        proj_dist = np.sqrt((xB - xA)**2 + (yB - yA)**2)
        if proj_dist < (R1 + R2):
            overlap = (R1 + R2) - proj_dist
            if zA < zB:
                frac = min(1.0, overlap / R2)
                lc[i] = L1 + L2 * (1 - 0.5 * frac**2)
            else:
                frac = min(1.0, overlap / R1)
                lc[i] = L2 + L1 * (1 - 0.5 * frac**2)
    return lc

solA = sol_rot[:, :3]
solB = sol_rot[:, 3:6]
sol_for_lc = np.hstack((solA, solB))
light_curve = calc_light_curve(sol_for_lc, R1, R2, L1, L2)

# --- ВИЗУАЛИЗАЦИЯ КРИВОЙ БЛЕСКА ---
plt.figure(figsize=(10, 5))
plt.plot(t / day, light_curve, 'b-', linewidth=2)
plt.xlabel("Время (сутки)")
plt.ylabel("Относительная яркость")
plt.title(f"Кривая блеска Spica при наклоне i={inclination_deg}°")
plt.grid(True)
plt.tight_layout()
plt.show()