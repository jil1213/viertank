# -*- coding: utf-8 -*-
import numpy as np
# ----------------------------------------------------------------------------------------------------------
# Globale Einstellungen für die physikalische Simulation für 4-Tanksystem bestehend aus Versuchsstand 2 u. 3
# ----------------------------------------------------------------------------------------------------------
# Dabei ist der obere Versuchsstand V3 und der untere V2
# Bei den Ventilabflüssen wird folgende Nomenklatur verwendet: 
# Prefix immer AS dann zwei Zahlen: erste Zahl = welcher Tank, zweite Zahl = welches Ventil

# Pumpe
Ku1 = 2.1e-5                 # m**3 / (s V) - Verstärkung der Pumpe 1
Ku2 = 2.3e-5                 # m**3 / (s V) - Verstärkung der Pumpe 2
uA01 = 7.0                  # V - Spannung ab welcher die Pumpe 1 Wasser fördert 
uA02 = 6.0                  # V - Spannung ab welcher die Pumpe 2 Wasser fördert 
uAmax=12.0                 # Maximalspannung 

#Generische Parameter Tank
rT = 0.06      # m - effektiver Radius des Tank
AT = np.pi * rT ** 2       # m**2 - Tankquerschnitt 1
hV = 0.02                  # m - Höhe zwischen Nullniveau und Ventil
rR = 0.01                  # m Radius Abflussschlauch
AR = np.pi * rR ** 2       # m**2 - Querschnitt Abflussschlauch
hT = 0.3                   # m - Höhe Tank

# Tank 1 (Tank 1 V3)
AS13 = 3.5e-05          # m**2 - Abflussquerschnitt von Tank 1 zu Tank 3 mit 2.75 Umdrehung geschlossen
AS12 = 1e-05          # m**2 - Abflussquerschnitt von Tank 1 zu Tank 2 mit 2.75 Umdrehung geschlossen

# Tank 2 (Tank 1 V2)
AS23 = 1.5e-05          # m**2 - Abflussquerschnitt von Tank 2 zu Tank 3 mit 2.00 Umdrehung geschlossen
AS24 = 3e-05          # m**2 - Abflussquerschnitt von Tank 2 zu Tank 4 mit 2.75 Umdrehung geschlossen

# Tank 3 (Tank 2 V3)
AS34 = 1e-05          # m**2 - Abflussquerschnitt von Tank 3 zu Tank 4 mit 2.00 Umdrehungen geschlossen
AS30 = 3e-05          # m**2 - Abflussquerschnitt von Tank 3 zu Reservoir mit 2.75 Umdrehungen geschlossen

# Tank 4 (Tank 2 V2)
AS40 = 4e-05          # m**2 - Abflussquerschnitt von Tank 4 zu Reservoir mit 2.00 Umdrehungen geschlossen

g = 9.81                     # m / s**2 - Erdbeschleunigung


initial_state = [0, 0, 0, 0]
