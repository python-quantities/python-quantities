"""
"""

from quantities.units.unitquantity import UnitQuantity
from quantities.units.energy import Btu, J
from quantities.units.time import s, h
from quantities.units.electromagnetism import A, V


W = watt = watts = UnitQuantity('W', J/s)
mW = milliwatt = milliwatts = UnitQuantity('mW', W/1000)
kW = kilowatt = kilowatts = UnitQuantity('kW', 1000*W)
MW = megawatt = megawatts = UnitQuantity('MW', 1000*kW)
VA = voltampere = voltamperes = UnitQuantity('VA', 1000*kW)
boiler_horsepower = boiler_horsepowers = UnitQuantity('boiler_horsepower', 9.80950e3*W)
hp = horsepower = horsepowers = UnitQuantity('hp', 7.456999e2*W)
metric_horsepower = metric_horsepowers = UnitQuantity('metric_horsepower', 7.35499*W)
electric_horsepower = electric_horsepowers = UnitQuantity('electric_horsepower', 7.460000e2*W)
water_horsepower = water_horsepowers = UnitQuantity('water_horsepower', 7.46043e2*W)
UK_horsepower = UK_horsepowers = UnitQuantity('UK_horsepower', 7.4570e2*W)
refrigeration_ton = refrigeration_tons = ton_of_refrigeration = \
    UnitQuantity('refrigeration_ton', 12000*Btu/h)

del UnitQuantity, Btu, J, s, h, A, V
