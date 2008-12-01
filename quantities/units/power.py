"""
"""

from quantities.units.unitquantities import compound
from quantities.units.energy import Btu, J
from quantities.units.time import s, h
from quantities.units.electromagnetism import A, V

W = watt = watts = J/s
mW = milliwatt = milliwatts = W/1000
kW = kilowatt = kilowatts = J/s*1000
MW = megawatt = megawatts = kW * 1000
VA = voltampere = voltamperes = V*A
boiler_horsepower = boiler_horsepowers = 9.80950e3 * W
hp = horsepower = horsepowers = shaft_horsepower = shaft_horsepowers = \
    7.456999e2 * W
metric_horsepower = metric_horsepowers = 7.35499 * W
electric_horsepower = electric_horsepowers = 7.460000e2 * W
water_horsepower = water_horsepowers = 7.46043e2 * W
UK_horsepower = UK_horsepowers = 7.4570e2 * W
refrigeration_ton = refrigeration_tons = ton_of_refrigeration = 12000 * Btu/h

W_ = watt_ = watts_ = compound('W')
kW_ = kilowatt_ = kilowatts_ = compound('kW')
MW_ = megawatt_ = megawatts_ = compound('MW')
VA_ = voltampere_ = voltamperes_ = compound('VA')
boiler_horsepower_ = boiler_horsepowers_ = compound('boiler_horsepower')
hp_ = horsepower_ = horsepowers_ = compound('hp')
metric_horsepower_ = metric_horsepowers_ = compound('metric_horsepower')
electric_horsepower_ = electric_horsepowers_ = compound('electric_horsepower')
water_horsepower_ = water_horsepowers_ = compound('water_horsepower')
UK_horsepower_ = UK_horsepowers_ = compound('UK_horsepower')
refrigeration_ton_ = refrigeration_tons_ = ton_of_refrigeration_ = \
    compound('refrigeration_ton')

del compound, Btu, J, s, h, A, V
