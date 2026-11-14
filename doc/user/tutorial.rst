====================
Quick-start tutorial
====================

.. automodule:: quantities

Photometric units
=================

Luminous flux, illuminance and luminance can be expressed using the
following units:

.. list-table::
   :header-rows: 1

   * - Unit
     - Symbol
     - Definition
   * - lumen
     - lm
     - cd · sr
   * - lux
     - lx
     - lm / m²
   * - foot-candle
     - fc
     - lm / ft²
   * - phot
     - ph
     - lm / cm²
   * - nit
     - nt
     - cd / m²
   * - stilb
     - sb
     - cd / cm²

For example, convert luminous flux per area to illuminance::

   >>> import quantities as pq
   >>> flux = pq.Quantity([800, 1600], "lm")
   >>> area = pq.Quantity([2, 4], "m**2")
   >>> (flux / area).rescale("lx")
   array([400., 400.]) * lx
   >>> round(float((10 * pq.foot_candle).rescale(pq.lux).magnitude), 4)
   107.6391

The foot-candle is available as ``pq.foot_candle``, ``pq.footcandle``
or ``pq.fc``. Unit names and symbols are also accepted by the
``Quantity`` constructor.

As in the SI, the steradian simplifies to dimensionless, so lumen
simplifies to candela and lux to candela per square metre. This does not
distinguish the physical quantities illuminance and luminance; use their
appropriate units in calculations. Conversion factors are tabulated in the
`NIST Guide to the SI, Appendix B.9
<https://www.nist.gov/pml/special-publication-811/nist-guide-si-appendix-b-conversion-factors/nist-guide-si-appendix-b9>`_.
