#ifndef CTPPSHectorParameters_H
#define CTPPSHectorParameters_H
#include <math.h>

const double   ProtonMass = 0.93827;
const double   ProtonMassSQ = pow(ProtonMass,2);
const double   c_light_s = 2.99792458e+8;// m/s;
const double   s_to_ns  = 1.e9;
const double   c_light_ns = c_light_s/s_to_ns;
const double   urad     = 1./1000000.; //micro rad
const double   um_to_mm = 0.001;
const double   um_to_cm = 0.0001;
const double   mm_to_um = 1000.;
const double   mm_to_cm = 0.1;
const double   mm_to_m  = 0.001;
const double   cm_to_um = 10000.;
const double   cm_to_mm = 10.;
const double   cm_to_m  = 0.01;
const double   m_to_cm  = 100.;
const double   m_to_mm  = 1000.;
const double   m_to_um  = 1000000.;
#endif
