#ifndef PPS_UNITS
#define PPS_UNITS
#include <cmath>

static const double ProtonMass = 0.93827;
static const double ProtonMassSQ = pow(ProtonMass, 2);
static const double c_light_s = 2.99792458e+8;  // m/s;
static const double s_to_ns = 1.e9;
static const double c_light_ns = c_light_s / s_to_ns;
static const double urad = 1. / 1000000.;  //micro rad
static const double um_to_mm = 0.001;
static const double um_to_cm = 0.0001;
static const double mm_to_um = 1000.;
static const double mm_to_cm = 0.1;
static const double mm_to_m = 0.001;
static const double cm_to_um = 10000.;
static const double cm_to_mm = 10.;
static const double cm_to_m = 0.01;
static const double m_to_cm = 100.;
static const double m_to_mm = 1000.;
static const double m_to_um = 1000000.;
#endif
