#include "Validation/MuonRPCGeometry/interface/Constants.h"

double RPCpg::rate(double x){ // rate from pt = x [Gev/c] to inf

   double ret = 0;
   double a = -0.235801;
   double b = -2.82346;
   double c = 17.162;

   ret = std::pow( x,a*std::log(x) ) * std::pow(x,b)*std::exp(c);

   return ret;
}

