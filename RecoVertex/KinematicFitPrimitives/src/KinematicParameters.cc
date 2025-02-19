#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParameters.h"
 
GlobalVector KinematicParameters::momentum() const
{return GlobalVector(par[3], par[4], par[5]);}
  
GlobalPoint KinematicParameters::position() const
{return GlobalPoint(par[0], par[1], par[2]);}

double KinematicParameters::energy() const {
  return sqrt(par(3)*par(3)+par(4)*par(4)+par(5)*par(5)+par(6)*par(6));
}
 
