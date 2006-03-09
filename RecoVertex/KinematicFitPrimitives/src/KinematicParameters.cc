#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParameters.h"
 
GlobalVector KinematicParameters::momentum() const
{return GlobalVector(par(4),par(5),par(6));}
  
GlobalPoint KinematicParameters::position() const
{return GlobalPoint(par(1),par(2),par(3));}
  
