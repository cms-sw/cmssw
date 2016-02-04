#include "SimGeneral/GFlash/interface/GflashHit.h"

GflashHit::GflashHit() :
  theTime(0),
  theEnergy(0),
  thePosition(0)
{
}

GflashHit::GflashHit(double time, double energy, Gflash3Vector& pos) 
{
  theTime = time;
  theEnergy = energy;
  thePosition = pos;
}

GflashHit::~GflashHit() 
{
}
