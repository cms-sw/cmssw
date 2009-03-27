#include "SimG4Core/GFlash/interface/GflashEnergySpot.h"

GflashEnergySpot::GflashEnergySpot() :
  theEnergy(0),
  thePosition(0)
{
}

GflashEnergySpot::GflashEnergySpot(G4double energy, G4ThreeVector& pos) 
{
  theEnergy = energy;
  thePosition = pos;
}

GflashEnergySpot::~GflashEnergySpot() 
{
}
