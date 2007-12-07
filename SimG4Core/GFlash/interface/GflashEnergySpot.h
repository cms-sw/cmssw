#ifndef GflashEnergySpot_H
#define GflashEnergySpot_H

#include "G4ThreeVector.hh"

class GflashEnergySpot
{
public:
  
  GflashEnergySpot();
  GflashEnergySpot(G4double energy, G4ThreeVector& pos);
  ~GflashEnergySpot();
  
  inline G4double getEnergy() const { return theEnergy; }
  inline const G4ThreeVector& getPosition() const { return thePosition; }

  inline void setEnergy(const G4double energy) { theEnergy = energy; }
  inline void setPosition(const G4ThreeVector& pos) { thePosition = pos; }
    
private:
  G4double theEnergy;
  G4ThreeVector thePosition;
};

#endif
