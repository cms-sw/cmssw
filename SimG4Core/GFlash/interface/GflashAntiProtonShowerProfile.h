#ifndef GflashAntiProtonShowerProfile_H
#define GflashAntiProtonShowerProfile_H 

#include "SimG4Core/GFlash/interface/GflashHadronShowerProfile.h"

class GflashAntiProtonShowerProfile : public GflashHadronShowerProfile
{
public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashAntiProtonShowerProfile (edm::ParameterSet parSet) : 
    GflashHadronShowerProfile (parSet) {}; 
  ~GflashAntiProtonShowerProfile () {};

  void loadParameters(const G4FastTrack& fastTrack);

};

#endif




