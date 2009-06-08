#ifndef GflashProtonShowerProfile_H
#define GflashProtonShowerProfile_H 

#include "SimG4Core/GFlash/interface/GflashHadronShowerProfile.h"

class GflashProtonShowerProfile : public GflashHadronShowerProfile
{
public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashProtonShowerProfile (edm::ParameterSet parSet) : 
    GflashHadronShowerProfile (parSet) {}; 
  ~GflashProtonShowerProfile () {};

  void loadParameters(const G4FastTrack& fastTrack);

};

#endif




