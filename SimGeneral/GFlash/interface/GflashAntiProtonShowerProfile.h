#ifndef GflashAntiProtonShowerProfile_H
#define GflashAntiProtonShowerProfile_H 

#include "SimGeneral/GFlash/interface/GflashHadronShowerProfile.h"

class GflashAntiProtonShowerProfile : public GflashHadronShowerProfile
{
public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashAntiProtonShowerProfile (edm::ParameterSet parSet) : 
    GflashHadronShowerProfile (parSet) {}; 
  ~GflashAntiProtonShowerProfile () {};

  void loadParameters();

};

#endif




