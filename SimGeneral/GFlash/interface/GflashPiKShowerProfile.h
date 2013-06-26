#ifndef GflashPiKShowerProfile_H
#define GflashPiKShowerProfile_H 

#include "SimGeneral/GFlash/interface/GflashHadronShowerProfile.h"

class GflashPiKShowerProfile : public GflashHadronShowerProfile
{
public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashPiKShowerProfile (edm::ParameterSet parSet) : 
    GflashHadronShowerProfile (parSet) {}; 
  ~GflashPiKShowerProfile () {};

  void loadParameters();
};

#endif




