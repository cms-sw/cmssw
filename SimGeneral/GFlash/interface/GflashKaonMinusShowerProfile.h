#ifndef GflashKaonMinusShowerProfile_H
#define GflashKaonMinusShowerProfile_H 

#include "SimGeneral/GFlash/interface/GflashHadronShowerProfile.h"

class GflashKaonMinusShowerProfile : public GflashHadronShowerProfile
{
public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashKaonMinusShowerProfile (edm::ParameterSet parSet) : 
    GflashHadronShowerProfile (parSet) {}; 
  ~GflashKaonMinusShowerProfile () {};

  void loadParameters();
};

#endif




