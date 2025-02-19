#ifndef GflashKaonPlusShowerProfile_H
#define GflashKaonPlusShowerProfile_H 

#include "SimGeneral/GFlash/interface/GflashHadronShowerProfile.h"

class GflashKaonPlusShowerProfile : public GflashHadronShowerProfile
{
public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashKaonPlusShowerProfile (edm::ParameterSet parSet) : 
    GflashHadronShowerProfile (parSet) {}; 
  ~GflashKaonPlusShowerProfile () {};

  void loadParameters();
};

#endif




