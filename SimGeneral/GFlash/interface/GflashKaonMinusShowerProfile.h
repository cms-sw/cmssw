#ifndef GflashKaonMinusShowerProfile_H
#define GflashKaonMinusShowerProfile_H

#include "SimGeneral/GFlash/interface/GflashHadronShowerProfile.h"

class GflashKaonMinusShowerProfile : public GflashHadronShowerProfile {
public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashKaonMinusShowerProfile(const edm::ParameterSet &parSet) : GflashHadronShowerProfile(parSet){};
  ~GflashKaonMinusShowerProfile() override{};

  void loadParameters() override;
};

#endif
