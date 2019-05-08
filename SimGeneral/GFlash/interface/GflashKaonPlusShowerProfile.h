#ifndef GflashKaonPlusShowerProfile_H
#define GflashKaonPlusShowerProfile_H

#include "SimGeneral/GFlash/interface/GflashHadronShowerProfile.h"

class GflashKaonPlusShowerProfile : public GflashHadronShowerProfile {
public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashKaonPlusShowerProfile(const edm::ParameterSet &parSet) : GflashHadronShowerProfile(parSet){};
  ~GflashKaonPlusShowerProfile() override{};

  void loadParameters() override;
};

#endif
