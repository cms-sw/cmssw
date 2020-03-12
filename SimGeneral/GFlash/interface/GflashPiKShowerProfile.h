#ifndef GflashPiKShowerProfile_H
#define GflashPiKShowerProfile_H

#include "SimGeneral/GFlash/interface/GflashHadronShowerProfile.h"

class GflashPiKShowerProfile : public GflashHadronShowerProfile {
public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashPiKShowerProfile(const edm::ParameterSet &parSet) : GflashHadronShowerProfile(parSet){};
  ~GflashPiKShowerProfile() override{};

  void loadParameters() override;
};

#endif
