#ifndef GflashProtonShowerProfile_H
#define GflashProtonShowerProfile_H

#include "SimGeneral/GFlash/interface/GflashHadronShowerProfile.h"

class GflashProtonShowerProfile : public GflashHadronShowerProfile {
public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashProtonShowerProfile(const edm::ParameterSet &parSet) : GflashHadronShowerProfile(parSet){};
  ~GflashProtonShowerProfile() override{};

  void loadParameters() override;
};

#endif
