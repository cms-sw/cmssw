#ifndef GflashAntiProtonShowerProfile_H
#define GflashAntiProtonShowerProfile_H

#include "SimGeneral/GFlash/interface/GflashHadronShowerProfile.h"

class GflashAntiProtonShowerProfile : public GflashHadronShowerProfile {
public:
  //-------------------------
  // Constructor, destructor
  //-------------------------
  GflashAntiProtonShowerProfile(const edm::ParameterSet &parSet) : GflashHadronShowerProfile(parSet){};
  ~GflashAntiProtonShowerProfile() override{};

  void loadParameters() override;
};

#endif
