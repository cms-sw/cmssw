#ifndef HCALTBPARTICLEID_H
#define HCALTBPARTICLEID_H 1

#include <string>
#include <iostream>
#include <vector>
#include "boost/cstdint.hpp"

  class HcalTBParticleId {
  public:
    HcalTBParticleId();

    // Getter methods

    double TOF()     const { return TOF_;     }

  private:
	double    TOF_ ; 


  };

  std::ostream& operator<<(std::ostream& s, const HcalTBParticleId& htbpid);

#endif
