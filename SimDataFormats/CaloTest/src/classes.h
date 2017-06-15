#include "SimDataFormats/CaloTest/interface/HcalTestHistoClass.h"
#include "SimDataFormats/CaloTest/interface/ParticleFlux.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace SimDataFormats_CaloTest {
  struct dictionary {
    HcalTestHistoClass theHcalTestHistoClass;
    ParticleFlux                                         d1;
    edm::Wrapper<ParticleFlux>                           thed1;
    ParticleFlux::flux                                   d2;
    edm::Wrapper<ParticleFlux::flux>                     thed2;
    std::vector<ParticleFlux::flux>                      d3;
    edm::Wrapper<std::vector<ParticleFlux::flux> >       thed3;
  };
}
