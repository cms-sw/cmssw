#include "SimDataFormats/CaloTest/interface/HcalTestHistoClass.h"
#include "SimDataFormats/CaloTest/interface/ParticleFlux.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace SimDataFormats_CaloTest {
  struct dictionary {
    HcalTestHistoClass theHcalTestHistoClass;
    ParticleFlux                                         m_d1;
    edm::Wrapper<ParticleFlux>                           m_thed1;
    ParticleFlux::flux                                   m_d2;
    edm::Wrapper<ParticleFlux::flux>                     m_thed2;
    std::vector<ParticleFlux::flux>                      m_d3;
    edm::Wrapper<std::vector<ParticleFlux::flux> >       m_thed3;
  };
}
