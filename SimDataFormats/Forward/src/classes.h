#include "SimDataFormats/Forward/interface/TotemTestHistoClass.h"
#include "SimDataFormats/Forward/interface/LHCTransportLink.h"
#include "SimDataFormats/Forward/interface/LHCTransportLinkContainer.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include <vector>

namespace {
  struct dictionary {
    TotemTestHistoClass                   theTotemTestHisto;
    edm::Wrapper<TotemTestHistoClass>     theTotemTestHistoClass;
    std::vector<TotemTestHistoClass::Hit> theHits;

    LHCTransportLink                             lhcL;
    edm::LHCTransportLinkContainer               lhcLC;
    edm::Wrapper<edm::LHCTransportLinkContainer> wlhcLC;
  };
}
