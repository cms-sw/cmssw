#include "SimDataFormats/Forward/interface/TotemTestHistoClass.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include <vector>

namespace {
  namespace {
    TotemTestHistoClass                   theTotemTestHisto;
    edm::Wrapper<TotemTestHistoClass>     theTotemTestHistoClass;
    std::vector<TotemTestHistoClass::Hit> theHits;
  } 
}
