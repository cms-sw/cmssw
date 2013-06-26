#include "SimDataFormats/EcalTestBeam/interface/PEcalTBInfo.h"
#include "SimDataFormats/EcalTestBeam/interface/HodoscopeDetId.h"

#include "DataFormats/Common/interface/Wrapper.h"
#include <vector>

namespace {
  struct dictionary {
    PEcalTBInfo                 theInfo;
    edm::Wrapper<PEcalTBInfo>   theEcalTBInfo;
  };
}
