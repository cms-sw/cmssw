#include "SimDataFormats/EcalTestBeam/interface/PEcalTBInfo.h"
#include "SimDataFormats/EcalTestBeam/interface/HodoscopeDetId.h"

#include "DataFormats/Common/interface/Wrapper.h"
#include <vector>

namespace SimDataFormats_EcalTestBeam {
  struct dictionary {
    PEcalTBInfo                 theInfo;
    edm::Wrapper<PEcalTBInfo>   theEcalTBInfo;
  };
}
