#include "SimDataFormats/HcalTestBeam/interface/PHcalTB04Info.h"
#include "SimDataFormats/HcalTestBeam/interface/PHcalTB06Info.h"
#include "SimDataFormats/HcalTestBeam/interface/HcalTB02HistoClass.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include <vector>

namespace {
  struct dictionary {
    PHcalTB04Info                    theInfo4;
    edm::Wrapper<PHcalTB04Info>      theTB04Info;
    std::vector<PHcalTB06Info::Vtx>  dummy1;
    std::vector<PHcalTB06Info::Hit>  dummy2;
    PHcalTB06Info                    theInfo6;
    edm::Wrapper<PHcalTB06Info>      theTB06Info;
    HcalTB02HistoClass               theHcalTB02Histo;
    edm::Wrapper<HcalTB02HistoClass> theHcalTB02HistoClass;
  };
}
