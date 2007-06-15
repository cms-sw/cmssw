#include "SimDataFormats/CaloHit/interface/HFShowerLibraryEventInfo.h"
#include "SimDataFormats/CaloHit/interface/HFShowerPhoton.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace {
  namespace {
    HFShowerLibraryEventInfo                             rv1;
    std::vector<HFShowerLibraryEventInfo>                v1;
    edm::Wrapper<std::vector<HFShowerLibraryEventInfo> > wc1;
    HFShowerPhoton                                       rv2;
    std::vector<HFShowerPhoton>                          v2;
    edm::Wrapper<std::vector<HFShowerPhoton> >           wc2;
    PCaloHit                                             rv3;
    edm::PCaloHitContainer                               v3;
    edm::Wrapper<edm::PCaloHitContainer>                 wc3;
  }
}
