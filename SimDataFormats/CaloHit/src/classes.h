#include "SimDataFormats/CaloHit/interface/CastorShowerEvent.h"
#include "SimDataFormats/CaloHit/interface/CastorShowerLibraryInfo.h"
#include "SimDataFormats/CaloHit/interface/HFShowerLibraryEventInfo.h"
#include "SimDataFormats/CaloHit/interface/HFShowerPhoton.h"
#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "DataFormats/Common/interface/Wrapper.h"

namespace SimDataFormats_CaloHit {
  struct dictionary {
    HFShowerLibraryEventInfo                             rv1;
    edm::Wrapper<HFShowerLibraryEventInfo>               p1;
    std::vector<HFShowerLibraryEventInfo>                v1;
    edm::Wrapper<std::vector<HFShowerLibraryEventInfo> > wc1;

    HFShowerPhoton                                       rv2;
    std::vector<HFShowerPhoton>                          v2;
    edm::Wrapper<std::vector<HFShowerPhoton> >           wc2;

    PCaloHit                                             rv3;
    edm::PCaloHitContainer                               v3;
    std::vector<const PCaloHit*>                         vcp3;
    edm::Wrapper<edm::PCaloHitContainer>                 wc3;

    HFShowerPhotonCollection                             rv4;
    edm::Wrapper<HFShowerPhotonCollection>               wc4;
  };
}
