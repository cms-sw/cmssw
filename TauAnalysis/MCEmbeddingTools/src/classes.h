#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

namespace DataFormats_TrackReco {
struct dictionary {
    
  edm::ValueMap<reco::TrackRef> rtref_vm;
  edm::ValueMap<reco::TrackRef>::const_iterator rtref_vmci;
  edm::Wrapper<edm::ValueMap<reco::TrackRef> > rtref_wvm;

  };

}
