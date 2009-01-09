#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFramePlaybackInfo.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include <map>
#include <string>

namespace {
  struct dictionary {
 	CrossingFrame<PSimHit> dummy1;

 	CrossingFramePlaybackInfo dummy0;
 	CrossingFrame<PCaloHit> dummy2;
 	CrossingFrame<SimTrack> dummy3;
 	CrossingFrame<SimVertex> dummy4;
 	CrossingFrame<edm::HepMCProduct> dummy5;

        std::vector<int> dummy9;
        std::vector<const PSimHit *> dummy10;
        std::vector<const PCaloHit *> dummy11;
        std::vector<const SimTrack *> dummy12;
        std::vector<const SimVertex *> dummy13;
        std::vector<const edm::HepMCProduct *> dummy14;

        edm::Wrapper<CrossingFramePlaybackInfo > dummy19;
        edm::Wrapper<CrossingFrame<PSimHit> > dummy20;
        edm::Wrapper<CrossingFrame<PCaloHit> > dummy21;
        edm::Wrapper<CrossingFrame<SimTrack> > dummy22;
        edm::Wrapper<CrossingFrame<SimVertex> > dummy23;
        edm::Wrapper<CrossingFrame<edm::HepMCProduct> > dummy24;
  };
}
