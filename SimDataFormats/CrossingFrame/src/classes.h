#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/PCrossingFrame.h"
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

 	PCrossingFrame<PSimHit> dummy6;
 	PCrossingFrame<PCaloHit> dummy8;
 	PCrossingFrame<SimTrack> dummy9;
 	PCrossingFrame<SimVertex> dummy10;
 	PCrossingFrame<edm::HepMCProduct> dummy11;

        std::vector<int> dummy14;
        std::vector<const PSimHit *> dummy15;
        std::vector<const PCaloHit *> dummy16;
        std::vector<const SimTrack *> dummy17;
        std::vector<const SimVertex *> dummy18;
        std::vector<const edm::HepMCProduct *> dummy19;

        edm::Wrapper<CrossingFramePlaybackInfo > dummy24;
        edm::Wrapper<CrossingFrame<PSimHit> > dummy25;
        edm::Wrapper<CrossingFrame<PCaloHit> > dummy26;
        edm::Wrapper<CrossingFrame<SimTrack> > dummy27;
        edm::Wrapper<CrossingFrame<SimVertex> > dummy28;
        edm::Wrapper<CrossingFrame<edm::HepMCProduct> > dummy29;
	
        edm::Wrapper<PCrossingFrame<PSimHit> > dummy36;
        edm::Wrapper<PCrossingFrame<PCaloHit> > dummy37;
        edm::Wrapper<PCrossingFrame<SimTrack> > dummy38;
        edm::Wrapper<PCrossingFrame<SimVertex> > dummy39;
        edm::Wrapper<PCrossingFrame<edm::HepMCProduct> > dummy40;
  };
}
