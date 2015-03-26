#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/PCrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFramePlaybackInfoExtended.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFramePlaybackInfoNew.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/Common/interface/Wrapper.h"

#include <map>
#include <string>

namespace SimDataFormats_CrossingFrame {
  struct dictionary {
 	CrossingFrame<PSimHit> dummy0;
	
	CrossingFramePlaybackInfoNew dummy1;
	CrossingFramePlaybackInfoExtended dummy2;
 	CrossingFrame<PCaloHit> dummy3;
 	CrossingFrame<SimTrack> dummy4;
 	CrossingFrame<SimVertex> dummy5;
 	CrossingFrame<edm::HepMCProduct> dummy6;

 	PCrossingFrame<PSimHit> dummy7;
 	PCrossingFrame<PCaloHit> dummy8;
 	PCrossingFrame<SimTrack> dummy9;
 	PCrossingFrame<SimVertex> dummy10;
 	PCrossingFrame<edm::HepMCProduct> dummy11;

	edm::Wrapper<CrossingFramePlaybackInfoNew > dummy23;
	edm::Wrapper<CrossingFramePlaybackInfoExtended > dummy24;
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
