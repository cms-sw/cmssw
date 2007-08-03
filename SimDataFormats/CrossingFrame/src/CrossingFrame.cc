#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"

//using namespace std;
using namespace edm;


template <> const int  CrossingFrame<PSimHit>::lowTrackTof = -36;
template <> const int  CrossingFrame<PSimHit>::highTrackTof = 36;
template <> void CrossingFrame<SimTrack>::addPileups(const int bcr, const std::vector<SimTrack> *simtracks, unsigned int evtNr, int vertexoffset,bool checkTof) { 
  pileupOffsets_.push_back(pileups_.size());

  EncodedEventId id(bcr,evtNr);
  for (unsigned int i=0;i<simtracks->size();++i)
    if ((*simtracks)[i].noVertex()) {
      SimTrack track((*simtracks)[i]);
      track.setEventId(id);
      pileups_.push_back(track);
    }
    else {
      SimTrack track((*simtracks)[i].type(),(*simtracks)[i].momentum(),(*simtracks)[i].vertIndex()+vertexoffset, (*simtracks)[i].genpartIndex());
      track.setEventId(id);
      track.setTrackId((*simtracks)[i].trackId());
      pileups_.push_back(track);
    }
}

template <> void CrossingFrame<SimVertex>::addPileups(const int bcr, const std::vector<SimVertex> *simvertices, unsigned int evtNr, int vertexoffset,bool checkTof) { 
  pileupOffsets_.push_back(pileups_.size());

  EncodedEventId id(bcr,evtNr);
  for (unsigned int i=0;i<simvertices->size();++i) {
    SimVertex vertex((*simvertices)[i].position(),((*simvertices)[i].position())[3]+bcr*bunchSpace_,(*simvertices)[i].parentIndex());
    vertex.setEventId(EncodedEventId(bcr,evtNr));
    pileups_.push_back(vertex);
  }
}

template <> void CrossingFrame<PSimHit>::addPileups(const int bcr, const std::vector<PSimHit> *simhits, unsigned int evtNr, int vertexoffset,bool checkTof) { 

  EncodedEventId id(bcr,evtNr);

  int count=0;
  for (unsigned int i=0;i<simhits->size();++i) {
    bool accept=true;
    float newtof;
    if (checkTof) {
      newtof=(*simhits)[i].timeOfFlight() + bcr*bunchSpace_;
      accept=newtof>=lowTrackTof && newtof <=highTrackTof;
    }
    if (!checkTof || accept) {
      PSimHit hit((*simhits)[i].entryPoint(), (*simhits)[i].exitPoint(),(*simhits)[i].pabs(),
		  (*simhits)[i].timeOfFlight() + bcr*bunchSpace_, 
		  (*simhits)[i].energyLoss(), (*simhits)[i].particleType(),
		  (*simhits)[i].detUnitId(), (*simhits)[i].trackId(),
		  (*simhits)[i].thetaAtEntry(),  (*simhits)[i].phiAtEntry(),  (*simhits)[i].processType());
      hit.setEventId(id);
      pileups_.push_back(hit);
      count++;
    }  
  }
  pileupOffsets_.push_back(count);
}

template <> void CrossingFrame<PCaloHit>::addPileups(const int bcr, const std::vector<PCaloHit> *calohits, unsigned int evtNr, int vertexoffset,bool checkTof) { 
  pileupOffsets_.push_back(pileups_.size());

  EncodedEventId id(bcr,evtNr);
  for (unsigned int i=0;i<calohits->size();++i) {
    PCaloHit hit((*calohits)[i].id(),(*calohits)[i].energyEM(),(*calohits)[i].energyHad(),(*calohits)[i].time()+bcr*bunchSpace_,(*calohits)[i].geantTrackId());
    hit.setEventId(id);
    pileups_.push_back(hit);
  }
}


 
