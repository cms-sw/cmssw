#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "DataFormats/Math/interface/Vector3D.h"

using namespace edm;

template <> 
void CrossingFrame<SimTrack>::addPileups(const int bcr, std::vector<SimTrack> *simtracks, unsigned int evtNr, int vertexoffset) { 

  EncodedEventId id(bcr,evtNr);
  for (unsigned int i=0;i<simtracks->size();++i){
    (*simtracks)[i].setEventId(id);
    if (!(*simtracks)[i].noVertex()) 
      (*simtracks)[i].setVertexIndex((*simtracks)[i].vertIndex()+vertexoffset);
    pileups_.push_back(&((*simtracks)[i]));
  }
}

template <> 
void CrossingFrame<SimVertex>::addPileups(const int bcr, std::vector<SimVertex> *simvertices, unsigned int evtNr, int vertexoffset) { 

  EncodedEventId id(bcr,evtNr);
  for (unsigned int i=0;i<simvertices->size();++i) {
    (*simvertices)[i].setEventId(id);
    (*simvertices)[i].setTof((*simvertices)[i].position().t()+bcr*bunchSpace_);
    pileups_.push_back(&((*simvertices)[i]));
  }
}

template <> 
void CrossingFrame<PSimHit>::addPileups(const int bcr, std::vector<PSimHit> *simhits, unsigned int evtNr, int vertexoffset) { 

  EncodedEventId id(bcr,evtNr);

  for (unsigned int i=0;i<simhits->size();++i) {
    (*simhits)[i].setEventId(id);
    (*simhits)[i].setTof((*simhits)[i].timeOfFlight() + bcr*bunchSpace_);
    pileups_.push_back(&((*simhits)[i]));
  }
}

template <> 
void CrossingFrame<PCaloHit>::addPileups(const int bcr, std::vector<PCaloHit> *calohits, unsigned int evtNr, int vertexoffset) { 

  EncodedEventId id(bcr,evtNr);
  for (unsigned int i=0;i<calohits->size();++i) {
    PCaloHit hit((*calohits)[i].id(),(*calohits)[i].energyEM(),(*calohits)[i].energyHad(),(*calohits)[i].time()+bcr*bunchSpace_,(*calohits)[i].geantTrackId());
    (*calohits)[i].setEventId(id);
    (*calohits)[i].setTime((*calohits)[i].time()+bcr*bunchSpace_);
    pileups_.push_back(&((*calohits)[i]));
  }
}

//special for the upgrade hit relabeller code - don't add the bunchspace time offset because it
//has already been added
template <> 
void CrossingFrame<PCaloHit>::addPileupsRelabeller(const int bcr, std::vector<PCaloHit> *calohits, unsigned int evtNr, int vertexoffset) { 

  EncodedEventId id(bcr,evtNr);
  for (unsigned int i=0;i<calohits->size();++i) {
    PCaloHit hit((*calohits)[i].id(),(*calohits)[i].energyEM(),(*calohits)[i].energyHad(),(*calohits)[i].time()+bcr*bunchSpace_,(*calohits)[i].geantTrackId());
    (*calohits)[i].setEventId(id);
    pileups_.push_back(&((*calohits)[i]));
  }
}
