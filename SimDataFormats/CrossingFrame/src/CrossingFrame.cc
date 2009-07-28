#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "DataFormats/Math/interface/Vector3D.h"

using namespace edm;

template <> 
void CrossingFrame<SimTrack>::addPileups(const int bcr, std::vector<SimTrack> *simtracks, unsigned int evtNr, int vertexoffset,bool checkTof,bool high) { 

  EncodedEventId id(bcr,evtNr);
  for (unsigned int i=0;i<simtracks->size();++i){
    (*simtracks)[i].setEventId(id);
    if (!(*simtracks)[i].noVertex()) 
      (*simtracks)[i].setVertexIndex((*simtracks)[i].vertIndex()+vertexoffset);
    pileups_.push_back(&((*simtracks)[i]));
  }
}

template <> 
void CrossingFrame<SimVertex>::addPileups(const int bcr, std::vector<SimVertex> *simvertices, unsigned int evtNr, int vertexoffset,bool checkTof,bool high) { 

  EncodedEventId id(bcr,evtNr);
  for (unsigned int i=0;i<simvertices->size();++i) {
    (*simvertices)[i].setEventId(id);
    (*simvertices)[i].setTof((*simvertices)[i].position().t()+bcr*bunchSpace_);
    pileups_.push_back(&((*simvertices)[i]));
  }
}

template <> 
void CrossingFrame<PSimHit>::addPileups(const int bcr, std::vector<PSimHit> *simhits, unsigned int evtNr, int vertexoffset,bool checkTof,bool high) { 

  EncodedEventId id(bcr,evtNr);

  int count=0;
  for (unsigned int i=0;i<simhits->size();++i) {
    bool accept=true;
    float newtof;
    if (checkTof) {
      newtof=(*simhits)[i].timeOfFlight() + bcr*bunchSpace_;
      accept=high ? newtof>= limHighLowTof : newtof < limHighLowTof;
    }
    if (!checkTof || accept) {
      (*simhits)[i].setEventId(id);
      // For simhits a container may be used twice (high+low)
      // and the acceptance depends on ToF
      // Therefore we transform only at the end.
      pileups_.push_back(&((*simhits)[i]));
      count++;
    }  
  }
}

template <> 
void CrossingFrame<PCaloHit>::addPileups(const int bcr, std::vector<PCaloHit> *calohits, unsigned int evtNr, int vertexoffset,bool checkTof,bool high) { 

  EncodedEventId id(bcr,evtNr);
  for (unsigned int i=0;i<calohits->size();++i) {
    PCaloHit hit((*calohits)[i].id(),(*calohits)[i].energyEM(),(*calohits)[i].energyHad(),(*calohits)[i].time()+bcr*bunchSpace_,(*calohits)[i].geantTrackId());
    (*calohits)[i].setEventId(id);
    (*calohits)[i].setTime((*calohits)[i].time()+bcr*bunchSpace_);
    pileups_.push_back(&((*calohits)[i]));
  }
}

template <> 
void  CrossingFrame<PSimHit>::setTof() {
  // does something only for simhits: containers may be used twice, and result depends on ToF
  // that is why we have to do the ToF transformation right at the end
  for (unsigned int i=0;i<pileups_.size();++i) {
    const_cast<PSimHit *>(pileups_[i])->setTof(pileups_[i]->timeOfFlight() + getBunchCrossing(i)*bunchSpace_);
  } 
} 
