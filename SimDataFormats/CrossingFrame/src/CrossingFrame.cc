#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
//#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/Math/interface/Vector3D.h"

#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

//using namespace std;
using namespace edm;

template <> const int  CrossingFrame<PSimHit>::limHighLowTof = 36;

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
      // and the acceptance depends onb ToF
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
void CrossingFrame<edm::HepMCProduct>::addPileups(const int bcr, std::vector<edm::HepMCProduct> *mcps, unsigned int evtNr, int vertexoffset,bool checkTof,bool high) { 
  LogWarning("CrossingFrame")<<"addPileups should never be called for a HepMCProduct!!";
}

template <> 
void  CrossingFrame<PSimHit>::setTof() {
  // only for simhits: containers may be used twice, and result depends on ToF
  // that is why we have to do the ToF transformation right at the end
  for (unsigned int i=0;i<pileups_.size();++i) {
    const_cast<PSimHit *>(pileups_[i])->setTof(pileups_[i]->timeOfFlight() + getBunchCrossing(i)*bunchSpace_);
  } 
} 
//ATTENTION:======================================================================================
// THIS UGLY IMPLEMENTATION WAS DONE TO OVERCOME A TEMPLATE PROBLEM THAT REMAINS TO BE UNDERSTOOD
// AS SOON AS THERE IS A DEFAULT IMPLEMENTATION FOR setTof THIS DEFAULT WAS TAKEN IN ALL CASES
//================================================================================================
template <> 
void  CrossingFrame<PCaloHit>::setTof() {}

template <> 
void  CrossingFrame<SimTrack>::setTof() {}

template <> 
void  CrossingFrame<SimVertex>::setTof() {}

template <> 
void  CrossingFrame<edm::HepMCProduct>::setTof() {}
