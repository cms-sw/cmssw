#ifndef CROSSING_FRAME_H
#define CROSSING_FRAME_H

/** \class CrossingFrame
 *
 * CrossingFrame is the result of the Sim Mixing Module
 *
 * \author Ursula Berthon, Claude Charlot,  LLR Palaiseau
 *
 * \version   1st Version July 2005
 * \version   2nd Version Sep 2005
 *
 ************************************************************/

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <string>
#include <map>
#include <utility>

template <class T> 
class CrossingFrame 
{ 

 public:
  // con- and destructors

  CrossingFrame():  firstCrossing_(0), lastCrossing_(0), bunchSpace_(75) {;}
  CrossingFrame(int minb, int maxb, int bunchsp, std::string subdet );

  ~CrossingFrame();
  void clear();

  void addSignals(const std::vector<T> * vec,edm::EventID id);

  void addPileups(const int bcr,const std::vector<T> * vec, unsigned int evtId,int vertexoffset=0,bool checkTof=false);
  void print(int level=0) const ;
  void setBcrOffset() {
    pileupOffsetsBcr_.push_back(pileups_.size()); 
  }

  //getters
  edm::EventID getEventID() const {return id_;}
  std::pair<int,int> getBunchRange() const {return std::pair<int,int>(firstCrossing_,lastCrossing_);}
  int getBunchSpace() const {return bunchSpace_;}
  int getBunchCrossing(unsigned int ip) const {
    for (unsigned int ii=1;ii<pileupOffsetsBcr_.size();ii++){
      if (ip>=pileupOffsetsBcr_[ii-1] && ip<pileupOffsetsBcr_[ii]) return ii+firstCrossing_-1;
    }
    if (ip<pileups_.size()) return lastCrossing_;
    else return 999;
  }
  void getSignal(typename std::vector<T>::const_iterator &first,typename std::vector<T>::const_iterator &last) const {

    first=signals_.begin();
    last=signals_.end();
  }
  void getPileups(typename std::vector<T>::const_iterator &first, typename std::vector<T>::const_iterator &last) const;
  unsigned int getNrSignals() const {return signals_.size();} 
  unsigned int getNrPileups() const {return pileups_.size();} 

  // limits for tof to be considered for trackers
  static const int lowTrackTof; //nsec
  static const int highTrackTof;
  static const int minLowTof;
  static const int limHighLowTof;
					    
 private:

  int firstCrossing_;
  int lastCrossing_;
  int bunchSpace_;  //in nsec
  std::string subdet_;
  edm::EventID id_;
 

  // signal
  std::vector<T>  signals_; 
  //pileup
  std::vector<T>  pileups_;  
  std::vector<unsigned int> pileupOffsets_;  //is this one really necessary????
  std::vector<unsigned int> pileupOffsetsBcr_;
};

//==============================================================================
//                              implementations
//==============================================================================
template <class T> CrossingFrame<T>::CrossingFrame(int minb, int maxb, int bunchsp, std::string subdet ):firstCrossing_(minb), lastCrossing_(maxb), bunchSpace_(bunchsp),subdet_(subdet) {
  //FIXME: should we force around 0 or so??
  pileupOffsetsBcr_.reserve(-firstCrossing_+lastCrossing_+1);
}
template <class T> void CrossingFrame<T>::addSignals(const std::vector<T> * vec,edm::EventID id){
  id_=id;
  signals_=*vec;
}

template <class T>  CrossingFrame<T>::~CrossingFrame () {
  this->clear();
}

template <class T>  void  CrossingFrame<T>::getPileups(typename std::vector<T>::const_iterator &first,typename std::vector<T>::const_iterator &last) const {
  first=pileups_.begin();
  last=pileups_.end();
}

template <class T> void CrossingFrame<T>::clear() {
  // clear things up
  signals_.clear();
  pileups_.clear();
  pileupOffsets_.clear();
  pileupOffsetsBcr_.clear();
}
template <class T> void CrossingFrame<T>::print(int level) const {
}

#include<iosfwd>
#include<iostream>

template <class T>
std::ostream &operator<<(std::ostream& o, const CrossingFrame<T>& cf)
{
  std::pair<int,int> range=cf.getBunchRange();
  o <<"\nCrossingFrame for subdet "<<cf.getEventID()<<",  bunchrange = "<<range.first<<","<<range.second
    <<", bunchSpace "<<cf.getBunchSpace();

  return o;
}

#endif 
