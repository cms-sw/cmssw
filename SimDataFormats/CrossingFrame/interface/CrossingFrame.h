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
 * \version   3rd Version Nov 2007
 *
 ************************************************************/

#include "DataFormats/Provenance/interface/EventID.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>
#include <string>
#include <iostream>
#include <utility>

template <class T> 
class CrossingFrame 
{ 

 public:
  // con- and destructors

  CrossingFrame():  firstCrossing_(0), lastCrossing_(0), bunchSpace_(75),subdet_(""),idFirstPileup_(0,0),pileupFileNr_(0) {;}
  CrossingFrame(int minb, int maxb, int bunchsp, std::string subdet );

  ~CrossingFrame() {;}

  void addSignals(const std::vector<T> * vec,edm::EventID id);

  void addPileups(const int bcr,const std::vector<T> * vec, unsigned int evtId,int vertexoffset=0,bool checkTof=false);
  
  void print(int level=0) const ;
  void setBcrOffset() {
    pileupOffsetsBcr_.push_back(pileups_.size());
  }
  void setSourceOffset(const unsigned int s) {
    pileupOffsetsSource_[s].push_back(pileups_.size());//FIXME: timing
  }

  //getters
  edm::EventID getEventID() const {return id_;}
  std::pair<int,int> getBunchRange() const {return std::pair<int,int>(firstCrossing_,lastCrossing_);}
  int getBunchSpace() const {return bunchSpace_;}
  int getBunchCrossing(unsigned int ip) const;
  int getSourceType(unsigned int ip) const;
  void getSignal(typename std::vector<T>::const_iterator &first,typename std::vector<T>::const_iterator &last) const {
    first=signals_.begin();
    last=signals_.end();
  }
  void getPileups(typename std::vector<T>::const_iterator &first, typename std::vector<T>::const_iterator &last) const;
  unsigned int getNrSignals() const {return signals_.size();} 
  unsigned int getNrPileups() const {return pileups_.size();} 
  unsigned int getNrPileups(int bcr) const {
    return bcr==lastCrossing_ ? pileups_.size()-pileupOffsetsBcr_[lastCrossing_-firstCrossing_] :pileupOffsetsBcr_[bcr-firstCrossing_+1]- pileupOffsetsBcr_[bcr-firstCrossing_];} 

  // limits for tof to be considered for trackers
  static const int lowTrackTof; //nsec
  static const int highTrackTof;
  static const int minLowTof;
  static const int limHighLowTof;
					    
 private:
  // general information
  int firstCrossing_;
  int lastCrossing_;
  int bunchSpace_;  //in nsec
  std::string subdet_;  // for PSimHits/PCaloHits
  edm::EventID id_; // event id of the signal event

  // for playback option
  edm::EventID idFirstPileup_;   // EventId fof the first pileup event used for this signal event
  unsigned int pileupFileNr_;    // ordinal number of the pileup file this event was in

  static const unsigned int maxNbSources =4 ;

  // signal
  std::vector<T>  signals_; 

  //pileup
  std::vector<T>  pileups_;  
  std::vector<unsigned int> pileupOffsetsBcr_;
  std::vector<unsigned int> pileupOffsetsSource_[maxNbSources]; //one per source
};

//==============================================================================
//                              implementations
//==============================================================================

template <class T> 
CrossingFrame<T>::CrossingFrame(int minb, int maxb, int bunchsp, std::string subdet ):firstCrossing_(minb), lastCrossing_(maxb), bunchSpace_(bunchsp),subdet_(subdet),idFirstPileup_(0,0),pileupFileNr_(0) {
  //FIXME: should we force around 0 or so??
  pileupOffsetsBcr_.reserve(-firstCrossing_+lastCrossing_+1);
}

template <class T> 
void CrossingFrame<T>::addSignals(const std::vector<T> * vec,edm::EventID id){
  id_=id;
  signals_=*vec;
}

template <class T>  
void  CrossingFrame<T>::getPileups(typename std::vector<T>::const_iterator &first,typename std::vector<T>::const_iterator &last) const {
  first=pileups_.begin();
  last=pileups_.end();
}

template <class T> 
void CrossingFrame<T>::print(int level) const {
}

template <class T> 
int  CrossingFrame<T>::getSourceType(unsigned int ip) const {
  // decide to which source belongs object with index ip in the pileup vector
  // pileup=0, cosmics=2, beam halo+ =3, beam halo- =4
  unsigned int bcr= getBunchCrossing(ip)-firstCrossing_; //starts at 0

    for (unsigned int i=0;i<maxNbSources-1;++i) {
      if (ip>=(pileupOffsetsSource_[i])[bcr] && ip <(pileupOffsetsSource_[i+1])[bcr]) return i;
    }
    return maxNbSources-1;
}

template <class T>   
int CrossingFrame<T>::getBunchCrossing(unsigned int ip) const {
  // return the bcr for a certain position in the pileup vector
    for (unsigned int ii=1;ii<pileupOffsetsBcr_.size();ii++){
      if (ip>=pileupOffsetsBcr_[ii-1] && ip<pileupOffsetsBcr_[ii]) return ii+firstCrossing_-1;
    }
    if (ip<pileups_.size()) return lastCrossing_;
    else return 999;
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
