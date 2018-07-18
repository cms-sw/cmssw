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

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

template <class T>
class PCrossingFrame;

#include <vector>
#include <string>
#include <iostream>
#include <utility>
#include <algorithm>
#include <memory>


template <class T> 
class CrossingFrame 
{ 

 public:
  // con- and destructors

  CrossingFrame():  firstCrossing_(0), lastCrossing_(0), bunchSpace_(75),subdet_(""),maxNbSources_(0) {
}
  CrossingFrame(int minb, int maxb, int bunchsp, std::string subdet ,unsigned int maxNbSources);

  ~CrossingFrame() {;}

  void swap(CrossingFrame& other);

  CrossingFrame& operator=(CrossingFrame const& rhs);

  //standard version
  void addSignals(const std::vector<T> * vec,edm::EventID id);
  // version for HepMCProduct
  void addSignals(const T * vec,edm::EventID id);
 
  // standard version
  void addPileups(std::vector<T> const& vec);
  // version for HepMCProduct
  void addPileups(T const& product);

  void setTof( );

  // we keep the shared pointer in the object that will be only destroyed at the end of the event (transient object!)
  // because of HepMCProduct, we need 2 versions...
/*   void setPileupPtr(std::shared_ptr<edm::Wrapper<std::vector<T> > const> shPtr) {shPtrPileups_=shPtr;} */
/*   void setPileupPtr(std::shared_ptr<edm::Wrapper<T> const> shPtr) {shPtrPileups2_=shPtr;} */
  void setPileupPtr(std::shared_ptr<edm::Wrapper<std::vector<T> > const> shPtr) {shPtrPileups_.push_back( shPtr );}
  void setPileupPtr(std::shared_ptr<edm::Wrapper<T> const> shPtr) {shPtrPileups2_.push_back( shPtr );}
  // used in the Step2 to set the PCrossingFrame
  void setPileupPtr(std::shared_ptr<edm::Wrapper<PCrossingFrame<T> > const> shPtr);
  
  void print(int level=0) const ;

  void setBcrOffset() {
    pileupOffsetsBcr_.push_back(pileups_.size());
  }

  void setSourceOffset(const unsigned int s) {
    pileupOffsetsSource_[s].push_back(pileups_.size());
  }

  //getters
  edm::EventID getEventID() const {return id_;}
  std::pair<int,int> getBunchRange() const {return std::pair<int,int>(firstCrossing_,lastCrossing_);}
  int getBunchSpace() const {return bunchSpace_;}
  unsigned int getMaxNbSources() const {return maxNbSources_; }
  std::string getSubDet() const { return subdet_;}
  unsigned int getPileupFileNr() const {return pileupFileNr_;}
  edm::EventID getIdFirstPileup() const {return idFirstPileup_;}
  const std::vector<unsigned int>& getPileupOffsetsBcr() const {return pileupOffsetsBcr_;}
  const std::vector< std::vector<unsigned int> >& getPileupOffsetsSource() const {return pileupOffsetsSource_;} //one per source
  const std::vector<const T *>& getPileups() const {return pileups_;}
  const std::vector<const T *>& getSignal() const {return signals_;}
  
  
  void getSignal(typename std::vector<const T *>::const_iterator &first,typename std::vector<const T*>::const_iterator &last) const {
    first=signals_.begin();
    last=signals_.end();
  }
  void getPileups(typename std::vector<const T*>::const_iterator &first, typename std::vector<const T*>::const_iterator &last) const;
  unsigned int getNrSignals() const {return signals_.size();} 
  unsigned int getNrPileups() const {return pileups_.size();} 
  unsigned int getNrPileups(int bcr) const {
    return bcr==lastCrossing_ ? pileups_.size()-pileupOffsetsBcr_[lastCrossing_-firstCrossing_] :pileupOffsetsBcr_[bcr-firstCrossing_+1]- pileupOffsetsBcr_[bcr-firstCrossing_];} 

  // get pileup information in dependency from internal pointer
  int getBunchCrossing(unsigned int ip) const;

  int getSourceType(unsigned int ip) const;

  // get object in pileup when position in the vector is known (for DigiSimLink typically)

  const T & getObject(unsigned int ip) const { 
    //ip is position in the MixCollection (i.e. signal + pileup)
    if (ip>getNrSignals()+getNrPileups()) throw cms::Exception("BadIndex")<<"CrossingFrame::getObject called with an invalid index- index was "<<ip<<"!"; // ip >=0, since ip is unsigned
    if (ip<getNrSignals()) {
      return *(signals_[ip]);
    }
    else  {
      return *(pileups_[ip-getNrSignals()]);
    }
  }  


  // setters needed for step2 when using mixed secondary source  
  void setEventID(edm::EventID evId) { id_ = evId; }
  void setPileups(const std::vector<const T *>& p) { pileups_ = p; } 
  void setBunchSpace(int bSpace) { bunchSpace_ = bSpace; } 
  void setMaxNbSources(unsigned int mNbS) { maxNbSources_ = mNbS; } 
  void setSubDet(std::string det) { subdet_ = det; } 
  void setPileupFileNr(unsigned int pFileNr) { pileupFileNr_ = pFileNr;} 
  void setIdFirstPileup(edm::EventID idFP) {idFirstPileup_ = idFP;}
  void setPileupOffsetsBcr(const std::vector<unsigned int>& pOffsetsBcr) { pileupOffsetsBcr_ = pOffsetsBcr;}  
  void setPileupOffsetsSource(const std::vector< std::vector<unsigned int> >& pOffsetsS) { pileupOffsetsSource_ = pOffsetsS;}  //one per source
  void setBunchRange(std::pair<int,int> bunchRange) { firstCrossing_ = bunchRange.first;
  						      lastCrossing_ = bunchRange.second;} 
  
 private: 					    
  // please update the swap() function below if any data members are added.
  // general information
  int firstCrossing_;
  int lastCrossing_;
  int bunchSpace_;  //in nsec
  std::string subdet_;  // for PSimHits/PCaloHits
  edm::EventID id_; // event id of the signal event

  // for playback option
  edm::EventID idFirstPileup_;   // EventId fof the first pileup event used for this signal event
  unsigned int pileupFileNr_;    // ordinal number of the pileup file this event was in

  unsigned int maxNbSources_;

  // signal
  std::vector<const T * >  signals_; 

  //pileup
  std::vector<const T *>  pileups_; 
  std::vector< std::shared_ptr<edm::Wrapper<std::vector<T> > const> > shPtrPileups_; 
  std::vector< std::shared_ptr<edm::Wrapper<T> const> > shPtrPileups2_;   // fore HepMCProduct
/*   std::shared_ptr<edm::Wrapper<std::vector<T> > const> shPtrPileups_;  */
/*   std::shared_ptr<edm::Wrapper<T> const> shPtrPileups2_;   // fore HepMCProduct */
  std::shared_ptr<edm::Wrapper<PCrossingFrame<T> > const> shPtrPileupsPCF_;
//  std::shared_ptr<edm::Wrapper<PCrossingFrame<edm::HepMCProduct> const> shPtrPileupsHepMCProductPCF_;
  
  // these are informations stored in order to be able to have information
  // as a function of the position of an object in the pileups_ vector
  std::vector<unsigned int> pileupOffsetsBcr_;
  std::vector< std::vector<unsigned int> > pileupOffsetsSource_; //one per source
  
};

//==============================================================================
//                              implementations
//==============================================================================

template <class T> 
CrossingFrame<T>::CrossingFrame(int minb, int maxb, int bunchsp, std::string subdet ,unsigned int
maxNbSources):firstCrossing_(minb), lastCrossing_(maxb),
bunchSpace_(bunchsp),subdet_(subdet),maxNbSources_(maxNbSources) {
 pileupOffsetsSource_.resize(maxNbSources_);
 for (unsigned int i=0;i<maxNbSources_;++i)
   pileupOffsetsSource_[i].reserve(-firstCrossing_+lastCrossing_+1);

//FIXME: should we force around 0 or so??
  pileupOffsetsBcr_.reserve(-firstCrossing_+lastCrossing_+1);
}

template <typename T>
inline
void
CrossingFrame<T>::swap(CrossingFrame<T>& other) {
  std::swap(firstCrossing_, other.firstCrossing_);
  std::swap(lastCrossing_, other.lastCrossing_);
  std::swap(bunchSpace_, other.bunchSpace_);
  subdet_.swap(other.subdet_);
  std::swap(id_, other.id_);
  std::swap(idFirstPileup_, other.idFirstPileup_);
  std::swap(pileupFileNr_, other.pileupFileNr_);
  std::swap(maxNbSources_, other.maxNbSources_);
  signals_.swap(other.signals_);
  pileups_.swap(other.pileups_);
  shPtrPileups_.swap(other.shPtrPileups_);
  shPtrPileups2_.swap(other.shPtrPileups2_);
  shPtrPileupsPCF_.swap(other.shPtrPileupsPCF_);
  pileupOffsetsBcr_.swap(other.pileupOffsetsBcr_);
  pileupOffsetsSource_.resize(maxNbSources_);
  for (unsigned int i=0;i<pileupOffsetsSource_.size();++i) { 
    pileupOffsetsSource_[i].swap(other.pileupOffsetsSource_[i]);
  }
}

template <typename T>
inline
CrossingFrame<T>&
CrossingFrame<T>::operator=(CrossingFrame<T> const& rhs) {
  CrossingFrame<T> temp(rhs);
  this->swap(temp);
  return *this;
}


template <class T>  
void  CrossingFrame<T>::getPileups(typename std::vector<const T *>::const_iterator &first,typename std::vector<const T *>::const_iterator &last) const {
  first=pileups_.begin();
  last=pileups_.end();
}

template <class T> 
void CrossingFrame<T>::print(int level) const {
}

template <class T> 
int  CrossingFrame<T>::getSourceType(unsigned int ip) const {
  // ip is position in the pileup vector
  // decide to which source belongs object with index ip in the pileup vector
  // pileup=0, cosmics=1, beam halo+ =2, beam halo- =3 forward =4
  unsigned int bcr= getBunchCrossing(ip)-firstCrossing_; //starts at 0
  for (unsigned int i=0;i<pileupOffsetsSource_.size()-1;++i) {
    if (ip>=(pileupOffsetsSource_[i])[bcr] && ip <(pileupOffsetsSource_[i+1])[bcr]) return i;
  }
  return pileupOffsetsSource_.size()-1;
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

// Free swap function
template <typename T>
inline
void
swap(CrossingFrame<T>& lhs, CrossingFrame<T>& rhs) {
  lhs.swap(rhs);
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

#include "SimDataFormats/CrossingFrame/interface/PCrossingFrame.h"
template <class T>
void CrossingFrame<T>::setPileupPtr(std::shared_ptr<edm::Wrapper<PCrossingFrame<T> > const> shPtr) {shPtrPileupsPCF_=shPtr;}


template <class T>
void CrossingFrame<T>::addPileups(T const& product) {
  // default, valid for HepMCProduct
  pileups_.push_back(&product);
}

#ifndef __GCCXML__
template <class T>
void CrossingFrame<T>::addPileups(std::vector<T> const& product){
  for (auto const& item : product) {
    pileups_.push_back(&item);
  }
}
#endif

template <class T> 
void CrossingFrame<T>::addSignals(const std::vector<T> * vec,edm::EventID id){
  // valid (called) for all except HepMCProduct
  id_=id;
  for (unsigned int i=0;i<vec->size();++i) {
    signals_.push_back(&((*vec)[i]));
  }
}

template <class T> 
void CrossingFrame<T>::addSignals(const T * product,edm::EventID id){
  // valid (called) for all except HepMCProduct
  id_=id;
  signals_.push_back(product);
}

template <class T>
void CrossingFrame<T>::setTof() {;}

#endif 
