#ifndef MIX_COLLECTION_H
#define MIX_COLLECTION_H
#include <utility>
#include <string>
#include <vector>

class CrossingFrame;

template <class T> 
class MixCollection {

private:

 public:
  typedef std::pair<int,int> range;
  MixCollection(const CrossingFrame *cf, 
		   const std::string subdet="", const range bunchRange =range(-999,999));

  range bunchrange() const {return bunchRange_;}

  class MixItr;
  friend class MixItr;

  // nested class for iterator
  class MixItr {
  public:

    /** constructors */
    MixItr() {;}
    MixItr(typename std::vector<T>::iterator it) :    pMixItr_(it) {;}
     MixItr(MixCollection *shc, int firstcr,int lastcr) :     
       mixCol_(shc),curBunchCrossing_(firstcr),lastBunchCrossing_(lastcr),first_(true) {;}


    /**Default destructor*/
    virtual ~MixItr() {;}

    /**operators*/
    const T* operator->() const { return pMixItr_.operator->(); }
    const T& operator*() const { return pMixItr_.operator*(); }
    MixItr operator++ () {return next();}
    MixItr operator++ (int) {return next();}
    bool operator!= (const MixItr& itr){return pMixItr_!=itr.pMixItr_;}

    /**getters*/
    int bunch() const {return trigger_ ? 0: curBunchCrossing_;}
    bool getTrigger() const {return trigger_;}

  private:

    typename std::vector<T>::iterator pMixItr_;
    typename std::vector<T>::iterator pMixItrEnd_;
    MixCollection *mixCol_;
    int curBunchCrossing_;
    int lastBunchCrossing_;
    bool first_;
    bool trigger_;
    typename std::vector<std::vector<T> >::iterator pileupItr_;

    MixItr next();
    void reset() {;}
  };

  typedef MixItr iterator;
  iterator begin();
  iterator end() ;

 private:
       std::vector<T>  *signals_;
       std::vector<std::vector<T> > *pileups_;
       CrossingFrame *cf_;
       range bunchRange_;
       std::vector<T>  *getSignal() {return signals_;}
       std::vector<std::vector<T> > *getPileups() {return pileups_;}
};


#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
//
// Exceptions
//
#include "FWCore/Utilities/interface/Exception.h"


template <class T> 
MixCollection<T>::MixCollection(const CrossingFrame *cf, const std::string subdet, const std::pair<int,int> bunchRange) : 
  cf_(const_cast<CrossingFrame*>(cf))
{
  bunchRange_=bunchRange;

  //verify whether bunchrange is ok
  range defaultrange=cf_->getBunchRange();
  if (bunchRange_==range(-999,999)) bunchRange_=defaultrange;
  else if (bunchRange_!=defaultrange ) {
    int first=defaultrange.first;
    int last = defaultrange.second;
    if (bunchRange_.first<defaultrange.first || bunchRange_.second>defaultrange.second )  throw cms::Exception("BadRunRange")<<" You are asking for a runrange ("<<bunchRange_.first<<","<<bunchRange_.second<<"), outside of the existing runrange ("<<defaultrange.first<<", "<<defaultrange.second<<")\n";
    bunchRange_=range(first,last);
  }


  //verify whether detector is known
  if (subdet.size()>0 && !cf_->knownDetector(subdet))
     throw cms::Exception("UnknownSubdetector")<< " No detector "<<subdet<<" known in CrossingFrame\n";


  //verify whether detector is present
  cf_->getSignal(subdet,signals_);
  cf_->getPileups(subdet,pileups_);
  if (!signals_->size()) {
    std::cout<<" No signal for subdetector "<<subdet<<std::endl;
    return;
  }
 
  // this should never happen!!
  if (!pileups_->size()) {
     throw cms::Exception("UnknownSubdetector")<< " Should not happen: no pileups for detector "<<subdet<<" in CrossingFrame\n";
    return;
  }


}
template <class T>
typename MixCollection<T>::MixItr MixCollection<T>::MixItr::next() {

  // initialisation
  if (first_) {
    first_=false;
    trigger_=true;
    pMixItr_=(mixCol_->getSignal())->begin();
    pMixItrEnd_=(mixCol_->getSignal())->end();
    if((mixCol_->getSignal())->size()) return *this;
  } else {
    if (++pMixItr_!=pMixItrEnd_) return *this;
  }
  
  // container changes
  if (trigger_) {
    trigger_=false;
    pileupItr_=(mixCol_->getPileups())->begin();
  } else {
    if (curBunchCrossing_==lastBunchCrossing_) return pMixItrEnd_;
    pileupItr_++;
    curBunchCrossing_++;
  }
  pMixItr_=(*pileupItr_).begin();
  pMixItrEnd_=(*pileupItr_).end();  // end of this container

  // skip empty containers
  while (!pileupItr_->size()) {
    if (curBunchCrossing_==lastBunchCrossing_) return pMixItrEnd_;
    pileupItr_++;
    curBunchCrossing_++;
    pMixItr_=(*pileupItr_).begin();
    pMixItrEnd_=(*pileupItr_).end();  // end of this container
  }

  return *this;
}

template <class T>
typename MixCollection<T>::MixItr MixCollection<T>::begin() {
  return MixItr(this,bunchRange_.first,bunchRange_.second)++;
}

template <class T>
typename  MixCollection<T>::MixItr MixCollection<T>::end() {
  typename std::vector<std::vector<T> >::iterator it=pileups_->begin();
  for (int i=bunchRange_.first;i<bunchRange_.second;i++) it++;
  typename std::vector<T>::iterator itend=it->end();
  return itend;
}

#include<iosfwd>
#include<iostream>
template <class T>
std::ostream &operator<<(std::ostream& o, const MixCollection<T>& col)
{
  o << "MixCollection with bunchRange: "<<(col.bunchrange()).first<< "," << (col.bunchrange()).second;

  return o;
}

#endif

