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
		   const std::string subdet=" ", const range bunchRange =range(-999,999));

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

template <class T> 
MixCollection<T>::MixCollection(const CrossingFrame *cf, const std::string subdet, const std::pair<int,int> bunchRange) : 
  cf_(const_cast<CrossingFrame*>(cf))
{
  bunchRange_=bunchRange;

  //verify whether detector is present
  cf_->getSignal(subdet,signals_);
  if (!signals_) {
    std::cout<< " No detector "<<subdet<<" known in CrossingFrame!!"<<std::endl;
    return;
  }

  cf_->getPileups(subdet,pileups_);
  if (!pileups_) {
    std::cout<< " No pileups for detector "<<subdet<<" in CrossingFrame!!"<<std::endl;
    return;
  }

  //verify whether bunchrange is ok

  int firstcr=cf_->getFirstCrossingNr();
  range defaultrange=range(firstcr,firstcr+pileups_->size()-1);
  if (bunchRange_==range(-999,999)) bunchRange_=defaultrange;
  else if (bunchRange_!=defaultrange ) {
    int first=defaultrange.first;
    int last = defaultrange.second;
    if (bunchRange_.first<defaultrange.first)       std::cout <<" Existing runrange is "<<defaultrange.first<<", "<<defaultrange.second<<" you are asking for "<<bunchRange_.first<<", "<<bunchRange_.second<<", lower limit was reset!!"<<std::endl; //FIXME: who throws exception?
    else first=bunchRange_.first;
    if (bunchRange_.second>defaultrange.second)         std::cout <<" Existing runrange is "<<defaultrange.first<<", "<<defaultrange.second<<" you are asking for "<<bunchRange_.first<<", "<<bunchRange_.second<<", upper limit was reset!!"<<std::endl; //who throws exception?
    else last=bunchRange_.second;
    bunchRange_=range(first,last);
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

  // skip empty containers
  while (!pileupItr_->size()) {
    if (curBunchCrossing_==lastBunchCrossing_) return pMixItrEnd_;
    pileupItr_++;
    curBunchCrossing_++;
  }

  pMixItr_=(*pileupItr_).begin();
  pMixItrEnd_=(*pileupItr_).end();  // end of this container
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
#endif

