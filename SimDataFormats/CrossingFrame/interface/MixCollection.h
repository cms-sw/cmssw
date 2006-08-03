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
  MixCollection(const CrossingFrame *cf, 
		std::vector<std::string> subdets, const range bunchRange =range(-999,999));
  void init( const range bunchRange);

  range bunchrange() const {return bunchRange_;}
  int size() {return sizeSignal() + sizePileup();}
  int sizePileup();
  int sizeSignal();

  class MixItr;
  friend class MixItr;

  // nested class for iterator
  class MixItr {
  public:

    /** constructors */
    MixItr():first_(true) {;}
    MixItr(typename std::vector<T>::iterator it) :    pMixItr_(it),first_(true) {;}
    MixItr(MixCollection *shc, int firstcr,int lastcr) :     
      mixCol_(shc),curBunchCrossing_(firstcr),lastBunchCrossing_(lastcr),first_(true) {;}


    /**Default destructor*/
    virtual ~MixItr() {;}

    /**operators*/
    const T* operator->() const {return pMixItr_.operator->(); }
    const T& operator*() const {return pMixItr_.operator*(); }
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
       std::vector<T>  *getNewSignal();
       std::vector<std::vector<T> > *getPileups() {return pileups_;}
       std::vector<std::vector<T> > *getNewPileups();
       int nrDets_, iSignal_, iPileup_;
       std::vector<std::string> subdets_;
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

  init(bunchRange);

  //set necessary variables
  nrDets_=1;
  subdets_.push_back(subdet);
  getNewPileups(); // get first pileup collection
} 

template <class T> 
MixCollection<T>::MixCollection(const CrossingFrame *cf, std::vector<std::string> subdets, const std::pair<int,int> bunchRange) : 
  cf_(const_cast<CrossingFrame*>(cf))
{

  init(bunchRange);

  //set necessary variables
  nrDets_=subdets.size();
  
  for (int i=0;i<nrDets_;++i) subdets_.push_back(subdets[i]);
  getNewPileups(); // get first pileup collection
}

template <class T> 
void MixCollection<T>::init( const std::pair<int,int> bunchRange) {

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

  iSignal_=0;
  iPileup_=0;
}

template <class T>  int  MixCollection<T>::sizePileup() {
  int s=0; 
  std::vector<std::vector<T> > *pils=0;
  for (int i=0;i<nrDets_;++i) {
    cf_->getPileups(subdets_[i],pils);
    for (unsigned int j=0;j<pils->size();++j) {
      s+=(*pils)[j].size(); 
    }
  }
  return s;
}

template <class T>  int  MixCollection<T>::sizeSignal() {
  int s=0; 
  std::vector<T>  *sigs=0;
  for (int i=0;i<nrDets_;++i) {
    cf_->getSignal(subdets_[i],sigs);
    s+=sigs->size();  
  }
  return s;
}


template <class T>
std::vector<T> * MixCollection<T>::getNewSignal() {
  // gets the next signal collection with non-zero size
  //at the same time we verify that input is coherent
  for (int i=iSignal_;i<nrDets_;i++) {
    //verify whether detector is known
    if ( strstr(typeid(T).name(),"Hit")  && !cf_->knownDetector(subdets_[iSignal_]))
      throw cms::Exception("UnknownSubdetector")<< " No detector '"<<subdets_[iSignal_]<<"' for hits known in CrossingFrame (must be non-blank)\n";

    //verify whether detector/T type correspond
    std::string type=cf_->getType(subdets_[iSignal_]);

    if (!type.empty()) { //test only for SimHits and CaloHits
      if (!strstr(typeid(T).name(),type.c_str()))
	throw cms::Exception("TypeMismatch")<< "Given template type "<<type<<" does not correspond to detecetor "<<subdets_[iSignal_]<<"\n";
    }

    //everything ok
    cf_->getSignal(subdets_[iSignal_++],signals_);
    if (signals_->size()) return signals_;
  }
  
  return NULL;
}

template <class T>
std::vector<std::vector<T> > * MixCollection<T>::getNewPileups() {
  // gets the next pileup collection with non-zero size
  for (int i=iPileup_;i<nrDets_;i++) {
    cf_->getPileups(subdets_[iPileup_++],pileups_);
    if (pileups_->size()) return pileups_;
  }
  return NULL;
}

template <class T>
typename MixCollection<T>::MixItr MixCollection<T>::MixItr::next() {
  // initialisation
  if (first_) {
    first_=false;
    trigger_=true;
  } else {
    if (++pMixItr_!=pMixItrEnd_) return *this;
  }
  // look whether there are more signal collections
  if (trigger_) {
    std::vector<T> *p =mixCol_->getNewSignal();
    if (p) {
      pMixItr_=(p->begin());
      pMixItrEnd_=(p->end());
      return *this;  
    }
  }

  // pileup container changes
  if(trigger_) {
    trigger_=false;
    pileupItr_=(mixCol_->getPileups())->begin();//first pileups are already there
  } else {
    if (curBunchCrossing_==lastBunchCrossing_) {
      std::vector<std::vector<T> > *p =mixCol_->getNewPileups();
      if (!p) return pMixItrEnd_;
      pileupItr_=(mixCol_->getPileups())->begin();
      curBunchCrossing_=mixCol_->bunchrange().first;
    } else {
      pileupItr_++;
      curBunchCrossing_++;
    }
  }
  pMixItr_=(*pileupItr_).begin();
  pMixItrEnd_=(*pileupItr_).end();  // end of this container

  return *this;
}

template <class T>
typename MixCollection<T>::MixItr MixCollection<T>::begin() {
  //FIXME hack to make it possible to iterate over
  // a collection more than once.
  // iSignal_ & iPileup_ really should be moved into MixItr
  // rpw Aug 3 2006
  iSignal_ = 0;
  iPileup_ = 0;
  return MixItr(this,bunchRange_.first,bunchRange_.second)++;
}

template <class T>
typename  MixCollection<T>::MixItr MixCollection<T>::end() {
  typename std::vector<std::vector<T> >::iterator it=getPileups()->begin();
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

