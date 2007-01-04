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
  MixCollection();
  MixCollection(const CrossingFrame *cf, 
		const std::string subdet="", const range bunchRange =range(-999,999));
  MixCollection(const CrossingFrame *cf, 
		std::vector<std::string> subdets, const range bunchRange =range(-999,999));

  range bunchrange() const {return bunchRange_;}
  int size() const {return sizeSignal() + sizePileup();}
  int sizePileup() const;
  int sizeSignal() const;
  // false if at least one of the subdetectors was not found in registry
  bool inRegistry() const {return inRegistry_;}

  class MixItr;
  friend class MixItr;

  // nested class for iterator
  class MixItr {
  public:

    /** constructors */
    MixItr():first_(true) {;}
    MixItr(typename std::vector<T>::const_iterator it) :    pMixItr_(it),first_(true) {;}
    MixItr(MixCollection *shc, int firstcr,int lastcr) :     
      mixCol_(shc),curBunchCrossing_(firstcr),lastBunchCrossing_(lastcr),first_(true),iSignal_(0),iPileup_(0) {;}


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

    typename std::vector<T>::const_iterator pMixItr_;
    typename std::vector<T>::const_iterator pMixItrEnd_;
    MixCollection *mixCol_;
    int curBunchCrossing_;
    int lastBunchCrossing_;
    bool first_;
    int iSignal_, iPileup_;
    bool trigger_;
    
    typename std::vector<std::vector<T> >::const_iterator pileupItr_;

    MixItr next();
    void reset() {;}
    const std::vector<T> * getNewSignal();
    const std::vector<std::vector<T> > * getNewPileups();
  };

  typedef MixItr iterator;
  iterator begin();
  iterator end() ;

 private:
  void init( const range bunchRange);
  bool testSubdet( const std::string subdet);
  std::vector<T>  *getSignal() {return signals_;}
  const std::vector<T>  *getNewSignal(int signal);
  std::vector<std::vector<T> > *getPileups() {return pileups_;}
  const std::vector<std::vector<T> > *getNewPileups(int pileup);
  const std::vector<T>  *signals_;
  const std::vector<std::vector<T> > *pileups_;
  CrossingFrame *cf_;
  range bunchRange_;
  bool inRegistry_;
  int nrDets_;
  std::vector<std::string> subdets_;
};


#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
//
// Exceptions
//
#include "FWCore/Utilities/interface/Exception.h"
template <class T> 
MixCollection<T>::MixCollection() : 
  cf_(0),bunchRange_(0,0), nrDets_(0),inRegistry_(false)
{
}


template <class T> 
MixCollection<T>::MixCollection(const CrossingFrame *cf, const std::string subdet, const std::pair<int,int> bunchRange) : 
  cf_(const_cast<CrossingFrame*>(cf)),inRegistry_(false)
{
  init(bunchRange);

  //set necessary variables
  if (testSubdet(subdet) ) {
    nrDets_=1;
    subdets_.push_back(subdet);
    inRegistry_=true; 
   }
} 

template <class T> 
MixCollection<T>::MixCollection(const CrossingFrame *cf, std::vector<std::string> subdets, const std::pair<int,int> bunchRange) : 
  cf_(const_cast<CrossingFrame*>(cf)),inRegistry_(false)
{

  init(bunchRange);

  //set necessary variables
  for (unsigned int i=0;i<subdets.size();++i) {
    if (testSubdet(subdets[i])) {
      nrDets_++;
      subdets_.push_back(subdets[i]);
      inRegistry_=true;  // true if at least one is present
    }
  }
}

template <class T> 
void MixCollection<T>::init( const std::pair<int,int> bunchRange) {

  nrDets_=0;
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
}

template <class T> 
bool MixCollection<T>::testSubdet( const std::string subdet) {
  //verify whether detector is known
  //???????  if ( strstr(typeid(T).name(),"Hit")  && !cf_->knownDetector(subdet)) {
    //      throw cms::Exception("UnknownSubdetector")<< " No detector '"<<subdets_[signal]<<"' for hits known in CrossingFrame (must be non-blank)\n";
  // known detector has a meaning only for PSimHit/PCaloHit type
  if (!cf_->knownDetector(subdet) && strstr(typeid(T).name(),"Hit")) {
    std::cout<<"Warning: detector type "<<subdet<<" is unknown!"<<std::endl;
    return false;
  } else {

    //verify whether detector/T type correspond
    std::string type=cf_->getType(subdet);

    if (!type.empty()) { //test valid only for SimHits and CaloHits
      if (!strstr(typeid(T).name(),type.c_str()))
	throw cms::Exception("TypeMismatch")<< "Given template type "<<type<<" does not correspond to detector "<<subdet<<"\n";
    }
  }
  return true;

}

template <class T>  int  MixCollection<T>::sizePileup() const {
  int s=0; 
  const std::vector<std::vector<T> > *pils=0;
  for (int i=0;i<nrDets_;++i) {
    cf_->getPileups(subdets_[i],pils);
    for (unsigned int j=0;j<pils->size();++j) {
      s+=(*pils)[j].size(); 
    }
  }
  return s;
}

template <class T>  int  MixCollection<T>::sizeSignal() const {
  int s=0; 
  const std::vector<T>  *sigs=0;
  for (int i=0;i<nrDets_;++i) {
    cf_->getSignal(subdets_[i],sigs);
    s+=sigs->size();  
  }
  return s;
}

template <class T>
const std::vector<T> * MixCollection<T>::getNewSignal(int signal) {
  // gets signal collection 
  // at the same time we verify that input is coherent

    if (signal>=nrDets_) return NULL;

    //verify whether detector is known
    if ( strstr(typeid(T).name(),"Hit")  && !cf_->knownDetector(subdets_[signal]))
      throw cms::Exception("UnknownSubdetector")<< " No detector '"<<subdets_[signal]<<"' for hits known in CrossingFrame (must be non-blank)\n";

    //verify whether detector/T type correspond
    std::string type=cf_->getType(subdets_[signal]);

    if (!type.empty()) { //test only for SimHits and CaloHits
      if (!strstr(typeid(T).name(),type.c_str()))
	throw cms::Exception("TypeMismatch")<< "Given template type "<<type<<" does not correspond to detecetor "<<subdets_[signal]<<"\n";
    }

    //everything ok
    cf_->getSignal(subdets_[signal],signals_);
    return signals_;
}

template <class T>
const std::vector<T> *  MixCollection<T>::MixItr::getNewSignal() {
  // gets the next signal collection with non-zero size

  const std::vector<T> *signals;
  while (signals=mixCol_->getNewSignal(iSignal_++)) {
    if (signals->size()) return signals;
  }
  return NULL;
}

template <class T>
const std::vector<std::vector<T> > * MixCollection<T>::getNewPileups(int pileup) {
  // gets the next pileup collection 
  if (pileup>=nrDets_) return NULL;
  cf_->getPileups(subdets_[pileup],pileups_);
  return pileups_;
  
}

template <class T>
const std::vector<std::vector<T> > *  MixCollection<T>::MixItr::getNewPileups() {
  // gets the next pileup collection 
  const std::vector<std::vector<T> > * pileups;
  while (pileups=mixCol_->getNewPileups(iPileup_)) {
    iPileup_++;
    if (pileups->size()) return pileups;
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
    const std::vector<T> *p =this->getNewSignal();
    if (p) {
      pMixItr_=(p->begin());
      pMixItrEnd_=(p->end());
      return *this;  
    }
    trigger_=false;
    curBunchCrossing_=lastBunchCrossing_;
  }

  // pileup container changes
  unsigned int coll_size=0;
  while (coll_size==0) {
    if (curBunchCrossing_==lastBunchCrossing_) {
      const std::vector<std::vector<T> > *p =this->getNewPileups();
      if (!p) return pMixItrEnd_;
      pileupItr_=p->begin();
      curBunchCrossing_=mixCol_->bunchrange().first;
    } else {
      pileupItr_++;
      curBunchCrossing_++;
    }
    pMixItr_=(*pileupItr_).begin();
    pMixItrEnd_=(*pileupItr_).end();  // end of this container
    coll_size=(*pileupItr_).size();
  }

  return *this;
}

template <class T>
typename MixCollection<T>::MixItr MixCollection<T>::begin() {
  return MixItr(this,bunchRange_.first,bunchRange_.second)++;
}

template <class T>
typename  MixCollection<T>::MixItr MixCollection<T>::end() {
  if (nrDets_<=0)   return this->begin();
  const std::vector<std::vector<T> > * pil;
  cf_->getPileups(subdets_[nrDets_-1],pil);
  typename std::vector<std::vector<T> >::const_iterator it=pil->begin();
  for (int i=bunchRange_.first;i<bunchRange_.second;i++) it++;
  typename std::vector<T>::const_iterator itend=it->end();
  return itend;
}

#include<iosfwd>
#include<iostream>
template <class T>
std::ostream &operator<<(std::ostream& o, const MixCollection<T>& col)
{
  o << "MixCollection with bunchRange: "<<(col.bunchrange()).first<< "," << (col.bunchrange()).second <<" size of signal: "<<col.sizeSignal() <<" ,size of pileup: "<<col.sizePileup();

  return o;
}

#endif

