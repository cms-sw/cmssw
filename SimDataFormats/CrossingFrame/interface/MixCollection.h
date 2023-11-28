#ifndef MIX_COLLECTION_H
#define MIX_COLLECTION_H
#include <utility>
#include <string>
#include <vector>

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"

template <class T>
class MixCollection {
private:
public:
  typedef std::pair<int, int> range;
  MixCollection();
  MixCollection(const CrossingFrame<T> *cf, const range bunchRange = range(-999, 999));
  MixCollection(const std::vector<const CrossingFrame<T> *> &cfs, const range bunchRange = range(-999, 999));

  range bunchrange() const { return bunchRange_; }
  int size() const { return sizeSignal() + sizePileup(); }
  int sizePileup() const;
  int sizeSignal() const;
  // false if at least one of the subdetectors was not found in registry
  bool inRegistry() const { return inRegistry_; }

  // get object the index of which -in the whole collection- is known
  const T &getObject(unsigned int ip) const {
    if (ip >= (unsigned int)size())
      throw cms::Exception("BadIndex")
          << "MixCollection::getObject called with an invalid index!";  // ip >= 0, since ip is unsigned
    int n = ip;
    /*
-    int iframe=0;
-    for (unsigned int ii=0;ii<crossingFrames_.size();++ii) {
-      iframe=ii;
-      int s=crossingFrames_[iframe]->getNrSignals()+crossingFrames_[iframe]->getNrPileups();
-      if (n<s) break;
*/
    for (unsigned int iframe = 0; iframe < crossingFrames_.size(); ++iframe) {
      int s = crossingFrames_[iframe]->getNrSignals();
      if (n < s)
        return crossingFrames_[iframe]->getObject(n);
      n = n - s;
    }
    /*
    return crossingFrames_[iframe]->getObject(n);
*/
    for (unsigned int iframe = 0; iframe < crossingFrames_.size(); ++iframe) {
      int s = crossingFrames_[iframe]->getNrSignals();
      int p = crossingFrames_[iframe]->getNrPileups();
      if (n < p)
        return crossingFrames_[iframe]->getObject(s + n);
      n = n - p;
    }
    throw cms::Exception("InternalError") << "MixCollection::getObject reached impossible condition";
  }

  class MixItr;
  friend class MixItr;

  // nested class for iterator
  class MixItr {
  public:
    struct end_tag {};

    /** constructors */
    MixItr() : first_(true), internalCtr_(0) { ; }
    MixItr(const MixCollection *shc, int nrDets, end_tag) : internalCtr2_(0) {
      for (int i = 0; i < nrDets; ++i) {
        const auto &cf = shc->crossingFrames_[i];
        internalCtr2_ += cf->getSignal().size() + cf->getPileups().size();
      }
    }
    MixItr(const MixCollection *shc, int nrDets)
        : mixCol_(shc), nrDets_(nrDets), first_(true), iSignal_(0), iPileup_(0), internalCtr_(0) {
      ;
    }

    /**Default destructor*/
    virtual ~MixItr() { ; }

    /**operators*/
    // default version valid for HepMCProduct
    const T *operator->() const { return *(pMixItr_.operator->()); }
    const T &operator*() const { return *(pMixItr_.operator*()); }
    const MixItr operator++() { return next(); }
    const MixItr operator++(int) { return next(); }
    bool operator!=(const MixItr &itr) { return internalCtr2_ != itr.internalCtr2_; }

    /**getters*/
    int bunch() const {
      if (trigger_)
        return 0;
      int bcr = myCF_->getBunchCrossing(internalCtr_);
      return bcr;
    }

    bool getTrigger() const { return trigger_; }

    int getSourceType() const { return (getTrigger() ? -1 : myCF_->getSourceType(internalCtr_)); }
    int getPileupEventNr() const { return (getTrigger() ? 0 : myCF_->getPileupEventNr(internalCtr_)); }

  private:
    typename std::vector<const T *>::const_iterator pMixItr_;
    typename std::vector<const T *>::const_iterator pMixItrEnd_;

    const CrossingFrame<T> *myCF_;
    const MixCollection *mixCol_;
    int nrDets_;
    bool first_;
    int iSignal_, iPileup_;
    bool trigger_;
    unsigned int internalCtr_;  //this is the internal counter pointing into the vector of piled up objects
    unsigned int internalCtr2_ =
        0;  // this is the internal counter for the number of iterated elements, needed for operator!=

    const MixItr next();
    void reset() { ; }
    bool getNewSignal(typename std::vector<const T *>::const_iterator &first,
                      typename std::vector<const T *>::const_iterator &last);

    bool getNewPileups(typename std::vector<const T *>::const_iterator &first,
                       typename std::vector<const T *>::const_iterator &last);
  };

  typedef MixItr iterator;
  iterator begin() const;
  iterator end() const;

private:
  void init(const range bunchRange);

  range bunchRange_;
  bool inRegistry_;
  int nrDets_;

  std::vector<const CrossingFrame<T> *> crossingFrames_;
};

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
//
// Exceptions
//
#include "FWCore/Utilities/interface/Exception.h"
template <class T>
MixCollection<T>::MixCollection() : bunchRange_(0, 0), inRegistry_(false), nrDets_(0) {
  crossingFrames_.push_back(NULL);
}

template <class T>
MixCollection<T>::MixCollection(const CrossingFrame<T> *cf, const std::pair<int, int> bunchRange)
    : inRegistry_(false), nrDets_(0) {
  nrDets_ = 1;
  inRegistry_ = true;
  if (cf) {
    crossingFrames_.push_back(cf);
    init(bunchRange);
  } else
    throw cms::Exception("InvalidPtr") << "Could not construct MixCollection for " << typeid(T).name()
                                       << ", pointer to CrossingFrame invalid!";
}

template <class T>
MixCollection<T>::MixCollection(const std::vector<const CrossingFrame<T> *> &cfs, const std::pair<int, int> bunchRange)
    : inRegistry_(false), nrDets_(0) {
  // first, verify that all CrossingFrames have the same bunchrange
  range bR = cfs[0]->getBunchRange();
  for (unsigned int i = 1; i < cfs.size(); ++i) {
    if (bR != cfs[i]->getBunchRange())
      throw cms::Exception("Incompatible CrossingFrames")
          << "You gave as input CrossingFrames with different bunchRanges!";
  }

  //set necessary variables
  for (unsigned int i = 0; i < cfs.size(); ++i) {
    nrDets_++;
    crossingFrames_.push_back(cfs[i]);
    inRegistry_ = true;  // true if at least one is present
  }

  init(bunchRange);
}

template <class T>
void MixCollection<T>::init(const std::pair<int, int> bunchRange) {
  bunchRange_ = bunchRange;

  //verify whether bunchrange is ok
  // in case of several crossingFrames, we have verified before that they have the same bunchrange
  range defaultrange = crossingFrames_[0]->getBunchRange();
  if (bunchRange_ == range(-999, 999))
    bunchRange_ = defaultrange;
  else if (bunchRange_ != defaultrange) {
    int first = defaultrange.first;
    int last = defaultrange.second;
    if (bunchRange_.first < defaultrange.first || bunchRange_.second > defaultrange.second)
      throw cms::Exception("BadRunRange")
          << " You are asking for a runrange (" << bunchRange_.first << "," << bunchRange_.second
          << "), outside of the existing runrange (" << defaultrange.first << ", " << defaultrange.second << ")\n";
    bunchRange_ = range(first, last);
  }
}

template <class T>
int MixCollection<T>::sizePileup() const {
  // get size cumulated for all subdetectors
  int s = 0;
  for (int i = 0; i < nrDets_; ++i) {
    s += crossingFrames_[i]->getNrPileups();
  }
  return s;
}

template <class T>
int MixCollection<T>::sizeSignal() const {
  int s = 0;
  for (int i = 0; i < nrDets_; ++i) {
    s += crossingFrames_[i]->getNrSignals();
  }
  return s;
}

template <class T>
bool MixCollection<T>::MixItr::getNewSignal(typename std::vector<const T *>::const_iterator &first,
                                            typename std::vector<const T *>::const_iterator &last) {
  // gets the next signal collection with non-zero size

  while (iSignal_ < nrDets_) {
    mixCol_->crossingFrames_[iSignal_]->getSignal(first, last);
    myCF_ = mixCol_->crossingFrames_[iSignal_];
    iSignal_++;
    if (first != last)
      return true;
  }
  return false;
}

template <class T>
bool MixCollection<T>::MixItr::getNewPileups(typename std::vector<const T *>::const_iterator &first,
                                             typename std::vector<const T *>::const_iterator &last) {
  // gets the next pileup collection , changing subdet if necessary
  while (iPileup_ < nrDets_) {
    mixCol_->crossingFrames_[iPileup_]->getPileups(first, last);
    myCF_ = mixCol_->crossingFrames_[iPileup_];
    iPileup_++;
    if (first != last)
      return true;
  }
  return false;
}

template <class T>
const typename MixCollection<T>::MixItr MixCollection<T>::MixItr::next() {
  // initialisation
  if (first_) {
    first_ = false;
    trigger_ = true;
  } else {
    ++internalCtr2_;
    if (!trigger_)
      internalCtr_++;
    if (++pMixItr_ != pMixItrEnd_)
      return *this;
  }

  // we have an end condition, look whether there are more collections
  bool ok;
  if (trigger_) {
    ok = this->getNewSignal(pMixItr_, pMixItrEnd_);
    if (ok)
      return *this;
    trigger_ = false;
  }
  ok = this->getNewPileups(pMixItr_, pMixItrEnd_);
  if (ok) {
    // debug print start
    //    for (auto dbIt=pMixItr_;dbIt!=pMixItrEnd_;++dbIt)  printf("Found pointer %p\n",(*dbIt));fflush(stdout);
    // debug print end
    internalCtr_ = 0;
  }
  return *this;  // with internalCtr2_ we can always return *this
}

template <class T>
typename MixCollection<T>::MixItr MixCollection<T>::begin() const {
  return MixItr(this, nrDets_)++;
}

template <class T>
typename MixCollection<T>::MixItr MixCollection<T>::end() const {
  return MixItr(this, nrDets_, typename MixItr::end_tag());
}

#include <iosfwd>
#include <iostream>
template <class T>
std::ostream &operator<<(std::ostream &o, const MixCollection<T> &col) {
  o << "MixCollection with bunchRange: " << (col.bunchrange()).first << "," << (col.bunchrange()).second
    << " size of signal: " << col.sizeSignal() << " ,size of pileup: " << col.sizePileup();

  return o;
}

#endif
