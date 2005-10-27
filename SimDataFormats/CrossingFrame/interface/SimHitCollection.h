#ifndef SIMHIT_COLLECTION_H
#define SIMHIT_COLLECTION_H
#include <utility>
#include <string>
#include <vector>

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"

class PSimHit;
class CrossingFrame;
using namespace edm;

class SimHitCollection {

private:

 public:
  typedef std::pair<int,int> range;
  SimHitCollection(const CrossingFrame *cf, 
		   const std::string subdet, const range bunchRange =range(-999,999));

  class SimHitItr;
  friend class SimHitItr;

  // nested class for iterator
  class SimHitItr {
  public:

    /** constructors */
    SimHitItr() {;}
    SimHitItr(std::vector<PSimHit>::iterator it) :    pSimHitItr_(it) {;}

     SimHitItr(SimHitCollection *shc, int firstcr,int lastcr) :     
       simHitCol_(shc),curBunchCrossing_(firstcr),lastBunchCrossing_(lastcr),first_(true) {;}


    /**Default destructor*/
    virtual ~SimHitItr() {;}

    /**operators*/
    const PSimHit* operator->() const { return pSimHitItr_.operator->(); }
    SimHitItr operator++ () {return next();}
    SimHitItr operator++ (int) {return next();}
    bool operator!= (const SimHitItr& itr){return pSimHitItr_!=itr.pSimHitItr_;}

    /**getters*/
    int bunch() const {return trigger_ ? 0: curBunchCrossing_;}
    bool getTrigger() const {return trigger_;}

  private:

    std::vector<PSimHit>::iterator pSimHitItr_;
    std::vector<PSimHit>::iterator pSimHitItrEnd_;
    SimHitCollection *simHitCol_;
    int curBunchCrossing_;
    int lastBunchCrossing_;
    bool first_;
    bool trigger_;
    std::vector<PSimHitContainer>::iterator pileupItr_;

    SimHitItr next();
    void reset() {;}
  };

  typedef SimHitItr iterator;
  iterator begin();
  iterator end() ;

 private:
       PSimHitContainer *signals_;
       std::vector<PSimHitContainer> *pileups_;
       CrossingFrame *cf_;
       range bunchRange_;
       PSimHitContainer *getSignal() {return signals_;}
       std::vector<PSimHitContainer> *getPileups() {return pileups_;}
};
#endif
