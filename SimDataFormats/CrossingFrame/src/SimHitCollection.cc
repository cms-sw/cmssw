#include "SimDataFormats/CrossingFrame/interface/SimHitCollection.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"

SimHitCollection::SimHitCollection(const CrossingFrame *cf, const std::string subdet, const std::pair<int,int> bunchRange) : 
  cf_(const_cast<CrossingFrame*>(cf))
{
  bunchRange_=bunchRange;

  //verify whether detector is present
  signals_=cf_->getSignalSimHits(subdet);
  if (!signals_) {
    std::cout<< " No detector "<<subdet<<" known in CrossingFrame!!"<<std::endl;
    return;
  }
  pileups_=cf_->getPileupSimHits(subdet);
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

SimHitCollection::SimHitItr SimHitCollection::SimHitItr::next() {

  // initialisation
  if (first_) {
    first_=false;
    trigger_=true;
    pSimHitItr_=(simHitCol_->getSignal())->begin();
    pSimHitItrEnd_=(simHitCol_->getSignal())->end();
    if((simHitCol_->getSignal())->size()) return *this;
  } else {
    if (++pSimHitItr_!=pSimHitItrEnd_) return *this;
  }
  
  // container changes
  if (trigger_) {
    trigger_=false;
    pileupItr_=(simHitCol_->getPileups())->begin();
  } else {
    if (curBunchCrossing_==lastBunchCrossing_) return pSimHitItrEnd_;
    pileupItr_++;
    curBunchCrossing_++;
  }

  // skip empty containers
  while (!pileupItr_->size()) {
    if (curBunchCrossing_==lastBunchCrossing_) return pSimHitItrEnd_;
    pileupItr_++;
    curBunchCrossing_++;
  }

  pSimHitItr_=(*pileupItr_).begin();
  pSimHitItrEnd_=(*pileupItr_).end();  // end of this container
  return *this;
}

SimHitCollection::SimHitItr SimHitCollection::begin() {
  return SimHitItr(this,bunchRange_.first,bunchRange_.second)++;
}

 SimHitCollection::SimHitItr SimHitCollection::end() {
  std::vector<PSimHitContainer>::const_iterator it=pileups_->begin();
  for (int i=bunchRange_.first;i<bunchRange_.second;i++) it++;
  std::vector<PSimHit>::iterator itend=it->end();
  return itend;
}
