#ifndef TestAssociator_h
#define TestAssociator_h

/* \class TestAssociator
 *
 * \author Patrizia Azzi (INFN PD), Vincenzo Chiochia (Uni Zuerich), Bill Ford (Colorado)
 *
 *
 ************************************************************/

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"                                                                                                    

class TestAssociator : public edm::EDAnalyzer
{
 public:
  
  explicit TestAssociator(const edm::ParameterSet& conf);
  
  ~TestAssociator() override;

  void analyze(const edm::Event& e, const edm::EventSetup& c) override;

 private:
  
  TrackerHitAssociator::Config trackerHitAssociatorConfig_;
  bool doPixel_, doStrip_, useOTph2_;

  edm::EDGetTokenT<edmNew::DetSetVector<SiStripMatchedRecHit2D> > matchedRecHitToken;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripRecHit2D> > rphiRecHitToken,stereoRecHitToken;
  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelRecHit> > siPixelRecHitsToken;
  edm::EDGetTokenT<edmNew::DetSetVector<Phase2TrackerRecHit1D> > siPhase2RecHitsToken;

  template<typename rechitType>
    void printRechitSimhit(const edm::Handle<edmNew::DetSetVector<rechitType>> rechitCollection,
			   const char* rechitName, int hitCounter, TrackerHitAssociator& associate) const;

};

#endif
