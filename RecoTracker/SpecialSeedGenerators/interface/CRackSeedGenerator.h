#ifndef CRackSeedGenerator_h
#define CRackSeedGenerator_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"


#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/SpecialSeedGenerators/interface/SeedGeneratorForCRack.h"


class CRackSeedGenerator : public edm::EDProducer
{
 public:

  explicit CRackSeedGenerator(const edm::ParameterSet& conf);

  ~CRackSeedGenerator() override;

  void produce(edm::Event& e, const edm::EventSetup& c) override;

 private:
  edm::ParameterSet conf_;
  SeedGeneratorForCRack  cosmic_seed;
  edm::EDGetTokenT<SiStripMatchedRecHit2DCollection> matchedrecHitsToken_;
  edm::EDGetTokenT<SiStripRecHit2DCollection> rphirecHitsToken_;
  edm::EDGetTokenT<SiStripRecHit2DCollection> stereorecHitsToken_;
};

#endif
