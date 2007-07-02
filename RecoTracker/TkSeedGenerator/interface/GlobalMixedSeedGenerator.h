#ifndef GlobalMixedSeedGenerator_h
#define GlobalMixedSeedGenerator_h

//
// Package:         RecoTracker/GlobalPixelSeedGenerator
// Class:           GlobalPixelSeedGenerator
// 
// Description:     Calls RoadSeachSeedFinderAlgorithm
//                  to find TrackingSeeds.


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/EDProduct.h"
#include "RecoTracker/TkSeedGenerator/interface/CombinatorialSeedGeneratorFromMixed.h"


class GlobalMixedSeedGenerator : public edm::EDProducer
{
 public:

  explicit GlobalMixedSeedGenerator(const edm::ParameterSet& conf);

  virtual ~GlobalMixedSeedGenerator();

  virtual void produce(edm::Event& e, const edm::EventSetup& c);

 private:
  edm::ParameterSet conf_;
  CombinatorialSeedGeneratorFromMixed  combinatorialSeedGenerator;
};

#endif
