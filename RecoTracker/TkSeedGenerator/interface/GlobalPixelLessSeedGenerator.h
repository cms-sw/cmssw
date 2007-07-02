#ifndef GlobalPixelLessSeedGenerator_h
#define GlobalPixelLessSeedGenerator_h

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
#include "RecoTracker/TkSeedGenerator/interface/CombinatorialSeedGeneratorFromPixelLess.h"


class GlobalPixelLessSeedGenerator : public edm::EDProducer
{
 public:

  explicit GlobalPixelLessSeedGenerator(const edm::ParameterSet& conf);

  virtual ~GlobalPixelLessSeedGenerator();

  virtual void produce(edm::Event& e, const edm::EventSetup& c);

 private:
  edm::ParameterSet conf_;
  CombinatorialSeedGeneratorFromPixelLess  combinatorialSeedGenerator;
};

#endif
