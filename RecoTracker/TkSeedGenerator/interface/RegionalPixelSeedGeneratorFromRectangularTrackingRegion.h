#ifndef RegionalPixelSeedGeneratorFromRectangularTrackingRegion_h
#define RegionalPixelSeedGeneratorFromRectangularTrackingRegion_h

//
// Package:         RecoTracker/RegionalPixelSeedGeneratorFromRectangularTrackingRegion
// Class:           RegionalPixelSeedGeneratorFromRectangularTrackingRegion
// 
// Description:     Calls RoadSeachSeedFinderAlgorithm
//                  to find TrackingSeeds.


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/EDProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/TkSeedGenerator/interface/CombinatorialRegionalSeedGeneratorFromPixel.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"

class RegionalPixelSeedGeneratorFromRectangularTrackingRegion : public edm::EDProducer
{
 public:

  explicit RegionalPixelSeedGeneratorFromRectangularTrackingRegion(const edm::ParameterSet& conf);

  virtual ~RegionalPixelSeedGeneratorFromRectangularTrackingRegion();

  virtual void produce(edm::Event& e, const edm::EventSetup& c);

 private:
  edm::ParameterSet conf_;
  CombinatorialRegionalSeedGeneratorFromPixel  combinatorialSeedGenerator;
  float ptmin;
  float originradius;
  float halflength;
  float originz;
  float deltaEta;
  float deltaPhi;
  edm::InputTag regSrc;
};

#endif
