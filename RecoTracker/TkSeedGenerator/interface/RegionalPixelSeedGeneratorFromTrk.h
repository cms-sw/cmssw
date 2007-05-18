#ifndef RegionalPixelSeedGeneratorFromTrk_h
#define RegionalPixelSeedGeneratorFromTrk_h

//
// Package:         RecoTracker/RegionalPixelSeedGeneratorFromTrk
// Class:           RegionalPixelSeedGeneratorFromTrk
//
// Description:     Calls RoadSeachSeedFinderAlgorithm
//                  to find TrackingSeeds.


#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/Common/interface/EDProduct.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoTracker/TkSeedGenerator/interface/CombinatorialRegionalSeedGeneratorFromPixel.h"

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/JetReco/interface/Jet.h"

class RegionalPixelSeedGeneratorFromTrk : public edm::EDProducer
{
 public:

  explicit RegionalPixelSeedGeneratorFromTrk(const edm::ParameterSet& conf);

  virtual ~RegionalPixelSeedGeneratorFromTrk();

  virtual void produce(edm::Event& e, const edm::EventSetup& c);

 private:
  edm::ParameterSet conf_;
  CombinatorialRegionalSeedGeneratorFromPixel   combinatorialSeedGenerator;
  double ptmin;
  bool vertexZconstrained;
  double vertexzDefault;
  std::string vertexSrc;
  double originradius;
  double halflength;
  double originz;
  double deltaEta;
  double deltaPhi;
  edm::InputTag trkSrc;
};

#endif

