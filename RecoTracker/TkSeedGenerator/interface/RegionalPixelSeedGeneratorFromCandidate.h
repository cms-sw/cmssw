#ifndef RegionalPixelSeedGeneratorFromCandidate_h
#define RegionalPixelSeedGeneratorFromCandidate_h


//
// Package:         RecoTracker/RegionalPixelSeedGeneratorFromCandidate
// Class:           RegionalPixelSeedGeneratorFromCandidate
// 
// Description:     Calls RoadSeachSeedFinderAlgorithm
//                  to find TrackingSeeds.

/**\class RegionalPixelSeedGeneratorFromCandidate 

 generates pixel seeds from a Candidate

 Implementation:
     uses candidate vertex for the seeding
*/

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

class RegionalPixelSeedGeneratorFromCandidate : public edm::EDProducer
{
 public:

  explicit RegionalPixelSeedGeneratorFromCandidate(const edm::ParameterSet& conf);

  virtual ~RegionalPixelSeedGeneratorFromCandidate();

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
  edm::InputTag candSrc;
//   std::string vertexSrc;
};

#endif
