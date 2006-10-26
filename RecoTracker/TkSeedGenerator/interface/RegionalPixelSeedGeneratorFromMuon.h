#ifndef RegionalPixelSeedGeneratorFromMuon_h
#define RegionalPixelSeedGeneratorFromMuon_h

//
// Package:         RecoTracker/RegionalPixelSeedGeneratorFromMuon
// Class:           RegionalPixelSeedGeneratorFromMuon
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

class RegionalPixelSeedGeneratorFromMuon : public edm::EDProducer
{
 public:

  explicit RegionalPixelSeedGeneratorFromMuon(const edm::ParameterSet& conf);

  virtual ~RegionalPixelSeedGeneratorFromMuon();

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
  edm::InputTag jetSrc;
  std::string vertexSrc;
};

#endif
