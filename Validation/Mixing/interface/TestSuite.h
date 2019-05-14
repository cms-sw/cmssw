// -*- C++ -*-
//
// Class:      TestSuite
//
/**\class TestSuite

   Description: test suite for Mixing Module

*/
//
// Original Author:  Ursula Berthon
//         Created:  Fri Sep 23 11:38:38 CEST 2005
//
//

// system include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/Vertex/interface/SimVertexContainer.h"

// DQM services for histogram
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

class TFile;

//
// class declaration
//

class TestSuite : public edm::EDAnalyzer {
public:
  explicit TestSuite(const edm::ParameterSet &);
  ~TestSuite() override;

  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void beginJob() override;
  void endJob() override;

private:
  std::string filename_;
  int bunchcr_;
  int minbunch_;
  int maxbunch_;
  DQMStore *dbe_;

  edm::EDGetTokenT<CrossingFrame<SimTrack>> cfTrackToken_;
  edm::EDGetTokenT<CrossingFrame<SimTrack>> cfVertexToken_;
  edm::EDGetTokenT<CrossingFrame<PSimHit>> g4SimHits_Token_;
  edm::EDGetTokenT<CrossingFrame<PCaloHit>> g4SimHits_Ecal_Token_;
  edm::EDGetTokenT<CrossingFrame<PCaloHit>> g4SimHits_HCal_Token_;
};
