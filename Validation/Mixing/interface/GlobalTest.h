// -*- C++ -*-
//
// Class:      GlobalTest
//
/**\class GlobalTest

   Description: test suite for Mixing Module

*/
//
// Original Author:  Ursula Berthon
//         Created:  Fri Sep 23 11:38:38 CEST 2005
//
//


// system include files
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

//DQM services for histogram
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

class TFile;
class MonitorElement;

//
// class declaration
//

class GlobalTest : public DQMEDAnalyzer {
 public:
  explicit GlobalTest(const edm::ParameterSet&);
  ~GlobalTest();

  void bookHistograms(DQMStore::IBooker &,
      edm::Run const &, edm::EventSetup const &) override;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);

 private:
  std::string filename_;
  int minbunch_;
  int maxbunch_;
  TFile *histfile_;

  const static int nMaxH=10;
  MonitorElement * nrPileupsH_[nMaxH];
  MonitorElement * nrVerticesH_[nMaxH];
  MonitorElement * nrTracksH_[nMaxH];
  MonitorElement * trackPartIdH_[nMaxH];
  MonitorElement * caloEnergyEBH_[nMaxH];
  MonitorElement * caloEnergyEEH_[nMaxH];

  const static int nrHistos=6;
  char * labels[nrHistos];

  edm::EDGetTokenT<CrossingFrame<SimTrack> > cfTrackToken_;
  edm::EDGetTokenT<CrossingFrame<SimTrack> > cfVertexToken_;
  edm::EDGetTokenT<CrossingFrame<PCaloHit> > g4SimHits_EB_Token_;
  edm::EDGetTokenT<CrossingFrame<PCaloHit> > g4SimHits_EE_Token_;
};

