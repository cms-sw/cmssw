#ifndef TauValidationMiniAOD_h
#define TauValidationMiniAOD_h

// -*- C++ -*-
//
// Package:    TauValidationMiniAOD
// Class:      TauValidationMiniAOD
//
/* *\class TauValidationMiniAOD TauValidationMiniAOD.cc

 Description: EDAnalyzer to validate tau collection in miniAOD
 Implementation:

*/
// Original Author: Aniello Spiezia On August 13, 2019
// user include files

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/TauReco/interface/PFTau.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/Math/interface/deltaR.h"

// Include DQM core
#include <DQMServices/Core/interface/DQMStore.h>
#include <DQMServices/Core/interface/MonitorElement.h>
#include <DQMServices/Core/interface/DQMEDAnalyzer.h>

struct histoInfo {
  int nbins;
  double min;
  double max;
  histoInfo(int n, double m, double M) {
    nbins = n;
    min = m;
    max = M;
  }
  histoInfo(const edm::ParameterSet &config) {
    nbins = config.getParameter<int>("nbins");
    min = config.getParameter<double>("min");
    max = config.getParameter<double>("max");
  }
};

// class declaration
class TauValidationMiniAOD : public DQMEDAnalyzer {
public:
  explicit TauValidationMiniAOD(const edm::ParameterSet &);
  ~TauValidationMiniAOD() override;

  void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) override;

private:
  edm::EDGetTokenT<std::vector<pat::Tau> > tauCollection_;
  edm::EDGetTokenT<edm::View<reco::Candidate> > refCollectionInputTagToken_;
  std::map<std::string, MonitorElement *> ptMap, ptTightMap, etaMap, etaTightMap, phiMap, phiTightMap, massMap, massTightMap, decayModeFindingMap, decayModeMap,
      byDeepTau2017v2p1VSerawMap, byDeepTau2017v2p1VSjetrawMap, byDeepTau2017v2p1VSmurawMap, summaryMap;
  edm::ParameterSet histoSettings_;
  std::string extensionName_;
  std::vector<edm::ParameterSet> discriminators_;
  std::vector<edm::ParameterSet> againstXs_;
};

#endif
