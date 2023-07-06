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
// Updated April, 2020 by Ece Asilar and Gage DeZoort
// Updated July, 2023 by Gourab Saha

// user include files
//#include "FWCore/Framework/interface/EDAnalyzer.h"
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
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
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
  // https://twiki.cern.ch/twiki/bin/viewauth/CMS/TauIDRecommendationForRun2#Decay_Mode_Reconstruction
  int findDecayMode(int Nc, int Np, int Ng) {return (Ng == 0) ? 5*(Nc-1)+Np : -1;};

private:
  edm::EDGetTokenT<std::vector<pat::Tau> > tauCollection_;
  edm::EDGetTokenT<edm::View<reco::Candidate> > refCollectionInputTagToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex> > primaryVertexCollectionToken_;
  edm::EDGetTokenT<std::vector<reco::GenParticle> > prunedGenToken_;
  edm::EDGetTokenT<std::vector<reco::GenJet> > genJetsToken_;
  //edm::EDGetTokenT<std::vector<pat::PackedGenParticle> >packedGenToken_;

  std::map<std::string, MonitorElement *> ptMap, etaMap, phiMap, massMap, puMap;
  std::map<std::string, MonitorElement *> ptTightvsJetMap, phiTightvsJetMap, etaTightvsJetMap, massTightvsJetMap,
      puTightvsJetMap;
  std::map<std::string, MonitorElement *> ptTightvsEleMap, phiTightvsEleMap, etaTightvsEleMap, massTightvsEleMap,
      puTightvsEleMap;
  std::map<std::string, MonitorElement *> ptTightvsMuoMap, phiTightvsMuoMap, etaTightvsMuoMap, massTightvsMuoMap,
      puTightvsMuoMap;
  std::map<std::string, MonitorElement *> ptMediumvsJetMap, phiMediumvsJetMap, etaMediumvsJetMap, massMediumvsJetMap,
      puMediumvsJetMap;
  std::map<std::string, MonitorElement *> ptMediumvsEleMap, phiMediumvsEleMap, etaMediumvsEleMap, massMediumvsEleMap,
      puMediumvsEleMap;
  std::map<std::string, MonitorElement *> ptMediumvsMuoMap, phiMediumvsMuoMap, etaMediumvsMuoMap, massMediumvsMuoMap,
      puMediumvsMuoMap;
  std::map<std::string, MonitorElement *> ptLoosevsJetMap, phiLoosevsJetMap, etaLoosevsJetMap, massLoosevsJetMap,
      puLoosevsJetMap;
  std::map<std::string, MonitorElement *> ptLoosevsEleMap, phiLoosevsEleMap, etaLoosevsEleMap, massLoosevsEleMap,
      puLoosevsEleMap;
  std::map<std::string, MonitorElement *> ptLoosevsMuoMap, phiLoosevsMuoMap, etaLoosevsMuoMap, massLoosevsMuoMap,
      puLoosevsMuoMap;
  std::map<std::string, MonitorElement *> decayModeFindingMap, decayModeMap, byDeepTau2018v2p5VSerawMap,
      byDeepTau2018v2p5VSjetrawMap, byDeepTau2018v2p5VSmurawMap, summaryMap;
  std::map<std::string, MonitorElement *> mtau_dm0Map, mtau_dm1p2Map, /*mtau_dm2Map,*/ mtau_dm5Map, mtau_dm6Map, /*mtau_dm7Map,*/ mtau_dm10Map, mtau_dm11Map;
  std::map<std::string, MonitorElement *> dmMigrationMap, ntau_vs_dmMap;
  std::map<std::string, MonitorElement *> pTOverProng_dm0Map, pTOverProng_dm1p2Map, pTOverProng_dm5Map, pTOverProng_dm6Map, /*pTOverProng_dm7Map,*/ pTOverProng_dm10Map, pTOverProng_dm11Map;

  edm::ParameterSet histoSettings_;
  std::string extensionName_;
  std::vector<edm::ParameterSet> discriminators_;
  std::vector<edm::ParameterSet> againstXs_;
  std::string qcd;
  std::string real_data;
  std::string real_eledata;
  std::string real_mudata;
  std::string ztt;
  std::string zee;
  std::string zmm;
};

#endif
