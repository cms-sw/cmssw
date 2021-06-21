// -*- C++ -*-
//
// Package:    CherenkovAnalysis
// Class:      CherenkovAnalysis
//
/**\class CherenkovAnalysis CherenkovAnalysis.cpp
SimG4CMS/CherenkovAnalysis/test/CherenkovAnalysis.cpp

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Frederic Ronga
//         Created:  Wed Mar 12 17:39:55 CET 2008
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include <TH1F.h>

class CherenkovAnalysis : public edm::EDAnalyzer {
public:
  explicit CherenkovAnalysis(const edm::ParameterSet &);
  ~CherenkovAnalysis() override {}

private:
  edm::EDGetTokenT<edm::PCaloHitContainer> tok_calo_;
  TH1F *hEnergy_;
  double maxEnergy_;
  int nBinsEnergy_;

  TH1F *hTimeStructure_;

  void beginJob() override {}
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override {}
};

//__________________________________________________________________________________________________
CherenkovAnalysis::CherenkovAnalysis(const edm::ParameterSet &iConfig)
    : maxEnergy_(iConfig.getParameter<double>("maxEnergy")),
      nBinsEnergy_(iConfig.getParameter<unsigned>("nBinsEnergy")) {
  tok_calo_ = consumes<edm::PCaloHitContainer>(iConfig.getParameter<edm::InputTag>("caloHitSource"));

  // Book histograms
  edm::Service<TFileService> tfile;

  if (!tfile.isAvailable())
    throw cms::Exception("BadConfig") << "TFileService unavailable: "
                                      << "please add it to config file";

  hEnergy_ = tfile->make<TH1F>("hEnergy", "Total energy deposit [GeV]", nBinsEnergy_, 0, maxEnergy_);
  hTimeStructure_ = tfile->make<TH1F>("hTimeStructure", "Time structure [ns]", 100, 0, 0.3);
}

//__________________________________________________________________________________________________
void CherenkovAnalysis::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  edm::Handle<edm::PCaloHitContainer> pCaloHits;
  iEvent.getByToken(tok_calo_, pCaloHits);

  double totalEnergy = 0;

  // Loop on all hits and calculate total energy loss
  edm::PCaloHitContainer::const_iterator it = pCaloHits.product()->begin();
  edm::PCaloHitContainer::const_iterator itend = pCaloHits.product()->end();
  for (; it != itend; ++it) {
    totalEnergy += (*it).energy();
    hTimeStructure_->Fill((*it).time(),
                          (*it).energy());  // Time weighted by energy...
    //     edm::LogInfo("CherenkovAnalysis") << "Time = " << (*it).time() <<
    //     std::endl;
  }
#ifdef EDM_ML_DEBUG
  edm::LogVerbatim("CherenkovAnalysis") << "CherenkovAnalysis::Total energy = " << totalEnergy;
#endif
  hEnergy_->Fill(totalEnergy);
}

DEFINE_FWK_MODULE(CherenkovAnalysis);
