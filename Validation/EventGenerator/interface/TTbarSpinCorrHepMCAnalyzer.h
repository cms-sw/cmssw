// -*- C++ -*-
//
// Package:    TTbarSpinCorrHepMCAnalyzer
// Class:      TTbarSpinCorrHepMCAnalyzer
// 
/**\class TTbarSpinCorrHepMCAnalyzer TTbarAnalyzer.cc MCstuff/TTbarAnalyzer/src/TTbarAnalyzer.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Martijn Gosselink,,,
//         Created:  Thu Jan 19 18:40:35 CET 2012
//
//
// Added to: Validation/EventGenerator by Ian M. Nugent Oct 9, 2012


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

#include "CLHEP/Units/GlobalPhysicalConstants.h"

#include "TFile.h"
#include "TH1D.h"
#include "TLorentzVector.h"

//
// class declaration
//

class TTbarSpinCorrHepMCAnalyzer : public DQMEDAnalyzer {
public:
  explicit TTbarSpinCorrHepMCAnalyzer(const edm::ParameterSet&);
  ~TTbarSpinCorrHepMCAnalyzer() override;

  void bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------
  double weight ;

  MonitorElement*  nEvt;
  MonitorElement* _h_asym     ;
  MonitorElement* _h_deltaPhi ;

  MonitorElement* _h_llpairPt ;
  MonitorElement* _h_llpairM  ;

  edm::InputTag genEventInfoProductTag_,genParticlesTag_;

  edm::EDGetTokenT<GenEventInfoProduct> genEventInfoProductTagToken_;
  edm::EDGetTokenT<reco::GenParticleCollection> genParticlesTagToken_;

};
