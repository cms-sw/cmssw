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
// $Id: TTbarSpinCorrHepMCAnalyzer.h,v 1.3 2012/10/16 15:08:10 inugent Exp $
//
//
// Added to: Validation/EventGenerator by Ian M. Nugent Oct 9, 2012


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

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

class TTbarSpinCorrHepMCAnalyzer : public edm::EDAnalyzer {
public:
  explicit TTbarSpinCorrHepMCAnalyzer(const edm::ParameterSet&);
  ~TTbarSpinCorrHepMCAnalyzer();

private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;

  virtual void beginRun(edm::Run const&, edm::EventSetup const&);
  virtual void endRun(edm::Run const&, edm::EventSetup const&);
  virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
  virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

  // ----------member data ---------------------------
  DQMStore *dbe;
  double weight ;

  MonitorElement*  nEvt;
  MonitorElement* _h_asym     ;
  MonitorElement* _h_deltaPhi ;

  MonitorElement* _h_llpairPt ;
  MonitorElement* _h_llpairM  ;

  edm::InputTag genEventInfoProductTag_,genParticlesTag_;
};
