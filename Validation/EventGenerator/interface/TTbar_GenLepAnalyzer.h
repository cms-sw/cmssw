// -*- C++ -*-
//
// Package:    TTbar_GenLepAnalyzer
// Class:      TTbar_GenLepAnalyzer
// 
/**\class TTbar_GenLepAnalyzer 

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Martijn Gosselink,,,
//         Created:  Thu May 10 17:15:16 CEST 2012
// $Id: TTbar_GenLepAnalyzer.h,v 1.1 2012/07/04 18:24:21 inugent Exp $
//
//
// Added to: Validation/EventGenerator by Ian M. Nugent June 28, 2012

#ifndef TTbar_GenLepAnalyzer_H
#define TTbar_GenLepAnalyzer_H


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
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <map>
#include <string>

//
// class declaration
//

class TTbar_GenLepAnalyzer : public edm::EDAnalyzer {
   public:
      explicit TTbar_GenLepAnalyzer(const edm::ParameterSet&);
      ~TTbar_GenLepAnalyzer();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      virtual void beginRun(edm::Run const&, edm::EventSetup const&);
      virtual void endRun(edm::Run const&, edm::EventSetup const&);
      virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);
      virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&);

      // ----------member data ---------------------------
      ///ME's "container"
      DQMStore *dbe;

      edm::InputTag leps_;
      std::map<std::string, MonitorElement*> hists_;
  
      bool do_e_, do_mu_, do_tau_, do_nu_e_, do_nu_mu_, do_nu_tau_;
      double pt_cut_, eta_cut_;
      int pdgid;
};

#endif
