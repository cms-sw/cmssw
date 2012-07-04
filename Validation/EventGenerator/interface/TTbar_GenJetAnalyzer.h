// -*- C++ -*-
//
// Package:    ObjectAnalyzer
// Class:      TTbar_GenJetAnalyzer
// 
/**\class TTbar_GenJetAnalyzer 

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Martijn Gosselink,,,
//         Created:  Thu May 10 17:15:16 CEST 2012
// $Id: TTbar_GenJetAnalyzer.cc,v 1.3 2012/05/29 08:17:01 mgosseli Exp $
//
//
// Added to: Validation/EventGenerator by Ian M. Nugent June 28, 2012

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

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/Jet.h"


#include <map>
#include <string>


//
// class declaration
//

class TTbar_GenJetAnalyzer : public edm::EDAnalyzer {
   public:
      explicit TTbar_GenJetAnalyzer(const edm::ParameterSet&);
      ~TTbar_GenJetAnalyzer();

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

      edm::InputTag jets_;
      std::map<std::string, MonitorElement*> hists_;

      double weight ;
};

