// -*- C++ -*-
//
// Package:    TTbar_P4Violation
// Class:      TTbar_P4Violation
// 
/**\class TTbar_P4Violation 

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Martijn Gosselink,,,
//         Created:  Fri Jan 20 12:52:00 CET 2012
// $Id: TTbar_P4Violation.cc,v 1.4 2012/05/18 22:08:18 mgosseli Exp $
//
//
// Added to: Validation/EventGenerator by Ian M. Nugent June 28, 2012

#ifndef TTbar_P4Violation_H
#define TTbar_P4Violation_H


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

//
// class declaration
//

class TTbar_P4Violation : public edm::EDFilter {
   public:
      explicit TTbar_P4Violation(const edm::ParameterSet&);
      ~TTbar_P4Violation();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      virtual bool beginRun(edm::Run&, edm::EventSetup const&);
      virtual bool endRun(edm::Run&, edm::EventSetup const&);
      virtual bool beginLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);
      virtual bool endLuminosityBlock(edm::LuminosityBlock&, edm::EventSetup const&);

      // ----------member data ---------------------------
};

#endif
