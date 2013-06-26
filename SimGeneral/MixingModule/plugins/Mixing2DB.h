#ifndef Mixing2DB_H
#define Mixing2DB_H
// -*- C++ -*-
//
// Package:    Mixing2DB
// Class:      Mixing2DB
// 
/**\class Mixing2DB Mixing2DB.cc SimGeneral/Mixing2DB/src/Mixing2DB.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Jean-Roch Vlimant,40 3-A28,+41227671209,
//         Created:  Mon Jan  9 17:27:59 CET 2012
// $Id: Mixing2DB.h,v 1.2 2013/03/01 00:12:02 wmtan Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
// class declaration
//

class Mixing2DB : public edm::EDAnalyzer {
   public:
      explicit Mixing2DB(const edm::ParameterSet&);
      ~Mixing2DB();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;


      // ----------member data ---------------------------
      edm::ParameterSet cfi_;
};

#endif
