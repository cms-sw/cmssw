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
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
//
// class declaration
//

class Mixing2DB : public edm::one::EDAnalyzer<> {
   public:
      explicit Mixing2DB(const edm::ParameterSet&);
      ~Mixing2DB() override;

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      void beginJob() override ;
      void analyze(const edm::Event&, const edm::EventSetup&) override;
      void endJob()  override;


      // ----------member data ---------------------------
      edm::ParameterSet cfi_;
};

#endif
