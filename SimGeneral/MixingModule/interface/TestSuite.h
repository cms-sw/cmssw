// -*- C++ -*-
//
// Class:      TestSuite
// 
/**\class TestSuite

 Description: test suite for Mixing Module

*/
//
// Original Author:  Ursula Berthon
//         Created:  Fri Sep 23 11:38:38 CEST 2005
// $Id: TestSuite.h,v 1.1 2005/10/10 16:32:04 uberthon Exp $
//
//


// system include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
class TFile;

namespace edm
{

//
// class declaration
//

class TestSuite : public edm::EDAnalyzer {
   public:
      explicit TestSuite(const edm::ParameterSet&);
      ~TestSuite();


      virtual void analyze(const edm::Event&, const edm::EventSetup&);
   private:
      int bunchcr_;
      std::string filename_;
      TFile *histfile_;
};
}//edm
