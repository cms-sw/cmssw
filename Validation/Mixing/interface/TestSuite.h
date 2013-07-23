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
// $Id: TestSuite.h,v 1.4 2009/12/21 14:46:17 ebecheva Exp $
//
//


// system include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

//DQM services for histogram
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

class TFile;

//
// class declaration
//

class TestSuite : public edm::EDAnalyzer {
   public:
      explicit TestSuite(const edm::ParameterSet&);
      ~TestSuite();


      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void beginJob();
      virtual void endJob();
     
   private:
      std::string filename_;
      int bunchcr_;
      int minbunch_;
      int maxbunch_;
      DQMStore* dbe_;

      edm::InputTag cfTrackTag_;
      edm::InputTag cfVertexTag_;
};
