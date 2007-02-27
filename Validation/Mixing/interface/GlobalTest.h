// -*- C++ -*-
//
// Class:      GlobalTest
// 
/**\class GlobalTest

 Description: test suite for Mixing Module

*/
//
// Original Author:  Ursula Berthon
//         Created:  Fri Sep 23 11:38:38 CEST 2005
// $Id: GlobalTest.h,v 1.1 2006/03/14 14:23:26 uberthon Exp $
//
//


// system include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
class TFile;
class TH1I;
class  TH1F;

//
// class declaration
//

class GlobalTest : public edm::EDAnalyzer {
   public:
      explicit GlobalTest(const edm::ParameterSet&);
      ~GlobalTest();


      virtual void analyze(const edm::Event&, const edm::EventSetup&);
   private:
      std::string filename_;
      int minbunch_;
      int maxbunch_;
      TFile *histfile_;

      const static int nMaxH=10;
      TH1I * nrPileupsH_[nMaxH];
      TH1I * nrVerticesH_[nMaxH];
      TH1I * nrTracksH_[nMaxH];
      TH1I * trackPartIdH_[nMaxH];
      TH1F * caloEnergyEBH_[nMaxH];
      TH1F * caloEnergyEEH_[nMaxH];
};

