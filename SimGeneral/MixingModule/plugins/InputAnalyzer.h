// -*- C++ -*-
//
// Package:    InputAnalyzer
// Class:      InputAnalyzer
// 
/**\class InputAnalyzer InputAnalyzer.cc Analyzer/InputAnalyzer/src/InputAnalyzer.cc

*/
//
// Original Author:  Emilia Lubenova Becheva
//         Created:  Mon Apr 20 13:43:06 CEST 2009
//
//

#ifndef InputAnalyzer_h
#define InputAnalyzer_h


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "SimDataFormats/CrossingFrame/interface/PCrossingFrame.h"

//
// class decleration
//
namespace edm
{
class InputAnalyzer : public edm::EDAnalyzer {
   public:
      explicit InputAnalyzer(const edm::ParameterSet&);
      ~InputAnalyzer();


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() ;

      // ----------member data ---------------------------
      
      bool dataStep2_;
  edm::EDGetTokenT<PCrossingFrame<SimTrack>> labelPCF_;
  edm::EDGetTokenT<SimTrackContainer> labelSimTr_;

  
};
}//edm
#endif
