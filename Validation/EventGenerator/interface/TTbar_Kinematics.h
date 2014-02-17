// -*- C++ -*-
//
// Package:    TTbar_Kinematics
// Class:      TTbar_Kinematics
// 
/**\class TTbar_Kinematics 

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Martijn Gosselink,,,
//         Created:  Thu Jan 19 18:40:35 CET 2012
// $Id: TTbar_Kinematics.h,v 1.3 2012/08/24 21:47:01 wdd Exp $
//
//
// Added to: Validation/EventGenerator by Ian M. Nugent June 28, 2012


#ifndef TTbar_Kinematics_H
#define TTbar_Kinematics_H

// system include files
#include <memory>
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
//#include "Validation/EventGenerator/interface/WeightManager.h"



#include "TTree.h"
#include "TLorentzVector.h"

//
// class declaration
//

class TTbar_Kinematics : public edm::EDAnalyzer {
   public:
      explicit TTbar_Kinematics(const edm::ParameterSet&);
      ~TTbar_Kinematics();

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

      edm::InputTag hepmcCollection_;
      edm::InputTag genEventInfoProductTag_;


      double weight ;

      TLorentzVector tlv_Top       ;
      TLorentzVector tlv_TopBar    ;
      TLorentzVector tlv_Bottom    ;
      TLorentzVector tlv_BottomBar ;
      TLorentzVector tlv_Wplus     ;
      TLorentzVector tlv_Wmin      ;

      TLorentzVector tlv_TTbar     ;

      MonitorElement *nEvt;
      MonitorElement* hTopPt                 ;
      MonitorElement* hTopY                  ;
      MonitorElement* hTopMass               ;

      MonitorElement* hTTbarPt               ;
      MonitorElement* hTTbarY                ;
      MonitorElement* hTTbarMass             ;

      MonitorElement* hBottomPt              ;
      MonitorElement* hBottomEta             ;
      MonitorElement* hBottomY               ;
      MonitorElement* hBottomPz              ;
      MonitorElement* hBottomE               ;
      MonitorElement* hBottomMass            ;

      MonitorElement* hWplusPz               ;
      MonitorElement* hWminPz                ;

      MonitorElement* hBottomPtPz            ;
      MonitorElement* hBottomEtaPz           ;
      MonitorElement* hBottomEtaPt           ;
      MonitorElement* hBottomYPz             ;
      MonitorElement* hBottomMassPz          ;
      MonitorElement* hBottomMassEta         ;
      MonitorElement* hBottomMassY           ;
      MonitorElement* hBottomMassDeltaY      ;

};

#endif
