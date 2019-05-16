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
//
//
// Added to: Validation/EventGenerator by Ian M. Nugent June 28, 2012


#ifndef TTbar_Kinematics_H
#define TTbar_Kinematics_H

// system include files
#include <memory>
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

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

class TTbar_Kinematics : public DQMEDAnalyzer {
   public:
      explicit TTbar_Kinematics(const edm::ParameterSet&);
      ~TTbar_Kinematics() override;

  void bookHistograms(DQMStore::IBooker &i, edm::Run const &, edm::EventSetup const &) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;


   private:
      // ----------member data ---------------------------

  edm::InputTag hepmcCollection_;
  edm::InputTag genEventInfoProductTag_,genEvt_;


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

  edm::EDGetTokenT<GenEventInfoProduct> genEventInfoProductTagToken_;
  edm::EDGetTokenT<edm::HepMCProduct> hepmcCollectionToken_;

};

#endif
