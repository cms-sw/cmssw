#ifndef TopQuarkAnalysis_TopPairBSM_interface_BoostedTopProducer_h
#define TopQuarkAnalysis_TopPairBSM_interface_BoostedTopProducer_h

// -*- C++ -*-
//
// Package:    BoostedTopProducer
// Class:      BoostedTopProducer
// 
/**\class BoostedTopProducer BoostedTopProducer.cc BoostedTopProducer.cc

 Description: Class to examine boosted ttbar pairs in multiple mass regions.
   
    This will produce a ttbar solution, which will take one of two forms:
       a) lv jj jj   Full reconstruction.
          
      ttbar->
          (hadt -> (hadW -> hadp + hadq) + hadb) + 
          (lept -> (lepW -> lepton + neutrino) + lepb)
    
       b) lv jj (j)  Partial reconstruction, associate 
                     at least 1 jet to the lepton 
                     hemisphere, and at least one jet in 
                     the opposite hemisphere.
   
       ttbar->
           (hadt -> (hadJet1 [+ hadJet2] ) ) +
           (lept -> (lepW -> lepton + neutrino) + lepJet1 )
   
    There will also be two subcategories of (b) that 
    will correspond to physics cases:
    
       b1)           Lepton is isolated: Moderate ttbar mass.
       b2)           Lepton is nonisolated: High ttbar mass. 
   

 Implementation:
     To implement this, we use the NamedCompositeCandidate structures
     from the Candidate model. This provides flexibility in the definition of the
     output objects and allows automatic plotting in the Starter Kit. 
     We use the PAT objects to construct the ttbar solutions in the different ranges
     as follows:
         a) Full reconstruction: We use TtSemiEventSolutions made upstream of this module.
	 b) Partial reconstruction: Association of variables using the "Psi" variable,
	    which is a more rapidity-invariant version of deltaR. 
*/
//
// Original Author:  "Salvatore Rappoccio"
//         Created:  Thu May  1 11:37:48 CDT 2008
// $Id: BoostedTopProducer.h,v 1.1 2012/10/11 14:26:45 eulisse Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Hemisphere.h"

#include "METzCalculator.h"
#include "AnalysisDataFormats/TopObjects/interface/TtSemiLeptonicEvent.h"
#include "TLorentzVector.h"
#include "TMath.h"



//
// class decleration
//

class BoostedTopProducer : public edm::EDProducer {
   public:
      explicit BoostedTopProducer(const edm::ParameterSet&);
      ~BoostedTopProducer();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

      // data labels
      edm::InputTag eleLabel_;
      edm::InputTag muoLabel_;
      edm::InputTag jetLabel_;
      edm::InputTag metLabel_;
      edm::InputTag solLabel_;
  
      // Handles to the various objects in the event:
      edm::Handle<std::vector<pat::Muon> >     muonHandle_;
      edm::Handle<std::vector<pat::Electron> > electronHandle_;
      edm::Handle<std::vector<pat::Jet> >      jetHandle_;
      edm::Handle<std::vector<pat::MET> >      METHandle_;
      
      // Cut variables
      double        caloIsoCut_;     // isolation cut to consider a lepton isolated
      double        mTop_;           // input top mass
 
      // Rapidity-invariant deltaR
      double Psi(TLorentzVector p1, TLorentzVector p2, double mass);
};

#endif
