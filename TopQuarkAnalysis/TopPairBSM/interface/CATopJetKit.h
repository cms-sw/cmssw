#ifndef TopQuarkAnalysis_TopPairBSM_CATopJetKit_h
#define TopQuarkAnalysis_TopPairBSM_CATopJetKit_h



// -*- C++ -*-
//// -*- C++ -*-
//
// Package:    CATopJetKit
// Class:      CATopJetKit
//
/**


*/
//-------------------------------------------------------------------------------------
//!\class CATopJetKit CATopJetKit.cc PhysicsTools/StarterKit/plugins/CATopJetKit.cc
//!\brief CATopJetKit is an EDAnalyzer to examine Boosted Tops with the CATopJet algorithm
//!
//-------------------------------------------------------------------------------------
//
// Original Author:  Salvatore Rappoccio
//         Created:  Wed Nov 28 15:31:57 CST 2007
// $Id: CATopJetKit.h,v 1.3 2008/09/22 22:18:07 yumiceva Exp $
//
//-------------------------------------------------------------------------------------

// system include files
#include <memory>
#include <fstream>

// user include files
#include "PhysicsTools/StarterKit/interface/PatKitHelper.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
//#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "PhysicsTools/UtilAlgos/interface/TFileDirectory.h"

//
// class declaration
//

class CATopJetKit : public edm::EDProducer  // public edm::EDAnalyzer
{
public:
  explicit CATopJetKit(const edm::ParameterSet&);
  virtual ~CATopJetKit();

protected:

  // beginJob
  virtual void beginJob(const edm::EventSetup&) ;
  // produce is where the ntuples are made
  virtual void produce( edm::Event &, const edm::EventSetup & );
  // endJob
  virtual void endJob() ;

  // The main sub-object which does the real work
  pat::PatKitHelper    helper_;

  // Verbosity
  int             verboseLevel_;

  // Physics objects handles
  edm::Handle<std::vector<pat::Muon> >                    muonHandle_;
  edm::Handle<std::vector<pat::Electron> >                electronHandle_;
  edm::Handle<std::vector<pat::Tau> >                     tauHandle_;
  edm::Handle<std::vector<pat::Jet> >                     jetHandle_;
  edm::Handle<std::vector<pat::MET> >                     METHandle_;
  edm::Handle<std::vector<pat::Photon> >                  photonHandle_;
  edm::Handle<std::vector<reco::RecoChargedCandidate> >   trackHandle_;
  edm::Handle<std::vector<reco::GenParticle> >            genParticles_;

};



#endif
