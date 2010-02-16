// -*- C++ -*-
//
// Package:    MCValidation
// Class:      MCValidation
// 
/**\class MCValidation MCValidation.cc testAnalysis/MCValidation/src/MCValidation.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>

   ----------------------------
   | particle | pdgId | icode |
   ----------------------------
   | e        |    11 |   1   |
   | mu       |    13 |   2   |
   | gamma    |    22 |   3   |
   | Z        |    23 |   4   |
   | W        |    24 |   5   |
   | H0       |    35 |   6   |
   | Z'       |    32 |   7   |
   ----------------------------
   
*/
//
// Original Author:  Devdatta MAJUMDER
//         Created:  Thu Apr 10 19:55:14 CEST 2008
// $Id: MCValidation.h,v 1.3 2010/01/12 06:11:24 hegner Exp $
//
//

// system include files
#include <memory>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Common/interface/TriggerNames.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CLHEP/Vector/LorentzVector.h"

///#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include <DataFormats/TrackReco/interface/HitPattern.h>
#include <DataFormats/TrackReco/interface/TrackBase.h>
#include <DataFormats/TrackReco/interface/TrackFwd.h>

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

#include "TFile.h"
#include "TH1F.h"
#include "TTree.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TPostScript.h"
#include "TStyle.h"
#include "TGraphErrors.h"

class MCValidation : public edm::EDAnalyzer {
public:
  explicit MCValidation(const edm::ParameterSet&);
  ~MCValidation();
  
private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  void genparticles(const reco::CandidateCollection&) ;
//      void genparticles(const GenParticleCollection&) ;

      // ----------member data ---------------------------

      edm::ESHandle<MagneticField> theMagneticField ;

//     edm::InputTag l_genEvt ;
     std::string s_genEvt ;
     
     std::string theOutfileName ;
     ofstream outfile ;
     
     edm::Service<TFileService> fs ;

     TTree* Tgen ;
     
     TH1F *mom_pT_,           *mom_eta_,           *mom_phi_            ;
      
     TH1F *gen_e_pT_,         *gen_eP_eta_,        *gen_eP_phi_,
                              *gen_eM_eta_,        *gen_eM_phi_         ;
     TH1F *gen_mu_pT_,        *gen_muP_eta_,       *gen_muP_phi_, 
                              *gen_muM_eta_,       *gen_muM_phi_        ; 
     TH1F *gen_gamma_pT_,     *gen_gamma_eta_,     *gen_gamma_phi_      ;
     TH1F *gen_W_pT_,         *gen_Wp_eta_,        *gen_Wp_phi_,
                              *gen_Wm_eta_,        *gen_Wm_phi_         ;
     TH1F *gen_Z_pT_,         *gen_Z_eta_,         *gen_Z_phi_          ;
     TH1F *gen_H0_pT_,        *gen_H0_eta_,        *gen_H0_phi_         ;
     TH1F *gen_Zprime_pT_,    *gen_Zprime_eta_,    *gen_Zprime_phi_     ;
     
     TH1F *dau_e_pT_,         *dau_eP_eta_,        *dau_eP_phi_, 
                              *dau_eM_eta_,        *dau_eM_phi_         ; 
     TH1F *dau_mu_pT_,        *dau_muP_eta_,       *dau_muP_phi_, 
                              *dau_muM_eta_,       *dau_muM_phi_        ;
     TH1F *dau_gamma_pT_,     *dau_gamma_eta_,     *dau_gamma_phi_      ;
     
     unsigned nEvt, iEvt, iRun ;
     short icode ;
     
};
