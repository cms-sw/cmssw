//
// Original Author:  Isabel R Ojalvo
//         Created:  Tue Jul 22 12:21:36 CEST 2008
// $Id: CaloTriggerAnalyzerOnDataTrees.h,v 1.2 2013/04/20 03:52:16 dlange Exp $

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "Math/GenVector/VectorUtil.h"

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TH1F.h"
#include "TProfile.h"

#include "TTree.h"

class CaloTriggerAnalyzerOnDataTrees : public edm::EDAnalyzer {
   public:
      explicit CaloTriggerAnalyzerOnDataTrees(const edm::ParameterSet&);
      ~CaloTriggerAnalyzerOnDataTrees();

   private:
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void matchSLHC(const reco::Candidate * recoCAND);



      TTree * RRTree;

      //Inputs
      edm::InputTag vertices_;
      edm::InputTag SLHCsrc_;
      edm::InputTag LHCsrc_;
      edm::InputTag LHCisosrc_;
      edm::InputTag ref_;
      edm::InputTag electrons_;
      double iso_;
      double DR_;
      double threshold_;
      double maxEta_;
      float numVertices;

      float highestGenPt;
      float secondGenPt;
      float highPt;
      float highPhi;
      float highEta;
      float highRecoPt;
      float secondPtf;

      float LHCL1pt;
      float LHCL1eta;
      float LHCL1phi;
      float SLHCL1pt;
      float SLHCL1phi;
      float SLHCL1eta;
      float SLHCCentralTowerE;

      float RecoEpt;
      float RecoEeta;
      float RecoEphi;

      TH1F * SLHCpt;
      TH1F * LHCpt;

      TH1F * eta;
      TH1F * RECOpt;
      TH1F * ptNum;   
      TH1F * ptDenom;
      TH1F * etaNum;
      TH1F * etaDenom;
      TH1F * pt;
      TH1F * dPt;
      TH1F * dEta;
      TH1F * dPhi;

      TH1F * LHChighestPt;
      TH1F * LHCsecondPt;
      TH1F * SLHChighestPt;
      TH1F * SLHCsecondPt;

      TH1F * highestPt;
      TH1F * secondPt;
      TH1F * highestPtGen;
      TH1F * secondPtGen;
      TH1F * RPt;
      TH1F * absEta;
      TH1F * dR;
      TProfile * RPtEta;
      TProfile * RPtEtaFull;
};





