// Simple code to access L1 objects

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Math/interface/LorentzVector.h"

#include "SimDataFormats/SLHC/interface/L1CaloCluster.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

#include <TFile.h>
#include <TTree.h>

// class declaration
class SLHCCaloTriggerAccessor : public edm::EDAnalyzer {
   public:
      explicit SLHCCaloTriggerAccessor(const edm::ParameterSet&);
      ~SLHCCaloTriggerAccessor();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      //Inputs
      edm::InputTag l1egamma_;
      edm::InputTag l1isoegamma_;
      edm::InputTag l1tau_;
      edm::InputTag l1isotau_;
      edm::InputTag jets_;

      //File
      std::string filename_;

      TFile *f;
      TTree *t;

      int nJets,nL1EG,nL1IsoEG,nL1Tau,nL1IsoTau;

      float *jet_et,    *jet_eta,     *jet_phi;
      float *l1eg_et,   *l1eg_eta,    *l1eg_phi;
      float *l1isoeg_et,*l1isoeg_eta, *l1isoeg_phi;
      float *l1tau_et,  *l1tau_eta,   *l1tau_phi;
      float *l1isotau_et,  *l1isotau_eta,   *l1isotau_phi;

};





