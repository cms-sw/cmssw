// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/SLHC/interface/L1CaloCluster.h"
#include "SimDataFormats/SLHC/interface/L1CaloClusterFwd.h"

#include "SimDataFormats/SLHC/interface/L1CaloJet.h"
#include "SimDataFormats/SLHC/interface/L1CaloJetFwd.h"

#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TH1F.h"
#include "TProfile.h"

#include "TTree.h"


//
// class declaration
//

class L1CaloClusterAnalyzer : public edm::EDAnalyzer {
   public:
      explicit L1CaloClusterAnalyzer(const edm::ParameterSet&);
      ~L1CaloClusterAnalyzer();

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

      TTree * RRTree;

      edm::InputTag src_;
      edm::InputTag electrons_; 
      float coneE;
      float centralPt;
      float RecoPt;
      float RecoMatch;
      float ClusterMatch;
      float TwoLeadTowerEnergy;
      int TowerEnergy1;
      int TowerEnergy2;
      int TowerEnergy3;
      int TowerEnergy4;
      int Ring1E;
      int Ring2E;
      int Ring3E;
      int Ring4E;

      float ClusterEnergy;

      float ClusterPtMatch;

      float RecoEpt;
      float RecoEeta;
      float RecoEphi;
      float CentralIso;


};

