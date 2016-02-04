#ifndef RecoEcal_EgammaClusterProducers_EnergyScaleAnalyzer_h
#define RecoEcal_EgammaClusterProducers_EnergyScaleAnalyzer_h
/**\class EnergyScaleAnalyzerx

 Description: Analyzer to fetch collection of objects from event and make simple plots

 Implementation:
     \\\author: Keti Kaadze, June 2007
*/
//
// $Id: EnergyScaleAnalyzer.h,v 1.5 2009/12/14 22:24:33 wmtan Exp $
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include <string>
#include "TH1.h"
#include "TH2.h"
#include "TTree.h"
#include "TNtuple.h"
class TFile;

//
// class declaration
//

class EnergyScaleAnalyzer : public edm::EDAnalyzer {
   public:
      explicit EnergyScaleAnalyzer( const edm::ParameterSet& );
      ~EnergyScaleAnalyzer();


      virtual void analyze( const edm::Event&, const edm::EventSetup& );
      virtual void beginJob();
      virtual void endJob();

      
 private:
      
      std::string outputFile_; // output file
  
      std::string hepMCLabel_;

       // root file to store histograms
      TFile*  rootFile_;

      //Tree
      TTree* mytree_;
      struct tree_structure_ {
	// MC information
	int   mc_npar;
	int   parID;
	float mc_sep;
	float mc_e;
	float mc_et;
	float mc_phi;
	float mc_eta;
	float mc_theta;

	// MC-EM matching info
	float em_dR;
	
	// EM SC info (uncorrected)
	int   em_isInCrack;
	int   em_scType;
	float em_e;
	float em_et;
	float em_phi;
	float em_eta;
	float em_theta;
	int   em_nCell;
	int   em_nBC;
	
	// physics variables
	float em_pet;
	float em_pe;
	float em_peta;
	float em_ptheta;

	// corrected collectin
	float emCorr_e;
	float emCorr_et;
	float emCorr_eta;
	float emCorr_phi;
	float emCorr_theta;
	float emCorr_pet;
	float emCorr_peta;
	float emCorr_ptheta;

	// EM widths, pw -- phiWidth, ew -- etaWidth
	float em_pw;
	float em_ew;
	// ratios of widths pw/ew
	float em_br;
      };
      tree_structure_ tree_;

      //
      float xVtx_;
      float yVtx_;
      float zVtx_;
      //
      float xClust_zero_;
      float yClust_zero_;
      float zClust_zero_;
      //
      float xClust_vtx_;
      float yClust_vtx_;
      float zClust_vtx_;
      //
      float rClust_vtx_;
      //
      float energyMax_;
      float eTMax_;
      float eTMaxVtx_;
      float etaMax_;
      float etaMaxVtx_;
      float phiMax_;
      float phiMaxVtx_;
      float thetaMax_;
      float thetaMaxVtx_;
      //

      int evtN;

      //Function
      void fillTree ( const reco::SuperClusterCollection* scColl, const reco::SuperClusterCollection* corrSCColl,
		      HepMC::GenParticle* mc,  tree_structure_& tree_, float xV, float yV, float zV, int scType);
};
#endif

