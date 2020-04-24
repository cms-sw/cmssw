// -*- C++ -*-
//
// Package:    EnergyScaleAnalyzer
// Class:      EnergyScaleAnalyzer
// 
/**\class EnergyScaleAnalyzer EnergyScaleAnalyzer.cc Validation/EcalClusters/src/EnergyScaleAnalyzer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
// Original Author:  Keti Kaadze
//         Created:  Thu Jun 21 08:59:42 CDT 2007
//

//#include "RecoEcal/EnergyScaleAnalyzer/interface/EnergyScaleAnalyzer.h"
#include "Validation/EcalClusters/interface/EnergyScaleAnalyzer.h"
#include "RecoEcal/EgammaClusterProducers/interface/PreshowerClusterProducer.h"

//Framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

//Geometry
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/CaloTopology/interface/EcalBarrelHardcodedTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"

#include "TFile.h"
//Reconstruction classes
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/EgammaReco/interface/PreshowerCluster.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"

#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "DataFormats/Math/interface/LorentzVector.h"

//// Class header file
#include "RecoEcal/EgammaClusterProducers/interface/HybridClusterProducer.h"
#include "RecoEcal/EgammaCoreTools/interface/ClusterShapeAlgo.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <ios>
#include <map>
#include "TString.h"
#include "Validation/EcalClusters/interface/AnglesUtil.h"

//========================================================================
EnergyScaleAnalyzer::EnergyScaleAnalyzer( const edm::ParameterSet& ps )
//========================================================================
{
  
  hepMCLabel_ = consumes<edm::HepMCProduct>(ps.getParameter<std::string>("hepMCLabel"));
  hybridSuperClusters_token = consumes<reco::SuperClusterCollection>(ps.getUntrackedParameter<edm::InputTag>("hybridSuperClusters",edm::InputTag("hybridSuperClusters")));
  dynamicHybridSuperClusters_token = consumes<reco::SuperClusterCollection>(ps.getUntrackedParameter<edm::InputTag>("dynamicHybridSuperClusters",edm::InputTag("dynamicHybridSuperClusters")));
  correctedHybridSuperClusters_token = consumes<reco::SuperClusterCollection>(ps.getUntrackedParameter<edm::InputTag>("correctedHybridSuperClusters",edm::InputTag("correctedHybridSuperClusters")));
  correctedDynamicHybridSuperClusters_token = consumes<reco::SuperClusterCollection>(ps.getUntrackedParameter<edm::InputTag>("correctedDynamicHybridSuperClusters",edm::InputTag("correctedDynamicHybridSuperClusters")));
  correctedFixedMatrixSuperClustersWithPreshower_token = consumes<reco::SuperClusterCollection>(ps.getUntrackedParameter<edm::InputTag>("correctedFixedMatrixSuperClustersWithPreshower",edm::InputTag("correctedFixedMatrixSuperClustersWithPreshower")));
  fixedMatrixSuperClustersWithPreshower_token = consumes<reco::SuperClusterCollection>(ps.getUntrackedParameter<edm::InputTag>("fixedMatrixSuperClustersWithPreshower",edm::InputTag("fixedMatrixSuperClustersWithPreshower")));

  outputFile_   = ps.getParameter<std::string>("outputFile");
  rootFile_ = TFile::Open(outputFile_.c_str(),"RECREATE"); // open output file to store histograms
  
  evtN = 0;
}


//========================================================================
EnergyScaleAnalyzer::~EnergyScaleAnalyzer()
//========================================================================
{
  delete rootFile_;
}

//========================================================================
void
EnergyScaleAnalyzer::beginJob() {
//========================================================================

  mytree_ = new TTree("energyScale","");
  TString treeVariables = "mc_npar/I:parID:mc_sep/F:mc_e:mc_et:mc_phi:mc_eta:mc_theta:";    // MC information
  treeVariables += "em_dR/F:";                 // MC <-> EM matching information
  treeVariables += "em_isInCrack/I:em_scType:em_e/F:em_et:em_phi:em_eta:em_theta:em_nCell/I:em_nBC:";  // EM SC info    
  treeVariables += "em_pet/F:em_pe:em_peta:em_ptheta:"                     ;  // EM SC physics (eta corrected information)

  treeVariables += "emCorr_e/F:emCorr_et:emCorr_eta:emCorr_phi:emCorr_theta:";// CMSSW standard corrections
  treeVariables += "emCorr_pet/F:emCorr_peta:emCorr_ptheta:"                 ;// CMSSW standard physics  

  treeVariables += "em_pw/F:em_ew:em_br"                                     ;  // EM widths pw -- phiWidth, ew -- etaWidth, ratios of pw/ew

  mytree_->Branch("energyScale",&(tree_.mc_npar),treeVariables);
      
}

//========================================================================
void
EnergyScaleAnalyzer::analyze( const edm::Event& evt, const edm::EventSetup& es ) {
  using namespace edm; // needed for all fwk related classes
  using namespace std;

  //  std::cout << "Proceccing event # " << ++evtN << std::endl;
  
  //Get containers for MC truth, SC etc. ===================================================
  // =======================================================================================
  // =======================================================================================
  Handle<HepMCProduct> hepMC;
  evt.getByToken( hepMCLabel_, hepMC ) ;
  
  Labels l;
  labelsForToken(hepMCLabel_, l);

  const HepMC::GenEvent* genEvent = hepMC->GetEvent();
  if ( !(hepMC.isValid())) {
    LogInfo("EnergyScaleAnalyzer") << "Could not get MC Product!";
    return;
  }
  

  //=======================For Vertex correction
  std::vector< Handle< HepMCProduct > > evtHandles ;
  evt.getManyByType( evtHandles ) ;
  
  for ( unsigned int i=0; i<evtHandles.size(); ++i) {
    if ( evtHandles[i].isValid() ) {
      const HepMC::GenEvent* evt = evtHandles[i]->GetEvent() ;
      
      // take only 1st vertex for now - it's been tested only of PGuns...
      //
      HepMC::GenEvent::vertex_const_iterator vtx = evt->vertices_begin() ;
      if ( evtHandles[i].provenance()->moduleLabel() == std::string(l.module) ) {
	//Corrdinates of Vertex w.r.o. the point (0,0,0)
	xVtx_ = 0.1*(*vtx)->position().x();      
	yVtx_ = 0.1*(*vtx)->position().y();
	zVtx_ = 0.1*(*vtx)->position().z();  
      }
    }
  }
  //==============================================================================
  //Get handle to SC collections

  Handle<reco::SuperClusterCollection> hybridSuperClusters;
  try {
    evt.getByToken(hybridSuperClusters_token,hybridSuperClusters);
  }catch (cms::Exception& ex) {
    edm::LogError("EnergyScaleAnalyzer") << "Can't get collection with producer hybridSuperClusters.";
  }

  Handle<reco::SuperClusterCollection> dynamicHybridSuperClusters;
  try {
    evt.getByToken(dynamicHybridSuperClusters_token,dynamicHybridSuperClusters);
  }catch (cms::Exception& ex) {
    edm::LogError("EnergyScaleAnalyzer") << "Can't get collection with producer dynamicHybridSuperClusters.";
  }

  Handle<reco::SuperClusterCollection> fixedMatrixSuperClustersWithPS;
  try {
    evt.getByToken(fixedMatrixSuperClustersWithPreshower_token,fixedMatrixSuperClustersWithPS);
  }catch (cms::Exception& ex) {
    edm::LogError("EnergyScaleAnalyzer") << "Can't get collection with producer fixedMatrixSuperClustersWithPreshower.";
  }

  //Corrected collections
  Handle<reco::SuperClusterCollection> correctedHybridSC;
  try {
    evt.getByToken(correctedHybridSuperClusters_token,correctedHybridSC);
  }catch (cms::Exception& ex) {
    edm::LogError("EnergyScaleAnalyzer") << "Can't get collection with producer correctedHybridSuperClusters.";
  }

  Handle<reco::SuperClusterCollection> correctedDynamicHybridSC;
  try{
    evt.getByToken(correctedDynamicHybridSuperClusters_token,correctedDynamicHybridSC);
  }catch (cms::Exception& ex) {
    edm::LogError("EnergyScaleAnalyzer") << "Can't get collection with producer correctedDynamicHybridSuperClusters.";
  }
  
  Handle<reco::SuperClusterCollection> correctedFixedMatrixSCWithPS;
  try {
    evt.getByToken(correctedFixedMatrixSuperClustersWithPreshower_token,correctedFixedMatrixSCWithPS);
  }catch (cms::Exception& ex ) {
    edm::LogError("EnergyScaleAnalyzer") << "Can't get collection with producer correctedFixedMatrixSuperClustersWithPreshower.";
  }

  const reco::SuperClusterCollection*  dSC  = dynamicHybridSuperClusters.product();
  const reco::SuperClusterCollection*  hSC  = hybridSuperClusters.product();
  const reco::SuperClusterCollection* fmSC  = fixedMatrixSuperClustersWithPS.product();
  const reco::SuperClusterCollection* chSC  = correctedHybridSC.product();
  const reco::SuperClusterCollection* cdSC  = correctedDynamicHybridSC.product();
  const reco::SuperClusterCollection* cfmSC = correctedFixedMatrixSCWithPS.product();
 
  // -----------------------  Print outs for debugging
  /*
  std::cout << "MC truth" << std::endl;
  int counterI = 0;
  for(  HepMC::GenEvent::particle_const_iterator p = genEvent->particles_begin();
	p != genEvent->particles_end(); ++p ) {
    if ( fabs((*p)->momentum().eta()) < 1.5 ) {
      std::cout << ++counterI << " " << (*p)->momentum().e() << " " 
		<< (*p)->momentum().phi() << " " << (*p)->momentum().eta() << std::endl;
    }
  }

  std::cout << "Standard clusters" << std::endl;
  counterI = 0;
  for(reco::SuperClusterCollection::const_iterator em = hSC->begin();
      em != hSC->end(); ++em ) 
    std::cout << ++counterI << " " << em->energy() << " " << em->position().phi() << " " << em->position().eta() << std::endl;

  std::cout << "Dynamic clusters" << std::endl;
  counterI = 0;
  for(reco::SuperClusterCollection::const_iterator em = dSC->begin();
      em != dSC->end(); ++em ) 
    std::cout << ++counterI << " " << em->energy() << " " << em->position().phi() << " " << em->position().eta() << std::endl;

  std::cout << "FixedMatrix clusters with PS" << std::endl;
  counterI = 0;
  for(reco::SuperClusterCollection::const_iterator em = fmSC->begin();
      em != fmSC->end(); ++em )
    std::cout << ++counterI << " " << em->energy() << " " << em->position().phi() << " " << em->position().eta() << std::endl;
  */
  // -----------------------------
  //=====================================================================  
  // All containers are loaded, perform the analysis 
  //====================================================================

  // --------------------------- Store MC particles
  HepMC::GenEvent::particle_const_iterator p = genEvent->particles_begin();
  
  // Search for MC electrons or photons that satisfy the criteria
  float min_eT = 5;
  float max_eta = 2.5;
  
  std::vector<HepMC::GenParticle* > mcParticles;
  //int counter = 0;
  for ( HepMC::GenEvent::particle_const_iterator p = genEvent->particles_begin(); 
	p != genEvent->particles_end(); 
	++p ) {
    //LogInfo("EnergyScaleAnalyzer") << "Particle " << ++counter 
    //<< " PDG ID = " << (*p)->pdg_id() << " pT = " << (*p)->momentum().perp();
    // require photon or electron
    if ( (*p)->pdg_id() != 22 && abs((*p)->pdg_id()) != 11 ) continue;
    
    // require selection criteria
    bool satisfySelectionCriteria = 
      (*p)->momentum().perp() > min_eT &&
      fabs((*p)->momentum().eta()) < max_eta;
    
    if ( ! satisfySelectionCriteria ) continue;
    
    // EM MC particle is found, save it in the vector
    mcParticles.push_back(*p);
  }
  // separation in dR between 2 first MC particles
  // should not be used for MC samples with > 2 em objects generated!
  if ( mcParticles.size() == 2 ) {
      HepMC::GenParticle* mc1 = mcParticles[0];
      HepMC::GenParticle* mc2 = mcParticles[1];
      tree_.mc_sep = kinem::delta_R(mc1->momentum().eta(), mc1->momentum().phi(),
				    mc2->momentum().eta(), mc2->momentum().phi());
  } else
    tree_.mc_sep = -100;

  
  // now loop over MC particles, find the match with SC and do everything we need
  // then save info in the tree for every MC particle
  for(std::vector<HepMC::GenParticle* >::const_iterator p = mcParticles.begin();
      p != mcParticles.end(); ++p) {
    HepMC::GenParticle* mc = *p;
    
    // Fill MC information
    tree_.mc_npar  = mcParticles.size();
    tree_.parID    = mc->pdg_id();
    tree_.mc_e     = mc->momentum().e();
    tree_.mc_et    = mc->momentum().e()*sin(mc->momentum().theta());
    tree_.mc_phi   = mc->momentum().phi();
    tree_.mc_eta   = mc->momentum().eta();
    tree_.mc_theta = mc->momentum().theta();


    //Call function to fill tree
    // scType coprreponds:
    // HybridSuperCluster                     -- 1
    // DynamicHybridSuperCluster              -- 2
    // FixedMatrixSuperClustersWithPreshower  -- 3

    fillTree( hSC, chSC, mc, tree_, xVtx_, yVtx_, zVtx_, 1);
    //    std::cout << " TYPE " << 1 << " : " << tree_.em_e << " : " << tree_.em_phi << " : " << tree_.em_eta << std::endl;

    fillTree( dSC, cdSC, mc, tree_, xVtx_, yVtx_, zVtx_, 2);
    //    std::cout << " TYPE " << 2 << " : " << tree_.em_e << " : " << tree_.em_phi << " : " << tree_.em_eta << std::endl;

    fillTree( fmSC, cfmSC, mc, tree_, xVtx_, yVtx_, zVtx_, 3);
    //    std::cout << " TYPE " << 3 << " : " << tree_.em_e << " : " << tree_.em_phi << " : " << tree_.em_eta << std::endl;

    //   mytree_->Fill();
  } // loop over particles  
}


void EnergyScaleAnalyzer::fillTree ( const reco::SuperClusterCollection* scColl, const reco::SuperClusterCollection* corrSCColl, 
				     HepMC::GenParticle* mc, tree_structure_& tree_, float xV, float yV, float zV, int scType) {
  
  // -----------------------------  SuperClusters before energy correction
  reco::SuperClusterCollection::const_iterator em = scColl->end();
  float energyMax = -100.0; // dummy energy of the matched SC
  for(reco::SuperClusterCollection::const_iterator aClus = scColl->begin();
      aClus != scColl->end(); ++aClus) {
    // check the matching
    float dR = kinem::delta_R(mc   ->momentum().eta(), mc   ->momentum().phi(), 
			      aClus->position().eta(), aClus->position().phi());
    if (dR <  0.4) { // a rather loose matching cut
      float energy = aClus->energy();
      if ( energy < energyMax ) continue;
      energyMax = energy;
      em = aClus;
    }
  }
  
  if (em == scColl->end() ) {
    //    std::cout << "No matching SC with type " << scType << " was found for MC particle! "  << std::endl;
    //    std::cout << "Going to next type of SC. " << std::endl;
    return; 
  }
  //  ------------  

  tree_.em_scType = scType;

  tree_.em_isInCrack = 0;
  double emAbsEta = fabs(em->position().eta());
  // copied from RecoEgama/EgammaElectronAlgos/src/EgammaElectronClassification.cc
  if ( emAbsEta < 0.018 ||
       (emAbsEta > 0.423 && emAbsEta < 0.461) || 
       (emAbsEta > 0.770 && emAbsEta < 0.806) || 
       (emAbsEta > 1.127 && emAbsEta < 1.163) || 
       (emAbsEta > 1.460 && emAbsEta < 1.558) )
    tree_.em_isInCrack = 1;
  
  tree_.em_dR = kinem::delta_R(mc->momentum().eta(), mc->momentum().phi(), 
			      em->position().eta(), em->position().phi());
  tree_.em_e     = em->energy();
  tree_.em_et    = em->energy() * sin(em->position().theta());
  tree_.em_phi   = em->position().phi();
  tree_.em_eta   = em->position().eta();
  tree_.em_theta = em->position().theta();
  tree_.em_nCell = em->size();
  tree_.em_nBC   = em->clustersSize();
  
  //Get physics e, et etc:
  //Coordinates of EM object with respect of the point (0,0,0)
  xClust_zero_ = em->position().x();
  yClust_zero_ = em->position().y();
  zClust_zero_ = em->position().z();
  //Coordinates of EM object w.r.o. the Vertex position
  xClust_vtx_ = xClust_zero_ - xV;
  yClust_vtx_ = yClust_zero_ - yV;
  zClust_vtx_ = zClust_zero_ - zV;
    
  energyMax_ = em->energy();
  thetaMax_ = em->position().theta();
  etaMax_ = em->position().eta();
  phiMax_ = em->position().phi();
  eTMax_ = energyMax_ * sin(thetaMax_);
  if ( phiMax_ < 0) phiMax_ += 2*3.14159;
  
  rClust_vtx_ = sqrt (xClust_vtx_*xClust_vtx_ + yClust_vtx_*yClust_vtx_ + zClust_vtx_*zClust_vtx_);
  thetaMaxVtx_ = acos(zClust_vtx_/rClust_vtx_);
  etaMaxVtx_   = - log(tan(thetaMaxVtx_/2));
  eTMaxVtx_    = energyMax_ * sin(thetaMaxVtx_); 
  phiMaxVtx_   = atan2(yClust_vtx_,xClust_vtx_); 
  if ( phiMaxVtx_ < 0) phiMaxVtx_ += 2*3.14159;
  //=============================
  //parametres of EM object after vertex correction
  tree_.em_pet    = eTMaxVtx_;
  tree_.em_pe     = tree_.em_pet/sin(thetaMaxVtx_);
  tree_.em_peta   = etaMaxVtx_;
  tree_.em_ptheta = thetaMaxVtx_;
  
  
  //-------------------------------   Get SC after energy correction 
  em = corrSCColl->end();
  energyMax = -100.0; // dummy energy of the matched SC
  for(reco::SuperClusterCollection::const_iterator aClus = corrSCColl->begin();
      aClus != corrSCColl->end(); ++aClus) {
    // check the matching
    float dR = kinem::delta_R(mc   ->momentum().eta(), mc   ->momentum().phi(), 
			      aClus->position().eta(), aClus->position().phi());
    if (dR <  0.4) {
      float energy = aClus->energy();
      if ( energy < energyMax ) continue;
      energyMax = energy;
      em = aClus;
    }
  }
  
  if (em == corrSCColl->end() ) {
    //    std::cout << "No matching corrected SC with type " << scType << " was found for MC particle! "  << std::endl;
    //    std::cout << "Going to next type of SC. " << std::endl;
    return; 
  }
  //  ------------  
  
  ///fill tree with kinematic variables of corrected Super Cluster
    tree_.emCorr_e     = em->energy();
    tree_.emCorr_et    = em->energy() * sin(em->position().theta());
    tree_.emCorr_phi   = em->position().phi();
    tree_.emCorr_eta   = em->position().eta();
    tree_.emCorr_theta = em->position().theta();
      
    // =========== Eta and Theta wrt Vertex does not change after energy corrections are applied
    // =========== So, no need to calculate them again
    
    tree_.emCorr_peta   = tree_.em_peta;
    tree_.emCorr_ptheta = tree_.em_ptheta;
    tree_.emCorr_pet    = tree_.emCorr_e/cosh(tree_.emCorr_peta);

    tree_.em_pw = em->phiWidth();
    tree_.em_ew = em->etaWidth();
    tree_.em_br = tree_.em_pw/tree_.em_ew;
        
    mytree_->Fill();
    
}

//==========================================================================
void
EnergyScaleAnalyzer::endJob() {
  //========================================================================
  //Fill ROOT tree
  rootFile_->Write();
}

DEFINE_FWK_MODULE(EnergyScaleAnalyzer);
