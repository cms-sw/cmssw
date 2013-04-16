// This file is imported from:
//http://cmssw.cvs.cern.ch/cgi-bin/cmssw.cgi/UserCode/Mangano/WWAnalysis/AnalysisStep/plugins/CalibratedPatElectronProducer.cc?revision=1.2&view=markup


// -*- C++ -*-
//
// Package:    EgammaElectronProducers
// Class:      CalibratedPatElectronProducer
//
/**\class CalibratedPatElectronProducer 

 Description: EDProducer of PatElectron objects

 Implementation:
     <Notes on implementation>
*/

//#if CMSSW_VERSION>500


#include "EgammaAnalysis/ElectronTools/plugins/CalibratedPatElectronProducer.h"
#include "EgammaAnalysis/ElectronTools/interface/EpCombinationTool.h"


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHitFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

#include "EgammaAnalysis/ElectronTools/interface/ElectronEnergyCalibrator.h"


#include <iostream>

using namespace edm ;
using namespace std ;
using namespace reco ;
using namespace pat ;

CalibratedPatElectronProducer::CalibratedPatElectronProducer( const edm::ParameterSet & cfg )
// : PatElectronBaseProducer(cfg)
 {

  produces<ElectronCollection>();

  inputPatElectrons = cfg.getParameter<edm::InputTag>("inputPatElectronsTag");
  dataset = cfg.getParameter<std::string>("inputDataset");
  isMC = cfg.getParameter<bool>("isMC");
  updateEnergyError = cfg.getParameter<bool>("updateEnergyError");
  lumiRatio = cfg.getParameter<double>("lumiRatio");
  correctionsType = cfg.getParameter<int>("correctionsType");
  combinationType = cfg.getParameter<int>("combinationType");
  verbose = cfg.getParameter<bool>("verbose");
  synchronization = cfg.getParameter<bool>("synchronization");
  combinationRegressionInputPath = cfg.getParameter<std::string>("combinationRegressionInputPath");
  
  //basic checks
  if (isMC&&(dataset!="Summer11"&&dataset!="Fall11"&&dataset!="Summer12"&&dataset!="Summer12_DR53X_HCP2012"))
   { throw cms::Exception("CalibratedgsfElectronProducer|ConfigError")<<"Unknown MC dataset" ; }
  if (!isMC&&(dataset!="Prompt"&&dataset!="ReReco"&&dataset!="Jan16ReReco"&&dataset!="ICHEP2012"&&dataset!="Moriond2013"))
   { throw cms::Exception("CalibratedgsfElectronProducer|ConfigError")<<"Unknown Data dataset" ; }
   cout << "[CalibratedPATElectronProducer] Correcting scale for dataset " << dataset << endl;
 }
 
CalibratedPatElectronProducer::~CalibratedPatElectronProducer()
 {}

void CalibratedPatElectronProducer::produce( edm::Event & event, const edm::EventSetup & setup )
 {

  edm::Handle<edm::View<reco::Candidate> > oldElectrons ;
  event.getByLabel(inputPatElectrons,oldElectrons) ;
  std::auto_ptr<ElectronCollection> electrons( new ElectronCollection ) ;
  ElectronCollection::const_iterator electron ;
  ElectronCollection::iterator ele ;
  // first clone the initial collection
  for(edm::View<reco::Candidate>::const_iterator ele=oldElectrons->begin(); ele!=oldElectrons->end(); ++ele){    
    const pat::ElectronRef elecsRef = edm::RefToBase<reco::Candidate>(oldElectrons,ele-oldElectrons->begin()).castTo<pat::ElectronRef>();
    pat::Electron clone = *edm::RefToBase<reco::Candidate>(oldElectrons,ele-oldElectrons->begin()).castTo<pat::ElectronRef>();
    electrons->push_back(clone);
  }

  std::string pathToDataCorr;
  if (correctionsType != 0 ){

  switch (correctionsType){

	  case 1: pathToDataCorr = "../data/scalesMoriond.csv"; 
		  if (verbose) {std::cout<<"You choose regression 1 scale corrections"<<std::endl;}
		  break;
	  case 2: throw cms::Exception("CalibratedgsfElectronProducer|ConfigError")<<"You choose regression 2 scale corrections. They are not implemented yet." ;
		 // pathToDataCorr = "../data/data.csv";
		 // if (verbose) {std::cout<<"You choose regression 2 scale corrections."<<std::endl;}
		  break;
	  case 3: pathToDataCorr = "../data/data.csv";
		  if (verbose) {std::cout<<"You choose standard ecal energy scale corrections"<<std::endl;}
		  break;
	  default: throw cms::Exception("CalibratedgsfElectronProducer|ConfigError")<<"Unknown correctionsType !!!" ;
  }

  ElectronEnergyCalibrator theEnCorrector(pathToDataCorr, dataset, correctionsType, lumiRatio, isMC, updateEnergyError, verbose, synchronization);

  if (verbose) {std::cout<<"ElectronEnergyCalibrator object is created "<<std::endl;}

  for
   ( ele = electrons->begin() ;
     ele != electrons->end() ;
     ++ele )
   {     
    int elClass = -1;
    int run = event.run(); 

    float r9 = ele->r9(); 
    double correctedEcalEnergy = ele->correctedEcalEnergy();
    double correctedEcalEnergyError = ele->correctedEcalEnergyError();
    double trackMomentum = ele->trackMomentumAtVtx().R();
    double trackMomentumError = ele->trackMomentumError();
    
    if (ele->classification() == reco::GsfElectron::GOLDEN) {elClass = 0;}
    if (ele->classification() == reco::GsfElectron::BIGBREM) {elClass = 1;}
    if (ele->classification() == reco::GsfElectron::BADTRACK) {elClass = 2;}
    if (ele->classification() == reco::GsfElectron::SHOWERING) {elClass = 3;}
    if (ele->classification() == reco::GsfElectron::GAP) {elClass = 4;}

    SimpleElectron mySimpleElectron(run, elClass, r9, correctedEcalEnergy, correctedEcalEnergyError, trackMomentum, trackMomentumError, ele->ecalRegressionEnergy(), ele->ecalRegressionError(), ele->superCluster()->eta(), ele->isEB(), isMC, ele->ecalDriven(), ele->trackerDrivenSeed());

      // energy calibration for ecalDriven electrons
      if (ele->core()->ecalDrivenSeed()) {        
	      theEnCorrector.calibrate(mySimpleElectron);
      }
          // E-p combination  
      ElectronEPcombinator myCombinator;
      EpCombinationTool MyEpCombinationTool;
      MyEpCombinationTool.init(edm::FileInPath(combinationRegressionInputPath.c_str()).fullPath().c_str(),"CombinationWeight");

       switch (combinationType){
	  case 0: 
		  if (verbose) {std::cout<<"You choose not to combine."<<std::endl;}
		  break;
	  case 1: 
		  if (verbose) {std::cout<<"You choose corrected regression energy for standard combination"<<std::endl;}
		  myCombinator.setCombinationMode(1);
		  myCombinator.combine(mySimpleElectron);
		  break;
	  case 2: 
		  if (verbose) {std::cout<<"You choose uncorrected regression energy for standard combination"<<std::endl;}
		  myCombinator.setCombinationMode(2);
		  myCombinator.combine(mySimpleElectron);
		  break;
	  case 3: 
		  if (verbose) {std::cout<<"You choose regression combination."<<std::endl;}
		  MyEpCombinationTool.combine(mySimpleElectron);
		  break;
	  default: 
		  throw cms::Exception("CalibratedgsfElectronProducer|ConfigError")<<"Unknown combination Type !!!" ;
      }

  math::XYZTLorentzVector oldMomentum = ele->p4() ;
  math::XYZTLorentzVector newMomentum_ ;
  newMomentum_ = math::XYZTLorentzVector
   ( oldMomentum.x()*mySimpleElectron.getCombinedMomentum()/oldMomentum.t(),
     oldMomentum.y()*mySimpleElectron.getCombinedMomentum()/oldMomentum.t(),
     oldMomentum.z()*mySimpleElectron.getCombinedMomentum()/oldMomentum.t(),
     mySimpleElectron.getCombinedMomentum() ) ;

  if (verbose) {std::cout<<"Combined momentum before saving  "<<ele->p4().t()<<std::endl;}
  if (verbose) {std::cout<<"Combined momentum FOR saving  "<<mySimpleElectron.getCombinedMomentum()<<std::endl;}

  ele->correctMomentum(newMomentum_,mySimpleElectron.getTrackerMomentumError(),mySimpleElectron.getCombinedMomentumError());

  if (verbose) {std::cout<<"Combined momentum after saving  "<<ele->p4().t()<<std::endl;}


   }
  } else {std::cout<<"You choose not to calibrate. "<<std::endl;}
   event.put(electrons) ;

 }


#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_MODULE(CalibratedPatElectronProducer);

//#endif
