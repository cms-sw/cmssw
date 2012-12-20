// This file is imported from:

// -*- C++ -*-
//
// Package:    EgammaElectronProducers
// Class:      CalibratedElectronProducer
//
/**\class CalibratedElectronProducer 

 Description: EDProducer of GsfElectron objects

 Implementation:
     <Notes on implementation>
*/




#include "EgammaAnalysis/ElectronTools/plugins/CalibratedElectronProducer.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "EgammaAnalysis/ElectronTools/interface/PatElectronEnergyCalibrator.h"
#include "EgammaAnalysis/ElectronTools/interface/SuperClusterHelper.h"

#include <iostream>

using namespace edm ;
using namespace std ;
using namespace reco ;

CalibratedElectronProducer::CalibratedElectronProducer( const edm::ParameterSet & cfg )
// : PatElectronBaseProducer(cfg)
 {


  

  inputElectrons_ = cfg.getParameter<edm::InputTag>("inputElectronsTag");

  nameEnergyReg_      = cfg.getParameter<edm::InputTag>("nameEnergyReg");
  nameEnergyErrorReg_ = cfg.getParameter<edm::InputTag>("nameEnergyErrorReg");

  recHitCollectionEB_ = cfg.getParameter<edm::InputTag>("recHitCollectionEB");
  recHitCollectionEE_ = cfg.getParameter<edm::InputTag>("recHitCollectionEE");


  nameNewEnergyReg_      = cfg.getParameter<std::string>("nameNewEnergyReg");
  nameNewEnergyErrorReg_ = cfg.getParameter<std::string>("nameNewEnergyErrorReg");
  newElectronName_ = cfg.getParameter<std::string>("outputGsfElectronCollectionLabel");


  dataset = cfg.getParameter<std::string>("inputDataset");
  isMC = cfg.getParameter<bool>("isMC");
  isAOD = cfg.getParameter<bool>("isAOD");
  updateEnergyError = cfg.getParameter<bool>("updateEnergyError");
  applyCorrections = cfg.getParameter<int>("applyCorrections");
  verbose = cfg.getParameter<bool>("verbose");
  synchronization = cfg.getParameter<bool>("synchronization");
  
  //basic checks
  if (isMC&&(dataset!="Summer11"&&dataset!="Fall11"&&dataset!="Summer12"&&dataset!="Summer12_DR53X_HCP2012"))
   { throw cms::Exception("CalibratedgsfElectronProducer|ConfigError")<<"Unknown MC dataset" ; }
  if (!isMC&&(dataset!="Prompt"&&dataset!="ReReco"&&dataset!="Jan16ReReco"&&dataset!="ICHEP2012"&&dataset!="2012Jul13ReReco"))
   { throw cms::Exception("CalibratedgsfElectronProducer|ConfigError")<<"Unknown Data dataset" ; }
   cout << "[CalibratedGsfElectronProducer] Correcting scale for dataset " << dataset << endl;


   produces<edm::ValueMap<double> >(nameNewEnergyReg_);
   produces<edm::ValueMap<double> >(nameNewEnergyErrorReg_);
   produces<GsfElectronCollection> (newElectronName_);
   geomInitialized_ = false;
 }
 
CalibratedElectronProducer::~CalibratedElectronProducer()
 {}

void CalibratedElectronProducer::produce( edm::Event & event, const edm::EventSetup & setup )
 {

  if (!geomInitialized_) {
    edm::ESHandle<CaloTopology> theCaloTopology;
    setup.get<CaloTopologyRecord>().get(theCaloTopology);
    ecalTopology_ = & (*theCaloTopology);
    
    edm::ESHandle<CaloGeometry> theCaloGeometry;
    setup.get<CaloGeometryRecord>().get(theCaloGeometry); 
    caloGeometry_ = & (*theCaloGeometry);
    geomInitialized_ = true;
  }

   // Read GsfElectrons
   edm::Handle<reco::GsfElectronCollection>  oldElectronsH ;
   event.getByLabel(inputElectrons_,oldElectronsH) ;
   
   // Read RecHits
  edm::Handle< EcalRecHitCollection > pEBRecHits;
  edm::Handle< EcalRecHitCollection > pEERecHits;
  event.getByLabel( recHitCollectionEB_, pEBRecHits );
  event.getByLabel( recHitCollectionEE_, pEERecHits );

   // ReadValueMaps
  edm::Handle<edm::ValueMap<double> > valMapEnergyH;
  event.getByLabel(nameEnergyReg_,valMapEnergyH);
  edm::Handle<edm::ValueMap<double> > valMapEnergyErrorH;
  event.getByLabel(nameEnergyErrorReg_,valMapEnergyErrorH);
  

  // Prepare output collections
  std::auto_ptr<GsfElectronCollection> electrons( new reco::GsfElectronCollection ) ;
  // Fillers for ValueMaps:
  std::auto_ptr<edm::ValueMap<double> > regrNewEnergyMap(new edm::ValueMap<double>() );
  edm::ValueMap<double>::Filler energyFiller(*regrNewEnergyMap);

  std::auto_ptr<edm::ValueMap<double> > regrNewEnergyErrorMap(new edm::ValueMap<double>() );
  edm::ValueMap<double>::Filler energyErrorFiller(*regrNewEnergyErrorMap);

  // first clone the initial collection
  unsigned nElectrons = oldElectronsH->size();
  for(unsigned iele = 0; iele < nElectrons; ++iele){    
    electrons->push_back((*oldElectronsH)[iele]);
  }

  ElectronEnergyCalibrator theEnCorrector(dataset, isAOD, isMC, updateEnergyError, applyCorrections, verbose, synchronization);

  std::vector<double> regressionValues;
  std::vector<double> regressionErrorValues;
  regressionValues.reserve(nElectrons);
  regressionErrorValues.reserve(nElectrons);

  for ( unsigned iele = 0; iele < nElectrons ; ++iele) {
    
    reco::GsfElectron & ele  ( (*electrons)[iele]);
    reco::GsfElectronRef elecRef(oldElectronsH,iele);
    double regressionEnergy = (*valMapEnergyH)[elecRef];
    double regressionEnergyError = (*valMapEnergyErrorH)[elecRef];
    
    regressionValues.push_back(regressionEnergy);
    regressionErrorValues.push_back(regressionEnergyError);

    //    r9 
    const EcalRecHitCollection * recHits=0;
    if(ele.isEB()) {
      recHits = pEBRecHits.product();
    } else
      recHits = pEERecHits.product();

    SuperClusterHelper mySCHelper(&(ele),recHits,ecalTopology_,caloGeometry_);

    // energy calibration for ecalDriven electrons
      if (ele.core()->ecalDrivenSeed()) {        
        theEnCorrector.correct(ele, mySCHelper.r9(),event, setup, regressionEnergy,regressionEnergyError);
      }
      else {
        //std::cout << "[CalibratedElectronProducer] is tracker driven only!!" << std::endl;
      }
   }


  // Save the electrons
  const edm::OrphanHandle<reco::GsfElectronCollection> gsfNewElectronHandle = event.put(electrons, newElectronName_) ;
  energyFiller.insert(gsfNewElectronHandle,regressionValues.begin(),regressionValues.end());
  energyFiller.fill();
  energyErrorFiller.insert(gsfNewElectronHandle,regressionValues.begin(),regressionValues.end());
  energyErrorFiller.fill();


  event.put(regrNewEnergyMap,nameNewEnergyReg_);
  event.put(regrNewEnergyErrorMap,nameNewEnergyErrorReg_);
 }


#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
DEFINE_FWK_MODULE(CalibratedElectronProducer);

//#endif
