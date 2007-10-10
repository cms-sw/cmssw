#include "TopQuarkAnalysis/TopObjectProducers/interface/TopObjectEnergyScale.h"

// -*- C++ -*-
//
// Package:    TopObjectProducers
// Class:      TopObjectEnergyScale
// 
/**\class TopObjectEnergyScale TopObjectEnergyScale.cc TopQuarkAnalysis/TopObjectProducers/src/TopObjectEnergyScale.cc

 Description: <This class provides energy scale shifting & smearing to certain objects for systematic error studies.>

 Implementation:
     <See corresponding header file "TopQuarkAnalysis/TopObjectProducers/interface/TopObjectEnergyScale.h">
*/
//
// Original Author:  Volker Adler
//         Created:  Fri Oct  5 20:20:59 CEST 2007
// $Id$
//
//


#include "AnalysisDataFormats/TopObjects/interface/TopElectron.h"
#include "AnalysisDataFormats/TopObjects/interface/TopMuon.h"
#include "AnalysisDataFormats/TopObjects/interface/TopTau.h"
#include "AnalysisDataFormats/TopObjects/interface/TopJet.h"
#include "AnalysisDataFormats/TopObjects/interface/TopMET.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"

#include "FWCore/Framework/interface/MakerMacros.h"


template<class T>
TopObjectEnergyScale<T>::TopObjectEnergyScale(const edm::ParameterSet& iConfig) :
  theFactor(1.),
  theIniRes(0.),
  theFinalRes(0.),
  theGaussian(0)
{
  theObjects  = iConfig.getParameter<edm::InputTag>("scaledTopObject");
  theFactor   = iConfig.getParameter<double>       ("shiftFactor");
  theIniRes   = iConfig.getParameter<double>       ("initialResolution");
  theFinalRes = iConfig.getParameter<double>       ("finalResolution");

  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine& engine = rng->getEngine();
  theGaussian = new CLHEP::RandGaussQ(engine);

  produces<std::vector<T> >();
}

template<class T>
TopObjectEnergyScale<T>::~TopObjectEnergyScale() {
  delete theGaussian;
}


template<class T>
void TopObjectEnergyScale<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<std::vector<T> > objectsHandle;
  iEvent.getByLabel(theObjects,objectsHandle);
  std::vector<T> objects = *objectsHandle;
  std::vector<T>* objectsVector = new std::vector<T>();

  for ( unsigned int i = 0; i < objects.size(); i++ ) {
    if ( theFactor >= 0. && theFactor != 1. )
      shiftScale(objects[i]);
    if ( theFinalRes > theIniRes && objects[i].energy() >= 0. )
      smearScale(objects[i]);
    objectsVector->push_back(objects[i]);
  }
  std::auto_ptr<std::vector<T> > ptr(objectsVector);
  iEvent.put(ptr);

  return;
}

template<class T>
void TopObjectEnergyScale<T>::shiftScale(T& object)
{
  const reco::Particle::LorentzVector newVector(theFactor*object.px(),
                                                theFactor*object.py(),
                                                theFactor*object.pz(),
                                                theFactor*object.energy());
  object.setP4(newVector);

  return;
}

template<class T>
void TopObjectEnergyScale<T>::smearScale(T& object)
{
  double smearFactor = std::max(theGaussian->fire(object.energy(),object.energy()*sqrt(pow(theFinalRes,2)-pow(theIniRes,2))),0.) / object.energy();
  const reco::Particle::LorentzVector newVector(smearFactor*object.px(),
                                                smearFactor*object.py(),
                                                smearFactor*object.pz(),
                                                smearFactor*object.energy());
  object.setP4(newVector);

  return;
}


typedef TopObjectEnergyScale<TopElectron> TopElectronEnergyScale;
typedef TopObjectEnergyScale<TopMuon>     TopMuonEnergyScale;
typedef TopObjectEnergyScale<TopTau>      TopTauEnergyScale;
typedef TopObjectEnergyScale<TopJet>      TopJetEnergyScale;
typedef TopObjectEnergyScale<TopMET>      TopMETEnergyScale;

DEFINE_FWK_MODULE(TopElectronEnergyScale);
DEFINE_FWK_MODULE(TopMuonEnergyScale);
DEFINE_FWK_MODULE(TopTauEnergyScale);
DEFINE_FWK_MODULE(TopJetEnergyScale);
DEFINE_FWK_MODULE(TopMETEnergyScale);
