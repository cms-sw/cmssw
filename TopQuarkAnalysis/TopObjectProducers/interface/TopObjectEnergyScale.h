#ifndef TopObjectProducers_TopObjectEnergyScale_h
#define TopObjectProducers_TopObjectEnergyScale_h

// -*- C++ -*-
//
// Package:    TopObjectProducers
// Class:      TopObjectEnergyScale
//
/**\class TopObjectEnergyScale TopObjectEnergyScale.h TopQuarkAnalysis/TopObjectProducers/interface/TopObjectEnergyScale.h

 Description: <This class provides energy scale shifting & smearing to certain objects for systematic error studies.>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Volker Adler
//         Created:  Fri Oct  5 20:20:59 CEST 2007
// $Id$
//


#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/RandomNumberGenerator.h"
#include "CLHEP/Random/RandGaussQ.h"


template<class T>
class TopObjectEnergyScale : public edm::EDProducer {

  public:

    explicit TopObjectEnergyScale(const edm::ParameterSet&);
    ~TopObjectEnergyScale();

  private:

    virtual void produce( edm::Event&, const edm::EventSetup&);

    void shiftScale(T&);
    void smearScale(T&);

    edm::InputTag theObjects;
    double        theFactor,
                  theIniRes,
                  theFinalRes;

    CLHEP::RandGaussQ* theGaussian;

};


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
//  double smearFactor = std::max(theGaussian->fire(object.energy(),object.energy()*sqrt(pow(theFinalRes,2)-pow(theIniRes,2))),0.) / object.energy();
  double smearFactor = std::max(theGaussian->fire(1, sqrt(pow(theFinalRes,2)-pow(theIniRes,2))), 0.);
  const reco::Particle::LorentzVector newVector(smearFactor*object.px(),
                                                smearFactor*object.py(),
                                                smearFactor*object.pz(),
                                                smearFactor*object.energy());
  object.setP4(newVector);

  return;
}


#endif
