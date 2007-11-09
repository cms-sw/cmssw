#ifndef TopObjectProducers_TopObjectSpatialResolution_h
#define TopObjectProducers_TopObjectSpatialResolution_h

// -*- C++ -*-
//
// Package:    TopObjectProducers
// Class:      TopObjectSpatialResolution
// 
/**\class TopObjectSpatialResolution TopObjectSpatialResolution.h TopQuarkAnalysis/TopObjectproducers/interface/TopObjectSpatialResolution.h

 Description: This class provides angular smearing to (inherited) TopObjects for systematic error studies.

 Implementation:
     A detailed documentation is found in
     TopQuarkAnalysis/TopObjectProducers/data/TopObjectSpatialResolution.cfi
*/
//
// Original Author:  Volker Adler
//         Created:  Mon Oct 15 16:15:05 CEST 2007
// $Id$
//
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
class TopObjectSpatialResolution : public edm::EDProducer {

  public:

    explicit TopObjectSpatialResolution(const edm::ParameterSet&);
    ~TopObjectSpatialResolution();

  private:

    virtual void produce(edm::Event&, const edm::EventSetup&);

    void smearAngles(T& object);

    edm::InputTag objects_;
    double        iniResTheta_,
                  worsenResTheta_,
                  finalResTheta_,
                  iniResPhi_,
                  worsenResPhi_,
                  finalResPhi_;
    bool          useDefaultIniRes_,
                  useWorsenResThetaByFactor_,
                  useWorsenResPhiByFactor_;

    CLHEP::RandGaussQ* gaussian_;

};


template<class T>
TopObjectSpatialResolution<T>::TopObjectSpatialResolution(const edm::ParameterSet& iConfig)
{
  objects_                   = iConfig.getParameter<edm::InputTag>("movedTopObject");
  useDefaultIniRes_          = iConfig.getParameter<bool>         ("useDefaultInitialResolutions");
  iniResTheta_               = iConfig.getParameter<double>       ("initialResolutionTheta");
  worsenResTheta_            = iConfig.getParameter<double>       ("worsenResolutionTheta");
  useWorsenResThetaByFactor_ = iConfig.getParameter<bool>         ("worsenResolutionThetaByFactor");
  iniResPhi_                 = iConfig.getParameter<double>       ("initialResolutionPhi");
  worsenResPhi_              = iConfig.getParameter<double>       ("worsenResolutionPhi");
  useWorsenResPhiByFactor_   = iConfig.getParameter<bool>         ("worsenResolutionPhiByFactor");


  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine& engine = rng->getEngine();
  gaussian_ = new CLHEP::RandGaussQ(engine);

  produces<std::vector<T> >();
}


template<class T>
TopObjectSpatialResolution<T>::~TopObjectSpatialResolution()
{
  delete gaussian_;
}


template<class T>
void TopObjectSpatialResolution<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<std::vector<T> > objectsHandle;
  iEvent.getByLabel(objects_, objectsHandle);
  std::vector<T> objects = *objectsHandle;
  std::auto_ptr<std::vector<T> > objectsVector(new std::vector<T>);
  objectsVector->reserve(objectsHandle->size());

  for ( unsigned int i = 0; i < objects.size(); i++ ) {
    smearAngles(objects[i]);
    objectsVector->push_back(objects[i]);
  }
  iEvent.put(objectsVector);
}


/// Sets initial resolution to resolution provided by input TopObject if required,
/// smears theta and phi and sets the 4-vector accordingl.
template<class T>
void TopObjectSpatialResolution<T>::smearAngles(T& object)
{
  // overwrite config file parameters 'initialResolution...' if required
  if ( useDefaultIniRes_ ) {
    // get initial resolutions from input TopObject
    iniResTheta_ = object.getResTheta(); // overwrites config file parameter "initialResolution"
    iniResPhi_   = object.getResPhi();   // overwrites config file parameter "initialResolution"
  }
  // Is 'worsenResolution<Angle>' a factor or an summand?
  finalResTheta_ = ( useWorsenResThetaByFactor_ ) ? (1.+fabs(1.-fabs(worsenResTheta_))) * fabs(iniResTheta_) : fabs(worsenResTheta_) + fabs(iniResTheta_); // conversion as protection from "finalRes_<iniRes_"
  finalResPhi_   = ( useWorsenResPhiByFactor_   ) ? (1.+fabs(1.-fabs(worsenResPhi_  ))) * fabs(iniResPhi_  ) : fabs(worsenResPhi_  ) + fabs(iniResPhi_  ); // conversion as protection from "finalRes_<iniRes_"
  // smear angles
  double smearedTheta = gaussian_->fire(object.theta(), sqrt(pow(finalResTheta_,2)-pow(iniResTheta_,2)));
  double smearedPhi   = gaussian_->fire(object.phi()  , sqrt(pow(finalResPhi_  ,2)-pow(iniResPhi_  ,2)));
  // set smeared new 4-vector
  object.setP4(reco::Particle::LorentzVector(object.p()*sin(smearedTheta)*cos(smearedPhi),
                                             object.p()*sin(smearedTheta)*sin(smearedPhi),
                                             object.p()*cos(smearedTheta),
                                             object.energy()));
}


#endif
