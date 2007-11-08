#ifndef TopObjectProducers_TopObjectEnergyScale_h
#define TopObjectProducers_TopObjectEnergyScale_h

// -*- C++ -*-
//
// Package:    TopObjectProducers
// Class:      TopObjectEnergyScale
//
/**\class TopObjectEnergyScale TopObjectEnergyScale.h TopQuarkAnalysis/TopObjectProducers/interface/TopObjectEnergyScale.h

 Description: This class provides energy scale shifting & smearing to (inherited) TopObjects for systematic error studies.

 Implementation:
     A detailed documentation is found in
     TopQuarkAnalysis/TopObjectProducers/data/TopObjectEnergyScale.cfi
*/
//
// Original Author:  Volker Adler
//         Created:  Fri Oct  5 20:20:59 CEST 2007
// $Id: TopObjectEnergyScale.h,v 1.1 2007/10/10 02:18:04 lowette Exp $
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

    explicit TopObjectEnergyScale(const edm::ParameterSet& iConfig);
    ~TopObjectEnergyScale();

  private:

    virtual void produce( edm::Event& iEvent, const edm::EventSetup& iSetup);

    double getSmearing(T& object);
    void   setScale(T& object);

    edm::InputTag objects_;
    double        factor_,
                  shiftFactor_,
                  iniRes_,
                  worsenRes_;
    bool          useFixedMass_,
                  useDefaultIniRes_,
                  useIniResByFraction_,
                  useWorsenResByFactor_;

    CLHEP::RandGaussQ* gaussian_;

};


template<class T>
TopObjectEnergyScale<T>::TopObjectEnergyScale(const edm::ParameterSet& iConfig)
{
  objects_              = iConfig.getParameter<edm::InputTag>("scaledTopObject");
  useFixedMass_         = iConfig.getParameter<bool>         ("fixMass");
  shiftFactor_          = iConfig.getParameter<double>       ("shiftFactor");
  useDefaultIniRes_     = iConfig.getParameter<bool>         ("useDefaultInitialResolution");
  iniRes_               = iConfig.getParameter<double>       ("initialResolution");
  useIniResByFraction_  = iConfig.getParameter<bool>         ("initialResolutionByFraction");
  worsenRes_            = iConfig.getParameter<double>       ("worsenResolution");
  useWorsenResByFactor_ = iConfig.getParameter<bool>         ("worsenResolutionByFactor");

  edm::Service<edm::RandomNumberGenerator> rng;
  CLHEP::HepRandomEngine& engine = rng->getEngine();
  gaussian_ = new CLHEP::RandGaussQ(engine);

  produces<std::vector<T> >();
}


template<class T>
TopObjectEnergyScale<T>::~TopObjectEnergyScale()
{
  delete gaussian_;
}


template<class T>
void TopObjectEnergyScale<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  edm::Handle<std::vector<T> > objectsHandle;
  iEvent.getByLabel(objects_, objectsHandle);
  std::vector<T> objects = *objectsHandle;
  std::auto_ptr<std::vector<T> > objectsVector(new std::vector<T>);
  objectsVector->reserve(objectsHandle->size());

  for ( unsigned int i = 0; i < objects.size(); i++ ) {
    factor_ = shiftFactor_ * ( objects[i].energy() > 0. ?
                               getSmearing(objects[i])  :
                               0.);
    setScale(objects[i]);
    objectsVector->push_back(objects[i]);
  }
  iEvent.put(objectsVector);
}


/// Returns a smearing factor which is multiplied to the initial value then to get it smeared,
/// sets initial resolution to resolution provided by input TopObject if required
/// and converts the 'worsenResolution' parameter to protect from meaningless final resolution values.
template<class T>
double TopObjectEnergyScale<T>::getSmearing(T& object)
{
  // overwrite config file parameter 'initialResolution' if required
  if ( useDefaultIniRes_ ) {
    // get initial resolution from input TopObject (and calculate relative initial resolution from absolute value)
    iniRes_ = (1. / sin(object.theta()) * object.getResET() - object.et() * cos(object.theta()) / pow(sin(object.theta()),2) * object.getResTheta()) / object.energy(); // conversion of TopObject::resET and TopObject::resTheta into energy resolution
  } else if ( ! useIniResByFraction_ ) {
    // calculate relative initial resolution from absolute value
    iniRes_ = iniRes_ / object.energy();
  }
  // Is 'worsenResolution' a factor or a summand?
  double finalRes = useWorsenResByFactor_                            ?
                    (1.+fabs(1.-fabs(worsenRes_)))   * fabs(iniRes_) :
                    fabs(worsenRes_)/object.energy() + fabs(iniRes_); // conversion as protection from "finalRes_<iniRes_"
  // return smearing factor
  return std::max( gaussian_->fire(1., sqrt(pow(finalRes,2)-pow(iniRes_,2))), 0. ); // protection from negative smearing factors
}


/// Mutliplies the final factor (consisting of shifting and smearing factors) to the object's 4-vector
/// and takes care of preserved masses.
template<class T>
void TopObjectEnergyScale<T>::setScale(T& object)
{
  if ( factor_ < 0. ) {
    factor_ = 0.;
  }
  // calculate the momentum factor for fixed or not fixed mass
  double factorMomentum = useFixedMass_ && object.p() > 0.                                   ?
                          sqrt(pow(factor_*object.energy(),2)-object.massSqr()) / object.p() :
                          factor_;
  // set shifted & smeared new 4-vector
  object.setP4(reco::Particle::LorentzVector(factorMomentum*object.px(),
                                             factorMomentum*object.py(),
                                             factorMomentum*object.pz(),
                                             factor_       *object.energy()));
}


#endif
