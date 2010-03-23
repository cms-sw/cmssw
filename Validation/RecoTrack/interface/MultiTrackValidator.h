#ifndef MultiTrackValidator_h
#define MultiTrackValidator_h

/** \class MultiTrackValidator
 *  Class that prodecs histrograms to validate Track Reconstruction performances
 *
 *  $Date: 2009/09/04 22:25:04 $
 *  $Revision: 1.48 $
 *  \author cerati
 */
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "Validation/RecoTrack/interface/MultiTrackValidatorBase.h"
#include "Validation/RecoTrack/interface/MTVHistoProducerAlgo.h"


class MultiTrackValidator : public edm::EDAnalyzer, protected MultiTrackValidatorBase {
 public:
  /// Constructor
  MultiTrackValidator(const edm::ParameterSet& pset);
  
  /// Destructor
  virtual ~MultiTrackValidator(){ }

  /// Method called before the event loop
  void beginRun(edm::Run const&, edm::EventSetup const&);
  /// Method called once per event
  void analyze(const edm::Event&, const edm::EventSetup& );
  /// Method called at the end of the event loop
  void endRun(edm::Run const&, edm::EventSetup const&);

private:

 private:
  std::string dirName_;
  edm::InputTag associatormap;
  bool UseAssociators;

  // B.M. why these lines are here again? They were already in MTVBase.h file
  //double minPhi, maxPhi;  int nintPhi;

  bool useGsf;
  // select tracking particles 
  //(i.e. "denominator" of the efficiency ratio)
  TrackingParticleSelector tpSelector;				      
  CosmicTrackingParticleSelector cosmictpSelector;
  MTVHistoProducerAlgo* histoProducerAlgo_;
};


#endif
