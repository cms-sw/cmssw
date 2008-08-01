#ifndef SeedValidator_h
#define SeedValidator_h

/** \class SeedValidator
 *  Class that prodecs histrograms to validate Track Reconstruction performances
 *
 *  $Date: 2007/11/13 10:46:45 $
 *  $Revision: 1.33 $
 *  \author cerati
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "Validation/RecoTrack/interface/MultiTrackValidatorBase.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

class SeedValidator : public edm::EDAnalyzer, protected MultiTrackValidatorBase {
 public:
  /// Constructor
  SeedValidator(const edm::ParameterSet& pset):MultiTrackValidatorBase(pset){
    builderName = pset.getParameter<std::string>("TTRHBuilder");
  }
  
  /// Destructor
  ~SeedValidator(){ }

  /// Method called before the event loop
  void beginJob( const edm::EventSetup &);
  /// Method called once per event
  void analyze(const edm::Event&, const edm::EventSetup& );
  /// Method called at the end of the event loop
  void endJob();
  
 private:
  std::string builderName;
  edm::ESHandle<TransientTrackingRecHitBuilder> theTTRHBuilder;
};


#endif
