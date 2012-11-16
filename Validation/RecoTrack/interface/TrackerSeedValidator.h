#ifndef TrackerSeedValidator_h
#define TrackerSeedValidator_h

/** \class TrackerSeedValidator
 *  Class that prodecs histrograms to validate Track Reconstruction performances
 *
 *  $Date: 2008/06/30 13:20:55 $
 *  $Revision: 1.3 $
 *  \author cerati
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "Validation/RecoTrack/interface/MultiTrackValidatorBase.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

class TrackerSeedValidator : public edm::EDAnalyzer, protected MultiTrackValidatorBase {
 public:
  /// Constructor
  TrackerSeedValidator(const edm::ParameterSet& pset):MultiTrackValidatorBase(pset){
    builderName = pset.getParameter<std::string>("TTRHBuilder");
  }
  
  /// Destructor
  ~TrackerSeedValidator(){ }

  /// Method called before the event loop
  void beginRun(edm::Run const&, edm::EventSetup const&);
  /// Method called once per event
  void analyze(const edm::Event&, const edm::EventSetup& );
  /// Method called at the end of the event loop
  void endRun(edm::Run const&, edm::EventSetup const&);
  
 private:
  std::string builderName;
  edm::ESHandle<TransientTrackingRecHitBuilder> theTTRHBuilder;
};


#endif
