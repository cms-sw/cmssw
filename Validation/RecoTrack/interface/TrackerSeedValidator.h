#ifndef TrackerSeedValidator_h
#define TrackerSeedValidator_h

/** \class TrackerSeedValidator
 *  Class that prodecs histrograms to validate Track Reconstruction performances
 *
 *  \author cerati
 */

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "Validation/RecoTrack/interface/MultiTrackValidatorBase.h"
#include "Validation/RecoTrack/interface/MTVHistoProducerAlgoForTracker.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"

class TrackerSeedValidator : public DQMEDAnalyzer, protected MultiTrackValidatorBase {
 public:
  /// Constructor
  TrackerSeedValidator(const edm::ParameterSet& pset);
  
  /// Destructor
  virtual ~TrackerSeedValidator();


  /// Method called once per event
  void analyze(const edm::Event&, const edm::EventSetup& ) override;
  /// Method called to book the DQM histograms
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  
 private:
  std::vector<edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator>> associatorTokens;

  std::string builderName;
  edm::ESHandle<TransientTrackingRecHitBuilder> theTTRHBuilder;
  std::string dirName_;

  // select tracking particles 
  //(i.e. "denominator" of the efficiency ratio)
  TrackingParticleSelector tpSelector;				      
  CosmicTrackingParticleSelector cosmictpSelector;
  std::unique_ptr<MTVHistoProducerAlgoForTracker> histoProducerAlgo_;

};


#endif
