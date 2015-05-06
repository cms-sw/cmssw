#ifndef MultiTrackValidator_h
#define MultiTrackValidator_h

/** \class MultiTrackValidator
 *  Class that prodecs histrograms to validate Track Reconstruction performances
 *
 *  \author cerati
 */
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "Validation/RecoTrack/interface/MultiTrackValidatorBase.h"
#include "Validation/RecoTrack/interface/MTVHistoProducerAlgo.h"


class MultiTrackValidator : public DQMEDAnalyzer, protected MultiTrackValidatorBase {
 public:
  /// Constructor
  MultiTrackValidator(const edm::ParameterSet& pset);
  
  /// Destructor
  virtual ~MultiTrackValidator();


  /// Method called once per event
  void analyze(const edm::Event&, const edm::EventSetup& ) override;
  /// Method called at the end of the event loop
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  /// Method called to book the DQM histograms
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;


 protected:
  //these are used by MTVGenPs
  bool UseAssociators;
  MTVHistoProducerAlgo* histoProducerAlgo_;

 private:
  std::vector<edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator>> associatorTokens;
  std::vector<edm::EDGetTokenT<reco::SimToRecoCollection>> associatormapStRs;
  std::vector<edm::EDGetTokenT<reco::RecoToSimCollection>> associatormapRtSs;

  std::string dirName_;

  bool useGsf;
  bool runStandalone;
  // select tracking particles 
  //(i.e. "denominator" of the efficiency ratio)
  TrackingParticleSelector tpSelector;				      
  CosmicTrackingParticleSelector cosmictpSelector;
  TrackingParticleSelector dRtpSelector;				      

  edm::EDGetTokenT<SimHitTPAssociationProducer::SimHitTPAssociationList> _simHitTpMapTag;
  edm::EDGetTokenT<edm::View<reco::Track> > labelTokenForDrCalculation;

};


#endif
