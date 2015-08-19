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
#include "Validation/RecoTrack/interface/MTVHistoProducerAlgoForTracker.h"
#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociator.h"

class MultiTrackValidator : public DQMEDAnalyzer, protected MultiTrackValidatorBase {
 public:
  /// Constructor
  MultiTrackValidator(const edm::ParameterSet& pset);
  
  /// Destructor
  virtual ~MultiTrackValidator();


  /// Method called once per event
  void analyze(const edm::Event&, const edm::EventSetup& ) override;
  /// Method called to book the DQM histograms
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;


 protected:
  //these are used by MTVGenPs
  bool UseAssociators;
  const bool parametersDefinerIsCosmic_;
  const bool calculateDrSingleCollection_;
  const bool doPlotsOnlyForTruePV_;
  const bool doSummaryPlots_;
  const bool doSimPlots_;
  const bool doSimTrackPlots_;
  const bool doRecoTrackPlots_;
  const bool dodEdxPlots_;
  const bool doPVAssociationPlots_;
  std::unique_ptr<MTVHistoProducerAlgoForTracker> histoProducerAlgo_;

 private:
  std::vector<edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator>> associatorTokens;
  std::vector<edm::EDGetTokenT<reco::SimToRecoCollection>> associatormapStRs;
  std::vector<edm::EDGetTokenT<reco::RecoToSimCollection>> associatormapRtSs;

  std::string dirName_;

  bool useGsf;
  // select tracking particles 
  //(i.e. "denominator" of the efficiency ratio)
  TrackingParticleSelector tpSelector;				      
  CosmicTrackingParticleSelector cosmictpSelector;
  TrackingParticleSelector dRtpSelector;				      
  TrackingParticleSelector dRtpSelectorNoPtCut;

  edm::EDGetTokenT<SimHitTPAssociationProducer::SimHitTPAssociationList> _simHitTpMapTag;
  edm::EDGetTokenT<edm::View<reco::Track> > labelTokenForDrCalculation;
  edm::EDGetTokenT<edm::View<reco::Vertex> > recoVertexToken_;
  edm::EDGetTokenT<reco::VertexToTrackingVertexAssociator> vertexAssociatorToken_;

  std::vector<MonitorElement *> h_reco_coll, h_assoc_coll, h_assoc2_coll, h_simul_coll, h_looper_coll, h_pileup_coll;
  std::vector<MonitorElement *> h_assoc_coll_allPt, h_simul_coll_allPt;
};


#endif
