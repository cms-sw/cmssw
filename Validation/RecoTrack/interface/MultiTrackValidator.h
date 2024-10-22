#ifndef MultiTrackValidator_h
#define MultiTrackValidator_h

/** \class MultiTrackValidator
 *  Class that prodecs histrograms to validate Track Reconstruction performances
 *
 *  \author cerati
 */
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"

#include "Validation/RecoTrack/interface/MTVHistoProducerAlgoForTracker.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociator.h"
#include "CommonTools/RecoAlgos/interface/CosmicTrackingParticleSelector.h"
#include "SimTracker/Common/interface/TrackingParticleSelector.h"
#include "CommonTools/RecoAlgos/interface/RecoTrackSelectorBase.h"
#include "SimTracker/TrackAssociation/interface/ParametersDefinerForTP.h"
#include "CommonTools/Utils/interface/DynArray.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Candidate/interface/Candidate.h"

class PileupSummaryInfo;
namespace reco {
  class DeDxData;
}

struct MultiTrackValidatorHistograms {
  MTVHistoProducerAlgoForTrackerHistograms histoProducerAlgo;
  std::vector<dqm::reco::MonitorElement*> h_reco_coll, h_assoc_coll, h_assoc2_coll, h_simul_coll, h_looper_coll,
      h_pileup_coll;
};

class MultiTrackValidator : public DQMGlobalEDAnalyzer<MultiTrackValidatorHistograms> {
public:
  using Histograms = MultiTrackValidatorHistograms;

  /// Constructor
  MultiTrackValidator(const edm::ParameterSet& pset);

  /// Destructor
  ~MultiTrackValidator() override;

  /// Method called once per event
  void dqmAnalyze(const edm::Event&, const edm::EventSetup&, const Histograms&) const override;
  /// Method called to book the DQM histograms
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&, Histograms&) const override;

protected:
  // ES Tokens
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> tTopoEsToken;

  std::unique_ptr<ParametersDefinerForTP> parametersDefinerTP_;
  const bool parametersDefinerIsCosmic_;

  //these are used by MTVGenPs
  // MTV-specific data members
  std::vector<edm::InputTag> associators;
  edm::EDGetTokenT<TrackingParticleCollection> label_tp_effic;
  edm::EDGetTokenT<TrackingParticleCollection> label_tp_fake;
  edm::EDGetTokenT<TrackingParticleRefVector> label_tp_effic_refvector;
  edm::EDGetTokenT<TrackingParticleRefVector> label_tp_fake_refvector;
  edm::EDGetTokenT<TrackingVertexCollection> label_tv;
  edm::EDGetTokenT<std::vector<PileupSummaryInfo>> label_pileupinfo;

  std::vector<edm::EDGetTokenT<std::vector<PSimHit>>> simHitTokens_;

  std::vector<edm::InputTag> label;
  std::vector<edm::EDGetTokenT<edm::View<reco::Track>>> labelToken;
  std::vector<edm::EDGetTokenT<edm::View<TrajectorySeed>>> labelTokenSeed;
  edm::EDGetTokenT<reco::BeamSpot> bsSrc;

  edm::EDGetTokenT<edm::ValueMap<reco::DeDxData>> m_dEdx1Tag;
  edm::EDGetTokenT<edm::ValueMap<reco::DeDxData>> m_dEdx2Tag;

  const bool ignoremissingtkcollection_;
  const bool useAssociators_;
  const bool calculateDrSingleCollection_;
  const bool doPlotsOnlyForTruePV_;
  const bool doSummaryPlots_;
  const bool doSimPlots_;
  const bool doSimTrackPlots_;
  const bool doRecoTrackPlots_;
  const bool dodEdxPlots_;
  const bool doPVAssociationPlots_;
  const bool doSeedPlots_;
  const bool doMVAPlots_;
  const bool applyTPSelToSimMatch_;

  std::vector<bool> doResolutionPlots_;

  std::unique_ptr<MTVHistoProducerAlgoForTracker> histoProducerAlgo_;

private:
  const TrackingVertex::LorentzVector* getSimPVPosition(const edm::Handle<TrackingVertexCollection>& htv) const;
  const reco::Vertex::Point* getRecoPVPosition(const edm::Event& event,
                                               const edm::Handle<TrackingVertexCollection>& htv) const;
  void tpParametersAndSelection(
      const Histograms& histograms,
      const TrackingParticleRefVector& tPCeff,
      const edm::Event& event,
      const edm::EventSetup& setup,
      const reco::BeamSpot& bs,
      std::vector<std::tuple<TrackingParticle::Vector, TrackingParticle::Point>>& momVert_tPCeff,
      std::vector<size_t>& selected_tPCeff) const;
  size_t tpDR(const TrackingParticleRefVector& tPCeff,
              const std::vector<size_t>& selected_tPCeff,
              DynArray<float>& dR_tPCeff,
              DynArray<float>& dR_tPCeff_jet,
              const edm::View<reco::Candidate>* cores) const;
  void trackDR(const edm::View<reco::Track>& trackCollection,
               const edm::View<reco::Track>& trackCollectionDr,
               DynArray<float>& dR_trk,
               DynArray<float>& dR_trk_jet,
               const edm::View<reco::Candidate>* cores) const;

  std::vector<edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator>> associatorTokens;
  std::vector<edm::EDGetTokenT<reco::SimToRecoCollection>> associatormapStRs;
  std::vector<edm::EDGetTokenT<reco::RecoToSimCollection>> associatormapRtSs;

  edm::EDGetTokenT<edm::ValueMap<unsigned int>> tpNLayersToken_;
  edm::EDGetTokenT<edm::ValueMap<unsigned int>> tpNPixelLayersToken_;
  edm::EDGetTokenT<edm::ValueMap<unsigned int>> tpNStripStereoLayersToken_;

  using MVACollection = std::vector<float>;
  using QualityMaskCollection = std::vector<unsigned char>;
  std::vector<std::vector<std::tuple<edm::EDGetTokenT<MVACollection>, edm::EDGetTokenT<QualityMaskCollection>>>>
      mvaQualityCollectionTokens_;

  std::string dirName_;

  bool useGsf;
  const double simPVMaxZ_;

  edm::EDGetTokenT<edm::View<reco::Candidate>> cores_;
  double ptMinJet_;
  // select tracking particles
  //(i.e. "denominator" of the efficiency ratio)
  TrackingParticleSelector tpSelector;
  CosmicTrackingParticleSelector cosmictpSelector;
  TrackingParticleSelector dRtpSelector;
  std::unique_ptr<RecoTrackSelectorBase> dRTrackSelector;

  edm::EDGetTokenT<SimHitTPAssociationProducer::SimHitTPAssociationList> _simHitTpMapTag;
  edm::EDGetTokenT<edm::View<reco::Track>> labelTokenForDrCalculation;
  edm::EDGetTokenT<edm::View<reco::Vertex>> recoVertexToken_;
  edm::EDGetTokenT<reco::VertexToTrackingVertexAssociator> vertexAssociatorToken_;
};

#endif
