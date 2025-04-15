#include <string>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/ValueMap.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Common/interface/ValidHandle.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "DataFormats/Math/interface/angle_units.h"
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDTopology.h"
#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

#include "RecoMTD/DetLayers/interface/MTDDetLayerGeometry.h"
#include "RecoMTD/Records/interface/MTDRecoGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoMTD/DetLayers/interface/MTDDetLayerGeometry.h"
#include "RecoMTD/DetLayers/interface/MTDTrayBarrelLayer.h"
#include "RecoMTD/DetLayers/interface/MTDDetTray.h"
#include "RecoMTD/DetLayers/interface/MTDSectorForwardDoubleLayer.h"
#include "RecoMTD/DetLayers/interface/MTDDetSector.h"
#include "RecoMTD/Records/interface/MTDRecoGeometryRecord.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeomUtil.h"

#include "TrackPropagation/SteppingHelixPropagator/interface/SteppingHelixPropagator.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementError.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/MeasurementPoint.h"
#include "DataFormats/FTLRecHit/interface/FTLRecHitCollections.h"
#include "DataFormats/FTLRecHit/interface/FTLClusterCollections.h"
#include "DataFormats/TrackerRecHit2D/interface/MTDTrackingRecHit.h"

#include "DataFormats/Common/interface/OneToMany.h"
#include "DataFormats/Common/interface/AssociationMap.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/Associations/interface/MtdSimLayerClusterToTPAssociatorBaseImpl.h"
#include "SimDataFormats/CaloAnalysis/interface/MtdSimLayerCluster.h"
#include "SimDataFormats/Associations/interface/MtdRecoClusterToSimLayerClusterAssociationMap.h"
#include "SimDataFormats/Associations/interface/MtdSimLayerClusterToRecoClusterAssociationMap.h"

#include "CLHEP/Units/PhysicalConstants.h"
#include "MTDHit.h"

class MtdTracksValidation : public DQMEDAnalyzer {
public:
  explicit MtdTracksValidation(const edm::ParameterSet&);
  ~MtdTracksValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  const std::pair<bool, bool> checkAcceptance(
      const reco::Track&, const edm::Event&, const edm::EventSetup&, size_t&, float&, float&, float&, float&);

  const bool trkTPSelLV(const TrackingParticle&);
  const bool trkTPSelAll(const TrackingParticle&);
  const bool trkRecSel(const reco::TrackBase&);
  const bool trkRecSelLowPt(const reco::TrackBase&);
  const edm::Ref<std::vector<TrackingParticle>>* getMatchedTP(const reco::TrackBaseRef&);

  const unsigned long int uniqueId(const uint32_t x, const EncodedEventId& y) {
    const uint64_t a = static_cast<uint64_t>(x);
    const uint64_t b = static_cast<uint64_t>(y.rawId());

    if (x < y.rawId())
      return (b << 32) | a;
    else
      return (a << 32) | b;
  }

  bool isETL(const double eta) const { return (std::abs(eta) > trackMinEtlEta_) && (std::abs(eta) < trackMaxEtlEta_); }

  void fillTrackClusterMatchingHistograms(MonitorElement* me1,
                                          MonitorElement* me2,
                                          MonitorElement* me3,
                                          MonitorElement* me4,
                                          MonitorElement* me5,
                                          float var1,
                                          float var2,
                                          float var3,
                                          float var4,
                                          float var5,
                                          bool flag);

  // ------------ member data ------------

  const std::string folder_;
  const bool optionalPlots_;
  const float trackMaxPt_;
  const float trackMaxBtlEta_;
  const float trackMinEtlEta_;
  const float trackMaxEtlEta_;

  static constexpr double simUnit_ = 1e9;                // sim time in s while reco time in ns
  static constexpr double etacutGEN_ = 4.;               // |eta| < 4;
  static constexpr double etacutREC_ = 3.;               // |eta| < 3;
  static constexpr double pTcutBTL_ = 0.7;               // PT > 0.7 GeV
  static constexpr double pTcutETL_ = 0.2;               // PT > 0.2 GeV
  static constexpr double depositBTLthreshold_ = 1;      // threshold for energy deposit in BTL cell [MeV]
  static constexpr double depositETLthreshold_ = 0.001;  // threshold for energy deposit in ETL cell [MeV]
  static constexpr double rBTL_ = 110.0;
  static constexpr double zETL_ = 290.0;
  static constexpr double etaMatchCut_ = 0.05;
  static constexpr double cluDRradius_ = 0.05;  // to cluster rechits around extrapolated track

  const reco::RecoToSimCollection* r2s_;

  edm::EDGetTokenT<reco::TrackCollection> GenRecTrackToken_;
  edm::EDGetTokenT<reco::TrackCollection> RecTrackToken_;

  edm::EDGetTokenT<TrackingParticleCollection> trackingParticleCollectionToken_;
  edm::EDGetTokenT<reco::SimToRecoCollection> simToRecoAssociationToken_;
  edm::EDGetTokenT<reco::RecoToSimCollection> recoToSimAssociationToken_;
  edm::EDGetTokenT<reco::TPToSimCollectionMtd> tp2SimAssociationMapToken_;
  edm::EDGetTokenT<reco::SimToTPCollectionMtd> Sim2tpAssociationMapToken_;
  edm::EDGetTokenT<MtdRecoClusterToSimLayerClusterAssociationMap> r2sAssociationMapToken_;

  edm::EDGetTokenT<FTLRecHitCollection> btlRecHitsToken_;
  edm::EDGetTokenT<FTLRecHitCollection> etlRecHitsToken_;
  edm::EDGetTokenT<FTLClusterCollection> btlRecCluToken_;
  edm::EDGetTokenT<FTLClusterCollection> etlRecCluToken_;

  edm::EDGetTokenT<edm::ValueMap<int>> trackAssocToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> pathLengthToken_;
  
  edm::EDGetTokenT<edm::ValueMap<float>> btlMatchTimeChi2Token_;
  edm::EDGetTokenT<edm::ValueMap<float>> etlMatchTimeChi2Token_;
  edm::EDGetTokenT<edm::ValueMap<float>> btlMatchChi2Token_;

  edm::EDGetTokenT<edm::ValueMap<float>> tmtdToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> SigmatmtdToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> t0SrcToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> Sigmat0SrcToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> t0PidToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> Sigmat0PidToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> t0SafePidToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> Sigmat0SafePidToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> SigmaTofPiToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> SigmaTofKToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> SigmaTofPToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> trackMVAQualToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> outermostHitPositionToken_;

  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdgeoToken_;
  edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdtopoToken_;
  edm::ESGetToken<MTDDetLayerGeometry, MTDRecoGeometryRecord> mtdlayerToken_;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magfieldToken_;
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> builderToken_;

  MonitorElement* meBTLTrackRPTime_;
  MonitorElement* meBTLTrackEtaTot_;
  MonitorElement* meBTLTrackPhiTot_;
  MonitorElement* meBTLTrackPtTot_;
  MonitorElement* meBTLTrackEtaMtd_;
  MonitorElement* meBTLTrackPhiMtd_;
  MonitorElement* meBTLTrackPtMtd_;
  MonitorElement* meBTLTrackPtRes_;

  MonitorElement* meETLTrackRPTime_;
  MonitorElement* meETLTrackEtaTot_;
  MonitorElement* meETLTrackPhiTot_;
  MonitorElement* meETLTrackPtTot_;
  MonitorElement* meETLTrackEtaMtd_;
  MonitorElement* meETLTrackPhiMtd_;
  MonitorElement* meETLTrackPtMtd_;
  MonitorElement* meETLTrackEta2Mtd_;
  MonitorElement* meETLTrackPhi2Mtd_;
  MonitorElement* meETLTrackPt2Mtd_;
  MonitorElement* meETLTrackPtRes_;

  MonitorElement* meETLTrackEtaTotLowPt_[2];
  MonitorElement* meETLTrackEtaMtdLowPt_[2];
  MonitorElement* meETLTrackEta2MtdLowPt_[2];

  MonitorElement* meBTLTrackMatchedTPEtaTot_;
  MonitorElement* meBTLTrackMatchedTPPtTot_;
  MonitorElement* meBTLTrackMatchedTPEtaMtd_;
  MonitorElement* meBTLTrackMatchedTPPtMtd_;
  MonitorElement* meETLTrackMatchedTPEtaTot_;
  MonitorElement* meETLTrackMatchedTPPtTot_;
  MonitorElement* meETLTrackMatchedTPEtaMtd_;
  MonitorElement* meETLTrackMatchedTPPtMtd_;
  MonitorElement* meETLTrackMatchedTPEta2Mtd_;
  MonitorElement* meETLTrackMatchedTPPt2Mtd_;
  MonitorElement* meETLTrackMatchedTPEtaMtdCorrect_;
  MonitorElement* meETLTrackMatchedTPPtMtdCorrect_;

  MonitorElement* meTracktmtd_;
  MonitorElement* meTrackt0Src_;
  MonitorElement* meTrackSigmat0Src_;
  MonitorElement* meTrackt0Pid_;
  MonitorElement* meTrackSigmat0Pid_;
  MonitorElement* meTrackt0SafePid_;
  MonitorElement* meTrackSigmat0SafePid_;
  MonitorElement* meTrackNumHits_;
  MonitorElement* meTrackNumHitsNT_;
  MonitorElement* meTrackMVAQual_;
  MonitorElement* meTrackPathLenghtvsEta_;
  MonitorElement* meTrackOutermostHitR_;
  MonitorElement* meTrackOutermostHitZ_;

  MonitorElement* meTrackSigmaTof_[3];
  MonitorElement* meTrackSigmaTofvsP_[3];

  MonitorElement* meBTLTrackMatchedTPPtResMtd_;
  MonitorElement* meETLTrackMatchedTPPtResMtd_;
  MonitorElement* meETLTrackMatchedTP2PtResMtd_;
  MonitorElement* meBTLTrackMatchedTPPtRatioGen_;
  MonitorElement* meETLTrackMatchedTPPtRatioGen_;
  MonitorElement* meETLTrackMatchedTP2PtRatioGen_;
  MonitorElement* meBTLTrackMatchedTPPtRatioMtd_;
  MonitorElement* meETLTrackMatchedTPPtRatioMtd_;
  MonitorElement* meETLTrackMatchedTP2PtRatioMtd_;
  MonitorElement* meBTLTrackMatchedTPPtResvsPtMtd_;
  MonitorElement* meETLTrackMatchedTPPtResvsPtMtd_;
  MonitorElement* meETLTrackMatchedTP2PtResvsPtMtd_;
  MonitorElement* meBTLTrackMatchedTPDPtvsPtGen_;
  MonitorElement* meETLTrackMatchedTPDPtvsPtGen_;
  MonitorElement* meETLTrackMatchedTP2DPtvsPtGen_;
  MonitorElement* meBTLTrackMatchedTPDPtvsPtMtd_;
  MonitorElement* meETLTrackMatchedTPDPtvsPtMtd_;
  MonitorElement* meETLTrackMatchedTP2DPtvsPtMtd_;

  MonitorElement* meTrackResTot_;
  MonitorElement* meTrackPullTot_;
  MonitorElement* meTrackResTotvsMVAQual_;
  MonitorElement* meTrackPullTotvsMVAQual_;

  MonitorElement* meTrackMatchedTPPtTotLV_;
  MonitorElement* meTrackMatchedTPEtaTotLV_;
  MonitorElement* meExtraPtMtd_;
  MonitorElement* meExtraPtEtl2Mtd_;
  MonitorElement* meExtraEtaMtd_;
  MonitorElement* meExtraEtaEtl2Mtd_;
  MonitorElement* meExtraPhiAtBTL_;
  MonitorElement* meExtraPhiAtBTLmatched_;
  MonitorElement* meExtraBTLeneInCone_;
  MonitorElement* meExtraMTDfailExtenderEta_;
  MonitorElement* meExtraMTDfailExtenderPt_;

  // ====== Trak-cluster matching based on MC truth
  // - BTL TPmtd Direct, TPmtd Other, TPnomtd
  MonitorElement* meBTLTrackMatchedTPmtdDirectEta_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectPt_;

  MonitorElement* meBTLTrackMatchedTPmtdOtherEta_;
  MonitorElement* meBTLTrackMatchedTPmtdOtherPt_;

  MonitorElement* meBTLTrackMatchedTPnomtdEta_;
  MonitorElement* meBTLTrackMatchedTPnomtdPt_;
  // - BTL TPmtd Direct hits: correct, wrong, missing association in MTD
  MonitorElement* meBTLTrackMatchedTPmtdDirectCorrectAssocEta_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectCorrectAssocPt_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectCorrectAssocMVAQual_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectCorrectAssocTimeRes_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectCorrectAssocTimePull_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectCorrectAssocTrackOutermostHitR_; 
  MonitorElement* meBTLTrackMatchedTPmtdDirectCorrectAssocTrackChi2_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectCorrectAssocTimeChi2_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectCorrectAssocTimeChi2vsMVAQual_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectCorrectAssocSpaceChi2_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectCorrectAssocTrackNdf_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectCorrectAssocSimClusSize_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectCorrectAssocRecoClusSize_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectCorrectAssocTrackPathLenghtvsEta_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectCorrectAssocTrackPathLenght_;

  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocEta_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocPt_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocMVAQual_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTimeRes_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTimePull_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTimeChi2_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTimeChi2vsMVAQual_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocSpaceChi2_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocDeltaT_; 
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocDeltaZ_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocDeltaPhi_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTrackIdOff_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTrackOutermostHitR_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTrackChi2_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTrackNdf_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocSimClusSize_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocRecoClusSize_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocDeltaZOutR_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenghtvsEta_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenght_;
  
  // wrong association with reco from same TP direct hit
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocEta1_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocPt1_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocMVAQual1_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTimeRes1_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTimePull1_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocDeltaT1_; 
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocDeltaZ1_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocDeltaPhi1_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTrackIdOff1_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTrackOutermostHitR1_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTrackChi21_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTimeChi21_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTimeChi2vsMVAQual1_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocSpaceChi21_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTrackNdf1_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocDeltaZOutR1_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenghtvsEta1_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenght1_;

  // wrong association with reco from same TP other hit
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocEta2_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocPt2_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocMVAQual2_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTimeRes2_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTimePull2_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocDeltaT2_; 
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocDeltaZ2_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocDeltaPhi2_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTrackIdOff2_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTrackOutermostHitR2_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTrackChi22_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTimeChi22_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTimeChi2vsMVAQual2_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocSpaceChi22_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTrackNdf2_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocDeltaZOutR2_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenghtvsEta2_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenght2_;

  // wrong association to reco from another TP
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocEta3_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocPt3_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocMVAQual3_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTimeRes3_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTimePull3_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocDeltaT3_; 
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocDeltaZ3_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocDeltaPhi3_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTrackIdOff3_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTrackOutermostHitR3_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTrackChi23_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTimeChi23_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTimeChi2vsMVAQual3_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocSpaceChi23_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTrackNdf3_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocDeltaZOutR3_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenghtvsEta3_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenght3_;

  MonitorElement* meBTLTrackMatchedTPmtdDirectNoAssocEta_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectNoAssocPt_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectNoAssocTrackOutermostHitR_; 
  MonitorElement* meBTLTrackMatchedTPmtdDirectNoAssocTrackChi2_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectNoAssocTrackNdf_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectNoAssocSimClusSize_;
  MonitorElement* meBTLTrackMatchedTPmtdDirectNoAssocRecoClusSize_;


  // - BTL TPmtd "other" hits: correct, wrong, missing association in MTD
  MonitorElement* meBTLTrackMatchedTPmtdOtherCorrectAssocEta_;
  MonitorElement* meBTLTrackMatchedTPmtdOtherCorrectAssocPt_;
  MonitorElement* meBTLTrackMatchedTPmtdOtherCorrectAssocMVAQual_;
  MonitorElement* meBTLTrackMatchedTPmtdOtherCorrectAssocTimeRes_;
  MonitorElement* meBTLTrackMatchedTPmtdOtherCorrectAssocTimePull_;

  MonitorElement* meBTLTrackMatchedTPmtdOtherWrongAssocEta_;
  MonitorElement* meBTLTrackMatchedTPmtdOtherWrongAssocPt_;
  MonitorElement* meBTLTrackMatchedTPmtdOtherWrongAssocMVAQual_;
  MonitorElement* meBTLTrackMatchedTPmtdOtherWrongAssocTimeRes_;
  MonitorElement* meBTLTrackMatchedTPmtdOtherWrongAssocTimePull_;

  MonitorElement* meBTLTrackMatchedTPmtdOtherNoAssocEta_;
  MonitorElement* meBTLTrackMatchedTPmtdOtherNoAssocPt_;

  // - BTL TPnomtd but a reco cluster is associated
  MonitorElement* meBTLTrackMatchedTPnomtdAssocEta_;
  MonitorElement* meBTLTrackMatchedTPnomtdAssocPt_;
  MonitorElement* meBTLTrackMatchedTPnomtdAssocMVAQual_;
  MonitorElement* meBTLTrackMatchedTPnomtdAssocTimeRes_;
  MonitorElement* meBTLTrackMatchedTPnomtdAssocTimePull_;
  MonitorElement* meBTLTrackMatchedTPnomtdAssocTrackChi2_;		    
  MonitorElement* meBTLTrackMatchedTPnomtdAssocTrackNdf_ ;		   
  MonitorElement* meBTLTrackMatchedTPnomtdAssocTrackOutermostHitR_; 
  MonitorElement* meBTLTrackMatchedTPnomtdAssocTrackIdOff_; 
  MonitorElement* meBTLTrackMatchedTPnomtdAssocSimClusSize_;
  MonitorElement* meBTLTrackMatchedTPnomtdAssocRecoClusSize_;
  MonitorElement* meBTLTrackMatchedTPnomtdAssocTrackID_;

  // - ETL: one, two o no sim hits
  MonitorElement* meETLTrackMatchedTPmtd1Eta_;  // -- sim hit in >=1 etl disk
  MonitorElement* meETLTrackMatchedTPmtd1Pt_;
  MonitorElement* meETLTrackMatchedTPmtd2Eta_;  // -- sim hits in 2 etl disks
  MonitorElement* meETLTrackMatchedTPmtd2Pt_;
  MonitorElement* meETLTrackMatchedTPnomtdEta_;  // -- no sim hits in etl
  MonitorElement* meETLTrackMatchedTPnomtdPt_;

  // - ETL >=1 sim hit: each correct, at least one wrong, each sim hit missing reco association
  MonitorElement* meETLTrackMatchedTPmtd1CorrectAssocEta_;
  MonitorElement* meETLTrackMatchedTPmtd1CorrectAssocPt_;
  MonitorElement* meETLTrackMatchedTPmtd1CorrectAssocMVAQual_;
  MonitorElement* meETLTrackMatchedTPmtd1CorrectAssocTimeRes_;
  MonitorElement* meETLTrackMatchedTPmtd1CorrectAssocTimePull_;
  MonitorElement* meETLTrackMatchedTPmtd1CorrectAssocTimeChi2_;
  MonitorElement* meETLTrackMatchedTPmtd1CorrectAssocTimeChi2vsMVAQual_;

  MonitorElement* meETLTrackMatchedTPmtd1WrongAssocEta_;
  MonitorElement* meETLTrackMatchedTPmtd1WrongAssocPt_;
  MonitorElement* meETLTrackMatchedTPmtd1WrongAssocMVAQual_;
  MonitorElement* meETLTrackMatchedTPmtd1WrongAssocTimeRes_;
  MonitorElement* meETLTrackMatchedTPmtd1WrongAssocTimePull_;
  MonitorElement* meETLTrackMatchedTPmtd1WrongAssocTimeChi2_;
  MonitorElement* meETLTrackMatchedTPmtd1WrongAssocTimeChi2vsMVAQual_;

  MonitorElement* meETLTrackMatchedTPmtd1NoAssocEta_;
  MonitorElement* meETLTrackMatchedTPmtd1NoAssocPt_;
  MonitorElement* meETLTrackMatchedTPmtd1NoAssocMVAQual_;
  MonitorElement* meETLTrackMatchedTPmtd1NoAssocTimeRes_;
  MonitorElement* meETLTrackMatchedTPmtd1NoAssocTimePull_;

  // - ETL - 2 sim hits: both correct, at least one wrong or one missing, both missing reco association
  MonitorElement* meETLTrackMatchedTPmtd2CorrectAssocEta_;
  MonitorElement* meETLTrackMatchedTPmtd2CorrectAssocPt_;
  MonitorElement* meETLTrackMatchedTPmtd2CorrectAssocMVAQual_;
  MonitorElement* meETLTrackMatchedTPmtd2CorrectAssocTimeRes_;
  MonitorElement* meETLTrackMatchedTPmtd2CorrectAssocTimePull_;
  MonitorElement* meETLTrackMatchedTPmtd2CorrectAssocTimeChi2_;
  MonitorElement* meETLTrackMatchedTPmtd2CorrectAssocTimeChi2vsMVAQual_;

  MonitorElement* meETLTrackMatchedTPmtd2WrongAssocEta_;
  MonitorElement* meETLTrackMatchedTPmtd2WrongAssocPt_;
  MonitorElement* meETLTrackMatchedTPmtd2WrongAssocMVAQual_;
  MonitorElement* meETLTrackMatchedTPmtd2WrongAssocTimeRes_;
  MonitorElement* meETLTrackMatchedTPmtd2WrongAssocTimePull_;
  MonitorElement* meETLTrackMatchedTPmtd2WrongAssocTimeChi2_;
  MonitorElement* meETLTrackMatchedTPmtd2WrongAssocTimeChi2vsMVAQual_;

  MonitorElement* meETLTrackMatchedTPmtd2NoAssocEta_;
  MonitorElement* meETLTrackMatchedTPmtd2NoAssocPt_;
  MonitorElement* meETLTrackMatchedTPmtd2NoAssocMVAQual_;
  MonitorElement* meETLTrackMatchedTPmtd2NoAssocTimeRes_;
  MonitorElement* meETLTrackMatchedTPmtd2NoAssocTimePull_;

  // - ETL - no sim hits, but reco hit associated to the track
  MonitorElement* meETLTrackMatchedTPnomtdAssocEta_;
  MonitorElement* meETLTrackMatchedTPnomtdAssocPt_;
  MonitorElement* meETLTrackMatchedTPnomtdAssocMVAQual_;
  MonitorElement* meETLTrackMatchedTPnomtdAssocTimeRes_;
  MonitorElement* meETLTrackMatchedTPnomtdAssocTimePull_;
};

// ------------ constructor and destructor --------------
MtdTracksValidation::MtdTracksValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      optionalPlots_(iConfig.getParameter<bool>("optionalPlots")),
      trackMaxPt_(iConfig.getParameter<double>("trackMaximumPt")),
      trackMaxBtlEta_(iConfig.getParameter<double>("trackMaximumBtlEta")),
      trackMinEtlEta_(iConfig.getParameter<double>("trackMinimumEtlEta")),
      trackMaxEtlEta_(iConfig.getParameter<double>("trackMaximumEtlEta")) {
  GenRecTrackToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("inputTagG"));
  RecTrackToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("inputTagT"));

  trackingParticleCollectionToken_ =
      consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("SimTag"));
  simToRecoAssociationToken_ =
      consumes<reco::SimToRecoCollection>(iConfig.getParameter<edm::InputTag>("TPtoRecoTrackAssoc"));
  recoToSimAssociationToken_ =
      consumes<reco::RecoToSimCollection>(iConfig.getParameter<edm::InputTag>("TPtoRecoTrackAssoc"));
  tp2SimAssociationMapToken_ =
      consumes<reco::TPToSimCollectionMtd>(iConfig.getParameter<edm::InputTag>("tp2SimAssociationMapTag"));
  Sim2tpAssociationMapToken_ =
      consumes<reco::SimToTPCollectionMtd>(iConfig.getParameter<edm::InputTag>("Sim2tpAssociationMapTag"));
  r2sAssociationMapToken_ = consumes<MtdRecoClusterToSimLayerClusterAssociationMap>(
      iConfig.getParameter<edm::InputTag>("r2sAssociationMapTag"));
  btlRecHitsToken_ = consumes<FTLRecHitCollection>(iConfig.getParameter<edm::InputTag>("btlRecHits"));
  etlRecHitsToken_ = consumes<FTLRecHitCollection>(iConfig.getParameter<edm::InputTag>("etlRecHits"));
  btlRecCluToken_ = consumes<FTLClusterCollection>(iConfig.getParameter<edm::InputTag>("recCluTagBTL"));
  etlRecCluToken_ = consumes<FTLClusterCollection>(iConfig.getParameter<edm::InputTag>("recCluTagETL"));
  trackAssocToken_ = consumes<edm::ValueMap<int>>(iConfig.getParameter<edm::InputTag>("trackAssocSrc"));
  pathLengthToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("pathLengthSrc"));
  btlMatchTimeChi2Token_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("btlMatchTimeChi2"));
  etlMatchTimeChi2Token_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("etlMatchTimeChi2"));
  btlMatchChi2Token_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("btlMatchChi2"));
  tmtdToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("tmtd"));
  SigmatmtdToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("sigmatmtd"));
  t0SrcToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("t0Src"));
  Sigmat0SrcToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("sigmat0Src"));
  t0PidToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("t0PID"));
  Sigmat0PidToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("sigmat0PID"));
  t0SafePidToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("t0SafePID"));
  Sigmat0SafePidToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("sigmat0SafePID"));
  SigmaTofPiToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("sigmaTofPi"));
  SigmaTofKToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("sigmaTofK"));
  SigmaTofPToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("sigmaTofP"));
  trackMVAQualToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("trackMVAQual"));
  outermostHitPositionToken_ =
      consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("outermostHitPositionSrc"));
  mtdgeoToken_ = esConsumes<MTDGeometry, MTDDigiGeometryRecord>();
  mtdtopoToken_ = esConsumes<MTDTopology, MTDTopologyRcd>();
  mtdlayerToken_ = esConsumes<MTDDetLayerGeometry, MTDRecoGeometryRecord>();
  magfieldToken_ = esConsumes<MagneticField, IdealMagneticFieldRecord>();
  builderToken_ = esConsumes<TransientTrackBuilder, TransientTrackRecord>(edm::ESInputTag("", "TransientTrackBuilder"));
}

MtdTracksValidation::~MtdTracksValidation() {}

// ------------ method called for each event  ------------
void MtdTracksValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace geant_units::operators;
  using namespace std;

  auto GenRecTrackHandle = makeValid(iEvent.getHandle(GenRecTrackToken_));

  auto btlRecCluHandle = makeValid(iEvent.getHandle(btlRecCluToken_));
  auto etlRecCluHandle = makeValid(iEvent.getHandle(etlRecCluToken_));

  std::unordered_map<uint32_t, MTDHit> m_btlHits;
  std::unordered_map<uint32_t, MTDHit> m_etlHits;
  std::unordered_map<uint32_t, std::set<unsigned long int>> m_btlTrkPerCell;
  std::unordered_map<uint32_t, std::set<unsigned long int>> m_etlTrkPerCell;
  const auto& tp2SimAssociationMap = iEvent.get(tp2SimAssociationMapToken_);
  const auto& Sim2tpAssociationMap = iEvent.get(Sim2tpAssociationMapToken_);
  const auto& r2sAssociationMap = iEvent.get(r2sAssociationMapToken_);

  const auto& tMtd = iEvent.get(tmtdToken_);
  const auto& SigmatMtd = iEvent.get(SigmatmtdToken_);
  const auto& t0Src = iEvent.get(t0SrcToken_);
  const auto& Sigmat0Src = iEvent.get(Sigmat0SrcToken_);
  const auto& t0Pid = iEvent.get(t0PidToken_);
  const auto& Sigmat0Pid = iEvent.get(Sigmat0PidToken_);
  const auto& t0Safe = iEvent.get(t0SafePidToken_);
  const auto& Sigmat0Safe = iEvent.get(Sigmat0SafePidToken_);
  const auto& SigmaTofPi = iEvent.get(SigmaTofPiToken_);
  const auto& SigmaTofK = iEvent.get(SigmaTofKToken_);
  const auto& SigmaTofP = iEvent.get(SigmaTofPToken_);
  const auto& mtdQualMVA = iEvent.get(trackMVAQualToken_);
  const auto& trackAssoc = iEvent.get(trackAssocToken_);
  const auto& pathLength = iEvent.get(pathLengthToken_);
  const auto& btlMatchTimeChi2 = iEvent.get(btlMatchTimeChi2Token_);
  const auto& etlMatchTimeChi2 = iEvent.get(etlMatchTimeChi2Token_);
  const auto& btlMatchChi2 = iEvent.get(btlMatchChi2Token_);
  const auto& outermostHitPosition = iEvent.get(outermostHitPositionToken_);

  auto recoToSimH = makeValid(iEvent.getHandle(recoToSimAssociationToken_));
  r2s_ = recoToSimH.product();
  auto geometryHandle = iSetup.getTransientHandle(mtdgeoToken_); 
  const MTDGeometry* geom = geometryHandle.product();
  auto topologyHandle = iSetup.getTransientHandle(mtdtopoToken_);
  const MTDTopology *topology = topologyHandle.product();
  
  mtd::MTDGeomUtil geomUtil;
  geomUtil.setGeometry(geom);
  geomUtil.setTopology(topology);


  unsigned int index = 0;

  // --- Loop over all RECO tracks ---
  for (const auto& trackGen : *GenRecTrackHandle) {
    const reco::TrackRef trackref(iEvent.getHandle(GenRecTrackToken_), index);
    index++;

    if (trackAssoc[trackref] == -1) {
      LogWarning("mtdTracks") << "Extended track not associated";
      continue;
    }

    const reco::TrackRef mtdTrackref = reco::TrackRef(iEvent.getHandle(RecTrackToken_), trackAssoc[trackref]);
    const reco::Track& track = *mtdTrackref;

    bool isBTL = false;
    bool isETL = false;
    bool ETLdisc1 = false;
    bool ETLdisc2 = false;
    bool twoETLdiscs = false;
    bool noCrack = std::abs(trackGen.eta()) < trackMaxBtlEta_ || std::abs(trackGen.eta()) > trackMinEtlEta_;

    if (trkRecSel(trackGen)) {
      meTracktmtd_->Fill(tMtd[trackref]);
      if (std::round(SigmatMtd[trackref] - Sigmat0Pid[trackref]) != 0) {
        LogWarning("mtdTracks")
            << "TimeError associated to refitted track is different from TimeError stored in tofPID "
               "sigmat0 ValueMap: this should not happen";
      }

      meTrackt0Src_->Fill(t0Src[trackref]);
      meTrackSigmat0Src_->Fill(Sigmat0Src[trackref]);

      meTrackt0Pid_->Fill(t0Pid[trackref]);
      meTrackSigmat0Pid_->Fill(Sigmat0Pid[trackref]);
      meTrackt0SafePid_->Fill(t0Safe[trackref]);
      meTrackSigmat0SafePid_->Fill(std::log10(std::max(Sigmat0Safe[trackref], 0.001f)));
      meTrackMVAQual_->Fill(mtdQualMVA[trackref]);

      meTrackSigmaTof_[0]->Fill(SigmaTofPi[trackref] * 1e3);  //save as ps
      meTrackSigmaTof_[1]->Fill(SigmaTofK[trackref] * 1e3);
      meTrackSigmaTof_[2]->Fill(SigmaTofP[trackref] * 1e3);
      meTrackSigmaTofvsP_[0]->Fill(trackGen.p(), SigmaTofPi[trackref] * 1e3);
      meTrackSigmaTofvsP_[1]->Fill(trackGen.p(), SigmaTofK[trackref] * 1e3);
      meTrackSigmaTofvsP_[2]->Fill(trackGen.p(), SigmaTofP[trackref] * 1e3);

      meTrackPathLenghtvsEta_->Fill(std::abs(trackGen.eta()), pathLength[trackref]);
      bool MTDEtlZnegD1 = false;
      bool MTDEtlZnegD2 = false;
      bool MTDEtlZposD1 = false;
      bool MTDEtlZposD2 = false;
      std::vector<edm::Ref<edmNew::DetSetVector<FTLCluster>, FTLCluster>> recoClustersRefs;

      if (std::abs(trackGen.eta()) < trackMaxBtlEta_) {
        // --- all BTL tracks (with and without hit in MTD) ---
        meBTLTrackEtaTot_->Fill(std::abs(trackGen.eta()));
        meBTLTrackPhiTot_->Fill(trackGen.phi());
        meBTLTrackPtTot_->Fill(trackGen.pt());

        bool MTDBtl = false;
        int numMTDBtlvalidhits = 0;
        for (const auto hit : track.recHits()) {
          if (hit->isValid() == false)
            continue;
          MTDDetId Hit = hit->geographicalId();
          if ((Hit.det() == 6) && (Hit.subdetId() == 1) && (Hit.mtdSubDetector() == 1)) {
            MTDBtl = true;
            numMTDBtlvalidhits++;
            const auto* mtdhit = static_cast<const MTDTrackingRecHit*>(hit);
            const auto& hitCluster = mtdhit->mtdCluster();
            if (hitCluster.size() != 0) {
              auto recoClusterRef = edmNew::makeRefTo(btlRecCluHandle, &hitCluster);
              recoClustersRefs.push_back(recoClusterRef);
            }
          }
        }
        meTrackNumHits_->Fill(numMTDBtlvalidhits);
        // --- keeping only tracks with last hit in MTD ---
        if (MTDBtl == true) {
          isBTL = true;
          meBTLTrackEtaMtd_->Fill(std::abs(trackGen.eta()));
          meBTLTrackPhiMtd_->Fill(trackGen.phi());
          meBTLTrackPtMtd_->Fill(trackGen.pt());
          meBTLTrackRPTime_->Fill(track.t0());
          meBTLTrackPtRes_->Fill((trackGen.pt() - track.pt()) / trackGen.pt());
        }
        if (isBTL && Sigmat0Safe[trackref] < 0.) {
          meTrackNumHitsNT_->Fill(numMTDBtlvalidhits);
        }
      }  //loop over (geometrical) BTL tracks

      else {
        // --- all ETL tracks (with and without hit in MTD) ---
        meETLTrackEtaTot_->Fill(std::abs(trackGen.eta()));
        meETLTrackPhiTot_->Fill(trackGen.phi());
        meETLTrackPtTot_->Fill(trackGen.pt());

        int numMTDEtlvalidhits = 0;
        for (const auto hit : track.recHits()) {
          if (hit->isValid() == false)
            continue;
          MTDDetId Hit = hit->geographicalId();
          if ((Hit.det() == 6) && (Hit.subdetId() == 1) && (Hit.mtdSubDetector() == 2)) {
            isETL = true;
            ETLDetId ETLHit = hit->geographicalId();

            const auto* mtdhit = static_cast<const MTDTrackingRecHit*>(hit);
            const auto& hitCluster = mtdhit->mtdCluster();
            if (hitCluster.size() != 0) {
              auto recoClusterRef = edmNew::makeRefTo(etlRecCluHandle, &hitCluster);
              recoClustersRefs.push_back(recoClusterRef);
            }

            if ((ETLHit.zside() == -1) && (ETLHit.nDisc() == 1)) {
              MTDEtlZnegD1 = true;
              meETLTrackRPTime_->Fill(track.t0());
              meETLTrackPtRes_->Fill((trackGen.pt() - track.pt()) / trackGen.pt());
              numMTDEtlvalidhits++;
            }
            if ((ETLHit.zside() == -1) && (ETLHit.nDisc() == 2)) {
              MTDEtlZnegD2 = true;
              meETLTrackRPTime_->Fill(track.t0());
              meETLTrackPtRes_->Fill((trackGen.pt() - track.pt()) / trackGen.pt());
              numMTDEtlvalidhits++;
            }
            if ((ETLHit.zside() == 1) && (ETLHit.nDisc() == 1)) {
              MTDEtlZposD1 = true;
              meETLTrackRPTime_->Fill(track.t0());
              meETLTrackPtRes_->Fill((trackGen.pt() - track.pt()) / trackGen.pt());
              numMTDEtlvalidhits++;
            }
            if ((ETLHit.zside() == 1) && (ETLHit.nDisc() == 2)) {
              MTDEtlZposD2 = true;
              meETLTrackRPTime_->Fill(track.t0());
              meETLTrackPtRes_->Fill((trackGen.pt() - track.pt()) / trackGen.pt());
              numMTDEtlvalidhits++;
            }
          }
        }
        meTrackNumHits_->Fill(-numMTDEtlvalidhits);
        if (isETL && Sigmat0Safe[trackref] < 0.) {
          meTrackNumHitsNT_->Fill(-numMTDEtlvalidhits);
        }

        // --- keeping only tracks with last hit in MTD ---
        ETLdisc1 = (MTDEtlZnegD1 || MTDEtlZposD1);
        ETLdisc2 = (MTDEtlZnegD2 || MTDEtlZposD2);
        twoETLdiscs =
            ((MTDEtlZnegD1 == true) && (MTDEtlZnegD2 == true)) || ((MTDEtlZposD1 == true) && (MTDEtlZposD2 == true));
        if (ETLdisc1 || ETLdisc2) {
          meETLTrackEtaMtd_->Fill(std::abs(trackGen.eta()));
          meETLTrackPhiMtd_->Fill(trackGen.phi());
          meETLTrackPtMtd_->Fill(trackGen.pt());
          if (twoETLdiscs) {
            meETLTrackEta2Mtd_->Fill(std::abs(trackGen.eta()));
            meETLTrackPhi2Mtd_->Fill(trackGen.phi());
            meETLTrackPt2Mtd_->Fill(trackGen.pt());
          }
        }
      }

      if (isBTL)
        meTrackOutermostHitR_->Fill(outermostHitPosition[trackref]);
      if (isETL)
        meTrackOutermostHitZ_->Fill(std::abs(outermostHitPosition[trackref]));

      LogDebug("MtdTracksValidation") << "Track p/pt = " << trackGen.p() << " " << trackGen.pt() << " eta "
                                      << trackGen.eta() << " BTL " << isBTL << " ETL " << isETL << " 2disks "
                                      << twoETLdiscs;

      // == TrackingParticle based matching
      const reco::TrackBaseRef trkrefb(trackref);
      auto tp_info = getMatchedTP(trkrefb);
      if (tp_info != nullptr && trkTPSelAll(**tp_info)) {
        // -- pT resolution plots
        if (optionalPlots_) {
          if (trackGen.pt() < trackMaxPt_) {
            if (isBTL) {
              meBTLTrackMatchedTPPtResMtd_->Fill(std::abs(track.pt() - (*tp_info)->pt()) /
                                                 std::abs(trackGen.pt() - (*tp_info)->pt()));
              meBTLTrackMatchedTPPtRatioGen_->Fill(trackGen.pt() / (*tp_info)->pt());
              meBTLTrackMatchedTPPtRatioMtd_->Fill(track.pt() / (*tp_info)->pt());
              meBTLTrackMatchedTPPtResvsPtMtd_->Fill(
                  (*tp_info)->pt(),
                  std::abs(track.pt() - (*tp_info)->pt()) / std::abs(trackGen.pt() - (*tp_info)->pt()));
              meBTLTrackMatchedTPDPtvsPtGen_->Fill((*tp_info)->pt(),
                                                   (trackGen.pt() - (*tp_info)->pt()) / (*tp_info)->pt());
              meBTLTrackMatchedTPDPtvsPtMtd_->Fill((*tp_info)->pt(),
                                                   (track.pt() - (*tp_info)->pt()) / (*tp_info)->pt());
            }
            if (isETL && !twoETLdiscs && (std::abs(trackGen.eta()) > trackMinEtlEta_) &&
                (std::abs(trackGen.eta()) < trackMaxEtlEta_)) {
              meETLTrackMatchedTPPtResMtd_->Fill(std::abs(track.pt() - (*tp_info)->pt()) /
                                                 std::abs(trackGen.pt() - (*tp_info)->pt()));
              meETLTrackMatchedTPPtRatioGen_->Fill(trackGen.pt() / (*tp_info)->pt());
              meETLTrackMatchedTPPtRatioMtd_->Fill(track.pt() / (*tp_info)->pt());
              meETLTrackMatchedTPPtResvsPtMtd_->Fill(
                  (*tp_info)->pt(),
                  std::abs(track.pt() - (*tp_info)->pt()) / std::abs(trackGen.pt() - (*tp_info)->pt()));
              meETLTrackMatchedTPDPtvsPtGen_->Fill((*tp_info)->pt(),
                                                   (trackGen.pt() - (*tp_info)->pt()) / ((*tp_info)->pt()));
              meETLTrackMatchedTPDPtvsPtMtd_->Fill((*tp_info)->pt(),
                                                   (track.pt() - (*tp_info)->pt()) / ((*tp_info)->pt()));
            }
            if (isETL && twoETLdiscs) {
              meETLTrackMatchedTP2PtResMtd_->Fill(std::abs(track.pt() - (*tp_info)->pt()) /
                                                  std::abs(trackGen.pt() - (*tp_info)->pt()));
              meETLTrackMatchedTP2PtRatioGen_->Fill(trackGen.pt() / (*tp_info)->pt());
              meETLTrackMatchedTP2PtRatioMtd_->Fill(track.pt() / (*tp_info)->pt());
              meETLTrackMatchedTP2PtResvsPtMtd_->Fill(
                  (*tp_info)->pt(),
                  std::abs(track.pt() - (*tp_info)->pt()) / std::abs(trackGen.pt() - (*tp_info)->pt()));
              meETLTrackMatchedTP2DPtvsPtGen_->Fill((*tp_info)->pt(),
                                                    (trackGen.pt() - (*tp_info)->pt()) / ((*tp_info)->pt()));
              meETLTrackMatchedTP2DPtvsPtMtd_->Fill((*tp_info)->pt(),
                                                    (track.pt() - (*tp_info)->pt()) / ((*tp_info)->pt()));
            }
          }
        }

        // -- Track matched to TP: all and with last hit in MTD
        if (std::abs(trackGen.eta()) < trackMaxBtlEta_) {
          meBTLTrackMatchedTPEtaTot_->Fill(std::abs(trackGen.eta()));
          meBTLTrackMatchedTPPtTot_->Fill(trackGen.pt());
          if (isBTL) {
            meBTLTrackMatchedTPEtaMtd_->Fill(std::abs(trackGen.eta()));
            meBTLTrackMatchedTPPtMtd_->Fill(trackGen.pt());
          }
        } else {
          meETLTrackMatchedTPEtaTot_->Fill(std::abs(trackGen.eta()));
          meETLTrackMatchedTPPtTot_->Fill(trackGen.pt());
          if (isETL) {
            meETLTrackMatchedTPEtaMtd_->Fill(std::abs(trackGen.eta()));
            meETLTrackMatchedTPPtMtd_->Fill(trackGen.pt());
            if (twoETLdiscs) {
              meETLTrackMatchedTPEta2Mtd_->Fill(std::abs(trackGen.eta()));
              meETLTrackMatchedTPPt2Mtd_->Fill(trackGen.pt());
            }
          }
        }

        if (noCrack) {
          if (trkTPSelLV(**tp_info)) {
            meTrackMatchedTPEtaTotLV_->Fill(std::abs(trackGen.eta()));
            meTrackMatchedTPPtTotLV_->Fill(trackGen.pt());
          }
        }

        bool hasTime = false;
        double tsim = (*tp_info)->parentVertex()->position().t() * simUnit_;
        double dT(-9999.);
        double pullT(-9999.);
        if (Sigmat0Safe[trackref] != -1.) {
          dT = t0Safe[trackref] - tsim;
          pullT = dT / Sigmat0Safe[trackref];
          hasTime = true;
        }



        // ==  MC truth matching
        double simClusterRef_RecoMatch_trackIdOff(-9999.); 
        double simClusterRef_RecoMatch_DeltaZ(-9999.);
        double simClusterRef_RecoMatch_DeltaPhi(-9999.);
        double simClusterRef_RecoMatch_DeltaT(-9999.);
        int simClusSize(-9999);
        int recoClusSize(-9999);

 
        bool isTPmtdDirectBTL = false, isTPmtdOtherBTL = false, isTPmtdDirectCorrectBTL = false,
             isTPmtdOtherCorrectBTL = false, isTPmtdETLD1 = false, isTPmtdETLD2 = false, isTPmtdCorrectETLD1 = false,
             isTPmtdCorrectETLD2 = false, isFromSameTP = false;

        auto simClustersRefsIt = tp2SimAssociationMap.find(*tp_info);
        const bool withMTD = (simClustersRefsIt != tp2SimAssociationMap.end());
        if (withMTD) {
          // -- Get the refs to MtdSimLayerClusters associated to the TP
          std::vector<edm::Ref<MtdSimLayerClusterCollection>> simClustersRefs;
          for (const auto& ref : simClustersRefsIt->val) {
            simClustersRefs.push_back(ref);
            MTDDetId mtddetid = ref->detIds_and_rows().front().first;
            if (mtddetid.mtdSubDetector() == 2) {
              ETLDetId detid(mtddetid.rawId());
              if (detid.nDisc() == 1)
                isTPmtdETLD1 = true;
              if (detid.nDisc() == 2)
                isTPmtdETLD2 = true;
            }
          }
          // === BTL
          // -- Sort BTL sim clusters by time
          std::vector<edm::Ref<MtdSimLayerClusterCollection>>::iterator directSimClusIt;
          if (std::abs(trackGen.eta()) < trackMaxBtlEta_ && !simClustersRefs.empty()) {
            std::sort(simClustersRefs.begin(), simClustersRefs.end(), [](const auto& a, const auto& b) {
              return a->simLCTime() < b->simLCTime();
            });
            // Find the first direct hit in time
            directSimClusIt = std::find_if(simClustersRefs.begin(), simClustersRefs.end(), [](const auto& simCluster) {
              MTDDetId mtddetid = simCluster->detIds_and_rows().front().first;
              return (mtddetid.mtdSubDetector() == 1 && simCluster->trackIdOffset() == 0);
            });
            // Check if TP has direct or other sim cluster for BTL
            for (const auto& simClusterRef : simClustersRefs) {
              if (directSimClusIt != simClustersRefs.end() && simClusterRef == *directSimClusIt) {
                isTPmtdDirectBTL = true;
              } else if (simClusterRef->trackIdOffset() != 0) {
                isTPmtdOtherBTL = true;
              }
            }
          }

          // ==  Check if the track-cluster association is correct: Track->RecoClus->SimClus == Track->TP->SimClus
	  recoClusSize = recoClustersRefs.size();
          for (const auto& recClusterRef : recoClustersRefs) {
            if (recClusterRef.isNonnull()) {
              auto itp = r2sAssociationMap.equal_range(recClusterRef);
	      simClusSize = 0;
              if (itp.first != itp.second) {
                auto& simClustersRefs_RecoMatch = (*itp.first).second;
		
                BTLDetId RecoDetId((*recClusterRef).id());
	        simClusSize = simClustersRefs_RecoMatch.size();
	
                for (const auto& simClusterRef_RecoMatch : simClustersRefs_RecoMatch) {
		  	
                  // Check if simClusterRef_RecoMatch  exists in SimClusters
                  auto simClusterIt =
                      std::find(simClustersRefs.begin(), simClustersRefs.end(), simClusterRef_RecoMatch);
	          if (optionalPlots_ && isTPmtdDirectBTL){
	  
                      // simCluster matched to TP 
		      // NB we are taking the position and id of the first hit in the cluster.
		      auto directSimClus = *directSimClusIt;
		      MTDDetId mtddetid = directSimClus->detIds_and_rows().front().first;
                      BTLDetId detid(mtddetid.rawId());
	              LocalPoint simClusLocalPos = directSimClus->hits_and_positions().front().second;
                      GlobalPoint simClusGlobalPos = geomUtil.globalPosition(detid, simClusLocalPos);
       
		      // simClusterRef_RecoMatch infos
  	              MTDDetId mtddetidRecoMatch = simClusterRef_RecoMatch->detIds_and_rows().front().first;
                      BTLDetId detidRecoMatch(mtddetidRecoMatch.rawId());
	              LocalPoint simClusRecoMatchLocalPos = simClusterRef_RecoMatch->hits_and_positions().front().second;
                      GlobalPoint simClusRecoMatchGlobalPos = geomUtil.globalPosition(detidRecoMatch, simClusRecoMatchLocalPos);

		      simClusterRef_RecoMatch_trackIdOff = simClusterRef_RecoMatch->trackIdOffset();
                      simClusterRef_RecoMatch_DeltaZ = simClusRecoMatchGlobalPos.z() - simClusGlobalPos.z();
                      simClusterRef_RecoMatch_DeltaPhi = simClusRecoMatchGlobalPos.phi() - simClusGlobalPos.phi();
                      simClusterRef_RecoMatch_DeltaT = simClusterRef_RecoMatch->simLCTime() - directSimClus->simLCTime();
      
		    }


                  // SimCluster found in SimClusters
                  if (simClusterIt != simClustersRefs.end()) {
		    isFromSameTP = true;
                    if (isBTL) {
                      if (directSimClusIt != simClustersRefs.end() && simClusterRef_RecoMatch == *directSimClusIt) {
                        isTPmtdDirectCorrectBTL = true;
	              } else if (simClusterRef_RecoMatch->trackIdOffset() != 0) {
                        isTPmtdOtherCorrectBTL = true;
                      }
                    }
                    if (isETL) {
                      MTDDetId mtddetid = (*simClusterIt)->detIds_and_rows().front().first;
                      ETLDetId detid(mtddetid.rawId());
                      if (detid.nDisc() == 1)
                        isTPmtdCorrectETLD1 = true;
                      if (detid.nDisc() == 2)
                        isTPmtdCorrectETLD2 = true;
                    }
                  }
                }
              }
            }
          }  /// end loop over reco clusters associated to this track.

          // == BTL
          if (std::abs(trackGen.eta()) < trackMaxBtlEta_) {
            // -- Track matched to TP with sim hit in MTD
            if (isTPmtdDirectBTL) {
              meBTLTrackMatchedTPmtdDirectEta_->Fill(std::abs(trackGen.eta()));
              meBTLTrackMatchedTPmtdDirectPt_->Fill(trackGen.pt());
            } else if (isTPmtdOtherBTL) {
              meBTLTrackMatchedTPmtdOtherEta_->Fill(std::abs(trackGen.eta()));
              meBTLTrackMatchedTPmtdOtherPt_->Fill(trackGen.pt());
            }
            //-- Track matched to TP with sim hit in MTD, with associated reco cluster
            if (isBTL) {
              if (isTPmtdDirectBTL) {
                // -- Track matched to TP with sim hit (direct), correctly associated reco cluster
                if (isTPmtdDirectCorrectBTL) {
	          
                  if (optionalPlots_) {
		    meBTLTrackMatchedTPmtdDirectCorrectAssocSimClusSize_->Fill(simClusSize);
                    meBTLTrackMatchedTPmtdDirectCorrectAssocRecoClusSize_->Fill(recoClusSize);
      	            meBTLTrackMatchedTPmtdDirectCorrectAssocTrackOutermostHitR_->Fill(outermostHitPosition[trackref]);
	            meBTLTrackMatchedTPmtdDirectCorrectAssocTrackNdf_->Fill(trackGen.ndof());
	            meBTLTrackMatchedTPmtdDirectCorrectAssocTrackChi2_->Fill(trackGen.chi2());
                    meBTLTrackMatchedTPmtdDirectCorrectAssocTimeChi2_->Fill(btlMatchTimeChi2[trackref]);
                    meBTLTrackMatchedTPmtdDirectCorrectAssocTimeChi2vsMVAQual_->Fill(btlMatchTimeChi2[trackref],  mtdQualMVA[trackref]);
                    meBTLTrackMatchedTPmtdDirectCorrectAssocSpaceChi2_->Fill(btlMatchChi2[trackref]);
                    meBTLTrackMatchedTPmtdDirectCorrectAssocTrackPathLenghtvsEta_->Fill(std::abs(trackGen.eta()), pathLength[trackref]);
                    meBTLTrackMatchedTPmtdDirectCorrectAssocTrackPathLenght_->Fill(pathLength[trackref]);

		  }
                  fillTrackClusterMatchingHistograms(meBTLTrackMatchedTPmtdDirectCorrectAssocEta_,
                                                     meBTLTrackMatchedTPmtdDirectCorrectAssocPt_,
                                                     meBTLTrackMatchedTPmtdDirectCorrectAssocMVAQual_,
                                                     meBTLTrackMatchedTPmtdDirectCorrectAssocTimeRes_,
                                                     meBTLTrackMatchedTPmtdDirectCorrectAssocTimePull_,
                                                     std::abs(trackGen.eta()),
                                                     trackGen.pt(),
                                                     mtdQualMVA[trackref],
                                                     dT,
                                                     pullT,
                                                     hasTime);
                }
                // -- Track matched to TP with sim hit (direct), incorrectly associated reco cluster
                else {
                  if (optionalPlots_) {
		    meBTLTrackMatchedTPmtdDirectWrongAssocSimClusSize_->Fill(simClusSize);
                    meBTLTrackMatchedTPmtdDirectWrongAssocRecoClusSize_->Fill(recoClusSize);
		    meBTLTrackMatchedTPmtdDirectWrongAssocDeltaT_->Fill(simClusterRef_RecoMatch_DeltaT);
		    meBTLTrackMatchedTPmtdDirectWrongAssocDeltaPhi_->Fill(simClusterRef_RecoMatch_DeltaPhi);
		    meBTLTrackMatchedTPmtdDirectWrongAssocDeltaZ_->Fill(simClusterRef_RecoMatch_DeltaZ);
		    meBTLTrackMatchedTPmtdDirectWrongAssocTrackIdOff_->Fill(simClusterRef_RecoMatch_trackIdOff);
		    meBTLTrackMatchedTPmtdDirectWrongAssocTrackOutermostHitR_->Fill(outermostHitPosition[trackref]);
                    meBTLTrackMatchedTPmtdDirectWrongAssocTrackNdf_->Fill(trackGen.ndof());
                    meBTLTrackMatchedTPmtdDirectWrongAssocTrackChi2_->Fill(trackGen.chi2());
                    meBTLTrackMatchedTPmtdDirectWrongAssocDeltaZOutR_->Fill(outermostHitPosition[trackref], simClusterRef_RecoMatch_DeltaZ);
                    meBTLTrackMatchedTPmtdDirectWrongAssocTimeChi2_->Fill(btlMatchTimeChi2[trackref]);
                    meBTLTrackMatchedTPmtdDirectWrongAssocTimeChi2vsMVAQual_->Fill(btlMatchTimeChi2[trackref],  mtdQualMVA[trackref]);
                    meBTLTrackMatchedTPmtdDirectWrongAssocSpaceChi2_->Fill(btlMatchChi2[trackref]);
                    meBTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenghtvsEta_->Fill(std::abs(trackGen.eta()), pathLength[trackref]);
                    meBTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenght_->Fill(pathLength[trackref]);



                    if (simClusterRef_RecoMatch_trackIdOff == 0 && isFromSameTP){

      			    fillTrackClusterMatchingHistograms(meBTLTrackMatchedTPmtdDirectWrongAssocEta1_,
                                     meBTLTrackMatchedTPmtdDirectWrongAssocPt1_,
                                     meBTLTrackMatchedTPmtdDirectWrongAssocMVAQual1_,
                                     meBTLTrackMatchedTPmtdDirectWrongAssocTimeRes1_,
                                     meBTLTrackMatchedTPmtdDirectWrongAssocTimePull1_,
                                     std::abs(trackGen.eta()),
                                     trackGen.pt(),
                                     mtdQualMVA[trackref],
                                     dT,
                                     pullT,
                                     hasTime);
        	    
		      meBTLTrackMatchedTPmtdDirectWrongAssocDeltaT1_->Fill(simClusterRef_RecoMatch_DeltaT);
		      meBTLTrackMatchedTPmtdDirectWrongAssocDeltaPhi1_->Fill(simClusterRef_RecoMatch_DeltaPhi);
		      meBTLTrackMatchedTPmtdDirectWrongAssocDeltaZ1_->Fill(simClusterRef_RecoMatch_DeltaZ);
		      meBTLTrackMatchedTPmtdDirectWrongAssocTrackIdOff1_->Fill(simClusterRef_RecoMatch_trackIdOff);
		      meBTLTrackMatchedTPmtdDirectWrongAssocTrackOutermostHitR1_->Fill(outermostHitPosition[trackref]);
                      meBTLTrackMatchedTPmtdDirectWrongAssocTrackNdf1_->Fill(trackGen.ndof());
                      meBTLTrackMatchedTPmtdDirectWrongAssocTrackChi21_->Fill(trackGen.chi2());
                      meBTLTrackMatchedTPmtdDirectWrongAssocDeltaZOutR1_->Fill(outermostHitPosition[trackref], simClusterRef_RecoMatch_DeltaZ);
                      meBTLTrackMatchedTPmtdDirectWrongAssocTimeChi21_->Fill(btlMatchTimeChi2[trackref]);
                      meBTLTrackMatchedTPmtdDirectWrongAssocTimeChi2vsMVAQual1_->Fill(btlMatchTimeChi2[trackref],  mtdQualMVA[trackref]);
                      meBTLTrackMatchedTPmtdDirectWrongAssocSpaceChi21_->Fill(btlMatchChi2[trackref]);
                      meBTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenghtvsEta1_->Fill(std::abs(trackGen.eta()), pathLength[trackref]);
                      meBTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenght1_->Fill(pathLength[trackref]);

                    } else if (simClusterRef_RecoMatch_trackIdOff > 0 && isFromSameTP){


	            		     fillTrackClusterMatchingHistograms(meBTLTrackMatchedTPmtdDirectWrongAssocEta2_,
                                     meBTLTrackMatchedTPmtdDirectWrongAssocPt2_,
                                     meBTLTrackMatchedTPmtdDirectWrongAssocMVAQual2_,
                                     meBTLTrackMatchedTPmtdDirectWrongAssocTimeRes2_,
                                     meBTLTrackMatchedTPmtdDirectWrongAssocTimePull2_,
                                     std::abs(trackGen.eta()),
                                     trackGen.pt(),
                                     mtdQualMVA[trackref],
                                     dT,
                                     pullT,
                                     hasTime);
        	    
		      meBTLTrackMatchedTPmtdDirectWrongAssocDeltaT2_->Fill(simClusterRef_RecoMatch_DeltaT);
		      meBTLTrackMatchedTPmtdDirectWrongAssocDeltaPhi2_->Fill(simClusterRef_RecoMatch_DeltaPhi);
		      meBTLTrackMatchedTPmtdDirectWrongAssocDeltaZ2_->Fill(simClusterRef_RecoMatch_DeltaZ);
		      meBTLTrackMatchedTPmtdDirectWrongAssocTrackIdOff2_->Fill(simClusterRef_RecoMatch_trackIdOff);
		      meBTLTrackMatchedTPmtdDirectWrongAssocTrackOutermostHitR2_->Fill(outermostHitPosition[trackref]);
                      meBTLTrackMatchedTPmtdDirectWrongAssocTrackNdf2_->Fill(trackGen.ndof());
                      meBTLTrackMatchedTPmtdDirectWrongAssocTrackChi22_->Fill(trackGen.chi2());
                      meBTLTrackMatchedTPmtdDirectWrongAssocDeltaZOutR2_->Fill(outermostHitPosition[trackref], simClusterRef_RecoMatch_DeltaZ);
                      meBTLTrackMatchedTPmtdDirectWrongAssocTimeChi22_->Fill(btlMatchTimeChi2[trackref]);
                      meBTLTrackMatchedTPmtdDirectWrongAssocTimeChi2vsMVAQual2_->Fill(btlMatchTimeChi2[trackref],  mtdQualMVA[trackref]);
                      meBTLTrackMatchedTPmtdDirectWrongAssocSpaceChi22_->Fill(btlMatchChi2[trackref]);
                      meBTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenghtvsEta2_->Fill(std::abs(trackGen.eta()), pathLength[trackref]);
                      meBTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenght2_->Fill(pathLength[trackref]);



                    } else if (!isFromSameTP){

	            		     fillTrackClusterMatchingHistograms(meBTLTrackMatchedTPmtdDirectWrongAssocEta3_,
                                     meBTLTrackMatchedTPmtdDirectWrongAssocPt3_,
                                     meBTLTrackMatchedTPmtdDirectWrongAssocMVAQual3_,
                                     meBTLTrackMatchedTPmtdDirectWrongAssocTimeRes3_,
                                     meBTLTrackMatchedTPmtdDirectWrongAssocTimePull3_,
                                     std::abs(trackGen.eta()),
                                     trackGen.pt(),
                                     mtdQualMVA[trackref],
                                     dT,
                                     pullT,
                                     hasTime);
        	    
		      meBTLTrackMatchedTPmtdDirectWrongAssocDeltaT3_->Fill(simClusterRef_RecoMatch_DeltaT);
		      meBTLTrackMatchedTPmtdDirectWrongAssocDeltaPhi3_->Fill(simClusterRef_RecoMatch_DeltaPhi);
		      meBTLTrackMatchedTPmtdDirectWrongAssocDeltaZ3_->Fill(simClusterRef_RecoMatch_DeltaZ);
		      meBTLTrackMatchedTPmtdDirectWrongAssocTrackIdOff3_->Fill(simClusterRef_RecoMatch_trackIdOff);
		      meBTLTrackMatchedTPmtdDirectWrongAssocTrackOutermostHitR3_->Fill(outermostHitPosition[trackref]);
                      meBTLTrackMatchedTPmtdDirectWrongAssocTrackNdf3_->Fill(trackGen.ndof());
                      meBTLTrackMatchedTPmtdDirectWrongAssocTrackChi23_->Fill(trackGen.chi2());
                      meBTLTrackMatchedTPmtdDirectWrongAssocDeltaZOutR3_->Fill(outermostHitPosition[trackref], simClusterRef_RecoMatch_DeltaZ);
                      meBTLTrackMatchedTPmtdDirectWrongAssocTimeChi23_->Fill(btlMatchTimeChi2[trackref]);
                      meBTLTrackMatchedTPmtdDirectWrongAssocTimeChi2vsMVAQual3_->Fill(btlMatchTimeChi2[trackref],  mtdQualMVA[trackref]);
                      meBTLTrackMatchedTPmtdDirectWrongAssocSpaceChi23_->Fill(btlMatchChi2[trackref]);
                      meBTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenghtvsEta3_->Fill(std::abs(trackGen.eta()), pathLength[trackref]);
                      meBTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenght3_->Fill(pathLength[trackref]);
		    }

                  }
		    fillTrackClusterMatchingHistograms(meBTLTrackMatchedTPmtdDirectWrongAssocEta_,
                                                     meBTLTrackMatchedTPmtdDirectWrongAssocPt_,
                                                     meBTLTrackMatchedTPmtdDirectWrongAssocMVAQual_,
                                                     meBTLTrackMatchedTPmtdDirectWrongAssocTimeRes_,
                                                     meBTLTrackMatchedTPmtdDirectWrongAssocTimePull_,
                                                     std::abs(trackGen.eta()),
                                                     trackGen.pt(),
                                                     mtdQualMVA[trackref],
                                                     dT,
                                                     pullT,
                                                     hasTime);

                }
              }
		
              // -- Track matched to TP with sim hit (other), correctly associated reco cluster
              else if (isTPmtdOtherBTL) {
                if (isTPmtdOtherCorrectBTL) {
                  fillTrackClusterMatchingHistograms(meBTLTrackMatchedTPmtdOtherCorrectAssocEta_,
                                                     meBTLTrackMatchedTPmtdOtherCorrectAssocPt_,
                                                     meBTLTrackMatchedTPmtdOtherCorrectAssocMVAQual_,
                                                     meBTLTrackMatchedTPmtdOtherCorrectAssocTimeRes_,
                                                     meBTLTrackMatchedTPmtdOtherCorrectAssocTimePull_,
                                                     std::abs(trackGen.eta()),
                                                     trackGen.pt(),
                                                     mtdQualMVA[trackref],
                                                     dT,
                                                     pullT,
                                                     hasTime);
                }
                // -- Track matched to TP with sim hit (other), incorrectly associated reco cluster
                else {
                  fillTrackClusterMatchingHistograms(meBTLTrackMatchedTPmtdOtherWrongAssocEta_,
                                                     meBTLTrackMatchedTPmtdOtherWrongAssocPt_,
                                                     meBTLTrackMatchedTPmtdOtherWrongAssocMVAQual_,
                                                     meBTLTrackMatchedTPmtdOtherWrongAssocTimeRes_,
                                                     meBTLTrackMatchedTPmtdOtherWrongAssocTimePull_,
                                                     std::abs(trackGen.eta()),
                                                     trackGen.pt(),
                                                     mtdQualMVA[trackref],
                                                     dT,
                                                     pullT,
                                                     hasTime);
                }
              }
            }
            // -- Track matched to TP with sim hit in MTD, missing associated reco cluster
            else {
              if (isTPmtdDirectBTL) {
		      
                if (optionalPlots_) {
		  meBTLTrackMatchedTPmtdDirectNoAssocSimClusSize_->Fill(simClusSize);
                  meBTLTrackMatchedTPmtdDirectNoAssocRecoClusSize_->Fill(recoClusSize);
	          meBTLTrackMatchedTPmtdDirectNoAssocTrackOutermostHitR_->Fill(outermostHitPosition[trackref]);
                  meBTLTrackMatchedTPmtdDirectNoAssocTrackNdf_->Fill(trackGen.ndof());
                  meBTLTrackMatchedTPmtdDirectNoAssocTrackChi2_->Fill(trackGen.chi2());
	        }

                meBTLTrackMatchedTPmtdDirectNoAssocEta_->Fill(std::abs(trackGen.eta()));
                meBTLTrackMatchedTPmtdDirectNoAssocPt_->Fill(trackGen.pt());
              } else if (isTPmtdOtherBTL) {
                meBTLTrackMatchedTPmtdOtherNoAssocEta_->Fill(std::abs(trackGen.eta()));
                meBTLTrackMatchedTPmtdOtherNoAssocPt_->Fill(trackGen.pt());
              }
            }
          }  // == end BTL
          // == ETL
          else {
	    // -- Track matched to TP with reco hits (one or two) correctly matched
	    if ((ETLdisc1 && isTPmtdCorrectETLD1) || (ETLdisc2 && isTPmtdCorrectETLD2)) {
              meETLTrackMatchedTPEtaMtdCorrect_->Fill(std::abs(trackGen.eta()));
	      meETLTrackMatchedTPPtMtdCorrect_->Fill(trackGen.pt());
            }
	    // -- Track matched to TP with sim hit in one etl layer
            if (isTPmtdETLD1 || isTPmtdETLD2) {  // at least one hit (D1 or D2 or both)
              meETLTrackMatchedTPmtd1Eta_->Fill(std::abs(trackGen.eta()));
              meETLTrackMatchedTPmtd1Pt_->Fill(trackGen.pt());
            }
            // -- Track matched to TP with sim hits in both etl layers (D1 and D2)
            if (isTPmtdETLD1 && isTPmtdETLD2) {
              meETLTrackMatchedTPmtd2Eta_->Fill(std::abs(trackGen.eta()));
              meETLTrackMatchedTPmtd2Pt_->Fill(trackGen.pt());
            }
            if (isETL) {
              // -- Track matched to TP with sim hit in >=1 etl layer
              if (isTPmtdETLD1 || isTPmtdETLD2) {
                // - each hit is correctly associated to the track
                if ((isTPmtdETLD1 && !isTPmtdETLD2 && ETLdisc1 && isTPmtdCorrectETLD1) ||
                    (isTPmtdETLD2 && !isTPmtdETLD1 && ETLdisc2 && isTPmtdCorrectETLD2) ||
                    (isTPmtdETLD1 && isTPmtdETLD2 && ETLdisc1 && ETLdisc2 && isTPmtdCorrectETLD1 &&
                     isTPmtdCorrectETLD2)) {
	          if (optionalPlots_){
			  meETLTrackMatchedTPmtd1CorrectAssocTimeChi2_->Fill(etlMatchTimeChi2[trackref]);		
			  meETLTrackMatchedTPmtd1CorrectAssocTimeChi2vsMVAQual_->Fill(etlMatchTimeChi2[trackref], mtdQualMVA[trackref]);
		  }	  
                  fillTrackClusterMatchingHistograms(meETLTrackMatchedTPmtd1CorrectAssocEta_,
                                                     meETLTrackMatchedTPmtd1CorrectAssocPt_,
                                                     meETLTrackMatchedTPmtd1CorrectAssocMVAQual_,
                                                     meETLTrackMatchedTPmtd1CorrectAssocTimeRes_,
                                                     meETLTrackMatchedTPmtd1CorrectAssocTimePull_,
                                                     std::abs(trackGen.eta()),
                                                     trackGen.pt(),
                                                     mtdQualMVA[trackref],
                                                     dT,
                                                     pullT,
                                                     hasTime);
                }
                // - at least one reco hit is incorrectly associated or, if two sim hits, one reco hit is missing
                else if ((isTPmtdETLD1 && !isTPmtdCorrectETLD1) || (isTPmtdETLD2 && !isTPmtdCorrectETLD2)) {
		  	
	          if (optionalPlots_) {
			  meETLTrackMatchedTPmtd1WrongAssocTimeChi2_->Fill(etlMatchTimeChi2[trackref]);		
			  meETLTrackMatchedTPmtd1WrongAssocTimeChi2vsMVAQual_->Fill(etlMatchTimeChi2[trackref], mtdQualMVA[trackref]);
		  }	  

                  fillTrackClusterMatchingHistograms(meETLTrackMatchedTPmtd1WrongAssocEta_,
                                                     meETLTrackMatchedTPmtd1WrongAssocPt_,
                                                     meETLTrackMatchedTPmtd1WrongAssocMVAQual_,
                                                     meETLTrackMatchedTPmtd1WrongAssocTimeRes_,
                                                     meETLTrackMatchedTPmtd1WrongAssocTimePull_,
                                                     std::abs(trackGen.eta()),
                                                     trackGen.pt(),
                                                     mtdQualMVA[trackref],
                                                     dT,
                                                     pullT,
                                                     hasTime);
                }
              }
              // -- Track matched to TP with sim hits in both etl layers (D1 and D2)
              if (isTPmtdETLD1 && isTPmtdETLD2) {
                // - each hit correctly associated to the track
                if (ETLdisc1 && ETLdisc2 && isTPmtdCorrectETLD1 && isTPmtdCorrectETLD2) {
	          if (optionalPlots_){
			  meETLTrackMatchedTPmtd2CorrectAssocTimeChi2_->Fill(etlMatchTimeChi2[trackref]);		
			  meETLTrackMatchedTPmtd2CorrectAssocTimeChi2vsMVAQual_->Fill(etlMatchTimeChi2[trackref], mtdQualMVA[trackref]);
		  }
		  fillTrackClusterMatchingHistograms(meETLTrackMatchedTPmtd2CorrectAssocEta_,
                                                     meETLTrackMatchedTPmtd2CorrectAssocPt_,
                                                     meETLTrackMatchedTPmtd2CorrectAssocMVAQual_,
                                                     meETLTrackMatchedTPmtd2CorrectAssocTimeRes_,
                                                     meETLTrackMatchedTPmtd2CorrectAssocTimePull_,
                                                     std::abs(trackGen.eta()),
                                                     trackGen.pt(),
                                                     mtdQualMVA[trackref],
                                                     dT,
                                                     pullT,
                                                     hasTime);
                }
                // - at least one reco hit incorrectly associated or one hit missing
                else if ((ETLdisc1 || ETLdisc2) && (!isTPmtdCorrectETLD1 || !isTPmtdCorrectETLD2)) {
	          if (optionalPlots_){ 
			  meETLTrackMatchedTPmtd2WrongAssocTimeChi2_->Fill(etlMatchTimeChi2[trackref]);		
			  meETLTrackMatchedTPmtd2WrongAssocTimeChi2vsMVAQual_->Fill(etlMatchTimeChi2[trackref], mtdQualMVA[trackref]);	
		  }	  
                  fillTrackClusterMatchingHistograms(meETLTrackMatchedTPmtd2WrongAssocEta_,
                                                     meETLTrackMatchedTPmtd2WrongAssocPt_,
                                                     meETLTrackMatchedTPmtd2WrongAssocMVAQual_,
                                                     meETLTrackMatchedTPmtd2WrongAssocTimeRes_,
                                                     meETLTrackMatchedTPmtd2WrongAssocTimePull_,
                                                     std::abs(trackGen.eta()),
                                                     trackGen.pt(),
                                                     mtdQualMVA[trackref],
                                                     dT,
                                                     pullT,
                                                     hasTime);
                }
              }
            }
            // -- Missing association with reco hits in MTD
            else {
              // -- Track matched to TP with sim hit in >=1 etl layers, no reco hits associated to the track
              if (isTPmtdETLD1 || isTPmtdETLD2) {
                meETLTrackMatchedTPmtd1NoAssocEta_->Fill(std::abs(trackGen.eta()));
                meETLTrackMatchedTPmtd1NoAssocPt_->Fill(trackGen.pt());
              }
              // -- Track matched to TP with sim hit in 2 etl layers, no reco hits associated to the track
              if (isTPmtdETLD1 && isTPmtdETLD2) {
                meETLTrackMatchedTPmtd2NoAssocEta_->Fill(std::abs(trackGen.eta()));
                meETLTrackMatchedTPmtd2NoAssocPt_->Fill(trackGen.pt());
              }
            }
          }  // == end ETL
        }  // --- end "withMTD"

        // - Track matched to TP without sim hit in MTD, but with reco cluster associated
        // - BTL
        if (std::abs(trackGen.eta()) < trackMaxBtlEta_) {
          if (!isTPmtdDirectBTL && !isTPmtdOtherBTL) {
            meBTLTrackMatchedTPnomtdEta_->Fill(std::abs(trackGen.eta()));
            meBTLTrackMatchedTPnomtdPt_->Fill(trackGen.pt());
            if (isBTL) {
		    
              if (optionalPlots_) {
	        for (const auto& recClusterRef : recoClustersRefs) { // having a look at these recos
                  if (recClusterRef.isNonnull()) {
                    auto itp = r2sAssociationMap.equal_range(recClusterRef);
                    if (itp.first != itp.second) {
                      auto& simClustersRefs_RecoMatch = (*itp.first).second;
	              simClusSize = simClustersRefs_RecoMatch.size();
                      for (const auto& sc : simClustersRefs_RecoMatch) {
                        auto mytps = Sim2tpAssociationMap.find(sc);
		        if (mytps != Sim2tpAssociationMap.end()){
		          for (const auto& mytp : mytps->val) {
		             if (((**tp_info).eventId().rawId() - (*mytp).eventId().rawId()) == 0 ) meBTLTrackMatchedTPnomtdAssocTrackID_->Fill(0); 
		             else meBTLTrackMatchedTPnomtdAssocTrackID_->Fill(1);
		          }
		        }
	                  meBTLTrackMatchedTPnomtdAssocTrackIdOff_->Fill(sc->trackIdOffset());

	                  }
	              }
	            }
	          }

	    
		  meBTLTrackMatchedTPnomtdAssocSimClusSize_->Fill(simClusSize);
		  meBTLTrackMatchedTPnomtdAssocRecoClusSize_->Fill(recoClusSize);
                  meBTLTrackMatchedTPnomtdAssocTrackChi2_ -> Fill(trackGen.chi2());
                  meBTLTrackMatchedTPnomtdAssocTrackNdf_ -> Fill(trackGen.ndof());		   
	          meBTLTrackMatchedTPnomtdAssocTrackOutermostHitR_->Fill(outermostHitPosition[trackref]) ; 
	      }
              fillTrackClusterMatchingHistograms(meBTLTrackMatchedTPnomtdAssocEta_,
                                                 meBTLTrackMatchedTPnomtdAssocPt_,
                                                 meBTLTrackMatchedTPnomtdAssocMVAQual_,
                                                 meBTLTrackMatchedTPnomtdAssocTimeRes_,
                                                 meBTLTrackMatchedTPnomtdAssocTimePull_,
                                                 std::abs(trackGen.eta()),
                                                 trackGen.pt(),
                                                 mtdQualMVA[trackref],
                                                 dT,
                                                 pullT,
                                                 hasTime);
            }
          }
        }
        // - ETL
        else if (!isTPmtdETLD1 && !isTPmtdETLD2) {
          meETLTrackMatchedTPnomtdEta_->Fill(std::abs(trackGen.eta()));
          meETLTrackMatchedTPnomtdPt_->Fill(trackGen.pt());
          if (isETL) {
            fillTrackClusterMatchingHistograms(meETLTrackMatchedTPnomtdAssocEta_,
                                               meETLTrackMatchedTPnomtdAssocPt_,
                                               meETLTrackMatchedTPnomtdAssocMVAQual_,
                                               meETLTrackMatchedTPnomtdAssocTimeRes_,
                                               meETLTrackMatchedTPnomtdAssocTimePull_,
                                               std::abs(trackGen.eta()),
                                               trackGen.pt(),
                                               mtdQualMVA[trackref],
                                               dT,
                                               pullT,
                                               hasTime);
          }
        }

        // == Time pull and detailed extrapolation check only on tracks associated to TP from signal event
        if (!trkTPSelLV(**tp_info)) {
          continue;
        }
        size_t nlayers(0);
        float extrho(0.);
        float exteta(0.);
        float extphi(0.);
        float selvar(0.);
        auto accept = checkAcceptance(trackGen, iEvent, iSetup, nlayers, extrho, exteta, extphi, selvar);
        if (accept.first && std::abs(exteta) < trackMaxBtlEta_) {
          meExtraPhiAtBTL_->Fill(angle_units::operators::convertRadToDeg(extphi));
          meExtraBTLeneInCone_->Fill(selvar);
        }
        if (accept.second) {
          if (std::abs(exteta) < trackMaxBtlEta_) {
            meExtraPhiAtBTLmatched_->Fill(angle_units::operators::convertRadToDeg(extphi));
          }
          if (noCrack) {
            meExtraPtMtd_->Fill(trackGen.pt());
            if (nlayers == 2) {
              meExtraPtEtl2Mtd_->Fill(trackGen.pt());
            }
          }
          meExtraEtaMtd_->Fill(std::abs(trackGen.eta()));
          if (nlayers == 2) {
            meExtraEtaEtl2Mtd_->Fill(std::abs(trackGen.eta()));
          }
          if (accept.first && accept.second && !(isBTL || isETL)) {
            edm::LogInfo("MtdTracksValidation")
                << "MtdTracksValidation: extender fail in " << iEvent.id().run() << " " << iEvent.id().event()
                << " pt= " << trackGen.pt() << " eta= " << trackGen.eta();
            meExtraMTDfailExtenderEta_->Fill(std::abs(trackGen.eta()));
            if (noCrack) {
              meExtraMTDfailExtenderPt_->Fill(trackGen.pt());
            }
          }
        }  // detailed extrapolation check

        // time res and time pull
        if (Sigmat0Safe[trackref] != -1.) {
          if (isBTL || isETL) {
            meTrackResTot_->Fill(dT);
            meTrackPullTot_->Fill(pullT);
            meTrackResTotvsMVAQual_->Fill(mtdQualMVA[trackref], dT);
            meTrackPullTotvsMVAQual_->Fill(mtdQualMVA[trackref], pullT);
          }
        }  // time res and time pull
      }  // TP matching
    }  // trkRecSel

    // ETL tracks with low pt (0.2 < Pt [GeV] < 0.7)
    if (trkRecSelLowPt(trackGen)) {
      if ((std::abs(trackGen.eta()) > trackMinEtlEta_) && (std::abs(trackGen.eta()) < trackMaxEtlEta_)) {
        if (trackGen.pt() < 0.45) {
          meETLTrackEtaTotLowPt_[0]->Fill(std::abs(trackGen.eta()));
        } else {
          meETLTrackEtaTotLowPt_[1]->Fill(std::abs(trackGen.eta()));
        }
      }
      bool MTDEtlZnegD1 = false;
      bool MTDEtlZnegD2 = false;
      bool MTDEtlZposD1 = false;
      bool MTDEtlZposD2 = false;
      for (const auto hit : track.recHits()) {
        if (hit->isValid() == false)
          continue;
        MTDDetId Hit = hit->geographicalId();
        if ((Hit.det() == 6) && (Hit.subdetId() == 1) && (Hit.mtdSubDetector() == 2)) {
          isETL = true;
          ETLDetId ETLHit = hit->geographicalId();
          if ((ETLHit.zside() == -1) && (ETLHit.nDisc() == 1)) {
            MTDEtlZnegD1 = true;
          }
          if ((ETLHit.zside() == -1) && (ETLHit.nDisc() == 2)) {
            MTDEtlZnegD2 = true;
          }
          if ((ETLHit.zside() == 1) && (ETLHit.nDisc() == 1)) {
            MTDEtlZposD1 = true;
          }
          if ((ETLHit.zside() == 1) && (ETLHit.nDisc() == 2)) {
            MTDEtlZposD2 = true;
          }
        }
      }
      if ((trackGen.eta() < -trackMinEtlEta_) && (trackGen.eta() > -trackMaxEtlEta_)) {
        twoETLdiscs = (MTDEtlZnegD1 == true) && (MTDEtlZnegD2 == true);
      }
      if ((trackGen.eta() > trackMinEtlEta_) && (trackGen.eta() < trackMaxEtlEta_)) {
        twoETLdiscs = (MTDEtlZposD1 == true) && (MTDEtlZposD2 == true);
      }
      if (isETL && (std::abs(trackGen.eta()) > trackMinEtlEta_) && (std::abs(trackGen.eta()) < trackMaxEtlEta_)) {
        if (trackGen.pt() < 0.45) {
          meETLTrackEtaMtdLowPt_[0]->Fill(std::abs(trackGen.eta()));
        } else {
          meETLTrackEtaMtdLowPt_[1]->Fill(std::abs(trackGen.eta()));
        }
      }
      if (isETL && twoETLdiscs) {
        if (trackGen.pt() < 0.45) {
          meETLTrackEta2MtdLowPt_[0]->Fill(std::abs(trackGen.eta()));
        } else {
          meETLTrackEta2MtdLowPt_[1]->Fill(std::abs(trackGen.eta()));
        }
      }
    }  // trkRecSelLowPt

  }  // RECO tracks loop
}

const std::pair<bool, bool> MtdTracksValidation::checkAcceptance(const reco::Track& track,
                                                                 const edm::Event& iEvent,
                                                                 edm::EventSetup const& iSetup,
                                                                 size_t& nlayers,
                                                                 float& extrho,
                                                                 float& exteta,
                                                                 float& extphi,
                                                                 float& selvar) {
  bool isMatched(false);
  nlayers = 0;
  extrho = 0.;
  exteta = -999.;
  extphi = -999.;
  selvar = 0.;

  auto geometryHandle = iSetup.getTransientHandle(mtdgeoToken_);
  const MTDGeometry* geom = geometryHandle.product();
  auto topologyHandle = iSetup.getTransientHandle(mtdtopoToken_);
  const MTDTopology* topology = topologyHandle.product();

  auto layerHandle = iSetup.getTransientHandle(mtdlayerToken_);
  const MTDDetLayerGeometry* layerGeo = layerHandle.product();

  auto magfieldHandle = iSetup.getTransientHandle(magfieldToken_);
  const MagneticField* mfield = magfieldHandle.product();

  auto ttrackBuilder = iSetup.getTransientHandle(builderToken_);

  auto tTrack = ttrackBuilder->build(track);
  TrajectoryStateOnSurface tsos = tTrack.outermostMeasurementState();
  float theMaxChi2 = 500.;
  float theNSigma = 10.;
  std::unique_ptr<MeasurementEstimator> theEstimator =
      std::make_unique<Chi2MeasurementEstimator>(theMaxChi2, theNSigma);
  SteppingHelixPropagator prop(mfield, anyDirection);

  auto btlRecHitsHandle = makeValid(iEvent.getHandle(btlRecHitsToken_));
  auto etlRecHitsHandle = makeValid(iEvent.getHandle(etlRecHitsToken_));

  edm::LogVerbatim("MtdTracksValidation")
      << "MtdTracksValidation: extrapolating track, pt= " << track.pt() << " eta= " << track.eta();

  //try BTL
  bool inBTL = false;
  float eneSum(0.);
  const std::vector<const DetLayer*>& layersBTL = layerGeo->allBTLLayers();
  for (const DetLayer* ilay : layersBTL) {
    std::pair<bool, TrajectoryStateOnSurface> comp = ilay->compatible(tsos, prop, *theEstimator);
    if (!comp.first)
      continue;
    if (!inBTL) {
      inBTL = true;
      extrho = comp.second.globalPosition().perp();
      exteta = comp.second.globalPosition().eta();
      extphi = comp.second.globalPosition().phi();
      edm::LogVerbatim("MtdTracksValidation") << "MtdTracksValidation: extrapolation at BTL surface, rho= " << extrho
                                              << " eta= " << exteta << " phi= " << extphi;
    }
    std::vector<DetLayer::DetWithState> compDets = ilay->compatibleDets(tsos, prop, *theEstimator);
    for (const auto& detWithState : compDets) {
      const auto& det = detWithState.first;

      // loop on compatible rechits and check energy in a fixed size cone around the extrapolation point

      edm::LogVerbatim("MtdTracksValidation")
          << "MtdTracksValidation: DetId= " << det->geographicalId().rawId()
          << " gp= " << detWithState.second.globalPosition().x() << " " << detWithState.second.globalPosition().y()
          << " " << detWithState.second.globalPosition().z() << " rho= " << detWithState.second.globalPosition().perp()
          << " eta= " << detWithState.second.globalPosition().eta()
          << " phi= " << detWithState.second.globalPosition().phi();

      for (const auto& recHit : *btlRecHitsHandle) {
        BTLDetId detId = recHit.id();
        DetId geoId = detId.geographicalId(MTDTopologyMode::crysLayoutFromTopoMode(topology->getMTDTopologyMode()));
        const MTDGeomDet* thedet = geom->idToDet(geoId);
        if (thedet == nullptr)
          throw cms::Exception("MtdTracksValidation") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                                      << detId.rawId() << ") is invalid!" << std::dec << std::endl;
        if (geoId == det->geographicalId()) {
          const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
          const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

          Local3DPoint local_point(0., 0., 0.);
          local_point = topo.pixelToModuleLocalPoint(local_point, detId.row(topo.nrows()), detId.column(topo.nrows()));
          const auto& global_point = thedet->toGlobal(local_point);
          edm::LogVerbatim("MtdTracksValidation")
              << "MtdTracksValidation: Hit id= " << detId.rawId() << " ene= " << recHit.energy()
              << " dr= " << reco::deltaR(global_point, detWithState.second.globalPosition());
          if (reco::deltaR(global_point, detWithState.second.globalPosition()) < cluDRradius_) {
            eneSum += recHit.energy();
            //extrho = detWithState.second.globalPosition().perp();
            //exteta = detWithState.second.globalPosition().eta();
            //extphi = detWithState.second.globalPosition().phi();
          }
        }
      }
    }
    if (eneSum > depositBTLthreshold_) {
      nlayers++;
      selvar = eneSum;
      isMatched = true;
      edm::LogVerbatim("MtdTracksValidation")
          << "MtdTracksValidation: BTL matched, energy= " << eneSum << " #layers= " << nlayers;
    }
  }
  if (inBTL) {
    return std::make_pair(inBTL, isMatched);
  }

  //try ETL
  bool inETL = false;
  const std::vector<const DetLayer*>& layersETL = layerGeo->allETLLayers();
  for (const DetLayer* ilay : layersETL) {
    size_t hcount(0);
    const BoundDisk& disk = static_cast<const MTDSectorForwardDoubleLayer*>(ilay)->specificSurface();
    const double diskZ = disk.position().z();
    if (tsos.globalPosition().z() * diskZ < 0)
      continue;  // only propagate to the disk that's on the same side
    std::pair<bool, TrajectoryStateOnSurface> comp = ilay->compatible(tsos, prop, *theEstimator);
    if (!comp.first)
      continue;
    if (!inETL) {
      inETL = true;
      extrho = comp.second.globalPosition().perp();
      exteta = comp.second.globalPosition().eta();
      extphi = comp.second.globalPosition().phi();
    }
    edm::LogVerbatim("MtdTracksValidation") << "MtdTracksValidation: extrapolation at ETL surface, rho= " << extrho
                                            << " eta= " << exteta << " phi= " << extphi;
    std::vector<DetLayer::DetWithState> compDets = ilay->compatibleDets(tsos, prop, *theEstimator);
    for (const auto& detWithState : compDets) {
      const auto& det = detWithState.first;

      // loop on compatible rechits and check hits in a fixed size cone around the extrapolation point

      edm::LogVerbatim("MtdTracksValidation")
          << "MtdTracksValidation: DetId= " << det->geographicalId().rawId()
          << " gp= " << detWithState.second.globalPosition().x() << " " << detWithState.second.globalPosition().y()
          << " " << detWithState.second.globalPosition().z() << " rho= " << detWithState.second.globalPosition().perp()
          << " eta= " << detWithState.second.globalPosition().eta()
          << " phi= " << detWithState.second.globalPosition().phi();

      for (const auto& recHit : *etlRecHitsHandle) {
        ETLDetId detId = recHit.id();
        DetId geoId = detId.geographicalId();
        const MTDGeomDet* thedet = geom->idToDet(geoId);
        if (thedet == nullptr)
          throw cms::Exception("MtdTracksValidation") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                                      << detId.rawId() << ") is invalid!" << std::dec << std::endl;
        if (geoId == det->geographicalId()) {
          const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
          const RectangularMTDTopology& topo = static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

          Local3DPoint local_point(topo.localX(recHit.row()), topo.localY(recHit.column()), 0.);
          const auto& global_point = thedet->toGlobal(local_point);
          edm::LogVerbatim("MtdTracksValidation")
              << "MtdTracksValidation: Hit id= " << detId.rawId() << " time= " << recHit.time()
              << " dr= " << reco::deltaR(global_point, detWithState.second.globalPosition());
          if (reco::deltaR(global_point, detWithState.second.globalPosition()) < cluDRradius_) {
            hcount++;
            if (hcount == 1) {
              //extrho = detWithState.second.globalPosition().perp();
              //exteta = detWithState.second.globalPosition().eta();
              //extphi = detWithState.second.globalPosition().phi();
            }
          }
        }
      }
    }
    if (hcount > 0) {
      nlayers++;
      selvar = (float)hcount;
      isMatched = true;
      edm::LogVerbatim("MtdTracksValidation")
          << "MtdTracksValidation: ETL matched, counts= " << hcount << " #layers= " << nlayers;
    }
  }

  if (!inBTL && !inETL) {
    edm::LogVerbatim("MtdTracksValidation")
        << "MtdTracksValidation: track not extrapolating to MTD: pt= " << track.pt() << " eta= " << track.eta()
        << " phi= " << track.phi() << " vz= " << track.vz()
        << " vxy= " << std::sqrt(track.vx() * track.vx() + track.vy() * track.vy());
  }
  return std::make_pair(inETL, isMatched);
}

// ------------ method for histogram booking ------------
void MtdTracksValidation::bookHistograms(DQMStore::IBooker& ibook, edm::Run const& run, edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(folder_);

  // histogram booking
  meBTLTrackRPTime_ = ibook.book1D("TrackBTLRPTime", "Track t0 with respect to R.P.;t0 [ns]", 100, -1, 3);
  meBTLTrackEtaTot_ = ibook.book1D("TrackBTLEtaTot", "Eta of tracks (Tot);#eta_{RECO}", 30, 0., 1.5);
  meBTLTrackPhiTot_ = ibook.book1D("TrackBTLPhiTot", "Phi of tracks (Tot);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meBTLTrackPtTot_ = ibook.book1D("TrackBTLPtTot", "Pt of tracks (Tot);pt_{RECO} [GeV]", 50, 0, 10);
  meBTLTrackEtaMtd_ = ibook.book1D("TrackBTLEtaMtd", "Eta of tracks (Mtd);#eta_{RECO}", 30, 0., 1.5);
  meBTLTrackPhiMtd_ = ibook.book1D("TrackBTLPhiMtd", "Phi of tracks (Mtd);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meBTLTrackPtMtd_ = ibook.book1D("TrackBTLPtMtd", "Pt of tracks (Mtd);pt_{RECO} [GeV]", 50, 0, 10);
  meBTLTrackPtRes_ =
      ibook.book1D("TrackBTLPtRes", "Track pT resolution  ;pT_{Gentrack}-pT_{MTDtrack}/pT_{Gentrack} ", 100, -0.1, 0.1);
  meETLTrackRPTime_ = ibook.book1D("TrackETLRPTime", "Track t0 with respect to R.P.;t0 [ns]", 100, -1, 3);
  meETLTrackEtaTot_ = ibook.book1D("TrackETLEtaTot", "Eta of tracks (Tot);#eta_{RECO}", 30, 1.5, 3.0);
  meETLTrackPhiTot_ = ibook.book1D("TrackETLPhiTot", "Phi of tracks (Tot);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meETLTrackPhiTot_ = ibook.book1D("TrackETLPhiTot", "Phi of tracks (Tot);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meETLTrackPtTot_ = ibook.book1D("TrackETLPtTot", "Pt of tracks (Tot);pt_{RECO} [GeV]", 50, 0, 10);

  meETLTrackEtaTotLowPt_[0] =
      ibook.book1D("TrackETLEtaTotLowPt0", "Eta of tracks, 0.2 < pt < 0.45 (Tot);#eta_{RECO}", 30, 1.5, 3.0);
  meETLTrackEtaTotLowPt_[1] =
      ibook.book1D("TrackETLEtaTotLowPt1", "Eta of tracks, 0.45 < pt < 0.7 (Tot);#eta_{RECO}", 30, 1.5, 3.0);

  meETLTrackEtaMtd_ = ibook.book1D("TrackETLEtaMtd", "Eta of tracks (Mtd);#eta_{RECO}", 30, 1.5, 3.0);
  meETLTrackEtaMtdLowPt_[0] =
      ibook.book1D("TrackETLEtaMtdLowPt0", "Eta of tracks, 0.2 < pt < 0.45 (Mtd);#eta_{RECO}", 30, 1.5, 3.0);
  meETLTrackEtaMtdLowPt_[1] =
      ibook.book1D("TrackETLEtaMtdLowPt1", "Eta of tracks, 0.45 < pt < 0.7 (Mtd);#eta_{RECO}", 30, 1.5, 3.0);
  meETLTrackEta2MtdLowPt_[0] =
      ibook.book1D("TrackETLEta2MtdLowPt0", "Eta of tracks, 0.2 < pt < 0.45 (Mtd 2 hit);#eta_{RECO}", 30, 1.5, 3.0);
  meETLTrackEta2MtdLowPt_[1] =
      ibook.book1D("TrackETLEta2MtdLowPt1", "Eta of tracks, 0.45 < pt < 0.7 (Mtd 2 hit);#eta_{RECO}", 30, 1.5, 3.0);

  meETLTrackPhiMtd_ = ibook.book1D("TrackETLPhiMtd", "Phi of tracks (Mtd);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meETLTrackPtMtd_ = ibook.book1D("TrackETLPtMtd", "Pt of tracks (Mtd);pt_{RECO} [GeV]", 50, 0, 10);
  meETLTrackEta2Mtd_ = ibook.book1D("TrackETLEta2Mtd", "Eta of tracks (Mtd 2 hit);#eta_{RECO}", 30, 1.5, 3.0);
  meETLTrackPhi2Mtd_ = ibook.book1D("TrackETLPhi2Mtd", "Phi of tracks (Mtd 2 hit);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meETLTrackPt2Mtd_ = ibook.book1D("TrackETLPt2Mtd", "Pt of tracks (Mtd 2 hit);pt_{RECO} [GeV]", 50, 0, 10);

  meETLTrackPtRes_ =
      ibook.book1D("TrackETLPtRes", "Track pT resolution;pT_{Gentrack}-pT_{MTDtrack}/pT_{Gentrack} ", 100, -0.1, 0.1);

  meTracktmtd_ = ibook.book1D("Tracktmtd", "Track time from TrackExtenderWithMTD;tmtd [ns]", 150, 1, 16);
  meTrackt0Src_ = ibook.book1D("Trackt0Src", "Track time from TrackExtenderWithMTD;t0Src [ns]", 100, -1.5, 1.5);
  meTrackSigmat0Src_ =
      ibook.book1D("TrackSigmat0Src", "Time Error from TrackExtenderWithMTD; #sigma_{t0Src} [ns]", 100, 0, 0.1);

  meTrackt0Pid_ = ibook.book1D("Trackt0Pid", "Track t0 as stored in TofPid;t0 [ns]", 100, -1, 1);
  meTrackSigmat0Pid_ = ibook.book1D("TrackSigmat0Pid", "Sigmat0 as stored in TofPid; #sigma_{t0} [ns]", 100, 0, 0.1);
  meTrackt0SafePid_ = ibook.book1D("Trackt0SafePID", "Track t0 Safe as stored in TofPid;t0 [ns]", 100, -1, 1);
  meTrackSigmat0SafePid_ = ibook.book1D(
      "TrackSigmat0SafePID", "Log10(Sigmat0 Safe) as stored in TofPid; Log10(#sigma_{t0} [ns])", 80, -3, 1);
  meTrackNumHits_ = ibook.book1D("TrackNumHits", "Number of valid MTD hits per track ; Number of hits", 10, -5, 5);
  meTrackNumHitsNT_ = ibook.book1D(
      "TrackNumHitsNT", "Number of valid MTD hits per track no time associated; Number of hits", 10, -5, 5);
  meTrackMVAQual_ = ibook.book1D("TrackMVAQual", "Track MVA Quality as stored in Value Map ; MVAQual", 100, -1, 1);
  meTrackPathLenghtvsEta_ = ibook.bookProfile(
      "TrackPathLenghtvsEta", "MTD Track pathlength vs MTD track Eta;|#eta|;Pathlength", 100, 0, 3.2, 100.0, 400.0, "S");

  meTrackOutermostHitR_ = ibook.book1D("TrackOutermostHitR", "Track outermost hit position R; R[cm]", 40, 0, 120.);
  meTrackOutermostHitZ_ = ibook.book1D("TrackOutermostHitZ", "Track outermost hit position Z; z[cm]", 100, 0, 300.);

  meTrackSigmaTof_[0] =
      ibook.book1D("TrackSigmaTof_Pion", "Sigma(TOF) for pion hypothesis; #sigma_{t0} [ps]", 10, 0, 5);
  meTrackSigmaTof_[1] =
      ibook.book1D("TrackSigmaTof_Kaon", "Sigma(TOF) for kaon hypothesis; #sigma_{t0} [ps]", 25, 0, 25);
  meTrackSigmaTof_[2] =
      ibook.book1D("TrackSigmaTof_Proton", "Sigma(TOF) for proton hypothesis; #sigma_{t0} [ps]", 50, 0, 50);

  meTrackSigmaTofvsP_[0] = ibook.bookProfile("TrackSigmaTofvsP_Pion",
                                             "Sigma(TOF) for pion hypothesis vs p; p [GeV]; #sigma_{t0} [ps]",
                                             20,
                                             0,
                                             10.,
                                             0,
                                             50.,
                                             "S");
  meTrackSigmaTofvsP_[1] = ibook.bookProfile("TrackSigmaTofvsP_Kaon",
                                             "Sigma(TOF) for kaon hypothesis vs p; p [GeV]; #sigma_{t0} [ps]",
                                             20,
                                             0,
                                             10.,
                                             0,
                                             50.,
                                             "S");
  meTrackSigmaTofvsP_[2] = ibook.bookProfile("TrackSigmaTofvsP_Proton",
                                             "Sigma(TOF) for proton hypothesis vs p; p [GeV]; #sigma_{t0} [ps]",
                                             20,
                                             0,
                                             10.,
                                             0,
                                             50.,
                                             "S");

  meExtraPtMtd_ =
      ibook.book1D("ExtraPtMtd", "Pt of tracks associated to LV extrapolated to hits; track pt [GeV] ", 110, 0., 11.);
  meExtraPtEtl2Mtd_ = ibook.book1D("ExtraPtEtl2Mtd",
                                   "Pt of tracks associated to LV extrapolated to hits, 2 ETL layers; track pt [GeV] ",
                                   110,
                                   0.,
                                   11.);
  meExtraEtaMtd_ =
      ibook.book1D("ExtraEtaMtd", "Eta of tracks associated to LV extrapolated to hits; track eta ", 66, 0., 3.3);
  meExtraEtaEtl2Mtd_ = ibook.book1D(
      "ExtraEtaEtl2Mtd", "Eta of tracks associated to LV extrapolated to hits, 2 ETL layers; track eta ", 66, 0., 3.3);
  meTrackMatchedTPEtaTotLV_ =
      ibook.book1D("MatchedTPEtaTotLV", "Eta of tracks associated to LV matched to TP; track eta ", 66, 0., 3.3);
  meTrackMatchedTPPtTotLV_ =
      ibook.book1D("MatchedTPPtTotLV", "Pt of tracks associated to LV matched to TP; track pt [GeV] ", 110, 0., 11.);

  meBTLTrackMatchedTPEtaTot_ =
      ibook.book1D("BTLTrackMatchedTPEtaTot", "Eta of tracks matched to TP; track eta ", 30, 0., 1.5);
  meBTLTrackMatchedTPEtaMtd_ =
      ibook.book1D("BTLTrackMatchedTPEtaMtd", "Eta of tracks matched to TP with time; track eta ", 30, 0., 1.5);
  meBTLTrackMatchedTPPtTot_ =
      ibook.book1D("BTLTrackMatchedTPPtTot", "Pt of tracks matched to TP; track pt [GeV] ", 50, 0., 10.);
  meBTLTrackMatchedTPPtMtd_ =
      ibook.book1D("BTLTrackMatchedTPPtMtd", "Pt of tracks matched to TP with time; track pt [GeV] ", 50, 0., 10.);
  meETLTrackMatchedTPEtaTot_ =
      ibook.book1D("ETLTrackMatchedTPEtaTot", "Eta of tracks matched to TP; track eta ", 30, 1.5, 3.0);
  meETLTrackMatchedTPEtaMtd_ = ibook.book1D(
      "ETLTrackMatchedTPEtaMtd", "Eta of tracks matched to TP with time (>=1 ETL hit); track eta ", 30, 1.5, 3.0);
   meETLTrackMatchedTPEtaMtdCorrect_ =
	   ibook.book1D("ETLTrackMatchedTPEtaMtdCorrect",
                         "Eta of tracks matched to TP with time (>=1 ETL hit), correct reco match; track eta ",
			 30,
			 1.5,
			 3.0);
  meETLTrackMatchedTPEta2Mtd_ = ibook.book1D(
      "ETLTrackMatchedTPEta2Mtd", "Eta of tracks matched to TP with time (2 ETL hits); track eta ", 30, 1.5, 3.0);
  meETLTrackMatchedTPPtTot_ =
      ibook.book1D("ETLTrackMatchedTPPtTot", "Pt of tracks matched to TP; track pt [GeV] ", 50, 0., 10.);
  meETLTrackMatchedTPPtMtd_ = ibook.book1D(
      "ETLTrackMatchedTPPtMtd", "Pt of tracks matched to TP with time (>=1 ETL hit); track pt [GeV] ", 50, 0., 10.);
  meETLTrackMatchedTPPtMtdCorrect_ =
	  ibook.book1D("ETLTrackMatchedTPPtMtdCorrect",
			   "Pt of tracks matched to TP with time (>=1 ETL hit), correct reco match; track pt [GeV] ",
			    50,
			    0.,
			    10.);
  meETLTrackMatchedTPPt2Mtd_ = ibook.book1D(
      "ETLTrackMatchedTPPt2Mtd", "Pt of tracks matched to TP with time (2 ETL hits); track pt [GeV] ", 50, 0., 10.);

  if (optionalPlots_) {
    meBTLTrackMatchedTPPtResMtd_ = ibook.book1D(
        "TrackMatchedTPBTLPtResMtd",
        "Pt resolution of tracks matched to TP-BTL hit  ;|pT_{MTDtrack}-pT_{truth}|/|pT_{Gentrack}-pT_{truth}| ",
        100,
        0.,
        4.);
    meETLTrackMatchedTPPtResMtd_ = ibook.book1D(
        "TrackMatchedTPETLPtResMtd",
        "Pt resolution of tracks matched to TP-ETL hit  ;|pT_{MTDtrack}-pT_{truth}|/|pT_{Gentrack}-pT_{truth}| ",
        100,
        0.,
        4.);
    meETLTrackMatchedTP2PtResMtd_ = ibook.book1D(
        "TrackMatchedTPETL2PtResMtd",
        "Pt resolution of tracks matched to TP-ETL 2hits  ;|pT_{MTDtrack}-pT_{truth}|/|pT_{Gentrack}-pT_{truth}| ",
        100,
        0.,
        4.);
    meBTLTrackMatchedTPPtRatioGen_ = ibook.book1D(
        "TrackMatchedTPBTLPtRatioGen", "Pt ratio of Gentracks (BTL)  ;pT_{Gentrack}/pT_{truth} ", 100, 0.9, 1.1);
    meETLTrackMatchedTPPtRatioGen_ = ibook.book1D(
        "TrackMatchedTPETLPtRatioGen", "Pt ratio of Gentracks (ETL 1hit)  ;pT_{Gentrack}/pT_{truth} ", 100, 0.9, 1.1);
    meETLTrackMatchedTP2PtRatioGen_ = ibook.book1D(
        "TrackMatchedTPETL2PtRatioGen", "Pt ratio of Gentracks (ETL 2hits)  ;pT_{Gentrack}/pT_{truth} ", 100, 0.9, 1.1);
    meBTLTrackMatchedTPPtRatioMtd_ =
        ibook.book1D("TrackMatchedTPBTLPtRatioMtd",
                     "Pt ratio of tracks matched to TP-BTL hit  ;pT_{MTDtrack}/pT_{truth} ",
                     100,
                     0.9,
                     1.1);
    meETLTrackMatchedTPPtRatioMtd_ =
        ibook.book1D("TrackMatchedTPETLPtRatioMtd",
                     "Pt ratio of tracks matched to TP-ETL hit  ;pT_{MTDtrack}/pT_{truth} ",
                     100,
                     0.9,
                     1.1);
    meETLTrackMatchedTP2PtRatioMtd_ =
        ibook.book1D("TrackMatchedTPETL2PtRatioMtd",
                     "Pt ratio of tracks matched to TP-ETL 2hits  ;pT_{MTDtrack}/pT_{truth} ",
                     100,
                     0.9,
                     1.1);
    meBTLTrackMatchedTPPtResvsPtMtd_ =
        ibook.bookProfile("TrackMatchedTPBTLPtResvsPtMtd",
                          "Pt resolution of tracks matched to TP-BTL hit vs Pt;pT_{truth} "
                          "[GeV];|pT_{MTDtrack}-pT_{truth}|/|pT_{Gentrack}-pT_{truth}| ",
                          20,
                          0.7,
                          10.,
                          0.,
                          4.,
                          "s");
    meETLTrackMatchedTPPtResvsPtMtd_ =
        ibook.bookProfile("TrackMatchedTPETLPtResvsPtMtd",
                          "Pt resolution of tracks matched to TP-ETL hit vs Pt;pT_{truth} "
                          "[GeV];|pT_{MTDtrack}-pT_{truth}|/|pT_{Gentrack}-pT_{truth}| ",
                          20,
                          0.7,
                          10.,
                          0.,
                          4.,
                          "s");
    meETLTrackMatchedTP2PtResvsPtMtd_ =
        ibook.bookProfile("TrackMatchedTPETL2PtResvsPtMtd",
                          "Pt resolution of tracks matched to TP-ETL 2hits Pt pT;pT_{truth} "
                          "[GeV];|pT_{MTDtrack}-pT_{truth}|/|pT_{Gentrack}-pT_{truth}| ",
                          20,
                          0.7,
                          10.,
                          0.,
                          4.,
                          "s");
    meBTLTrackMatchedTPDPtvsPtGen_ = ibook.bookProfile(
        "TrackMatchedTPBTLDPtvsPtGen",
        "Pt relative difference of Gentracks (BTL) vs Pt;pT_{truth} [GeV];pT_{Gentrack}-pT_{truth}/pT_{truth} ",
        20,
        0.7,
        10.,
        -0.1,
        0.1,
        "s");
    meETLTrackMatchedTPDPtvsPtGen_ = ibook.bookProfile(
        "TrackMatchedTPETLDPtvsPtGen",
        "Pt relative difference of Gentracks (ETL 1hit) vs Pt;pT_{truth} [GeV];pT_{Gentrack}-pT_{truth}/pT_{truth} ",
        20,
        0.7,
        10.,
        -0.1,
        0.1,
        "s");
    meETLTrackMatchedTP2DPtvsPtGen_ = ibook.bookProfile(
        "TrackMatchedTPETL2DPtvsPtGen",
        "Pt relative difference  of Gentracks (ETL 2hits) vs Pt;pT_{truth} [GeV];pT_{Gentrack}-pT_{truth}/pT_{truth} ",
        20,
        0.7,
        10.,
        -0.1,
        0.1,
        "s");
    meBTLTrackMatchedTPDPtvsPtMtd_ = ibook.bookProfile("TrackMatchedTPBTLDPtvsPtMtd",
                                                       "Pt relative difference of tracks matched to TP-BTL hits vs "
                                                       "Pt;pT_{truth} [GeV];pT_{MTDtrack}-pT_{truth}/pT_{truth} ",
                                                       20,
                                                       0.7,
                                                       10.,
                                                       -0.1,
                                                       0.1,
                                                       "s");
    meETLTrackMatchedTPDPtvsPtMtd_ = ibook.bookProfile("TrackMatchedTPETLDPtvsPtMtd",
                                                       "Pt relative difference of tracks matched to TP-ETL hits vs "
                                                       "Pt;pT_{truth} [GeV];pT_{MTDtrack}-pT_{truth}/pT_{truth} ",
                                                       20,
                                                       0.7,
                                                       10.,
                                                       -0.1,
                                                       0.1,
                                                       "s");
    meETLTrackMatchedTP2DPtvsPtMtd_ = ibook.bookProfile("TrackMatchedTPETL2DPtvsPtMtd",
                                                        "Pt relative difference of tracks matched to TP-ETL 2hits vs "
                                                        "Pt;pT_{truth} [GeV];pT_{MTDtrack}-pT_{truth}/pT_{truth} ",
                                                        20,
                                                        0.7,
                                                        10.,
                                                        -0.1,
                                                        0.1,
                                                        "s");

    meBTLTrackMatchedTPmtdDirectCorrectAssocTrackNdf_ = ibook.book1D(
       "BTLTrackMatchedTPmtdDirectCorrectAssocTrackNdf",
       "Ndf of tracks matched to TP with sim hit in MTD (direct), correct track-MTD association; Ndof ",
       80,
       0.,
       220);
    meBTLTrackMatchedTPmtdDirectCorrectAssocTrackChi2_ = ibook.book1D(
       "BTLTrackMatchedTPmtdDirectCorrectAssocTrackChi2",
       "Chi2 of tracks matched to TP with sim hit in MTD (direct), correct track-MTD association; Chi2 ",
       80,
       0.,
       220);
    meBTLTrackMatchedTPmtdDirectCorrectAssocTrackOutermostHitR_ = ibook.book1D(
       "BTLTrackMatchedTPmtdDirectCorrectAssocTrackOutermostHitR",
       "Outermost hit position R of tracks matched to TP with sim hit in MTD (direct), correct track-MTD association; R [cm] ",
       40,
       0.,
       120);
    meBTLTrackMatchedTPmtdDirectCorrectAssocSimClusSize_ =  ibook.book1D(
         "BTLTrackMatchedTPmtdDirectCorrectAssocSimClusSize",
         "Size of the sim clusters associated to the reco cluster associated to the track (direct), correct track-MTD association; Number of clusters ",
         10,
         -5.,
         5.);
    meBTLTrackMatchedTPmtdDirectCorrectAssocRecoClusSize_ =  ibook.book1D(
         "BTLTrackMatchedTPmtdDirectCorrectAssocRecoClusSize",
         "Size of the reco cluster associated to the track (direct), correct track-MTD association; Number of clusters ",
         10,
         -5.,
         5.);
    meBTLTrackMatchedTPmtdDirectCorrectAssocTimeChi2_ = ibook.book1D(
       "BTLTrackMatchedTPmtdDirectCorrectAssocTimeChi2",
       "Time chi2 of tracks matched to TP with sim hit in MTD (direct), correct track-MTD association; Chi2 ",
       200,
       0.,
       100);
     meBTLTrackMatchedTPmtdDirectCorrectAssocTimeChi2vsMVAQual_ = ibook.book2D(
       "BTLTrackMatchedTPmtdDirectCorrectAssocTimeChi2vsMVAQual", "Time chi2 of tracks matched to TP with sim hit in MTD (direct), correct track-MTD association; Chi2; MVAQual ",
       200, 0., 100, 100, -1, 1);
   
    meBTLTrackMatchedTPmtdDirectCorrectAssocSpaceChi2_ = ibook.book1D(
       "BTLTrackMatchedTPmtdDirectCorrectAssocSpaceChi2",
       "Space chi2 of tracks matched to TP with sim hit in MTD (direct), correct track-MTD association; Chi2 ",
       250,
       0.,
       250.);
    meBTLTrackMatchedTPmtdDirectCorrectAssocTrackPathLenghtvsEta_ = ibook.book2D(
      "BTLTrackMatchedTPmtdDirectCorrectAssocTrackPathLenghtvsEta", "MTD Track pathlength vs MTD track Eta;|#eta|;Pathlength", 30, 0, 1.5, 350, 0.0, 350.0);
    meBTLTrackMatchedTPmtdDirectCorrectAssocTrackPathLenght_ = ibook.book1D(
      "BTLTrackMatchedTPmtdDirectCorrectAssocTrackPathLenght", "MTD Track pathlength ; ;Pathlength", 400, 0, 400);


    meBTLTrackMatchedTPmtdDirectWrongAssocTrackNdf1_ = ibook.book1D(
       "BTLTrackMatchedTPmtdDirectWrongAssocTrackNdf1",
       "Ndf of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; Ndof ",
       80,
       0.,
       220);
    meBTLTrackMatchedTPmtdDirectWrongAssocTrackChi21_ = ibook.book1D(
       "BTLTrackMatchedTPmtdDirectWrongAssocTrackChi21",
       "Chi2 of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; Chi2 ",
       80,
       0.,
       220);
    meBTLTrackMatchedTPmtdDirectWrongAssocTrackOutermostHitR1_ = ibook.book1D(
        "BTLTrackMatchedTPmtdDirectWrongAssocTrackOutermostHitR1",
        "Outermost hit position R of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; R [cm]",
        40,
        0.,
        120);
    meBTLTrackMatchedTPmtdDirectWrongAssocTrackIdOff1_ = ibook.book1D(
        "BTLTrackMatchedTPmtdDirectWrongAssocTrackIdOff1",
        "Track Id offset of reco (wrong) cluster associated to the track (direct)  - wrong track-MTD association; trackId (wrong)",
        6,
        -1.,
        5.);
    meBTLTrackMatchedTPmtdDirectWrongAssocDeltaZ1_ = ibook.book1D(
         "BTLTrackMatchedTPmtdDirectWrongAssocDeltaZ1",
         "Z of sim matched wrong cluster - Z of true sim cluster (direct) - wrong track-MTD association; DeltaZ (wrong - true) [cm]",
         1000,
         -50.,
         50.);
     meBTLTrackMatchedTPmtdDirectWrongAssocDeltaPhi1_ = ibook.book1D(
         "BTLTrackMatchedTPmtdDirectWrongAssocDeltaPhi1",
         "Phi of sim matched wrong cluster - Z of true sim cluster - wrong track-MTD association; DeltaPhi (wrong - true) ",
         500,
         -0.1,
         0.1);
    meBTLTrackMatchedTPmtdDirectWrongAssocDeltaT1_ = ibook.book1D(
         "BTLTrackMatchedTPmtdDirectWrongAssocDeltaT1",
         "Time of sim matched wrong cluster - time of true sim cluster (direct) - wrong track-MTD association; DeltaT (wrong - true) [ns]",
         480,
         -0.6,
         0.6);
    meBTLTrackMatchedTPmtdDirectWrongAssocEta1_ =
       ibook.book1D("BTLTrackMatchedTPmtdDirectWrongAssocEta1",
                    "Eta of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association;#eta_{RECO}",
                    30,
                    0.,
                    1.5);
    meBTLTrackMatchedTPmtdDirectWrongAssocPt1_ = ibook.book1D(
       "BTLTrackMatchedTPmtdDirectWrongAssocPt1",
       "Pt of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association;track pt [GeV]",
       50,
       0.,
       10.);
    meBTLTrackMatchedTPmtdDirectWrongAssocMVAQual1_ =
       ibook.book1D("BTLTrackMatchedTPmtdDirectWrongAssocMVAQual1",
                    "MVA of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; MVA score",
                    100,
                    -1.,
                    1.);
    meBTLTrackMatchedTPmtdDirectWrongAssocTimeRes1_ =
       ibook.book1D("BTLTrackMatchedTPmtdDirectWrongAssocTimeRes1",
                    "Time resolution of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD "
                    "association; t_{rec} - t_{sim} [ns] ",
                    240,
                    -0.3,
                    0.3);
    meBTLTrackMatchedTPmtdDirectWrongAssocTimePull1_ =
       ibook.book1D("BTLTrackMatchedTPmtdDirectWrongAssocTimePull1",
                    "Time pull of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; "
                    "(t_{rec}-t_{sim})/#sigma_{t}",
                    50,
                    -5.,
                    5.);
    meBTLTrackMatchedTPmtdDirectWrongAssocDeltaZOutR1_ = ibook.book2D(
       "BTLTrackMatchedTPmtdDirectWrongAssocDeltaZOutR1", "; Outer R [cm]; DeltaZ [cm]", 120, 0., 120., 1000, -40., 40);
    meBTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenghtvsEta1_ = ibook.book2D(
      "BTLTrackMatchedTPmtdDirectCorrectAssocTrackPathLenghtvsEta1", "MTD Track pathlength vs MTD track Eta;|#eta|;Pathlength", 30, 0, 1.5, 350, 0.0, 350.0);
    meBTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenght1_ = ibook.book1D(
      "BTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenght1", "MTD Track pathlength ; ;Pathlength", 400, 0, 400);



    meBTLTrackMatchedTPmtdDirectWrongAssocTrackNdf2_ = ibook.book1D(
        "BTLTrackMatchedTPmtdDirectWrongAssocTrackNdf2",
        "Chi2 of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; Ndof ",
        80,
        0.,
        220);
    meBTLTrackMatchedTPmtdDirectWrongAssocTrackChi22_ = ibook.book1D(
        "BTLTrackMatchedTPmtdDirectWrongAssocTrackChi22",
        "Chi2 of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; Chi2 ",
        80,
        0.,
        220);
    meBTLTrackMatchedTPmtdDirectWrongAssocTimeChi21_ = ibook.book1D(
       "BTLTrackMatchedTPmtdDirectWrongAssocTimeChi21",
       "Time chi2 of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; Chi2 ",
       200,
       0.,
       100);
     meBTLTrackMatchedTPmtdDirectWrongAssocTimeChi2vsMVAQual1_ = ibook.book2D(
       "BTLTrackMatchedTPmtdDirectWrongAssocTimeChi2vsMVAQual1", "Time chi2 of tracks matched to TP with sim hit in MTD (direct), correct track-MTD association; Chi2; MVAQual ",
       200, 0., 100, 100, -1, 1);
    meBTLTrackMatchedTPmtdDirectWrongAssocSpaceChi21_ = ibook.book1D(
       "BTLTrackMatchedTPmtdDirectWrongAssocSpaceChi21",
       "Space chi2 of tracks matched to TP with sim hit in MTD (direct), correct track-MTD association; Chi2 ",
       250,
       0.,
       250.);
    meBTLTrackMatchedTPmtdDirectWrongAssocTrackOutermostHitR2_ = ibook.book1D(
        "BTLTrackMatchedTPmtdDirectWrongAssocTrackOutermostHitR2",
        "Outermost hit position R of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; R [cm]",
        40,
        0.,
        120);
    meBTLTrackMatchedTPmtdDirectWrongAssocTrackIdOff2_ = ibook.book1D(
        "BTLTrackMatchedTPmtdDirectWrongAssocTrackIdOff2",
        "Track Id offset of reco (wrong) cluster associated to the track (direct)  - wrong track-MTD association; trackId (wrong)",
        6,
        -1.,
        5.);
    meBTLTrackMatchedTPmtdDirectWrongAssocDeltaZ2_ = ibook.book1D(
         "BTLTrackMatchedTPmtdDirectWrongAssocDeltaZ2",
         "Z of sim matched wrong cluster - Z of true sim cluster (direct) - wrong track-MTD association; DeltaZ (wrong - true) [cm]",
         1000,
         -50.,
         50.);
    meBTLTrackMatchedTPmtdDirectWrongAssocDeltaPhi2_ = ibook.book1D(
         "BTLTrackMatchedTPmtdDirectWrongAssocDeltaPhi2",
         "Phi of sim matched wrong cluster - Z of true sim cluster - wrong track-MTD association; DeltaPhi (wrong - true) ",
         500,
         -0.1,
         0.1);
    meBTLTrackMatchedTPmtdDirectWrongAssocDeltaT2_ = ibook.book1D(
         "BTLTrackMatchedTPmtdDirectWrongAssocDeltaT2",
         "Time of sim matched wrong cluster - time of true sim cluster (direct) - wrong track-MTD association; DeltaT (wrong - true) [ns]",
         480,
         -0.6,
         0.6);
    meBTLTrackMatchedTPmtdDirectWrongAssocEta2_ =
       ibook.book1D("BTLTrackMatchedTPmtdDirectWrongAssocEta2",
                    "Eta of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association;#eta_{RECO}",
                    30,
                    0.,
                    1.5);
    meBTLTrackMatchedTPmtdDirectWrongAssocPt2_ = ibook.book1D(
       "BTLTrackMatchedTPmtdDirectWrongAssocPt2",
       "Pt of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association;track pt [GeV]",
       50,
       0.,
       10.);
    meBTLTrackMatchedTPmtdDirectWrongAssocMVAQual2_ =
       ibook.book1D("BTLTrackMatchedTPmtdDirectWrongAssocMVAQual2",
                    "MVA of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; MVA score",
                    100,
                    -1.,
                    1.);
    meBTLTrackMatchedTPmtdDirectWrongAssocTimeRes2_ =
       ibook.book1D("BTLTrackMatchedTPmtdDirectWrongAssocTimeRes2",
                    "Time resolution of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD "
                    "association; t_{rec} - t_{sim} [ns] ",
                    240,
                    -0.3,
                    0.3);
    meBTLTrackMatchedTPmtdDirectWrongAssocTimePull2_ =
      ibook.book1D("BTLTrackMatchedTPmtdDirectWrongAssocTimePull2",
                   "Time pull of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; "
                   "(t_{rec}-t_{sim})/#sigma_{t}",
                   50,
                   -5.,
                   5.);
    meBTLTrackMatchedTPmtdDirectWrongAssocDeltaZOutR2_ = ibook.book2D(
       "BTLTrackMatchedTPmtdDirectWrongAssocDeltaZOutR2", "; Outer R [cm]; DeltaZ [cm]", 120, 0., 120., 1000, -40., 40);

     meBTLTrackMatchedTPmtdDirectWrongAssocTimeChi22_ = ibook.book1D(
       "BTLTrackMatchedTPmtdDirectWrongAssocTimeChi22",
       "Time chi2 of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; Chi2 ",
       200,
       0.,
       100);
     meBTLTrackMatchedTPmtdDirectWrongAssocTimeChi2vsMVAQual2_ = ibook.book2D(
       "BTLTrackMatchedTPmtdDirectWrongAssocTimeChi2vsMVAQual2", "Time chi2 of tracks matched to TP with sim hit in MTD (direct), correct track-MTD association; Chi2; MVAQual ",
       200, 0., 100, 100, -1, 1);
     meBTLTrackMatchedTPmtdDirectWrongAssocSpaceChi22_ = ibook.book1D(
       "BTLTrackMatchedTPmtdDirectWrongAssocSpaceChi22",
       "Space chi2 of tracks matched to TP with sim hit in MTD (direct), correct track-MTD association; Chi2 ",
       250,
       0.,
       250.);
      meBTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenghtvsEta2_ = ibook.book2D(
      "BTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenghtvsEta2", "MTD Track pathlength vs MTD track Eta;|#eta|;Pathlength", 30, 0, 1.5,350, 0.0, 350.0);
     meBTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenght2_ = ibook.book1D(
      "BTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenght2", "MTD Track pathlength ; ;Pathlength", 400, 0, 400);



    meBTLTrackMatchedTPmtdDirectWrongAssocTrackNdf3_ = ibook.book1D(
       "BTLTrackMatchedTPmtdDirectWrongAssocTrackNdf3",
       "Chi2 of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; Ndof ",
       80,
       0.,
       220);
    meBTLTrackMatchedTPmtdDirectWrongAssocTrackChi23_ = ibook.book1D(
       "BTLTrackMatchedTPmtdDirectWrongAssocTrackChi23",
       "Chi2 of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; Chi2 ",
       80,
       0.,
       220);
    meBTLTrackMatchedTPmtdDirectWrongAssocTrackOutermostHitR3_ = ibook.book1D(
       "BTLTrackMatchedTPmtdDirectWrongAssocTrackOutermostHitR3",
       "Outermost hit position R of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; R [cm]",
       40,
       0.,
       120);
    meBTLTrackMatchedTPmtdDirectWrongAssocTrackIdOff3_ = ibook.book1D(
        "BTLTrackMatchedTPmtdDirectWrongAssocTrackIdOff3",
        "Track Id offset of reco (wrong) cluster associated to the track (direct)  - wrong track-MTD association; trackId (wrong)",
        6,
        -1.,
        5.);
    meBTLTrackMatchedTPmtdDirectWrongAssocDeltaZ3_ = ibook.book1D(
         "BTLTrackMatchedTPmtdDirectWrongAssocDeltaZ3",
         "Z of sim matched wrong cluster - Z of true sim cluster (direct) - wrong track-MTD association; DeltaZ (wrong - true) [cm]",
         1000,
         -50.,
	 50.);
    meBTLTrackMatchedTPmtdDirectWrongAssocDeltaPhi3_ = ibook.book1D(
         "BTLTrackMatchedTPmtdDirectWrongAssocDeltaPhi3",
         "Phi of sim matched wrong cluster - Z of true sim cluster - wrong track-MTD association; DeltaPhi (wrong - true) ",
         500,
         -0.1,
         0.1);
    meBTLTrackMatchedTPmtdDirectWrongAssocDeltaT3_ = ibook.book1D(
         "BTLTrackMatchedTPmtdDirectWrongAssocDeltaT3",
         "Time of sim matched wrong cluster - time of true sim cluster (direct) - wrong track-MTD association; DeltaT (wrong - true) [ns]",
         480,
         -0.6,
         0.6);
    meBTLTrackMatchedTPmtdDirectWrongAssocEta3_ =
       ibook.book1D("BTLTrackMatchedTPmtdDirectWrongAssocEta3",
                    "Eta of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association;#eta_{RECO}",
                    30,
                    0.,
                    1.5);
    meBTLTrackMatchedTPmtdDirectWrongAssocPt3_ = ibook.book1D(
       "BTLTrackMatchedTPmtdDirectWrongAssocPt3",
       "Pt of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association;track pt [GeV]",
       50,
       0.,
       10.);
    meBTLTrackMatchedTPmtdDirectWrongAssocMVAQual3_ =
       ibook.book1D("BTLTrackMatchedTPmtdDirectWrongAssocMVAQual3",
                    "MVA of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; MVA score",
                    100,
                    -1.,
                    1.);
    meBTLTrackMatchedTPmtdDirectWrongAssocTimeRes3_ =
       ibook.book1D("BTLTrackMatchedTPmtdDirectWrongAssocTimeRes3",
                    "Time resolution of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD "
                    "association; t_{rec} - t_{sim} [ns] ",
                    240,
                    -0.3,
                    0.3);
    meBTLTrackMatchedTPmtdDirectWrongAssocTimePull3_ =
      ibook.book1D("BTLTrackMatchedTPmtdDirectWrongAssocTimePull3",
                   "Time pull of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; "
                   "(t_{rec}-t_{sim})/#sigma_{t}",
                   50,
                   -5.,
                   5.);
    meBTLTrackMatchedTPmtdDirectWrongAssocDeltaZOutR3_ = ibook.book2D(
       "BTLTrackMatchedTPmtdDirectWrongAssocDeltaZOutR3", "; Outer R [cm]; DeltaZ [cm]", 120, 0., 120., 1000, -40., 40);
    meBTLTrackMatchedTPmtdDirectWrongAssocTimeChi23_ = ibook.book1D(
       "BTLTrackMatchedTPmtdDirectWrongAssocTimeChi23",
       "Time chi2 of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; Chi2 ",
       200,
       0.,
       100);
      meBTLTrackMatchedTPmtdDirectWrongAssocTimeChi2vsMVAQual3_ = ibook.book2D(
       "BTLTrackMatchedTPmtdDirectWrongAssocTimeChi2vsMVAQual3", "Time chi2 of tracks matched to TP with sim hit in MTD (direct), correct track-MTD association; Chi2; MVAQual ",
       200, 0., 100, 100, -1, 1);
   

    meBTLTrackMatchedTPmtdDirectWrongAssocSpaceChi23_ = ibook.book1D(
       "BTLTrackMatchedTPmtdDirectWrongAssocSpaceChi23",
       "Space chi2 of tracks matched to TP with sim hit in MTD (direct), correct track-MTD association; Chi2 ",
       250,
       0.,
       250.);
    meBTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenghtvsEta3_ = ibook.book2D(
      "BTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenghtvsEta3", "MTD Track pathlength vs MTD track Eta;|#eta|;Pathlength", 30, 0, 1.5,350, 0.0, 350.0);
    meBTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenght3_ = ibook.book1D(
      "BTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenght3", "MTD Track pathlength ; ;Pathlength", 400, 0, 400);



 
    meBTLTrackMatchedTPmtdDirectWrongAssocTrackNdf_ = ibook.book1D(
       "BTLTrackMatchedTPmtdDirectWrongAssocTrackNdf",
       "Chi2 of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; Ndof ",
       80,
       0.,
       220);
    meBTLTrackMatchedTPmtdDirectWrongAssocTrackChi2_ = ibook.book1D(
       "BTLTrackMatchedTPmtdDirectWrongAssocTrackChi2",
       "Chi2 of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; Chi2 ",
       80,
       0.,
       220);
    meBTLTrackMatchedTPmtdDirectWrongAssocTrackOutermostHitR_ = ibook.book1D(
       "BTLTrackMatchedTPmtdDirectWrongAssocTrackOutermostHitR",
       "Outermost hit position R of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; R [cm]",
       40,
       0.,
       120);
     meBTLTrackMatchedTPmtdDirectWrongAssocTrackIdOff_ = ibook.book1D(
        "BTLTrackMatchedTPmtdDirectWrongAssocTrackIdOff",
        "Track Id offset of reco (wrong) cluster associated to the track (direct)  - wrong track-MTD association; trackId (wrong)",
        6,
        -1.,
        5.);
    meBTLTrackMatchedTPmtdDirectWrongAssocDeltaZ_ = ibook.book1D(
         "BTLTrackMatchedTPmtdDirectWrongAssocDeltaZ",
         "Z of sim matched wrong cluster - Z of true sim cluster (direct) - wrong track-MTD association; DeltaZ (wrong - true) [cm]",
         1000,
         -50.,
         50.);
   meBTLTrackMatchedTPmtdDirectWrongAssocDeltaPhi_ = ibook.book1D(
        "BTLTrackMatchedTPmtdDirectWrongAssocDeltaPhi",
        "Phi of sim matched wrong cluster - Z of true sim cluster - wrong track-MTD association; DeltaPhi (wrong - true) ",
        500,
        -0.1,
        0.1);
    meBTLTrackMatchedTPmtdDirectWrongAssocDeltaT_ = ibook.book1D(
         "BTLTrackMatchedTPmtdDirectWrongAssocDeltaT",
         "Time of sim matched wrong cluster - time of true sim cluster (direct) - wrong track-MTD association; DeltaT (wrong - true) [ns]",
         480,
         -0.6,
         0.6);
    meBTLTrackMatchedTPmtdDirectWrongAssocSimClusSize_ =  ibook.book1D(
        "BTLTrackMatchedTPmtdDirectWrongAssocSimClusSize",
        "Size of the sim clusters associated to the reco cluster associated to the track (direct) - wrong track-MTD association; Number of clusters ",
        10,
        -5.,
        5.);
    meBTLTrackMatchedTPmtdDirectWrongAssocRecoClusSize_ =  ibook.book1D(
        "BTLTrackMatchedTPmtdDirectWrongAssocRecoClusSize",
        "Size of the reco cluster associated to the track (direct) - wrong track-MTD association; Number of clusters ",
        10,
        -5.,
        5.);
    meBTLTrackMatchedTPmtdDirectWrongAssocDeltaZOutR_ = ibook.book2D(
        "BTLTrackMatchedTPmtdDirectWrongAssocDeltaZOutR", "; Outer R [cm]; DeltaZ [cm]", 120, 0., 120., 1000, -40., 40);
    meBTLTrackMatchedTPmtdDirectWrongAssocTimeChi2_ = ibook.book1D(
       "BTLTrackMatchedTPmtdDirectWrongAssocTimeChi2",
       "Time chi2 of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; Chi2 ",
       200,
       0.,
       100);
     meBTLTrackMatchedTPmtdDirectWrongAssocTimeChi2vsMVAQual_ = ibook.book2D(
       "BTLTrackMatchedTPmtdDirectWrongAssocTimeChi2vsMVAQual", "Time chi2 of tracks matched to TP with sim hit in MTD (direct), correct track-MTD association; Chi2; MVAQual ",
       200, 0., 100, 100, -1, 1);

    meBTLTrackMatchedTPmtdDirectWrongAssocSpaceChi2_ = ibook.book1D(
       "BTLTrackMatchedTPmtdDirectWrongAssocSpaceChi2",
       "Space chi2 of tracks matched to TP with sim hit in MTD (direct), correct track-MTD association; Chi2 ",
       250,
       0.,
       250.);
    meBTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenghtvsEta_ = ibook.book2D(
      "BTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenghtvsEta", "MTD Track pathlength vs MTD track Eta;|#eta|;Pathlength", 30, 0, 1.5,350,  0.0, 350.0);
    meBTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenght_ = ibook.book1D(
      "BTLTrackMatchedTPmtdDirectWrongAssocTrackPathLenght", "MTD Track pathlength ; ;Pathlength", 400, 0, 400);



    meBTLTrackMatchedTPmtdDirectNoAssocTrackNdf_ = ibook.book1D(
        "BTLTrackMatchedTPmtdDirectNoAssocTrackNdf",
        "Ndof of tracks matched to TP with sim hit in MTD (direct) - no track-MTD association; Ndof ",
        80,
        0.,
        220);
    meBTLTrackMatchedTPmtdDirectNoAssocTrackChi2_ = ibook.book1D(
        "BTLTrackMatchedTPmtdDirectNoAssocTrackChi2",
        "Chi2 of tracks matched to TP with sim hit in MTD (direct) - no track-MTD association; Chi2 ",
        80,
        0.,
        220);
    meBTLTrackMatchedTPmtdDirectNoAssocTrackOutermostHitR_ = ibook.book1D(
        "BTLTrackMatchedTPmtdDirectNoAssocTrackOutermostHitR",
        "Outermost hit position R of tracks matched to TP with sim hit in MTD (direct) - no track-MTD association; R [cm]",
        40,
        0.,
        120);
    meBTLTrackMatchedTPmtdDirectNoAssocSimClusSize_ =  ibook.book1D(
           "BTLTrackMatchedTPmtdDirectNoAssocSimClusSize",
           "Size of the sim clusters associated to the reco cluster associated to the track (direct) - no track-MTD association; Number of clusters ",
           10,
           -5.,
           5.);
    meBTLTrackMatchedTPmtdDirectNoAssocRecoClusSize_ =  ibook.book1D(
           "BTLTrackMatchedTPmtdDirectNoAssocRecoClusSize",
           "Size of the reco cluster associated to the track (direct) - no track-MTD association; Number of clusters ",
           10,
           -5.,
           5.);


    meBTLTrackMatchedTPnomtdAssocTrackChi2_ = ibook.book1D(
       "BTLTrackMatchedTPnomtdAssocTrackChi2", "Chi2 of tracks matched to TP w/o sim hit in MTD; Chi2", 80, 0., 220);
    meBTLTrackMatchedTPnomtdAssocTrackOutermostHitR_ = ibook.book1D(
       "BTLTrackMatchedTPnomtdAssocTrackOutermostHitR", "Outermost hit position R of tracks matched to TP w/o sim hit in MTD; R [cm]", 40, 0., 120.);
    meBTLTrackMatchedTPnomtdAssocTrackNdf_ = ibook.book1D(
       "BTLTrackMatchedTPnomtdAssocTrackNdf", "Ndf of tracks matched to TP w/o sim hit in MTD; Ndof", 80, 0., 220);
    meBTLTrackMatchedTPnomtdAssocTrackIdOff_ = ibook.book1D(
       "BTLTrackMatchedTPnomtdAssocTrackIdOff", "TrackIdOff of simCluster matched to the reco cluster associated to the track,  TP w/o sim hit in MTD; Track Id Off", 6, -1., 5.);
    meBTLTrackMatchedTPnomtdAssocSimClusSize_ =  ibook.book1D(
          "BTLTrackMatchedTPnomtdAssocSimClusSize",
          "Size of the sim clusters associated to the reco cluster associated to the track (direct),  TP w/o sim hit in MTD; Number of clusters ",
          10,
          -5.,
          5.);
    meBTLTrackMatchedTPnomtdAssocRecoClusSize_ =  ibook.book1D(
          "BTLTrackMatchedTPnomtdAssocRecoClusSize",
          "Size of the reco cluster associated to the track (direct),  TP w/o sim hit in MTD; Number of clusters ",
          10,
          -5.,
          5.);
    meBTLTrackMatchedTPnomtdAssocTrackID_ = ibook.book1D(
          "BTLTrackMatchedTPnomtdAssocTrackID", 
 	  "Diff track raw ID, TP w/o sim hit in MTD ; diff track raw Id",
 	  5,
         -1., 
 	  4.);

    meETLTrackMatchedTPmtd1CorrectAssocTimeChi2_ = ibook.book1D(
       "ETLTrackMatchedTPmtd1CorrectAssocTimeChi2",
       "Time chi2 of tracks matched to TP with sim hit in MTD (direct) - correct track-MTD association; Chi2 ",
       200,
       0.,
       100);
     meETLTrackMatchedTPmtd1CorrectAssocTimeChi2vsMVAQual_ = ibook.book2D(
       "ETLTrackMatchedTPmtd1CorrectAssocTimeChi2vsMVAQual", "Time chi2 of tracks matched to TP with sim hit in MTD (direct) - correct track-MTD association; Chi2; MVAQual ",
       200, 0., 100, 100, -1, 1);

    meETLTrackMatchedTPmtd1WrongAssocTimeChi2_ = ibook.book1D(
       "ETLTrackMatchedTPmtd1WrongAssocTimeChi2",
       "Time chi2 of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; Chi2 ",
       200,
       0.,
       100);
     meETLTrackMatchedTPmtd1WrongAssocTimeChi2vsMVAQual_ = ibook.book2D(
       "ETLTrackMatchedTPmtd1WrongAssocTimeChi2vsMVAQual", "Time chi2 of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; Chi2; MVAQual ",
       200, 0., 100, 100, -1, 1);



    meETLTrackMatchedTPmtd2CorrectAssocTimeChi2_ = ibook.book1D(
       "ETLTrackMatchedTPmtd2CorrectAssocTimeChi2",
       "Time chi2 of tracks matched to TP with sim hit in MTD (direct) - correct track-MTD association; Chi2 ",
       200,
       0.,
       100);
     meETLTrackMatchedTPmtd2CorrectAssocTimeChi2vsMVAQual_ = ibook.book2D(
       "ETLTrackMatchedTPmtd2CorrectAssocTimeChi2vsMVAQual", "Time chi2 of tracks matched to TP with sim hit in MTD (direct) - correct track-MTD association; Chi2; MVAQual ",
       200, 0., 100, 100, -1, 1);


    meETLTrackMatchedTPmtd2WrongAssocTimeChi2_ = ibook.book1D(
       "ETLTrackMatchedTPmtd2WrongAssocTimeChi2",
       "Time chi2 of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; Chi2 ",
       200,
       0.,
       100);
     meETLTrackMatchedTPmtd2WrongAssocTimeChi2vsMVAQual_ = ibook.book2D(
       "ETLTrackMatchedTPmtd2WrongAssocTimeChi2vsMVAQual", "Time chi2 of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; Chi2; MVAQual ",
       200, 0., 100, 100, -1, 1);


  }  // end optional plots

  meTrackResTot_ = ibook.book1D(
      "TrackRes", "t_{rec} - t_{sim} for LV associated tracks matched to TP; t_{rec} - t_{sim} [ns] ", 120, -0.15, 0.15);
  meTrackPullTot_ = ibook.book1D(
      "TrackPull", "Pull for LV associated tracks matched to TP; (t_{rec}-t_{sim})/#sigma_{t}", 50, -5., 5.);
  meTrackResTotvsMVAQual_ = ibook.bookProfile(
      "TrackResvsMVA",
      "t_{rec} - t_{sim} for LV associated tracks matched to TP vs MVA Quality; MVAQual; t_{rec} - t_{sim} [ns] ",
      100,
      -1.,
      1.,
      -0.15,
      0.15,
      "s");
  meTrackPullTotvsMVAQual_ = ibook.bookProfile(
      "TrackPullvsMVA",
      "Pull for LV associated tracks matched to TP vs MVA Quality; MVAQual; (t_{rec}-t_{sim})/#sigma_{t}",
      100,
      -1.,
      1.,
      -5.,
      5.,
      "s");

  meExtraPhiAtBTL_ = ibook.book1D(
      "ExtraPhiAtBTL", "Phi at BTL surface of extrapolated tracks associated to LV; phi [deg]", 720, -180., 180.);
  meExtraPhiAtBTLmatched_ =
      ibook.book1D("ExtraPhiAtBTLmatched",
                   "Phi at BTL surface of extrapolated tracks associated to LV matched with BTL hits; phi [deg]",
                   720,
                   -180.,
                   180.);
  meExtraBTLeneInCone_ =
      ibook.book1D("ExtraBTLeneInCone",
                   "BTL reconstructed energy in cone arounnd extrapolated track associated to LV; E [MeV]",
                   100,
                   0.,
                   50.);
  meExtraMTDfailExtenderEta_ =
      ibook.book1D("ExtraMTDfailExtenderEta",
                   "Eta of tracks associated to LV extrapolated to MTD with no track extender match to hits; track eta",
                   66,
                   0.,
                   3.3);
  meExtraMTDfailExtenderPt_ = ibook.book1D(
      "ExtraMTDfailExtenderPt",
      "Pt of tracks associated to LV extrapolated to MTD with no track extender match to hits; track pt [GeV] ",
      110,
      0.,
      11.);

  // Book the histograms for track-hit matching based on MC truth
  meBTLTrackMatchedTPmtdDirectEta_ =
      ibook.book1D("BTLTrackMatchedTPmtdDirectEta",
                   "Eta of tracks matched to TP with sim hit in MTD (direct);#eta_{RECO}",
                   30,
                   0.,
                   1.5);
  meBTLTrackMatchedTPmtdDirectPt_ =
      ibook.book1D("BTLTrackMatchedTPmtdDirectPt",
                   "Pt of tracks matched to TP with sim hit in MTD (direct); track pt [GeV]",
                   50,
                   0.,
                   10.);

  meBTLTrackMatchedTPmtdOtherEta_ = ibook.book1D("BTLTrackMatchedTPmtdOtherEta",
                                                 "Eta of tracks matched to TP with sim hit in MTD (other);#eta_{RECO}",
                                                 30,
                                                 0.,
                                                 1.5);
  meBTLTrackMatchedTPmtdOtherPt_ =
      ibook.book1D("BTLTrackMatchedTPmtdOtherPt",
                   "Pt of tracks matched to TP with sim hit in MTD (other); track pt [GeV]",
                   50,
                   0.,
                   10.);

  meBTLTrackMatchedTPnomtdEta_ = ibook.book1D(
      "BTLTrackMatchedTPnomtdEta", "Eta of tracks matched to TP w/o sim hit in MTD;#eta_{RECO}", 30, 0., 1.5);
  meBTLTrackMatchedTPnomtdPt_ = ibook.book1D(
      "BTLTrackMatchedTPnomtdPt", "Pt of tracks matched to TP w/o sim hit in MTD; track pt [GeV]", 50, 0., 10.);

  meBTLTrackMatchedTPmtdDirectCorrectAssocEta_ = ibook.book1D(
      "BTLTrackMatchedTPmtdDirectCorrectAssocEta",
      "Eta of tracks matched to TP with sim hit in MTD (direct), correct track-MTD association;#eta_{RECO}",
      30,
      0.,
      1.5);
  meBTLTrackMatchedTPmtdDirectCorrectAssocPt_ = ibook.book1D(
      "BTLTrackMatchedTPmtdDirectCorrectAssocPt",
      "Pt of tracks matched to TP with sim hit in MTD (direct) - correct track-MTD association;track pt [GeV]",
      50,
      0.,
      10.);
  meBTLTrackMatchedTPmtdDirectCorrectAssocMVAQual_ = ibook.book1D(
      "BTLTrackMatchedTPmtdDirectCorrectAssocMVAQual",
      "MVA of tracks matched to TP with sim hit in MTD (direct) - correct track-MTD association; MVA score",
      100,
      -1.,
      1.);
  meBTLTrackMatchedTPmtdDirectCorrectAssocTimeRes_ =
      ibook.book1D("BTLTrackMatchedTPmtdDirectCorrectAssocTimeRes",
                   "Time resolution of tracks matched to TP with sim hit in MTD (direct) - correct track-MTD "
                   "association; t_{rec} - t_{sim} [ns] ",
                   120,
                   -0.15,
                   0.15);
  meBTLTrackMatchedTPmtdDirectCorrectAssocTimePull_ =
      ibook.book1D("BTLTrackMatchedTPmtdDirectCorrectAssocTimePull",
                   "Time pull of tracks matched to TP with sim hit in MTD (direct) - correct track-MTD association; "
                   "(t_{rec}-t_{sim})/#sigma_{t}",
                   50,
                   -5.,
                   5.);

  meBTLTrackMatchedTPmtdDirectWrongAssocEta_ =
      ibook.book1D("BTLTrackMatchedTPmtdDirectWrongAssocEta",
                   "Eta of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association;#eta_{RECO}",
                   30,
                   0.,
                   1.5);

  meBTLTrackMatchedTPmtdDirectWrongAssocPt_ = ibook.book1D(
      "BTLTrackMatchedTPmtdDirectWrongAssocPt",
      "Pt of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association;track pt [GeV]",
      50,
      0.,
      10.);
  meBTLTrackMatchedTPmtdDirectWrongAssocMVAQual_ =
      ibook.book1D("BTLTrackMatchedTPmtdDirectWrongAssocMVAQual",
                   "MVA of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; MVA score",
                   100,
                   -1.,
                   1.);
  meBTLTrackMatchedTPmtdDirectWrongAssocTimeRes_ =
      ibook.book1D("BTLTrackMatchedTPmtdDirectWrongAssocTimeRes",
                   "Time resolution of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD "
                   "association; t_{rec} - t_{sim} [ns] ",
                   120,
                   -0.15,
                   0.15);
  meBTLTrackMatchedTPmtdDirectWrongAssocTimePull_ =
      ibook.book1D("BTLTrackMatchedTPmtdDirectWrongAssocTimePull",
                   "Time pull of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; "
                   "(t_{rec}-t_{sim})/#sigma_{t}",
                   50,
                   -5.,
                   5.);

  meBTLTrackMatchedTPmtdDirectNoAssocEta_ =
      ibook.book1D("BTLTrackMatchedTPmtdDirectNoAssocEta",
                   "Eta of tracks matched to TP with sim hit in MTD (direct) - no track-MTD association;#eta_{RECO}",
                   30,
                   0.,
                   1.5);
  meBTLTrackMatchedTPmtdDirectNoAssocPt_ =
      ibook.book1D("BTLTrackMatchedTPmtdDirectNoAssocPt",
                   "Pt of tracks matched to TP with sim hit in MTD (direct) - no track-MTD association;track pt [GeV]",
                   50,
                   0.,
                   10.);

  meBTLTrackMatchedTPmtdOtherCorrectAssocEta_ = ibook.book1D(
      "BTLTrackMatchedTPmtdOtherCorrectAssocEta",
      "Eta of tracks matched to TP with sim hit in MTD (direct), correct track-MTD association;#eta_{RECO}",
      30,
      0.,
      1.5);
  meBTLTrackMatchedTPmtdOtherCorrectAssocPt_ = ibook.book1D(
      "BTLTrackMatchedTPmtdOtherCorrectAssocPt",
      "Pt of tracks matched to TP with sim hit in MTD (direct) - correct track-MTD association;track pt [GeV]",
      50,
      0.,
      10.);
  meBTLTrackMatchedTPmtdOtherCorrectAssocMVAQual_ = ibook.book1D(
      "BTLTrackMatchedTPmtdOtherCorrectAssocMVAQual",
      "MVA of tracks matched to TP with sim hit in MTD (direct) - correct track-MTD association; MVA score",
      100,
      -1.,
      1.);
  meBTLTrackMatchedTPmtdOtherCorrectAssocTimeRes_ =
      ibook.book1D("BTLTrackMatchedTPmtdOtherCorrectAssocTimeRes",
                   "Time resolution of tracks matched to TP with sim hit in MTD (direct) - correct track-MTD "
                   "association; t_{rec} - t_{sim} [ns] ",
                   120,
                   -0.15,
                   0.15);
  meBTLTrackMatchedTPmtdOtherCorrectAssocTimePull_ =
      ibook.book1D("BTLTrackMatchedTPmtdOtherCorrectAssocTimePull",
                   "Time pull of tracks matched to TP with sim hit in MTD (direct) - correct track-MTD association; "
                   "(t_{rec}-t_{sim})/#sigma_{t}",
                   50,
                   -5.,
                   5.);

  meBTLTrackMatchedTPmtdOtherWrongAssocEta_ =
      ibook.book1D("BTLTrackMatchedTPmtdOtherWrongAssocEta",
                   "Eta of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association;#eta_{RECO}",
                   30,
                   0.,
                   1.5);
  meBTLTrackMatchedTPmtdOtherWrongAssocPt_ = ibook.book1D(
      "BTLTrackMatchedTPmtdOtherWrongAssocPt",
      "Pt of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association;track pt [GeV]",
      50,
      0.,
      10.);
  meBTLTrackMatchedTPmtdOtherWrongAssocMVAQual_ =
      ibook.book1D("BTLTrackMatchedTPmtdOtherWrongAssocMVAQual",
                   "MVA of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; MVA score",
                   100,
                   -1.,
                   1.);
  meBTLTrackMatchedTPmtdOtherWrongAssocTimeRes_ =
      ibook.book1D("BTLTrackMatchedTPmtdOtherWrongAssocTimeRes",
                   "Time resolution of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD "
                   "association; t_{rec} - t_{sim} [ns] ",
                   120,
                   -0.15,
                   0.15);
  meBTLTrackMatchedTPmtdOtherWrongAssocTimePull_ =
      ibook.book1D("BTLTrackMatchedTPmtdOtherWrongAssocTimePull",
                   "Time pull of tracks matched to TP with sim hit in MTD (direct) - wrong track-MTD association; "
                   "(t_{rec}-t_{sim})/#sigma_{t}",
                   50,
                   -5.,
                   5.);

  meBTLTrackMatchedTPmtdOtherNoAssocEta_ =
      ibook.book1D("BTLTrackMatchedTPmtdOtherNoAssocEta",
                   "Eta of tracks matched to TP with sim hit in MTD (direct) - no track-MTD association;#eta_{RECO}",
                   30,
                   0.,
                   1.5);
  meBTLTrackMatchedTPmtdOtherNoAssocPt_ =
      ibook.book1D("BTLTrackMatchedTPmtdOtherNoAssocPt",
                   "Pt of tracks matched to TP with sim hit in MTD (direct) - no track-MTD association;track pt [GeV]",
                   50,
                   0.,
                   10.);

  meBTLTrackMatchedTPnomtdAssocEta_ =
      ibook.book1D("BTLTrackMatchedTPnomtdAssocEta",
                   "Eta of tracks matched to TP w/o sim hit in MTD, with associated reco cluster;#eta_{RECO}",
                   30,
                   0.,
                   1.5);
  meBTLTrackMatchedTPnomtdAssocPt_ =
      ibook.book1D("BTLTrackMatchedTPnomtdAssocPt",
                   "Pt of tracks matched to TP w/o sim hit in MTD, with associated reco cluster;track pt [GeV]",
                   50,
                   0.,
                   10.);
  meBTLTrackMatchedTPnomtdAssocMVAQual_ =
      ibook.book1D("BTLTrackMatchedTPnomtdAssocMVAQual",
                   "MVA of tracks matched to TP w/o sim hit in MTD, with associated reco cluster; MVA score",
                   100,
                   -1.,
                   1.);
  meBTLTrackMatchedTPnomtdAssocTimeRes_ = ibook.book1D("BTLTrackMatchedTPnomtdAssocTimeRes",
                                                       "Time resolution of tracks matched to TP w/o sim hit in MTD, "
                                                       "with associated reco cluster; t_{rec} - t_{sim} [ns] ",
                                                       120,
                                                       -0.15,
                                                       0.15);
  meBTLTrackMatchedTPnomtdAssocTimePull_ = ibook.book1D("BTLTrackMatchedTPnomtdAssocTimePull",
                                                        "Time pull of tracks matched to TP w/o sim hit in MTD, with "
                                                        "associated reco cluster; (t_{rec}-t_{sim})/#sigma_{t}",
                                                        50,
                                                        -5.,
                                                        5.);

  meETLTrackMatchedTPmtd1Eta_ = ibook.book1D("ETLTrackMatchedTPmtd1Eta",
                                             "Eta of tracks matched to TP with sim hit in MTD (>= 1 hit);#eta_{RECO}",
                                             30,
                                             1.5,
                                             3.0);
  meETLTrackMatchedTPmtd1Pt_ = ibook.book1D("ETLTrackMatchedTPmtd1Pt",
                                            "Pt of tracks matched to TP with sim hit in MTD (>= 1 hit); track pt [GeV]",
                                            50,
                                            0.,
                                            10.);

  meETLTrackMatchedTPmtd2Eta_ = ibook.book1D(
      "ETLTrackMatchedTPmtd2Eta", "Eta of tracks matched to TP with sim hit in MTD (2 hits);#eta_{RECO}", 30, 1.5, 3.0);
  meETLTrackMatchedTPmtd2Pt_ = ibook.book1D(
      "ETLTrackMatchedTPmtd2Pt", "Pt of tracks matched to TP with sim hit in MTD (2 hits); track pt [GeV]", 50, 0., 10.);

  meETLTrackMatchedTPnomtdEta_ = ibook.book1D(
      "ETLTrackMatchedTPnomtdEta", "Eta of tracks matched to TP w/o sim hit in MTD;#eta_{RECO}", 30, 1.5, 3.0);
  meETLTrackMatchedTPnomtdPt_ = ibook.book1D(
      "ETLTrackMatchedTPnomtdPt", "Pt of tracks matched to TP w/o sim hit in MTD; track pt [GeV]", 50, 0., 10.);

  meETLTrackMatchedTPmtd1CorrectAssocEta_ = ibook.book1D(
      "ETLTrackMatchedTPmtd1CorrectAssocEta",
      "Eta of tracks matched to TP with sim hit in MTD (>= 1 hit), correct track-MTD association;#eta_{RECO}",
      30,
      1.5,
      3.0);
  meETLTrackMatchedTPmtd1CorrectAssocPt_ = ibook.book1D(
      "ETLTrackMatchedTPmtd1CorrectAssocPt",
      "Pt of tracks matched to TP with sim hit in MTD (>= 1 hit) - correct track-MTD association;track pt [GeV]",
      50,
      0.,
      10.);
  meETLTrackMatchedTPmtd1CorrectAssocMVAQual_ = ibook.book1D(
      "ETLTrackMatchedTPmtd1CorrectAssocMVAQual",
      "MVA of tracks matched to TP with sim hit in MTD (>= 1 hit) - correct track-MTD association; MVA score",
      100,
      -1.,
      1.);
  meETLTrackMatchedTPmtd1CorrectAssocTimeRes_ =
      ibook.book1D("ETLTrackMatchedTPmtd1CorrectAssocTimeRes",
                   "Time resolution of tracks matched to TP with sim hit in MTD (>= 1 hit) - correct track-MTD "
                   "association; t_{rec} - t_{sim} [ns] ",
                   120,
                   -0.15,
                   0.15);
  meETLTrackMatchedTPmtd1CorrectAssocTimePull_ =
      ibook.book1D("ETLTrackMatchedTPmtd1CorrectAssocTimePull",
                   "Time pull of tracks matched to TP with sim hit in MTD (>= 1 hit) - correct track-MTD association; "
                   "(t_{rec}-t_{sim})/#sigma_{t}",
                   50,
                   -5.,
                   5.);

  meETLTrackMatchedTPmtd2CorrectAssocEta_ = ibook.book1D(
      "ETLTrackMatchedTPmtd2CorrectAssocEta",
      "Eta of tracks matched to TP with sim hit in MTD (2 hits), correct track-MTD association;#eta_{RECO}",
      30,
      1.5,
      3.0);
  meETLTrackMatchedTPmtd2CorrectAssocPt_ = ibook.book1D(
      "ETLTrackMatchedTPmtd2CorrectAssocPt",
      "Pt of tracks matched to TP with sim hit in MTD (2 hits) - correct track-MTD association;track pt [GeV]",
      50,
      0.,
      10.);
  meETLTrackMatchedTPmtd2CorrectAssocMVAQual_ = ibook.book1D(
      "ETLTrackMatchedTPmtd2CorrectAssocMVAQual",
      "MVA of tracks matched to TP with sim hit in MTD (2 hits) - correct track-MTD association; MVA score",
      100,
      -1.,
      1.);
  meETLTrackMatchedTPmtd2CorrectAssocTimeRes_ =
      ibook.book1D("ETLTrackMatchedTPmtd2CorrectAssocTimeRes",
                   "Time resolution of tracks matched to TP with sim hit in MTD (2 hits) - correct track-MTD "
                   "association; t_{rec} - t_{sim} [ns] ",
                   120,
                   -0.15,
                   0.15);
  meETLTrackMatchedTPmtd2CorrectAssocTimePull_ =
      ibook.book1D("ETLTrackMatchedTPmtd2CorrectAssocTimePull",
                   "Time pull of tracks matched to TP with sim hit in MTD (2 hits) - correct track-MTD association; "
                   "(t_{rec}-t_{sim})/#sigma_{t}",
                   50,
                   -5.,
                   5.);

  meETLTrackMatchedTPmtd1WrongAssocEta_ = ibook.book1D(
      "ETLTrackMatchedTPmtd1WrongAssocEta",
      "Eta of tracks matched to TP with sim hit in MTD (>= 1 hit), wrong track-MTD association;#eta_{RECO}",
      30,
      1.5,
      3.0);
  meETLTrackMatchedTPmtd1WrongAssocPt_ = ibook.book1D(
      "ETLTrackMatchedTPmtd1WrongAssocPt",
      "Pt of tracks matched to TP with sim hit in MTD (>= 1 hit) - wrong track-MTD association;track pt [GeV]",
      50,
      0.,
      10.);
  meETLTrackMatchedTPmtd1WrongAssocMVAQual_ = ibook.book1D(
      "ETLTrackMatchedTPmtd1WrongAssocMVAQual",
      "MVA of tracks matched to TP with sim hit in MTD (>= 1 hit) - wrong track-MTD association; MVA score",
      100,
      -1.,
      1.);
  meETLTrackMatchedTPmtd1WrongAssocTimeRes_ =
      ibook.book1D("ETLTrackMatchedTPmtd1WrongAssocTimeRes",
                   "Time resolution of tracks matched to TP with sim hit in MTD (>= 1 hit) - wrong track-MTD "
                   "association; t_{rec} - t_{sim} [ns] ",
                   120,
                   -0.15,
                   0.15);
  meETLTrackMatchedTPmtd1WrongAssocTimePull_ =
      ibook.book1D("ETLTrackMatchedTPmtd1WrongAssocTimePull",
                   "Time pull of tracks matched to TP with sim hit in MTD (>= 1 hit) - wrong track-MTD association; "
                   "(t_{rec}-t_{sim})/#sigma_{t}",
                   50,
                   -5.,
                   5.);

  meETLTrackMatchedTPmtd2WrongAssocEta_ =
      ibook.book1D("ETLTrackMatchedTPmtd2WrongAssocEta",
                   "Eta of tracks matched to TP with sim hit in MTD (2 hits), wrong track-MTD association;#eta_{RECO}",
                   30,
                   1.5,
                   3.0);
  meETLTrackMatchedTPmtd2WrongAssocPt_ = ibook.book1D(
      "ETLTrackMatchedTPmtd2WrongAssocPt",
      "Pt of tracks matched to TP with sim hit in MTD (2 hits) - wrong track-MTD association;track pt [GeV]",
      50,
      0.,
      10.);
  meETLTrackMatchedTPmtd2WrongAssocMVAQual_ =
      ibook.book1D("ETLTrackMatchedTPmtd2WrongAssocMVAQual",
                   "MVA of tracks matched to TP with sim hit in MTD (2 hits) - wrong track-MTD association; MVA score",
                   100,
                   -1.,
                   1.);
  meETLTrackMatchedTPmtd2WrongAssocTimeRes_ =
      ibook.book1D("ETLTrackMatchedTPmtd2WrongAssocTimeRes",
                   "Time resolution of tracks matched to TP with sim hit in MTD (2 hits) - wrong track-MTD "
                   "association; t_{rec} - t_{sim} [ns] ",
                   120,
                   -0.15,
                   0.15);
  meETLTrackMatchedTPmtd2WrongAssocTimePull_ =
      ibook.book1D("ETLTrackMatchedTPmtd2WrongAssocTimePull",
                   "Time pull of tracks matched to TP with sim hit in MTD (2 hits) - wrong track-MTD association; "
                   "(t_{rec}-t_{sim})/#sigma_{t}",
                   50,
                   -5.,
                   5.);

  meETLTrackMatchedTPmtd1NoAssocEta_ = ibook.book1D(
      "ETLTrackMatchedTPmtd1NoAssocEta",
      "Eta of tracks matched to TP with sim hit in MTD (>= 1 hit), missing track-MTD association;#eta_{RECO}",
      30,
      1.5,
      3.0);
  meETLTrackMatchedTPmtd1NoAssocPt_ = ibook.book1D(
      "ETLTrackMatchedTPmtd1NoAssocPt",
      "Pt of tracks matched to TP with sim hit in MTD (>= 1 hit) - missing track-MTD association;track pt [GeV]",
      50,
      0.,
      10.);

  meETLTrackMatchedTPmtd2NoAssocEta_ = ibook.book1D(
      "ETLTrackMatchedTPmtd2NoAssocEta",
      "Eta of tracks matched to TP with sim hit in MTD (2 hits), missing track-MTD association;#eta_{RECO}",
      30,
      1.5,
      3.0);
  meETLTrackMatchedTPmtd2NoAssocPt_ = ibook.book1D(
      "ETLTrackMatchedTPmtd2NoAssocPt",
      "Pt of tracks matched to TP with sim hit in MTD (2 hits) - missing track-MTD association;track pt [GeV]",
      50,
      0.,
      10.);
  meETLTrackMatchedTPnomtdAssocEta_ =
      ibook.book1D("ETLTrackMatchedTPnomtdAssocEta",
                   "Eta of tracks matched to TP w/o sim hit in MTD, with associated reco cluster;#eta_{RECO}",
                   30,
                   1.5,
                   3.0);
  meETLTrackMatchedTPnomtdAssocPt_ =
      ibook.book1D("ETLTrackMatchedTPnomtdAssocPt",
                   "Pt of tracks matched to TP w/o sim hit in MTD, with associated reco cluster;track pt [GeV]",
                   50,
                   0.,
                   10.);
  meETLTrackMatchedTPnomtdAssocMVAQual_ =
      ibook.book1D("ETLTrackMatchedTPnomtdAssocMVAQual",
                   "MVA of tracks matched to TP w/o sim hit in MTD, with associated reco cluster; MVA score",
                   100,
                   -1.,
                   1.);
  meETLTrackMatchedTPnomtdAssocTimeRes_ = ibook.book1D("ETLTrackMatchedTPnomtdAssocTimeRes",
                                                       "Time resolution of tracks matched to TP w/o sim hit in MTD, "
                                                       "with associated reco cluster; t_{rec} - t_{sim} [ns] ",
                                                       120,
                                                       -0.15,
                                                       0.15);
  meETLTrackMatchedTPnomtdAssocTimePull_ = ibook.book1D("ETLTrackMatchedTPnomtdAssocTimePull",
                                                        "Time pull of tracks matched to TP w/o sim hit in MTD, with "
                                                        "associated reco cluster; (t_{rec}-t_{sim})/#sigma_{t}",
                                                        50,
                                                        -5.,
                                                        5.);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------

void MtdTracksValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/Tracks");
  desc.add<bool>("optionalPlots", false);
  desc.add<edm::InputTag>("inputTagG", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("inputTagT", edm::InputTag("trackExtenderWithMTD"));
  desc.add<edm::InputTag>("inputTagV", edm::InputTag("offlinePrimaryVertices4D"));
  desc.add<edm::InputTag>("inputTagH", edm::InputTag("generatorSmeared"));
  desc.add<edm::InputTag>("SimTag", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("TPtoRecoTrackAssoc", edm::InputTag("trackingParticleRecoTrackAsssociation"));
  desc.add<edm::InputTag>("tp2SimAssociationMapTag", edm::InputTag("mtdSimLayerClusterToTPAssociation"));
  desc.add<edm::InputTag>("Sim2tpAssociationMapTag", edm::InputTag("mtdSimLayerClusterToTPAssociation"));
  desc.add<edm::InputTag>("r2sAssociationMapTag", edm::InputTag("mtdRecoClusterToSimLayerClusterAssociation"));
  desc.add<edm::InputTag>("btlRecHits", edm::InputTag("mtdRecHits", "FTLBarrel"));
  desc.add<edm::InputTag>("etlRecHits", edm::InputTag("mtdRecHits", "FTLEndcap"));
  desc.add<edm::InputTag>("recCluTagBTL", edm::InputTag("mtdClusters", "FTLBarrel"));
  desc.add<edm::InputTag>("recCluTagETL", edm::InputTag("mtdClusters", "FTLEndcap"));
  desc.add<edm::InputTag>("tmtd", edm::InputTag("trackExtenderWithMTD:generalTracktmtd"));
  desc.add<edm::InputTag>("sigmatmtd", edm::InputTag("trackExtenderWithMTD:generalTracksigmatmtd"));
  desc.add<edm::InputTag>("t0Src", edm::InputTag("trackExtenderWithMTD:generalTrackt0"));
  desc.add<edm::InputTag>("sigmat0Src", edm::InputTag("trackExtenderWithMTD:generalTracksigmat0"));
  desc.add<edm::InputTag>("trackAssocSrc", edm::InputTag("trackExtenderWithMTD:generalTrackassoc"))
      ->setComment("Association between General and MTD Extended tracks");
  desc.add<edm::InputTag>("pathLengthSrc", edm::InputTag("trackExtenderWithMTD:generalTrackPathLength"));
  desc.add<edm::InputTag>("btlMatchTimeChi2", edm::InputTag("trackExtenderWithMTD:btlMatchTimeChi2"));
  desc.add<edm::InputTag>("etlMatchTimeChi2", edm::InputTag("trackExtenderWithMTD:etlMatchTimeChi2"));
  desc.add<edm::InputTag>("btlMatchChi2", edm::InputTag("trackExtenderWithMTD:btlMatchChi2"));
  desc.add<edm::InputTag>("t0SafePID", edm::InputTag("tofPID:t0safe"));
  desc.add<edm::InputTag>("sigmat0SafePID", edm::InputTag("tofPID:sigmat0safe"));
  desc.add<edm::InputTag>("sigmat0PID", edm::InputTag("tofPID:sigmat0"));
  desc.add<edm::InputTag>("t0PID", edm::InputTag("tofPID:t0"));
  desc.add<edm::InputTag>("sigmaTofPi", edm::InputTag("trackExtenderWithMTD:generalTrackSigmaTofPi"));
  desc.add<edm::InputTag>("sigmaTofK", edm::InputTag("trackExtenderWithMTD:generalTrackSigmaTofK"));
  desc.add<edm::InputTag>("sigmaTofP", edm::InputTag("trackExtenderWithMTD:generalTrackSigmaTofP"));
  desc.add<edm::InputTag>("trackMVAQual", edm::InputTag("mtdTrackQualityMVA:mtdQualMVA"));
  desc.add<edm::InputTag>("outermostHitPositionSrc",
                          edm::InputTag("trackExtenderWithMTD:generalTrackOutermostHitPosition"));
  desc.add<double>("trackMaximumPt", 12.);  // [GeV]
  desc.add<double>("trackMaximumBtlEta", 1.5);
  desc.add<double>("trackMinimumEtlEta", 1.6);
  desc.add<double>("trackMaximumEtlEta", 3.);

  descriptions.add("mtdTracksValid", desc);
}

const bool MtdTracksValidation::trkTPSelLV(const TrackingParticle& tp) {
  bool match = (tp.status() != 1) ? false : true;
  return match;
}

const bool MtdTracksValidation::trkTPSelAll(const TrackingParticle& tp) {
  bool match = false;

  auto x_pv = tp.parentVertex()->position().x();
  auto y_pv = tp.parentVertex()->position().y();
  auto z_pv = tp.parentVertex()->position().z();

  auto r_pv = std::sqrt(x_pv * x_pv + y_pv * y_pv);

  match = tp.charge() != 0 && std::abs(tp.eta()) < etacutGEN_ && tp.pt() > pTcutBTL_ && r_pv < rBTL_ &&
          std::abs(z_pv) < zETL_;
  return match;
}

const bool MtdTracksValidation::trkRecSel(const reco::TrackBase& trk) {
  bool match = false;
  match = std::abs(trk.eta()) <= etacutREC_ && trk.pt() > pTcutBTL_;
  return match;
}

const bool MtdTracksValidation::trkRecSelLowPt(const reco::TrackBase& trk) {
  bool match = false;
  match = std::abs(trk.eta()) <= etacutREC_ && trk.pt() > pTcutETL_ && trk.pt() < pTcutBTL_;
  return match;
}

const edm::Ref<std::vector<TrackingParticle>>* MtdTracksValidation::getMatchedTP(const reco::TrackBaseRef& recoTrack) {
  auto found = r2s_->find(recoTrack);

  // reco track not matched to any TP
  if (found == r2s_->end())
    return nullptr;

  //matched TP equal to any TP associated to in time events
  for (const auto& tp : found->val) {
    if (tp.first->eventId().bunchCrossing() == 0)
      return &tp.first;
  }

  // reco track not matched to any TP from vertex
  return nullptr;
}

void MtdTracksValidation::fillTrackClusterMatchingHistograms(MonitorElement* me1,
                                                             MonitorElement* me2,
                                                             MonitorElement* me3,
                                                             MonitorElement* me4,
                                                             MonitorElement* me5,
                                                             float var1,
                                                             float var2,
                                                             float var3,
                                                             float var4,
                                                             float var5,
                                                             bool flag) {
  me1->Fill(var1);
  me2->Fill(var2);
  if (flag) {
    me3->Fill(var3);
    me4->Fill(var4);
    me5->Fill(var5);
  }
}

DEFINE_FWK_MODULE(MtdTracksValidation);
