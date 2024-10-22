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
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"
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

#include "DataFormats/Common/interface/OneToMany.h"
#include "DataFormats/Common/interface/AssociationMap.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/Associations/interface/MtdSimLayerClusterToTPAssociatorBaseImpl.h"

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

  // ------------ member data ------------

  const std::string folder_;
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
  const reco::SimToRecoCollection* s2r_;

  edm::EDGetTokenT<reco::TrackCollection> GenRecTrackToken_;
  edm::EDGetTokenT<reco::TrackCollection> RecTrackToken_;

  edm::EDGetTokenT<TrackingParticleCollection> trackingParticleCollectionToken_;
  edm::EDGetTokenT<reco::SimToRecoCollection> simToRecoAssociationToken_;
  edm::EDGetTokenT<reco::RecoToSimCollection> recoToSimAssociationToken_;
  edm::EDGetTokenT<reco::TPToSimCollectionMtd> tp2SimAssociationMapToken_;
  edm::EDGetTokenT<FTLRecHitCollection> btlRecHitsToken_;
  edm::EDGetTokenT<FTLRecHitCollection> etlRecHitsToken_;

  edm::EDGetTokenT<edm::ValueMap<int>> trackAssocToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> pathLengthToken_;

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
  MonitorElement* meBTLTrackEffEtaTot_;
  MonitorElement* meBTLTrackEffPhiTot_;
  MonitorElement* meBTLTrackEffPtTot_;
  MonitorElement* meBTLTrackEffEtaMtd_;
  MonitorElement* meBTLTrackEffPhiMtd_;
  MonitorElement* meBTLTrackEffPtMtd_;
  MonitorElement* meBTLTrackPtRes_;

  MonitorElement* meETLTrackRPTime_;
  MonitorElement* meETLTrackEffEtaTot_[2];
  MonitorElement* meETLTrackEffEtaTotLowPt_[2];
  MonitorElement* meETLTrackEffPhiTot_[2];
  MonitorElement* meETLTrackEffPtTot_[2];
  MonitorElement* meETLTrackEffEtaMtd_[2];
  MonitorElement* meETLTrackEffEtaMtdLowPt_[2];
  MonitorElement* meETLTrackEffEta2MtdLowPt_[2];
  MonitorElement* meETLTrackEffPhiMtd_[2];
  MonitorElement* meETLTrackEffPtMtd_[2];
  MonitorElement* meETLTrackEffEta2Mtd_[2];
  MonitorElement* meETLTrackEffPhi2Mtd_[2];
  MonitorElement* meETLTrackEffPt2Mtd_[2];
  MonitorElement* meETLTrackPtRes_;

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

  MonitorElement* meTrackPtTot_;
  MonitorElement* meExtraPtMtd_;
  MonitorElement* meExtraPtEtl2Mtd_;

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

  MonitorElement* meTrackMatchedTPEffPtTot_;
  MonitorElement* meTrackMatchedTPEffPtTotLV_;
  MonitorElement* meTrackMatchedTPEffPtMtd_;
  MonitorElement* meTrackMatchedTPEffPtEtl2Mtd_;
  MonitorElement* meTrackMatchedTPmtdEffPtTot_;
  MonitorElement* meTrackMatchedTPmtdEffPtMtd_;
  MonitorElement* meTrackEtaTot_;
  MonitorElement* meExtraEtaMtd_;
  MonitorElement* meExtraEtaEtl2Mtd_;
  MonitorElement* meTrackMatchedTPEffEtaTot_;
  MonitorElement* meTrackMatchedTPEffEtaTotLV_;
  MonitorElement* meTrackMatchedTPEffEtaMtd_;
  MonitorElement* meTrackMatchedTPEffEtaEtl2Mtd_;
  MonitorElement* meTrackMatchedTPmtdEffEtaTot_;
  MonitorElement* meTrackMatchedTPmtdEffEtaMtd_;
  MonitorElement* meTrackResTot_;
  MonitorElement* meTrackPullTot_;
  MonitorElement* meTrackResTotvsMVAQual_;
  MonitorElement* meTrackPullTotvsMVAQual_;

  MonitorElement* meExtraPhiAtBTL_;
  MonitorElement* meExtraPhiAtBTLmatched_;
  MonitorElement* meExtraBTLeneInCone_;
  MonitorElement* meExtraMTDfailExtenderEta_;
  MonitorElement* meExtraMTDfailExtenderPt_;
};

// ------------ constructor and destructor --------------
MtdTracksValidation::MtdTracksValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
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
  btlRecHitsToken_ = consumes<FTLRecHitCollection>(iConfig.getParameter<edm::InputTag>("btlRecHits"));
  etlRecHitsToken_ = consumes<FTLRecHitCollection>(iConfig.getParameter<edm::InputTag>("etlRecHits"));
  trackAssocToken_ = consumes<edm::ValueMap<int>>(iConfig.getParameter<edm::InputTag>("trackAssocSrc"));
  pathLengthToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("pathLengthSrc"));
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

  std::unordered_map<uint32_t, MTDHit> m_btlHits;
  std::unordered_map<uint32_t, MTDHit> m_etlHits;
  std::unordered_map<uint32_t, std::set<unsigned long int>> m_btlTrkPerCell;
  std::unordered_map<uint32_t, std::set<unsigned long int>> m_etlTrkPerCell;
  const auto& tp2SimAssociationMap = iEvent.get(tp2SimAssociationMapToken_);

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
  const auto& outermostHitPosition = iEvent.get(outermostHitPositionToken_);

  auto simToRecoH = makeValid(iEvent.getHandle(simToRecoAssociationToken_));
  s2r_ = simToRecoH.product();

  auto recoToSimH = makeValid(iEvent.getHandle(recoToSimAssociationToken_));
  r2s_ = recoToSimH.product();

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

      if (std::abs(trackGen.eta()) < trackMaxBtlEta_) {
        // --- all BTL tracks (with and without hit in MTD) ---
        meBTLTrackEffEtaTot_->Fill(trackGen.eta());
        meBTLTrackEffPhiTot_->Fill(trackGen.phi());
        meBTLTrackEffPtTot_->Fill(trackGen.pt());

        bool MTDBtl = false;
        int numMTDBtlvalidhits = 0;
        for (const auto hit : track.recHits()) {
          if (hit->isValid() == false)
            continue;
          MTDDetId Hit = hit->geographicalId();
          if ((Hit.det() == 6) && (Hit.subdetId() == 1) && (Hit.mtdSubDetector() == 1)) {
            MTDBtl = true;
            numMTDBtlvalidhits++;
          }
        }
        meTrackNumHits_->Fill(numMTDBtlvalidhits);

        // --- keeping only tracks with last hit in MTD ---
        if (MTDBtl == true) {
          isBTL = true;
          meBTLTrackEffEtaMtd_->Fill(trackGen.eta());
          meBTLTrackEffPhiMtd_->Fill(trackGen.phi());
          meBTLTrackEffPtMtd_->Fill(trackGen.pt());
          meBTLTrackRPTime_->Fill(track.t0());
          meBTLTrackPtRes_->Fill((trackGen.pt() - track.pt()) / trackGen.pt());
        }
        if (isBTL && Sigmat0Safe[trackref] < 0.) {
          meTrackNumHitsNT_->Fill(numMTDBtlvalidhits);
        }
      }  //loop over (geometrical) BTL tracks

      else {
        // --- all ETL tracks (with and without hit in MTD) ---
        if ((trackGen.eta() < -trackMinEtlEta_) && (trackGen.eta() > -trackMaxEtlEta_)) {
          meETLTrackEffEtaTot_[0]->Fill(trackGen.eta());
          meETLTrackEffPhiTot_[0]->Fill(trackGen.phi());
          meETLTrackEffPtTot_[0]->Fill(trackGen.pt());
        }

        if ((trackGen.eta() > trackMinEtlEta_) && (trackGen.eta() < trackMaxEtlEta_)) {
          meETLTrackEffEtaTot_[1]->Fill(trackGen.eta());
          meETLTrackEffPhiTot_[1]->Fill(trackGen.phi());
          meETLTrackEffPtTot_[1]->Fill(trackGen.pt());
        }

        bool MTDEtlZnegD1 = false;
        bool MTDEtlZnegD2 = false;
        bool MTDEtlZposD1 = false;
        bool MTDEtlZposD2 = false;
        int numMTDEtlvalidhits = 0;
        for (const auto hit : track.recHits()) {
          if (hit->isValid() == false)
            continue;
          MTDDetId Hit = hit->geographicalId();
          if ((Hit.det() == 6) && (Hit.subdetId() == 1) && (Hit.mtdSubDetector() == 2)) {
            isETL = true;
            ETLDetId ETLHit = hit->geographicalId();

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
        if ((trackGen.eta() < -trackMinEtlEta_) && (trackGen.eta() > -trackMaxEtlEta_)) {
          twoETLdiscs = (MTDEtlZnegD1 == true) && (MTDEtlZnegD2 == true);
          if ((MTDEtlZnegD1 == true) || (MTDEtlZnegD2 == true)) {
            meETLTrackEffEtaMtd_[0]->Fill(trackGen.eta());
            meETLTrackEffPhiMtd_[0]->Fill(trackGen.phi());
            meETLTrackEffPtMtd_[0]->Fill(trackGen.pt());
            if (twoETLdiscs) {
              meETLTrackEffEta2Mtd_[0]->Fill(trackGen.eta());
              meETLTrackEffPhi2Mtd_[0]->Fill(trackGen.phi());
              meETLTrackEffPt2Mtd_[0]->Fill(trackGen.pt());
            }
          }
        }
        if ((trackGen.eta() > trackMinEtlEta_) && (trackGen.eta() < trackMaxEtlEta_)) {
          twoETLdiscs = (MTDEtlZposD1 == true) && (MTDEtlZposD2 == true);
          if ((MTDEtlZposD1 == true) || (MTDEtlZposD2 == true)) {
            meETLTrackEffEtaMtd_[1]->Fill(trackGen.eta());
            meETLTrackEffPhiMtd_[1]->Fill(trackGen.phi());
            meETLTrackEffPtMtd_[1]->Fill(trackGen.pt());
            if (twoETLdiscs) {
              meETLTrackEffEta2Mtd_[1]->Fill(trackGen.eta());
              meETLTrackEffPhi2Mtd_[1]->Fill(trackGen.phi());
              meETLTrackEffPt2Mtd_[1]->Fill(trackGen.pt());
            }
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

      // TrackingParticle based matching

      const reco::TrackBaseRef trkrefb(trackref);
      auto tp_info = getMatchedTP(trkrefb);

      meTrackPtTot_->Fill(trackGen.pt());
      meTrackEtaTot_->Fill(std::abs(trackGen.eta()));
      if (tp_info != nullptr && trkTPSelAll(**tp_info)) {
        if (trackGen.pt() < trackMaxPt_) {
          if (isBTL) {
            meBTLTrackMatchedTPPtResMtd_->Fill(std::abs(track.pt() - (*tp_info)->pt()) /
                                               std::abs(trackGen.pt() - (*tp_info)->pt()));
            meBTLTrackMatchedTPPtRatioGen_->Fill(trackGen.pt() / (*tp_info)->pt());
            meBTLTrackMatchedTPPtRatioMtd_->Fill(track.pt() / (*tp_info)->pt());
            meBTLTrackMatchedTPPtResvsPtMtd_->Fill(
                (*tp_info)->pt(), std::abs(track.pt() - (*tp_info)->pt()) / std::abs(trackGen.pt() - (*tp_info)->pt()));
            meBTLTrackMatchedTPDPtvsPtGen_->Fill((*tp_info)->pt(),
                                                 (trackGen.pt() - (*tp_info)->pt()) / (*tp_info)->pt());
            meBTLTrackMatchedTPDPtvsPtMtd_->Fill((*tp_info)->pt(), (track.pt() - (*tp_info)->pt()) / (*tp_info)->pt());
          }
          if (isETL && !twoETLdiscs && (std::abs(trackGen.eta()) > trackMinEtlEta_) &&
              (std::abs(trackGen.eta()) < trackMaxEtlEta_)) {
            meETLTrackMatchedTPPtResMtd_->Fill(std::abs(track.pt() - (*tp_info)->pt()) /
                                               std::abs(trackGen.pt() - (*tp_info)->pt()));
            meETLTrackMatchedTPPtRatioGen_->Fill(trackGen.pt() / (*tp_info)->pt());
            meETLTrackMatchedTPPtRatioMtd_->Fill(track.pt() / (*tp_info)->pt());
            meETLTrackMatchedTPPtResvsPtMtd_->Fill(
                (*tp_info)->pt(), std::abs(track.pt() - (*tp_info)->pt()) / std::abs(trackGen.pt() - (*tp_info)->pt()));
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
                (*tp_info)->pt(), std::abs(track.pt() - (*tp_info)->pt()) / std::abs(trackGen.pt() - (*tp_info)->pt()));
            meETLTrackMatchedTP2DPtvsPtGen_->Fill((*tp_info)->pt(),
                                                  (trackGen.pt() - (*tp_info)->pt()) / ((*tp_info)->pt()));
            meETLTrackMatchedTP2DPtvsPtMtd_->Fill((*tp_info)->pt(),
                                                  (track.pt() - (*tp_info)->pt()) / ((*tp_info)->pt()));
          }
        }
        auto simClustersRefs = tp2SimAssociationMap.find(*tp_info);
        const bool withMTD = (simClustersRefs != tp2SimAssociationMap.end());
        if (noCrack) {
          meTrackMatchedTPEffPtTot_->Fill(trackGen.pt());
          if (trkTPSelLV(**tp_info)) {
            meTrackMatchedTPEffPtTotLV_->Fill(trackGen.pt());
          }
          if (withMTD) {
            meTrackMatchedTPmtdEffPtTot_->Fill(trackGen.pt());
          }
        }
        meTrackMatchedTPEffEtaTot_->Fill(std::abs(trackGen.eta()));
        if (trkTPSelLV(**tp_info)) {
          meTrackMatchedTPEffEtaTotLV_->Fill(std::abs(trackGen.eta()));
        }
        if (withMTD) {
          meTrackMatchedTPmtdEffEtaTot_->Fill(std::abs(trackGen.eta()));
        }
        if (isBTL || isETL) {
          if (noCrack) {
            meTrackMatchedTPEffPtMtd_->Fill(trackGen.pt());
            if (isBTL || twoETLdiscs) {
              meTrackMatchedTPEffPtEtl2Mtd_->Fill(trackGen.pt());
            }
            if (withMTD) {
              meTrackMatchedTPmtdEffPtMtd_->Fill(trackGen.pt());
            }
          }
          meTrackMatchedTPEffEtaMtd_->Fill(std::abs(trackGen.eta()));
          if (isBTL || twoETLdiscs) {
            meTrackMatchedTPEffEtaEtl2Mtd_->Fill(std::abs(trackGen.eta()));
          }
          if (withMTD) {
            meTrackMatchedTPmtdEffEtaMtd_->Fill(std::abs(trackGen.eta()));
          }
        }

        // time pull and detailed extrapolation check only on tracks associated to TP from signal event
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
        double tsim = (*tp_info)->parentVertex()->position().t() * simUnit_;
        double dT(-9999.);
        double pullT(-9999.);
        if (Sigmat0Safe[trackref] != -1.) {
          dT = t0Safe[trackref] - tsim;
          pullT = dT / Sigmat0Safe[trackref];
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
          meETLTrackEffEtaTotLowPt_[0]->Fill(std::abs(trackGen.eta()));
        } else {
          meETLTrackEffEtaTotLowPt_[1]->Fill(std::abs(trackGen.eta()));
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
          meETLTrackEffEtaMtdLowPt_[0]->Fill(std::abs(trackGen.eta()));
        } else {
          meETLTrackEffEtaMtdLowPt_[1]->Fill(std::abs(trackGen.eta()));
        }
      }
      if (isETL && twoETLdiscs) {
        if (trackGen.pt() < 0.45) {
          meETLTrackEffEta2MtdLowPt_[0]->Fill(std::abs(trackGen.eta()));
        } else {
          meETLTrackEffEta2MtdLowPt_[1]->Fill(std::abs(trackGen.eta()));
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
  meBTLTrackEffEtaTot_ = ibook.book1D("TrackBTLEffEtaTot", "Eta of tracks (Tot);#eta_{RECO}", 100, -1.6, 1.6);
  meBTLTrackEffPhiTot_ = ibook.book1D("TrackBTLEffPhiTot", "Phi of tracks (Tot);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meBTLTrackEffPtTot_ = ibook.book1D("TrackBTLEffPtTot", "Pt of tracks (Tot);pt_{RECO} [GeV]", 50, 0, 10);
  meBTLTrackEffEtaMtd_ = ibook.book1D("TrackBTLEffEtaMtd", "Eta of tracks (Mtd);#eta_{RECO}", 100, -1.6, 1.6);
  meBTLTrackEffPhiMtd_ = ibook.book1D("TrackBTLEffPhiMtd", "Phi of tracks (Mtd);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meBTLTrackEffPtMtd_ = ibook.book1D("TrackBTLEffPtMtd", "Pt of tracks (Mtd);pt_{RECO} [GeV]", 50, 0, 10);
  meBTLTrackPtRes_ =
      ibook.book1D("TrackBTLPtRes", "Track pT resolution  ;pT_{Gentrack}-pT_{MTDtrack}/pT_{Gentrack} ", 100, -0.1, 0.1);
  meETLTrackRPTime_ = ibook.book1D("TrackETLRPTime", "Track t0 with respect to R.P.;t0 [ns]", 100, -1, 3);
  meETLTrackEffEtaTot_[0] =
      ibook.book1D("TrackETLEffEtaTotZneg", "Eta of tracks (Tot) (-Z);#eta_{RECO}", 100, -3.2, -1.4);
  meETLTrackEffEtaTot_[1] =
      ibook.book1D("TrackETLEffEtaTotZpos", "Eta of tracks (Tot) (+Z);#eta_{RECO}", 100, 1.4, 3.2);
  meETLTrackEffEtaTotLowPt_[0] =
      ibook.book1D("TrackETLEffEtaTotLowPt0", "Eta of tracks, 0.2 < pt < 0.45 (Tot);#eta_{RECO}", 100, 1.4, 3.2);
  meETLTrackEffEtaTotLowPt_[1] =
      ibook.book1D("TrackETLEffEtaTotLowPt1", "Eta of tracks, 0.45 < pt < 0.7 (Tot);#eta_{RECO}", 100, 1.4, 3.2);
  meETLTrackEffPhiTot_[0] =
      ibook.book1D("TrackETLEffPhiTotZneg", "Phi of tracks (Tot) (-Z);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meETLTrackEffPhiTot_[1] =
      ibook.book1D("TrackETLEffPhiTotZpos", "Phi of tracks (Tot) (+Z);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meETLTrackEffPtTot_[0] = ibook.book1D("TrackETLEffPtTotZneg", "Pt of tracks (Tot) (-Z);pt_{RECO} [GeV]", 50, 0, 10);
  meETLTrackEffPtTot_[1] = ibook.book1D("TrackETLEffPtTotZpos", "Pt of tracks (Tot) (+Z);pt_{RECO} [GeV]", 50, 0, 10);
  meETLTrackEffEtaMtd_[0] =
      ibook.book1D("TrackETLEffEtaMtdZneg", "Eta of tracks (Mtd) (-Z);#eta_{RECO}", 100, -3.2, -1.4);
  meETLTrackEffEtaMtd_[1] =
      ibook.book1D("TrackETLEffEtaMtdZpos", "Eta of tracks (Mtd) (+Z);#eta_{RECO}", 100, 1.4, 3.2);
  meETLTrackEffEtaMtdLowPt_[0] =
      ibook.book1D("TrackETLEffEtaMtdLowPt0", "Eta of tracks, 0.2 < pt < 0.45 (Mtd);#eta_{RECO}", 100, 1.4, 3.2);
  meETLTrackEffEtaMtdLowPt_[1] =
      ibook.book1D("TrackETLEffEtaMtdLowPt1", "Eta of tracks, 0.45 < pt < 0.7 (Mtd);#eta_{RECO}", 100, 1.4, 3.2);
  meETLTrackEffEta2MtdLowPt_[0] =
      ibook.book1D("TrackETLEffEta2MtdLowPt0", "Eta of tracks, 0.2 < pt < 0.45 (Mtd 2 hit);#eta_{RECO}", 100, 1.4, 3.2);
  meETLTrackEffEta2MtdLowPt_[1] =
      ibook.book1D("TrackETLEffEta2MtdLowPt1", "Eta of tracks, 0.45 < pt < 0.7 (Mtd 2 hit);#eta_{RECO}", 100, 1.4, 3.2);
  meETLTrackEffPhiMtd_[0] =
      ibook.book1D("TrackETLEffPhiMtdZneg", "Phi of tracks (Mtd) (-Z);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meETLTrackEffPhiMtd_[1] =
      ibook.book1D("TrackETLEffPhiMtdZpos", "Phi of tracks (Mtd) (+Z);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meETLTrackEffPtMtd_[0] = ibook.book1D("TrackETLEffPtMtdZneg", "Pt of tracks (Mtd) (-Z);pt_{RECO} [GeV]", 50, 0, 10);
  meETLTrackEffPtMtd_[1] = ibook.book1D("TrackETLEffPtMtdZpos", "Pt of tracks (Mtd) (+Z);pt_{RECO} [GeV]", 50, 0, 10);
  meETLTrackEffEta2Mtd_[0] =
      ibook.book1D("TrackETLEffEta2MtdZneg", "Eta of tracks (Mtd 2 hit) (-Z);#eta_{RECO}", 100, -3.2, -1.4);
  meETLTrackEffEta2Mtd_[1] =
      ibook.book1D("TrackETLEffEta2MtdZpos", "Eta of tracks (Mtd 2 hit) (+Z);#eta_{RECO}", 100, 1.4, 3.2);
  meETLTrackEffPhi2Mtd_[0] =
      ibook.book1D("TrackETLEffPhi2MtdZneg", "Phi of tracks (Mtd 2 hit) (-Z);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meETLTrackEffPhi2Mtd_[1] =
      ibook.book1D("TrackETLEffPhi2MtdZpos", "Phi of tracks (Mtd 2 hit) (+Z);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meETLTrackEffPt2Mtd_[0] =
      ibook.book1D("TrackETLEffPt2MtdZneg", "Pt of tracks (Mtd 2 hit) (-Z);pt_{RECO} [GeV]", 50, 0, 10);
  meETLTrackEffPt2Mtd_[1] =
      ibook.book1D("TrackETLEffPt2MtdZpos", "Pt of tracks (Mtd 2 hit) (+Z);pt_{RECO} [GeV]", 50, 0, 10);
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

  meTrackPtTot_ = ibook.book1D("TrackPtTot", "Pt of tracks ; track pt [GeV] ", 110, 0., 11.);
  meTrackMatchedTPEffPtTot_ =
      ibook.book1D("MatchedTPEffPtTot", "Pt of tracks matched to TP; track pt [GeV] ", 110, 0., 11.);
  meTrackMatchedTPEffPtTotLV_ =
      ibook.book1D("MatchedTPEffPtTotLV", "Pt of tracks associated to LV matched to TP; track pt [GeV] ", 110, 0., 11.);
  meTrackMatchedTPEffPtMtd_ =
      ibook.book1D("MatchedTPEffPtMtd", "Pt of tracks matched to TP with time; track pt [GeV] ", 110, 0., 11.);
  meTrackMatchedTPEffPtEtl2Mtd_ = ibook.book1D(
      "MatchedTPEffPtEtl2Mtd", "Pt of tracks matched to TP with time, 2 ETL hits; track pt [GeV] ", 110, 0., 11.);

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
  meBTLTrackMatchedTPPtRatioMtd_ = ibook.book1D("TrackMatchedTPBTLPtRatioMtd",
                                                "Pt ratio of tracks matched to TP-BTL hit  ;pT_{MTDtrack}/pT_{truth} ",
                                                100,
                                                0.9,
                                                1.1);
  meETLTrackMatchedTPPtRatioMtd_ = ibook.book1D("TrackMatchedTPETLPtRatioMtd",
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
  meBTLTrackMatchedTPPtResvsPtMtd_ = ibook.bookProfile("TrackMatchedTPBTLPtResvsPtMtd",
                                                       "Pt resolution of tracks matched to TP-BTL hit vs Pt;pT_{truth} "
                                                       "[GeV];|pT_{MTDtrack}-pT_{truth}|/|pT_{Gentrack}-pT_{truth}| ",
                                                       20,
                                                       0.7,
                                                       10.,
                                                       0.,
                                                       4.,
                                                       "s");
  meETLTrackMatchedTPPtResvsPtMtd_ = ibook.bookProfile("TrackMatchedTPETLPtResvsPtMtd",
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

  meTrackMatchedTPmtdEffPtTot_ =
      ibook.book1D("MatchedTPmtdEffPtTot", "Pt of tracks matched to TP-mtd hit; track pt [GeV] ", 110, 0., 11.);
  meTrackMatchedTPmtdEffPtMtd_ = ibook.book1D(
      "MatchedTPmtdEffPtMtd", "Pt of tracks matched to TP-mtd hit with time; track pt [GeV] ", 110, 0., 11.);

  meExtraEtaMtd_ =
      ibook.book1D("ExtraEtaMtd", "Eta of tracks associated to LV extrapolated to hits; track eta ", 66, 0., 3.3);
  meExtraEtaEtl2Mtd_ = ibook.book1D(
      "ExtraEtaEtl2Mtd", "Eta of tracks associated to LV extrapolated to hits, 2 ETL layers; track eta ", 66, 0., 3.3);

  meTrackEtaTot_ = ibook.book1D("TrackEtaTot", "Eta of tracks ; track eta ", 66, 0., 3.3);
  meTrackMatchedTPEffEtaTot_ =
      ibook.book1D("MatchedTPEffEtaTot", "Eta of tracks matched to TP; track eta ", 66, 0., 3.3);
  meTrackMatchedTPEffEtaTotLV_ =
      ibook.book1D("MatchedTPEffEtaTotLV", "Eta of tracks associated to LV matched to TP; track eta ", 66, 0., 3.3);
  meTrackMatchedTPEffEtaMtd_ =
      ibook.book1D("MatchedTPEffEtaMtd", "Eta of tracks matched to TP with time; track eta ", 66, 0., 3.3);
  meTrackMatchedTPEffEtaEtl2Mtd_ = ibook.book1D(
      "MatchedTPEffEtaEtl2Mtd", "Eta of tracks matched to TP with time, 2 ETL hits; track eta ", 66, 0., 3.3);

  meTrackMatchedTPmtdEffEtaTot_ =
      ibook.book1D("MatchedTPmtdEffEtaTot", "Eta of tracks matched to TP-mtd hit; track eta ", 66, 0., 3.3);
  meTrackMatchedTPmtdEffEtaMtd_ =
      ibook.book1D("MatchedTPmtdEffEtaMtd", "Eta of tracks matched to TP-mtd hit with time; track eta ", 66, 0., 3.3);

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
  ;
  meExtraMTDfailExtenderPt_ = ibook.book1D(
      "ExtraMTDfailExtenderPt",
      "Pt of tracks associated to LV extrapolated to MTD with no track extender match to hits; track pt [GeV] ",
      110,
      0.,
      11.);
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------

void MtdTracksValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/Tracks");
  desc.add<edm::InputTag>("inputTagG", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("inputTagT", edm::InputTag("trackExtenderWithMTD"));
  desc.add<edm::InputTag>("inputTagV", edm::InputTag("offlinePrimaryVertices4D"));
  desc.add<edm::InputTag>("inputTagH", edm::InputTag("generatorSmeared"));
  desc.add<edm::InputTag>("SimTag", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("TPtoRecoTrackAssoc", edm::InputTag("trackingParticleRecoTrackAsssociation"));
  desc.add<edm::InputTag>("tp2SimAssociationMapTag", edm::InputTag("mtdSimLayerClusterToTPAssociation"));
  desc.add<edm::InputTag>("btlRecHits", edm::InputTag("mtdRecHits", "FTLBarrel"));
  desc.add<edm::InputTag>("etlRecHits", edm::InputTag("mtdRecHits", "FTLEndcap"));
  desc.add<edm::InputTag>("tmtd", edm::InputTag("trackExtenderWithMTD:generalTracktmtd"));
  desc.add<edm::InputTag>("sigmatmtd", edm::InputTag("trackExtenderWithMTD:generalTracksigmatmtd"));
  desc.add<edm::InputTag>("t0Src", edm::InputTag("trackExtenderWithMTD:generalTrackt0"));
  desc.add<edm::InputTag>("sigmat0Src", edm::InputTag("trackExtenderWithMTD:generalTracksigmat0"));
  desc.add<edm::InputTag>("trackAssocSrc", edm::InputTag("trackExtenderWithMTD:generalTrackassoc"))
      ->setComment("Association between General and MTD Extended tracks");
  desc.add<edm::InputTag>("pathLengthSrc", edm::InputTag("trackExtenderWithMTD:generalTrackPathLength"));
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

DEFINE_FWK_MODULE(MtdTracksValidation);
