#include <numeric>
#include <string>
#include <vector>
#include <map>
#include <algorithm>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "DataFormats/Common/interface/ValidHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/isFinite.h"

#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/Math/interface/Point3D.h"

// reco track and vertex
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"

// TrackingParticle
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoVertex/PrimaryVertexProducer/interface/HITrackFilterForPVFinding.h"
// Fastjet
#include <fastjet/internal/base.hh>
#include "fastjet/PseudoJet.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/ClusterSequence.hh"
#include "fastjet/Selector.hh"

// HepPDTRecord
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

// pile-up
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

// associator
#include "SimTracker/VertexAssociation/interface/calculateVertexSharedTracks.h"

// vertexing
#include "RecoVertex/PrimaryVertexProducer/interface/TrackFilterForPVFinding.h"

// simulated vertex
#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociator.h"

// DQM
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "CLHEP/Units/PhysicalConstants.h"

// class declaration
class Primary4DVertexValidation : public DQMEDAnalyzer {
  typedef math::XYZTLorentzVector LorentzVector;

  // auxiliary class holding simulated vertices
  struct simPrimaryVertex {
    simPrimaryVertex(double x1, double y1, double z1, double t1)
        : x(x1),
          y(y1),
          z(z1),
          t(t1),
          pt(0),
          ptsq(0),
          closest_vertex_distance_z(-1.),
          nGenTrk(0),
          num_matched_reco_tracks(0),
          average_match_quality(0.0) {
      ptot.setPx(0);
      ptot.setPy(0);
      ptot.setPz(0);
      ptot.setE(0);
      p4 = LorentzVector(0, 0, 0, 0);
      r = sqrt(x * x + y * y);
    };
    double x, y, z, r, t;
    HepMC::FourVector ptot;
    LorentzVector p4;
    double pt;
    double ptsq;
    double closest_vertex_distance_z;
    int nGenTrk;
    int num_matched_reco_tracks;
    float average_match_quality;
    EncodedEventId eventId;
    TrackingVertexRef sim_vertex;
    int OriginalIndex = -1;

    unsigned int nwosmatch = 0;                    // number of rec vertices dominated by this sim evt (by wos)
    unsigned int nwntmatch = 0;                    // number of rec vertices dominated by this sim evt  (by wnt)
    std::vector<unsigned int> wos_dominated_recv;  // list of dominated rec vertices (by wos, size==nwosmatch)

    std::map<unsigned int, double> wnt;  // weighted number of tracks in recvtx (by index)
    std::map<unsigned int, double> wos;  // weight over sigma**2 in recvtx (by index)
    double sumwos = 0;                   // sum of wos in any recvtx
    double sumwnt = 0;                   // sum of weighted tracks
    unsigned int rec = NOT_MATCHED;      // best match (NO_MATCH if not matched)
    unsigned int matchQuality = 0;       // quality flag

    void addTrack(unsigned int irecv, double twos, double twt) {
      sumwnt += twt;
      if (wnt.find(irecv) == wnt.end()) {
        wnt[irecv] = twt;
      } else {
        wnt[irecv] += twt;
      }

      sumwos += twos;
      if (wos.find(irecv) == wos.end()) {
        wos[irecv] = twos;
      } else {
        wos[irecv] += twos;
      }
    }
  };

  // auxiliary class holding reconstructed vertices
  struct recoPrimaryVertex {
    recoPrimaryVertex(double x1, double y1, double z1)
        : x(x1),
          y(y1),
          z(z1),
          pt(0),
          ptsq(0),
          closest_vertex_distance_z(-1.),
          nRecoTrk(0),
          num_matched_sim_tracks(0),
          ndof(0.),
          recVtx(nullptr) {
      r = sqrt(x * x + y * y);
    };
    double x, y, z, r;
    double pt;
    double ptsq;
    double closest_vertex_distance_z;
    int nRecoTrk;
    int num_matched_sim_tracks;
    double ndof;
    const reco::Vertex* recVtx;
    reco::VertexBaseRef recVtxRef;
    int OriginalIndex = -1;

    std::map<unsigned int, double> wos;   // sim event -> wos
    std::map<unsigned int, double> wnt;   // sim event -> weighted number of truth matched tracks
    unsigned int wosmatch = NOT_MATCHED;  // index of the sim event providing the largest contribution to wos
    unsigned int wntmatch = NOT_MATCHED;  // index of the sim event providing the highest number of tracks
    double sumwos = 0;                    // total sum of wos of all truth matched tracks
    double sumwnt = 0;                    // total weighted number of truth matchted tracks
    double maxwos = 0;                    // largest wos sum from one sim event (wosmatch)
    double maxwnt = 0;                    // largest weighted number of tracks from one sim event (ntmatch)
    unsigned int maxwosnt = 0;            // number of tracks from the sim event with highest wos
    unsigned int sim = NOT_MATCHED;       // best match (NO_MATCH if not matched)
    unsigned int matchQuality = 0;        // quality flag

    bool is_real() { return (matchQuality > 0) && (matchQuality < 99); }

    bool is_fake() { return (matchQuality <= 0) || (matchQuality >= 99); }

    bool is_signal() { return (sim == 0); }

    int split_from() {
      if (is_real())
        return -1;
      if ((maxwos > 0) && (maxwos > 0.3 * sumwos))
        return wosmatch;
      return -1;
    }
    bool other_fake() { return (is_fake() && (split_from() < 0)); }

    void addTrack(unsigned int iev, double twos, double wt) {
      sumwnt += wt;
      if (wnt.find(iev) == wnt.end()) {
        wnt[iev] = wt;
      } else {
        wnt[iev] += wt;
      }

      sumwos += twos;
      if (wos.find(iev) == wos.end()) {
        wos[iev] = twos;
      } else {
        wos[iev] += twos;
      }
    }
  };

public:
  explicit Primary4DVertexValidation(const edm::ParameterSet&);
  ~Primary4DVertexValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker& i, edm::Run const&, edm::EventSetup const&) override;

private:
  void matchReco2Sim(std::vector<recoPrimaryVertex>&,
                     std::vector<simPrimaryVertex>&,
                     const edm::ValueMap<float>&,
                     const edm::ValueMap<float>&,
                     const edm::Handle<reco::BeamSpot>&);
  bool matchRecoTrack2SimSignal(const reco::TrackBaseRef&);
  std::pair<const edm::Ref<std::vector<TrackingParticle>>*, int> getMatchedTP(const reco::TrackBaseRef&,
                                                                              const TrackingVertexRef&);
  double timeFromTrueMass(double, double, double, double);
  bool select(const reco::Vertex&, int level = 0);
  void observablesFromJets(const std::vector<reco::Track>&,
                           const std::vector<double>&,
                           const std::vector<int>&,
                           const std::string&,
                           unsigned int&,
                           double&,
                           double&,
                           double&,
                           double&);
  void isParticle(const reco::TrackBaseRef&,
                  const edm::ValueMap<float>&,
                  const edm::ValueMap<float>&,
                  const edm::ValueMap<float>&,
                  const edm::ValueMap<float>&,
                  const edm::ValueMap<float>&,
                  unsigned int&,
                  bool&,
                  bool&,
                  bool&,
                  bool&);
  void getWosWnt(const reco::Vertex&,
                 const reco::TrackBaseRef&,
                 const edm::ValueMap<float>&,
                 const edm::ValueMap<float>&,
                 const edm::Handle<reco::BeamSpot>&,
                 double&,
                 double&);
  std::vector<Primary4DVertexValidation::simPrimaryVertex> getSimPVs(const edm::Handle<TrackingVertexCollection>&);
  std::vector<Primary4DVertexValidation::recoPrimaryVertex> getRecoPVs(const edm::Handle<edm::View<reco::Vertex>>&);

  void printMatchedRecoTrackInfo(const reco::Vertex&,
                                 const reco::TrackBaseRef&,
                                 const TrackingParticleRef&,
                                 const unsigned int&);
  void printSimVtxRecoVtxInfo(const struct Primary4DVertexValidation::simPrimaryVertex&,
                              const struct Primary4DVertexValidation::recoPrimaryVertex&);
  const bool trkTPSelLV(const TrackingParticle&);
  const bool trkRecSel(const reco::TrackBase&);

  // ----------member data ---------------------------

  const edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> theTTBToken;
  TrackFilterForPVFindingBase* theTrackFilter;
  const std::string folder_;
  static constexpr unsigned int NOT_MATCHED = 66666;
  static constexpr double simUnit_ = 1e9;     // sim time in s while reco time in ns
  static constexpr double c_ = 2.99792458e1;  // c in cm/ns
  static constexpr double mvaL_ = 0.5;        // MVA cuts for MVA categories
  static constexpr double mvaH_ = 0.8;
  static constexpr double selNdof_ = 4.;
  static constexpr double maxRank_ = 8.;
  static constexpr double maxTry_ = 10.;
  static constexpr double zWosMatchMax_ = 1.;
  static constexpr double etacutGEN_ = 4.;   // |eta| < 4;
  static constexpr double etacutREC_ = 3.;   // |eta| < 3;
  static constexpr double pTcut_ = 0.7;      // PT > 0.7 GeV
  static constexpr double deltaZcut_ = 0.1;  // dz separation 1 mm
  static constexpr double trackMaxBtlEta_ = 1.5;
  static constexpr double trackMinEtlEta_ = 1.6;
  static constexpr double trackMaxEtlEta_ = 3.;
  static constexpr double tol_ = 1.e-4;          // tolerance on reconstructed track time, [ns]
  static constexpr double minThrSumWnt_ = 0.01;  // min threshold for filling histograms with logarithmic scale
  static constexpr double minThrSumWos_ = 0.1;
  static constexpr double minThrSumPt_ = 0.01;
  static constexpr double minThrSumPt2_ = 1.e-3;
  static constexpr double minThrMetPt_ = 1.e-3;
  static constexpr double minThrSumPz_ = 1.e-4;
  static constexpr double rBTL_ = 110.0;
  static constexpr double zETL_ = 290.0;

  static constexpr float c_cm_ns = geant_units::operators::convertMmToCm(CLHEP::c_light);  // [mm/ns] -> [cm/ns]

  bool use_only_charged_tracks_;
  bool optionalPlots_;
  bool use3dNoTime_;
  const double minProbHeavy_;
  const double trackweightTh_;
  const double mvaTh_;
  const reco::RecoToSimCollection* r2s_;
  const reco::SimToRecoCollection* s2r_;

  edm::EDGetTokenT<reco::TrackCollection> RecTrackToken_;

  edm::EDGetTokenT<std::vector<PileupSummaryInfo>> vecPileupSummaryInfoToken_;

  edm::EDGetTokenT<reco::TrackCollection> trkToken;
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticleCollectionToken_;
  edm::EDGetTokenT<TrackingVertexCollection> trackingVertexCollectionToken_;
  edm::EDGetTokenT<reco::SimToRecoCollection> simToRecoAssociationToken_;
  edm::EDGetTokenT<reco::RecoToSimCollection> recoToSimAssociationToken_;
  edm::EDGetTokenT<reco::BeamSpot> RecBeamSpotToken_;
  edm::EDGetTokenT<edm::View<reco::Vertex>> Rec4DVerToken_;

  edm::EDGetTokenT<edm::ValueMap<int>> trackAssocToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> pathLengthToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> momentumToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> timeToken_;

  edm::EDGetTokenT<edm::ValueMap<float>> t0PidToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> sigmat0PidToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> t0SafePidToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> sigmat0SafePidToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> trackMVAQualToken_;

  edm::EDGetTokenT<edm::ValueMap<float>> tmtdToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> tofPiToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> tofKToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> tofPToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> probPiToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> probKToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> probPToken_;
  edm::ESGetToken<HepPDT::ParticleDataTable, edm::DefaultRecord> pdtToken_;

  // histogram declaration
  MonitorElement* meUnAssocTracks_;
  MonitorElement* meUnAssocTracksFake_;
  MonitorElement* meFractionUnAssocTracks_;
  MonitorElement* meFractionUnAssocTracksFake_;
  MonitorElement* meTrackEffPtTot_;
  MonitorElement* meTrackMatchedTPEffPtTot_;
  MonitorElement* meTrackMatchedTPEffPtMtd_;
  MonitorElement* meTrackEffEtaTot_;
  MonitorElement* meTrackMatchedTPEffEtaTot_;
  MonitorElement* meTrackMatchedTPEffEtaMtd_;
  MonitorElement* meTrackMatchedTPResTot_;
  MonitorElement* meTrackMatchedTPPullTot_;
  MonitorElement* meTrackResTot_;
  MonitorElement* meTrackPullTot_;
  MonitorElement* meTrackRes_[3];
  MonitorElement* meTrackPull_[3];
  MonitorElement* meTrackResMass_[3];
  MonitorElement* meTrackResMassTrue_[3];
  MonitorElement* meTrackMatchedTPZposResTot_;
  MonitorElement* meTrackZposResTot_;
  MonitorElement* meTrackZposRes_[3];
  MonitorElement* meTrack3DposRes_[3];
  MonitorElement* meTimeRes_;
  MonitorElement* meTimePull_;
  MonitorElement* meTimeSignalRes_;
  MonitorElement* meTimeSignalPull_;
  MonitorElement* mePUvsRealV_;
  MonitorElement* mePUvsFakeV_;
  MonitorElement* mePUvsOtherFakeV_;
  MonitorElement* mePUvsSplitV_;
  MonitorElement* meMatchQual_;
  MonitorElement* meDeltaZrealreal_;
  MonitorElement* meDeltaZfakefake_;
  MonitorElement* meDeltaZfakereal_;
  MonitorElement* meDeltaTrealreal_;
  MonitorElement* meDeltaTfakefake_;
  MonitorElement* meDeltaTfakereal_;
  MonitorElement* meRecoPosInSimCollection_;
  MonitorElement* meRecoPosInRecoOrigCollection_;
  MonitorElement* meSimPosInSimOrigCollection_;
  MonitorElement* meRecoPVPosSignal_;
  MonitorElement* meRecoPVPosSignalNotHighestPt_;
  MonitorElement* meRecoVtxVsLineDensity_;
  MonitorElement* meRecVerNumber_;
  MonitorElement* meRecPVZ_;
  MonitorElement* meRecPVT_;
  MonitorElement* meSimVerNumber_;
  MonitorElement* meSimPVZ_;
  MonitorElement* meSimPVT_;
  MonitorElement* meSimPVTvsZ_;

  MonitorElement* meVtxTrackMult_;
  MonitorElement* meVtxTrackMultPassNdof_;
  MonitorElement* meVtxTrackMultFailNdof_;
  MonitorElement* meVtxTrackW_;
  MonitorElement* meVtxTrackWnt_;
  MonitorElement* meVtxTrackRecLVMult_;
  MonitorElement* meVtxTrackRecLVW_;
  MonitorElement* meVtxTrackRecLVWnt_;

  MonitorElement* mePUTrackMult_;
  MonitorElement* mePUTrackRelMult_;
  MonitorElement* meFakeTrackRelMult_;
  MonitorElement* mePUTrackSumWnt_;
  MonitorElement* mePUTrackRelSumWnt_;
  MonitorElement* mePUTrackSumWos_;
  MonitorElement* mePUTrackRelSumWos_;
  MonitorElement* meSecTrackSumWos_;
  MonitorElement* meSecTrackRelSumWos_;
  MonitorElement* meFakeTrackRelSumWos_;
  MonitorElement* mePUTrackWnt_;
  MonitorElement* mePUTrackSumPt_;
  MonitorElement* mePUTrackRelSumPt_;
  MonitorElement* mePUTrackSumPt2_;
  MonitorElement* mePUTrackRelSumPt2_;

  MonitorElement* mePUTrackRelMultvsMult_;
  MonitorElement* meFakeTrackRelMultvsMult_;
  MonitorElement* mePUTrackRelSumWntvsSumWnt_;
  MonitorElement* mePUTrackRelSumWosvsSumWos_;
  MonitorElement* meFakeTrackRelSumWosvsSumWos_;
  MonitorElement* meSecTrackRelSumWosvsSumWos_;
  MonitorElement* mePUTrackRelSumPtvsSumPt_;
  MonitorElement* mePUTrackRelSumPt2vsSumPt2_;

  MonitorElement* mePUTrackRecLVMult_;
  MonitorElement* mePUTrackRecLVRelMult_;
  MonitorElement* meFakeTrackRecLVRelMult_;
  MonitorElement* mePUTrackRecLVSumWnt_;
  MonitorElement* mePUTrackRecLVRelSumWnt_;
  MonitorElement* mePUTrackRecLVSumWos_;
  MonitorElement* mePUTrackRecLVRelSumWos_;
  MonitorElement* meSecTrackRecLVSumWos_;
  MonitorElement* meSecTrackRecLVRelSumWos_;
  MonitorElement* meFakeTrackRecLVRelSumWos_;
  MonitorElement* mePUTrackRecLVWnt_;
  MonitorElement* mePUTrackRecLVSumPt_;
  MonitorElement* mePUTrackRecLVRelSumPt_;
  MonitorElement* mePUTrackRecLVSumPt2_;
  MonitorElement* mePUTrackRecLVRelSumPt2_;

  MonitorElement* mePUTrackRecLVRelMultvsMult_;
  MonitorElement* meFakeTrackRecLVRelMultvsMult_;
  MonitorElement* mePUTrackRecLVRelSumWntvsSumWnt_;
  MonitorElement* mePUTrackRecLVRelSumWosvsSumWos_;
  MonitorElement* meSecTrackRecLVRelSumWosvsSumWos_;
  MonitorElement* meFakeTrackRecLVRelSumWosvsSumWos_;
  MonitorElement* mePUTrackRecLVRelSumPtvsSumPt_;
  MonitorElement* mePUTrackRecLVRelSumPt2vsSumPt2_;

  MonitorElement* meJetsPUMult_;
  MonitorElement* meJetsPUHt_;
  MonitorElement* meJetsPUSumPt2_;
  MonitorElement* meJetsPUMetPt_;
  MonitorElement* meJetsPUSumPz_;
  MonitorElement* meJetsPURelMult_;
  MonitorElement* meJetsPURelHt_;
  MonitorElement* meJetsPURelSumPt2_;
  MonitorElement* meJetsFakeRelSumPt2_;
  MonitorElement* meJetsPURelMetPt_;
  MonitorElement* meJetsPURelSumPz_;

  MonitorElement* meJetsPURelMultvsMult_;
  MonitorElement* meJetsPURelHtvsHt_;
  MonitorElement* meJetsPURelSumPt2vsSumPt2_;
  MonitorElement* meJetsFakeRelSumPt2vsSumPt2_;
  MonitorElement* meJetsPURelMetPtvsMetPt_;
  MonitorElement* meJetsPURelSumPzvsSumPz_;

  MonitorElement* meJetsRecLVPUMult_;
  MonitorElement* meJetsRecLVPUHt_;
  MonitorElement* meJetsRecLVPUSumPt2_;
  MonitorElement* meJetsRecLVPUMetPt_;
  MonitorElement* meJetsRecLVPUSumPz_;
  MonitorElement* meJetsRecLVPURelMult_;
  MonitorElement* meJetsRecLVPURelHt_;
  MonitorElement* meJetsRecLVPURelSumPt2_;
  MonitorElement* meJetsRecLVFakeRelSumPt2_;
  MonitorElement* meJetsRecLVPURelMetPt_;
  MonitorElement* meJetsRecLVPURelSumPz_;

  MonitorElement* meJetsRecLVPURelMultvsMult_;
  MonitorElement* meJetsRecLVPURelHtvsHt_;
  MonitorElement* meJetsRecLVPURelSumPt2vsSumPt2_;
  MonitorElement* meJetsRecLVFakeRelSumPt2vsSumPt2_;
  MonitorElement* meJetsRecLVPURelMetPtvsMetPt_;
  MonitorElement* meJetsRecLVPURelSumPzvsSumPz_;

  // some tests
  MonitorElement* meTrackResLowPTot_;
  MonitorElement* meTrackResHighPTot_;
  MonitorElement* meTrackPullLowPTot_;
  MonitorElement* meTrackPullHighPTot_;

  MonitorElement* meTrackResLowP_[3];
  MonitorElement* meTrackResHighP_[3];
  MonitorElement* meTrackPullLowP_[3];
  MonitorElement* meTrackPullHighP_[3];

  MonitorElement* meTrackResMassProtons_[3];
  MonitorElement* meTrackResMassTrueProtons_[3];
  MonitorElement* meTrackResMassPions_[3];
  MonitorElement* meTrackResMassTruePions_[3];

  MonitorElement* meBarrelPIDp_;
  MonitorElement* meEndcapPIDp_;

  MonitorElement* meBarrelNoPIDtype_;
  MonitorElement* meEndcapNoPIDtype_;

  MonitorElement* meBarrelTruePiNoPID_;
  MonitorElement* meBarrelTrueKNoPID_;
  MonitorElement* meBarrelTruePNoPID_;
  MonitorElement* meEndcapTruePiNoPID_;
  MonitorElement* meEndcapTrueKNoPID_;
  MonitorElement* meEndcapTruePNoPID_;

  MonitorElement* meBarrelTruePiAsPi_;
  MonitorElement* meBarrelTruePiAsK_;
  MonitorElement* meBarrelTruePiAsP_;
  MonitorElement* meEndcapTruePiAsPi_;
  MonitorElement* meEndcapTruePiAsK_;
  MonitorElement* meEndcapTruePiAsP_;

  MonitorElement* meBarrelTrueKAsPi_;
  MonitorElement* meBarrelTrueKAsK_;
  MonitorElement* meBarrelTrueKAsP_;
  MonitorElement* meEndcapTrueKAsPi_;
  MonitorElement* meEndcapTrueKAsK_;
  MonitorElement* meEndcapTrueKAsP_;

  MonitorElement* meBarrelTruePAsPi_;
  MonitorElement* meBarrelTruePAsK_;
  MonitorElement* meBarrelTruePAsP_;
  MonitorElement* meEndcapTruePAsPi_;
  MonitorElement* meEndcapTruePAsK_;
  MonitorElement* meEndcapTruePAsP_;

  // Histograms for study of no PID tracks

  //Time residual
  MonitorElement* meTrackTimeResCorrectPID_;
  MonitorElement* meTrackTimeResWrongPID_;
  MonitorElement* meTrackTimeResNoPID_;
  MonitorElement* meNoPIDTrackTimeResNoPIDType_[3];
  MonitorElement* meTrackTimeResNoPIDtruePi_;
  MonitorElement* meTrackTimeResNoPIDtrueK_;
  MonitorElement* meTrackTimeResNoPIDtrueP_;

  //Time pull
  MonitorElement* meTrackTimePullCorrectPID_;
  MonitorElement* meTrackTimePullWrongPID_;
  MonitorElement* meTrackTimePullNoPID_;
  MonitorElement* meNoPIDTrackTimePullNoPIDType_[3];
  MonitorElement* meTrackTimePullNoPIDtruePi_;
  MonitorElement* meTrackTimePullNoPIDtrueK_;
  MonitorElement* meTrackTimePullNoPIDtrueP_;

  //Sigma
  MonitorElement* meTrackTimeSigmaCorrectPID_;
  MonitorElement* meTrackTimeSigmaWrongPID_;
  MonitorElement* meTrackTimeSigmaNoPID_;
  MonitorElement* meNoPIDTrackSigmaNoPIDType_[3];

  //MVA
  MonitorElement* meTrackMVACorrectPID_;
  MonitorElement* meTrackMVAWrongPID_;
  MonitorElement* meTrackMVANoPID_;
  MonitorElement* meNoPIDTrackMVANoPIDType_[3];
};

// constructors and destructor
Primary4DVertexValidation::Primary4DVertexValidation(const edm::ParameterSet& iConfig)
    : theTTBToken(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
      folder_(iConfig.getParameter<std::string>("folder")),
      use_only_charged_tracks_(iConfig.getParameter<bool>("useOnlyChargedTracks")),
      optionalPlots_(iConfig.getUntrackedParameter<bool>("optionalPlots")),
      use3dNoTime_(iConfig.getParameter<bool>("use3dNoTime")),
      minProbHeavy_(iConfig.getParameter<double>("minProbHeavy")),
      trackweightTh_(iConfig.getParameter<double>("trackweightTh")),
      mvaTh_(iConfig.getParameter<double>("mvaTh")),
      pdtToken_(esConsumes<HepPDT::ParticleDataTable, edm::DefaultRecord>()) {
  vecPileupSummaryInfoToken_ = consumes<std::vector<PileupSummaryInfo>>(edm::InputTag(std::string("addPileupInfo")));
  trkToken = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("TrackLabel"));
  trackingParticleCollectionToken_ =
      consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("SimTag"));
  trackingVertexCollectionToken_ = consumes<TrackingVertexCollection>(iConfig.getParameter<edm::InputTag>("SimTag"));
  simToRecoAssociationToken_ =
      consumes<reco::SimToRecoCollection>(iConfig.getParameter<edm::InputTag>("TPtoRecoTrackAssoc"));
  recoToSimAssociationToken_ =
      consumes<reco::RecoToSimCollection>(iConfig.getParameter<edm::InputTag>("TPtoRecoTrackAssoc"));
  RecTrackToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("mtdTracks"));
  RecBeamSpotToken_ = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("offlineBS"));
  Rec4DVerToken_ = consumes<edm::View<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("offline4DPV"));
  trackAssocToken_ = consumes<edm::ValueMap<int>>(iConfig.getParameter<edm::InputTag>("trackAssocSrc"));
  pathLengthToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("pathLengthSrc"));
  momentumToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("momentumSrc"));
  timeToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("timeSrc"));
  t0PidToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("t0PID"));
  sigmat0PidToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("sigmat0PID"));
  t0SafePidToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("t0SafePID"));
  sigmat0SafePidToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("sigmat0SafePID"));
  trackMVAQualToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("trackMVAQual"));
  tmtdToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("tmtd"));
  tofPiToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("tofPi"));
  tofKToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("tofK"));
  tofPToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("tofP"));
  probPiToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("probPi"));
  probKToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("probK"));
  probPToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("probP"));
  std::string trackSelectionAlgorithm =
      iConfig.getParameter<edm::ParameterSet>("TkFilterParameters").getParameter<std::string>("algorithm");
  if (trackSelectionAlgorithm == "filter") {
    theTrackFilter = new TrackFilterForPVFinding(iConfig.getParameter<edm::ParameterSet>("TkFilterParameters"));
  } else if (trackSelectionAlgorithm == "filterWithThreshold") {
    theTrackFilter = new HITrackFilterForPVFinding(iConfig.getParameter<edm::ParameterSet>("TkFilterParameters"));
  } else {
    edm::LogWarning("Primary4DVertexValidation")
        << "unknown track selection algorithm: " + trackSelectionAlgorithm << std::endl;
  }
}

Primary4DVertexValidation::~Primary4DVertexValidation() {}

//
// member functions
//
void Primary4DVertexValidation::bookHistograms(DQMStore::IBooker& ibook,
                                               edm::Run const& iRun,
                                               edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(folder_);
  // --- histograms booking
  meUnAssocTracks_ = ibook.book1D("UnAssocTracks", "Log10(Unassociated tracks)", 160, 0.5, 4.5);
  meUnAssocTracksFake_ = ibook.book1D("UnAssocTracksFake", "Log10(Unassociated fake tracks)", 160, 0.5, 4.5);
  meFractionUnAssocTracks_ = ibook.book1D("FractionUnAssocTracks", "Fraction Unassociated tracks", 160, 0.0, 1.);
  meFractionUnAssocTracksFake_ =
      ibook.book1D("FractionUnAssocTracksFake", "Fraction Unassociated fake tracks", 160, 0.0, 1.);
  meTrackEffPtTot_ = ibook.book1D("EffPtTot", "Pt of tracks associated to LV; track pt [GeV] ", 110, 0., 11.);
  meTrackEffEtaTot_ = ibook.book1D("EffEtaTot", "Eta of tracks associated to LV; track eta ", 66, 0., 3.3);
  meTrackMatchedTPEffPtTot_ =
      ibook.book1D("MatchedTPEffPtTot", "Pt of tracks associated to LV matched to TP; track pt [GeV] ", 110, 0., 11.);
  meTrackMatchedTPEffPtMtd_ = ibook.book1D(
      "MatchedTPEffPtMtd", "Pt of tracks associated to LV matched to TP with time; track pt [GeV] ", 110, 0., 11.);
  meTrackMatchedTPEffEtaTot_ =
      ibook.book1D("MatchedTPEffEtaTot", "Eta of tracks associated to LV matched to TP; track eta ", 66, 0., 3.3);
  meTrackMatchedTPEffEtaMtd_ = ibook.book1D(
      "MatchedTPEffEtaMtd", "Eta of tracks associated to LV matched to TP with time; track eta ", 66, 0., 3.3);
  meTrackMatchedTPResTot_ =
      ibook.book1D("MatchedTPTrackRes",
                   "t_{rec} - t_{sim} for tracks associated to LV matched to TP; t_{rec} - t_{sim} [ns] ",
                   120,
                   -0.15,
                   0.15);
  meTrackResTot_ = ibook.book1D("TrackRes", "t_{rec} - t_{sim} for tracks; t_{rec} - t_{sim} [ns] ", 120, -0.15, 0.15);
  meTrackRes_[0] = ibook.book1D(
      "TrackRes-LowMVA", "t_{rec} - t_{sim} for tracks with MVA < 0.5; t_{rec} - t_{sim} [ns] ", 100, -1., 1.);
  meTrackRes_[1] = ibook.book1D(
      "TrackRes-MediumMVA", "t_{rec} - t_{sim} for tracks with 0.5 < MVA < 0.8; t_{rec} - t_{sim} [ns] ", 100, -1., 1.);
  meTrackRes_[2] = ibook.book1D(
      "TrackRes-HighMVA", "t_{rec} - t_{sim} for tracks with MVA > 0.8; t_{rec} - t_{sim} [ns] ", 100, -1., 1.);
  if (optionalPlots_) {
    meTrackResMass_[0] = ibook.book1D(
        "TrackResMass-LowMVA", "t_{rec} - t_{est} for tracks with MVA < 0.5; t_{rec} - t_{est} [ns] ", 100, -1., 1.);
    meTrackResMass_[1] = ibook.book1D("TrackResMass-MediumMVA",
                                      "t_{rec} - t_{est} for tracks with 0.5 < MVA < 0.8; t_{rec} - t_{est} [ns] ",
                                      100,
                                      -1.,
                                      1.);
    meTrackResMass_[2] = ibook.book1D(
        "TrackResMass-HighMVA", "t_{rec} - t_{est} for tracks with MVA > 0.8; t_{rec} - t_{est} [ns] ", 100, -1., 1.);
    meTrackResMassTrue_[0] = ibook.book1D(
        "TrackResMassTrue-LowMVA", "t_{est} - t_{sim} for tracks with MVA < 0.5; t_{est} - t_{sim} [ns] ", 100, -1., 1.);
    meTrackResMassTrue_[1] = ibook.book1D("TrackResMassTrue-MediumMVA",
                                          "t_{est} - t_{sim} for tracks with 0.5 < MVA < 0.8; t_{est} - t_{sim} [ns] ",
                                          100,
                                          -1.,
                                          1.);
    meTrackResMassTrue_[2] = ibook.book1D("TrackResMassTrue-HighMVA",
                                          "t_{est} - t_{sim} for tracks with MVA > 0.8; t_{est} - t_{sim} [ns] ",
                                          100,
                                          -1.,
                                          1.);
  }
  meTrackMatchedTPPullTot_ = ibook.book1D(
      "MatchedTPTrackPull", "Pull for tracks associated to LV matched to TP; (t_{rec}-t_{sim})/#sigma_{t}", 50, -5., 5.);
  meTrackPullTot_ = ibook.book1D("TrackPull", "Pull for tracks; (t_{rec}-t_{sim})/#sigma_{t}", 100, -10., 10.);
  meTrackPull_[0] =
      ibook.book1D("TrackPull-LowMVA", "Pull for tracks with MVA < 0.5; (t_{rec}-t_{sim})/#sigma_{t}", 100, -10., 10.);
  meTrackPull_[1] = ibook.book1D(
      "TrackPull-MediumMVA", "Pull for tracks with 0.5 < MVA < 0.8; (t_{rec}-t_{sim})/#sigma_{t}", 100, -10., 10.);
  meTrackPull_[2] =
      ibook.book1D("TrackPull-HighMVA", "Pull for tracks with MVA > 0.8; (t_{rec}-t_{sim})/#sigma_{t}", 100, -10., 10.);
  meTrackMatchedTPZposResTot_ =
      ibook.book1D("MatchedTPTrackZposResTot",
                   "Z_{PCA} - Z_{sim} for tracks associated to LV matched to TP;Z_{PCA} - Z_{sim} [cm] ",
                   50,
                   -0.1,
                   0.1);
  meTrackZposResTot_ =
      ibook.book1D("TrackZposResTot", "Z_{PCA} - Z_{sim} for tracks;Z_{PCA} - Z_{sim} [cm] ", 50, -0.5, 0.5);
  meTrackZposRes_[0] = ibook.book1D(
      "TrackZposRes-LowMVA", "Z_{PCA} - Z_{sim} for tracks with MVA < 0.5;Z_{PCA} - Z_{sim} [cm] ", 50, -0.5, 0.5);
  meTrackZposRes_[1] = ibook.book1D("TrackZposRes-MediumMVA",
                                    "Z_{PCA} - Z_{sim} for tracks with 0.5 < MVA < 0.8 ;Z_{PCA} - Z_{sim} [cm] ",
                                    50,
                                    -0.5,
                                    0.5);
  meTrackZposRes_[2] = ibook.book1D(
      "TrackZposRes-HighMVA", "Z_{PCA} - Z_{sim} for tracks with MVA > 0.8 ;Z_{PCA} - Z_{sim} [cm] ", 50, -0.5, 0.5);
  meTrack3DposRes_[0] =
      ibook.book1D("Track3DposRes-LowMVA",
                   "3dPos_{PCA} - 3dPos_{sim} for tracks with MVA < 0.5 ;3dPos_{PCA} - 3dPos_{sim} [cm] ",
                   50,
                   -0.5,
                   0.5);
  meTrack3DposRes_[1] =
      ibook.book1D("Track3DposRes-MediumMVA",
                   "3dPos_{PCA} - 3dPos_{sim} for tracks with 0.5 < MVA < 0.8 ;3dPos_{PCA} - 3dPos_{sim} [cm] ",
                   50,
                   -0.5,
                   0.5);
  meTrack3DposRes_[2] =
      ibook.book1D("Track3DposRes-HighMVA",
                   "3dPos_{PCA} - 3dPos_{sim} for tracks with MVA > 0.8;3dPos_{PCA} - 3dPos_{sim} [cm] ",
                   50,
                   -0.5,
                   0.5);
  meTimeRes_ = ibook.book1D("TimeRes", "t_{rec} - t_{sim} ;t_{rec} - t_{sim} [ns] ", 100, -0.2, 0.2);
  meTimePull_ = ibook.book1D("TimePull", "Pull; t_{rec} - t_{sim}/#sigma_{t rec}", 100, -10., 10.);
  meTimeSignalRes_ =
      ibook.book1D("TimeSignalRes", "t_{rec} - t_{sim} for signal ;t_{rec} - t_{sim} [ns] ", 50, -0.1, 0.1);
  meTimeSignalPull_ =
      ibook.book1D("TimeSignalPull", "Pull for signal; t_{rec} - t_{sim}/#sigma_{t rec}", 100, -10., 10.);
  mePUvsRealV_ =
      ibook.bookProfile("PUvsReal", "#PU vertices vs #real matched vertices;#PU;#real ", 100, 0, 300, 100, 0, 200);
  mePUvsFakeV_ =
      ibook.bookProfile("PUvsFake", "#PU vertices vs #fake matched vertices;#PU;#fake ", 100, 0, 300, 100, 0, 20);
  mePUvsOtherFakeV_ = ibook.bookProfile(
      "PUvsOtherFake", "#PU vertices vs #other fake matched vertices;#PU;#other fake ", 100, 0, 300, 100, 0, 20);
  mePUvsSplitV_ =
      ibook.bookProfile("PUvsSplit", "#PU vertices vs #split matched vertices;#PU;#split ", 100, 0, 300, 100, 0, 20);
  meMatchQual_ = ibook.book1D("MatchQuality", "RECO-SIM vertex match quality; ", 8, 0, 8.);
  meDeltaZrealreal_ = ibook.book1D("DeltaZrealreal", "#Delta Z real-real; |#Delta Z (r-r)| [cm]", 100, 0, 0.5);
  meDeltaZfakefake_ = ibook.book1D("DeltaZfakefake", "#Delta Z fake-fake; |#Delta Z (f-f)| [cm]", 100, 0, 0.5);
  meDeltaZfakereal_ = ibook.book1D("DeltaZfakereal", "#Delta Z fake-real; |#Delta Z (f-r)| [cm]", 100, 0, 0.5);
  meDeltaTrealreal_ = ibook.book1D("DeltaTrealreal", "#Delta T real-real; |#Delta T (r-r)| [sigma]", 60, 0., 30.);
  meDeltaTfakefake_ = ibook.book1D("DeltaTfakefake", "#Delta T fake-fake; |#Delta T (f-f)| [sigma]", 60, 0., 30.);
  meDeltaTfakereal_ = ibook.book1D("DeltaTfakereal", "#Delta T fake-real; |#Delta T (f-r)| [sigma]", 60, 0., 30.);
  if (optionalPlots_) {
    meRecoPosInSimCollection_ = ibook.book1D(
        "RecoPosInSimCollection", "Sim signal vertex index associated to Reco signal vertex; Sim PV index", 200, 0, 200);
    meRecoPosInRecoOrigCollection_ =
        ibook.book1D("RecoPosInRecoOrigCollection", "Reco signal index in OrigCollection; Reco index", 200, 0, 200);
    meSimPosInSimOrigCollection_ =
        ibook.book1D("SimPosInSimOrigCollection", "Sim signal index in OrigCollection; Sim index", 200, 0, 200);
  }
  meRecoPVPosSignal_ =
      ibook.book1D("RecoPVPosSignal", "Position in reco collection of PV associated to sim signal", 20, 0, 20);
  meRecoPVPosSignalNotHighestPt_ =
      ibook.book1D("RecoPVPosSignalNotHighestPt",
                   "Position in reco collection of PV associated to sim signal not highest Pt",
                   20,
                   0,
                   20);
  meRecVerNumber_ = ibook.book1D("RecVerNumber", "RECO Vertex Number: Number of vertices", 50, 0, 250);
  meSimVerNumber_ = ibook.book1D("SimVerNumber", "SIM Vertex Number: Number of vertices", 50, 0, 250);
  meRecPVZ_ = ibook.book1D("recPVZ", "#Rec vertices/10 mm", 30, -15., 15.);
  meRecPVT_ = ibook.book1D("recPVT", "#Rec vertices/50 ps", 30, -0.75, 0.75);
  meSimPVZ_ = ibook.book1D("simPVZ", "#Sim vertices/10 mm", 30, -15., 15.);
  meSimPVT_ = ibook.book1D("simPVT", "#Sim vertices/50 ps", 30, -0.75, 0.75);
  meSimPVTvsZ_ = ibook.bookProfile("simPVTvsZ", "PV Time vs Z", 30, -15., 15., 30, -0.75, 0.75);

  meVtxTrackMult_ = ibook.book1D("VtxTrackMult", "Log10(Vertex track multiplicity)", 80, 0.5, 2.5);
  meVtxTrackMultPassNdof_ =
      ibook.book1D("VtxTrackMultPassNdof", "Log10(Vertex track multiplicity for ndof>4)", 80, 0.5, 2.5);
  meVtxTrackMultFailNdof_ = ibook.book1D("VtxTrackMultFailNdof", "Vertex track multiplicity for ndof<4", 10, 0., 10.);
  meVtxTrackW_ = ibook.book1D("VtxTrackW", "Vertex track weight (all)", 50, 0., 1.);
  meVtxTrackWnt_ = ibook.book1D("VtxTrackWnt", "Vertex track Wnt", 50, 0., 1.);
  meVtxTrackRecLVMult_ =
      ibook.book1D("VtxTrackRecLVMult", "Log10(Vertex track multiplicity) for matched LV", 80, 0.5, 2.5);
  meVtxTrackRecLVW_ = ibook.book1D("VtxTrackRecLVW", "Vertex track weight for matched LV (all)", 50, 0., 1.);
  meVtxTrackRecLVWnt_ = ibook.book1D("VtxTrackRecLVWnt", "Vertex track Wnt for matched LV", 50, 0., 1.);

  mePUTrackRelMult_ = ibook.book1D(
      "PUTrackRelMult", "Relative multiplicity of PU tracks for matched vertices; #PUTrks/#Trks", 50, 0., 1.);
  meFakeTrackRelMult_ = ibook.book1D(
      "FakeTrackRelMult", "Relative multiplicity of fake tracks for matched vertices; #fakeTrks/#Trks", 50, 0., 1.);
  mePUTrackRelSumWnt_ =
      ibook.book1D("PUTrackRelSumWnt",
                   "Relative Sum of wnt of PU tracks for matched vertices; PUSumW*min(Pt, 1.)/SumW*min(Pt, 1.)",
                   50,
                   0.,
                   1.);
  mePUTrackRelSumWos_ = ibook.book1D(
      "PUTrackRelSumWos", "Relative Sum of wos of PU tracks for matched vertices; PUSumWos/SumWos", 50, 0., 1.);
  meSecTrackRelSumWos_ =
      ibook.book1D("SecTrackRelSumWos",
                   "Relative Sum of wos of tracks from secondary vtx for matched vertices; SecSumWos/SumWos",
                   50,
                   0.,
                   1.);
  meFakeTrackRelSumWos_ = ibook.book1D(
      "FakeTrackRelSumWos", "Relative Sum of wos of fake tracks for matched vertices; FakeSumWos/SumWos", 50, 0., 1.);
  mePUTrackRelSumPt_ = ibook.book1D(
      "PUTrackRelSumPt", "Relative Sum of Pt of PU tracks for matched vertices; PUSumPt/SumPt", 50, 0., 1.);
  mePUTrackRelSumPt2_ = ibook.book1D(
      "PUTrackRelSumPt2", "Relative Sum of Pt2 for PU tracks for matched vertices; PUSumPt2/SumPt2", 50, 0., 1.);
  mePUTrackRecLVRelMult_ = ibook.book1D(
      "PUTrackRecLVRelMult", "Relative multiplicity of PU tracks for matched LV; #PUTrks/#Trks", 50, 0., 1.);
  meFakeTrackRecLVRelMult_ = ibook.book1D(
      "FakeTrackRecLVRelMult", "Relative multiplicity of fake tracks for matched LV; #FakeTrks/#Trks", 50, 0., 1.);
  mePUTrackRecLVRelSumWnt_ =
      ibook.book1D("PUTrackRecLVRelSumWnt",
                   "Relative Sum of Wnt of PU tracks for matched LV; PUSumW*min(Pt, 1.)/SumW*min(Pt, 1.)",
                   50,
                   0.,
                   1.);
  mePUTrackRecLVRelSumWos_ = ibook.book1D(
      "PUTrackRecLVRelSumWos", "Relative Sum of Wos of PU tracks for matched LV; PUSumWos/SumWos", 50, 0., 1.);
  meSecTrackRecLVRelSumWos_ =
      ibook.book1D("SecTrackRecLVRelSumWos",
                   "Relative Sum of wos of tracks from secondary vtx for matched LV; SecSumWos/SumWos",
                   50,
                   0.,
                   1.);
  meFakeTrackRecLVRelSumWos_ = ibook.book1D(
      "FakeTrackRecLVRelSumWos", "Relative Sum of wos of fake tracks for matched LV; FakeSumWos/SumWos", 50, 0., 1.);
  mePUTrackRecLVRelSumPt_ =
      ibook.book1D("PUTrackRecLVRelSumPt", "Relative Sum of Pt of PU tracks for matched LV; PUSumPt/SumPt", 50, 0., 1.);
  mePUTrackRecLVRelSumPt2_ = ibook.book1D(
      "PUTrackRecLVRelSumPt2", "Relative Sum of Pt2 of PU tracks for matched LV; PUSumPt2/SumPt2", 50, 0., 1.);

  if (optionalPlots_) {
    mePUTrackMult_ = ibook.book1D("PUTrackMult", "Number of PU tracks for matched vertices; #PUTrks", 50, 0., 100.);
    mePUTrackWnt_ = ibook.book1D("PUTrackWnt", "Wnt of PU tracks for matched vertices; PUTrkW*min(Pt, 1.)", 50, 0., 1.);
    mePUTrackSumWnt_ = ibook.book1D(
        "PUTrackSumWnt", "Sum of wnt of PU tracks for matched vertices; log10(PUSumW*min(Pt, 1.))", 50, -2., 3.);
    mePUTrackSumWos_ =
        ibook.book1D("PUTrackSumWos", "Sum of wos of PU tracks for matched vertices; log10(PUSumWos)", 50, -1., 7.);
    meSecTrackSumWos_ = ibook.book1D(
        "SecTrackSumWos", "Sum of wos of tracks from secondary vtx for matched vertices; log10(SecSumWos)", 50, -1., 7.);
    mePUTrackSumPt_ =
        ibook.book1D("PUTrackSumPt", "Sum of Pt of PU tracks for matched vertices; log10(PUSumPt)", 50, -2., 3.);
    mePUTrackSumPt2_ =
        ibook.book1D("PUTrackSumPt2", "Sum of Pt2 of PU tracks for matched vertices; log10(PUSumPt2)", 50, -3., 3.);

    mePUTrackRelMultvsMult_ =
        ibook.bookProfile("PUTrackRelMultvsMult",
                          "Relative PU multiplicity vs Number of tracks for matched vertices; #Trks; #PUTrks/#Trks",
                          50,
                          0.,
                          200.,
                          0.,
                          1.,
                          "s");
    meFakeTrackRelMultvsMult_ = ibook.bookProfile(
        "FakeTrackRelMultvsMult",
        "Relative multiplicity of fake tracks vs Number of tracks for matched vertices; #Trks; #FakeTrks/#Trks",
        50,
        0.,
        200.,
        0.,
        1.,
        "s");
    mePUTrackRelSumWntvsSumWnt_ =
        ibook.bookProfile("PUTrackRelSumWntvsSumWnt",
                          "Relative PU Sum of Wnt vs Sum of Wnt of tracks for matched vertices; log10(SumW*min(Pt, "
                          "1.)); PUSumW*min(Pt, 1.)/SumW*min(Pt, 1.)",
                          50,
                          0.,
                          2.5,
                          0.,
                          1.,
                          "s");
    mePUTrackRelSumWosvsSumWos_ = ibook.bookProfile(
        "PUTrackRelSumWosvsSumWos",
        "Relative PU Sum of Wos vs Sum of Wos of tracks for matched vertices; log10(SumWos); PUSumWos/SumWos",
        50,
        2.5,
        7.,
        0.,
        1.,
        "s");
    meSecTrackRelSumWosvsSumWos_ = ibook.bookProfile("SecTrackRelSumWosvsSumWos",
                                                     "Relative Sum of Wos of tracks from secondary vtx vs Sum of Wos "
                                                     "of tracks for matched vertices; log10(SumWos); SecSumWos/SumWos",
                                                     50,
                                                     2.,
                                                     7.,
                                                     0.,
                                                     1.,
                                                     "s");
    meFakeTrackRelSumWosvsSumWos_ = ibook.bookProfile("FakeTrackRelSumWosvsSumWos",
                                                      "Relative Sum of Wos of fake tracks vs Sum of Wos of tracks for "
                                                      "matched vertices; log10(SumWos); FakeSumWos/SumWos",
                                                      50,
                                                      2.5,
                                                      7.5,
                                                      0.,
                                                      1.,
                                                      "s");
    mePUTrackRelSumPtvsSumPt_ = ibook.bookProfile(
        "PUTrackRelSumPtvsSumPt",
        "Relative PU Sum of Pt vs Sum of Pt of tracks for matched vertices; log10(SumPt); PUSumPt/SumPt",
        50,
        0.,
        3.,
        0.,
        1.,
        "s");
    mePUTrackRelSumPt2vsSumPt2_ = ibook.bookProfile(
        "PUTrackRelSumPt2vsSumPt2",
        "Relative PU Sum of Pt2 vs Sum of Pt2 of tracks for matched vertices; log10(SumPt2); PUSumPt2/SumPt2",
        50,
        0.,
        4.,
        0.,
        1.,
        "s");

    mePUTrackRecLVMult_ = ibook.book1D("PUTrackRecLVMult", "Number of PU tracks for matched LV; #PUTrks", 50, 0., 100.);
    mePUTrackRecLVWnt_ =
        ibook.book1D("PUTrackRecLVWnt", "Wnt of PU tracks for matched LV; PUTrkW*min(Pt, 1.)", 50, 0., 1.);
    mePUTrackRecLVSumWnt_ = ibook.book1D(
        "PUTrackRecLVSumWnt", "Sum of wnt of PU tracks for matched LV; log10(PUSumW*min(Pt, 1.))", 50, -2., 3.);
    mePUTrackRecLVSumWos_ =
        ibook.book1D("PUTrackRecLVSumWos", "Sum of wos of PU tracks for matched LV; log10(PUSumWos)", 50, -1., 7.);
    meSecTrackRecLVSumWos_ = ibook.book1D(
        "SecTrackRecLVSumWos", "Sum of wos of tracks from secondary vtx for matched LV; log10(SecSumWos)", 50, -1., 7.);
    mePUTrackRecLVSumPt_ =
        ibook.book1D("PUTrackRecLVSumPt", "Sum of Pt of PU tracks for matched LV; log10(PUSumPt)", 50, -2., 3.);
    mePUTrackRecLVSumPt2_ =
        ibook.book1D("PUTrackRecLVSumPt2", "Sum of Pt2 of PU tracks for matched LV; log10(PUSumPt2)", 50, -3., 3.);

    mePUTrackRecLVRelMultvsMult_ =
        ibook.bookProfile("PUTrackRecLVRelMultvsMult",
                          "Relative PU multiplicity vs Number of tracks for matched LV; #Trks; #PUTrks/#Trks",
                          50,
                          0.,
                          200.,
                          0.,
                          1.,
                          "s");
    meFakeTrackRecLVRelMultvsMult_ = ibook.bookProfile(
        "FakeTrackRecLVRelMultvsMult",
        "Relative multiplicity of fake tracks  vs Number of tracks for matched LV; #Trks; #FakeTrks/#Trks",
        50,
        0.,
        200.,
        0.,
        1.,
        "s");
    mePUTrackRecLVRelSumWntvsSumWnt_ =
        ibook.bookProfile("PUTrackRecLVRelSumWntvsSumWnt",
                          "Relative PU Sum of Wnt vs Sum of Wnt of tracks for matched LV; log10(SumW*min(Pt, 1.)); "
                          "PUSumW*min(Pt, 1.)/SumW*min(Pt, 1.)",
                          50,
                          1.,
                          2.3,
                          0.,
                          1.,
                          "s");
    mePUTrackRecLVRelSumWosvsSumWos_ = ibook.bookProfile(
        "PUTrackRecLVRelSumWosvsSumWos",
        "Relative PU Sum of Wos vs Sum of Wos of tracks for matched vertices; log10(SumWos); PUSumWos/SumWos",
        50,
        5.5,
        7.,
        0.,
        1.,
        "s");
    meSecTrackRecLVRelSumWosvsSumWos_ =
        ibook.bookProfile("SecTrackRecLVRelSumWosvsSumWos",
                          "Relative Sum of Wos of tracks from secondary vtx vs Sum of Wos of tracks for matched "
                          "vertices; log10(SumWos); SecSumWos/SumWos",
                          50,
                          5.5,
                          7.,
                          0.,
                          1.,
                          "s");
    meFakeTrackRecLVRelSumWosvsSumWos_ = ibook.bookProfile("FakeTrackRecLVRelSumWosvsSumWos",
                                                           "Relative Sum of Wos of fake tracks vs Sum of Wos of tracks "
                                                           "for matched vertices; log10(SumWos); FakeSumWos/SumWos",
                                                           50,
                                                           5.5,
                                                           7.,
                                                           0.,
                                                           1.,
                                                           "s");
    mePUTrackRecLVRelSumPtvsSumPt_ =
        ibook.bookProfile("PUTrackRecLVRelSumPtvsSumPt",
                          "Relative PU Sum of Pt vs Sum of Pt of tracks for matched LV; log10(SumPt); PUSumPt/SumPt",
                          50,
                          1.4,
                          3.,
                          0.,
                          1.,
                          "s");
    mePUTrackRecLVRelSumPt2vsSumPt2_ =
        ibook.bookProfile("PUTrackRecLVRelSumPt2vsSumPt2",
                          "Relative PU Sum of Pt2 vs Sum of tracks for matched LV; log10(SumPt2); PUSumPt2/SumPt2",
                          50,
                          2.,
                          4.,
                          0.,
                          1.,
                          "s");
  }

  meJetsPURelMult_ =
      ibook.book1D("JetsPURelMult",
                   "Relative contribution of PU to jet multiplicity for matched vertices; (#Jets-#JetsNoPU)/#Jets",
                   50,
                   0.,
                   1.);
  meJetsPURelHt_ =
      ibook.book1D("JetsPURelHt",
                   "Relative contribution of PU to scalar sum of Et of jets for matched vertices; (Ht-HtNoPU)/HT",
                   50,
                   0.,
                   1.);
  meJetsPURelSumPt2_ =
      ibook.book1D("JetsPURelSumPt2",
                   "Relative contribution of PU to sum of Pt2 of jets for matched vertices; (SumPt2-SumPt2NoPU)/SumPt2",
                   50,
                   0.,
                   1.);
  meJetsFakeRelSumPt2_ = ibook.book1D(
      "JetsFakeRelSumPt2",
      "Relative contribution of fake tracks to sum of Pt2 of jets for matched vertices; (SumPt2-SumPt2NoFake)/SumPt2",
      50,
      0.,
      1.);
  meJetsPURelMetPt_ =
      ibook.book1D("JetsPURelMetPt",
                   "Relative contribution of PU to Missing Transverse Energy for matched vertices; (Met-MetNoPU)/Met",
                   50,
                   -1.,
                   1.);
  meJetsPURelSumPz_ =
      ibook.book1D("JetsPURelSumPz",
                   "Relative contribution of PU to sum of Pz of jets for matched vertices; (SumPz-SumPzNoPU)/SumPz",
                   50,
                   -1.,
                   1.);

  meJetsRecLVPURelMult_ =
      ibook.book1D("JetsRecLVPURelMult",
                   "Relative contribution of PU to jet multiplicity for matched LV; (#Jets-#JetsNoPU)/#Jets",
                   50,
                   0.,
                   1.);
  meJetsRecLVPURelHt_ =
      ibook.book1D("JetsRecLVPURelHt",
                   "Relative contribution of PU to scalar sum of Et of jets for matched LV; (Ht-HtNoPU)/HT",
                   50,
                   0.,
                   1.);
  meJetsRecLVPURelSumPt2_ =
      ibook.book1D("JetsRecLVPURelSumPt2",
                   "Relative contribution of PU to sum of Pt2 of jets for matched LV; (SumPt2-SumPt2NoPU)/SumPt2",
                   50,
                   0.,
                   1.);
  meJetsRecLVFakeRelSumPt2_ = ibook.book1D(
      "JetsRecLVFakeRelSumPt2",
      "Relative contribution of fake tracks to sum of Pt2 of jets for matched LV; (SumPt2-SumPt2NoFake)/SumPt2",
      50,
      0.,
      1.);
  meJetsRecLVPURelMetPt_ =
      ibook.book1D("JetsRecLVPURelMetPt",
                   "Relative contribution of PU to Missing Transverse Energy for matched LV; (Met-MetNoPU)/Met",
                   50,
                   -1.,
                   1.);
  meJetsRecLVPURelSumPz_ =
      ibook.book1D("JetsRecLVPURelSumPz",
                   "Relative contribution of PU to sum of Pz of jets for matched LV; (SumPz-SumPzNoPU)/SumPz",
                   50,
                   -1.,
                   1.);

  if (optionalPlots_) {
    meJetsPUMult_ = ibook.book1D(
        "JetsPUMult", "Contribution of PU to jet multiplicity for matched vertices; #Jets-#JetsNoPU", 50, 0., 100.);
    meJetsPUHt_ = ibook.book1D("JetsPUHt",
                               "Contribution of PU to scalar sum of Et of jets for matched vertices; log10(Ht-HtNoPU)",
                               50,
                               -2.,
                               3.);
    meJetsPUSumPt2_ =
        ibook.book1D("JetsPUSumPt2",
                     "Contribution of PU to sum of Pt2 of jets for matched vertices; log10(sumPt2-SumPt2NoPU)",
                     50,
                     -3.,
                     3.);
    meJetsPUMetPt_ =
        ibook.book1D("JetsPUMetPt",
                     "Contribution of PU to Missing Transverse Energy for matched vertices; log10(Met-MetNoPU)",
                     50,
                     -3.,
                     2.);
    meJetsPUSumPz_ =
        ibook.book1D("JetsPUSumPz",
                     "Contribution of PU to sum of Pz of jets for matched vertices; log10(abs(SumPz-SumPzNoPU))",
                     50,
                     -4.,
                     3.);

    meJetsPURelMultvsMult_ = ibook.bookProfile("JetsPURelMultvsMult",
                                               "Relative contribution of PU to jet multiplicity vs number of jets for "
                                               "matched vertices; #Jets; (#Jets-#JetsNoPU)/#Jets",
                                               50,
                                               0.,
                                               120.,
                                               0.,
                                               1.,
                                               "s");
    meJetsPURelHtvsHt_ = ibook.bookProfile("JetsPURelHtvsHt",
                                           "Relative contribution of PU to scalar sum of Et of jets vs scalar sum of "
                                           "Et for matched vertices; log10(Ht); (Ht-HtNoPU)/HT",
                                           50,
                                           0.,
                                           3.,
                                           0.,
                                           1.,
                                           "s");
    meJetsPURelSumPt2vsSumPt2_ = ibook.bookProfile("JetsPURelSumPt2vsSumPt2",
                                                   "Relative contribution of PU to sum of Pt2 of jets vs sum of Pt2 "
                                                   "for matched vertices; log10(SumPt2); (SumPt2-SumPt2NoPU)/SumPt2",
                                                   50,
                                                   -1.,
                                                   4.,
                                                   0.,
                                                   1.,
                                                   "s");
    meJetsFakeRelSumPt2vsSumPt2_ =
        ibook.bookProfile("JetsFakeRelSumPt2vsSumPt2",
                          "Relative contribution of fake tracks to sum of Pt2 of jets vs sum of Pt2 for matched "
                          "vertices; log10(SumPt2); (SumPt2-SumPt2NoFake)/SumPt2",
                          50,
                          -1.,
                          4.,
                          0.,
                          1.,
                          "s");
    meJetsPURelMetPtvsMetPt_ = ibook.bookProfile("JetsPURelMetPtvsMetPt",
                                                 "Relative contribution of PU to Missing Transverse Energy vs MET for "
                                                 "matched vertices; log10(Met); (Met-MetNoPU)/Met",
                                                 50,
                                                 -1.,
                                                 2.,
                                                 -1.,
                                                 1.,
                                                 "s");
    meJetsPURelSumPzvsSumPz_ = ibook.bookProfile("JetsPURelSumPzvsSumPz",
                                                 "Relative contribution of PU to sum of Pz of jets vs Sum of Pz for "
                                                 "matched vertices; log10(abs SumPz); (SumPz-SumPzNoPU)/SumPz",
                                                 50,
                                                 -2.,
                                                 3.,
                                                 -1.,
                                                 1.,
                                                 "s");

    meJetsRecLVPUMult_ = ibook.book1D(
        "JetsRecLVPUMult", "Contribution of PU to jet multiplicity for matched LV; #Jets-#JetsNoPU", 50, 0., 100.);
    meJetsRecLVPUHt_ = ibook.book1D(
        "JetsRecLVPUHt", "Contribution of PU to scalar sum of Et of jets for matched LV; log10(Ht-HtNoPU)", 50, -2., 3.);
    meJetsRecLVPUSumPt2_ =
        ibook.book1D("JetsRecLVPUSumPt2",
                     "Contribution of PU to sum of Pt2 of jets for matched LV; log10(sumPt2-SumPt2NoPU)",
                     50,
                     -3.,
                     3.);
    meJetsRecLVPUMetPt_ =
        ibook.book1D("JetsRecLVPUMetPt",
                     "Contribution of PU to Missing Transverse Energy for matched LV; log10(Met-MetNoPU)",
                     50,
                     -3.,
                     2.);
    meJetsRecLVPUSumPz_ =
        ibook.book1D("JetsRecLVPUSumPz",
                     "Contribution of PU to sum of Pz of jets for matched LV; log10(abs(SumPz-SumPzNoPU))",
                     50,
                     -4.,
                     3.);

    meJetsRecLVPURelMultvsMult_ = ibook.bookProfile("JetsRecLVPURelMultvsMult",
                                                    "Relative contribution of PU to jet multiplicity vs number of jets "
                                                    "for matched vertices; #Jets; (#Jets-#JetsNoPU)/#Jets",
                                                    50,
                                                    0.,
                                                    120.,
                                                    0.,
                                                    1.,
                                                    "s");
    meJetsRecLVPURelHtvsHt_ = ibook.bookProfile("JetsRecLVPURelHtvsHt",
                                                "Relative contribution of PU to scalar sum of Et of jets vs scalar sum "
                                                "of Et for matched vertices; log10(Ht); (Ht-HtNoPU)/HT",
                                                50,
                                                1.5,
                                                3.,
                                                0.,
                                                1.,
                                                "s");
    meJetsRecLVPURelSumPt2vsSumPt2_ =
        ibook.bookProfile("JetsRecLVPURelSumPt2vsSumPt2",
                          "Relative contribution of PU to sum of Pt2 of jets vs sum of Pt2 for matched vertices; "
                          "log10(SumPt2); (SumPt2-SumPt2NoPU)/SumPt2",
                          50,
                          2.,
                          5.,
                          0.,
                          1.,
                          "s");
    meJetsRecLVFakeRelSumPt2vsSumPt2_ =
        ibook.bookProfile("JetsRecLVFakeRelSumPt2vsSumPt2",
                          "Relative contribution of fake tracks to sum of Pt2 of jets vs sum of Pt2 for matched "
                          "vertices; log10(SumPt2); (SumPt2-SumPt2NoFake)/SumPt2",
                          50,
                          2.,
                          5.,
                          0.,
                          1.,
                          "s");
    meJetsRecLVPURelMetPtvsMetPt_ = ibook.bookProfile("JetsRecLVPURelMetPtvsMetPt",
                                                      "Relative contribution of PU to Missing Transverse Energy vs MET "
                                                      "for matched vertices; log10(Met); (Met-MetNoPU)/Met",
                                                      50,
                                                      0.,
                                                      2.5,
                                                      -1.,
                                                      1.,
                                                      "s");
    meJetsRecLVPURelSumPzvsSumPz_ = ibook.bookProfile("JetsRecLVPURelSumPzvsSumPz",
                                                      "Relative contribution of PU to sum of Pz of jets vs Sum of Pz "
                                                      "for matched vertices; log10(abs SumPz); (SumPz-SumPzNoPU)/SumPz",
                                                      50,
                                                      0.5,
                                                      3.5,
                                                      -1.,
                                                      1.,
                                                      "s");

    meTrackTimeResCorrectPID_ = ibook.book1D(
        "TrackTimeResCorrectPID", "Time residual of tracks with correct PID; t_{rec} - t_{sim} [ns]; ", 100, -5., 5.);

    meTrackTimeResWrongPID_ = ibook.book1D(
        "TrackTimeResWrongPID", "Time residual of tracks with wrong PID; t_{rec} - t_{sim} [ns]; ", 100, -5., 5.);
    meTrackTimeResNoPID_ = ibook.book1D(
        "TrackTimeResNoPID", "Time residual of tracks with no PID; t_{rec} - t_{sim} [ns]; ", 100, -5., 5.);
    meTrackTimeResNoPIDtruePi_ = ibook.book1D(
        "TrackTimeResNoPIDtruePi", "Time residual of no PID tracks, true Pi; t_{rec} - t_{sim} [ns]; ", 100, -5., 5.);

    meTrackTimeResNoPIDtrueK_ = ibook.book1D(
        "TrackTimeResNoPIDtrueK", "Time residual of no PID tracks, true K; t_{rec} - t_{sim} [ns]; ", 100, -5., 5.);
    meTrackTimeResNoPIDtrueP_ = ibook.book1D(
        "TrackTimeResNoPIDtrueP", "Time residual of no PID tracks, true P; t_{rec} - t_{sim} [ns]; ", 100, -5., 5.);

    meNoPIDTrackTimeResNoPIDType_[0] =
        ibook.book1D("NoPIDTrackTimeResNoPIDType1",
                     "Time residual of no PID tracks, no PID type 1; t_{rec} - t_{sim} [ns];",
                     100,
                     -5.,
                     5.);

    meNoPIDTrackTimeResNoPIDType_[1] =
        ibook.book1D("NoPIDTrackTimeResNoPIDType2",
                     "Time residual of no PID tracks, no PID type 2; t_{rec} - t_{sim} [ns];",
                     100,
                     -5.,
                     5.);
    meNoPIDTrackTimeResNoPIDType_[2] =
        ibook.book1D("NoPIDTrackTimeResNoPIDType3",
                     "Time residual of no PID tracks, no PID type 3; t_{rec} - t_{sim} [ns];",
                     100,
                     -5.,
                     5.);

    meTrackTimePullCorrectPID_ =
        ibook.book1D("TrackTimePullCorrectPID",
                     "Time pull of tracks with correct PID; (t_{rec} - t_{sim})/#sigma_{t rec}; ",
                     100,
                     -10.,
                     10.);

    meTrackTimePullWrongPID_ = ibook.book1D("TrackTimePullWrongPID",
                                            "Time pull of tracks with wrong PID; (t_{rec} - t_{sim})/#sigma_{t rec}; ",
                                            100,
                                            -10.,
                                            10.);
    meTrackTimePullNoPID_ = ibook.book1D(
        "TrackTimePullNoPID", "Time pull of tracks with no PID; (t_{rec} - t_{sim})/#sigma_{t rec}; ", 100, -10., 10.);

    meTrackTimePullNoPIDtruePi_ =
        ibook.book1D("TrackTimePullNoPIDtruePi",
                     "Time pull of no PID tracks, true Pi; (t_{rec} - t_{sim})/#sigma_{t rec}; ",
                     100,
                     -10.,
                     10.);
    meTrackTimePullNoPIDtrueK_ =
        ibook.book1D("TrackTimePullNoPIDtrueK",
                     "Time pull of no PID tracks, true K; (t_{rec} - t_{sim})/#sigma_{t rec}; ",
                     100,
                     -10.,
                     10.);
    meTrackTimePullNoPIDtrueP_ =
        ibook.book1D("TrackTimePullNoPIDtrueP",
                     "Time pull of no PID tracks, true P; (t_{rec} - t_{sim})/#sigma_{t rec}; ",
                     100,
                     -10.,
                     10.);

    meNoPIDTrackTimePullNoPIDType_[0] =
        ibook.book1D("NoPIDTrackTimePullNoPIDType1",
                     "Time pull of no PID tracks, no PID type 1; (t_{rec} - t_{sim})/#sigma_{t rec}; ",
                     100,
                     -10.,
                     10.);
    meNoPIDTrackTimePullNoPIDType_[1] =
        ibook.book1D("NoPIDTrackTimePullNoPIDType2",
                     "Time pull of no PID tracks, no PID type 2; (t_{rec} - t_{sim})/#sigma_{t rec}; ",
                     100,
                     -10.,
                     10.);
    meNoPIDTrackTimePullNoPIDType_[2] =
        ibook.book1D("NoPIDTrackTimePullNoPIDType3",
                     "Time pull of no PID tracks, no PID type 3; (t_{rec} - t_{sim})/#sigma_{t rec}; ",
                     100,
                     -10.,
                     10.);

    meTrackTimeSigmaCorrectPID_ = ibook.book1D(
        "TrackTimeSigmaCorrectPID", "Time sigma of tracks with correct PID; #sigma_{t0Safe} [ns]; ", 100, 0., 4.);
    meTrackTimeSigmaWrongPID_ = ibook.book1D(
        "TrackTimeSigmaWrongPID", "Time sigma of tracks with wrong PID; #sigma_{t0Safe} [ns]; ", 100, 0., 4.);
    meTrackTimeSigmaNoPID_ =
        ibook.book1D("TrackTimeSigmaNoPID", "Time sigma of tracks with no PID; #sigma_{t0Safe} [ns]; ", 100, 0., 4.);
    meNoPIDTrackSigmaNoPIDType_[0] = ibook.book1D(
        "NoPIDTrackSigmaNoPIDType1", "Time sigma of no PID tracks, no PID type 1; #sigma_{t0Safe} [ns]; ", 100, 0., 4.);
    meNoPIDTrackSigmaNoPIDType_[1] = ibook.book1D(
        "NoPIDTrackSigmaNoPIDType2", "Time sigma of no PID tracks, no PID type 2; #sigma_{t0Safe} [ns]; ", 100, 0., 4.);
    meNoPIDTrackSigmaNoPIDType_[2] = ibook.book1D(
        "NoPIDTrackSigmaNoPIDType3", "Time sigma of no PID tracks, no PID type 3; #sigma_{t0Safe} [ns]; ", 100, 0., 4.);
    meTrackMVACorrectPID_ =
        ibook.book1D("TrackMVACorrectPID", "MVA of tracks with correct PID; MVA score; ", 100, 0., 1.);

    meTrackMVAWrongPID_ = ibook.book1D("TrackMVAWrongPID", "MVA of tracks with wrong PID; MVA score; ", 100, 0., 1.);
    meTrackMVANoPID_ = ibook.book1D("TrackMVANoPID", "MVA of tracks with no PID; MVA score; ", 100, 0., 1.);
    meNoPIDTrackMVANoPIDType_[0] =
        ibook.book1D("NoPIDTrackMVANoPIDType1", "MVA of no PID tracks, no PID type 1; MVA score; ", 100, 0., 1.);
    meNoPIDTrackMVANoPIDType_[1] =
        ibook.book1D("NoPIDTrackMVANoPIDType2", "MVA of no PID tracks, no PID type 2; MVA score; ", 100, 0., 1.);
    meNoPIDTrackMVANoPIDType_[2] =
        ibook.book1D("NoPIDTrackMVANoPIDType3", "MVA of no PID tracks, no PID type 3; MVA score; ", 100, 0., 1.);
  }

  // some tests
  meTrackResLowPTot_ = ibook.book1D(
      "TrackResLowP", "t_{rec} - t_{sim} for tracks with p < 2 GeV; t_{rec} - t_{sim} [ns] ", 70, -0.15, 0.15);
  meTrackResLowP_[0] =
      ibook.book1D("TrackResLowP-LowMVA",
                   "t_{rec} - t_{sim} for tracks with MVA < 0.5 and p < 2 GeV; t_{rec} - t_{sim} [ns] ",
                   100,
                   -1.,
                   1.);
  meTrackResLowP_[1] =
      ibook.book1D("TrackResLowP-MediumMVA",
                   "t_{rec} - t_{sim} for tracks with 0.5 < MVA < 0.8 and p < 2 GeV; t_{rec} - t_{sim} [ns] ",
                   100,
                   -1.,
                   1.);
  meTrackResLowP_[2] =
      ibook.book1D("TrackResLowP-HighMVA",
                   "t_{rec} - t_{sim} for tracks with MVA > 0.8 and p < 2 GeV; t_{rec} - t_{sim} [ns] ",
                   100,
                   -1.,
                   1.);
  meTrackResHighPTot_ = ibook.book1D(
      "TrackResHighP", "t_{rec} - t_{sim} for tracks with p > 2 GeV; t_{rec} - t_{sim} [ns] ", 70, -0.15, 0.15);
  meTrackResHighP_[0] =
      ibook.book1D("TrackResHighP-LowMVA",
                   "t_{rec} - t_{sim} for tracks with MVA < 0.5 and p > 2 GeV; t_{rec} - t_{sim} [ns] ",
                   100,
                   -1.,
                   1.);
  meTrackResHighP_[1] =
      ibook.book1D("TrackResHighP-MediumMVA",
                   "t_{rec} - t_{sim} for tracks with 0.5 < MVA < 0.8 and p > 2 GeV; t_{rec} - t_{sim} [ns] ",
                   100,
                   -1.,
                   1.);
  meTrackResHighP_[2] =
      ibook.book1D("TrackResHighP-HighMVA",
                   "t_{rec} - t_{sim} for tracks with MVA > 0.8 and p > 2 GeV; t_{rec} - t_{sim} [ns] ",
                   100,
                   -1.,
                   1.);
  meTrackPullLowPTot_ =
      ibook.book1D("TrackPullLowP", "Pull for tracks with p < 2 GeV; (t_{rec}-t_{sim})/#sigma_{t}", 100, -10., 10.);
  meTrackPullLowP_[0] = ibook.book1D("TrackPullLowP-LowMVA",
                                     "Pull for tracks with MVA < 0.5 and p < 2 GeV; (t_{rec}-t_{sim})/#sigma_{t}",
                                     100,
                                     -10.,
                                     10.);
  meTrackPullLowP_[1] = ibook.book1D("TrackPullLowP-MediumMVA",
                                     "Pull for tracks with 0.5 < MVA < 0.8 and p < 2 GeV; (t_{rec}-t_{sim})/#sigma_{t}",
                                     100,
                                     -10.,
                                     10.);
  meTrackPullLowP_[2] = ibook.book1D("TrackPullLowP-HighMVA",
                                     "Pull for tracks with MVA > 0.8 and p < 2 GeV; (t_{rec}-t_{sim})/#sigma_{t}",
                                     100,
                                     -10.,
                                     10.);
  meTrackPullHighPTot_ =
      ibook.book1D("TrackPullHighP", "Pull for tracks with p > 2 GeV; (t_{rec}-t_{sim})/#sigma_{t}", 100, -10., 10.);
  meTrackPullHighP_[0] = ibook.book1D("TrackPullHighP-LowMVA",
                                      "Pull for tracks with MVA < 0.5 and p > 2 GeV; (t_{rec}-t_{sim})/#sigma_{t}",
                                      100,
                                      -10.,
                                      10.);
  meTrackPullHighP_[1] =
      ibook.book1D("TrackPullHighP-MediumMVA",
                   "Pull for tracks with 0.5 < MVA < 0.8 and p > 2 GeV; (t_{rec}-t_{sim})/#sigma_{t}",
                   100,
                   -10.,
                   10.);
  meTrackPullHighP_[2] = ibook.book1D("TrackPullHighP-HighMVA",
                                      "Pull for tracks with MVA > 0.8 and p > 2 GeV; (t_{rec}-t_{sim})/#sigma_{t}",
                                      100,
                                      -10.,
                                      10.);
  if (optionalPlots_) {
    meTrackResMassProtons_[0] =
        ibook.book1D("TrackResMass-Protons-LowMVA",
                     "t_{rec} - t_{est} for proton tracks with MVA < 0.5; t_{rec} - t_{est} [ns] ",
                     100,
                     -1.,
                     1.);
    meTrackResMassProtons_[1] =
        ibook.book1D("TrackResMass-Protons-MediumMVA",
                     "t_{rec} - t_{est} for proton tracks with 0.5 < MVA < 0.8; t_{rec} - t_{est} [ns] ",
                     100,
                     -1.,
                     1.);
    meTrackResMassProtons_[2] =
        ibook.book1D("TrackResMass-Protons-HighMVA",
                     "t_{rec} - t_{est} for proton tracks with MVA > 0.8; t_{rec} - t_{est} [ns] ",
                     100,
                     -1.,
                     1.);
    meTrackResMassTrueProtons_[0] =
        ibook.book1D("TrackResMassTrue-Protons-LowMVA",
                     "t_{est} - t_{sim} for proton tracks with MVA < 0.5; t_{est} - t_{sim} [ns] ",
                     100,
                     -1.,
                     1.);
    meTrackResMassTrueProtons_[1] =
        ibook.book1D("TrackResMassTrue-Protons-MediumMVA",
                     "t_{est} - t_{sim} for proton tracks with 0.5 < MVA < 0.8; t_{est} - t_{sim} [ns] ",
                     100,
                     -1.,
                     1.);
    meTrackResMassTrueProtons_[2] =
        ibook.book1D("TrackResMassTrue-Protons-HighMVA",
                     "t_{est} - t_{sim} for proton tracks with MVA > 0.8; t_{est} - t_{sim} [ns] ",
                     100,
                     -1.,
                     1.);

    meTrackResMassPions_[0] = ibook.book1D("TrackResMass-Pions-LowMVA",
                                           "t_{rec} - t_{est} for pion tracks with MVA < 0.5; t_{rec} - t_{est} [ns] ",
                                           100,
                                           -1.,
                                           1.);
    meTrackResMassPions_[1] =
        ibook.book1D("TrackResMass-Pions-MediumMVA",
                     "t_{rec} - t_{est} for pion tracks with 0.5 < MVA < 0.8; t_{rec} - t_{est} [ns] ",
                     100,
                     -1.,
                     1.);
    meTrackResMassPions_[2] = ibook.book1D("TrackResMass-Pions-HighMVA",
                                           "t_{rec} - t_{est} for pion tracks with MVA > 0.8; t_{rec} - t_{est} [ns] ",
                                           100,
                                           -1.,
                                           1.);
    meTrackResMassTruePions_[0] =
        ibook.book1D("TrackResMassTrue-Pions-LowMVA",
                     "t_{est} - t_{sim} for pion tracks with MVA < 0.5; t_{est} - t_{sim} [ns] ",
                     100,
                     -1.,
                     1.);
    meTrackResMassTruePions_[1] =
        ibook.book1D("TrackResMassTrue-Pions-MediumMVA",
                     "t_{est} - t_{sim} for pion tracks with 0.5 < MVA < 0.8; t_{est} - t_{sim} [ns] ",
                     100,
                     -1.,
                     1.);
    meTrackResMassTruePions_[2] =
        ibook.book1D("TrackResMassTrue-Pions-HighMVA",
                     "t_{est} - t_{sim} for pion tracks with MVA > 0.8; t_{est} - t_{sim} [ns] ",
                     100,
                     -1.,
                     1.);
  }

  meBarrelPIDp_ = ibook.book1D("BarrelPIDp", "PID track MTD momentum spectrum, |eta| < 1.5;p [GeV]", 25, 0., 10.);
  meEndcapPIDp_ = ibook.book1D("EndcapPIDp", "PID track with MTD momentum spectrum, |eta| > 1.6;p [GeV]", 25, 0., 10.);

  meBarrelNoPIDtype_ = ibook.book1D("BarrelNoPIDtype", "Barrel PID failure category", 4, 0., 4.);
  meEndcapNoPIDtype_ = ibook.book1D("EndcapNoPIDtype", "Endcap PID failure category", 4, 0., 4.);

  meBarrelTruePiNoPID_ =
      ibook.book1D("BarrelTruePiNoPID", "True pi NoPID momentum spectrum, |eta| < 1.5;p [GeV]", 25, 0., 10.);
  meBarrelTrueKNoPID_ =
      ibook.book1D("BarrelTrueKNoPID", "True k NoPID momentum spectrum, |eta| < 1.5;p [GeV]", 25, 0., 10.);
  meBarrelTruePNoPID_ =
      ibook.book1D("BarrelTruePNoPID", "True p NoPID momentum spectrum, |eta| < 1.5;p [GeV]", 25, 0., 10.);
  meEndcapTruePiNoPID_ =
      ibook.book1D("EndcapTruePiNoPID", "True pi NoPID momentum spectrum, |eta| > 1.6;p [GeV]", 25, 0., 10.);
  meEndcapTrueKNoPID_ =
      ibook.book1D("EndcapTrueKNoPID", "True k NoPID momentum spectrum, |eta| > 1.6;p [GeV]", 25, 0., 10.);
  meEndcapTruePNoPID_ =
      ibook.book1D("EndcapTruePNoPID", "True p NoPID momentum spectrum, |eta| > 1.6;p [GeV]", 25, 0., 10.);

  meBarrelTruePiAsPi_ =
      ibook.book1D("BarrelTruePiAsPi", "True pi as pi momentum spectrum, |eta| < 1.5;p [GeV]", 25, 0., 10.);
  meBarrelTruePiAsK_ =
      ibook.book1D("BarrelTruePiAsK", "True pi as k momentum spectrum, |eta| < 1.5;p [GeV]", 25, 0., 10.);
  meBarrelTruePiAsP_ =
      ibook.book1D("BarrelTruePiAsP", "True pi as p momentum spectrum, |eta| < 1.5;p [GeV]", 25, 0., 10.);
  meEndcapTruePiAsPi_ =
      ibook.book1D("EndcapTruePiAsPi", "True pi as pi momentum spectrum, |eta| > 1.6;p [GeV]", 25, 0., 10.);
  meEndcapTruePiAsK_ =
      ibook.book1D("EndcapTruePiAsK", "True pi as k momentum spectrum, |eta| > 1.6;p [GeV]", 25, 0., 10.);
  meEndcapTruePiAsP_ =
      ibook.book1D("EndcapTruePiAsP", "True pi as p momentum spectrum, |eta| > 1.6;p [GeV]", 25, 0., 10.);

  meBarrelTrueKAsPi_ =
      ibook.book1D("BarrelTrueKAsPi", "True k as pi momentum spectrum, |eta| < 1.5;p [GeV]", 25, 0., 10.);
  meBarrelTrueKAsK_ = ibook.book1D("BarrelTrueKAsK", "True k as k momentum spectrum, |eta| < 1.5;p [GeV]", 25, 0., 10.);
  meBarrelTrueKAsP_ = ibook.book1D("BarrelTrueKAsP", "True k as p momentum spectrum, |eta| < 1.5;p [GeV]", 25, 0., 10.);
  meEndcapTrueKAsPi_ =
      ibook.book1D("EndcapTrueKAsPi", "True k as pi momentum spectrum, |eta| > 1.6;p [GeV]", 25, 0., 10.);
  meEndcapTrueKAsK_ = ibook.book1D("EndcapTrueKAsK", "True k as k momentum spectrum, |eta| > 1.6;p [GeV]", 25, 0., 10.);
  meEndcapTrueKAsP_ = ibook.book1D("EndcapTrueKAsP", "True k as p momentum spectrum, |eta| > 1.6;p [GeV]", 25, 0., 10.);

  meBarrelTruePAsPi_ =
      ibook.book1D("BarrelTruePAsPi", "True p as pi momentum spectrum, |eta| < 1.5;p [GeV]", 25, 0., 10.);
  meBarrelTruePAsK_ = ibook.book1D("BarrelTruePAsK", "True p as k momentum spectrum, |eta| < 1.5;p [GeV]", 25, 0., 10.);
  meBarrelTruePAsP_ = ibook.book1D("BarrelTruePAsP", "True p as p momentum spectrum, |eta| < 1.5;p [GeV]", 25, 0., 10.);
  meEndcapTruePAsPi_ =
      ibook.book1D("EndcapTruePAsPi", "True p as pi momentum spectrum, |eta| > 1.6;p [GeV]", 25, 0., 10.);
  meEndcapTruePAsK_ = ibook.book1D("EndcapTruePAsK", "True p as k momentum spectrum, |eta| > 1.6;p [GeV]", 25, 0., 10.);
  meEndcapTruePAsP_ = ibook.book1D("EndcapTruePAsP", "True p as p momentum spectrum, |eta| > 1.6;p [GeV]", 25, 0., 10.);
}

bool Primary4DVertexValidation::matchRecoTrack2SimSignal(const reco::TrackBaseRef& recoTrack) {
  auto found = r2s_->find(recoTrack);

  // reco track not matched to any TP
  if (found == r2s_->end())
    return false;

  //// reco track matched to some TP from signal vertex
  for (const auto& tp : found->val) {
    if (tp.first->eventId().bunchCrossing() == 0 && tp.first->eventId().event() == 0)
      return true;
  }

  // reco track not matched to any TP from signal vertex
  return false;
}

std::pair<const edm::Ref<std::vector<TrackingParticle>>*, int> Primary4DVertexValidation::getMatchedTP(
    const reco::TrackBaseRef& recoTrack, const TrackingVertexRef& vsim) {
  auto found = r2s_->find(recoTrack);

  // reco track not matched to any TP (fake tracks)
  if (found == r2s_->end())
    return std::make_pair(nullptr, -1);

  // matched TP equal to any TP of a given sim vertex
  for (const auto& tp : found->val) {
    if (std::find_if(vsim->daughterTracks_begin(), vsim->daughterTracks_end(), [&](const TrackingParticleRef& vtp) {
          return tp.first == vtp;
        }) != vsim->daughterTracks_end())
      return std::make_pair(&tp.first, 0);
    // matched TP not associated to any daughter track of a given sim vertex but having the same eventID (track from secondary vtx)
    else if (tp.first->eventId().bunchCrossing() == vsim->eventId().bunchCrossing() &&
             tp.first->eventId().event() == vsim->eventId().event()) {
      return std::make_pair(&tp.first, 1);
    }
    // matched TP not associated to any sim vertex of a given simulated event (PU track)
    else {
      return std::make_pair(&tp.first, 2);
    }
  }

  // reco track not matched to any TP from vertex
  return std::make_pair(nullptr, -1);
}

double Primary4DVertexValidation::timeFromTrueMass(double mass, double pathlength, double momentum, double time) {
  if (time > 0 && pathlength > 0 && mass > 0) {
    double gammasq = 1. + momentum * momentum / (mass * mass);
    double v = c_ * std::sqrt(1. - 1. / gammasq);  // cm / ns
    double t_est = time - (pathlength / v);

    return t_est;
  } else {
    return -1;
  }
}

bool Primary4DVertexValidation::select(const reco::Vertex& v, int level) {
  /* level
   0  !isFake  && ndof>4  (default)
   1  !isFake  && ndof>4 && prob > 0.01
   2  !isFake  && ndof>4 && prob > 0.01 && ptmax2 > 0.4
   */
  if (v.isFake())
    return false;
  if ((level == 0) && (v.ndof() > selNdof_))
    return true;
  /*if ((level == 1) && (v.ndof() > selNdof_) && (vertex_pxy(v) > 0.01))
    return true;
  if ((level == 2) && (v.ndof() > selNdof_) && (vertex_pxy(v) > 0.01) && (vertex_ptmax2(v) > 0.4))
    return true;
  if ((level == 3) && (v.ndof() > selNdof_) && (vertex_ptmax2(v) < 0.4))
    return true;*/
  return false;
}

void Primary4DVertexValidation::observablesFromJets(const std::vector<reco::Track>& reco_Tracks,
                                                    const std::vector<double>& mass_Tracks,
                                                    const std::vector<int>& category_Tracks,
                                                    const std::string& skip_Tracks,
                                                    unsigned int& n_Jets,
                                                    double& sum_EtJets,
                                                    double& sum_Pt2Jets,
                                                    double& met_Pt,
                                                    double& sum_PzJets) {
  double sum_PtJets = 0;
  n_Jets = 0;
  sum_EtJets = 0;
  sum_Pt2Jets = 0;
  met_Pt = 0;
  sum_PzJets = 0;
  auto met = LorentzVector(0, 0, 0, 0);
  std::vector<fastjet::PseudoJet> fjInputs_;
  fjInputs_.clear();
  size_t countScale0 = 0;
  for (size_t i = 0; i < reco_Tracks.size(); i++) {
    const auto& recotr = reco_Tracks[i];
    const auto mass = mass_Tracks[i];
    float scale = 1.;
    if (recotr.charge() == 0) {
      continue;
    }
    // skip PU tracks in jet definition if skip_PU is required
    if (skip_Tracks == "skip_PU" && category_Tracks[i] == 2) {
      continue;
    }
    // skip fake tracks in jet definition if skip_Fake is required
    if (skip_Tracks == "skip_Fake" && category_Tracks[i] == -1) {
      continue;
    }
    if (recotr.pt() != 0) {
      scale = (recotr.pt() - recotr.ptError()) / recotr.pt();
    }
    if (edm::isNotFinite(scale)) {
      edm::LogWarning("Primary4DVertexValidation") << "Scaling is NAN ignoring this recotrack" << std::endl;
      scale = 0;
    }
    if (scale < 0) {
      scale = 0;
      countScale0++;
    }
    if (scale != 0) {
      fjInputs_.push_back(fastjet::PseudoJet(recotr.px() * scale,
                                             recotr.py() * scale,
                                             recotr.pz() * scale,
                                             std::sqrt(recotr.p() * recotr.p() + mass * mass) * scale));
    }
  }
  fastjet::ClusterSequence sequence(fjInputs_, fastjet::JetDefinition(fastjet::antikt_algorithm, 0.4));
  auto jets = fastjet::sorted_by_pt(sequence.inclusive_jets(0));
  for (const auto& pj : jets) {
    auto p4 = LorentzVector(pj.px(), pj.py(), pj.pz(), pj.e());
    sum_EtJets += std::sqrt(p4.e() * p4.e() - p4.P() * p4.P() + p4.pt() * p4.pt());
    sum_PtJets += p4.pt();
    sum_Pt2Jets += (p4.pt() * p4.pt() * 0.8 * 0.8);
    met += p4;
    sum_PzJets += p4.pz();
    n_Jets++;
  }
  met_Pt = met.pt();
  double metAbove = met_Pt - 2 * std::sqrt(sum_PtJets);
  if (metAbove > 0) {
    sum_Pt2Jets += (metAbove * metAbove);
  }
  if (countScale0 == reco_Tracks.size()) {
    sum_Pt2Jets = countScale0 * 0.01;  //leave some epsilon value to sort vertices with unknown pt
  }
}

void Primary4DVertexValidation::isParticle(const reco::TrackBaseRef& recoTrack,
                                           const edm::ValueMap<float>& sigmat0,
                                           const edm::ValueMap<float>& sigmat0Safe,
                                           const edm::ValueMap<float>& probPi,
                                           const edm::ValueMap<float>& probK,
                                           const edm::ValueMap<float>& probP,
                                           unsigned int& no_PIDtype,
                                           bool& no_PID,
                                           bool& is_Pi,
                                           bool& is_K,
                                           bool& is_P) {
  no_PIDtype = 0;
  no_PID = false;
  is_Pi = false;
  is_K = false;
  is_P = false;
  if (probPi[recoTrack] == -1) {
    no_PIDtype = 1;
  } else if (edm::isNotFinite(probPi[recoTrack])) {
    no_PIDtype = 2;
  } else if (probPi[recoTrack] == 1 && probK[recoTrack] == 0 && probP[recoTrack] == 0 &&
             sigmat0[recoTrack] < sigmat0Safe[recoTrack]) {
    no_PIDtype = 3;
  }
  no_PID = no_PIDtype > 0;
  is_Pi = !no_PID && 1. - probPi[recoTrack] < minProbHeavy_;
  is_K = !no_PID && !is_Pi && probK[recoTrack] > probP[recoTrack];
  is_P = !no_PID && !is_Pi && !is_K;
}

void Primary4DVertexValidation::getWosWnt(const reco::Vertex& recoVtx,
                                          const reco::TrackBaseRef& recoTrk,
                                          const edm::ValueMap<float>& sigmat0,
                                          const edm::ValueMap<float>& mtdQualMVA,
                                          const edm::Handle<reco::BeamSpot>& BS,
                                          double& wos,
                                          double& wnt) {
  double dz2_beam = pow((*BS).BeamWidthX() * cos(recoTrk->phi()) / tan(recoTrk->theta()), 2) +
                    pow((*BS).BeamWidthY() * sin(recoTrk->phi()) / tan(recoTrk->theta()), 2);
  double dz2 =
      pow(recoTrk->dzError(), 2) + dz2_beam + pow(0.0020, 2);  // added 20 um, some tracks have crazy small resolutions
  wos = recoVtx.trackWeight(recoTrk) / dz2;
  wnt = recoVtx.trackWeight(recoTrk) *
        std::min(recoTrk->pt(), 1.0);  // pt-weighted number of tracks (downweights pt < 1 GeV tracks)

  // If tracks have time information, give more weight to tracks with good resolution
  if (sigmat0[recoTrk] > 0 && mtdQualMVA[recoTrk] > mvaTh_) {
    double sigmaZ = (*BS).sigmaZ();
    double sigmaT = sigmaZ / c_;  // c in cm/ns
    wos = wos / erf(sigmat0[recoTrk] / sigmaT);
  }
}

/* Extract information form TrackingParticles/TrackingVertex and fill
 * the helper class simPrimaryVertex with proper generation-level
 * information */
std::vector<Primary4DVertexValidation::simPrimaryVertex> Primary4DVertexValidation::getSimPVs(
    const edm::Handle<TrackingVertexCollection>& tVC) {
  std::vector<Primary4DVertexValidation::simPrimaryVertex> simpv;
  int current_event = -1;
  int s = -1;
  for (TrackingVertexCollection::const_iterator v = tVC->begin(); v != tVC->end(); ++v) {
    // We keep only the first vertex from all the events at BX=0.
    if (v->eventId().bunchCrossing() != 0)
      continue;
    if (v->eventId().event() != current_event) {
      current_event = v->eventId().event();
    } else {
      continue;
    }
    s++;
    if (std::abs(v->position().z()) > 1000)
      continue;  // skip junk vertices

    // could be a new vertex, check  all primaries found so far to avoid multiple entries
    simPrimaryVertex sv(v->position().x(), v->position().y(), v->position().z(), v->position().t());
    sv.eventId = v->eventId();
    sv.sim_vertex = TrackingVertexRef(tVC, std::distance(tVC->begin(), v));
    sv.OriginalIndex = s;

    for (TrackingParticleRefVector::iterator iTrack = v->daughterTracks_begin(); iTrack != v->daughterTracks_end();
         ++iTrack) {
      assert((**iTrack).eventId().bunchCrossing() == 0);
    }
    simPrimaryVertex* vp = nullptr;  // will become non-NULL if a vertex is found and then point to it
    for (std::vector<simPrimaryVertex>::iterator v0 = simpv.begin(); v0 != simpv.end(); v0++) {
      if ((sv.eventId == v0->eventId) && (std::abs(sv.x - v0->x) < 1e-5) && (std::abs(sv.y - v0->y) < 1e-5) &&
          (std::abs(sv.z - v0->z) < 1e-5)) {
        vp = &(*v0);
        break;
      }
    }
    if (!vp) {
      // this is a new vertex, add it to the list of sim-vertices
      simpv.push_back(sv);
      vp = &simpv.back();
    }

    // Loop over daughter track(s) as Tracking Particles
    for (TrackingVertex::tp_iterator iTP = v->daughterTracks_begin(); iTP != v->daughterTracks_end(); ++iTP) {
      auto momentum = (*(*iTP)).momentum();
      const reco::Track* matched_best_reco_track = nullptr;
      double match_quality = -1;
      if (use_only_charged_tracks_ && (**iTP).charge() == 0)
        continue;
      if (s2r_->find(*iTP) != s2r_->end()) {
        matched_best_reco_track = (*s2r_)[*iTP][0].first.get();
        match_quality = (*s2r_)[*iTP][0].second;
      }

      vp->ptot.setPx(vp->ptot.x() + momentum.x());
      vp->ptot.setPy(vp->ptot.y() + momentum.y());
      vp->ptot.setPz(vp->ptot.z() + momentum.z());
      vp->ptot.setE(vp->ptot.e() + (**iTP).energy());
      vp->pt += (**iTP).pt();
      vp->ptsq += ((**iTP).pt() * (**iTP).pt());
      vp->nGenTrk++;

      if (matched_best_reco_track) {
        vp->num_matched_reco_tracks++;
        vp->average_match_quality += match_quality;
      }
    }  // End of for loop on daughters sim-particles
    if (vp->num_matched_reco_tracks) {
      vp->average_match_quality /= static_cast<float>(vp->num_matched_reco_tracks);
    }
    LogTrace("Primary4DVertexValidation")
        << "average number of associated tracks: " << vp->num_matched_reco_tracks / static_cast<float>(vp->nGenTrk)
        << " with average quality: " << vp->average_match_quality;
  }  // End of for loop on tracking vertices

  // In case of no simulated vertices, break here
  if (simpv.empty())
    return simpv;

  // Now compute the closest distance in z between all simulated vertex
  // first initialize
  auto prev_z = simpv.back().z;
  for (simPrimaryVertex& vsim : simpv) {
    vsim.closest_vertex_distance_z = std::abs(vsim.z - prev_z);
    prev_z = vsim.z;
  }
  // then calculate
  for (std::vector<simPrimaryVertex>::iterator vsim = simpv.begin(); vsim != simpv.end(); vsim++) {
    std::vector<simPrimaryVertex>::iterator vsim2 = vsim;
    vsim2++;
    for (; vsim2 != simpv.end(); vsim2++) {
      double distance = std::abs(vsim->z - vsim2->z);
      // need both to be complete
      vsim->closest_vertex_distance_z = std::min(vsim->closest_vertex_distance_z, distance);
      vsim2->closest_vertex_distance_z = std::min(vsim2->closest_vertex_distance_z, distance);
    }
  }
  return simpv;
}

/* Extract information form reco Vertex and fill the helper class
 * recoPrimaryVertex with proper reco-level information */
std::vector<Primary4DVertexValidation::recoPrimaryVertex> Primary4DVertexValidation::getRecoPVs(
    const edm::Handle<edm::View<reco::Vertex>>& tVC) {
  std::vector<Primary4DVertexValidation::recoPrimaryVertex> recopv;
  int r = -1;
  for (auto v = tVC->begin(); v != tVC->end(); ++v) {
    r++;
    // Skip junk vertices
    if (std::abs(v->z()) > 1000)
      continue;
    if (v->isFake() || !v->isValid())
      continue;

    recoPrimaryVertex sv(v->position().x(), v->position().y(), v->position().z());
    sv.recVtx = &(*v);
    sv.recVtxRef = reco::VertexBaseRef(tVC, std::distance(tVC->begin(), v));

    sv.OriginalIndex = r;
    sv.ndof = v->ndof();
    // this is a new vertex, add it to the list of reco-vertices
    recopv.push_back(sv);
    Primary4DVertexValidation::recoPrimaryVertex* vp = &recopv.back();

    // Loop over daughter track(s)
    for (auto iTrack = v->tracks_begin(); iTrack != v->tracks_end(); ++iTrack) {
      auto momentum = (*(*iTrack)).innerMomentum();
      if (momentum.mag2() == 0)
        momentum = (*(*iTrack)).momentum();
      vp->pt += std::sqrt(momentum.perp2());
      vp->ptsq += (momentum.perp2());
      vp->nRecoTrk++;

      auto matched = r2s_->find(*iTrack);
      if (matched != r2s_->end()) {
        vp->num_matched_sim_tracks++;
      }

    }  // End of for loop on daughters reconstructed tracks
  }  // End of for loop on tracking vertices

  // In case of no reco vertices, break here
  if (recopv.empty())
    return recopv;

  // Now compute the closest distance in z between all reconstructed vertex
  // first initialize
  auto prev_z = recopv.back().z;
  for (recoPrimaryVertex& vreco : recopv) {
    vreco.closest_vertex_distance_z = std::abs(vreco.z - prev_z);
    prev_z = vreco.z;
  }
  for (std::vector<recoPrimaryVertex>::iterator vreco = recopv.begin(); vreco != recopv.end(); vreco++) {
    std::vector<recoPrimaryVertex>::iterator vreco2 = vreco;
    vreco2++;
    for (; vreco2 != recopv.end(); vreco2++) {
      double distance = std::abs(vreco->z - vreco2->z);
      // need both to be complete
      vreco->closest_vertex_distance_z = std::min(vreco->closest_vertex_distance_z, distance);
      vreco2->closest_vertex_distance_z = std::min(vreco2->closest_vertex_distance_z, distance);
    }
  }
  return recopv;
}

// ------------ method called to produce the data  ------------
void Primary4DVertexValidation::matchReco2Sim(std::vector<recoPrimaryVertex>& recopv,
                                              std::vector<simPrimaryVertex>& simpv,
                                              const edm::ValueMap<float>& sigmat0,
                                              const edm::ValueMap<float>& MVA,
                                              const edm::Handle<reco::BeamSpot>& BS) {
  // Initialization (clear wos and wnt)
  for (auto vv : simpv) {
    vv.wnt.clear();
    vv.wos.clear();
  }
  for (auto rv : recopv) {
    rv.wnt.clear();
    rv.wos.clear();
  }

  // Filling infos for matching rec and sim vertices
  for (unsigned int iv = 0; iv < recopv.size(); iv++) {
    const reco::Vertex* vertex = recopv.at(iv).recVtx;
    LogTrace("Primary4DVertexValidation") << "iv (rec): " << iv;

    for (unsigned int iev = 0; iev < simpv.size(); iev++) {
      double wnt = 0;
      double wos = 0;
      double evwnt = 0;       // weighted number of tracks from sim event iev in the current recvtx
      double evwos = 0;       // weight over sigma**2 of sim event iev in the current recvtx
      unsigned int evnt = 0;  // number of tracks from sim event iev in the current recvtx

      for (auto iTrack = vertex->tracks_begin(); iTrack != vertex->tracks_end(); ++iTrack) {
        if (vertex->trackWeight(*iTrack) < trackweightTh_) {
          continue;
        }

        auto tp_info = getMatchedTP(*iTrack, simpv.at(iev).sim_vertex).first;
        int matchCategory = getMatchedTP(*iTrack, simpv.at(iev).sim_vertex).second;
        // matched TP equal to any TP of a given sim vertex
        if (tp_info != nullptr && matchCategory == 0) {
          getWosWnt(*vertex, *iTrack, MVA, sigmat0, BS, wos, wnt);
          simpv.at(iev).addTrack(iv, wos, wnt);
          recopv.at(iv).addTrack(iev, wos, wnt);
          evwos += wos;
          evwnt += wnt;
          evnt++;
        }
      }  // RecoTracks loop

      // require 2 tracks for a wos-match
      if ((evwos > 0) && (evwos > recopv.at(iv).maxwos) && (evnt > 1)) {
        recopv.at(iv).wosmatch = iev;
        recopv.at(iv).maxwos = evwos;
        recopv.at(iv).maxwosnt = evnt;
        LogTrace("Primary4DVertexValidation") << "dominating sim event (iev): " << iev << " evwos = " << evwos;
      }

      // weighted track counting match, require at least one track
      if ((evnt > 0) && (evwnt > recopv.at(iv).maxwnt)) {
        recopv.at(iv).wntmatch = iev;
        recopv.at(iv).maxwnt = evwnt;
      }
    }  // TrackingVertex loop
    if (recopv.at(iv).maxwos > 0) {
      simpv.at(recopv.at(iv).wosmatch).wos_dominated_recv.push_back(iv);
      simpv.at(recopv.at(iv).wosmatch).nwosmatch++;  // count the rec vertices dominated by a sim vertex using wos
      assert(iv < recopv.size());
    }
    LogTrace("Primary4DVertexValidation") << "largest contribution to wos: wosmatch (iev) = " << recopv.at(iv).wosmatch
                                          << " maxwos = " << recopv.at(iv).maxwos;
    if (recopv.at(iv).maxwnt > 0) {
      simpv.at(recopv.at(iv).wntmatch).nwntmatch++;  // count the rec vertices dominated by a sim vertex using wnt
    }
  }  // RecoPrimaryVertex

  // reset
  for (auto& vrec : recopv) {
    vrec.sim = NOT_MATCHED;
    vrec.matchQuality = 0;
  }
  unsigned int iev = 0;
  for (auto& vv : simpv) {
    LogTrace("Primary4DVertexValidation") << "iev (sim): " << iev;
    LogTrace("Primary4DVertexValidation") << "wos_dominated_recv.size: " << vv.wos_dominated_recv.size();
    for (unsigned int i = 0; i < vv.wos_dominated_recv.size(); i++) {
      auto recov = vv.wos_dominated_recv.at(i);
      LogTrace("Primary4DVertexValidation")
          << "index of reco vertex: " << recov << " that has a wos: " << vv.wos.at(recov) << " at position " << i;
    }
    vv.rec = NOT_MATCHED;
    vv.matchQuality = 0;
    iev++;
  }
  // after filling infos, goes for the sim-reco match
  // this tries a one-to-one match, taking simPV with highest wos if there are > 1 simPV candidates
  for (unsigned int rank = 1; rank < maxRank_; rank++) {
    for (unsigned int iev = 0; iev < simpv.size(); iev++) {  //loop on SimPV
      if (simpv.at(iev).rec != NOT_MATCHED) {
        continue;  // only sim vertices not already matched
      }
      if (simpv.at(iev).nwosmatch == 0) {
        continue;  // the sim vertex does not dominate any reco vertex
      }
      if (simpv.at(iev).nwosmatch > rank) {
        continue;  // start with sim vertices dominating one rec vertex (rank 1), then go with the ones dominating more
      }
      unsigned int iv = NOT_MATCHED;  // select a rec vertex index
      for (unsigned int k = 0; k < simpv.at(iev).wos_dominated_recv.size(); k++) {
        unsigned int rec = simpv.at(iev).wos_dominated_recv.at(k);  //candidate rec vertex index
        auto vrec = recopv.at(rec);
        if (vrec.sim != NOT_MATCHED) {
          continue;  // already matched
        }
        if (std::abs(simpv.at(iev).z - vrec.z) > zWosMatchMax_) {
          continue;  // insanely far away
        }
        if ((iv == NOT_MATCHED) || simpv.at(iev).wos.at(rec) > simpv.at(iev).wos.at(iv)) {
          iv = rec;
        }
      }
      // if a viable candidate was found, make the link
      if (iv !=
          NOT_MATCHED) {  // if the rec vertex has already been associated is possible that iv remains NOT_MATCHED at this point
        recopv.at(iv).sim = iev;
        simpv.at(iev).rec = iv;
        recopv.at(iv).matchQuality = rank;
        simpv.at(iev).matchQuality = rank;
      }
    }  // iev
  }  // rank

  // Reco vertices that are not necessarily dominated by a sim vertex, or whose dominating sim-vertex
  // has been matched already to another overlapping reco vertex, can still be associated to a specific
  // sim vertex (without being classified as dominating).
  // In terms of fitting 1d-distributions, this corresponds to a small peak close to a much larger nearby peak
  unsigned int ntry = 0;
  while (ntry++ < maxTry_) {
    unsigned nmatch = 0;
    for (unsigned int iev = 0; iev < simpv.size(); iev++) {
      if ((simpv.at(iev).rec != NOT_MATCHED) || (simpv.at(iev).wos.empty())) {
        continue;
      }
      // find a rec vertex for the NOT_MATCHED sim vertex
      unsigned int rec = NOT_MATCHED;
      for (auto rv : simpv.at(iev).wos) {
        if ((rec == NOT_MATCHED) || (rv.second > simpv.at(iev).wos.at(rec))) {
          rec = rv.first;
        }
      }

      if (rec == NOT_MATCHED) {  // try with wnt match
        for (auto rv : simpv.at(iev).wnt) {
          if ((rec == NOT_MATCHED) || (rv.second > simpv.at(iev).wnt.at(rec))) {
            rec = rv.first;
          }
        }
      }

      if (rec == NOT_MATCHED) {
        continue;  // should not happen
      }
      if (recopv.at(rec).sim != NOT_MATCHED) {
        continue;  // already gone
      }

      // check if the rec vertex can be matched
      unsigned int rec2sim = NOT_MATCHED;
      for (auto sv : recopv.at(rec).wos) {
        if (simpv.at(sv.first).rec != NOT_MATCHED) {
          continue;  // already used
        }
        if ((rec2sim == NOT_MATCHED) || (sv.second > recopv.at(rec).wos.at(rec2sim))) {
          rec2sim = sv.first;
        }
      }
      if (iev == rec2sim) {
        // do the match and assign lowest quality (i.e. max rank)
        recopv.at(rec).sim = iev;
        recopv.at(rec).matchQuality = maxRank_;
        simpv.at(iev).rec = rec;
        simpv.at(iev).matchQuality = maxRank_;
        nmatch++;
      }
    }  // sim loop
    if (nmatch == 0) {
      break;
    }
  }  // ntry

// Debugging
#ifdef EDM_ML_DEBUG
  unsigned int nmatch_tot = 0, n_dzgtsz = 0;
  unsigned int n_rank1 = 0, n_rank2 = 0, n_rank3 = 0, n_rank8 = 0;

  for (unsigned int iev = 0; iev < simpv.size(); iev++) {  //loop on SimPV
    if (simpv.at(iev).rec != NOT_MATCHED) {
      unsigned int rec = simpv.at(iev).rec;
      unsigned int wosmatch = recopv.at(rec).wosmatch;
      LogTrace("Primary4DVertexValidation")
          << "Final match: iev (sim) = " << std::setw(4) << iev << "  sim.rec = " << std::setw(4) << rec
          << "  rec.wosmatch = " << std::setw(5) << wosmatch << "  dZ/sigmaZ = " << std::setw(6) << std::setprecision(2)
          << std::abs((recopv.at(rec).z - simpv.at(iev).z) / recopv.at(rec).recVtx->zError())
          << "  match qual = " << std::setw(1) << recopv.at(rec).matchQuality;
      nmatch_tot++;
      if (std::abs((recopv.at(rec).z - simpv.at(iev).z) / recopv.at(rec).recVtx->zError()) > 3.) {
        n_dzgtsz++;
      }
      if (recopv.at(rec).matchQuality == 1) {
        n_rank1++;
      }
      if (recopv.at(rec).matchQuality == 2) {
        n_rank2++;
      }
      if (recopv.at(rec).matchQuality == 3) {
        n_rank3++;
      }
      if (recopv.at(rec).matchQuality == 8) {
        n_rank8++;
      }
    }
  }
  LogTrace("Primary4DVertexValidation") << "n_sim = " << simpv.size() << " n_rec = " << recopv.size()
                                        << " nmatch_tot = " << nmatch_tot << " n(dZ>sigmaZ) = " << n_dzgtsz
                                        << " n_rank1 = " << n_rank1 << " n_rank2 = " << n_rank2
                                        << " n_rank3 = " << n_rank3 << " n_rank8 = " << n_rank8;
#endif
}

void Primary4DVertexValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using edm::Handle;
  using edm::View;
  using std::cout;
  using std::endl;
  using std::vector;
  using namespace reco;

  std::vector<float> pileUpInfo_z;

  // get the pileup information
  edm::Handle<std::vector<PileupSummaryInfo>> puinfoH;
  if (iEvent.getByToken(vecPileupSummaryInfoToken_, puinfoH)) {
    for (auto const& pu_info : *puinfoH.product()) {
      if (pu_info.getBunchCrossing() == 0) {
        pileUpInfo_z = pu_info.getPU_zpositions();
        break;
      }
    }
  }

  edm::Handle<TrackingParticleCollection> TPCollectionH;
  iEvent.getByToken(trackingParticleCollectionToken_, TPCollectionH);
  if (!TPCollectionH.isValid())
    edm::LogWarning("Primary4DVertexValidation") << "TPCollectionH is not valid";

  edm::Handle<TrackingVertexCollection> TVCollectionH;
  iEvent.getByToken(trackingVertexCollectionToken_, TVCollectionH);
  if (!TVCollectionH.isValid())
    edm::LogWarning("Primary4DVertexValidation") << "TVCollectionH is not valid";

  edm::Handle<reco::SimToRecoCollection> simToRecoH;
  iEvent.getByToken(simToRecoAssociationToken_, simToRecoH);
  if (simToRecoH.isValid())
    s2r_ = simToRecoH.product();
  else
    edm::LogWarning("Primary4DVertexValidation") << "simToRecoH is not valid";

  edm::Handle<reco::RecoToSimCollection> recoToSimH;
  iEvent.getByToken(recoToSimAssociationToken_, recoToSimH);
  if (recoToSimH.isValid())
    r2s_ = recoToSimH.product();
  else
    edm::LogWarning("Primary4DVertexValidation") << "recoToSimH is not valid";

  reco::BeamSpot beamSpot;
  edm::Handle<reco::BeamSpot> BeamSpotH;
  iEvent.getByToken(RecBeamSpotToken_, BeamSpotH);
  if (!BeamSpotH.isValid())
    edm::LogWarning("Primary4DVertexValidation") << "BeamSpotH is not valid";
  beamSpot = *BeamSpotH;

  edm::Handle<reco::TrackCollection> tks;
  iEvent.getByToken(trkToken, tks);
  const auto& theB = &iSetup.getData(theTTBToken);
  std::vector<reco::TransientTrack> t_tks;

  std::vector<simPrimaryVertex> simpv;  // a list of simulated primary MC vertices
  simpv = getSimPVs(TVCollectionH);
  // this bool check if first vertex in that with highest pT
  bool signal_is_highest_pt =
      std::max_element(simpv.begin(), simpv.end(), [](const simPrimaryVertex& lhs, const simPrimaryVertex& rhs) {
        return lhs.ptsq < rhs.ptsq;
      }) == simpv.begin();

  std::vector<recoPrimaryVertex> recopv;  // a list of reconstructed primary MC vertices
  edm::Handle<edm::View<reco::Vertex>> recVtxs;
  iEvent.getByToken(Rec4DVerToken_, recVtxs);
  if (!recVtxs.isValid())
    edm::LogWarning("Primary4DVertexValidation") << "recVtxs is not valid";
  recopv = getRecoPVs(recVtxs);

  const auto& trackAssoc = iEvent.get(trackAssocToken_);
  const auto& pathLength = iEvent.get(pathLengthToken_);
  const auto& momentum = iEvent.get(momentumToken_);
  const auto& time = iEvent.get(timeToken_);
  const auto& t0Pid = iEvent.get(t0PidToken_);
  const auto& sigmat0 = iEvent.get(sigmat0PidToken_);
  const auto& t0Safe = iEvent.get(t0SafePidToken_);
  const auto& sigmat0Safe = iEvent.get(sigmat0SafePidToken_);
  const auto& mtdQualMVA = iEvent.get(trackMVAQualToken_);
  const auto& tMtd = iEvent.get(tmtdToken_);
  const auto& tofPi = iEvent.get(tofPiToken_);
  const auto& tofK = iEvent.get(tofKToken_);
  const auto& tofP = iEvent.get(tofPToken_);
  const auto& probPi = iEvent.get(probPiToken_);
  const auto& probK = iEvent.get(probKToken_);
  const auto& probP = iEvent.get(probPToken_);
  const auto& fPDGTable = iSetup.getHandle(pdtToken_);

  // I have simPV and recoPV collections
  matchReco2Sim(recopv, simpv, sigmat0Safe, mtdQualMVA, BeamSpotH);

  t_tks = (*theB).build(tks, beamSpot, t0Safe, sigmat0Safe);

  // track filter
  std::vector<reco::TransientTrack>&& seltks = theTrackFilter->select(t_tks);

  int unassociatedCount = 0;
  int unassociatedCountFake = 0;
  for (std::vector<reco::TransientTrack>::const_iterator itk = seltks.begin(); itk != seltks.end(); itk++) {
    reco::TrackBaseRef trackref = (*itk).trackBaseRef();
    bool isAssociated = false;
    for (unsigned int iv = 0; iv < recopv.size(); iv++) {
      const reco::Vertex* vertex = recopv.at(iv).recVtx;
      for (auto iTrack = vertex->tracks_begin(); iTrack != vertex->tracks_end(); ++iTrack) {
        if (*iTrack == trackref) {
          isAssociated = true;
          break;
        }
      }
      if (isAssociated)
        break;
    }

    if (!isAssociated) {
      unassociatedCount++;
      auto found = r2s_->find(trackref);
      if (found == r2s_->end())
        unassociatedCountFake++;
    }
  }
  double fraction = double(unassociatedCount) / (seltks.size());
  meUnAssocTracks_->Fill(log10(unassociatedCount));
  meFractionUnAssocTracks_->Fill(fraction);

  double fractionFake = double(unassociatedCountFake) / (seltks.size());
  meUnAssocTracksFake_->Fill(log10(unassociatedCountFake));
  meFractionUnAssocTracksFake_->Fill(fractionFake);

  // Loop on tracks
  for (unsigned int iv = 0; iv < recopv.size(); iv++) {
    if (recopv.at(iv).ndof > selNdof_) {
      const reco::Vertex* vertex = recopv.at(iv).recVtx;

      for (unsigned int iev = 0; iev < simpv.size(); iev++) {
        auto vsim = simpv.at(iev).sim_vertex;

        bool selectedVtxMatching = recopv.at(iv).sim == iev && simpv.at(iev).rec == iv;
        bool selectedLV = simpv.at(iev).eventId.bunchCrossing() == 0 && simpv.at(iev).eventId.event() == 0 &&
                          recopv.at(iv).OriginalIndex == 0;
        bool selectedLVMatching = selectedVtxMatching && selectedLV;  // bool for reco vtx leading match
        if (selectedLVMatching && !recopv.at(iv).is_signal()) {
          edm::LogWarning("Primary4DVertexValidation")
              << "Reco vtx leading match inconsistent: BX/ID " << simpv.at(iev).eventId.bunchCrossing() << " "
              << simpv.at(iev).eventId.event();
        }
#ifdef EDM_ML_DEBUG
        if (selectedLVMatching) {
          printSimVtxRecoVtxInfo(simpv.at(iev), recopv.at(iv));
        }
#endif
        double vzsim = simpv.at(iev).z;
        double vtsim = simpv.at(iev).t * simUnit_;

        double wnt = 0, wos = 0;
        double PUsumWnt = 0, PUsumWos = 0, SecsumWos = 0, FakesumWos = 0, PUsumPt = 0, PUsumPt2 = 0;
        double sumWnt = 0, sumWos = 0, sumPt = 0, sumPt2 = 0;
        unsigned int nt = 0, PUnt = 0, Fakent = 0;

        std::vector<double> massVector;
        std::vector<reco::Track> recotracks;
        std::vector<int> categoryVector;
        double sumEtJets = 0, sumPt2Jets = 0, metPt = 0, sumPzJets = 0;
        double sumEtJetsnoPU = 0, sumPt2JetsnoPU = 0, metPtnoPU = 0, sumPzJetsnoPU = 0;
        double sumEtJetsnoFake = 0, sumPt2JetsnoFake = 0, metPtnoFake = 0, sumPzJetsnoFake = 0;
        unsigned int nJets = 0, nJetsnoPU = 0, nJetsnoFake = 0;
        for (auto iTrack = vertex->tracks_begin(); iTrack != vertex->tracks_end(); ++iTrack) {
          if (trackAssoc[*iTrack] == -1) {
            edm::LogWarning("mtdTracks") << "Extended track not associated";
            continue;
          }

          // monitor all track weights associated to a vertex before selection on it
          if (selectedVtxMatching) {
            meVtxTrackW_->Fill(vertex->trackWeight(*iTrack));
            if (selectedLV) {
              meVtxTrackRecLVW_->Fill(vertex->trackWeight(*iTrack));
            }
          }

          if (vertex->trackWeight(*iTrack) < trackweightTh_)
            continue;
          bool noCrack = std::abs((*iTrack)->eta()) < trackMaxBtlEta_ || std::abs((*iTrack)->eta()) > trackMinEtlEta_;

          bool selectRecoTrk = trkRecSel(**iTrack);
          if (selectedLVMatching && selectRecoTrk) {
            if (noCrack) {
              meTrackEffPtTot_->Fill((*iTrack)->pt());
            }
            meTrackEffEtaTot_->Fill(std::abs((*iTrack)->eta()));
          }

          auto tp_info = getMatchedTP(*iTrack, vsim).first;
          int matchCategory = getMatchedTP(*iTrack, vsim).second;

          // PU, fake and secondary tracks
          if (selectedVtxMatching) {
            unsigned int no_PIDtype = 0;
            bool no_PID, is_Pi, is_K, is_P;
            int PartID = 211;  // pion
            isParticle(*iTrack, sigmat0, sigmat0Safe, probPi, probK, probP, no_PIDtype, no_PID, is_Pi, is_K, is_P);
            if (!use3dNoTime_) {
              if (no_PID || is_Pi) {
                PartID = 211;
              } else if (is_K) {
                PartID = 321;
              } else if (is_P) {
                PartID = 2212;
              }
            }
            const HepPDT::ParticleData* PData = fPDGTable->particle(HepPDT::ParticleID(abs(PartID)));
            double mass = PData->mass().value();
            massVector.push_back(mass);
            recotracks.push_back(**iTrack);
            getWosWnt(*vertex, *iTrack, sigmat0Safe, mtdQualMVA, BeamSpotH, wos, wnt);
            meVtxTrackWnt_->Fill(wnt);
            if (selectedLV) {
              meVtxTrackRecLVWnt_->Fill(wnt);
            }
            // reco track matched to any TP
            if (tp_info != nullptr) {
#ifdef EDM_ML_DEBUG
              if (selectedLV) {
                printMatchedRecoTrackInfo(*vertex, *iTrack, *tp_info, matchCategory);
              }
#endif
              // matched TP not associated to any daughter track of a given sim vertex but having the same eventID (track from secondary vtx)
              if (matchCategory == 1) {
                categoryVector.push_back(matchCategory);
                SecsumWos += wos;
              }
              // matched TP not associated to any sim vertex of a given simulated event (PU track)
              if (matchCategory == 2) {
                if (optionalPlots_) {
                  mePUTrackWnt_->Fill(wnt);
                  if (selectedLV) {
                    mePUTrackRecLVWnt_->Fill(wnt);
                  }
                }
                PUsumWnt += wnt;
                PUsumWos += wos;
                PUsumPt += (*iTrack)->pt();
                PUsumPt2 += ((*iTrack)->pt() * (*iTrack)->pt());
                PUnt++;
                categoryVector.push_back(2);
              }
            }
            // reco track not matched to any TP (fake tracks)
            else {
              categoryVector.push_back(matchCategory);
              FakesumWos += wos;
              Fakent++;
            }
            nt++;
            sumWnt += wnt;
            sumWos += wos;
            sumPt += (*iTrack)->pt();
            sumPt2 += ((*iTrack)->pt() * (*iTrack)->pt());
          }

          // matched TP equal to any TP of a given sim vertex
          if (tp_info != nullptr && matchCategory == 0) {
            categoryVector.push_back(matchCategory);
            double mass = (*tp_info)->mass();
            double tsim = (*tp_info)->parentVertex()->position().t() * simUnit_;
            double tEst = timeFromTrueMass(mass, pathLength[*iTrack], momentum[*iTrack], time[*iTrack]);

            double xsim = (*tp_info)->parentVertex()->position().x();
            double ysim = (*tp_info)->parentVertex()->position().y();
            double zsim = (*tp_info)->parentVertex()->position().z();
            double xPCA = (*iTrack)->vx();
            double yPCA = (*iTrack)->vy();
            double zPCA = (*iTrack)->vz();

            double dZ = zPCA - zsim;
            double d3D = std::sqrt((xPCA - xsim) * (xPCA - xsim) + (yPCA - ysim) * (yPCA - ysim) + dZ * dZ);
            // orient d3D according to the projection of RECO - SIM onto simulated momentum
            if ((xPCA - xsim) * ((*tp_info)->px()) + (yPCA - ysim) * ((*tp_info)->py()) + dZ * ((*tp_info)->pz()) <
                0.) {
              d3D = -d3D;
            }

            // select TPs associated to the signal event
            bool selectTP = trkTPSelLV(**tp_info);

            if (selectedLVMatching && selectRecoTrk && selectTP) {
              meTrackMatchedTPZposResTot_->Fill((*iTrack)->vz() - vzsim);
              if (noCrack) {
                meTrackMatchedTPEffPtTot_->Fill((*iTrack)->pt());
              }
              meTrackMatchedTPEffEtaTot_->Fill(std::abs((*iTrack)->eta()));
            }

            if (sigmat0Safe[*iTrack] == -1)
              continue;

            if (selectedLVMatching && selectRecoTrk && selectTP) {
              meTrackMatchedTPResTot_->Fill(t0Safe[*iTrack] - vtsim);
              meTrackMatchedTPPullTot_->Fill((t0Safe[*iTrack] - vtsim) / sigmat0Safe[*iTrack]);
              if (noCrack) {
                meTrackMatchedTPEffPtMtd_->Fill((*iTrack)->pt());
              }
              meTrackMatchedTPEffEtaMtd_->Fill(std::abs((*iTrack)->eta()));

              unsigned int noPIDtype = 0;
              bool noPID = false, isPi = false, isK = false, isP = false;
              isParticle(*iTrack, sigmat0, sigmat0Safe, probPi, probK, probP, noPIDtype, noPID, isPi, isK, isP);

              if ((isPi && std::abs(tMtd[*iTrack] - tofPi[*iTrack] - t0Pid[*iTrack]) > tol_) ||
                  (isK && std::abs(tMtd[*iTrack] - tofK[*iTrack] - t0Pid[*iTrack]) > tol_) ||
                  (isP && std::abs(tMtd[*iTrack] - tofP[*iTrack] - t0Pid[*iTrack]) > tol_)) {
                edm::LogWarning("Primary4DVertexValidation")
                    << "No match between mass hyp. and time: " << std::abs((*tp_info)->pdgId()) << " mass hyp pi/k/p "
                    << isPi << " " << isK << " " << isP << " t0/t0safe " << t0Pid[*iTrack] << " " << t0Safe[*iTrack]
                    << " tMtd - tof pi/K/p " << tMtd[*iTrack] - tofPi[*iTrack] << " " << tMtd[*iTrack] - tofK[*iTrack]
                    << " " << tMtd[*iTrack] - tofP[*iTrack] << " Prob pi/K/p " << probPi[*iTrack] << " "
                    << probK[*iTrack] << " " << probP[*iTrack];
              }

              if (std::abs((*iTrack)->eta()) < trackMaxBtlEta_) {
                meBarrelPIDp_->Fill((*iTrack)->p());
                meBarrelNoPIDtype_->Fill(noPIDtype + 0.5);
                if (std::abs((*tp_info)->pdgId()) == 211) {
                  if (noPID) {
                    meBarrelTruePiNoPID_->Fill((*iTrack)->p());
                  } else if (isPi) {
                    meBarrelTruePiAsPi_->Fill((*iTrack)->p());
                  } else if (isK) {
                    meBarrelTruePiAsK_->Fill((*iTrack)->p());
                  } else if (isP) {
                    meBarrelTruePiAsP_->Fill((*iTrack)->p());
                  } else {
                    edm::LogWarning("Primary4DVertexValidation")
                        << "No PID class: " << std::abs((*tp_info)->pdgId()) << " t0/t0safe " << t0Pid[*iTrack] << " "
                        << t0Safe[*iTrack] << " Prob pi/K/p " << probPi[*iTrack] << " " << probK[*iTrack] << " "
                        << probP[*iTrack];
                  }
                } else if (std::abs((*tp_info)->pdgId()) == 321) {
                  if (noPID) {
                    meBarrelTrueKNoPID_->Fill((*iTrack)->p());
                  } else if (isPi) {
                    meBarrelTrueKAsPi_->Fill((*iTrack)->p());
                  } else if (isK) {
                    meBarrelTrueKAsK_->Fill((*iTrack)->p());
                  } else if (isP) {
                    meBarrelTrueKAsP_->Fill((*iTrack)->p());
                  } else {
                    edm::LogWarning("Primary4DVertexValidation")
                        << "No PID class: " << std::abs((*tp_info)->pdgId()) << " t0/t0safe " << t0Pid[*iTrack] << " "
                        << t0Safe[*iTrack] << " Prob pi/K/p " << probPi[*iTrack] << " " << probK[*iTrack] << " "
                        << probP[*iTrack];
                  }
                } else if (std::abs((*tp_info)->pdgId()) == 2212) {
                  if (noPID) {
                    meBarrelTruePNoPID_->Fill((*iTrack)->p());
                  } else if (isPi) {
                    meBarrelTruePAsPi_->Fill((*iTrack)->p());
                  } else if (isK) {
                    meBarrelTruePAsK_->Fill((*iTrack)->p());
                  } else if (isP) {
                    meBarrelTruePAsP_->Fill((*iTrack)->p());
                  } else {
                    edm::LogWarning("Primary4DVertexValidation")
                        << "No PID class: " << std::abs((*tp_info)->pdgId()) << " t0/t0safe " << t0Pid[*iTrack] << " "
                        << t0Safe[*iTrack] << " Prob pi/K/p " << probPi[*iTrack] << " " << probK[*iTrack] << " "
                        << probP[*iTrack];
                  }
                }
              } else if (std::abs((*iTrack)->eta()) > trackMinEtlEta_ && std::abs((*iTrack)->eta()) < trackMaxEtlEta_) {
                meEndcapPIDp_->Fill((*iTrack)->p());
                meEndcapNoPIDtype_->Fill(noPIDtype + 0.5);
                if (std::abs((*tp_info)->pdgId()) == 211) {
                  if (noPID) {
                    meEndcapTruePiNoPID_->Fill((*iTrack)->p());
                  } else if (isPi) {
                    meEndcapTruePiAsPi_->Fill((*iTrack)->p());
                  } else if (isK) {
                    meEndcapTruePiAsK_->Fill((*iTrack)->p());
                  } else if (isP) {
                    meEndcapTruePiAsP_->Fill((*iTrack)->p());
                  } else {
                    edm::LogWarning("Primary4DVertexValidation")
                        << "No PID class: " << std::abs((*tp_info)->pdgId()) << " t0/t0safe " << t0Pid[*iTrack] << " "
                        << t0Safe[*iTrack] << " Prob pi/K/p " << probPi[*iTrack] << " " << probK[*iTrack] << " "
                        << probP[*iTrack];
                  }
                } else if (std::abs((*tp_info)->pdgId()) == 321) {
                  if (noPID) {
                    meEndcapTrueKNoPID_->Fill((*iTrack)->p());
                  } else if (isPi) {
                    meEndcapTrueKAsPi_->Fill((*iTrack)->p());
                  } else if (isK) {
                    meEndcapTrueKAsK_->Fill((*iTrack)->p());
                  } else if (isP) {
                    meEndcapTrueKAsP_->Fill((*iTrack)->p());
                  } else {
                    edm::LogWarning("Primary4DVertexValidation")
                        << "No PID class: " << std::abs((*tp_info)->pdgId()) << " t0/t0safe " << t0Pid[*iTrack] << " "
                        << t0Safe[*iTrack] << " Prob pi/K/p " << probPi[*iTrack] << " " << probK[*iTrack] << " "
                        << probP[*iTrack];
                  }
                } else if (std::abs((*tp_info)->pdgId()) == 2212) {
                  if (noPID) {
                    meEndcapTruePNoPID_->Fill((*iTrack)->p());
                  } else if (isPi) {
                    meEndcapTruePAsPi_->Fill((*iTrack)->p());
                  } else if (isK) {
                    meEndcapTruePAsK_->Fill((*iTrack)->p());
                  } else if (isP) {
                    meEndcapTruePAsP_->Fill((*iTrack)->p());
                  } else {
                    edm::LogWarning("Primary4DVertexValidation")
                        << "No PID class: " << std::abs((*tp_info)->pdgId()) << " t0/t0safe " << t0Pid[*iTrack] << " "
                        << t0Safe[*iTrack] << " Prob pi/K/p " << probPi[*iTrack] << " " << probK[*iTrack] << " "
                        << probP[*iTrack];
                  }
                }
              }
            }
            meTrackResTot_->Fill(t0Safe[*iTrack] - tsim);
            meTrackPullTot_->Fill((t0Safe[*iTrack] - tsim) / sigmat0Safe[*iTrack]);
            meTrackZposResTot_->Fill(dZ);
            if (selectRecoTrk && optionalPlots_) {
              unsigned int no_PIDtype = 0;
              bool no_PID, is_Pi, is_K, is_P;
              isParticle(*iTrack, sigmat0, sigmat0Safe, probPi, probK, probP, no_PIDtype, no_PID, is_Pi, is_K, is_P);
              if (no_PID) {
                meTrackTimeResNoPID_->Fill(t0Safe[*iTrack] - tsim);
                meTrackTimePullNoPID_->Fill((t0Safe[*iTrack] - tsim) / sigmat0Safe[*iTrack]);
                meTrackTimeSigmaNoPID_->Fill(sigmat0Safe[*iTrack]);
                meTrackMVANoPID_->Fill(mtdQualMVA[(*iTrack)]);
                if (no_PIDtype == 1) {
                  meNoPIDTrackTimeResNoPIDType_[0]->Fill(t0Safe[*iTrack] - tsim);
                  meNoPIDTrackTimePullNoPIDType_[0]->Fill((t0Safe[*iTrack] - tsim) / sigmat0Safe[*iTrack]);
                  meNoPIDTrackSigmaNoPIDType_[0]->Fill(sigmat0Safe[*iTrack]);
                  meNoPIDTrackMVANoPIDType_[0]->Fill(mtdQualMVA[(*iTrack)]);
                } else if (no_PIDtype == 2) {
                  meNoPIDTrackTimeResNoPIDType_[1]->Fill(t0Safe[*iTrack] - tsim);
                  meNoPIDTrackTimePullNoPIDType_[1]->Fill((t0Safe[*iTrack] - tsim) / sigmat0Safe[*iTrack]);
                  meNoPIDTrackSigmaNoPIDType_[1]->Fill(sigmat0Safe[*iTrack]);
                  meNoPIDTrackMVANoPIDType_[1]->Fill(mtdQualMVA[(*iTrack)]);
                } else if (no_PIDtype == 3) {
                  meNoPIDTrackTimeResNoPIDType_[2]->Fill(t0Safe[*iTrack] - tsim);
                  meNoPIDTrackTimePullNoPIDType_[2]->Fill((t0Safe[*iTrack] - tsim) / sigmat0Safe[*iTrack]);
                  meNoPIDTrackSigmaNoPIDType_[2]->Fill(sigmat0Safe[*iTrack]);
                  meNoPIDTrackMVANoPIDType_[2]->Fill(mtdQualMVA[(*iTrack)]);
                }
                if (std::abs((*tp_info)->pdgId()) == 211) {
                  meTrackTimeResNoPIDtruePi_->Fill(t0Safe[*iTrack] - tsim);
                  meTrackTimePullNoPIDtruePi_->Fill((t0Safe[*iTrack] - tsim) / sigmat0Safe[*iTrack]);
                } else if (std::abs((*tp_info)->pdgId()) == 321) {
                  meTrackTimeResNoPIDtrueK_->Fill(t0Safe[*iTrack] - tsim);
                  meTrackTimePullNoPIDtrueK_->Fill((t0Safe[*iTrack] - tsim) / sigmat0Safe[*iTrack]);
                } else if (std::abs((*tp_info)->pdgId()) == 2212) {
                  meTrackTimeResNoPIDtrueP_->Fill(t0Safe[*iTrack] - tsim);
                  meTrackTimePullNoPIDtrueP_->Fill((t0Safe[*iTrack] - tsim) / sigmat0Safe[*iTrack]);
                }
              } else if ((is_Pi && std::abs((*tp_info)->pdgId()) == 211) ||
                         (is_K && std::abs((*tp_info)->pdgId()) == 321) ||
                         (is_P && std::abs((*tp_info)->pdgId()) == 2212)) {
                meTrackTimeResCorrectPID_->Fill(t0Safe[*iTrack] - tsim);
                meTrackTimePullCorrectPID_->Fill((t0Safe[*iTrack] - tsim) / sigmat0Safe[*iTrack]);
                meTrackTimeSigmaCorrectPID_->Fill(sigmat0Safe[*iTrack]);
                meTrackMVACorrectPID_->Fill(mtdQualMVA[(*iTrack)]);
              } else {
                meTrackTimeResWrongPID_->Fill(t0Safe[*iTrack] - tsim);
                meTrackTimePullWrongPID_->Fill((t0Safe[*iTrack] - tsim) / sigmat0Safe[*iTrack]);
                meTrackTimeSigmaWrongPID_->Fill(sigmat0Safe[*iTrack]);
                meTrackMVAWrongPID_->Fill(mtdQualMVA[(*iTrack)]);
              }
            }

            if ((*iTrack)->p() <= 2) {
              meTrackResLowPTot_->Fill(t0Safe[*iTrack] - tsim);
              meTrackPullLowPTot_->Fill((t0Safe[*iTrack] - tsim) / sigmat0Safe[*iTrack]);
            } else {
              meTrackResHighPTot_->Fill(t0Safe[*iTrack] - tsim);
              meTrackPullHighPTot_->Fill((t0Safe[*iTrack] - tsim) / sigmat0Safe[*iTrack]);
            }

            if (mtdQualMVA[(*iTrack)] < mvaL_) {
              meTrackZposRes_[0]->Fill(dZ);
              meTrack3DposRes_[0]->Fill(d3D);
              meTrackRes_[0]->Fill(t0Safe[*iTrack] - tsim);
              meTrackPull_[0]->Fill((t0Safe[*iTrack] - tsim) / sigmat0Safe[*iTrack]);

              if (optionalPlots_) {
                meTrackResMass_[0]->Fill(t0Safe[*iTrack] - tEst);
                meTrackResMassTrue_[0]->Fill(tEst - tsim);
              }

              if ((*iTrack)->p() <= 2) {
                meTrackResLowP_[0]->Fill(t0Safe[*iTrack] - tsim);
                meTrackPullLowP_[0]->Fill((t0Safe[*iTrack] - tsim) / sigmat0Safe[*iTrack]);
              } else if ((*iTrack)->p() > 2) {
                meTrackResHighP_[0]->Fill(t0Safe[*iTrack] - tsim);
                meTrackPullHighP_[0]->Fill((t0Safe[*iTrack] - tsim) / sigmat0Safe[*iTrack]);
              }

              if (optionalPlots_) {
                if (std::abs((*tp_info)->pdgId()) == 2212) {
                  meTrackResMassProtons_[0]->Fill(t0Safe[*iTrack] - tEst);
                  meTrackResMassTrueProtons_[0]->Fill(tEst - tsim);
                } else if (std::abs((*tp_info)->pdgId()) == 211) {
                  meTrackResMassPions_[0]->Fill(t0Safe[*iTrack] - tEst);
                  meTrackResMassTruePions_[0]->Fill(tEst - tsim);
                }
              }

            } else if (mtdQualMVA[(*iTrack)] > mvaL_ && mtdQualMVA[(*iTrack)] < mvaH_) {
              meTrackZposRes_[1]->Fill(dZ);
              meTrack3DposRes_[1]->Fill(d3D);
              meTrackRes_[1]->Fill(t0Safe[*iTrack] - tsim);
              meTrackPull_[1]->Fill((t0Safe[*iTrack] - tsim) / sigmat0Safe[*iTrack]);

              if (optionalPlots_) {
                meTrackResMass_[1]->Fill(t0Safe[*iTrack] - tEst);
                meTrackResMassTrue_[1]->Fill(tEst - tsim);
              }

              if ((*iTrack)->p() <= 2) {
                meTrackResLowP_[1]->Fill(t0Safe[*iTrack] - tsim);
                meTrackPullLowP_[1]->Fill((t0Safe[*iTrack] - tsim) / sigmat0Safe[*iTrack]);
              } else if ((*iTrack)->p() > 2) {
                meTrackResHighP_[1]->Fill(t0Safe[*iTrack] - tsim);
                meTrackPullHighP_[1]->Fill((t0Safe[*iTrack] - tsim) / sigmat0Safe[*iTrack]);
              }

              if (optionalPlots_) {
                if (std::abs((*tp_info)->pdgId()) == 2212) {
                  meTrackResMassProtons_[1]->Fill(t0Safe[*iTrack] - tEst);
                  meTrackResMassTrueProtons_[1]->Fill(tEst - tsim);
                } else if (std::abs((*tp_info)->pdgId()) == 211) {
                  meTrackResMassPions_[1]->Fill(t0Safe[*iTrack] - tEst);
                  meTrackResMassTruePions_[1]->Fill(tEst - tsim);
                }
              }

            } else if (mtdQualMVA[(*iTrack)] > mvaH_) {
              meTrackZposRes_[2]->Fill(dZ);
              meTrack3DposRes_[2]->Fill(d3D);
              meTrackRes_[2]->Fill(t0Safe[*iTrack] - tsim);
              meTrackPull_[2]->Fill((t0Safe[*iTrack] - tsim) / sigmat0Safe[*iTrack]);

              if (optionalPlots_) {
                meTrackResMass_[2]->Fill(t0Safe[*iTrack] - tEst);
                meTrackResMassTrue_[2]->Fill(tEst - tsim);
              }

              if ((*iTrack)->p() <= 2) {
                meTrackResLowP_[2]->Fill(t0Safe[*iTrack] - tsim);
                meTrackPullLowP_[2]->Fill((t0Safe[*iTrack] - tsim) / sigmat0Safe[*iTrack]);
              } else if ((*iTrack)->p() > 2) {
                meTrackResHighP_[2]->Fill(t0Safe[*iTrack] - tsim);
                meTrackPullHighP_[2]->Fill((t0Safe[*iTrack] - tsim) / sigmat0Safe[*iTrack]);
              }

              if (optionalPlots_) {
                if (std::abs((*tp_info)->pdgId()) == 2212) {
                  meTrackResMassProtons_[2]->Fill(t0Safe[*iTrack] - tEst);
                  meTrackResMassTrueProtons_[2]->Fill(tEst - tsim);
                } else if (std::abs((*tp_info)->pdgId()) == 211) {
                  meTrackResMassPions_[2]->Fill(t0Safe[*iTrack] - tEst);
                  meTrackResMassTruePions_[2]->Fill(tEst - tsim);
                }
              }
            }
          }  // if tp_info != nullptr && MatchCategory == 0
        }  // loop on reco tracks
        if (selectedVtxMatching) {
          meVtxTrackMult_->Fill(log10(nt));
          mePUTrackRelMult_->Fill(static_cast<double>(PUnt) / nt);
          meFakeTrackRelMult_->Fill(static_cast<double>(Fakent) / nt);
          mePUTrackRelSumWnt_->Fill(PUsumWnt / sumWnt);
          mePUTrackRelSumWos_->Fill(PUsumWos / sumWos);
          meSecTrackRelSumWos_->Fill(SecsumWos / sumWos);
          meFakeTrackRelSumWos_->Fill(FakesumWos / sumWos);
          mePUTrackRelSumPt_->Fill(PUsumPt / sumPt);
          mePUTrackRelSumPt2_->Fill(PUsumPt2 / sumPt2);

          observablesFromJets(
              recotracks, massVector, categoryVector, "use_allTracks", nJets, sumEtJets, sumPt2Jets, metPt, sumPzJets);
          observablesFromJets(recotracks,
                              massVector,
                              categoryVector,
                              "skip_Fake",
                              nJetsnoFake,
                              sumEtJetsnoFake,
                              sumPt2JetsnoFake,
                              metPtnoFake,
                              sumPzJetsnoFake);
          observablesFromJets(recotracks,
                              massVector,
                              categoryVector,
                              "skip_PU",
                              nJetsnoPU,
                              sumEtJetsnoPU,
                              sumPt2JetsnoPU,
                              metPtnoPU,
                              sumPzJetsnoPU);

          meJetsPURelMult_->Fill(static_cast<double>(nJets - nJetsnoPU) / nJets);
          meJetsPURelHt_->Fill((sumEtJets - sumEtJetsnoPU) / sumEtJets);
          meJetsPURelSumPt2_->Fill((sumPt2Jets - sumPt2JetsnoPU) / sumPt2Jets);
          meJetsFakeRelSumPt2_->Fill((sumPt2Jets - sumPt2JetsnoFake) / sumPt2Jets);
          meJetsPURelMetPt_->Fill((metPt - metPtnoPU) / metPt);
          meJetsPURelSumPz_->Fill((sumPzJets - sumPzJetsnoPU) / sumPzJets);

          if (optionalPlots_) {
            mePUTrackMult_->Fill(PUnt);
            mePUTrackSumWnt_->Fill(log10(std::max(minThrSumWnt_, PUsumWnt)));
            mePUTrackSumWos_->Fill(log10(std::max(minThrSumWos_, PUsumWos)));
            meSecTrackSumWos_->Fill(log10(std::max(minThrSumWos_, SecsumWos)));
            mePUTrackSumPt_->Fill(log10(std::max(minThrSumPt_, PUsumPt)));
            mePUTrackSumPt2_->Fill(log10(std::max(minThrSumPt2_, PUsumPt2)));

            mePUTrackRelMultvsMult_->Fill(nt, static_cast<double>(PUnt) / nt);
            meFakeTrackRelMultvsMult_->Fill(nt, static_cast<double>(Fakent) / nt);
            mePUTrackRelSumWntvsSumWnt_->Fill(log10(std::max(minThrSumWnt_, sumWnt)), PUsumWnt / sumWnt);
            mePUTrackRelSumWosvsSumWos_->Fill(log10(std::max(minThrSumWos_, sumWos)), PUsumWos / sumWos);
            meSecTrackRelSumWosvsSumWos_->Fill(log10(std::max(minThrSumWos_, sumWos)), SecsumWos / sumWos);
            meFakeTrackRelSumWosvsSumWos_->Fill(log10(std::max(minThrSumWos_, sumWos)), FakesumWos / sumWos);
            mePUTrackRelSumPtvsSumPt_->Fill(log10(std::max(minThrSumPt_, sumPt)), PUsumPt / sumPt);
            mePUTrackRelSumPt2vsSumPt2_->Fill(log10(std::max(minThrSumPt2_, sumPt2)), PUsumPt2 / sumPt2);

            meJetsPUMult_->Fill(nJets - nJetsnoPU);
            meJetsPUHt_->Fill(log10(std::max(minThrSumPt_, sumEtJets - sumEtJetsnoPU)));
            meJetsPUSumPt2_->Fill(log10(std::max(minThrSumPt2_, sumPt2Jets - sumPt2JetsnoPU)));
            meJetsPUMetPt_->Fill(log10(std::max(minThrMetPt_, metPt - metPtnoPU)));
            meJetsPUSumPz_->Fill(log10(std::max(minThrSumPz_, std::abs(sumPzJets - sumPzJetsnoPU))));

            meJetsPURelMultvsMult_->Fill(nJets, static_cast<double>(nJets - nJetsnoPU) / nJets);
            meJetsPURelHtvsHt_->Fill(log10(std::max(minThrSumPt_, sumEtJets)), (sumEtJets - sumEtJetsnoPU) / sumEtJets);
            meJetsPURelSumPt2vsSumPt2_->Fill(log10(std::max(minThrSumPt2_, sumPt2Jets)),
                                             (sumPt2Jets - sumPt2JetsnoPU) / sumPt2Jets);
            meJetsFakeRelSumPt2vsSumPt2_->Fill(log10(std::max(minThrSumPt2_, sumPt2Jets)),
                                               (sumPt2Jets - sumPt2JetsnoFake) / sumPt2Jets);
            meJetsPURelMetPtvsMetPt_->Fill(log10(std::max(minThrMetPt_, metPt)), (metPt - metPtnoPU) / metPt);
            meJetsPURelSumPzvsSumPz_->Fill(log10(std::max(minThrSumPz_, std::abs(sumPzJets))),
                                           (sumPzJets - sumPzJetsnoPU) / sumPzJets);
          }
          if (selectedLV) {
            meVtxTrackRecLVMult_->Fill(log10(nt));
            mePUTrackRecLVRelMult_->Fill(static_cast<double>(PUnt) / nt);
            meFakeTrackRecLVRelMult_->Fill(static_cast<double>(Fakent) / nt);
            mePUTrackRecLVRelSumWnt_->Fill(PUsumWnt / sumWnt);
            mePUTrackRecLVRelSumWos_->Fill(PUsumWos / sumWos);
            meSecTrackRecLVRelSumWos_->Fill(SecsumWos / sumWos);
            meFakeTrackRecLVRelSumWos_->Fill(FakesumWos / sumWos);
            mePUTrackRecLVRelSumPt_->Fill(PUsumPt / sumPt);
            mePUTrackRecLVRelSumPt2_->Fill(PUsumPt2 / sumPt2);

            meJetsRecLVPURelMult_->Fill(static_cast<double>(nJets - nJetsnoPU) / nJets);
            meJetsRecLVPURelHt_->Fill((sumEtJets - sumEtJetsnoPU) / sumEtJets);
            meJetsRecLVPURelSumPt2_->Fill((sumPt2Jets - sumPt2JetsnoPU) / sumPt2Jets);
            meJetsRecLVFakeRelSumPt2_->Fill((sumPt2Jets - sumPt2JetsnoFake) / sumPt2Jets);
            meJetsRecLVPURelMetPt_->Fill((metPt - metPtnoPU) / metPt);
            meJetsRecLVPURelSumPz_->Fill((sumPzJets - sumPzJetsnoPU) / sumPzJets);

            LogTrace("Primary4DVertexValidation")
                << "#PUTrks = " << PUnt << " #Trks = " << nt << " PURelMult = " << std::setprecision(3)
                << static_cast<double>(PUnt) / nt;
            LogTrace("Primary4DVertexValidation")
                << "PUsumWnt = " << std::setprecision(3) << PUsumWnt << " sumWnt = " << std::setprecision(3) << sumWnt
                << " PURelsumWnt = " << std::setprecision(3) << PUsumWnt / sumWnt;
            LogTrace("Primary4DVertexValidation")
                << "PUsumWos = " << std::setprecision(3) << PUsumWos << " sumWos = " << std::setprecision(3) << sumWos
                << " PURelsumWos = " << std::setprecision(3) << PUsumWos / sumWos;
            LogTrace("Primary4DVertexValidation")
                << "PuSumPt = " << std::setprecision(3) << PUsumPt << " SumPt = " << std::setprecision(4) << sumPt
                << " PURelSumPt = " << std::setprecision(3) << PUsumPt / sumPt;
            LogTrace("Primary4DVertexValidation")
                << "PuSumPt2 = " << std::setprecision(3) << PUsumPt2 << " SumPt2 = " << std::setprecision(4) << sumPt2
                << " PURelSumPt2 = " << std::setprecision(3) << PUsumPt2 / sumPt2;
            if (optionalPlots_) {
              mePUTrackRecLVMult_->Fill(PUnt);
              mePUTrackRecLVSumWnt_->Fill(log10(std::max(minThrSumWnt_, PUsumWnt)));
              mePUTrackRecLVSumWos_->Fill(log10(std::max(minThrSumWos_, PUsumWos)));
              meSecTrackRecLVSumWos_->Fill(log10(std::max(minThrSumWos_, PUsumWos)));
              mePUTrackRecLVSumPt_->Fill(log10(std::max(minThrSumPt_, PUsumPt)));
              mePUTrackRecLVSumPt2_->Fill(log10(std::max(minThrSumPt2_, PUsumPt2)));

              mePUTrackRecLVRelMultvsMult_->Fill(nt, static_cast<double>(PUnt) / nt);
              meFakeTrackRecLVRelMultvsMult_->Fill(nt, static_cast<double>(Fakent) / nt);
              mePUTrackRecLVRelSumWntvsSumWnt_->Fill(log10(std::max(minThrSumWnt_, sumWnt)), PUsumWnt / sumWnt);
              mePUTrackRecLVRelSumWosvsSumWos_->Fill(log10(std::max(minThrSumWos_, sumWos)), PUsumWos / sumWos);
              meSecTrackRecLVRelSumWosvsSumWos_->Fill(log10(std::max(minThrSumWos_, sumWos)), SecsumWos / sumWos);
              meFakeTrackRecLVRelSumWosvsSumWos_->Fill(log10(std::max(minThrSumWos_, sumWos)), FakesumWos / sumWos);
              mePUTrackRecLVRelSumPtvsSumPt_->Fill(log10(std::max(minThrSumPt_, sumPt)), PUsumPt / sumPt);
              mePUTrackRecLVRelSumPt2vsSumPt2_->Fill(log10(std::max(minThrSumPt2_, sumPt2)), PUsumPt2 / sumPt2);

              meJetsRecLVPUMult_->Fill(nJets - nJetsnoPU);
              meJetsRecLVPUHt_->Fill(log10(std::max(minThrSumPt_, sumEtJets - sumEtJetsnoPU)));
              meJetsRecLVPUSumPt2_->Fill(log10(std::max(minThrSumPt2_, sumPt2Jets - sumPt2JetsnoPU)));
              meJetsRecLVPUMetPt_->Fill(log10(std::max(minThrMetPt_, metPt - metPtnoPU)));
              meJetsRecLVPUSumPz_->Fill(log10(std::max(minThrSumPz_, std::abs(sumPzJets - sumPzJetsnoPU))));

              meJetsRecLVPURelMultvsMult_->Fill(nJets, static_cast<double>(nJets - nJetsnoPU) / nJets);
              meJetsRecLVPURelHtvsHt_->Fill(log10(std::max(minThrSumPt_, sumEtJets)),
                                            (sumEtJets - sumEtJetsnoPU) / sumEtJets);
              meJetsRecLVPURelSumPt2vsSumPt2_->Fill(log10(std::max(minThrSumPt2_, sumPt2Jets)),
                                                    (sumPt2Jets - sumPt2JetsnoPU) / sumPt2Jets);
              meJetsRecLVFakeRelSumPt2vsSumPt2_->Fill(log10(std::max(minThrSumPt2_, sumPt2Jets)),
                                                      (sumPt2Jets - sumPt2JetsnoFake) / sumPt2Jets);
              meJetsRecLVPURelMetPtvsMetPt_->Fill(log10(std::max(minThrMetPt_, metPt)), (metPt - metPtnoPU) / metPt);
              meJetsRecLVPURelSumPzvsSumPz_->Fill(log10(std::max(minThrSumPz_, std::abs(sumPzJets))),
                                                  (sumPzJets - sumPzJetsnoPU) / sumPzJets);
            }
          }
        }
      }  // loop on simpv
    }  // ndof
  }  // loop on recopv

  int real = 0;
  int fake = 0;
  int other_fake = 0;
  int split = 0;

  meRecVerNumber_->Fill(recopv.size());
  for (unsigned int ir = 0; ir < recopv.size(); ir++) {
    const reco::Vertex* vertex = recopv.at(ir).recVtx;
    if (recopv.at(ir).ndof > selNdof_) {
      meRecPVZ_->Fill(recopv.at(ir).z);
      meVtxTrackMultPassNdof_->Fill(log10(vertex->tracksSize()));

      if (recopv.at(ir).recVtx->tError() > 0.) {
        meRecPVT_->Fill(recopv.at(ir).recVtx->t());
      }
      LogTrace("Primary4DVertexValidation") << "************* IR: " << ir;
      LogTrace("Primary4DVertexValidation") << "is_real: " << recopv.at(ir).is_real();
      LogTrace("Primary4DVertexValidation") << "is_fake: " << recopv.at(ir).is_fake();
      LogTrace("Primary4DVertexValidation") << "is_signal: " << recopv.at(ir).is_signal();
      LogTrace("Primary4DVertexValidation") << "split_from: " << recopv.at(ir).split_from();
      LogTrace("Primary4DVertexValidation") << "other fake: " << recopv.at(ir).other_fake();
      if (recopv.at(ir).is_real()) {
        real++;
      }
      if (recopv.at(ir).is_fake()) {
        fake++;
      }
      if (recopv.at(ir).other_fake()) {
        other_fake++;
      }
      if (recopv.at(ir).split_from() != -1) {
        split++;
      }
    }  // ndof
    else {
      meVtxTrackMultFailNdof_->Fill(vertex->tracksSize());
    }
  }

  LogTrace("Primary4DVertexValidation") << "is_real: " << real;
  LogTrace("Primary4DVertexValidation") << "is_fake: " << fake;
  LogTrace("Primary4DVertexValidation") << "split_from: " << split;
  LogTrace("Primary4DVertexValidation") << "other fake: " << other_fake;
  mePUvsRealV_->Fill(simpv.size(), real);
  mePUvsFakeV_->Fill(simpv.size(), fake);
  mePUvsOtherFakeV_->Fill(simpv.size(), other_fake);
  mePUvsSplitV_->Fill(simpv.size(), split);

  // fill vertices histograms here in a new loop
  meSimVerNumber_->Fill(simpv.size());
  for (unsigned int is = 0; is < simpv.size(); is++) {
    meSimPVZ_->Fill(simpv.at(is).z);
    meSimPVT_->Fill(simpv.at(is).t * simUnit_);
    meSimPVTvsZ_->Fill(simpv.at(is).z, simpv.at(is).t * simUnit_);
    if (is == 0 && optionalPlots_) {
      meSimPosInSimOrigCollection_->Fill(simpv.at(is).OriginalIndex);
    }

    if (simpv.at(is).rec == NOT_MATCHED) {
      LogTrace("Primary4DVertexValidation") << "sim vertex: " << is << " is not matched with any reco";
      continue;
    }

    for (unsigned int ir = 0; ir < recopv.size(); ir++) {
      if (recopv.at(ir).ndof > selNdof_) {
        if (recopv.at(ir).sim == is && simpv.at(is).rec == ir) {
          meTimeRes_->Fill(recopv.at(ir).recVtx->t() - simpv.at(is).t * simUnit_);
          meTimePull_->Fill((recopv.at(ir).recVtx->t() - simpv.at(is).t * simUnit_) / recopv.at(ir).recVtx->tError());
          meMatchQual_->Fill(recopv.at(ir).matchQuality - 0.5);
          if (ir == 0) {  // signal vertex plots
            meTimeSignalRes_->Fill(recopv.at(ir).recVtx->t() - simpv.at(is).t * simUnit_);
            meTimeSignalPull_->Fill((recopv.at(ir).recVtx->t() - simpv.at(is).t * simUnit_) /
                                    recopv.at(ir).recVtx->tError());
            if (optionalPlots_) {
              meRecoPosInSimCollection_->Fill(recopv.at(ir).sim);
              meRecoPosInRecoOrigCollection_->Fill(recopv.at(ir).OriginalIndex);
            }
          }
          if (simpv.at(is).eventId.bunchCrossing() == 0 && simpv.at(is).eventId.event() == 0) {
            if (!recopv.at(ir).is_signal()) {
              edm::LogWarning("Primary4DVertexValidation")
                  << "Reco vtx leading match inconsistent: BX/ID " << simpv.at(is).eventId.bunchCrossing() << " "
                  << simpv.at(is).eventId.event();
            }
            meRecoPVPosSignal_->Fill(
                recopv.at(ir).OriginalIndex);  // position in reco vtx correction associated to sim signal
            if (!signal_is_highest_pt) {
              meRecoPVPosSignalNotHighestPt_->Fill(
                  recopv.at(ir).OriginalIndex);  // position in reco vtx correction associated to sim signal
            }
          }
          LogTrace("Primary4DVertexValidation") << "*** Matching RECO: " << ir << "with SIM: " << is << " ***";
          LogTrace("Primary4DVertexValidation") << "Match Quality is " << recopv.at(ir).matchQuality;
          LogTrace("Primary4DVertexValidation") << "****";
        }
      }  // ndof
    }
  }

  // dz histos
  for (unsigned int iv = 0; iv < recVtxs->size() - 1; iv++) {
    if (recVtxs->at(iv).ndof() > selNdof_) {
      double mindistance_realreal = 1e10;

      for (unsigned int jv = iv; jv < recVtxs->size(); jv++) {
        if ((!(jv == iv)) && select(recVtxs->at(jv))) {
          double dz = recVtxs->at(iv).z() - recVtxs->at(jv).z();
          double dtsigma = std::sqrt(recVtxs->at(iv).covariance(3, 3) + recVtxs->at(jv).covariance(3, 3));
          double dt = (std::abs(dz) <= deltaZcut_ && dtsigma > 0.)
                          ? (recVtxs->at(iv).t() - recVtxs->at(jv).t()) / dtsigma
                          : -9999.;
          if (recopv.at(iv).is_real() && recopv.at(jv).is_real()) {
            meDeltaZrealreal_->Fill(std::abs(dz));
            if (dt != -9999.) {
              meDeltaTrealreal_->Fill(std::abs(dt));
            }
            if (std::abs(dz) < std::abs(mindistance_realreal)) {
              mindistance_realreal = dz;
            }
          } else if (recopv.at(iv).is_fake() && recopv.at(jv).is_fake()) {
            meDeltaZfakefake_->Fill(std::abs(dz));
            if (dt != -9999.) {
              meDeltaTfakefake_->Fill(std::abs(dt));
            }
          }
        }
      }

      double mindistance_fakereal = 1e10;
      double mindistance_realfake = 1e10;
      for (unsigned int jv = 0; jv < recVtxs->size(); jv++) {
        if ((!(jv == iv)) && select(recVtxs->at(jv))) {
          double dz = recVtxs->at(iv).z() - recVtxs->at(jv).z();
          double dtsigma = std::sqrt(recVtxs->at(iv).covariance(3, 3) + recVtxs->at(jv).covariance(3, 3));
          double dt = (std::abs(dz) <= deltaZcut_ && dtsigma > 0.)
                          ? (recVtxs->at(iv).t() - recVtxs->at(jv).t()) / dtsigma
                          : -9999.;
          if (recopv.at(iv).is_fake() && recopv.at(jv).is_real()) {
            meDeltaZfakereal_->Fill(std::abs(dz));
            if (dt != -9999.) {
              meDeltaTfakereal_->Fill(std::abs(dt));
            }
            if (std::abs(dz) < std::abs(mindistance_fakereal)) {
              mindistance_fakereal = dz;
            }
          }

          if (recopv.at(iv).is_real() && recopv.at(jv).is_fake() && (std::abs(dz) < std::abs(mindistance_realfake))) {
            mindistance_realfake = dz;
          }
        }
      }
    }  // ndof
  }

}  // end of analyze

void Primary4DVertexValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/Vertices");
  desc.add<edm::InputTag>("TPtoRecoTrackAssoc", edm::InputTag("trackingParticleRecoTrackAsssociation"));
  desc.add<edm::InputTag>("TrackLabel", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("mtdTracks", edm::InputTag("trackExtenderWithMTD"));
  desc.add<edm::InputTag>("SimTag", edm::InputTag("mix", "MergedTrackTruth"));
  desc.add<edm::InputTag>("offlineBS", edm::InputTag("offlineBeamSpot"));
  desc.add<edm::InputTag>("offline4DPV", edm::InputTag("offlinePrimaryVertices4D"));
  desc.add<edm::InputTag>("trackAssocSrc", edm::InputTag("trackExtenderWithMTD:generalTrackassoc"))
      ->setComment("Association between General and MTD Extended tracks");
  desc.add<edm::InputTag>("pathLengthSrc", edm::InputTag("trackExtenderWithMTD:generalTrackPathLength"));
  desc.add<edm::InputTag>("momentumSrc", edm::InputTag("trackExtenderWithMTD:generalTrackp"));
  desc.add<edm::InputTag>("tmtd", edm::InputTag("trackExtenderWithMTD:generalTracktmtd"));
  desc.add<edm::InputTag>("timeSrc", edm::InputTag("trackExtenderWithMTD:generalTracktmtd"));
  desc.add<edm::InputTag>("sigmaSrc", edm::InputTag("trackExtenderWithMTD:generalTracksigmatmtd"));
  desc.add<edm::InputTag>("t0PID", edm::InputTag("tofPID:t0"));
  desc.add<edm::InputTag>("sigmat0PID", edm::InputTag("tofPID:sigmat0"));
  desc.add<edm::InputTag>("t0SafePID", edm::InputTag("tofPID:t0safe"));
  desc.add<edm::InputTag>("sigmat0SafePID", edm::InputTag("tofPID:sigmat0safe"));
  desc.add<edm::InputTag>("trackMVAQual", edm::InputTag("mtdTrackQualityMVA:mtdQualMVA"));
  desc.add<edm::InputTag>("tofPi", edm::InputTag("trackExtenderWithMTD:generalTrackTofPi"));
  desc.add<edm::InputTag>("tofK", edm::InputTag("trackExtenderWithMTD:generalTrackTofK"));
  desc.add<edm::InputTag>("tofP", edm::InputTag("trackExtenderWithMTD:generalTrackTofP"));
  desc.add<edm::InputTag>("probPi", edm::InputTag("tofPID:probPi"));
  desc.add<edm::InputTag>("probK", edm::InputTag("tofPID:probK"));
  desc.add<edm::InputTag>("probP", edm::InputTag("tofPID:probP"));
  desc.add<bool>("useOnlyChargedTracks", true);
  desc.addUntracked<bool>("optionalPlots", false);
  desc.add<bool>("use3dNoTime", false);
  desc.add<double>("trackweightTh", 0.5);
  desc.add<double>("mvaTh", 0.8);
  desc.add<double>("minProbHeavy", 0.75);
  {
    edm::ParameterSetDescription psd0;
    HITrackFilterForPVFinding::fillPSetDescription(psd0);  // extension of TrackFilterForPVFinding
    desc.add<edm::ParameterSetDescription>("TkFilterParameters", psd0);
  }

  descriptions.add("vertices4DValid", desc);
}

void Primary4DVertexValidation::printMatchedRecoTrackInfo(const reco::Vertex& vtx,
                                                          const reco::TrackBaseRef& trk,
                                                          const TrackingParticleRef& tp,
                                                          const unsigned int& categ) {
  std::string strTrk;
  switch (categ) {
    case 0:
      strTrk = "Reco_Track:";
      break;
    case 1:
      strTrk = "SecRecoTrk:";
      break;
    case 2:
      strTrk = "PU_RecoTrk:";
      break;
  }
  LogTrace("Primary4DVertexValidation") << strTrk << " w =" << std::setw(6) << std::setprecision(2)
                                        << vtx.trackWeight(trk) << " pt =" << std::setw(6) << std::setprecision(2)
                                        << trk->pt() << " eta =" << std::setw(6) << std::setprecision(2) << trk->eta()
                                        << "  MatchedTP: Pt =" << std::setw(6) << std::setprecision(2) << tp->pt()
                                        << " eta =" << std::setw(6) << std::setprecision(2) << tp->eta()
                                        << "  Parent vtx: z =" << std::setw(8) << std::setprecision(4)
                                        << tp->parentVertex()->position().z() << " t =" << std::setw(8)
                                        << std::setprecision(4) << tp->parentVertex()->position().t() * simUnit_
                                        << " BX =" << tp->parentVertex()->eventId().bunchCrossing()
                                        << " ev =" << tp->parentVertex()->eventId().event() << std::endl;
}

void Primary4DVertexValidation::printSimVtxRecoVtxInfo(
    const struct Primary4DVertexValidation::simPrimaryVertex& simpVtx,
    const struct Primary4DVertexValidation::recoPrimaryVertex& recopVtx) {
  LogTrace("Primary4DVertexValidation") << "Sim vtx (x,y,z,t) = (" << std::setprecision(4) << simpVtx.x << ","
                                        << std::setprecision(4) << simpVtx.y << "," << std::setprecision(4) << simpVtx.z
                                        << "," << std::setprecision(4) << simpVtx.t * simUnit_ << ")"
                                        << " Simvtx.rec = " << simpVtx.rec;
  LogTrace("Primary4DVertexValidation") << "Sim vtx: pt = " << std::setprecision(4) << simpVtx.pt
                                        << " ptsq = " << std::setprecision(6) << simpVtx.ptsq
                                        << " nGenTrk = " << simpVtx.nGenTrk
                                        << " nmatch recotrks = " << simpVtx.num_matched_reco_tracks;
  LogTrace("Primary4DVertexValidation") << "Reco vtx (x,y,z) = (" << std::setprecision(4) << recopVtx.x << ","
                                        << std::setprecision(4) << recopVtx.y << "," << std::setprecision(4)
                                        << recopVtx.z << ")"
                                        << " Recovtx.sim = " << recopVtx.sim;
  LogTrace("Primary4DVertexValidation") << "Reco vtx: pt = " << std::setprecision(4) << recopVtx.pt
                                        << " ptsq = " << std::setprecision(6) << recopVtx.ptsq
                                        << " nrecotrks = " << recopVtx.nRecoTrk
                                        << " nmatch simtrks = " << recopVtx.num_matched_sim_tracks;
  LogTrace("Primary4DVertexValidation") << "wnt " << recopVtx.sumwnt << " wos = " << recopVtx.sumwos;
  for (auto iTP = simpVtx.sim_vertex->daughterTracks_begin(); iTP != simpVtx.sim_vertex->daughterTracks_end(); ++iTP) {
    if (use_only_charged_tracks_ && (**iTP).charge() == 0) {
      continue;
    }
    LogTrace("Primary4DVertexValidation")
        << "Daughter track of sim vertex: pt =" << std::setw(6) << std::setprecision(2) << (*iTP)->pt()
        << "  eta =" << std::setw(6) << std::setprecision(2) << (*iTP)->eta();
  }
}

const bool Primary4DVertexValidation::trkTPSelLV(const TrackingParticle& tp) {
  bool match = false;
  if (tp.status() != 1) {
    return match;
  }

  auto x_pv = tp.parentVertex()->position().x();
  auto y_pv = tp.parentVertex()->position().y();
  auto z_pv = tp.parentVertex()->position().z();

  auto r_pv = std::sqrt(x_pv * x_pv + y_pv * y_pv);

  match =
      tp.charge() != 0 && std::abs(tp.eta()) < etacutGEN_ && tp.pt() > pTcut_ && r_pv < rBTL_ && std::abs(z_pv) < zETL_;
  return match;
}

const bool Primary4DVertexValidation::trkRecSel(const reco::TrackBase& trk) {
  bool match = false;
  match = std::abs(trk.eta()) <= etacutREC_ && trk.pt() > pTcut_;
  return match;
}

DEFINE_FWK_MODULE(Primary4DVertexValidation);
