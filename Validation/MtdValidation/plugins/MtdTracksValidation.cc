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
#include "DataFormats/ForwardDetId/interface/ETLDetId.h"
#include "DataFormats/ForwardDetId/interface/BTLDetId.h"

#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "Geometry/Records/interface/MTDDigiGeometryRecord.h"
#include "Geometry/Records/interface/MTDTopologyRcd.h"
#include "Geometry/MTDNumberingBuilder/interface/MTDTopology.h"
#include "Geometry/MTDCommonData/interface/MTDTopologyMode.h"
#include "Geometry/MTDGeometryBuilder/interface/MTDGeometry.h"
#include "Geometry/MTDGeometryBuilder/interface/ProxyMTDTopology.h"
#include "Geometry/MTDGeometryBuilder/interface/RectangularMTDTopology.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "HepMC/GenRanges.h"
#include "DataFormats/Math/interface/GeantUnits.h"
#include "CLHEP/Units/PhysicalConstants.h"

class MtdTracksValidation : public DQMEDAnalyzer {
public:
  explicit MtdTracksValidation(const edm::ParameterSet&);
  ~MtdTracksValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  const bool mvaGenSel(const HepMC::GenParticle&, const float&);
  const bool mvaRecSel(const reco::TrackBase&, const reco::Vertex&, const double&, const double&);
  const bool mvaGenRecMatch(const HepMC::GenParticle&, const double&, const reco::TrackBase&);

  // ------------ member data ------------

  const std::string folder_;
  const float trackMinPt_;
  const float trackMinEta_;
  const float trackMaxEta_;

  static constexpr double etacutGEN_ = 4.;     // |eta| < 4;
  static constexpr double etacutREC_ = 3.;     // |eta| < 3;
  static constexpr double pTcut_ = 0.7;        // PT > 0.7 GeV
  static constexpr double deltaZcut_ = 0.1;    // dz separation 1 mm
  static constexpr double deltaPTcut_ = 0.05;  // dPT < 5%
  static constexpr double deltaDRcut_ = 0.03;  // DeltaR separation

  static constexpr float c_cm_ns = geant_units::operators::convertMmToCm(CLHEP::c_light);  // [mm/ns] -> [cm/ns]

  const bool testPID_;

  edm::EDGetTokenT<reco::TrackCollection> GenRecTrackToken_;
  edm::EDGetTokenT<reco::TrackCollection> RecTrackToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> RecVertexToken_;

  edm::EDGetTokenT<edm::HepMCProduct> HepMCProductToken_;

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
  edm::EDGetTokenT<edm::ValueMap<float>> trackMVAQualToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> tofPiToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> tofKToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> tofPToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> probPiToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> probKToken_;
  edm::EDGetTokenT<edm::ValueMap<float>> probPToken_;

  edm::ESGetToken<MTDTopology, MTDTopologyRcd> mtdtopoToken_;
  edm::ESGetToken<HepPDT::ParticleDataTable, edm::DefaultRecord> particleTableToken_;

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
  MonitorElement* meETLTrackEffPhiTot_[2];
  MonitorElement* meETLTrackEffPtTot_[2];
  MonitorElement* meETLTrackEffEtaMtd_[2];
  MonitorElement* meETLTrackEffPhiMtd_[2];
  MonitorElement* meETLTrackEffPtMtd_[2];
  MonitorElement* meETLTrackPtRes_;

  MonitorElement* meTracktmtd_;
  MonitorElement* meTrackt0Src_;
  MonitorElement* meTrackSigmat0Src_;
  MonitorElement* meTrackt0Pid_;
  MonitorElement* meTrackSigmat0Pid_;
  MonitorElement* meTrackt0SafePid_;
  MonitorElement* meTrackSigmat0SafePid_;
  MonitorElement* meTrackNumHits_;
  MonitorElement* meTrackMVAQual_;
  MonitorElement* meTrackPathLenghtvsEta_;

  MonitorElement* meMVATrackEffPtTot_;
  MonitorElement* meMVATrackMatchedEffPtTot_;
  MonitorElement* meMVATrackMatchedEffPtMtd_;
  MonitorElement* meMVATrackEffEtaTot_;
  MonitorElement* meMVATrackMatchedEffEtaTot_;
  MonitorElement* meMVATrackMatchedEffEtaMtd_;
  MonitorElement* meMVATrackResTot_;
  MonitorElement* meMVATrackPullTot_;
  MonitorElement* meMVATrackZposResTot_;

  MonitorElement* meBarrelPiDBetavsp_;
  MonitorElement* meEndcapPiDBetavsp_;
  MonitorElement* meBarrelKDBetavsp_;
  MonitorElement* meEndcapKDBetavsp_;
  MonitorElement* meBarrelPDBetavsp_;
  MonitorElement* meEndcapPDBetavsp_;

  MonitorElement* meBarrelPiprobPivsp_;
  MonitorElement* meBarrelPiprobKvsp_;
  MonitorElement* meEndcapPiprobPivsp_;
  MonitorElement* meEndcapPiprobKvsp_;

  MonitorElement* meBarrelKprobPivsp_;
  MonitorElement* meBarrelKprobKvsp_;
  MonitorElement* meBarrelKprobPvsp_;
  MonitorElement* meEndcapKprobPivsp_;
  MonitorElement* meEndcapKprobKvsp_;
  MonitorElement* meEndcapKprobPvsp_;

  MonitorElement* meBarrelPprobPvsp_;
  MonitorElement* meBarrelPprobKvsp_;
  MonitorElement* meEndcapPprobPvsp_;
  MonitorElement* meEndcapPprobKvsp_;

  MonitorElement* meBarrelTruePi_;
  MonitorElement* meBarrelTrueK_;
  MonitorElement* meBarrelTrueP_;
  MonitorElement* meEndcapTruePi_;
  MonitorElement* meEndcapTrueK_;
  MonitorElement* meEndcapTrueP_;
};

// ------------ constructor and destructor --------------
MtdTracksValidation::MtdTracksValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      trackMinPt_(iConfig.getParameter<double>("trackMinimumPt")),
      trackMinEta_(iConfig.getParameter<double>("trackMinimumEta")),
      trackMaxEta_(iConfig.getParameter<double>("trackMaximumEta")),
      testPID_(iConfig.getParameter<bool>("testPID")) {
  GenRecTrackToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("inputTagG"));
  RecTrackToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("inputTagT"));
  RecVertexToken_ = consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("inputTagV"));
  HepMCProductToken_ = consumes<edm::HepMCProduct>(iConfig.getParameter<edm::InputTag>("inputTagH"));
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
  trackMVAQualToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("trackMVAQual"));
  tofPiToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("tofPi"));
  tofKToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("tofK"));
  tofPToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("tofP"));
  probPiToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("probPi"));
  probKToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("probK"));
  probPToken_ = consumes<edm::ValueMap<float>>(iConfig.getParameter<edm::InputTag>("probP"));
  mtdtopoToken_ = esConsumes<MTDTopology, MTDTopologyRcd>();
  particleTableToken_ = esConsumes<HepPDT::ParticleDataTable, edm::DefaultRecord>();
}

MtdTracksValidation::~MtdTracksValidation() {}

// ------------ method called for each event  ------------
void MtdTracksValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace geant_units::operators;
  using namespace std;

  auto topologyHandle = iSetup.getTransientHandle(mtdtopoToken_);
  const MTDTopology* topology = topologyHandle.product();

  bool topo1Dis = false;
  bool topo2Dis = false;
  if (topology->getMTDTopologyMode() <= static_cast<int>(MTDTopologyMode::Mode::barphiflat)) {
    topo1Dis = true;
  }
  if (topology->getMTDTopologyMode() > static_cast<int>(MTDTopologyMode::Mode::barphiflat)) {
    topo2Dis = true;
  }

  auto GenRecTrackHandle = makeValid(iEvent.getHandle(GenRecTrackToken_));
  auto RecVertexHandle = makeValid(iEvent.getHandle(RecVertexToken_));

  const auto& tMtd = iEvent.get(tmtdToken_);
  const auto& SigmatMtd = iEvent.get(SigmatmtdToken_);
  const auto& t0Src = iEvent.get(t0SrcToken_);
  const auto& Sigmat0Src = iEvent.get(Sigmat0SrcToken_);
  const auto& t0Pid = iEvent.get(t0PidToken_);
  const auto& Sigmat0Pid = iEvent.get(Sigmat0PidToken_);
  const auto& t0Safe = iEvent.get(t0SafePidToken_);
  const auto& Sigmat0Safe = iEvent.get(Sigmat0SafePidToken_);
  const auto& mtdQualMVA = iEvent.get(trackMVAQualToken_);
  const auto& trackAssoc = iEvent.get(trackAssocToken_);
  const auto& pathLength = iEvent.get(pathLengthToken_);
  const auto& tofPi = iEvent.get(tofPiToken_);
  const auto& tofK = iEvent.get(tofKToken_);
  const auto& tofP = iEvent.get(tofPToken_);
  const auto& probPi = iEvent.get(probPiToken_);
  const auto& probK = iEvent.get(probKToken_);
  const auto& probP = iEvent.get(probPToken_);

  unsigned int index = 0;
  // --- Loop over all RECO tracks ---
  for (const auto& trackGen : *GenRecTrackHandle) {
    const reco::TrackRef trackref(iEvent.getHandle(GenRecTrackToken_), index);
    index++;

    if (trackAssoc[trackref] == -1) {
      LogInfo("mtdTracks") << "Extended track not associated";
      continue;
    }

    const reco::TrackRef mtdTrackref = reco::TrackRef(iEvent.getHandle(RecTrackToken_), trackAssoc[trackref]);
    const reco::Track track = *mtdTrackref;

    if (track.pt() < trackMinPt_ || std::abs(track.eta()) > trackMaxEta_)
      continue;

    meTracktmtd_->Fill(tMtd[trackref]);
    if (std::round(SigmatMtd[trackref] - Sigmat0Pid[trackref]) != 0) {
      LogWarning("mtdTracks") << "TimeError associated to refitted track is different from TimeError stored in tofPID "
                                 "sigmat0 ValueMap: this should not happen";
    }

    meTrackt0Src_->Fill(t0Src[trackref]);
    meTrackSigmat0Src_->Fill(Sigmat0Src[trackref]);

    meTrackt0Pid_->Fill(t0Pid[trackref]);
    meTrackSigmat0Pid_->Fill(Sigmat0Pid[trackref]);
    meTrackt0SafePid_->Fill(t0Safe[trackref]);
    meTrackSigmat0SafePid_->Fill(Sigmat0Safe[trackref]);
    meTrackMVAQual_->Fill(mtdQualMVA[trackref]);

    meTrackPathLenghtvsEta_->Fill(std::abs(track.eta()), pathLength[trackref]);

    if (std::abs(track.eta()) < trackMinEta_) {
      // --- all BTL tracks (with and without hit in MTD) ---
      meBTLTrackEffEtaTot_->Fill(track.eta());
      meBTLTrackEffPhiTot_->Fill(track.phi());
      meBTLTrackEffPtTot_->Fill(track.pt());

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
        meBTLTrackEffEtaMtd_->Fill(track.eta());
        meBTLTrackEffPhiMtd_->Fill(track.phi());
        meBTLTrackEffPtMtd_->Fill(track.pt());
        meBTLTrackRPTime_->Fill(track.t0());
        meBTLTrackPtRes_->Fill((trackGen.pt() - track.pt()) / trackGen.pt());
      }
    }  //loop over (geometrical) BTL tracks

    else {
      // --- all ETL tracks (with and without hit in MTD) ---
      if ((track.eta() < -trackMinEta_) && (track.eta() > -trackMaxEta_)) {
        meETLTrackEffEtaTot_[0]->Fill(track.eta());
        meETLTrackEffPhiTot_[0]->Fill(track.phi());
        meETLTrackEffPtTot_[0]->Fill(track.pt());
      }

      if ((track.eta() > trackMinEta_) && (track.eta() < trackMaxEta_)) {
        meETLTrackEffEtaTot_[1]->Fill(track.eta());
        meETLTrackEffPhiTot_[1]->Fill(track.phi());
        meETLTrackEffPtTot_[1]->Fill(track.pt());
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
          ETLDetId ETLHit = hit->geographicalId();

          if (topo2Dis) {
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

          if (topo1Dis) {
            if (ETLHit.zside() == -1) {
              MTDEtlZnegD1 = true;
              meETLTrackRPTime_->Fill(track.t0());
              numMTDEtlvalidhits++;
            }
            if (ETLHit.zside() == 1) {
              MTDEtlZposD1 = true;
              meETLTrackRPTime_->Fill(track.t0());
              numMTDEtlvalidhits++;
            }
          }
        }
      }
      meTrackNumHits_->Fill(-numMTDEtlvalidhits);

      // --- keeping only tracks with last hit in MTD ---
      if ((track.eta() < -trackMinEta_) && (track.eta() > -trackMaxEta_)) {
        if ((MTDEtlZnegD1 == true) || (MTDEtlZnegD2 == true)) {
          meETLTrackEffEtaMtd_[0]->Fill(track.eta());
          meETLTrackEffPhiMtd_[0]->Fill(track.phi());
          meETLTrackEffPtMtd_[0]->Fill(track.pt());
        }
      }
      if ((track.eta() > trackMinEta_) && (track.eta() < trackMaxEta_)) {
        if ((MTDEtlZposD1 == true) || (MTDEtlZposD2 == true)) {
          meETLTrackEffEtaMtd_[1]->Fill(track.eta());
          meETLTrackEffPhiMtd_[1]->Fill(track.phi());
          meETLTrackEffPtMtd_[1]->Fill(track.pt());
        }
      }
    }
  }  //RECO tracks loop

  // reco-gen matching used for MVA quality flag
  const auto& primRecoVtx = *(RecVertexHandle.product()->begin());
  double treco = primRecoVtx.t();

  auto GenEventHandle = makeValid(iEvent.getHandle(HepMCProductToken_));
  const HepMC::GenEvent* mc = GenEventHandle->GetEvent();
  double zsim = convertMmToCm((*(mc->vertices_begin()))->position().z());
  double tsim = (*(mc->vertices_begin()))->position().t() * CLHEP::mm / CLHEP::c_light;

  auto pdt = iSetup.getHandle(particleTableToken_);
  const HepPDT::ParticleDataTable* pdTable = pdt.product();

  // select events with reco vertex close to true simulated primary vertex
  if (std::abs(primRecoVtx.z() - zsim) < deltaZcut_) {
    index = 0;
    for (const auto& trackGen : *GenRecTrackHandle) {
      const reco::TrackRef trackref(iEvent.getHandle(GenRecTrackToken_), index);
      index++;

      // select the reconstructed track

      if (trackAssoc[trackref] == -1) {
        continue;
      }

      if (mvaRecSel(trackGen, primRecoVtx, t0Safe[trackref], Sigmat0Safe[trackref])) {
        meMVATrackEffPtTot_->Fill(trackGen.pt());
        meMVATrackEffEtaTot_->Fill(std::abs(trackGen.eta()));

        double dZ = trackGen.vz() - zsim;
        double dT(-9999.);
        double pullT(-9999.);
        if (Sigmat0Safe[trackref] != -1.) {
          dT = t0Safe[trackref] - tsim;
          pullT = dT / Sigmat0Safe[trackref];
        }
        for (const auto& genP : mc->particle_range()) {
          // select status 1 genParticles and match them to the reconstructed track

          float charge = pdTable->particle(HepPDT::ParticleID(genP->pdg_id())) != nullptr
                             ? pdTable->particle(HepPDT::ParticleID(genP->pdg_id()))->charge()
                             : 0.f;
          if (mvaGenSel(*genP, charge)) {
            if (mvaGenRecMatch(*genP, zsim, trackGen)) {
              meMVATrackZposResTot_->Fill(dZ);
              meMVATrackMatchedEffPtTot_->Fill(trackGen.pt());
              meMVATrackMatchedEffEtaTot_->Fill(std::abs(trackGen.eta()));
              if (testPID_) {
                if (std::abs(trackGen.eta()) < trackMinEta_) {
                  if (std::abs(genP->pdg_id()) == 211) {
                    meBarrelTruePi_->Fill(trackGen.p());
                  } else if (std::abs(genP->pdg_id()) == 321) {
                    meBarrelTrueK_->Fill(trackGen.p());
                  } else if (std::abs(genP->pdg_id()) == 2212) {
                    meBarrelTrueP_->Fill(trackGen.p());
                  }
                } else if (std::abs(trackGen.eta()) > trackMinEta_) {
                  if (std::abs(genP->pdg_id()) == 211) {
                    meEndcapTruePi_->Fill(trackGen.p());
                  } else if (std::abs(genP->pdg_id()) == 321) {
                    meEndcapTrueK_->Fill(trackGen.p());
                  } else if (std::abs(genP->pdg_id()) == 2212) {
                    meEndcapTrueP_->Fill(trackGen.p());
                  }
                }
              }
              if (pullT > -9999.) {
                meMVATrackResTot_->Fill(dT);
                meMVATrackPullTot_->Fill(pullT);
                meMVATrackMatchedEffPtMtd_->Fill(trackGen.pt());
                meMVATrackMatchedEffEtaMtd_->Fill(std::abs(trackGen.eta()));

                if (testPID_) {
                  double dbetaPi = c_cm_ns * (tMtd[trackref] - treco - tofPi[trackref]) / pathLength[trackref];
                  double dbetaK = c_cm_ns * (tMtd[trackref] - treco - tofK[trackref]) / pathLength[trackref];
                  double dbetaP = c_cm_ns * (tMtd[trackref] - treco - tofP[trackref]) / pathLength[trackref];

                  if (std::abs(trackGen.eta()) < trackMinEta_) {
                    if (std::abs(genP->pdg_id()) == 211) {
                      meBarrelPiDBetavsp_->Fill(trackGen.p(), dbetaPi);
                      meBarrelPiprobPivsp_->Fill(trackGen.p(), probPi[trackref]);
                      meBarrelPiprobKvsp_->Fill(trackGen.p(), probK[trackref]);
                    } else if (std::abs(genP->pdg_id()) == 321) {
                      meBarrelKDBetavsp_->Fill(trackGen.p(), dbetaK);
                      meBarrelKprobPivsp_->Fill(trackGen.p(), probPi[trackref]);
                      meBarrelKprobKvsp_->Fill(trackGen.p(), probK[trackref]);
                      meBarrelKprobPvsp_->Fill(trackGen.p(), probP[trackref]);
                    } else if (std::abs(genP->pdg_id()) == 2212) {
                      meBarrelPDBetavsp_->Fill(trackGen.p(), dbetaP);
                      meBarrelPprobPvsp_->Fill(trackGen.p(), probP[trackref]);
                      meBarrelPprobKvsp_->Fill(trackGen.p(), probK[trackref]);
                    }
                  } else if (std::abs(trackGen.eta()) > trackMinEta_) {
                    if (std::abs(genP->pdg_id()) == 211) {
                      meEndcapPiDBetavsp_->Fill(trackGen.p(), dbetaPi);
                      meEndcapPiprobPivsp_->Fill(trackGen.p(), probPi[trackref]);
                      meEndcapPiprobKvsp_->Fill(trackGen.p(), probK[trackref]);
                    } else if (std::abs(genP->pdg_id()) == 321) {
                      meEndcapKDBetavsp_->Fill(trackGen.p(), dbetaK);
                      meEndcapKprobPivsp_->Fill(trackGen.p(), probPi[trackref]);
                      meEndcapKprobKvsp_->Fill(trackGen.p(), probK[trackref]);
                      meEndcapKprobPvsp_->Fill(trackGen.p(), probP[trackref]);
                    } else if (std::abs(genP->pdg_id()) == 2212) {
                      meEndcapPDBetavsp_->Fill(trackGen.p(), dbetaP);
                      meEndcapPprobPvsp_->Fill(trackGen.p(), probP[trackref]);
                      meEndcapPprobKvsp_->Fill(trackGen.p(), probK[trackref]);
                    }
                  }
                }
              }
              break;
            }
          }
        }
      }
    }
  }
}

// ------------ method for histogram booking ------------
void MtdTracksValidation::bookHistograms(DQMStore::IBooker& ibook, edm::Run const& run, edm::EventSetup const& iSetup) {
  ibook.setCurrentFolder(folder_);

  // histogram booking
  meBTLTrackRPTime_ = ibook.book1D("TrackBTLRPTime", "Track t0 with respect to R.P.;t0 [ns]", 100, -1, 3);
  meBTLTrackEffEtaTot_ = ibook.book1D("TrackBTLEffEtaTot", "Track efficiency vs eta (Tot);#eta_{RECO}", 100, -1.6, 1.6);
  meBTLTrackEffPhiTot_ =
      ibook.book1D("TrackBTLEffPhiTot", "Track efficiency vs phi (Tot);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meBTLTrackEffPtTot_ = ibook.book1D("TrackBTLEffPtTot", "Track efficiency vs pt (Tot);pt_{RECO} [GeV]", 50, 0, 10);
  meBTLTrackEffEtaMtd_ = ibook.book1D("TrackBTLEffEtaMtd", "Track efficiency vs eta (Mtd);#eta_{RECO}", 100, -1.6, 1.6);
  meBTLTrackEffPhiMtd_ =
      ibook.book1D("TrackBTLEffPhiMtd", "Track efficiency vs phi (Mtd);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meBTLTrackEffPtMtd_ = ibook.book1D("TrackBTLEffPtMtd", "Track efficiency vs pt (Mtd);pt_{RECO} [GeV]", 50, 0, 10);
  meBTLTrackPtRes_ =
      ibook.book1D("TrackBTLPtRes", "Track pT resolution  ;pT_{Gentrack}-pT_{MTDtrack}/pT_{Gentrack} ", 100, -0.1, 0.1);
  meETLTrackRPTime_ = ibook.book1D("TrackETLRPTime", "Track t0 with respect to R.P.;t0 [ns]", 100, -1, 3);
  meETLTrackEffEtaTot_[0] =
      ibook.book1D("TrackETLEffEtaTotZneg", "Track efficiency vs eta (Tot) (-Z);#eta_{RECO}", 100, -3.2, -1.4);
  meETLTrackEffEtaTot_[1] =
      ibook.book1D("TrackETLEffEtaTotZpos", "Track efficiency vs eta (Tot) (+Z);#eta_{RECO}", 100, 1.4, 3.2);
  meETLTrackEffPhiTot_[0] =
      ibook.book1D("TrackETLEffPhiTotZneg", "Track efficiency vs phi (Tot) (-Z);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meETLTrackEffPhiTot_[1] =
      ibook.book1D("TrackETLEffPhiTotZpos", "Track efficiency vs phi (Tot) (+Z);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meETLTrackEffPtTot_[0] =
      ibook.book1D("TrackETLEffPtTotZneg", "Track efficiency vs pt (Tot) (-Z);pt_{RECO} [GeV]", 50, 0, 10);
  meETLTrackEffPtTot_[1] =
      ibook.book1D("TrackETLEffPtTotZpos", "Track efficiency vs pt (Tot) (+Z);pt_{RECO} [GeV]", 50, 0, 10);
  meETLTrackEffEtaMtd_[0] =
      ibook.book1D("TrackETLEffEtaMtdZneg", "Track efficiency vs eta (Mtd) (-Z);#eta_{RECO}", 100, -3.2, -1.4);
  meETLTrackEffEtaMtd_[1] =
      ibook.book1D("TrackETLEffEtaMtdZpos", "Track efficiency vs eta (Mtd) (+Z);#eta_{RECO}", 100, 1.4, 3.2);
  meETLTrackEffPhiMtd_[0] =
      ibook.book1D("TrackETLEffPhiMtdZneg", "Track efficiency vs phi (Mtd) (-Z);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meETLTrackEffPhiMtd_[1] =
      ibook.book1D("TrackETLEffPhiMtdZpos", "Track efficiency vs phi (Mtd) (+Z);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meETLTrackEffPtMtd_[0] =
      ibook.book1D("TrackETLEffPtMtdZneg", "Track efficiency vs pt (Mtd) (-Z);pt_{RECO} [GeV]", 50, 0, 10);
  meETLTrackEffPtMtd_[1] =
      ibook.book1D("TrackETLEffPtMtdZpos", "Track efficiency vs pt (Mtd) (+Z);pt_{RECO} [GeV]", 50, 0, 10);
  meETLTrackPtRes_ =
      ibook.book1D("TrackETLPtRes", "Track pT resolution;pT_{Gentrack}-pT_{MTDtrack}/pT_{Gentrack} ", 100, -0.1, 0.1);

  meTracktmtd_ = ibook.book1D("Tracktmtd", "Track time from TrackExtenderWithMTD;tmtd [ns]", 150, 1, 16);
  meTrackt0Src_ = ibook.book1D("Trackt0Src", "Track time from TrackExtenderWithMTD;t0Src [ns]", 100, -1.5, 1.5);
  meTrackSigmat0Src_ =
      ibook.book1D("TrackSigmat0Src", "Time Error from TrackExtenderWithMTD; #sigma_{t0Src} [ns]", 100, 0, 0.1);

  meTrackt0Pid_ = ibook.book1D("Trackt0Pid", "Track t0 as stored in TofPid;t0 [ns]", 100, -1, 1);
  meTrackSigmat0Pid_ = ibook.book1D("TrackSigmat0Pid", "Sigmat0 as stored in TofPid; #sigma_{t0} [ns]", 100, 0, 0.1);
  meTrackt0SafePid_ = ibook.book1D("Trackt0SafePID", "Track t0 Safe as stored in TofPid;t0 [ns]", 100, -1, 1);
  meTrackSigmat0SafePid_ =
      ibook.book1D("TrackSigmat0SafePID", "Sigmat0 Safe as stored in TofPid; #sigma_{t0} [ns]", 100, 0, 0.1);
  meTrackNumHits_ = ibook.book1D("TrackNumHits", "Number of valid MTD hits per track ; Number of hits", 10, -5, 5);
  meTrackMVAQual_ = ibook.book1D("TrackMVAQual", "Track MVA Quality as stored in Value Map ; MVAQual", 100, 0, 1);
  meTrackPathLenghtvsEta_ = ibook.bookProfile(
      "TrackPathLenghtvsEta", "MTD Track pathlength vs MTD track Eta;|#eta|;Pathlength", 100, 0, 3.2, 100.0, 400.0, "S");
  meMVATrackEffPtTot_ = ibook.book1D("MVAEffPtTot", "Pt of tracks associated to LV; track pt [GeV] ", 110, 0., 11.);
  meMVATrackMatchedEffPtTot_ =
      ibook.book1D("MVAMatchedEffPtTot", "Pt of tracks associated to LV matched to GEN; track pt [GeV] ", 110, 0., 11.);
  meMVATrackMatchedEffPtMtd_ = ibook.book1D(
      "MVAMatchedEffPtMtd", "Pt of tracks associated to LV matched to GEN with time; track pt [GeV] ", 110, 0., 11.);
  meMVATrackEffEtaTot_ = ibook.book1D("MVAEffEtaTot", "Pt of tracks associated to LV; track eta ", 66, 0., 3.3);
  meMVATrackMatchedEffEtaTot_ =
      ibook.book1D("MVAMatchedEffEtaTot", "Pt of tracks associated to LV matched to GEN; track eta ", 66, 0., 3.3);
  meMVATrackMatchedEffEtaMtd_ = ibook.book1D(
      "MVAMatchedEffEtaMtd", "Pt of tracks associated to LV matched to GEN with time; track eta ", 66, 0., 3.3);
  meMVATrackResTot_ = ibook.book1D(
      "MVATrackRes", "t_{rec} - t_{sim} for LV associated tracks; t_{rec} - t_{sim} [ns] ", 120, -0.15, 0.15);
  meMVATrackPullTot_ =
      ibook.book1D("MVATrackPull", "Pull for associated tracks; (t_{rec}-t_{sim})/#sigma_{t}", 50, -5., 5.);
  meMVATrackZposResTot_ = ibook.book1D(
      "MVATrackZposResTot", "Z_{PCA} - Z_{sim} for associated tracks;Z_{PCA} - Z_{sim} [cm] ", 100, -0.1, 0.1);

  if (testPID_) {
    meBarrelPiDBetavsp_ = ibook.bookProfile(
        "BarrelPiDBetavsp", "DeltaBeta true pi as pi vs p, |eta| < 1.5;p [GeV]; dBeta", 25, 0., 10., -0.1, 0.1, "S");
    meEndcapPiDBetavsp_ = ibook.bookProfile(
        "EndcapPiDBetavsp", "DeltaBeta true pi as pi vs p, |eta| > 1.5;p [GeV]; dBeta", 25, 0., 10., -0.1, 0.1, "S");
    meBarrelKDBetavsp_ = ibook.bookProfile(
        "BarrelKDBetavsp", "DeltaBeta true K as K vs p, |eta| < 1.5;p [GeV]; dBeta", 25, 0., 10., -0.1, 0.1, "S");
    meEndcapKDBetavsp_ = ibook.bookProfile(
        "EndcapKDBetavsp", "DeltaBeta true K as K vs p, |eta| > 1.5;p [GeV]; dBeta", 25, 0., 10., -0.1, 0.1, "S");
    meBarrelPDBetavsp_ = ibook.bookProfile(
        "BarrelPDBetavsp", "DeltaBeta true p as p vs p, |eta| < 1.5;p [GeV]; dBeta", 25, 0., 10., -0.1, 0.1, "S");
    meEndcapPDBetavsp_ = ibook.bookProfile(
        "EndcapPDBetavsp", "DeltaBeta true p as p vs p, |eta| > 1.5;p [GeV]; dBeta", 25, 0., 10., -0.1, 0.1, "S");

    meBarrelPiprobPivsp_ = ibook.book2D(
        "BarrelPiprobPivsp", "Probability true pi as pi vs p, |eta| < 1.5;p [GeV]; prob", 25, 0., 10., 51, 0., 1.02);
    meBarrelPiprobKvsp_ = ibook.book2D(
        "BarrelPiprobKvsp", "Probability true pi as K vs p, |eta| < 1.5;p [GeV]; prob", 25, 0., 10., 51, 0., 1.02);
    meEndcapPiprobPivsp_ = ibook.book2D(
        "EndcapPiprobPivsp", "Probability true pi as pi vs p, |eta| > 1.5;p [GeV]; prob", 25, 0., 10., 51, 0., 1.02);
    meEndcapPiprobKvsp_ = ibook.book2D(
        "EndcapPiprobKvsp", "Probability true pi as K vs p, |eta| > 1.5;p [GeV]; prob", 25, 0., 10., 51, 0., 1.02);

    meBarrelKprobPivsp_ = ibook.book2D(
        "BarrelKprobPivsp", "Probability true K as pi vs p, |eta| < 1.5;p [GeV]; prob", 25, 0., 10., 51, 0., 1.02);
    meBarrelKprobKvsp_ = ibook.book2D(
        "BarrelKprobKvsp", "Probability true K as K vs p, |eta| < 1.5;p [GeV]; prob", 25, 0., 10., 51, 0., 1.02);
    meBarrelKprobPvsp_ = ibook.book2D(
        "BarrelKprobPvsp", "Probability true K as p vs p, |eta| < 1.5;p [GeV]; prob", 25, 0., 10., 51, 0., 1.02);
    meEndcapKprobPivsp_ = ibook.book2D(
        "EndcapKprobPivsp", "Probability true K as pi vs p, |eta| > 1.5;p [GeV]; prob", 25, 0., 10., 51, 0., 1.02);
    meEndcapKprobKvsp_ = ibook.book2D(
        "EndcapKprobKvsp", "Probability true K as K vs p, |eta| > 1.5;p [GeV]; prob", 25, 0., 10., 51, 0., 1.02);
    meEndcapKprobPvsp_ = ibook.book2D(
        "EndcapKprobPvsp", "Probability true K as p vs p, |eta| > 1.5;p [GeV]; prob", 25, 0., 10., 51, 0., 1.02);

    meBarrelPprobPvsp_ = ibook.book2D(
        "BarrelPprobPvsp", "Probability true p as p vs p, |eta| < 1.5;p [GeV]; prob", 25, 0., 10., 51, 0., 1.02);
    meBarrelPprobKvsp_ = ibook.book2D(
        "BarrelPprobKvsp", "Probability true p as K vs p, |eta| < 1.5;p [GeV]; prob", 25, 0., 10., 51, 0., 1.02);
    meEndcapPprobPvsp_ = ibook.book2D(
        "EndcapPprobPvsp", "Probability true p as p vs p, |eta| > 1.5;p [GeV]; prob", 25, 0., 10., 51, 0., 1.02);
    meEndcapPprobKvsp_ = ibook.book2D(
        "EndcapPprobKvsp", "Probability true p as K vs p, |eta| > 1.5;p [GeV]; prob", 25, 0., 10., 51, 0., 1.02);

    meBarrelTruePi_ = ibook.book1D("BarrelTruePi", "True pi momentum spectrum, |eta| < 1.5;p [GeV]", 25, 0., 10.);
    meBarrelTrueK_ = ibook.book1D("BarrelTrueK", "True k momentum spectrum, |eta| < 1.5;p [GeV]", 25, 0., 10.);
    meBarrelTrueP_ = ibook.book1D("BarrelTrueP", "True p momentum spectrum, |eta| < 1.5;p [GeV]", 25, 0., 10.);
    meEndcapTruePi_ = ibook.book1D("EndcapTruePi", "True pi momentum spectrum, |eta| > 1.5;p [GeV]", 25, 0., 10.);
    meEndcapTrueK_ = ibook.book1D("EndcapTrueK", "True k momentum spectrum, |eta| > 1.5;p [GeV]", 25, 0., 10.);
    meEndcapTrueP_ = ibook.book1D("EndcapTrueP", "True p momentum spectrum, |eta| > 1.5;p [GeV]", 25, 0., 10.);
  }
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------

void MtdTracksValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("folder", "MTD/Tracks");
  desc.add<edm::InputTag>("inputTagG", edm::InputTag("generalTracks"));
  desc.add<edm::InputTag>("inputTagT", edm::InputTag("trackExtenderWithMTD"));
  desc.add<edm::InputTag>("inputTagV", edm::InputTag("offlinePrimaryVertices4D"));
  desc.add<edm::InputTag>("inputTagH", edm::InputTag("generatorSmeared"));
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
  desc.add<edm::InputTag>("trackMVAQual", edm::InputTag("mtdTrackQualityMVA:mtdQualMVA"));
  desc.add<edm::InputTag>("tofPi", edm::InputTag("trackExtenderWithMTD:generalTrackTofPi"));
  desc.add<edm::InputTag>("tofK", edm::InputTag("trackExtenderWithMTD:generalTrackTofK"));
  desc.add<edm::InputTag>("tofP", edm::InputTag("trackExtenderWithMTD:generalTrackTofP"));
  desc.add<edm::InputTag>("probPi", edm::InputTag("tofPID:probPi"));
  desc.add<edm::InputTag>("probK", edm::InputTag("tofPID:probK"));
  desc.add<edm::InputTag>("probP", edm::InputTag("tofPID:probP"));
  desc.add<double>("trackMinimumPt", 0.7);  // [GeV]
  desc.add<double>("trackMinimumEta", 1.5);
  desc.add<double>("trackMaximumEta", 3.);
  desc.add<bool>("testPID", false);

  descriptions.add("mtdTracksValid", desc);
}

const bool MtdTracksValidation::mvaGenSel(const HepMC::GenParticle& gp, const float& charge) {
  bool match = false;
  if (gp.status() != 1) {
    return match;
  }
  match = charge != 0.f && gp.momentum().perp() > pTcut_ && std::abs(gp.momentum().eta()) < etacutGEN_;
  return match;
}

const bool MtdTracksValidation::mvaRecSel(const reco::TrackBase& trk,
                                          const reco::Vertex& vtx,
                                          const double& t0,
                                          const double& st0) {
  bool match = false;
  match = trk.pt() > pTcut_ && std::abs(trk.eta()) < etacutREC_ && std::abs(trk.vz() - vtx.z()) <= deltaZcut_;
  if (st0 > 0.) {
    match = match && std::abs(t0 - vtx.t()) < 3. * st0;
  }
  return match;
}

const bool MtdTracksValidation::mvaGenRecMatch(const HepMC::GenParticle& genP,
                                               const double& zsim,
                                               const reco::TrackBase& trk) {
  bool match = false;
  double dR = reco::deltaR(genP.momentum(), trk.momentum());
  double genPT = genP.momentum().perp();
  match =
      std::abs(genPT - trk.pt()) < trk.pt() * deltaPTcut_ && dR < deltaDRcut_ && std::abs(trk.vz() - zsim) < deltaZcut_;
  return match;
}

DEFINE_FWK_MODULE(MtdTracksValidation);
