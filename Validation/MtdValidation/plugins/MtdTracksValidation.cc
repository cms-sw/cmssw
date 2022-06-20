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

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "HepMC/GenRanges.h"
#include "CLHEP/Units/PhysicalConstants.h"

namespace {

  struct MTDHit {
    float energy;
    float time;
    float x;
    float y;
    float z;
  };

}  // namespace

class MtdTracksValidation : public DQMEDAnalyzer {
public:
  explicit MtdTracksValidation(const edm::ParameterSet&);
  ~MtdTracksValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  const bool mvaGenSel(const HepMC::GenParticle&, const float&);
  const bool mvaTPSel(const TrackingParticle&);
  const bool mvaRecSel(const reco::TrackBase&, const reco::Vertex&, const double&, const double&);
  const bool mvaGenRecMatch(const HepMC::GenParticle&, const double&, const reco::TrackBase&, const bool&);
  const edm::Ref<std::vector<TrackingParticle>>* getMatchedTP(const reco::TrackBaseRef&, const double&);
  const bool tpWithMTD(const TrackingParticle&, const std::unordered_set<unsigned long int>&);

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
  const float trackMinPt_;
  const float trackMaxBtlEta_;
  const float trackMinEtlEta_;
  const float trackMaxEtlEta_;

  static constexpr double etacutGEN_ = 4.;               // |eta| < 4;
  static constexpr double etacutREC_ = 3.;               // |eta| < 3;
  static constexpr double pTcut_ = 0.7;                  // PT > 0.7 GeV
  static constexpr double deltaZcut_ = 0.1;              // dz separation 1 mm
  static constexpr double deltaPTcut_ = 0.05;            // dPT < 5%
  static constexpr double deltaDRcut_ = 0.03;            // DeltaR separation
  static constexpr double depositBTLthreshold_ = 1;      // threshold for energy deposit in BTL cell [MeV]
  static constexpr double depositETLthreshold_ = 0.001;  // threshold for energy deposit in ETL cell [MeV]
  static constexpr double rBTL_ = 110.0;
  static constexpr double zETL_ = 290.0;
  static constexpr double etaMatchCut_ = 0.05;

  bool optionalPlots_;

  const reco::RecoToSimCollection* r2s_;
  const reco::SimToRecoCollection* s2r_;

  edm::EDGetTokenT<reco::TrackCollection> GenRecTrackToken_;
  edm::EDGetTokenT<reco::TrackCollection> RecTrackToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> RecVertexToken_;

  edm::EDGetTokenT<edm::HepMCProduct> HepMCProductToken_;
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticleCollectionToken_;
  edm::EDGetTokenT<reco::SimToRecoCollection> simToRecoAssociationToken_;
  edm::EDGetTokenT<reco::RecoToSimCollection> recoToSimAssociationToken_;
  edm::EDGetTokenT<CrossingFrame<PSimHit>> btlSimHitsToken_;
  edm::EDGetTokenT<CrossingFrame<PSimHit>> etlSimHitsToken_;

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

  edm::ESGetToken<MTDGeometry, MTDDigiGeometryRecord> mtdgeoToken_;
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
  MonitorElement* meTrackMVAQual_;
  MonitorElement* meTrackPathLenghtvsEta_;

  MonitorElement* meMVATrackEffPtTot_;
  MonitorElement* meMVATrackMatchedEffPtTot_;
  MonitorElement* meMVATrackMatchedEffPtMtd_;
  MonitorElement* meTrackMatchedTPEffPtTot_;
  MonitorElement* meTrackMatchedTPEffPtMtd_;
  MonitorElement* meTrackMatchedTPEffPtEtl2Mtd_;
  MonitorElement* meTrackMatchedTPmtdEffPtTot_;
  MonitorElement* meTrackMatchedTPmtdEffPtMtd_;
  MonitorElement* meMVATrackEffEtaTot_;
  MonitorElement* meMVATrackMatchedEffEtaTot_;
  MonitorElement* meMVATrackMatchedEffEtaMtd_;
  MonitorElement* meTrackMatchedTPEffEtaTot_;
  MonitorElement* meTrackMatchedTPEffEtaMtd_;
  MonitorElement* meTrackMatchedTPEffEtaEtl2Mtd_;
  MonitorElement* meTrackMatchedTPmtdEffEtaTot_;
  MonitorElement* meTrackMatchedTPmtdEffEtaMtd_;
  MonitorElement* meMVATrackResTot_;
  MonitorElement* meMVATrackPullTot_;
  MonitorElement* meMVATrackZposResTot_;

  MonitorElement* meUnassociatedDetId_;
  MonitorElement* meUnassCrysEnergy_;
  MonitorElement* meUnassLgadsEnergy_;
  MonitorElement* meUnassDeposit_;
  MonitorElement* meNTrackingParticles_;
};

// ------------ constructor and destructor --------------
MtdTracksValidation::MtdTracksValidation(const edm::ParameterSet& iConfig)
    : folder_(iConfig.getParameter<std::string>("folder")),
      trackMinPt_(iConfig.getParameter<double>("trackMinimumPt")),
      trackMaxBtlEta_(iConfig.getParameter<double>("trackMaximumBtlEta")),
      trackMinEtlEta_(iConfig.getParameter<double>("trackMinimumEtlEta")),
      trackMaxEtlEta_(iConfig.getParameter<double>("trackMaximumEtlEta")),
      optionalPlots_(iConfig.getUntrackedParameter<bool>("optionalPlots")) {
  GenRecTrackToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("inputTagG"));
  RecTrackToken_ = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("inputTagT"));
  RecVertexToken_ = consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("inputTagV"));
  HepMCProductToken_ = consumes<edm::HepMCProduct>(iConfig.getParameter<edm::InputTag>("inputTagH"));
  trackingParticleCollectionToken_ =
      consumes<TrackingParticleCollection>(iConfig.getParameter<edm::InputTag>("SimTag"));
  simToRecoAssociationToken_ =
      consumes<reco::SimToRecoCollection>(iConfig.getParameter<edm::InputTag>("TPtoRecoTrackAssoc"));
  recoToSimAssociationToken_ =
      consumes<reco::RecoToSimCollection>(iConfig.getParameter<edm::InputTag>("TPtoRecoTrackAssoc"));
  btlSimHitsToken_ = consumes<CrossingFrame<PSimHit>>(iConfig.getParameter<edm::InputTag>("btlSimHits"));
  etlSimHitsToken_ = consumes<CrossingFrame<PSimHit>>(iConfig.getParameter<edm::InputTag>("etlSimHits"));
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
  mtdgeoToken_ = esConsumes<MTDGeometry, MTDDigiGeometryRecord>();
  mtdtopoToken_ = esConsumes<MTDTopology, MTDTopologyRcd>();
  particleTableToken_ = esConsumes<HepPDT::ParticleDataTable, edm::DefaultRecord>();
}

MtdTracksValidation::~MtdTracksValidation() {}

// ------------ method called for each event  ------------
void MtdTracksValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace geant_units::operators;
  using namespace std;

  auto geometryHandle = iSetup.getTransientHandle(mtdgeoToken_);
  const MTDGeometry* geom = geometryHandle.product();
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

  std::unordered_map<uint32_t, MTDHit> m_btlHits;
  std::unordered_map<uint32_t, MTDHit> m_etlHits;
  std::unordered_map<uint32_t, std::set<unsigned long int>> m_btlTrkPerCell;
  std::unordered_map<uint32_t, std::set<unsigned long int>> m_etlTrkPerCell;

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

  const auto& primRecoVtx = *(RecVertexHandle.product()->begin());

  // generator level information (HepMC format)
  auto GenEventHandle = makeValid(iEvent.getHandle(HepMCProductToken_));
  const HepMC::GenEvent* mc = GenEventHandle->GetEvent();
  double zsim = convertMmToCm((*(mc->vertices_begin()))->position().z());
  double tsim = (*(mc->vertices_begin()))->position().t() * CLHEP::mm / CLHEP::c_light;

  auto pdt = iSetup.getHandle(particleTableToken_);
  const HepPDT::ParticleDataTable* pdTable = pdt.product();

  auto simToRecoH = makeValid(iEvent.getHandle(simToRecoAssociationToken_));
  s2r_ = simToRecoH.product();

  auto recoToSimH = makeValid(iEvent.getHandle(recoToSimAssociationToken_));
  r2s_ = recoToSimH.product();

  // find all signal event trackId corresponding to an MTD simHit
  std::unordered_set<unsigned long int> mtdTrackId;

  std::unordered_set<unsigned long int> tpTrackId;
  auto tpHandle = makeValid(iEvent.getHandle(trackingParticleCollectionToken_));
  TrackingParticleCollection tpColl = *(tpHandle.product());
  for (const auto& tp : tpColl) {
    if (tp.eventId().bunchCrossing() == 0 && tp.eventId().event() == 0) {
      if (!mvaTPSel(tp))
        continue;
      if (optionalPlots_) {
        if (std::abs(tp.eta()) < trackMaxBtlEta_) {
          meNTrackingParticles_->Fill(0.5);
        } else if ((std::abs(tp.eta()) < trackMaxEtlEta_) && (std::abs(tp.eta()) > trackMinEtlEta_)) {
          meNTrackingParticles_->Fill(1.5);
        }
      }
      for (const auto& simTrk : tp.g4Tracks()) {
        auto const thisTId = uniqueId(simTrk.trackId(), simTrk.eventId());
        tpTrackId.insert(thisTId);
        LogDebug("MtdTracksValidation") << "TP simTrack id : " << thisTId;
      }
    }
  }

  //Fill maps with simhits accumulated per DetId

  auto btlSimHitsHandle = makeValid(iEvent.getHandle(btlSimHitsToken_));
  MixCollection<PSimHit> btlSimHits(btlSimHitsHandle.product());
  for (auto const& simHit : btlSimHits) {
    if (simHit.tof() < 0 || simHit.tof() > 25.)
      continue;
    DetId id = simHit.detUnitId();
    auto const thisHId = uniqueId(simHit.trackId(), simHit.eventId());
    m_btlTrkPerCell[id.rawId()].insert(thisHId);
    auto simHitIt = m_btlHits.emplace(id.rawId(), MTDHit()).first;
    // --- Accumulate the energy (in MeV) of SIM hits in the same detector cell
    (simHitIt->second).energy += convertUnitsTo(0.001_MeV, simHit.energyLoss());
  }

  uint32_t hcount(0);
  for (auto const& cell : m_btlTrkPerCell) {
    bool foundAssocTP = false;
    auto detId_key = cell.first;
    for (auto const& simtrack : cell.second) {
      if (tpTrackId.find(simtrack) != tpTrackId.end()) {
        foundAssocTP = true;
        mtdTrackId.insert(simtrack);
      }
    }
    if (foundAssocTP == false) {
      meUnassCrysEnergy_->Fill(log10(m_btlHits[detId_key].energy));
      if (m_btlHits[detId_key].energy > depositBTLthreshold_) {
        hcount++;
      }
    }
  }
  meUnassociatedDetId_->Fill(0.5, hcount);

  auto etlSimHitsHandle = makeValid(iEvent.getHandle(etlSimHitsToken_));
  MixCollection<PSimHit> etlSimHits(etlSimHitsHandle.product());
  for (auto const& simHit : etlSimHits) {
    if (simHit.tof() < 0 || simHit.tof() > 25.) {
      continue;
    }
    DetId id = simHit.detUnitId();
    auto const thisHId = uniqueId(simHit.trackId(), simHit.eventId());
    m_etlTrkPerCell[id.rawId()].insert(thisHId);
    auto simHitIt = m_etlHits.emplace(id.rawId(), MTDHit()).first;
    // --- Accumulate the energy (in MeV) of SIM hits in the same detector cell
    (simHitIt->second).energy += convertUnitsTo(0.001_MeV, simHit.energyLoss());
  }

  hcount = 0;
  for (auto const& cell : m_etlTrkPerCell) {
    bool foundAssocTP = false;
    auto detId_key = cell.first;
    for (auto const& simtrack : cell.second) {
      if (tpTrackId.find(simtrack) != tpTrackId.end()) {
        foundAssocTP = true;
        mtdTrackId.insert(simtrack);
      }
    }
    if (foundAssocTP == false) {
      meUnassLgadsEnergy_->Fill(log10(m_etlHits[detId_key].energy));
      if (m_etlHits[detId_key].energy > depositETLthreshold_) {
        hcount++;
      }
    }
  }
  meUnassociatedDetId_->Fill(1.5, hcount);

  // Search for TP without associated hits and unassociated DetIds above threshold
  if (optionalPlots_) {
    for (const auto& tp : tpColl) {
      if (tp.eventId().bunchCrossing() == 0 && tp.eventId().event() == 0) {
        if (!mvaTPSel(tp)) {
          continue;
        }

        // test BTL crystals for association

        if (std::abs(tp.eta()) < trackMaxBtlEta_) {
          bool tpIsAssoc = false;
          bool goodCell = false;
          for (auto const& cell : m_btlTrkPerCell) {
            auto detId_key = cell.first;
            if (m_btlHits[detId_key].energy < depositBTLthreshold_) {
              continue;
            }

            BTLDetId detId(detId_key);
            DetId geoId = detId.geographicalId(MTDTopologyMode::crysLayoutFromTopoMode(topology->getMTDTopologyMode()));
            const MTDGeomDet* thedet = geom->idToDet(geoId);
            if (thedet == nullptr)
              throw cms::Exception("MtdTracksValidation") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                                          << detId.rawId() << ") is invalid!" << std::dec << std::endl;
            const ProxyMTDTopology& topoproxy = static_cast<const ProxyMTDTopology&>(thedet->topology());
            const RectangularMTDTopology& topo =
                static_cast<const RectangularMTDTopology&>(topoproxy.specificTopology());

            Local3DPoint local_point(convertMmToCm(m_btlHits[detId_key].x),
                                     convertMmToCm(m_btlHits[detId_key].y),
                                     convertMmToCm(m_btlHits[detId_key].z));

            local_point =
                topo.pixelToModuleLocalPoint(local_point, detId.row(topo.nrows()), detId.column(topo.nrows()));
            const auto& global_point = thedet->toGlobal(local_point);

            if (std::abs(tp.eta() - global_point.eta()) > etaMatchCut_) {
              continue;
            }
            goodCell = true;
            for (auto const& simtrack : cell.second) {
              for (auto const& TPsimtrack : tp.g4Tracks()) {
                auto const testId = uniqueId(TPsimtrack.trackId(), TPsimtrack.eventId());
                if (simtrack == testId) {
                  tpIsAssoc = true;
                  break;
                }
              }
            }
          }  //cell Loop
          if (!tpIsAssoc && goodCell) {
            meUnassDeposit_->Fill(0.5);
          }

        } else {
          // test ETL LGADs for association
          bool tpIsAssoc = false;
          bool goodCell = false;
          for (auto const& cell : m_etlTrkPerCell) {
            auto detId_key = cell.first;
            if (m_etlHits[detId_key].energy < depositETLthreshold_) {
              continue;
            }

            ETLDetId detId(detId_key);
            DetId geoId = detId.geographicalId();
            const MTDGeomDet* thedet = geom->idToDet(geoId);
            if (thedet == nullptr)
              throw cms::Exception("MtdTracksValidation") << "GeographicalID: " << std::hex << geoId.rawId() << " ("
                                                          << detId.rawId() << ") is invalid!" << std::dec << std::endl;

            Local3DPoint local_point(convertMmToCm(m_etlHits[detId_key].x),
                                     convertMmToCm(m_etlHits[detId_key].y),
                                     convertMmToCm(m_etlHits[detId_key].z));
            const auto& global_point = thedet->toGlobal(local_point);

            if (std::abs(tp.eta() - global_point.eta()) > etaMatchCut_) {
              continue;
            }
            goodCell = true;
            for (auto const& simtrack : cell.second) {
              for (auto const& TPsimtrack : tp.g4Tracks()) {
                auto const testId = uniqueId(TPsimtrack.trackId(), TPsimtrack.eventId());
                if (simtrack == testId) {
                  tpIsAssoc = true;
                  break;
                }
              }
            }
          }  //cell Loop
          if (!tpIsAssoc && goodCell) {
            meUnassDeposit_->Fill(1.5);
          }
        }  // tp BTL/ETL acceptance selection
      }
    }  //tp Loop
  }    //optionalPlots

  unsigned int index = 0;

  // flag to select events with reco vertex close to true simulated primary vertex, or PV fake (particle guns)
  const bool isGoodVtx = std::abs(primRecoVtx.z() - zsim) < deltaZcut_ || primRecoVtx.isFake();

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

    bool isBTL = false;
    bool twoETLdiscs = false;

    if (track.pt() >= trackMinPt_ && std::abs(track.eta()) <= trackMaxEtlEta_) {
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
      meTrackSigmat0SafePid_->Fill(Sigmat0Safe[trackref]);
      meTrackMVAQual_->Fill(mtdQualMVA[trackref]);

      meTrackPathLenghtvsEta_->Fill(std::abs(track.eta()), pathLength[trackref]);

      if (std::abs(track.eta()) < trackMaxBtlEta_) {
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
          isBTL = true;
          meBTLTrackEffEtaMtd_->Fill(track.eta());
          meBTLTrackEffPhiMtd_->Fill(track.phi());
          meBTLTrackEffPtMtd_->Fill(track.pt());
          meBTLTrackRPTime_->Fill(track.t0());
          meBTLTrackPtRes_->Fill((trackGen.pt() - track.pt()) / trackGen.pt());
        }
      }  //loop over (geometrical) BTL tracks

      else {
        // --- all ETL tracks (with and without hit in MTD) ---
        if ((track.eta() < -trackMinEtlEta_) && (track.eta() > -trackMaxEtlEta_)) {
          meETLTrackEffEtaTot_[0]->Fill(track.eta());
          meETLTrackEffPhiTot_[0]->Fill(track.phi());
          meETLTrackEffPtTot_[0]->Fill(track.pt());
        }

        if ((track.eta() > trackMinEtlEta_) && (track.eta() < trackMaxEtlEta_)) {
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
        if ((track.eta() < -trackMinEtlEta_) && (track.eta() > -trackMaxEtlEta_)) {
          twoETLdiscs = (MTDEtlZnegD1 == true) && (MTDEtlZnegD2 == true);
          if ((MTDEtlZnegD1 == true) || (MTDEtlZnegD2 == true)) {
            meETLTrackEffEtaMtd_[0]->Fill(track.eta());
            meETLTrackEffPhiMtd_[0]->Fill(track.phi());
            meETLTrackEffPtMtd_[0]->Fill(track.pt());
            if (twoETLdiscs) {
              meETLTrackEffEta2Mtd_[0]->Fill(track.eta());
              meETLTrackEffPhi2Mtd_[0]->Fill(track.phi());
              meETLTrackEffPt2Mtd_[0]->Fill(track.pt());
            }
          }
        }
        if ((track.eta() > trackMinEtlEta_) && (track.eta() < trackMaxEtlEta_)) {
          twoETLdiscs = (MTDEtlZposD1 == true) && (MTDEtlZposD2 == true);
          if ((MTDEtlZposD1 == true) || (MTDEtlZposD2 == true)) {
            meETLTrackEffEtaMtd_[1]->Fill(track.eta());
            meETLTrackEffPhiMtd_[1]->Fill(track.phi());
            meETLTrackEffPtMtd_[1]->Fill(track.pt());
            if (twoETLdiscs) {
              meETLTrackEffEta2Mtd_[1]->Fill(track.eta());
              meETLTrackEffPhi2Mtd_[1]->Fill(track.phi());
              meETLTrackEffPt2Mtd_[1]->Fill(track.pt());
            }
          }
        }
      }
    }

    if (isGoodVtx) {
      bool noCrack = std::abs(trackGen.eta()) < trackMaxBtlEta_ || std::abs(trackGen.eta()) > trackMinEtlEta_;
      const bool vtxFake = primRecoVtx.isFake();

      if (mvaRecSel(trackGen, primRecoVtx, t0Safe[trackref], Sigmat0Safe[trackref])) {
        // reco-gen matching used for MVA quality flag

        if (noCrack) {
          meMVATrackEffPtTot_->Fill(trackGen.pt());
        }
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
            if (mvaGenRecMatch(*genP, zsim, trackGen, vtxFake)) {
              meMVATrackZposResTot_->Fill(dZ);
              if (noCrack) {
                meMVATrackMatchedEffPtTot_->Fill(trackGen.pt());
              }
              meMVATrackMatchedEffEtaTot_->Fill(std::abs(trackGen.eta()));
              if (pullT > -9999.) {
                meMVATrackResTot_->Fill(dT);
                meMVATrackPullTot_->Fill(pullT);
                if (noCrack) {
                  meMVATrackMatchedEffPtMtd_->Fill(trackGen.pt());
                }
                meMVATrackMatchedEffEtaMtd_->Fill(std::abs(trackGen.eta()));
              }
              break;
            }
          }
        }

        // TrackingParticle based matching

        const reco::TrackBaseRef trkrefb(trackref);
        auto tp_info = getMatchedTP(trkrefb, zsim);

        if (tp_info != nullptr && mvaTPSel(**tp_info)) {
          const bool withMTD = tpWithMTD(**tp_info, mtdTrackId);
          if (noCrack) {
            meTrackMatchedTPEffPtTot_->Fill(trackGen.pt());
            if (withMTD) {
              meTrackMatchedTPmtdEffPtTot_->Fill(trackGen.pt());
            }
          }
          meTrackMatchedTPEffEtaTot_->Fill(std::abs(trackGen.eta()));
          if (withMTD) {
            meTrackMatchedTPmtdEffEtaTot_->Fill(std::abs(trackGen.eta()));
          }
          if (pullT > -9999.) {
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
        }
      }
    }  // MC truth matich analysis for good PV
  }    //RECO tracks loop
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
  meETLTrackEffEta2Mtd_[0] =
      ibook.book1D("TrackETLEffEta2MtdZneg", "Track efficiency vs eta (Mtd 2 hit) (-Z);#eta_{RECO}", 100, -3.2, -1.4);
  meETLTrackEffEta2Mtd_[1] =
      ibook.book1D("TrackETLEffEta2MtdZpos", "Track efficiency vs eta (Mtd 2 hit) (+Z);#eta_{RECO}", 100, 1.4, 3.2);
  meETLTrackEffPhi2Mtd_[0] = ibook.book1D(
      "TrackETLEffPhi2MtdZneg", "Track efficiency vs phi (Mtd 2 hit) (-Z);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meETLTrackEffPhi2Mtd_[1] = ibook.book1D(
      "TrackETLEffPhi2MtdZpos", "Track efficiency vs phi (Mtd 2 hit) (+Z);#phi_{RECO} [rad]", 100, -3.2, 3.2);
  meETLTrackEffPt2Mtd_[0] =
      ibook.book1D("TrackETLEffPt2MtdZneg", "Track efficiency vs pt (Mtd 2 hit) (-Z);pt_{RECO} [GeV]", 50, 0, 10);
  meETLTrackEffPt2Mtd_[1] =
      ibook.book1D("TrackETLEffPt2MtdZpos", "Track efficiency vs pt (Mtd 2 hit) (+Z);pt_{RECO} [GeV]", 50, 0, 10);
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

  meTrackMatchedTPEffPtTot_ =
      ibook.book1D("MatchedTPEffPtTot", "Pt of tracks associated to LV matched to TP; track pt [GeV] ", 110, 0., 11.);
  meTrackMatchedTPEffPtMtd_ = ibook.book1D(
      "MatchedTPEffPtMtd", "Pt of tracks associated to LV matched to TP with time; track pt [GeV] ", 110, 0., 11.);
  meTrackMatchedTPEffPtEtl2Mtd_ =
      ibook.book1D("MatchedTPEffPtEtl2Mtd",
                   "Pt of tracks associated to LV matched to TP with time, 2 ETL hits; track pt [GeV] ",
                   110,
                   0.,
                   11.);

  meTrackMatchedTPmtdEffPtTot_ = ibook.book1D(
      "MatchedTPmtdEffPtTot", "Pt of tracks associated to LV matched to TP-mtd hit; track pt [GeV] ", 110, 0., 11.);
  meTrackMatchedTPmtdEffPtMtd_ =
      ibook.book1D("MatchedTPmtdEffPtMtd",
                   "Pt of tracks associated to LV matched to TP-mtd hit with time; track pt [GeV] ",
                   110,
                   0.,
                   11.);

  meMVATrackEffEtaTot_ = ibook.book1D("MVAEffEtaTot", "Eta of tracks associated to LV; track eta ", 66, 0., 3.3);
  meMVATrackMatchedEffEtaTot_ =
      ibook.book1D("MVAMatchedEffEtaTot", "Eta of tracks associated to LV matched to GEN; track eta ", 66, 0., 3.3);
  meMVATrackMatchedEffEtaMtd_ = ibook.book1D(
      "MVAMatchedEffEtaMtd", "Eta of tracks associated to LV matched to GEN with time; track eta ", 66, 0., 3.3);

  meTrackMatchedTPEffEtaTot_ =
      ibook.book1D("MatchedTPEffEtaTot", "Eta of tracks associated to LV matched to TP; track eta ", 66, 0., 3.3);
  meTrackMatchedTPEffEtaMtd_ = ibook.book1D(
      "MatchedTPEffEtaMtd", "Eta of tracks associated to LV matched to TP with time; track eta ", 66, 0., 3.3);
  meTrackMatchedTPEffEtaEtl2Mtd_ =
      ibook.book1D("MatchedTPEffEtaEtl2Mtd",
                   "Eta of tracks associated to LV matched to TP with time, 2 ETL hits; track eta ",
                   66,
                   0.,
                   3.3);

  meTrackMatchedTPmtdEffEtaTot_ = ibook.book1D(
      "MatchedTPmtdEffEtaTot", "Eta of tracks associated to LV matched to TP-mtd hit; track eta ", 66, 0., 3.3);
  meTrackMatchedTPmtdEffEtaMtd_ =
      ibook.book1D("MatchedTPmtdEffEtaMtd",
                   "Eta of tracks associated to LV matched to TP-mtd hit with time; track eta ",
                   66,
                   0.,
                   3.3);

  meMVATrackResTot_ = ibook.book1D(
      "MVATrackRes", "t_{rec} - t_{sim} for LV associated tracks; t_{rec} - t_{sim} [ns] ", 120, -0.15, 0.15);
  meMVATrackPullTot_ =
      ibook.book1D("MVATrackPull", "Pull for associated tracks; (t_{rec}-t_{sim})/#sigma_{t}", 50, -5., 5.);
  meMVATrackZposResTot_ = ibook.book1D(
      "MVATrackZposResTot", "Z_{PCA} - Z_{sim} for associated tracks;Z_{PCA} - Z_{sim} [cm] ", 100, -0.1, 0.1);

  meUnassociatedDetId_ = ibook.bookProfile(
      "UnassociatedDetId", "Number of MTD cell not associated to any TP per event", 2, 0., 2., 0., 100000., "S");
  meNTrackingParticles_ = ibook.book1D("NTrackingParticles", "Total #Tracking particles", 2, 0, 2);
  meUnassDeposit_ =
      ibook.book1D("UnassDeposit",
                   "#Tracking particles with deposit over threshold in MTD cell, but with no cell associated to TP;",
                   2,
                   0,
                   2);
  meUnassCrysEnergy_ =
      ibook.book1D("UnassCrysEnergy",
                   "Energy deposit in BTL crystal with no associated SimTracks;log_{10}(Energy [MeV]) ",
                   100,
                   -3.5,
                   1.5);
  meUnassLgadsEnergy_ = ibook.book1D("UnassLgadsEnergy",
                                     "Energy deposit in ETL LGADs with no associated SimTracks;log_{10}(Energy [MeV]) ",
                                     100,
                                     -3.5,
                                     1.5);
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
  desc.add<edm::InputTag>("btlSimHits", edm::InputTag("mix", "g4SimHitsFastTimerHitsBarrel"));
  desc.add<edm::InputTag>("etlSimHits", edm::InputTag("mix", "g4SimHitsFastTimerHitsEndcap"));
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
  desc.add<double>("trackMinimumPt", 0.7);  // [GeV]
  desc.add<double>("trackMaximumBtlEta", 1.5);
  desc.add<double>("trackMinimumEtlEta", 1.6);
  desc.add<double>("trackMaximumEtlEta", 3.);
  desc.addUntracked<bool>("optionalPlots", true);

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

const bool MtdTracksValidation::mvaTPSel(const TrackingParticle& tp) {
  bool match = false;
  if (tp.status() != 1) {
    return match;
  }
  auto x_pv = tp.parentVertex()->position().x();
  auto y_pv = tp.parentVertex()->position().y();
  auto z_pv = tp.parentVertex()->position().z();

  auto r_pv = std::sqrt(x_pv * x_pv + y_pv * y_pv);

  match = tp.charge() != 0 && tp.pt() > pTcut_ && std::abs(tp.eta()) < etacutGEN_ && r_pv < rBTL_ && z_pv < zETL_;
  return match;
}

const bool MtdTracksValidation::mvaRecSel(const reco::TrackBase& trk,
                                          const reco::Vertex& vtx,
                                          const double& t0,
                                          const double& st0) {
  bool match = false;
  match = trk.pt() > pTcut_ && std::abs(trk.eta()) < etacutREC_ &&
          (std::abs(trk.vz() - vtx.z()) <= deltaZcut_ || vtx.isFake());
  if (st0 > 0.) {
    match = match && std::abs(t0 - vtx.t()) < 3. * st0;
  }
  return match;
}

const bool MtdTracksValidation::mvaGenRecMatch(const HepMC::GenParticle& genP,
                                               const double& zsim,
                                               const reco::TrackBase& trk,
                                               const bool& vtxFake) {
  bool match = false;
  double dR = reco::deltaR(genP.momentum(), trk.momentum());
  double genPT = genP.momentum().perp();
  match = std::abs(genPT - trk.pt()) < trk.pt() * deltaPTcut_ && dR < deltaDRcut_ &&
          (std::abs(trk.vz() - zsim) < deltaZcut_ || vtxFake);
  return match;
}

const edm::Ref<std::vector<TrackingParticle>>* MtdTracksValidation::getMatchedTP(const reco::TrackBaseRef& recoTrack,
                                                                                 const double& zsim) {
  auto found = r2s_->find(recoTrack);

  // reco track not matched to any TP
  if (found == r2s_->end())
    return nullptr;

  //matched TP equal to any TP associated to signal sim vertex
  for (const auto& tp : found->val) {
    if (tp.first->eventId().bunchCrossing() == 0 && tp.first->eventId().event() == 0 &&
        std::abs(tp.first->parentVertex()->position().z() - zsim) < deltaZcut_) {
      return &tp.first;
    }
  }

  // reco track not matched to any TP from vertex
  return nullptr;
}

const bool MtdTracksValidation::tpWithMTD(const TrackingParticle& tp,
                                          const std::unordered_set<unsigned long int>& trkList) {
  for (const auto& simTrk : tp.g4Tracks()) {
    for (const auto& mtdTrk : trkList) {
      if (uniqueId(simTrk.trackId(), simTrk.eventId()) == mtdTrk) {
        return true;
      }
    }
  }
  return false;
}

DEFINE_FWK_MODULE(MtdTracksValidation);
