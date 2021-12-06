#include "Validation/RecoB/plugins/BDHadronTrackMonitoringAnalyzer.h"

using namespace reco;
using namespace edm;
using namespace std;

const reco::TrackBaseRef toTrackRef(const edm::Ptr<reco::Candidate> &cnd) {
  const reco::PFCandidate *pfcand = dynamic_cast<const reco::PFCandidate *>(cnd.get());

  if ((std::abs(pfcand->pdgId()) == 11 || pfcand->pdgId() == 22) && pfcand->gsfTrackRef().isNonnull() &&
      pfcand->gsfTrackRef().isAvailable())
    return reco::TrackBaseRef(pfcand->gsfTrackRef());
  else if (pfcand->trackRef().isNonnull() && pfcand->trackRef().isAvailable())
    return reco::TrackBaseRef(pfcand->trackRef());
  else
    return reco::TrackBaseRef();
}

// ---------- Static member declaration -----------
const std::vector<std::string> BDHadronTrackMonitoringAnalyzer::TrkHistCat = {
    "BCWeakDecay", "BWeakDecay", "CWeakDecay", "PU", "Other", "Fake"};

// ---------- Constructor -----------
BDHadronTrackMonitoringAnalyzer::BDHadronTrackMonitoringAnalyzer(const edm::ParameterSet &pSet)
    : distJetAxis_(pSet.getParameter<double>("distJetAxisCut")),
      decayLength_(pSet.getParameter<double>("decayLengthCut")),
      minJetPt_(pSet.getParameter<double>("minJetPt")),
      maxJetEta_(pSet.getParameter<double>("maxJetEta")),
      ipTagInfos_(pSet.getParameter<std::string>("ipTagInfos")),
      PatJetSrc_(pSet.getParameter<InputTag>("PatJetSource")),
      TrackSrc_(pSet.getParameter<InputTag>("TrackSource")),
      PVSrc_(pSet.getParameter<InputTag>("PrimaryVertexSource")),
      ClusterTPMapSrc_(pSet.getParameter<InputTag>("clusterTPMap")),
      classifier_(pSet, consumesCollector())

{
  PatJetCollectionTag_ = consumes<pat::JetCollection>(PatJetSrc_);
  TrackCollectionTag_ = consumes<reco::TrackCollection>(TrackSrc_);
  PrimaryVertexColl_ = consumes<reco::VertexCollection>(PVSrc_);
  clusterTPMapToken_ = consumes<ClusterTPAssociation>(ClusterTPMapSrc_);
  ttrackToken_ = esConsumes(edm::ESInputTag("", "TransientTrackBuilder"));
  // TrkHistCat = {"BCWeakDecay", "BWeakDecay", "CWeakDecay", "PU", "Other",
  // "Fake"};
}

// ---------- BookHistograms -----------

void BDHadronTrackMonitoringAnalyzer::bookHistograms(DQMStore::IBooker &ibook,
                                                     edm::Run const &run,
                                                     edm::EventSetup const &es) {
  ibook.setCurrentFolder("BDHadronTracks/JetContent");
  //
  // Book all histograms.
  //
  RecoBTag::setTDRStyle();

  nTrkAll_bjet = ibook.book1D(
      "nTrkAll_bjet", "Number of selected tracks in b jets;number of selected tracks;jets", 16, -0.5, 15.5);

  nTrkAll_cjet = ibook.book1D(
      "nTrkAll_cjet", "Number of selected tracks in c jets;number of selected tracks;jets", 16, -0.5, 15.5);

  nTrkAll_dusgjet = ibook.book1D(
      "nTrkAll_dusgjet", "Number of selected tracks in dusg jets;number of selected tracks;jets", 16, -0.5, 15.5);

  // Loop over different Track History Categories
  for (unsigned int i = 0; i < TrkHistCat.size(); i++) {
    ibook.setCurrentFolder("BDHadronTracks/JetContent");
    // b jets
    nTrk_bjet[i] = ibook.book1D("nTrk_bjet_" + TrkHistCat[i],
                                "Number of selected tracks in b jets (" + TrkHistCat[i] +
                                    ");number of selected tracks (" + TrkHistCat[i] + ");jets",
                                16,
                                -0.5,
                                15.5);

    // c jets
    nTrk_cjet[i] = ibook.book1D("nTrk_cjet_" + TrkHistCat[i],
                                "Number of selected tracks in c jets (" + TrkHistCat[i] +
                                    ");number of selected tracks (" + TrkHistCat[i] + ");jets",
                                16,
                                -0.5,
                                15.5);

    // dusg jets
    nTrk_dusgjet[i] = ibook.book1D("nTrk_dusgjet_" + TrkHistCat[i],
                                   "Number of selected tracks in dusg jets (" + TrkHistCat[i] +
                                       ");number of selected tracks (" + TrkHistCat[i] + ");jets",
                                   16,
                                   -0.5,
                                   15.5);

    ibook.setCurrentFolder("BDHadronTracks/TrackInfo");
    // track properties for all flavours combined
    TrkPt_alljets[i] = ibook.book1D("TrkPt_" + TrkHistCat[i],
                                    "Track pT (" + TrkHistCat[i] + ");track p_{T} (" + TrkHistCat[i] + ");tracks",
                                    30,
                                    0,
                                    100);
    TrkEta_alljets[i] = ibook.book1D("TrkEta_" + TrkHistCat[i],
                                     "Track #eta (" + TrkHistCat[i] + ");track #eta (" + TrkHistCat[i] + ");tracks",
                                     30,
                                     -2.5,
                                     2.5);
    TrkPhi_alljets[i] = ibook.book1D("TrkPhi_" + TrkHistCat[i],
                                     "Track #phi (" + TrkHistCat[i] + ");track #phi (" + TrkHistCat[i] + ");tracks",
                                     30,
                                     -3.15,
                                     3.15);
    TrkDxy_alljets[i] = ibook.book1D("TrkDxy_" + TrkHistCat[i],
                                     "Track dxy (" + TrkHistCat[i] + ");track dxy (" + TrkHistCat[i] + ");tracks",
                                     30,
                                     -0.1,
                                     0.1);
    TrkDz_alljets[i] = ibook.book1D("TrkDz_" + TrkHistCat[i],
                                    "Track dz (" + TrkHistCat[i] + ");track dz (" + TrkHistCat[i] + ");tracks",
                                    30,
                                    -0.1,
                                    0.1);
    TrkHitAll_alljets[i] = ibook.book1D(
        "TrkHitAll_" + TrkHistCat[i],
        "Number of tracker hits (" + TrkHistCat[i] + ");track number of all hits (" + TrkHistCat[i] + ");tracks",
        31,
        -0.5,
        30.5);
    TrkHitStrip_alljets[i] = ibook.book1D(
        "TrkHitStrip_" + TrkHistCat[i],
        "Number of strip hits (" + TrkHistCat[i] + ");track number of strip hits (" + TrkHistCat[i] + ");tracks",
        31,
        -0.5,
        30.5);
    TrkHitPixel_alljets[i] = ibook.book1D(
        "TrkHitPixel_" + TrkHistCat[i],
        "Number of pixel hits (" + TrkHistCat[i] + ");track number of pixel hits (" + TrkHistCat[i] + ");tracks",
        9,
        -0.5,
        8.5);

    ibook.setCurrentFolder("BDHadronTracks/TrackTruthInfo");
    if (i < 5) {  // Fakes (i == 5) have no truth by definition!
      TrkTruthPt_alljets[i] =
          ibook.book1D("TrkTruthPt_" + TrkHistCat[i],
                       "Track pT (" + TrkHistCat[i] + " Truth);track p_{T} (" + TrkHistCat[i] + " Truth);tracks",
                       30,
                       0,
                       100);
      TrkTruthEta_alljets[i] =
          ibook.book1D("TrkTruthEta_" + TrkHistCat[i],
                       "Track #eta (" + TrkHistCat[i] + " Truth);track #eta (" + TrkHistCat[i] + " Truth);tracks",
                       30,
                       -2.5,
                       2.5);
      TrkTruthPhi_alljets[i] =
          ibook.book1D("TrkTruthPhi_" + TrkHistCat[i],
                       "Track #phi (" + TrkHistCat[i] + " Truth);track #phi (" + TrkHistCat[i] + " Truth);tracks",
                       30,
                       -3.15,
                       3.15);
      TrkTruthDxy_alljets[i] =
          ibook.book1D("TrkTruthDxy_" + TrkHistCat[i],
                       "Track dxy (" + TrkHistCat[i] + " Truth);track dxy (" + TrkHistCat[i] + " Truth);tracks",
                       30,
                       -0.1,
                       0.1);
      TrkTruthDz_alljets[i] =
          ibook.book1D("TrkTruthDz_" + TrkHistCat[i],
                       "Track dz (" + TrkHistCat[i] + " Truth);track dz (" + TrkHistCat[i] + " Truth);tracks",
                       30,
                       -0.1,
                       0.1);
      TrkTruthHitAll_alljets[i] =
          ibook.book1D("TrkTruthHitAll_" + TrkHistCat[i],
                       "Number of tracker hits (" + TrkHistCat[i] + " Truth);track number of all hits (" +
                           TrkHistCat[i] + " Truth);tracks",
                       31,
                       -0.5,
                       30.5);
      TrkTruthHitStrip_alljets[i] =
          ibook.book1D("TrkTruthHitStrip_" + TrkHistCat[i],
                       "Number of strip hits (" + TrkHistCat[i] + " Truth);track number of strip hits (" +
                           TrkHistCat[i] + " Truth);tracks",
                       31,
                       -0.5,
                       30.5);
      TrkTruthHitPixel_alljets[i] =
          ibook.book1D("TrkTruthHitPixel_" + TrkHistCat[i],
                       "Number of pixel hits (" + TrkHistCat[i] + " Truth);track number of pixel hits (" +
                           TrkHistCat[i] + " Truth);tracks",
                       9,
                       -0.5,
                       8.5);
    }
  }
}

// ---------- Destructor -----------

BDHadronTrackMonitoringAnalyzer::~BDHadronTrackMonitoringAnalyzer() {}

// ---------- Analyze -----------
// This is needed to get a TrackingParticle --> Cluster match (instead of
// Cluster-->TP)
using P = std::pair<OmniClusterRef, TrackingParticleRef>;
bool compare(const P &i, const P &j) { return i.second.index() > j.second.index(); }

void BDHadronTrackMonitoringAnalyzer::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  edm::Handle<pat::JetCollection> patJetsColl;
  iEvent.getByToken(PatJetCollectionTag_, patJetsColl);

  edm::Handle<reco::TrackCollection> tracksHandle;
  iEvent.getByToken(TrackCollectionTag_, tracksHandle);

  edm::Handle<ClusterTPAssociation> pCluster2TPListH;
  iEvent.getByToken(clusterTPMapToken_, pCluster2TPListH);
  const ClusterTPAssociation &clusterToTPMap = *pCluster2TPListH;

  //edm::ESHandle<TransientTrackBuilder>
  const auto &trackBuilder = iSetup.getHandle(ttrackToken_);

  classifier_.newEvent(iEvent, iSetup);

  // -----Primary Vertex-----
  const reco::Vertex *pv;

  edm::Handle<reco::VertexCollection> primaryVertex;
  iEvent.getByToken(PrimaryVertexColl_, primaryVertex);

  bool pvFound = (!primaryVertex->empty());
  if (pvFound) {
    pv = &(*primaryVertex->begin());
  } else {
    reco::Vertex::Error e;
    e(0, 0) = 0.0015 * 0.0015;
    e(1, 1) = 0.0015 * 0.0015;
    e(2, 2) = 15. * 15.;
    reco::Vertex::Point p(0, 0, 0);
    pv = new reco::Vertex(p, e, 1, 1, 1);
  }
  // -----------------------

  // -------- Loop Over Jets ----------
  for (pat::JetCollection::const_iterator jet = patJetsColl->begin(); jet != patJetsColl->end(); ++jet) {
    if (jet->pt() < minJetPt_ || std::fabs(jet->eta()) > maxJetEta_)
      continue;

    unsigned int flav = abs(jet->hadronFlavour());

    // std::cout << "patJet collection has pfImpactParameterTagInfo?: " <<
    // jet->hasTagInfo("pfImpactParameter") << std::endl;
    const CandIPTagInfo *trackIpTagInfo = jet->tagInfoCandIP(ipTagInfos_);
    const std::vector<edm::Ptr<reco::Candidate>> &selectedTracks(trackIpTagInfo->selectedTracks());

    unsigned int nseltracks = 0;
    std::vector<int> nseltracksCat(TrkHistCat.size(),
                                   0);  // following the order of TrkHistCat

    unsigned int nTrackSize = selectedTracks.size();  // number of tracks from IPInfos to loop over
    // -------- Loop Over (selected) Tracks ----------
    for (unsigned int itt = 0; itt < nTrackSize; ++itt) {
      const TrackBaseRef ptrackRef = toTrackRef(selectedTracks[itt]);
      const reco::Track *ptrackPtr = reco::btag::toTrack(ptrackRef);
      const reco::Track &ptrack = *ptrackPtr;

      reco::TransientTrack transientTrack = trackBuilder->build(ptrackPtr);
      GlobalVector direction(jet->px(), jet->py(), jet->pz());

      Double_t distJetAxis = IPTools::jetTrackDistance(transientTrack, direction, *pv).second.value();

      Double_t decayLength = 999;
      TrajectoryStateOnSurface closest =
          IPTools::closestApproachToJet(transientTrack.impactPointState(), *pv, direction, transientTrack.field());
      if (closest.isValid())
        decayLength = (closest.globalPosition() - RecoVertex::convertPos(pv->position())).mag();
      else
        decayLength = 999;

      // extra cut ons the tracks
      if (std::fabs(distJetAxis) > distJetAxis_ || decayLength > decayLength_) {
        continue;
      }
      nseltracks += 1;  // if it passed these cuts, nselectedtracks +1

      TrackCategories::Flags theFlag = classifier_.evaluate(toTrackRef(selectedTracks[itt])).flags();

      double TrkPt = ptrack.pt();
      double TrkEta = ptrack.eta();
      double TrkPhi = ptrack.phi();
      double TrkDxy = ptrack.dxy(pv->position());
      double TrkDz = ptrack.dz(pv->position());
      int TrknHitAll = ptrack.numberOfValidHits();
      int TrknHitPixel = ptrack.hitPattern().numberOfValidPixelHits();
      int TrknHitStrip = ptrack.hitPattern().numberOfValidStripHits();

      double TrkTruthPt = -99;
      double TrkTruthEta = -99;
      double TrkTruthPhi = -99;
      double TrkTruthDxy = -1;
      double TrkTruthDz = -1;
      int TrkTruthnHitAll = -1;
      int TrkTruthnHitPixel = -1;
      int TrkTruthnHitStrip = -1;

      // Get corresponding Trackingparticle
      std::pair<TrackingParticleRef, double> res = classifier_.history().getMatchedTrackingParticle();
      TrackingParticleRef tpr = res.first;
      double quality_tpr = res.second;

      // Match TP to hit-cluster (re-ordering according to TP rather than
      // clusters and look for equal_range of a given tpr)
      auto clusterTPmap = clusterToTPMap.map();
      std::sort(clusterTPmap.begin(), clusterTPmap.end(), compare);
      auto clusterRange =
          std::equal_range(clusterTPmap.begin(), clusterTPmap.end(), std::make_pair(OmniClusterRef(), tpr), compare);
      if (quality_tpr != 0) {
        TrkTruthPt = tpr->pt();
        TrkTruthEta = tpr->eta();
        TrkTruthPhi = tpr->phi();

        const TrackingParticle::Point &vertex_pv = pv->position();
        TrackingParticle::Point vertex_tpr = tpr->vertex();
        TrackingParticle::Vector momentum_tpr = tpr->momentum();
        TrkTruthDxy = (-(vertex_tpr.x() - vertex_pv.x()) * momentum_tpr.y() +
                       (vertex_tpr.y() - vertex_pv.y()) * momentum_tpr.x()) /
                      tpr->pt();
        TrkTruthDz = (vertex_tpr.z() - vertex_pv.z()) - ((vertex_tpr.x() - vertex_pv.x()) * momentum_tpr.x() +
                                                         (vertex_tpr.y() - vertex_pv.y()) * momentum_tpr.y()) /
                                                            sqrt(momentum_tpr.perp2()) * momentum_tpr.z() /
                                                            sqrt(momentum_tpr.perp2());

        TrkTruthnHitAll = 0;
        TrkTruthnHitPixel = 0;
        TrkTruthnHitStrip = 0;
        if (clusterRange.first != clusterRange.second) {
          for (auto ip = clusterRange.first; ip != clusterRange.second; ++ip) {
            const OmniClusterRef &cluster = ip->first;
            if (cluster.isPixel() && cluster.isValid()) {
              TrkTruthnHitPixel += 1;
            }
            if (cluster.isStrip() && cluster.isValid()) {
              TrkTruthnHitStrip += 1;
            }
          }
        }
        TrkTruthnHitAll = TrkTruthnHitPixel + TrkTruthnHitStrip;
      }

      // ----------- Filling the correct histograms based on jet flavour and
      // Track history Category --------

      // BCWeakDecay
      if (theFlag[TrackCategories::SignalEvent] && theFlag[TrackCategories::BWeakDecay] &&
          theFlag[TrackCategories::CWeakDecay]) {
        nseltracksCat[BDHadronTrackMonitoringAnalyzer::BCWeakDecay] += 1;
        TrkPt_alljets[BDHadronTrackMonitoringAnalyzer::BCWeakDecay]->Fill(TrkPt);
        TrkEta_alljets[BDHadronTrackMonitoringAnalyzer::BCWeakDecay]->Fill(TrkEta);
        TrkPhi_alljets[BDHadronTrackMonitoringAnalyzer::BCWeakDecay]->Fill(TrkPhi);
        TrkDxy_alljets[BDHadronTrackMonitoringAnalyzer::BCWeakDecay]->Fill(TrkDxy);
        TrkDz_alljets[BDHadronTrackMonitoringAnalyzer::BCWeakDecay]->Fill(TrkDz);
        TrkHitAll_alljets[BDHadronTrackMonitoringAnalyzer::BCWeakDecay]->Fill(TrknHitAll);
        TrkHitPixel_alljets[BDHadronTrackMonitoringAnalyzer::BCWeakDecay]->Fill(TrknHitPixel);
        TrkHitStrip_alljets[BDHadronTrackMonitoringAnalyzer::BCWeakDecay]->Fill(TrknHitStrip);
        if (quality_tpr != 0) {
          TrkTruthPt_alljets[BDHadronTrackMonitoringAnalyzer::BCWeakDecay]->Fill(TrkTruthPt);
          TrkTruthEta_alljets[BDHadronTrackMonitoringAnalyzer::BCWeakDecay]->Fill(TrkTruthEta);
          TrkTruthPhi_alljets[BDHadronTrackMonitoringAnalyzer::BCWeakDecay]->Fill(TrkTruthPhi);
          TrkTruthDxy_alljets[BDHadronTrackMonitoringAnalyzer::BCWeakDecay]->Fill(TrkTruthDxy);
          TrkTruthDz_alljets[BDHadronTrackMonitoringAnalyzer::BCWeakDecay]->Fill(TrkTruthDz);
          TrkTruthHitAll_alljets[BDHadronTrackMonitoringAnalyzer::BCWeakDecay]->Fill(TrkTruthnHitAll);
          TrkTruthHitPixel_alljets[BDHadronTrackMonitoringAnalyzer::BCWeakDecay]->Fill(TrkTruthnHitPixel);
          TrkTruthHitStrip_alljets[BDHadronTrackMonitoringAnalyzer::BCWeakDecay]->Fill(TrkTruthnHitStrip);
        }
      }
      // BWeakDecay
      else if (theFlag[TrackCategories::SignalEvent] && theFlag[TrackCategories::BWeakDecay] &&
               !theFlag[TrackCategories::CWeakDecay]) {
        nseltracksCat[BDHadronTrackMonitoringAnalyzer::BWeakDecay] += 1;
        TrkPt_alljets[BDHadronTrackMonitoringAnalyzer::BWeakDecay]->Fill(TrkPt);
        TrkEta_alljets[BDHadronTrackMonitoringAnalyzer::BWeakDecay]->Fill(TrkEta);
        TrkPhi_alljets[BDHadronTrackMonitoringAnalyzer::BWeakDecay]->Fill(TrkPhi);
        TrkDxy_alljets[BDHadronTrackMonitoringAnalyzer::BWeakDecay]->Fill(TrkDxy);
        TrkDz_alljets[BDHadronTrackMonitoringAnalyzer::BWeakDecay]->Fill(TrkDz);
        TrkHitAll_alljets[BDHadronTrackMonitoringAnalyzer::BWeakDecay]->Fill(TrknHitAll);
        TrkHitPixel_alljets[BDHadronTrackMonitoringAnalyzer::BWeakDecay]->Fill(TrknHitPixel);
        TrkHitStrip_alljets[BDHadronTrackMonitoringAnalyzer::BWeakDecay]->Fill(TrknHitStrip);
        if (quality_tpr != 0) {
          TrkTruthPt_alljets[BDHadronTrackMonitoringAnalyzer::BWeakDecay]->Fill(TrkTruthPt);
          TrkTruthEta_alljets[BDHadronTrackMonitoringAnalyzer::BWeakDecay]->Fill(TrkTruthEta);
          TrkTruthPhi_alljets[BDHadronTrackMonitoringAnalyzer::BWeakDecay]->Fill(TrkTruthPhi);
          TrkTruthDxy_alljets[BDHadronTrackMonitoringAnalyzer::BWeakDecay]->Fill(TrkTruthDxy);
          TrkTruthDz_alljets[BDHadronTrackMonitoringAnalyzer::BWeakDecay]->Fill(TrkTruthDz);
          TrkTruthHitAll_alljets[BDHadronTrackMonitoringAnalyzer::BWeakDecay]->Fill(TrkTruthnHitAll);
          TrkTruthHitPixel_alljets[BDHadronTrackMonitoringAnalyzer::BWeakDecay]->Fill(TrkTruthnHitPixel);
          TrkTruthHitStrip_alljets[BDHadronTrackMonitoringAnalyzer::BWeakDecay]->Fill(TrkTruthnHitStrip);
        }
      }
      // CWeakDecay
      else if (theFlag[TrackCategories::SignalEvent] && !theFlag[TrackCategories::BWeakDecay] &&
               theFlag[TrackCategories::CWeakDecay]) {
        nseltracksCat[BDHadronTrackMonitoringAnalyzer::CWeakDecay] += 1;
        TrkPt_alljets[BDHadronTrackMonitoringAnalyzer::CWeakDecay]->Fill(TrkPt);
        TrkEta_alljets[BDHadronTrackMonitoringAnalyzer::CWeakDecay]->Fill(TrkEta);
        TrkPhi_alljets[BDHadronTrackMonitoringAnalyzer::CWeakDecay]->Fill(TrkPhi);
        TrkDxy_alljets[BDHadronTrackMonitoringAnalyzer::CWeakDecay]->Fill(TrkDxy);
        TrkDz_alljets[BDHadronTrackMonitoringAnalyzer::CWeakDecay]->Fill(TrkDz);
        TrkHitAll_alljets[BDHadronTrackMonitoringAnalyzer::CWeakDecay]->Fill(TrknHitAll);
        TrkHitPixel_alljets[BDHadronTrackMonitoringAnalyzer::CWeakDecay]->Fill(TrknHitPixel);
        TrkHitStrip_alljets[BDHadronTrackMonitoringAnalyzer::CWeakDecay]->Fill(TrknHitStrip);
        if (quality_tpr != 0) {
          TrkTruthPt_alljets[BDHadronTrackMonitoringAnalyzer::CWeakDecay]->Fill(TrkTruthPt);
          TrkTruthEta_alljets[BDHadronTrackMonitoringAnalyzer::CWeakDecay]->Fill(TrkTruthEta);
          TrkTruthPhi_alljets[BDHadronTrackMonitoringAnalyzer::CWeakDecay]->Fill(TrkTruthPhi);
          TrkTruthDxy_alljets[BDHadronTrackMonitoringAnalyzer::CWeakDecay]->Fill(TrkTruthDxy);
          TrkTruthDz_alljets[BDHadronTrackMonitoringAnalyzer::CWeakDecay]->Fill(TrkTruthDz);
          TrkTruthHitAll_alljets[BDHadronTrackMonitoringAnalyzer::CWeakDecay]->Fill(TrkTruthnHitAll);
          TrkTruthHitPixel_alljets[BDHadronTrackMonitoringAnalyzer::CWeakDecay]->Fill(TrkTruthnHitPixel);
          TrkTruthHitStrip_alljets[BDHadronTrackMonitoringAnalyzer::CWeakDecay]->Fill(TrkTruthnHitStrip);
        }
      }
      // PU
      else if (!theFlag[TrackCategories::SignalEvent] && !theFlag[TrackCategories::Fake]) {
        nseltracksCat[BDHadronTrackMonitoringAnalyzer::PU] += 1;
        TrkPt_alljets[BDHadronTrackMonitoringAnalyzer::PU]->Fill(TrkPt);
        TrkEta_alljets[BDHadronTrackMonitoringAnalyzer::PU]->Fill(TrkEta);
        TrkPhi_alljets[BDHadronTrackMonitoringAnalyzer::PU]->Fill(TrkPhi);
        TrkDxy_alljets[BDHadronTrackMonitoringAnalyzer::PU]->Fill(TrkDxy);
        TrkDz_alljets[BDHadronTrackMonitoringAnalyzer::PU]->Fill(TrkDz);
        TrkHitAll_alljets[BDHadronTrackMonitoringAnalyzer::PU]->Fill(TrknHitAll);
        TrkHitPixel_alljets[BDHadronTrackMonitoringAnalyzer::PU]->Fill(TrknHitPixel);
        TrkHitStrip_alljets[BDHadronTrackMonitoringAnalyzer::PU]->Fill(TrknHitStrip);
        if (quality_tpr != 0) {
          TrkTruthPt_alljets[BDHadronTrackMonitoringAnalyzer::PU]->Fill(TrkTruthPt);
          TrkTruthEta_alljets[BDHadronTrackMonitoringAnalyzer::PU]->Fill(TrkTruthEta);
          TrkTruthPhi_alljets[BDHadronTrackMonitoringAnalyzer::PU]->Fill(TrkTruthPhi);
          TrkTruthDxy_alljets[BDHadronTrackMonitoringAnalyzer::PU]->Fill(TrkTruthDxy);
          TrkTruthDz_alljets[BDHadronTrackMonitoringAnalyzer::PU]->Fill(TrkTruthDz);
          TrkTruthHitAll_alljets[BDHadronTrackMonitoringAnalyzer::PU]->Fill(TrkTruthnHitAll);
          TrkTruthHitPixel_alljets[BDHadronTrackMonitoringAnalyzer::PU]->Fill(TrkTruthnHitPixel);
          TrkTruthHitStrip_alljets[BDHadronTrackMonitoringAnalyzer::PU]->Fill(TrkTruthnHitStrip);
        }
      }
      // Other
      else if (theFlag[TrackCategories::SignalEvent] && !theFlag[TrackCategories::BWeakDecay] &&
               !theFlag[TrackCategories::CWeakDecay]) {
        nseltracksCat[BDHadronTrackMonitoringAnalyzer::Other] += 1;
        TrkPt_alljets[BDHadronTrackMonitoringAnalyzer::Other]->Fill(TrkPt);
        TrkEta_alljets[BDHadronTrackMonitoringAnalyzer::Other]->Fill(TrkEta);
        TrkPhi_alljets[BDHadronTrackMonitoringAnalyzer::Other]->Fill(TrkPhi);
        TrkDxy_alljets[BDHadronTrackMonitoringAnalyzer::Other]->Fill(TrkDxy);
        TrkDz_alljets[BDHadronTrackMonitoringAnalyzer::Other]->Fill(TrkDz);
        TrkHitAll_alljets[BDHadronTrackMonitoringAnalyzer::Other]->Fill(TrknHitAll);
        TrkHitPixel_alljets[BDHadronTrackMonitoringAnalyzer::Other]->Fill(TrknHitPixel);
        TrkHitStrip_alljets[BDHadronTrackMonitoringAnalyzer::Other]->Fill(TrknHitStrip);
        if (quality_tpr != 0) {
          TrkTruthPt_alljets[BDHadronTrackMonitoringAnalyzer::Other]->Fill(TrkTruthPt);
          TrkTruthEta_alljets[BDHadronTrackMonitoringAnalyzer::Other]->Fill(TrkTruthEta);
          TrkTruthPhi_alljets[BDHadronTrackMonitoringAnalyzer::Other]->Fill(TrkTruthPhi);
          TrkTruthDxy_alljets[BDHadronTrackMonitoringAnalyzer::Other]->Fill(TrkTruthDxy);
          TrkTruthDz_alljets[BDHadronTrackMonitoringAnalyzer::Other]->Fill(TrkTruthDz);
          TrkTruthHitAll_alljets[BDHadronTrackMonitoringAnalyzer::Other]->Fill(TrkTruthnHitAll);
          TrkTruthHitPixel_alljets[BDHadronTrackMonitoringAnalyzer::Other]->Fill(TrkTruthnHitPixel);
          TrkTruthHitStrip_alljets[BDHadronTrackMonitoringAnalyzer::Other]->Fill(TrkTruthnHitStrip);
        }
      }
      // Fake
      else if (!theFlag[TrackCategories::SignalEvent] && theFlag[TrackCategories::Fake]) {
        nseltracksCat[BDHadronTrackMonitoringAnalyzer::Fake] += 1;
        TrkPt_alljets[BDHadronTrackMonitoringAnalyzer::Fake]->Fill(TrkPt);
        TrkEta_alljets[BDHadronTrackMonitoringAnalyzer::Fake]->Fill(TrkEta);
        TrkPhi_alljets[BDHadronTrackMonitoringAnalyzer::Fake]->Fill(TrkPhi);
        TrkDxy_alljets[BDHadronTrackMonitoringAnalyzer::Fake]->Fill(TrkDxy);
        TrkDz_alljets[BDHadronTrackMonitoringAnalyzer::Fake]->Fill(TrkDz);
        TrkHitAll_alljets[BDHadronTrackMonitoringAnalyzer::Fake]->Fill(TrknHitAll);
        TrkHitPixel_alljets[BDHadronTrackMonitoringAnalyzer::Fake]->Fill(TrknHitPixel);
        TrkHitStrip_alljets[BDHadronTrackMonitoringAnalyzer::Fake]->Fill(TrknHitStrip);
        // NO TRUTH FOR FAKES!!!
      }
    }
    // -------- END Loop Over (selected) Tracks ----------
    // Still have to fill some jet-flavour specific variables
    if (flav == 5) {
      nTrkAll_bjet->Fill(nseltracks);
      for (unsigned int i = 0; i < TrkHistCat.size(); i++) {
        nTrk_bjet[i]->Fill(nseltracksCat[i]);
      }
    } else if (flav == 4) {
      nTrkAll_cjet->Fill(nseltracks);
      for (unsigned int i = 0; i < TrkHistCat.size(); i++) {
        nTrk_cjet[i]->Fill(nseltracksCat[i]);
      }
    } else {
      nTrkAll_dusgjet->Fill(nseltracks);
      for (unsigned int i = 0; i < TrkHistCat.size(); i++) {
        nTrk_dusgjet[i]->Fill(nseltracksCat[i]);
      }
    }
  }
  // -------- END Loop Over Jets ----------

  if (!pvFound) {
    delete pv;
  }
}

// define this as a plug-in
DEFINE_FWK_MODULE(BDHadronTrackMonitoringAnalyzer);
