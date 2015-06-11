#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"

class PackedCandidateTrackValidator: public DQMEDAnalyzer{
 public:
  PackedCandidateTrackValidator(const edm::ParameterSet& pset);
  virtual ~PackedCandidateTrackValidator();

  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event&, const edm::EventSetup& ) override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 private:

  edm::EDGetTokenT<edm::View<reco::Track>> tracksToken_;
  edm::EDGetTokenT<edm::Association<pat::PackedCandidateCollection>> trackToPackedCandidateToken_;

  std::string rootFolder_;

  MonitorElement *h_selectionFlow;

  MonitorElement *h_diffPx;
  MonitorElement *h_diffPy;
  MonitorElement *h_diffPz;

  MonitorElement *h_diffVx;
  MonitorElement *h_diffVy;
  MonitorElement *h_diffVz;

  MonitorElement *h_diffNormalizedChi2;
  MonitorElement *h_diffNdof;

  MonitorElement *h_diffCharge;
  MonitorElement *h_diffIsHighPurity;

  MonitorElement *h_diffQoverp;
  MonitorElement *h_diffPt;
  MonitorElement *h_diffEta;
  MonitorElement *h_diffTheta;
  MonitorElement *h_diffPhi;
  MonitorElement *h_diffDxy;
  MonitorElement *h_diffDz;

  MonitorElement *h_diffQoverpError;
  MonitorElement *h_diffPtError;
  MonitorElement *h_diffEtaError;
  MonitorElement *h_diffThetaError;
  MonitorElement *h_diffPhiError;
  MonitorElement *h_diffDxyError;
  MonitorElement *h_diffDzError;

  MonitorElement *h_diffNumberOfPixelHits;
  MonitorElement *h_diffNumberOfHits;
  MonitorElement *h_diffLostInnerHits;

  MonitorElement *h_diffHitPatternNumberOfValidPixelHits;
  MonitorElement *h_diffHitPatternNumberOfValidHits;
  MonitorElement *h_diffHitPatternNumberOfLostInnerHits;
  MonitorElement *h_diffHitPatternHasValidHitInFirstPixelBarrel;
};

PackedCandidateTrackValidator::PackedCandidateTrackValidator(const edm::ParameterSet& iConfig):
  tracksToken_(consumes<edm::View<reco::Track>>(iConfig.getUntrackedParameter<edm::InputTag>("tracks"))),
  trackToPackedCandidateToken_(consumes<edm::Association<pat::PackedCandidateCollection>>(iConfig.getUntrackedParameter<edm::InputTag>("trackToPackedCandiadteAssociation"))),
  rootFolder_(iConfig.getUntrackedParameter<std::string>("rootFolder"))
{}

PackedCandidateTrackValidator::~PackedCandidateTrackValidator() {}

void PackedCandidateTrackValidator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.addUntracked<edm::InputTag>("tracks", edm::InputTag("generalTracks"));
  desc.addUntracked<edm::InputTag>("trackToPackedCandiadteAssociation", edm::InputTag("packedPFCandidates"));
  desc.addUntracked<std::string>("rootFolder", "Tracking/PackedCandidate");

  descriptions.add("packedCandidateTrackValidator", desc);
}

void PackedCandidateTrackValidator::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const&, edm::EventSetup const&) {
  iBooker.setCurrentFolder(rootFolder_);

  h_selectionFlow = iBooker.book1D("selectionFlow", "Track selection flow", 5, 0, 5);
  h_selectionFlow->setBinLabel(1, "All tracks");
  h_selectionFlow->setBinLabel(2, "Associated to PackedCandidate");
  h_selectionFlow->setBinLabel(3, "PackedCandidate has track");
  h_selectionFlow->setBinLabel(4, "PackedCandidate is not electron");
  h_selectionFlow->setBinLabel(5, "PackedCandidate has hits");

  constexpr int diffBins = 50;
  constexpr float diff = 1e-4;
  constexpr float diffP = 5e-3;

  h_diffPx = iBooker.book1D("diffPx", "PackedCandidate::bestTrack() - reco::Track in px()", diffBins, -diffP, diffP);
  h_diffPy = iBooker.book1D("diffPy", "PackedCandidate::bestTrack() - reco::Track in py()", diffBins, -diffP, diffP);
  h_diffPz = iBooker.book1D("diffPz", "PackedCandidate::bestTrack() - reco::Track in pz()", diffBins, -diffP, diffP);

  h_diffVx = iBooker.book1D("diffVx", "PackedCandidate::bestTrack() - reco::Track in vx()", diffBins, -diffP, diffP);
  h_diffVy = iBooker.book1D("diffVy", "PackedCandidate::bestTrack() - reco::Track in vy()", diffBins, -diffP, diffP);
  h_diffVz = iBooker.book1D("diffVz", "PackedCandidate::bestTrack() - reco::Track in vz()", diffBins, -diffP, diffP);

  h_diffNormalizedChi2 = iBooker.book1D("diffNormalizedChi2", "PackedCandidate::bestTrack() - reco::Track in normalizedChi2()", 30, -1.5, 1.5);
  h_diffNdof = iBooker.book1D("diffNdof", "PackedCandidate::bestTrack() - reco::Track in ndof()", 33, -30.5, 2.5);

  h_diffCharge = iBooker.book1D("diffCharge", "PackedCandidate::bestTrack() - reco::Track in charge()", 5, -2.5, 2.5);
  h_diffIsHighPurity = iBooker.book1D("diffIsHighPurity", "PackedCandidate::bestTrack() - reco::Track in quality(highPurity)", 3, -1.5, 1.5);

  h_diffQoverp = iBooker.book1D("diffQoverp", "PackedCandidate::bestTrack() - reco::Track in qoverp()", diffBins, -1e-3, 1e-3);
  h_diffPt     = iBooker.book1D("diffPt",     "PackedCandidate::bestTrack() - reco::Track in pt()",     diffBins, -diffP, diffP);
  h_diffEta    = iBooker.book1D("diffEta",    "PackedCandidate::bestTrack() - reco::Track in eta()",    diffBins, -diff, diff);
  h_diffTheta  = iBooker.book1D("diffTheta",  "PackedCandidate::bestTrack() - reco::Track in theta()",  diffBins, -diff, diff);
  h_diffPhi    = iBooker.book1D("diffPhi",    "PackedCandidate::bestTrack() - reco::Track in phi()",    diffBins, -diff, diff);
  h_diffDxy    = iBooker.book1D("diffDxy",    "PackedCandidate::bestTrack() - reco::Track in dxy()",    diffBins, -2e-5, 2e-5);
  h_diffDz     = iBooker.book1D("diffDz",     "PackedCandidate::bestTrack() - reco::Track in dz()",     diffBins, -4e-5, 4e-5);

  h_diffQoverpError = iBooker.book1D("diffQoverpError", "PackedCandidate::bestTrack() - reco::Track in qoverpError()", diffBins, -1e-3, 1e-3);
  h_diffPtError     = iBooker.book1D("diffPtError",     "PackedCandidate::bestTrack() - reco::Track in ptError()",     diffBins, -diffP, diffP);
  h_diffEtaError    = iBooker.book1D("diffEtaError",    "PackedCandidate::bestTrack() - reco::Track in etaError()",    diffBins, -diff, diff);
  h_diffThetaError  = iBooker.book1D("diffThetaError",  "PackedCandidate::bestTrack() - reco::Track in thetaError()",  diffBins, -diff, diff);
  h_diffPhiError    = iBooker.book1D("diffPhiError",    "PackedCandidate::bestTrack() - reco::Track in phiError()",    diffBins, -diff, diff);
  h_diffDxyError    = iBooker.book1D("diffDxyError",    "PackedCandidate::bestTrack() - reco::Track in dxyError()",    diffBins, -2e-5, 2e-5);
  h_diffDzError     = iBooker.book1D("diffDzError",     "PackedCandidate::bestTrack() - reco::Track in dzError()",     diffBins, -4e-5, 4e-5);

  h_diffNumberOfPixelHits = iBooker.book1D("diffNumberOfPixelHits", "PackedCandidate::numberOfPixelHits() - reco::Track::hitPattern::numberOfValidPixelHits()", 5, -2.5, 2.5);
  h_diffNumberOfHits      = iBooker.book1D("diffNumberOfHits",      "PackedCandidate::numberHits() - reco::Track::hitPattern::numberOfValidHits()",             5, -2.5, 2.5);
  h_diffLostInnerHits     = iBooker.book1D("diffLostInnerHits",     "PackedCandidate::lostInnerHits() - reco::Track::hitPattern::numberOfLostHits(MISSING_INNER_HITS)",      5, -2.5, 2.5);

  h_diffHitPatternNumberOfValidPixelHits = iBooker.book1D("diffHitPatternNumberOfValidPixelHits", "PackedCandidate::bestTrack() - reco::Track in hitPattern::numberOfValidPixelHits()",   5, -2.5, 2.5);
  h_diffHitPatternNumberOfValidHits      = iBooker.book1D("diffHitPatternNumberOfValidHits",      "PackedCandidate::bestTrack() - reco::Track in hitPattern::numberOfValidHits()",      5, -2.5, 2.5);
  h_diffHitPatternNumberOfLostInnerHits  = iBooker.book1D("diffHitPatternNumberOfLostPixelHits",  "PackedCandidate::bestTrack() - reco::Track in hitPattern::numberOfLostHits(MISSING_INNER_HITS)", 13, -10.5, 2.5);
  h_diffHitPatternHasValidHitInFirstPixelBarrel = iBooker.book1D("diffHitPatternHasValidHitInFirstPixelBarrel", "PackedCandidate::bestTrack() - reco::Track in hitPattern::hasValidHitInFirstPixelBarrel", 3, -1.5, 1.5);

}

namespace {
  template<typename T> void fillNoFlow(MonitorElement* h, T val){
    h->Fill(std::min(std::max(val,((T) h->getTH1()->GetXaxis()->GetXmin())),((T) h->getTH1()->GetXaxis()->GetXmax())));
  }
}

void PackedCandidateTrackValidator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<edm::View<reco::Track>> htracks;
  iEvent.getByToken(tracksToken_, htracks);
  const auto& tracks = *htracks;

  edm::Handle<edm::Association<pat::PackedCandidateCollection>> hassoc;
  iEvent.getByToken(trackToPackedCandidateToken_, hassoc);
  const auto& trackToPackedCandidate = *hassoc;

  for(size_t i=0; i<tracks.size(); ++i) {
    auto trackPtr = tracks.ptrAt(i);
    const reco::Track& track = *trackPtr;
    h_selectionFlow->Fill(0.5);

    pat::PackedCandidateRef pcRef = trackToPackedCandidate[trackPtr];
    if(pcRef.isNull()) {
      continue;
    }
    h_selectionFlow->Fill(1.5);

    const reco::Track *trackPcPtr = pcRef->bestTrack();
    if(!trackPcPtr) {
      continue;
    }
    h_selectionFlow->Fill(2.5);

    // Filter out electrons to avoid comparisons to PackedCandidates with GsfTrack
    if(std::abs(pcRef->pdgId()) == 11) {
      continue;
    }
    h_selectionFlow->Fill(3.5);

    // Filter out PackedCandidate-tracks with no hits, as they won't have their details filled
    const reco::Track& trackPc = *trackPcPtr;
    if(trackPc.hitPattern().numberOfValidHits() == 0) {
      continue;
    }
    h_selectionFlow->Fill(4.5);


    fillNoFlow(h_diffPx, trackPc.px() - track.px());
    fillNoFlow(h_diffPy, trackPc.py() - track.py());
    fillNoFlow(h_diffPz, trackPc.pz() - track.pz());

    fillNoFlow(h_diffVx, trackPc.vx() - track.vx());
    fillNoFlow(h_diffVy, trackPc.vy() - track.vy());
    fillNoFlow(h_diffVz, trackPc.vz() - track.vz());

    fillNoFlow(h_diffNormalizedChi2, trackPc.normalizedChi2() - track.normalizedChi2());
    fillNoFlow(h_diffNdof, trackPc.ndof() - track.ndof());

    fillNoFlow(h_diffCharge, trackPc.charge() - track.charge());
    fillNoFlow(h_diffIsHighPurity, trackPc.quality(reco::TrackBase::highPurity) - track.quality(reco::TrackBase::highPurity));

    fillNoFlow(h_diffQoverp, trackPc.qoverp() - track.qoverp());
    fillNoFlow(h_diffPt    , trackPc.pt()     - track.pt()    );
    fillNoFlow(h_diffEta   , trackPc.eta()    - track.eta()   );
    fillNoFlow(h_diffTheta , trackPc.theta()  - track.theta() );
    fillNoFlow(h_diffPhi   , trackPc.phi()    - track.phi()   );
    fillNoFlow(h_diffDxy   , trackPc.dxy()    - track.dxy()   );
    fillNoFlow(h_diffDz    , trackPc.dz()     - track.dz()    );

    fillNoFlow(h_diffQoverpError, trackPc.qoverpError() - track.qoverpError());
    fillNoFlow(h_diffPtError    , trackPc.ptError()     - track.ptError()    );
    fillNoFlow(h_diffEtaError   , trackPc.etaError()    - track.etaError()   );
    fillNoFlow(h_diffThetaError , trackPc.thetaError()  - track.thetaError() );
    fillNoFlow(h_diffPhiError   , trackPc.phiError()    - track.phiError()   );
    fillNoFlow(h_diffDxyError   , trackPc.dxyError()    - track.dxyError()   );
    fillNoFlow(h_diffDzError    , trackPc.dzError()     - track.dzError()    );

    fillNoFlow(h_diffNumberOfPixelHits, pcRef->numberOfPixelHits() - track.hitPattern().numberOfValidPixelHits());
    fillNoFlow(h_diffNumberOfHits, pcRef->numberOfHits() - track.hitPattern().numberOfValidHits());

    int diffLostInnerHits = 0;
    const auto trackLostInnerHits = track.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS);
    switch(pcRef->lostInnerHits()) {
    case pat::PackedCandidate::validHitInFirstPixelBarrelLayer:
    case pat::PackedCandidate::noLostInnerHits:
      diffLostInnerHits = -trackLostInnerHits;
      break;
    case pat::PackedCandidate::oneLostInnerHit:
      diffLostInnerHits = 1-trackLostInnerHits;
      break;
    case pat::PackedCandidate::moreLostInnerHits:
      diffLostInnerHits = trackLostInnerHits>=2 ? 0 : 2-trackLostInnerHits;
      break;
    }
    fillNoFlow(h_diffLostInnerHits, diffLostInnerHits);

    fillNoFlow(h_diffHitPatternNumberOfValidPixelHits, trackPc.hitPattern().numberOfValidPixelHits() - track.hitPattern().numberOfValidPixelHits());
    fillNoFlow(h_diffHitPatternNumberOfValidHits, trackPc.hitPattern().numberOfValidHits() - track.hitPattern().numberOfValidHits());
    fillNoFlow(h_diffHitPatternNumberOfLostInnerHits, trackPc.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS) - track.hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS));

    fillNoFlow(h_diffHitPatternHasValidHitInFirstPixelBarrel, trackPc.hitPattern().hasValidHitInFirstPixelBarrel() - track.hitPattern().hasValidHitInFirstPixelBarrel());
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PackedCandidateTrackValidator);
