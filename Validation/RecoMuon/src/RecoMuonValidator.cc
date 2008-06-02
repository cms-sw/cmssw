#include "Validation/RecoMuon/src/RecoMuonValidator.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

#include "Validation/RecoMuon/src/MuonSimRecoMatching.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <memory>
#include <algorithm>
#include <cmath>

using namespace std;
using namespace edm;
using namespace reco;

typedef TrajectoryStateOnSurface TSOS;
typedef TrackingParticleCollection TPColl;
typedef TrackingParticleRef TPRef;
typedef MuonCollection MuColl;
typedef reco::MuonRef MuRef;

class MuonHisto
{
public:
  typedef map<string, MonitorElement*> HistMap;

  MuonHisto():isParamsSet_(false), isBooked_(false) {};

  MuonHisto(const ParameterSet& pset)
  {
    theDQMService   = 0;

    isParamsSet_ = false;
    isBooked_ = false;

    subDir_ = pset.getUntrackedParameter<string>("subDir");

    nBinPt_  = pset.getUntrackedParameter<unsigned int>("nBinPt" );
    nBinEta_ = pset.getUntrackedParameter<unsigned int>("nBinEta");
    nBinPhi_ = pset.getUntrackedParameter<unsigned int>("nBinPhi");

    nBinPull_ = pset.getUntrackedParameter<unsigned int>("nBinPull");
    nBinErrPt_ = pset.getUntrackedParameter<unsigned int>("nBinErrPt");
    nBinErrQOverPt_ = pset.getUntrackedParameter<unsigned int>("nBinErrQOverPt");
    doAbsEta_ = pset.getUntrackedParameter<bool>("doAbsEta");

    minPt_  = pset.getUntrackedParameter<double>("minPt" );
    maxPt_  = pset.getUntrackedParameter<double>("maxPt" );
    minEta_ = pset.getUntrackedParameter<double>("minEta");
    maxEta_ = pset.getUntrackedParameter<double>("maxEta");
    minPhi_ = pset.getUntrackedParameter<double>("minPhi", -TMath::Pi());
    maxPhi_ = pset.getUntrackedParameter<double>("maxPhi",  TMath::Pi());

    wPull_ = pset.getUntrackedParameter<double>("wPull");
    wErrPt_ = pset.getUntrackedParameter<double>("wErrPt");
    wErrQOverPt_ = pset.getUntrackedParameter<double>("wErrQOverPt");

    isParamsSet_ = true;
  };

  bool isValid() { return isParamsSet() && isBooked(); };
  bool isParamsSet() { return isParamsSet_; };
  bool isBooked() { return isBooked_; };

  void bookHistograms(DQMStore * dqm)
  {
    if ( ! dqm ) return;
    if ( theDQMService ) return;
    if ( ! isParamsSet() ) return;
    if ( isBooked() ) return;

    theDQMService = dqm;
    dqm->setCurrentFolder(subDir_.c_str());

    book1D("SimPt", "Sim p_{T}", nBinPt_, minPt_, maxPt_);
    book1D("RecoPt" , "Reco p_{T}", nBinPt_ , minPt_ , maxPt_ );
    book1D("SimEta" , "Sim #eta"  , nBinEta_, minEta_, maxEta_);
    book1D("RecoEta", "Reco #eta" , nBinEta_, minEta_, maxEta_);
    book1D("SimPhi" , "Sim #phi"  , nBinPhi_, minPhi_, maxPhi_);
    book1D("RecoPhi", "Reco #phi" , nBinPhi_, minPhi_, maxPhi_);
    book2D("ErrPtVsEta", "#Delta{}p_{T}/p_{T} vs #eta",
           nBinEta_, minEta_, maxEta_, nBinErrPt_, -wErrPt_, wErrPt_);
    book2D("ErrQOverPtVsEta", "(#Delta{}q/p_{T})/(q/p_{T}) vs #eta",
           nBinEta_, minEta_, maxEta_, nBinErrQOverPt_, -wErrQOverPt_, wErrQOverPt_);

    // p_T pulls
    book2D("PullPtVsPt" , "Pull p_{T} vs p_{T}",
           nBinPt_ , minPt_ , maxPt_ , nBinPull_, -wPull_, wPull_);
    book2D("PullPtVsEta", "Pull p_{T} vs #eta",
           nBinEta_, minEta_, maxEta_, nBinPull_, -wPull_, wPull_);
    book2D("PullPtVsPhi", "Pull p_{T} vs #phi",
           nBinPhi_, minPhi_, maxPhi_, nBinPull_, -wPull_, wPull_);

    // eta pulls
    book2D("PullEtaVsPt" , "Pull #eta vs p_{T}",
           nBinPt_ , minPt_ , maxPt_ , nBinPull_, -wPull_, wPull_);
    book2D("PullEtaVsEta", "Pull #eta vs #eta",
           nBinEta_, minEta_, maxEta_, nBinPull_, -wPull_, wPull_);
    book2D("PullEtaVsPhi", "Pull #eta vs #phi",
           nBinPhi_, minPhi_, maxPhi_, nBinPull_, -wPull_, wPull_);

    // phi pulls
    book2D("PullPhiVsPt" , "Pull #phi vs #pt",
           nBinPt_ , minPt_ , maxPt_ , nBinPull_, -wPull_, wPull_);
    book2D("PullPhiVsEta", "Pull #phi vs #eta",
           nBinEta_, minEta_, maxEta_, nBinPull_, -wPull_, wPull_);
    book2D("PullPhiVsPhi", "Pull #phi vs #phi",
           nBinPhi_, minPhi_, maxPhi_, nBinPull_, -wPull_, wPull_);

    isBooked_ = true;

  };

  ~MuonHisto() {};

  // run this for unmatched sim particles
  void operator()(const TPRef simPtcl)
  {
    if ( ! isValid() ) return;

    const double simPt  = simPtcl->pt() ;
//    if ( isnan(simPt)  || isinf(simPt)  ) return;

    const double simEta = doAbsEta_ ? fabs(simPtcl->eta()) : simPtcl->eta();
//    if ( isnan(simEta) || isinf(simEta) ) return;

    const double simPhi = simPtcl->phi();
//    if ( isnan(simPhi) || isinf(simPhi) ) return;

//    if ( simPt < minPt_ ) return;
//    if ( simEta < minEta_ || simEta > maxEta_ ) return;
//    if ( simPhi < minPhi_ || simPhi > maxPhi_ ) return;

    MonitorElement* me;

    if ( me = theHistMap["SimPt" ] ) me->Fill(simPt );
    if ( me = theHistMap["SimEta"] ) me->Fill(simEta);
    if ( me = theHistMap["SimPhi"] ) me->Fill(simPhi);
  };

  // run this for matched sim-reco pairs
  void operator()(const std::pair<TPRef, MuRef>& matchedPair)
  {
    if ( ! isValid() ) return;

    TPRef simPtcl  = matchedPair.first;
    MuRef iRecoMuon = matchedPair.second;

    // all variables to be compared
    const double simPt  = simPtcl->pt() ;
//    if ( isnan(simPt)  || isinf(simPt)  ) return;

    const double simEta = doAbsEta_ ? fabs(simPtcl->eta()) : simPtcl->eta();
//    if ( isnan(simEta) || isinf(simEta) ) return;

    const double simPhi = simPtcl->phi();
//    if ( isnan(simPhi) || isinf(simPhi) ) return;

    const TrackCharge simQ = simPtcl->charge();
    const double simQOverPt = simQ/simPt;
//    if ( isnan(simQOverPt) || isinf(simQOverPt) ) return;

//    if ( simPt < minPt_ ) return;
//    if ( simEta < minEta_ || simEta > maxEta_ ) return;
//    if ( simPhi < minPhi_ || simPhi > maxPhi_ ) return;

    this->operator()(simPtcl);

    const double recoPt  = iRecoMuon->pt() ;
    if ( isnan(recoPt)  || isinf(recoPt)  ) return;
    if ( recoPt <= 0.0 ) return;

    const double recoEta = iRecoMuon->eta();
    if ( isnan(recoEta) || isinf(recoEta) ) return;

    const double recoPhi = iRecoMuon->phi();
    if ( isnan(recoPhi) || isinf(recoPhi) ) return;

    const TrackCharge recoQ = iRecoMuon->charge();
    const double recoQOverPt = recoQ/recoPt;
    if ( isnan(recoQOverPt) || isinf(recoQOverPt) ) return;

    MonitorElement* me;

    if ( me = theHistMap["RecoPt" ] ) me->Fill(simPt );
    if ( me = theHistMap["RecoEta"] ) me->Fill(simEta);
    if ( me = theHistMap["RecoPhi"] ) me->Fill(simPhi);

    if ( me = theHistMap["ErrPtVsEta"] ) me->Fill(simEta, (recoPt-simPt)/simPt);
    if ( me = theHistMap["ErrQOverPtVsEta"] ) me->Fill(simEta, (recoQOverPt-simQOverPt)/simQOverPt);
  };

protected:
  void book1D(string name, string title, int nBin, double min, double max)
  {
    MonitorElement* me = 0;
    if ( theDQMService ) {
      me = theDQMService->book1D(name, title, nBin, min, max);
    }
    theHistMap.insert(HistMap::value_type(name, me));
  };

  void book2D(string name, string title,
              int nBinX, double minX, double maxX,
              int nBinY, double minY, double maxY)
  {
    MonitorElement* me = 0;
    if ( theDQMService ) {
      me = theDQMService->book2D(name, title, nBinX, minX, maxX, nBinY, minY, maxY);
    }
    theHistMap.insert(HistMap::value_type(name,me));
  };

protected:
  bool isParamsSet_, isBooked_;

  string subDir_;
  unsigned int nBinPt_, nBinEta_, nBinPhi_;
  unsigned int nBinPull_, nBinErrPt_, nBinErrQOverPt_;
  double minPt_, maxPt_, minEta_, maxEta_, minPhi_, maxPhi_;
  double wPull_, wErrPt_, wErrQOverPt_;

  bool doAbsEta_;

  DQMStore * theDQMService;

  HistMap theHistMap;
};

RecoMuonValidator::RecoMuonValidator(const ParameterSet& pset)
{
  outputFileName_ = pset.getUntrackedParameter<string>("outputFileName");

  // Track Labels
  simPtclLabel_ = pset.getParameter<InputTag>("SimPtcl");
  recoMuonLabel_ = pset.getParameter<InputTag>("RecoMuon");
//  staMuonLabel_ = pset.getParameter<InputTag>("StaMuon");
//  glbMuonLabel_ = pset.getParameter<InputTag>("GlbMuon");

  // the service parameters
  ParameterSet serviceParameters
    = pset.getParameter<ParameterSet>("ServiceParameters");
  theMuonService = new MuonServiceProxy(serviceParameters);

  // Set histogram manager
  theDQMService   = 0;
  theDQMService = Service<DQMStore>().operator->();

  MuonHisto staMuonDeltaRHisto(pset.getParameter<ParameterSet>("StaMuonHistoParameters"));
  fillHisto_.insert(map<string, MuonHisto>::value_type("StaMuonDeltaR", staMuonDeltaRHisto));

  MuonHisto glbMuonDeltaRHisto(pset.getParameter<ParameterSet>("GlbMuonHistoParameters"));
  fillHisto_.insert(map<string, MuonHisto>::value_type("GlbMuonDeltaR", glbMuonDeltaRHisto));

}

RecoMuonValidator::~RecoMuonValidator()
{
  if ( theMuonService ) delete theMuonService;
}
void RecoMuonValidator::beginJob(const EventSetup& eventSetup)
{
  if ( ! theMuonService ) return;
  if ( ! theDQMService  ) return;

  if ( theDQMService ) {
    LogDebug("RecoMuonValidator::beginJob") << "Booking DQM histograms" << endl;

    fillHisto_["StaMuonDeltaR"].bookHistograms(theDQMService);
    fillHisto_["GlbMuonDeltaR"].bookHistograms(theDQMService);
  }
}

void RecoMuonValidator::endJob()
{
  if ( theDQMService && ! outputFileName_.empty() ) { 
    LogDebug("RecoMuonValidator::endJob") << "Saving DQM histograms to file " << outputFileName_ << endl;
    theDQMService->save(outputFileName_);
  }
}

class StaMuonDeltaR : public MuonDeltaR
{
 public:
  StaMuonDeltaR(const double maxDeltaR):
    MuonDeltaR::MuonDeltaR(maxDeltaR) {};

  bool operator()(const Muon& recoMuon) const
  {
    return recoMuon.isStandAloneMuon();
  };

  bool operator()(const TrackingParticle& simPtcl) const
  {
    return MuonDeltaR::operator()(simPtcl);
  };

  bool operator()(const TrackingParticle& simPtcl,
                  const Muon& recoMuon,
                  double& quality) const
  {
    return MuonDeltaR::operator()(simPtcl, recoMuon, quality);
  };
};

class GlbMuonDeltaR : public MuonDeltaR
{
 public:
  GlbMuonDeltaR(const double maxDeltaR):
    MuonDeltaR::MuonDeltaR(maxDeltaR) {};

  bool operator()(const Muon& recoMuon) const
  {
    return recoMuon.isGlobalMuon();
  };

  bool operator()(const TrackingParticle& simPtcl) const
  {
    return MuonDeltaR::operator()(simPtcl);
  };

  bool operator()(const TrackingParticle& simPtcl,
                  const Muon& recoMuon,
                  double& quality) const
  {
    return MuonDeltaR::operator()(simPtcl, recoMuon, quality);
  };
};

void RecoMuonValidator::analyze(const Event& event, const EventSetup& eventSetup)
{
  if ( ! theMuonService ) return;
  if ( ! theDQMService  ) return;

  theMuonService->update(eventSetup);

  // Retrieve sim particles, reco muons, etc.
  Handle<TPColl> staSimPtcls;
  Handle<TPColl> glbSimPtcls;
  Handle<MuColl> staRecoMuons;
  Handle<MuColl> glbRecoMuons;

  // Retrieve sim, reco objects from event handler
  event.getByLabel(simPtclLabel_, staSimPtcls);
  event.getByLabel(simPtclLabel_, glbSimPtcls);
  event.getByLabel(recoMuonLabel_, staRecoMuons);
  event.getByLabel(recoMuonLabel_, glbRecoMuons);

  // Create Sim to Reco Table
  LogDebug("RecoMuonValidator::analyze") << "Building Sim to Reco matching table" << endl; 

  typedef SimRecoTable<StaMuonDeltaR> StaDRTab;
  typedef SimRecoTable<GlbMuonDeltaR> GlbDRTab; 

  StaDRTab staDRTab(staSimPtcls, staRecoMuons, StaMuonDeltaR(1.));
  GlbDRTab glbDRTab(glbSimPtcls, glbRecoMuons, GlbMuonDeltaR(1.));

  // Get best matched pairs
  LogDebug("RecoMuonValidator::analyze") << "Get matching pairs from the table" << endl; 

  StaDRTab::Pairs staPairsDR;
  staDRTab.getBestMatched(staPairsDR);

  GlbDRTab::Pairs glbPairsDR;
  glbDRTab.getBestMatched(glbPairsDR);

  // Get un-matched pairs
  LogDebug("RecoMuonValidator::analyze") << "Get un-matched simulated muons" << endl; 

  StaDRTab::SimPtcls lostStaDR;
  staDRTab.getUnmatched(lostStaDR);

  GlbDRTab::SimPtcls lostGlbDR;
  glbDRTab.getUnmatched(lostGlbDR);

  // Calculate difference bet'n sim-reco objects, fill histograms.
  LogDebug("RecoMuonValidator::analyze") << "Calculate diffrences between sim-reco pairs and fill histograms" << endl; 

  for_each(lostStaDR.begin(), lostStaDR.end(), fillHisto_["StaMuonDeltaR"]);
  for_each(staPairsDR.begin(), staPairsDR.end(), fillHisto_["StaMuonDeltaR"]);

  for_each(lostGlbDR.begin(), lostGlbDR.end(), fillHisto_["GlbMuonDeltaR"]);
  for_each(glbPairsDR.begin(), glbPairsDR.end(), fillHisto_["GlbMuonDeltaR"]);

}
