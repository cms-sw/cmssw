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

#include "PhysicsTools/UtilAlgos/interface/TFileService.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include <TH1F.h>
#include <TH2F.h>

#include <memory>
#include <algorithm>
#include <cmath>

using namespace std;
using namespace edm;
using namespace reco;

typedef TrajectoryStateOnSurface TSOS;
typedef TrackingParticleCollection TPColl;
typedef TPColl::const_iterator TPCIter;
typedef MuonCollection MuColl;
typedef MuColl::const_iterator MuCIter;

class MuonHisto
{
public:
  typedef map<string, TH1*> HistMap;

  MuonHisto():isParamsSet_(false), isBooked_(false) {};

  MuonHisto(const ParameterSet& pset)
  {
    theDQMService   = 0;
    theTFileService = 0;

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
    if ( theDQMService || theTFileService ) return;

    theDQMService = dqm;
    dqm->setCurrentFolder(subDir_.c_str());
    bookHistograms();
  };

  void bookHistograms(TFileService* fs)
  {
    if ( ! fs ) return;
    if ( theDQMService || theTFileService ) return;

    theTFileService = fs;
    theTFileDirectory = new TFileDirectory(fs->mkdir(subDir_));
    bookHistograms();
  };

  ~MuonHisto() {};

  // run this for unmatched sim particles
  void operator()(const TPCIter& iSimPtcl)
  {
    if ( ! isValid() ) return;

    const double simPt  = iSimPtcl->pt() ;
    if ( isnan(simPt)  || isinf(simPt)  ) return;

    const double simEta = doAbsEta_ ? fabs(iSimPtcl->eta()) : iSimPtcl->eta();
    if ( isnan(simEta) || isinf(simEta) ) return;

    const double simPhi = iSimPtcl->phi();
    if ( isnan(simPhi) || isinf(simPhi) ) return;

    if ( simPt < minPt_ ) return;
    if ( simEta < minEta_ || simEta > maxEta_ ) return;
    if ( simPhi < minPhi_ || simPhi > maxPhi_ ) return;

    TH1F * h1;

    if ( h1 = (TH1F*)(theHistMap["SimPt" ]) ) h1->Fill(simPt );
    if ( h1 = (TH1F*)(theHistMap["SimEta"]) ) h1->Fill(simEta);
    if ( h1 = (TH1F*)(theHistMap["SimPhi"]) ) h1->Fill(simPhi);
  };

  // run this for matched sim-reco pairs
  void operator()(const std::pair<TPCIter, MuCIter>& matchedPair)
  {
    if ( ! isValid() ) return;

    TPCIter iSimPtcl  = matchedPair.first;
    MuCIter iRecoMuon = matchedPair.second;

    // all variables to be compared
    const double simPt  = iSimPtcl->pt() ;
    if ( isnan(simPt)  || isinf(simPt)  ) return;

    const double simEta = doAbsEta_ ? fabs(iSimPtcl->eta()) : iSimPtcl->eta();
    if ( isnan(simEta) || isinf(simEta) ) return;

    const double simPhi = iSimPtcl->phi();
    if ( isnan(simPhi) || isinf(simPhi) ) return;

    const TrackCharge simQ = iSimPtcl->charge();
    const double simQOverPt = simQ/simPt;
    if ( isnan(simQOverPt) || isinf(simQOverPt) ) return;

    if ( simPt < minPt_ ) return;
    if ( simEta < minEta_ || simEta > maxEta_ ) return;
    if ( simPhi < minPhi_ || simPhi > maxPhi_ ) return;

    this->operator()(iSimPtcl);

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

    TH1F * h1;
    TH2F * h2;

    if ( h1 = (TH1F*)(theHistMap["RecoPt" ]) ) h1->Fill(simPt );
    if ( h1 = (TH1F*)(theHistMap["RecoEta"]) ) h1->Fill(simEta);
    if ( h1 = (TH1F*)(theHistMap["RecoPhi"]) ) h1->Fill(simPhi);

    if ( h2 = (TH2F*)(theHistMap["ErrPtVsEta"]) ) h2->Fill(simEta, (recoPt-simPt)/simPt);
    if ( h2 = (TH2F*)(theHistMap["ErrQOverPtVsEta"]) ) h2->Fill(simEta, (recoQOverPt-simQOverPt)/simQOverPt);
  };

protected:
  void bookHistograms()
  {
    if ( ! isParamsSet() ) return;
    if ( isBooked() ) return;

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

  void book1D(string name, string title, int nBin, double min, double max)
  {
    TH1F * histo = 0;
    if ( theDQMService ) {
      histo = (TH1F*)(theDQMService->book1D(name, title, nBin, min, max));
    }
    else if ( theTFileService ) {
      histo = theTFileDirectory->make<TH1F>(name.c_str(), title.c_str(), nBin, min, max);
    }
    theHistMap.insert(HistMap::value_type(name, histo));
  };

  void book2D(string name, string title,
              int nBinX, double minX, double maxX,
              int nBinY, double minY, double maxY)
  {
    TH2F * histo = 0;
    if ( theDQMService ) {
      histo = (TH2F*)(theDQMService->book2D(name, title, nBinX, minX, maxX, nBinY, minY, maxY));
    }
    else if ( theTFileService ) {
      histo = theTFileDirectory->make<TH2F>(name.c_str(), title.c_str(), nBinX, minX, maxX, nBinY, minY, maxY);
    }
    theHistMap.insert(HistMap::value_type(name,histo));
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
  TFileService * theTFileService;
  TFileDirectory * theTFileDirectory;

  HistMap theHistMap;
};

RecoMuonValidator::RecoMuonValidator(const ParameterSet& pset)
{
  histoManager_ = pset.getUntrackedParameter<string>("histoManager", "DQM");
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
  theTFileService = 0;

  if ( histoManager_ == "DQM" ) {
    theDQMService = Service<DQMStore>().operator->();
  }
  else if ( histoManager_ == "TFileService" ) {
    theTFileService = Service<TFileService>().operator->();
  }

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
  if ( ! theDQMService && ! theTFileService  ) return;

  if ( theDQMService ) {
    LogDebug("RecoMuonValidator::beginJob") << "Booking DQM histograms" << endl;

    fillHisto_["StaMuonDeltaR"].bookHistograms(theDQMService);
    fillHisto_["GlbMuonDeltaR"].bookHistograms(theDQMService);
  }
  else if ( theTFileService ) {
    LogDebug("RecoMuonValidator::beginJob") << "Booking TFileService histograms" << endl;

    fillHisto_["StaMuonDeltaR"].bookHistograms(theTFileService);
    fillHisto_["GlbMuonDeltaR"].bookHistograms(theTFileService);
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
  if ( ! theDQMService  && ! theTFileService ) return;

  theMuonService->update(eventSetup);

  // Retrieve sim particles, reco muons, etc.
  Handle<TPColl> simPtcls;
  Handle<MuColl> recoMuons;

  // Retrieve sim, reco objects from event handler
  event.getByLabel(simPtclLabel_, simPtcls);
  event.getByLabel(recoMuonLabel_, recoMuons);

  // Create Sim to Reco Table
  LogDebug("RecoMuonValidator::analyze") << "Building Sim to Reco matching table" << endl; 

  SimRecoTable<StaMuonDeltaR> staMuonDeltaRTable(simPtcls, recoMuons, StaMuonDeltaR(1.));
  SimRecoTable<GlbMuonDeltaR> glbMuonDeltaRTable(simPtcls, recoMuons, GlbMuonDeltaR(1.));

  // Get best matched pairs
  LogDebug("RecoMuonValidator::analyze") << "Get matching pairs from the table" << endl; 

  SimRecoTable<StaMuonDeltaR>::Pairs staMuonPairsByDeltaR;
  staMuonDeltaRTable.getBestMatched(staMuonPairsByDeltaR);

  SimRecoTable<GlbMuonDeltaR>::Pairs glbMuonPairsByDeltaR;
  glbMuonDeltaRTable.getBestMatched(glbMuonPairsByDeltaR);

  // Get un-matched pairs
  LogDebug("RecoMuonValidator::analyze") << "Get un-matched simulated muons" << endl; 

  SimRecoTable<StaMuonDeltaR>::SimPtcls unmatchStaMuonDeltaR;
  staMuonDeltaRTable.getUnmatched(unmatchStaMuonDeltaR);

  SimRecoTable<GlbMuonDeltaR>::SimPtcls unmatchGlbMuonDeltaR;
  glbMuonDeltaRTable.getUnmatched(unmatchGlbMuonDeltaR);

  // Calculate difference bet'n sim-reco objects, fill histograms.
  LogDebug("RecoMuonValidator::analyze") << "Calculate diffrences between sim-reco pairs and fill histograms" << endl; 

  for_each(unmatchStaMuonDeltaR.begin(), unmatchStaMuonDeltaR.end(), fillHisto_["StaMuonDeltaR"]);
  for_each(staMuonPairsByDeltaR.begin(), staMuonPairsByDeltaR.end(), fillHisto_["StaMuonDeltaR"]);

  for_each(unmatchGlbMuonDeltaR.begin(), unmatchGlbMuonDeltaR.end(), fillHisto_["GlbMuonDeltaR"]);
  for_each(glbMuonPairsByDeltaR.begin(), glbMuonPairsByDeltaR.end(), fillHisto_["GlbMuonDeltaR"]);

}
