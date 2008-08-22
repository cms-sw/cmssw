#include "Validation/RecoMuon/src/RecoMuonValidator.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "SimTracker/Records/interface/TrackAssociatorRecord.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByChi2.h"
#include "SimTracker/TrackAssociation/interface/TrackAssociatorByHits.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include "TMath.h"

using namespace std;
using namespace edm;
using namespace reco;

typedef TrajectoryStateOnSurface TSOS;
typedef FreeTrajectoryState FTS;

namespace MEIdx {
  enum 
  {
    SimP, SimPt, SimEta, SimPhi,
    RecoP, RecoPt, RecoEta, RecoPhi,
    ErrP, ErrPt, ErrEta, ErrPhi, ErrDxy, ErrDz, 
    ErrP_vs_Eta, ErrPt_vs_Eta, ErrQPt_vs_Eta, 
    ErrP_vs_P, ErrPt_vs_Pt, ErrQPt_vs_Pt, ErrEta_vs_Eta,
    PullPt, PullEta, PullPhi, PullQPt, PullDxy, PullDz,
    PullPt_vs_Eta, PullPt_vs_Pt, PullEta_vs_Eta, PullPhi_vs_Eta, PullEta_vs_Pt,
    NSim, NReco,
    NDof, Chi2, Chi2Norm, Chi2Prob,
    NDof_vs_Eta, Chi2_vs_Eta, Chi2Norm_vs_Eta, Chi2Prob_vs_Eta,
    MisQPt, MisQEta,
    NAssocSimToReco, NAssocRecoToSim, 
    NSimToReco,
    SimNHits, 
    RecoNHits, 
    NRecoHits,
    NSimHits_vs_Pt, NSimHits_vs_Eta,
    NRecoHits_vs_Pt, NRecoHits_vs_Eta,
    NLostHits, NLostHits_vs_Pt, NLostHits_vs_Eta
  }; 
}

RecoMuonValidator::RecoMuonValidator(const ParameterSet& pset)
{
  outputFileName_ = pset.getUntrackedParameter<string>("outputFileName", "");

  // Set histogram dimensions
  unsigned int nBinP = pset.getUntrackedParameter<unsigned int>("nBinP");
  double minP = pset.getUntrackedParameter<double>("minP");
  double maxP = pset.getUntrackedParameter<double>("maxP");

  unsigned int nBinPt = pset.getUntrackedParameter<unsigned int>("nBinPt");
  double minPt = pset.getUntrackedParameter<double>("minPt");
  double maxPt = pset.getUntrackedParameter<double>("maxPt");

  doAbsEta_ = pset.getUntrackedParameter<bool>("doAbsEta");
  unsigned int nBinEta  = pset.getUntrackedParameter<unsigned int>("nBinEta");
  double minEta = pset.getUntrackedParameter<double>("minEta");
  double maxEta = pset.getUntrackedParameter<double>("maxEta");

  unsigned int nBinPhi  = pset.getUntrackedParameter<unsigned int>("nBinPhi");
  double minPhi = pset.getUntrackedParameter<double>("minPhi", -TMath::Pi());
  double maxPhi = pset.getUntrackedParameter<double>("maxPhi",  TMath::Pi());

  unsigned int nBinErr  = pset.getUntrackedParameter<unsigned int>("nBinErr");
  unsigned int nBinPull = pset.getUntrackedParameter<unsigned int>("nBinPull");

  double wPull = pset.getUntrackedParameter<double>("wPull");

  double minErrP = pset.getUntrackedParameter<double>("minErrP");
  double maxErrP = pset.getUntrackedParameter<double>("maxErrP");

  double minErrPt = pset.getUntrackedParameter<double>("minErrPt");
  double maxErrPt = pset.getUntrackedParameter<double>("maxErrPt");

  double minErrQPt = pset.getUntrackedParameter<double>("minErrQPt");
  double maxErrQPt = pset.getUntrackedParameter<double>("maxErrQPt");

  double minErrEta = pset.getUntrackedParameter<double>("minErrEta");
  double maxErrEta = pset.getUntrackedParameter<double>("maxErrEta");

  double minErrPhi = pset.getUntrackedParameter<double>("minErrPhi");
  double maxErrPhi = pset.getUntrackedParameter<double>("maxErrPhi");

  double minErrDxy = pset.getUntrackedParameter<double>("minErrDxy");
  double maxErrDxy = pset.getUntrackedParameter<double>("maxErrDxy");

  double minErrDz  = pset.getUntrackedParameter<double>("minErrDz" );
  double maxErrDz  = pset.getUntrackedParameter<double>("maxErrDz" );

  unsigned int nTrks = pset.getUntrackedParameter<unsigned int>("nTrks");
  unsigned int nAssoc = pset.getUntrackedParameter<unsigned int>("nAssoc");
  unsigned int nHits = pset.getUntrackedParameter<unsigned int>("nHits");
  unsigned int nDof = pset.getUntrackedParameter<unsigned int>("nDof", 55);

  // Labels for simulation and reconstruction tracks
  simLabel_  = pset.getParameter<InputTag>("simLabel" );
  recoLabel_ = pset.getParameter<InputTag>("recoLabel");

  // Labels for sim-reco association
  doAssoc_ = pset.getUntrackedParameter<bool>("doAssoc", true);
  assocLabel_ = pset.getParameter<InputTag>("assocLabel");

//  seedPropagatorName_ = pset.getParameter<string>("SeedPropagator");


  // the service parameters
  ParameterSet serviceParameters 
    = pset.getParameter<ParameterSet>("ServiceParameters");
  theMuonService = new MuonServiceProxy(serviceParameters);

  // retrieve the instance of DQMService
  theDQM = 0;
  theDQM = Service<DQMStore>().operator->();

  if ( ! theDQM ) {
    LogError("RecoMuonValidator") << "DQMService not initialized\n";
    return;
  }

  subDir_ = pset.getUntrackedParameter<string>("subDir");

  theDQM->showDirStructure();

  theDQM->cd();
  InputTag algo = recoLabel_;
  string dirName=subDir_;
  if (algo.process()!="")
    dirName+=algo.process()+"_";
  if(algo.label()!="")
    dirName+=algo.label()+"_";
  if(algo.instance()!="")
    dirName+=algo.instance()+"";
  if (dirName.find("Tracks")<dirName.length()){
    dirName.replace(dirName.find("Tracks"),6,"");
  }
  string assoc= assocLabel_.label();
  if (assoc.find("Track")<assoc.length()){
    assoc.replace(assoc.find("Track"),5,"");
  }
  dirName+=assoc;
  std::replace(dirName.begin(), dirName.end(), ':', '_');
  theDQM->setCurrentFolder(dirName.c_str());

  // book histograms
  // - 1d histograms on tracking variables
  // -- histograms for efficiency vs p, pt, eta, phi
  meMap_[MEIdx::SimP  ] = theDQM->book1D("SimP"  , "p of simTracks"    , nBinP  , minP  , maxP  );
  meMap_[MEIdx::SimPt ] = theDQM->book1D("SimPt" , "p_{T} of simTracks", nBinPt , minPt , maxPt );
  meMap_[MEIdx::SimEta] = theDQM->book1D("SimEta", "#eta of simTracks" , nBinEta, minEta, maxEta);
  meMap_[MEIdx::SimPhi] = theDQM->book1D("SimPhi", "#phi of simTracks" , nBinPhi, minPhi, maxPhi);

  meMap_[MEIdx::RecoP  ] = theDQM->book1D("RecoP"  , "p of recoTracks"    , nBinP  , minP  , maxP  );
  meMap_[MEIdx::RecoPt ] = theDQM->book1D("RecoPt" , "p_{T} of recoTracks", nBinPt , minPt , maxPt );
  meMap_[MEIdx::RecoEta] = theDQM->book1D("RecoEta", "#eta of recoTracks" , nBinEta, minEta, maxEta);
  meMap_[MEIdx::RecoPhi] = theDQM->book1D("RecoPhi", "#phi of recoTracks" , nBinPhi, minPhi, maxPhi);

  // - Resolutions
  meMap_[MEIdx::ErrP  ] = theDQM->book1D("ErrP"  , "#Delta(p)/p"        , nBinErr, minErrP  , maxErrP  );
  meMap_[MEIdx::ErrPt ] = theDQM->book1D("ErrPt" , "#Delta(p_{T})/p_{T}", nBinErr, minErrPt , maxErrPt );
  meMap_[MEIdx::ErrEta] = theDQM->book1D("ErrEta", "#sigma(#eta))"      , nBinErr, minErrEta, maxErrEta);
  meMap_[MEIdx::ErrPhi] = theDQM->book1D("ErrPhi", "#sigma(#phi)"       , nBinErr, minErrPhi, maxErrPhi);
  meMap_[MEIdx::ErrDxy] = theDQM->book1D("ErrDxy", "#sigma(d_{xy})"     , nBinErr, minErrDxy, maxErrDxy);
  meMap_[MEIdx::ErrDz ] = theDQM->book1D("ErrDz" , "#sigma(d_{z})"      , nBinErr, minErrDz , maxErrDz );

  // -- Resolutions vs Eta
  meMap_[MEIdx::ErrP_vs_Eta  ] = theDQM->book2D("ErrP_vs_Eta", "#Delta(p)/p vs #eta",
                                           nBinEta, minEta, maxEta, nBinErr, minErrP, maxErrP);
  meMap_[MEIdx::ErrPt_vs_Eta ] = theDQM->book2D("ErrPt_vs_Eta", "#Delta(p_{T})/p_{T} vs #eta",
                                           nBinEta, minEta, maxEta, nBinErr, minErrPt, maxErrPt);
  meMap_[MEIdx::ErrQPt_vs_Eta] = theDQM->book2D("ErrQPt_vs_Eta", "#Delta(q/p_{T})/(q/p_{T}) vs #eta",
                                            nBinEta, minEta, maxEta, nBinErr, minErrQPt, maxErrQPt);
  meMap_[MEIdx::ErrEta_vs_Eta] = theDQM->book2D("ErrEta_vs_Eta", "#sigma(#eta) vs #eta", 
                                           nBinEta, minEta, maxEta, nBinErr, minErrEta, maxErrEta);

  // -- Resolutions vs momentum
  meMap_[MEIdx::ErrP_vs_P   ] = theDQM->book2D("ErrP_vs_P", "#Delta(p)/p vs p",
                                          nBinP, minP, maxP, nBinErr, minErrP, maxErrP);

  meMap_[MEIdx::ErrPt_vs_Pt ] = theDQM->book2D("ErrPt_vs_Pt", "#Delta(p_{T})/p_{T} vs p_{T}",
                                           nBinPt, minPt, maxPt, nBinErr, minErrPt, maxErrPt);
  meMap_[MEIdx::ErrQPt_vs_Pt] = theDQM->book2D("ErrQPt_vs_Pt", "#Delta(q/p_{T})/(q/p_{T}) vs p_{T}",
                                           nBinPt, minPt, maxPt, nBinErr, minErrQPt, maxErrQPt);


  // - Pulls
  meMap_[MEIdx::PullPt ] = theDQM->book1D("PullPt" , "Pull(#p_{T})" , nBinPull, -wPull, wPull);
  meMap_[MEIdx::PullEta] = theDQM->book1D("PullEta", "Pull(#eta)"   , nBinPull, -wPull, wPull);
  meMap_[MEIdx::PullPhi] = theDQM->book1D("PullPhi", "Pull(#phi)"   , nBinPull, -wPull, wPull);
  meMap_[MEIdx::PullQPt] = theDQM->book1D("PullQPt", "Pull(q/p_{T})", nBinPull, -wPull, wPull);
  meMap_[MEIdx::PullDxy] = theDQM->book1D("PullDxy", "Pull(D_{xy})" , nBinPull, -wPull, wPull);
  meMap_[MEIdx::PullDz ] = theDQM->book1D("PullDz" , "Pull(D_{z})"  , nBinPull, -wPull, wPull);

  // -- Pulls vs Eta
  meMap_[MEIdx::PullPt_vs_Eta ] = theDQM->book2D("PullPt_vs_Eta", "Pull(p_{T}) vs #eta",
                                            nBinEta, minEta, maxEta, nBinPull, -wPull, wPull);
  meMap_[MEIdx::PullEta_vs_Eta] = theDQM->book2D("PullEta_vs_Eta", "Pull(#eta) vs #eta",
                                            nBinEta, minEta, maxEta, nBinPull, -wPull, wPull);
  meMap_[MEIdx::PullPhi_vs_Eta] = theDQM->book2D("PullPhi_vs_Eta", "Pull(#phi) vs #eta",
                                            nBinEta, minEta, maxEta, nBinPull, -wPull, wPull);

  // -- Pulls vs Pt
  meMap_[MEIdx::PullPt_vs_Pt ] = theDQM->book2D("PullPt_vs_Pt", "Pull(p_{T}) vs p_{T}",
                                           nBinPt, minPt, maxPt, nBinPull, -wPull, wPull);
  meMap_[MEIdx::PullEta_vs_Pt ] = theDQM->book2D("PullEta_vs_Pt", "Pull(#eta) vs p_{T}",
                                            nBinPt, minPt, maxPt, nBinPull, -wPull, wPull);

  // - Misc variables
  meMap_[MEIdx::NSim]  = theDQM->book1D("NSim" , "Number of particles per event"  , nTrks, 0, nTrks);
  meMap_[MEIdx::NReco] = theDQM->book1D("NReco", "Number of reco tracks per event", nTrks, 0, nTrks);

  meMap_[MEIdx::MisQPt ] = theDQM->book1D("MisQPt" , "Charge mis-id vs Pt" , nBinPt , minPt , maxPt );
  meMap_[MEIdx::MisQEta] = theDQM->book1D("MisQEta", "Charge mis-id vs Eta", nBinEta, minEta, maxEta);

//  meMap_["SimVtxPos"] = theDQM->book1D("SimVtxPos", "sim vertex position", nBinVtxPos, minVtxPos, maxVtxPos);

  // -- Association map
  meMap_[MEIdx::NAssocSimToReco] = theDQM->book1D("NAssocSimToReco", "Number of sim to reco associations", nAssoc, 0, nAssoc);
  meMap_[MEIdx::NAssocRecoToSim] = theDQM->book1D("NAssocRecoToSim", "Number of reco to sim associations", nAssoc, 0, nAssoc);

//  meMap_["NRecoToSim"] = theDQM->book1D("NRecoToSim", "Number of reco to sim associations", nAssoc, 0, nAssoc);
  meMap_[MEIdx::NSimToReco] = theDQM->book1D("NSimToReco", "Number of sim to reco associations", nAssoc, 0, nAssoc);

  // -- Number of Hits
  meMap_[MEIdx::SimNHits ] = theDQM->book1D("SimNHits" , "Number of simTracks vs nSimhits" , nHits, 0, nHits);
  meMap_[MEIdx::RecoNHits] = theDQM->book1D("RecoNHits", "Number of recoTracks vs nSimhits", nHits, 0, nHits);

  meMap_[MEIdx::NSimHits_vs_Pt ] = theDQM->book2D("NSimHits_vs_Pt", "Number of sim Hits vs p_{T}",
                                              nBinPt, minPt, maxPt, nHits, 0, nHits);
  meMap_[MEIdx::NSimHits_vs_Eta] = theDQM->book2D("NSimHits_vs_Eta", "Number of sim Hits vs #eta",
                                              nBinPt, minPt, maxPt, nHits, 0, nHits);

  meMap_[MEIdx::NRecoHits] = theDQM->book1D("NRecoHits", "Number of reco-hits", nHits, 0, nHits);
  meMap_[MEIdx::NRecoHits_vs_Pt ] = theDQM->book2D("NRecoHits_vs_Pt", "Number of reco Hits vs p_{T}", 
                                              nBinPt, minPt, maxPt, nHits, 0, nHits);
  meMap_[MEIdx::NRecoHits_vs_Eta] = theDQM->book2D("NRecoHits_vs_Eta", "Number of reco Hits vs #eta",
                                              nBinEta, minEta, maxEta, nHits, 0, nHits);

  meMap_[MEIdx::NLostHits] = theDQM->book1D("NLostHits", "Number of Lost hits", nHits, 0, nHits);
  meMap_[MEIdx::NLostHits_vs_Pt ] = theDQM->book2D("NLostHits_vs_Pt", "Number of lost Hits vs p_{T}", 
                                              nBinPt, minPt, maxPt, nHits, 0, nHits);
  meMap_[MEIdx::NLostHits_vs_Eta] = theDQM->book2D("NLostHits_vs_Eta", "Number of lost Hits vs #eta",
                                              nBinEta, minEta, maxEta, nHits, 0, nHits);

  meMap_[MEIdx::NDof] = theDQM->book1D("NDof", "Number of DoF", nDof, 0, nDof);
  meMap_[MEIdx::Chi2] = theDQM->book1D("Chi2", "#Chi^{2}", 200, 0, 200);
  meMap_[MEIdx::Chi2Norm] = theDQM->book1D("Chi2Norm", "Normalized #Chi^{2}", nBinErr, 0, 100);
  meMap_[MEIdx::Chi2Prob] = theDQM->book1D("Chi2Prob", "Prob(#Chi^{2})", nBinErr, 0, 1);

  meMap_[MEIdx::NDof_vs_Eta] = theDQM->book2D("NDof_vs_Eta", "Number of DoF vs #eta",
                                         nBinEta, minEta, maxEta, nDof, 0, nDof);
  meMap_[MEIdx::Chi2_vs_Eta] = theDQM->book2D("Chi2_vs_Eta", "#Chi^{2} vs #eta",
                                         nBinEta, minEta, maxEta, 200, 0, 200);
  meMap_[MEIdx::Chi2Norm_vs_Eta] = theDQM->book2D("Chi2Norm_vs_Eta", "Normalized #Chi^{2} vs #eta",
                                             nBinEta, minEta, maxEta, nBinErr, 0, 100);
  meMap_[MEIdx::Chi2Prob_vs_Eta] = theDQM->book2D("Chi2Prob_vs_Eta", "Prob(#Chi^{2}) vs #eta",
                                             nBinEta, minEta, maxEta, nBinErr, 0, 1);
}

RecoMuonValidator::~RecoMuonValidator()
{
  if ( theMuonService ) delete theMuonService;
}

void RecoMuonValidator::beginJob(const EventSetup& eventSetup)
{
  if ( theMuonService ) theMuonService->update(eventSetup);

  theAssociator = 0;
  if ( doAssoc_ ) {
    ESHandle<TrackAssociatorBase> assocHandle;
    eventSetup.get<TrackAssociatorRecord>().get(assocLabel_.label(), assocHandle);
    theAssociator = const_cast<TrackAssociatorBase*>(assocHandle.product());
  }
}

void RecoMuonValidator::endJob()
{
  if ( theDQM && ! outputFileName_.empty() ) theDQM->save(outputFileName_);
}

void RecoMuonValidator::analyze(const Event& event, const EventSetup& eventSetup)
{
  if ( ! theDQM ) {
    LogError("RecoMuonValidator") << "DQMService not initialized\n";
    return;
  }

  // Get TrackingParticles
  Handle<TrackingParticleCollection> simHandle;
  event.getByLabel(simLabel_, simHandle);
  const TrackingParticleCollection simColl = *(simHandle.product());

  // Get Tracks
  Handle<View<Track> > recoHandle;
  event.getByLabel(recoLabel_, recoHandle);
  //const TrackCollection recoColl = *(recoHandle.product());
  View<Track> recoColl = *(recoHandle.product());

  // Get Association maps
  SimToRecoCollection simToRecoColl;
  RecoToSimCollection recoToSimColl;
  if ( doAssoc_ ) {
    simToRecoColl = theAssociator->associateSimToReco(recoHandle, simHandle, &event);
    recoToSimColl = theAssociator->associateRecoToSim(recoHandle, simHandle, &event);
  }
  else {
    Handle<SimToRecoCollection> simToRecoHandle;
    event.getByLabel(assocLabel_, simToRecoHandle);
    simToRecoColl = *(simToRecoHandle.product());

    Handle<RecoToSimCollection> recoToSimHandle;
    event.getByLabel(assocLabel_, recoToSimHandle);
    recoToSimColl = *(recoToSimHandle.product());
  }

  const TrackingParticleCollection::size_type nSim = simColl.size();
  meMap_[MEIdx::NSim]->Fill(static_cast<double>(nSim));

  const TrackCollection::size_type nReco = recoColl.size();
  meMap_[MEIdx::NReco]->Fill(static_cast<double>(nReco));

  meMap_[MEIdx::NAssocSimToReco]->Fill(simToRecoColl.size());
  meMap_[MEIdx::NAssocRecoToSim]->Fill(recoToSimColl.size());

  for(TrackingParticleCollection::size_type i=0; i<nSim; i++) {
    TrackingParticleRef simRef(simHandle, i);
//    if ( !tpSelector(*simRef) ) continue;

    const double simP   = simRef->p();
    const double simPt  = simRef->pt();
    const double simEta = doAbsEta_ ? fabs(simRef->eta()) : simRef->eta();
    const double simPhi = simRef->phi();
    const double simQ   = simRef->charge();
    const double simQPt = simQ/simPt;

    GlobalPoint  simVtx(simRef->vertex().x(), simRef->vertex().y(), simRef->vertex().z());
    GlobalVector simMom(simRef->momentum().x(), simRef->momentum().y(), simRef->momentum().z());
    const double simDxy = -simVtx.x()*sin(simPhi)+simVtx.y()*cos(simPhi);
    const double simDz  = simVtx.z() - (simVtx.x()*simMom.x()+simVtx.y()*simMom.y())*simMom.z()/simMom.perp2();

    const unsigned int nSimHits = simRef->pSimHit_end() - simRef->pSimHit_begin();
//    const double simVtxPos = sqrt(simRef->vertex().perp2());

    meMap_[MEIdx::SimP  ]->Fill(simP  );
    meMap_[MEIdx::SimPt ]->Fill(simPt );
    meMap_[MEIdx::SimEta]->Fill(simEta);
    meMap_[MEIdx::SimPhi]->Fill(simPhi);

    meMap_[MEIdx::SimNHits]->Fill(nSimHits);
    meMap_[MEIdx::NSimHits_vs_Pt ]->Fill(simPt , nSimHits);
    meMap_[MEIdx::NSimHits_vs_Eta]->Fill(simEta, nSimHits);

    // Get sim-reco association for a simRef
    RefToBase<Track> recoRef;
    double assocQuality = 0;
    if ( simToRecoColl.find(simRef) != simToRecoColl.end() ) {
      vector<pair<RefToBase<Track>, double> > recoRefV(simToRecoColl[simRef]);

      unsigned int nSimToReco = recoRefV.size();
      meMap_[MEIdx::NSimToReco]->Fill(nSimToReco);

      if ( ! recoRefV.empty() ) {
        recoRef = recoRefV.begin()->first;
        assocQuality = recoRefV.begin()->second;
        LogVerbatim("RecoMuonValidator") << "TrackingParticle # " << i 
                                         << " with pT = " << simPt 
                                         << " associated with Q=" << assocQuality << '\n';
      }
      else {
        LogVerbatim("RecoMuonValidator") << "TrackingParticle # " << i
                                         << " with pT = " << simPt
                                         << " not associated to any reco::Track\n";
        continue;
      }

      // Histograms for efficiency plots
      meMap_[MEIdx::RecoP  ]->Fill(simP  );
      meMap_[MEIdx::RecoPt ]->Fill(simPt );
      meMap_[MEIdx::RecoEta]->Fill(simEta);
      meMap_[MEIdx::RecoPhi]->Fill(simPhi);

      meMap_[MEIdx::RecoNHits]->Fill(nSimHits);

      // Number of reco-hits
      const int nRecoHits = recoRef->numberOfValidHits();
      const int nLostHits = recoRef->numberOfLostHits();

      meMap_[MEIdx::NRecoHits]->Fill(nRecoHits);
      meMap_[MEIdx::NRecoHits_vs_Pt ]->Fill(simPt , nRecoHits);
      meMap_[MEIdx::NRecoHits_vs_Eta]->Fill(simEta, nRecoHits);

      meMap_[MEIdx::NLostHits]->Fill(nLostHits);
      meMap_[MEIdx::NLostHits_vs_Pt ]->Fill(simPt , nLostHits);
      meMap_[MEIdx::NLostHits_vs_Eta]->Fill(simEta, nLostHits);

      const double recoNDof = recoRef->ndof();
      const double recoChi2 = recoRef->chi2();
      const double recoChi2Norm = recoRef->normalizedChi2();
      const double recoChi2Prob = TMath::Prob(recoRef->chi2(), static_cast<int>(recoRef->ndof()));

      meMap_[MEIdx::NDof]->Fill(recoNDof);
      meMap_[MEIdx::Chi2]->Fill(recoChi2);
      meMap_[MEIdx::Chi2Norm]->Fill(recoChi2Norm);
      meMap_[MEIdx::Chi2Prob]->Fill(recoChi2Prob);

      meMap_[MEIdx::NDof_vs_Eta]->Fill(simEta, recoNDof);
      meMap_[MEIdx::Chi2_vs_Eta]->Fill(simEta, recoChi2);
      meMap_[MEIdx::Chi2Norm_vs_Eta]->Fill(simEta, recoChi2Norm);
      meMap_[MEIdx::Chi2Prob_vs_Eta]->Fill(simEta, recoChi2Prob);

      const double recoQ   = recoRef->charge();
      if ( simQ*recoQ < 0 ) {
        meMap_[MEIdx::MisQPt ]->Fill(simPt );
        meMap_[MEIdx::MisQEta]->Fill(simEta);
      }

      const double recoP   = sqrt(recoRef->momentum().mag2());
      const double recoPt  = sqrt(recoRef->momentum().perp2());
      const double recoEta = recoRef->momentum().eta();
      const double recoPhi = recoRef->momentum().phi();
      const double recoQPt = recoQ/recoPt;

      const double recoDxy = recoRef->dxy();
      const double recoDz  = recoRef->dz();

      const double errP   = (recoP-simP)/simP;
      const double errPt  = (recoPt-simPt)/simPt;
      const double errEta = (recoEta-simEta)/simEta;
      const double errPhi = (recoPhi-simPhi)/simPhi;
      const double errQPt = (recoQPt-simQPt)/simQPt;

      const double errDxy = (recoDxy-simDxy)/simDxy;
      const double errDz  = (recoDz-simDz)/simDz;

      meMap_[MEIdx::ErrP  ]->Fill(errP  );
      meMap_[MEIdx::ErrPt ]->Fill(errPt );
      meMap_[MEIdx::ErrEta]->Fill(errEta);
      meMap_[MEIdx::ErrPhi]->Fill(errPhi);
      meMap_[MEIdx::ErrDxy]->Fill(errDxy);
      meMap_[MEIdx::ErrDz ]->Fill(errDz );

      meMap_[MEIdx::ErrP_vs_Eta  ]->Fill(simEta, errP  );
      meMap_[MEIdx::ErrPt_vs_Eta ]->Fill(simEta, errPt );
      meMap_[MEIdx::ErrQPt_vs_Eta]->Fill(simEta, errQPt);

      meMap_[MEIdx::ErrP_vs_P   ]->Fill(simP  , errP  );
      meMap_[MEIdx::ErrPt_vs_Pt ]->Fill(simPt , errPt );
      meMap_[MEIdx::ErrQPt_vs_Pt]->Fill(simQPt, errQPt);

      meMap_[MEIdx::ErrEta_vs_Eta]->Fill(simEta, errEta);

      const double pullPt  = (recoPt-simPt)/recoRef->ptError();
      const double pullQPt = (recoQPt-simQPt)/recoRef->qoverpError();
      const double pullEta = (recoEta-simEta)/recoRef->etaError();
      const double pullPhi = (recoPhi-simPhi)/recoRef->phiError();
      const double pullDxy = (recoDxy-simDxy)/recoRef->dxyError();
      const double pullDz  = (recoDz-simDz)/recoRef->dzError();

      meMap_[MEIdx::PullPt ]->Fill(pullPt );
      meMap_[MEIdx::PullEta]->Fill(pullEta);
      meMap_[MEIdx::PullPhi]->Fill(pullPhi);
      meMap_[MEIdx::PullQPt]->Fill(pullQPt);
      meMap_[MEIdx::PullDxy]->Fill(pullDxy);
      meMap_[MEIdx::PullDz ]->Fill(pullDz );

      meMap_[MEIdx::PullPt_vs_Eta]->Fill(simEta, pullPt);
      meMap_[MEIdx::PullPt_vs_Pt ]->Fill(simPt, pullPt);

      meMap_[MEIdx::PullEta_vs_Eta]->Fill(simEta, pullEta);
      meMap_[MEIdx::PullPhi_vs_Eta]->Fill(simEta, pullPhi);

      meMap_[MEIdx::PullEta_vs_Pt]->Fill(simPt, pullEta);

    }
  }
}
/* vim:set ts=2 sts=2 sw=2 expandtab: */
