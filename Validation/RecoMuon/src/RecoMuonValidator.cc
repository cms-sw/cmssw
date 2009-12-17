#include "Validation/RecoMuon/src/RecoMuonValidator.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
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

struct HistoDimensions {
  unsigned int nBinP;
  double minP, maxP;

  unsigned int nBinPt;
  double minPt, maxPt;

  bool doAbsEta;
  unsigned int nBinEta;
  double minEta, maxEta;

  unsigned int nBinPhi;
  double minPhi, maxPhi;

  unsigned int nBinPull;
  double wPull;

  unsigned int nBinErr;
  double minErrP, maxErrP;
  double minErrPt, maxErrPt;
  double minErrQPt, maxErrQPt;
  double minErrEta, maxErrEta;
  double minErrPhi, maxErrPhi;
  double minErrDxy, maxErrDxy;
  double minErrDz, maxErrDz;

  unsigned int nTrks, nAssoc;
  unsigned int nDof;
};

struct RecoMuonValidator::MuonME {
  void bookHistograms(DQMStore* dqm, const string& dirName, const HistoDimensions& hDim)
  {
    dqm->cd();
    dqm->setCurrentFolder(dirName.c_str());

    doAbsEta_ = hDim.doAbsEta;

    hP_   = dqm->book1D("P"  , "p of recoTracks"    , hDim.nBinP  , hDim.minP  , hDim.maxP  );
    hPt_  = dqm->book1D("Pt" , "p_{T} of recoTracks", hDim.nBinPt , hDim.minPt , hDim.maxPt );
    hEta_ = dqm->book1D("Eta", "#eta of recoTracks" , hDim.nBinEta, hDim.minEta, hDim.maxEta);
    hPhi_ = dqm->book1D("Phi", "#phi of recoTracks" , hDim.nBinPhi, hDim.minPhi, hDim.maxPhi);

    // - Resolutions
    hErrP_   = dqm->book1D("ErrP"  , "#Delta(p)/p"        , hDim.nBinErr, hDim.minErrP  , hDim.maxErrP  );
    hErrPBarrel_   = dqm->book1D("ErrP_barrel"  , "#Delta(p)/p"        , hDim.nBinErr, hDim.minErrP  , hDim.maxErrP  );
    hErrPOverlap_   = dqm->book1D("ErrP_overlap"  , "#Delta(p)/p"        , hDim.nBinErr, hDim.minErrP  , hDim.maxErrP  );
    hErrPEndcap_   = dqm->book1D("ErrP_endcap"  , "#Delta(p)/p"        , hDim.nBinErr, hDim.minErrP  , hDim.maxErrP  );
    hErrPt_  = dqm->book1D("ErrPt" , "#Delta(p_{T})/p_{T}", hDim.nBinErr, hDim.minErrPt , hDim.maxErrPt );
    hErrPtBarrel_  = dqm->book1D("ErrPt_barrel" , "#Delta(p_{T})/p_{T}", hDim.nBinErr, hDim.minErrPt , hDim.maxErrPt );
    hErrPtOverlap_  = dqm->book1D("ErrPt_overlap" , "#Delta(p_{T})/p_{T}", hDim.nBinErr, hDim.minErrPt , hDim.maxErrPt );
    hErrPtEndcap_  = dqm->book1D("ErrPt_endcap" , "#Delta(p_{T})/p_{T}", hDim.nBinErr, hDim.minErrPt , hDim.maxErrPt );
    hErrEta_ = dqm->book1D("ErrEta", "#sigma(#eta))"      , hDim.nBinErr, hDim.minErrEta, hDim.maxErrEta);
    hErrPhi_ = dqm->book1D("ErrPhi", "#sigma(#phi)"       , hDim.nBinErr, hDim.minErrPhi, hDim.maxErrPhi);
    hErrDxy_ = dqm->book1D("ErrDxy", "#sigma(d_{xy})"     , hDim.nBinErr, hDim.minErrDxy, hDim.maxErrDxy);
    hErrDz_  = dqm->book1D("ErrDz" , "#sigma(d_{z})"      , hDim.nBinErr, hDim.minErrDz , hDim.maxErrDz );

    // -- Resolutions vs Eta
    hErrP_vs_Eta_   = dqm->book2D("ErrP_vs_Eta", "#Delta(p)/p vs #eta",
                                  hDim.nBinEta, hDim.minEta, hDim.maxEta, hDim.nBinErr, hDim.minErrP, hDim.maxErrP);
    hErrPt_vs_Eta_  = dqm->book2D("ErrPt_vs_Eta", "#Delta(p_{T})/p_{T} vs #eta",
                                  hDim.nBinEta, hDim.minEta, hDim.maxEta, hDim.nBinErr, hDim.minErrPt, hDim.maxErrPt);
    hErrQPt_vs_Eta_ = dqm->book2D("ErrQPt_vs_Eta", "#Delta(q/p_{T})/(q/p_{T}) vs #eta",
                                  hDim.nBinEta, hDim.minEta, hDim.maxEta, hDim.nBinErr, hDim.minErrQPt, hDim.maxErrQPt);
    hErrEta_vs_Eta_ = dqm->book2D("ErrEta_vs_Eta", "#sigma(#eta) vs #eta",
                                  hDim.nBinEta, hDim.minEta, hDim.maxEta, hDim.nBinErr, hDim.minErrEta, hDim.maxErrEta);

    // -- Resolutions vs momentum
    hErrP_vs_P_    = dqm->book2D("ErrP_vs_P", "#Delta(p)/p vs p",
                                 hDim.nBinP, hDim.minP, hDim.maxP, hDim.nBinErr, hDim.minErrP, hDim.maxErrP);

    hErrPt_vs_Pt_  = dqm->book2D("ErrPt_vs_Pt", "#Delta(p_{T})/p_{T} vs p_{T}",
                                 hDim.nBinPt, hDim.minPt, hDim.maxPt, hDim.nBinErr, hDim.minErrPt, hDim.maxErrPt);
    hErrQPt_vs_Pt_ = dqm->book2D("ErrQPt_vs_Pt", "#Delta(q/p_{T})/(q/p_{T}) vs p_{T}",
                                 hDim.nBinPt, hDim.minPt, hDim.maxPt, hDim.nBinErr, hDim.minErrQPt, hDim.maxErrQPt);


    // - Pulls
    hPullPt_  = dqm->book1D("PullPt" , "Pull(#p_{T})" , hDim.nBinPull, -hDim.wPull, hDim.wPull);
    hPullEta_ = dqm->book1D("PullEta", "Pull(#eta)"   , hDim.nBinPull, -hDim.wPull, hDim.wPull);
    hPullPhi_ = dqm->book1D("PullPhi", "Pull(#phi)"   , hDim.nBinPull, -hDim.wPull, hDim.wPull);
    hPullQPt_ = dqm->book1D("PullQPt", "Pull(q/p_{T})", hDim.nBinPull, -hDim.wPull, hDim.wPull);
    hPullDxy_ = dqm->book1D("PullDxy", "Pull(D_{xy})" , hDim.nBinPull, -hDim.wPull, hDim.wPull);
    hPullDz_  = dqm->book1D("PullDz" , "Pull(D_{z})"  , hDim.nBinPull, -hDim.wPull, hDim.wPull);

    // -- Pulls vs Eta
    hPullPt_vs_Eta_  = dqm->book2D("PullPt_vs_Eta", "Pull(p_{T}) vs #eta",
                                   hDim.nBinEta, hDim.minEta, hDim.maxEta, hDim.nBinPull, -hDim.wPull, hDim.wPull);
    hPullEta_vs_Eta_ = dqm->book2D("PullEta_vs_Eta", "Pull(#eta) vs #eta",
                                   hDim.nBinEta, hDim.minEta, hDim.maxEta, hDim.nBinPull, -hDim.wPull, hDim.wPull);
    hPullPhi_vs_Eta_ = dqm->book2D("PullPhi_vs_Eta", "Pull(#phi) vs #eta",
                                   hDim.nBinEta, hDim.minEta, hDim.maxEta, hDim.nBinPull, -hDim.wPull, hDim.wPull);

    // -- Pulls vs Pt
    hPullPt_vs_Pt_ = dqm->book2D("PullPt_vs_Pt", "Pull(p_{T}) vs p_{T}",
                                 hDim.nBinPt, hDim.minPt, hDim.maxPt, hDim.nBinPull, -hDim.wPull, hDim.wPull);
    hPullEta_vs_Pt_ = dqm->book2D("PullEta_vs_Pt", "Pull(#eta) vs p_{T}",
                                  hDim.nBinPt, hDim.minPt, hDim.maxPt, hDim.nBinPull, -hDim.wPull, hDim.wPull);

    // - Misc. variables
    hNTrks_ = dqm->book1D("NTrks", "Number of reco tracks per event", hDim.nTrks, 0, hDim.nTrks);
    hNTrksEta_ = dqm->book1D("NTrksEta", "Number of reco tracks vs #eta", hDim.nBinEta, hDim.minEta, hDim.maxEta);
    hNTrksPt_ = dqm->book1D("NTrksPt", "Number of reco tracks vs p_{T}", hDim.nBinPt, hDim.minPt, hDim.maxPt);

    hMisQPt_  = dqm->book1D("MisQPt" , "Charge mis-id vs Pt" , hDim.nBinPt , hDim.minPt , hDim.maxPt );
    hMisQEta_ = dqm->book1D("MisQEta", "Charge mis-id vs Eta", hDim.nBinEta, hDim.minEta, hDim.maxEta);

    // -- Number of Hits
    const int nHits = 80;
    hNHits_ = dqm->book1D("NHits", "Number of hits", nHits, 0, nHits);
    hNHits_vs_Pt_  = dqm->book2D("NHits_vs_Pt", "Number of hits vs p_{T}",
                                 hDim.nBinPt, hDim.minPt, hDim.maxPt, nHits/4, 0, nHits);
    hNHits_vs_Eta_ = dqm->book2D("NHits_vs_Eta", "Number of hits vs #eta",
                                 hDim.nBinEta, hDim.minEta, hDim.maxEta, nHits/4, 0, nHits);

    hNSimHits_ = dqm->book1D("NSimHits", "Number of simHits", nHits, 0, nHits);

    const int nLostHits = 5;
    hNLostHits_ = dqm->book1D("NLostHits", "Number of Lost hits", nLostHits, 0, nLostHits);
    hNLostHits_vs_Pt_  = dqm->book2D("NLostHits_vs_Pt", "Number of lost Hits vs p_{T}",
                                     hDim.nBinPt, hDim.minPt, hDim.maxPt, nLostHits, 0, nLostHits);
    hNLostHits_vs_Eta_ = dqm->book2D("NLostHits_vs_Eta", "Number of lost Hits vs #eta",
                                     hDim.nBinEta, hDim.minEta, hDim.maxEta, nLostHits, 0, nLostHits);

    const int nTrackerHits = 40;
    hNTrackerHits_ = dqm->book1D("NTrackerHits", "Number of valid tracker hits", nTrackerHits, 0, nTrackerHits);
    hNTrackerHits_vs_Pt_ = dqm->book2D("NTrackerHits_vs_Pt", "Number of valid traker hits vs p_{T}",
                                       hDim.nBinPt, hDim.minPt, hDim.maxPt, nTrackerHits/4, 0, nTrackerHits);
    hNTrackerHits_vs_Eta_ = dqm->book2D("NTrackerHits_vs_Eta", "Number of valid tracker hits vs #eta",
                                        hDim.nBinEta, hDim.minEta, hDim.maxEta, nTrackerHits/4, 0, nTrackerHits);

    const int nMuonHits = 40;
    hNMuonHits_ = dqm->book1D("NMuonHits", "Number of valid muon hits", nMuonHits, 0, nMuonHits);
    hNMuonHits_vs_Pt_  = dqm->book2D("NMuonHits_vs_Pt", "Number of valid muon hits vs p_{T}",
                                     hDim.nBinPt, hDim.minPt, hDim.maxPt, nMuonHits/4, 0, nMuonHits);
    hNMuonHits_vs_Eta_ = dqm->book2D("NMuonHits_vs_Eta", "Number of valid muon hits vs #eta",
                                     hDim.nBinEta, hDim.minEta, hDim.maxEta, nMuonHits/4, 0, nMuonHits);

    hNDof_ = dqm->book1D("NDof", "Number of DoF", hDim.nDof, 0, hDim.nDof);
    hChi2_ = dqm->book1D("Chi2", "#Chi^{2}", hDim.nBinErr, 0, 200);
    hChi2Norm_ = dqm->book1D("Chi2Norm", "Normalized #Chi^{2}", hDim.nBinErr, 0, 50);
    hChi2Prob_ = dqm->book1D("Chi2Prob", "Prob(#Chi^{2})", hDim.nBinErr, 0, 1);

    hNDof_vs_Eta_ = dqm->book2D("NDof_vs_Eta", "Number of DoF vs #eta",
                                hDim.nBinEta, hDim.minEta, hDim.maxEta, hDim.nBinErr, 0, hDim.nDof);
    hChi2_vs_Eta_ = dqm->book2D("Chi2_vs_Eta", "#Chi^{2} vs #eta",
                                hDim.nBinEta, hDim.minEta, hDim.maxEta, hDim.nBinErr, 0, 200);
    hChi2Norm_vs_Eta_ = dqm->book2D("Chi2Norm_vs_Eta", "Normalized #Chi^{2} vs #eta",
                                    hDim.nBinEta, hDim.minEta, hDim.maxEta, hDim.nBinErr, 0, 100);
    hChi2Prob_vs_Eta_ = dqm->book2D("Chi2Prob_vs_Eta", "Prob(#Chi^{2}) vs #eta",
                                    hDim.nBinEta, hDim.minEta, hDim.maxEta, hDim.nBinErr, 0, 1);

    hNSimToReco_ = dqm->book1D("NSimToReco", "Number of associated reco tracks", hDim.nAssoc, 0, hDim.nAssoc);
    hNRecoToSim_ = dqm->book1D("NRecoToSim", "Number of associated sim TP's", hDim.nAssoc, 0, hDim.nAssoc);
  };

  void fill(const TrackingParticle* simRef, const Track* recoRef)
  {
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

    // Histograms for efficiency plots
    hP_  ->Fill(simP  );
    hPt_ ->Fill(simPt );
    hEta_->Fill(simEta);
    hPhi_->Fill(simPhi);
    hNSimHits_->Fill(nSimHits);

    // Number of reco-hits
    const int nRecoHits = recoRef->numberOfValidHits();
    const int nLostHits = recoRef->numberOfLostHits();

    hNHits_->Fill(nRecoHits);
    hNHits_vs_Pt_ ->Fill(simPt , nRecoHits);
    hNHits_vs_Eta_->Fill(simEta, nRecoHits);

    hNLostHits_->Fill(nLostHits);
    hNLostHits_vs_Pt_ ->Fill(simPt , nLostHits);
    hNLostHits_vs_Eta_->Fill(simEta, nLostHits);

    const double recoNDof = recoRef->ndof();
    const double recoChi2 = recoRef->chi2();
    const double recoChi2Norm = recoRef->normalizedChi2();
    const double recoChi2Prob = TMath::Prob(recoRef->chi2(), static_cast<int>(recoRef->ndof()));

    hNDof_->Fill(recoNDof);
    hChi2_->Fill(recoChi2);
    hChi2Norm_->Fill(recoChi2Norm);
    hChi2Prob_->Fill(recoChi2Prob);

    hNDof_vs_Eta_->Fill(simEta, recoNDof);
    hChi2_vs_Eta_->Fill(simEta, recoChi2);
    hChi2Norm_vs_Eta_->Fill(simEta, recoChi2Norm);
    hChi2Prob_vs_Eta_->Fill(simEta, recoChi2Prob);

    const double recoQ   = recoRef->charge();
    if ( simQ*recoQ < 0 ) {
      hMisQPt_ ->Fill(simPt );
      hMisQEta_->Fill(simEta);
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

    hErrP_  ->Fill(errP  );
    hErrPt_ ->Fill(errPt );
    hErrEta_->Fill(errEta);
    hErrPhi_->Fill(errPhi);
    hErrDxy_->Fill(errDxy);
    hErrDz_ ->Fill(errDz );

    if(fabs(simEta) > 0. && fabs(simEta) < 0.8) {
      hErrPBarrel_->Fill(errP);
      hErrPtBarrel_->Fill(errPt);
    } else if (fabs(simEta) > 0.8 && fabs(simEta) < 1.2) {
      hErrPOverlap_->Fill(errP);
      hErrPtOverlap_->Fill(errPt);
    } else if (fabs(simEta) > 1.2 ){
      hErrPEndcap_->Fill(errP);
      hErrPtEndcap_->Fill(errPt);
    }

    hErrP_vs_Eta_  ->Fill(simEta, errP  );
    hErrPt_vs_Eta_ ->Fill(simEta, errPt );
    hErrQPt_vs_Eta_->Fill(simEta, errQPt);

    hErrP_vs_P_   ->Fill(simP  , errP  );
    hErrPt_vs_Pt_ ->Fill(simPt , errPt );
    hErrQPt_vs_Pt_->Fill(simQPt, errQPt);

    hErrEta_vs_Eta_->Fill(simEta, errEta);

    const double pullPt  = (recoPt-simPt)/recoRef->ptError();
    const double pullQPt = (recoQPt-simQPt)/recoRef->qoverpError();
    const double pullEta = (recoEta-simEta)/recoRef->etaError();
    const double pullPhi = (recoPhi-simPhi)/recoRef->phiError();
    const double pullDxy = (recoDxy-simDxy)/recoRef->dxyError();
    const double pullDz  = (recoDz-simDz)/recoRef->dzError();

    hPullPt_ ->Fill(pullPt );
    hPullEta_->Fill(pullEta);
    hPullPhi_->Fill(pullPhi);
    hPullQPt_->Fill(pullQPt);
    hPullDxy_->Fill(pullDxy);
    hPullDz_ ->Fill(pullDz );

    hPullPt_vs_Eta_->Fill(simEta, pullPt);
    hPullPt_vs_Pt_ ->Fill(simPt, pullPt);

    hPullEta_vs_Eta_->Fill(simEta, pullEta);
    hPullPhi_vs_Eta_->Fill(simEta, pullPhi);

    hPullEta_vs_Pt_->Fill(simPt, pullEta);


  };

  bool doAbsEta_;

  typedef MonitorElement* MEP;

  MEP hP_, hPt_, hEta_, hPhi_;
  MEP hErrP_, hErrPt_, hErrEta_, hErrPhi_;
  MEP hErrPBarrel_, hErrPOverlap_, hErrPEndcap_;
  MEP hErrPtBarrel_, hErrPtOverlap_, hErrPtEndcap_;
  MEP hErrDxy_, hErrDz_;

  MEP hErrP_vs_Eta_, hErrPt_vs_Eta_, hErrQPt_vs_Eta_;
  MEP hErrP_vs_P_, hErrPt_vs_Pt_, hErrQPt_vs_Pt_, hErrEta_vs_Eta_;

  MEP hPullPt_, hPullEta_, hPullPhi_, hPullQPt_, hPullDxy_, hPullDz_;
  MEP hPullPt_vs_Eta_, hPullPt_vs_Pt_, hPullEta_vs_Eta_, hPullPhi_vs_Eta_, hPullEta_vs_Pt_;

  MEP hNDof_, hChi2_, hChi2Norm_, hChi2Prob_;
  MEP hNDof_vs_Eta_, hChi2_vs_Eta_, hChi2Norm_vs_Eta_, hChi2Prob_vs_Eta_;

  MEP hNTrks_, hNTrksEta_,  hNTrksPt_;

  MEP hMisQPt_, hMisQEta_;

  MEP hNSimHits_;
  MEP hNHits_, hNLostHits_, hNTrackerHits_, hNMuonHits_;
  MEP hNHits_vs_Pt_, hNHits_vs_Eta_;
  MEP hNLostHits_vs_Pt_, hNLostHits_vs_Eta_;
  MEP hNTrackerHits_vs_Pt_, hNTrackerHits_vs_Eta_;
  MEP hNMuonHits_vs_Pt_, hNMuonHits_vs_Eta_;

  MEP hNSimToReco_, hNRecoToSim_;
};

struct RecoMuonValidator::CommonME {
  typedef MonitorElement* MEP;

  MEP hSimP_, hSimPt_, hSimEta_, hSimPhi_;
  MEP hNSim_, hNMuon_;
  MEP hNSimHits_;

  MEP hTrkToGlbDiffNTrackerHits_, hStaToGlbDiffNMuonHits_;
  MEP hTrkToGlbDiffNTrackerHitsEta_, hStaToGlbDiffNMuonHitsEta_;
  MEP hTrkToGlbDiffNTrackerHitsPt_, hStaToGlbDiffNMuonHitsPt_;
};

RecoMuonValidator::RecoMuonValidator(const ParameterSet& pset)
{
  verbose_ = pset.getUntrackedParameter<unsigned int>("verbose", 0);

  outputFileName_ = pset.getUntrackedParameter<string>("outputFileName", "");

  // Set histogram dimensions
  HistoDimensions hDim;

  hDim.nBinP = pset.getUntrackedParameter<unsigned int>("nBinP");
  hDim.minP = pset.getUntrackedParameter<double>("minP");
  hDim.maxP = pset.getUntrackedParameter<double>("maxP");

  hDim.nBinPt = pset.getUntrackedParameter<unsigned int>("nBinPt");
  hDim.minPt = pset.getUntrackedParameter<double>("minPt");
  hDim.maxPt = pset.getUntrackedParameter<double>("maxPt");

  doAbsEta_ = pset.getUntrackedParameter<bool>("doAbsEta");
  hDim.doAbsEta = doAbsEta_;
  hDim.nBinEta  = pset.getUntrackedParameter<unsigned int>("nBinEta");
  hDim.minEta = pset.getUntrackedParameter<double>("minEta");
  hDim.maxEta = pset.getUntrackedParameter<double>("maxEta");

  hDim.nBinPhi  = pset.getUntrackedParameter<unsigned int>("nBinPhi");
  hDim.minPhi = pset.getUntrackedParameter<double>("minPhi", -TMath::Pi());
  hDim.maxPhi = pset.getUntrackedParameter<double>("maxPhi",  TMath::Pi());

  hDim.nBinErr  = pset.getUntrackedParameter<unsigned int>("nBinErr");
  hDim.nBinPull = pset.getUntrackedParameter<unsigned int>("nBinPull");

  hDim.wPull = pset.getUntrackedParameter<double>("wPull");

  hDim.minErrP = pset.getUntrackedParameter<double>("minErrP");
  hDim.maxErrP = pset.getUntrackedParameter<double>("maxErrP");

  hDim.minErrPt = pset.getUntrackedParameter<double>("minErrPt");
  hDim.maxErrPt = pset.getUntrackedParameter<double>("maxErrPt");

  hDim.minErrQPt = pset.getUntrackedParameter<double>("minErrQPt");
  hDim.maxErrQPt = pset.getUntrackedParameter<double>("maxErrQPt");

  hDim.minErrEta = pset.getUntrackedParameter<double>("minErrEta");
  hDim.maxErrEta = pset.getUntrackedParameter<double>("maxErrEta");

  hDim.minErrPhi = pset.getUntrackedParameter<double>("minErrPhi");
  hDim.maxErrPhi = pset.getUntrackedParameter<double>("maxErrPhi");

  hDim.minErrDxy = pset.getUntrackedParameter<double>("minErrDxy");
  hDim.maxErrDxy = pset.getUntrackedParameter<double>("maxErrDxy");

  hDim.minErrDz  = pset.getUntrackedParameter<double>("minErrDz" );
  hDim.maxErrDz  = pset.getUntrackedParameter<double>("maxErrDz" );

  hDim.nTrks = pset.getUntrackedParameter<unsigned int>("nTrks");
  hDim.nAssoc = pset.getUntrackedParameter<unsigned int>("nAssoc");
  hDim.nDof = pset.getUntrackedParameter<unsigned int>("nDof", 55);

  // Labels for simulation and reconstruction tracks
  simLabel_  = pset.getParameter<InputTag>("simLabel" );
  trkMuLabel_ = pset.getParameter<InputTag>("trkMuLabel");
  staMuLabel_ = pset.getParameter<InputTag>("staMuLabel");
  glbMuLabel_ = pset.getParameter<InputTag>("glbMuLabel");
  muonLabel_ = pset.getParameter<InputTag>("muonLabel");

  // Labels for sim-reco association
  doAssoc_ = pset.getUntrackedParameter<bool>("doAssoc", true);
  trkMuAssocLabel_ = pset.getParameter<InputTag>("trkMuAssocLabel");
  staMuAssocLabel_ = pset.getParameter<InputTag>("staMuAssocLabel");
  glbMuAssocLabel_ = pset.getParameter<InputTag>("glbMuAssocLabel");

//  seedPropagatorName_ = pset.getParameter<string>("SeedPropagator");

  ParameterSet tpset = pset.getParameter<ParameterSet>("tpSelector");
  tpSelector_ = TrackingParticleSelector(tpset.getParameter<double>("ptMin"),
                                         tpset.getParameter<double>("minRapidity"),
                                         tpset.getParameter<double>("maxRapidity"),
                                         tpset.getParameter<double>("tip"),
                                         tpset.getParameter<double>("lip"),
                                         tpset.getParameter<int>("minHit"),
                                         tpset.getParameter<bool>("signalOnly"),
                                         tpset.getParameter<bool>("chargedOnly"),
                                         tpset.getParameter<std::vector<int> >("pdgId"));

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
  if ( subDir_.empty() ) subDir_ = "RecoMuonV";
  if ( subDir_[subDir_.size()-1] == '/' ) subDir_.erase(subDir_.size()-1);

  // book histograms
  theDQM->cd();

  theDQM->setCurrentFolder(subDir_+"/Muons");

  commonME_ = new CommonME;
  trkMuME_ = new MuonME;
  staMuME_ = new MuonME;
  glbMuME_ = new MuonME;

  commonME_->hSimP_   = theDQM->book1D("SimP"  , "p of simTracks"    , hDim.nBinP  , hDim.minP  , hDim.maxP  );
  commonME_->hSimPt_  = theDQM->book1D("SimPt" , "p_{T} of simTracks", hDim.nBinPt , hDim.minPt , hDim.maxPt );
  commonME_->hSimEta_ = theDQM->book1D("SimEta", "#eta of simTracks" , hDim.nBinEta, hDim.minEta, hDim.maxEta);
  commonME_->hSimPhi_ = theDQM->book1D("SimPhi", "#phi of simTracks" , hDim.nBinPhi, hDim.minPhi, hDim.maxPhi);

  commonME_->hNSim_  = theDQM->book1D("NSim" , "Number of particles per event", hDim.nTrks, 0, hDim.nTrks);
  commonME_->hNMuon_ = theDQM->book1D("NMuon", "Number of muons per event"    , hDim.nTrks, 0, hDim.nTrks);

  const int nHits = 201;
  commonME_->hNSimHits_ = theDQM->book1D("NSimHits", "Number of simHits", nHits, -100.5, 100.5);

  commonME_->hTrkToGlbDiffNTrackerHits_ = theDQM->book1D("TrkGlbDiffNTrackerHits", "Difference of number of tracker hits (tkMuon - globalMuon)", nHits, -100.5, 100.5);
  commonME_->hStaToGlbDiffNMuonHits_ = theDQM->book1D("StaGlbDiffNMuonHits", "Difference of number of muon hits (staMuon - globalMuon)", nHits, -100.5, 100.5);

  commonME_->hTrkToGlbDiffNTrackerHitsEta_ = theDQM->book2D("TrkGlbDiffNTrackerHitsEta", "Difference of number of tracker hits (tkMuon - globalMuon)",   hDim.nBinEta, hDim.minEta, hDim.maxEta,nHits, -100.5, 100.5);
  commonME_->hStaToGlbDiffNMuonHitsEta_ = theDQM->book2D("StaGlbDiffNMuonHitsEta", "Difference of number of muon hits (staMuon - globalMuon)",   hDim.nBinEta, hDim.minEta, hDim.maxEta,nHits, -100.5, 100.5);

  commonME_->hTrkToGlbDiffNTrackerHitsPt_ = theDQM->book2D("TrkGlbDiffNTrackerHitsPt", "Difference of number of tracker hits (tkMuon - globalMuon)",  hDim.nBinPt, hDim.minPt, hDim.maxPt,nHits, -100.5, 100.5);
  commonME_->hStaToGlbDiffNMuonHitsPt_ = theDQM->book2D("StaGlbDiffNMuonHitsPt", "Difference of number of muon hits (staMuon - globalMuon)",  hDim.nBinPt, hDim.minPt, hDim.maxPt,nHits, -100.5, 100.5);

  // - histograms on tracking variables
  theDQM->setCurrentFolder(subDir_+"/Trk");
  theDQM->bookString("TrackLabel", trkMuLabel_.label()+"_"+trkMuLabel_.instance());
  theDQM->bookString("AssocLabel", trkMuAssocLabel_.label());

  theDQM->setCurrentFolder(subDir_+"/Sta");
  theDQM->bookString("TrackLabel", staMuLabel_.label()+"_"+staMuLabel_.instance());
  theDQM->bookString("AssocLabel", staMuAssocLabel_.label());

  theDQM->setCurrentFolder(subDir_+"/Glb");
  theDQM->bookString("TrackLabel", glbMuLabel_.label()+"_"+glbMuLabel_.instance());
  theDQM->bookString("AssocLabel", glbMuAssocLabel_.label());
  
  trkMuME_->bookHistograms(theDQM, subDir_+"/Trk", hDim);
  staMuME_->bookHistograms(theDQM, subDir_+"/Sta", hDim);
  glbMuME_->bookHistograms(theDQM, subDir_+"/Glb", hDim);

  if ( verbose_ > 0 ) theDQM->showDirStructure();

}

RecoMuonValidator::~RecoMuonValidator()
{
  if ( theMuonService ) delete theMuonService;
}

void RecoMuonValidator::beginRun(const edm::Run& , const EventSetup& eventSetup)
{
  if ( theMuonService ) theMuonService->update(eventSetup);

  trkMuAssociator_ = 0;
  staMuAssociator_ = 0;
  glbMuAssociator_ = 0;
  if ( doAssoc_ ) {
    ESHandle<TrackAssociatorBase> trkMuAssocHandle;
    eventSetup.get<TrackAssociatorRecord>().get(trkMuAssocLabel_.label(), trkMuAssocHandle);
    trkMuAssociator_ = const_cast<TrackAssociatorBase*>(trkMuAssocHandle.product());

    ESHandle<TrackAssociatorBase> staMuAssocHandle;
    eventSetup.get<TrackAssociatorRecord>().get(staMuAssocLabel_.label(), staMuAssocHandle);
    staMuAssociator_ = const_cast<TrackAssociatorBase*>(staMuAssocHandle.product());

    ESHandle<TrackAssociatorBase> glbMuAssocHandle;
    eventSetup.get<TrackAssociatorRecord>().get(glbMuAssocLabel_.label(), glbMuAssocHandle);
    glbMuAssociator_ = const_cast<TrackAssociatorBase*>(glbMuAssocHandle.product());
  }
}

void RecoMuonValidator::endRun()
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

  // Get Muon Tracks
  Handle<View<Track> > trkMuHandle;
  event.getByLabel(trkMuLabel_, trkMuHandle);
  View<Track> trkMuColl = *(trkMuHandle.product());

  Handle<View<Track> > staMuHandle;
  event.getByLabel(staMuLabel_, staMuHandle);
  View<Track> staMuColl = *(staMuHandle.product());

  Handle<View<Track> > glbMuHandle;
  event.getByLabel(glbMuLabel_, glbMuHandle);
  View<Track> glbMuColl = *(glbMuHandle.product());

  // Get Muons
  Handle<View<Muon> > muonHandle;
  event.getByLabel(muonLabel_, muonHandle);
  View<Muon> muonColl = *(muonHandle.product());

  // Get Association maps
  SimToRecoCollection simToTrkMuColl;
  SimToRecoCollection simToStaMuColl;
  SimToRecoCollection simToGlbMuColl;

  RecoToSimCollection trkMuToSimColl;
  RecoToSimCollection staMuToSimColl;
  RecoToSimCollection glbMuToSimColl;

  if ( doAssoc_ ) {
    // SimToReco associations
    simToTrkMuColl = trkMuAssociator_->associateSimToReco(trkMuHandle, simHandle, &event);
    simToStaMuColl = staMuAssociator_->associateSimToReco(staMuHandle, simHandle, &event);
    simToGlbMuColl = glbMuAssociator_->associateSimToReco(glbMuHandle, simHandle, &event);

    // // RecoToSim associations
    trkMuToSimColl = trkMuAssociator_->associateRecoToSim(trkMuHandle, simHandle, &event);
    staMuToSimColl = staMuAssociator_->associateRecoToSim(staMuHandle, simHandle, &event);
    glbMuToSimColl = glbMuAssociator_->associateRecoToSim(glbMuHandle, simHandle, &event);
  }
  else {
    // SimToReco associations
    Handle<SimToRecoCollection> simToTrkMuHandle;
    event.getByLabel(trkMuAssocLabel_, simToTrkMuHandle);
    simToTrkMuColl = *(simToTrkMuHandle.product());

    Handle<SimToRecoCollection> simToStaMuHandle;
    event.getByLabel(staMuAssocLabel_, simToStaMuHandle);
    simToStaMuColl = *(simToStaMuHandle.product());

    Handle<SimToRecoCollection> simToGlbMuHandle;
    event.getByLabel(glbMuAssocLabel_, simToGlbMuHandle);
    simToGlbMuColl = *(simToGlbMuHandle.product());

    // RecoToSim associations
    Handle<RecoToSimCollection> trkMuToSimHandle;
    event.getByLabel(trkMuAssocLabel_, trkMuToSimHandle);
    trkMuToSimColl = *(trkMuToSimHandle.product());

    Handle<RecoToSimCollection> staMuToSimHandle;
    event.getByLabel(staMuAssocLabel_, staMuToSimHandle);
    staMuToSimColl = *(staMuToSimHandle.product());

    Handle<RecoToSimCollection> glbMuToSimHandle;
    event.getByLabel(glbMuAssocLabel_, glbMuToSimHandle);
    glbMuToSimColl = *(glbMuToSimHandle.product());
  }

  const TrackingParticleCollection::size_type nSim = simColl.size();
  commonME_->hNSim_->Fill(nSim);

  commonME_->hNMuon_->Fill(muonColl.size());

  trkMuME_->hNTrks_->Fill(trkMuColl.size());
  staMuME_->hNTrks_->Fill(staMuColl.size());
  glbMuME_->hNTrks_->Fill(glbMuColl.size());

  // Analyzer reco::Muon
  for(View<Muon>::const_iterator iMuon = muonColl.begin();
      iMuon != muonColl.end(); ++iMuon) {
    int trkNTrackerHits = 0, glbNTrackerHits = 0;
    int staNMuonHits = 0, glbNMuonHits = 0;

    if ( iMuon->isTrackerMuon() ) {
      const TrackRef trkTrack = iMuon->track();

      trkNTrackerHits = countTrackerHits(*trkTrack);

      trkMuME_->hNTrackerHits_->Fill(trkNTrackerHits);
      trkMuME_->hNTrackerHits_vs_Pt_->Fill(trkTrack->pt(), trkNTrackerHits);
      trkMuME_->hNTrackerHits_vs_Eta_->Fill(trkTrack->eta(), trkNTrackerHits);
    }

    if ( iMuon->isStandAloneMuon() ) {
      const TrackRef staTrack = iMuon->standAloneMuon();

      staNMuonHits = countMuonHits(*staTrack);

      staMuME_->hNMuonHits_->Fill(staNMuonHits);
      staMuME_->hNMuonHits_vs_Pt_->Fill(staTrack->pt(), staNMuonHits);
      staMuME_->hNMuonHits_vs_Eta_->Fill(staTrack->eta(), staNMuonHits);

      staMuME_->hNTrksEta_->Fill(staTrack->eta());
      staMuME_->hNTrksPt_->Fill(staTrack->pt());
      
    }

    if ( iMuon->isGlobalMuon() ) {
      const TrackRef glbTrack = iMuon->combinedMuon();

      glbNTrackerHits = countTrackerHits(*glbTrack);
      glbNMuonHits = countMuonHits(*glbTrack);

      glbMuME_->hNTrackerHits_->Fill(glbNTrackerHits);
      glbMuME_->hNTrackerHits_vs_Pt_->Fill(glbTrack->pt(), glbNTrackerHits);
      glbMuME_->hNTrackerHits_vs_Eta_->Fill(glbTrack->eta(), glbNTrackerHits);

      glbMuME_->hNMuonHits_->Fill(glbNMuonHits);
      glbMuME_->hNMuonHits_vs_Pt_->Fill(glbTrack->pt(), glbNMuonHits);
      glbMuME_->hNMuonHits_vs_Eta_->Fill(glbTrack->eta(), glbNMuonHits);

      glbMuME_->hNTrksEta_->Fill(glbTrack->eta());
      glbMuME_->hNTrksPt_->Fill(glbTrack->pt());
      
      commonME_->hTrkToGlbDiffNTrackerHitsEta_->Fill(glbTrack->eta(),trkNTrackerHits-glbNTrackerHits);
      commonME_->hStaToGlbDiffNMuonHitsEta_->Fill(glbTrack->eta(),staNMuonHits-glbNMuonHits);
      
      commonME_->hTrkToGlbDiffNTrackerHitsPt_->Fill(glbTrack->pt(),trkNTrackerHits-glbNTrackerHits);
      commonME_->hStaToGlbDiffNMuonHitsPt_->Fill(glbTrack->pt(),staNMuonHits-glbNMuonHits);
      
      commonME_->hTrkToGlbDiffNTrackerHits_->Fill(trkNTrackerHits-glbNTrackerHits);
      commonME_->hStaToGlbDiffNMuonHits_->Fill(staNMuonHits-glbNMuonHits);

    }

  }

  // Analyzer reco::Track
  for(TrackingParticleCollection::size_type i=0; i<nSim; i++) {
    TrackingParticleRef simRef(simHandle, i);
    const TrackingParticle* simTP = simRef.get();
    if ( ! tpSelector_(*simTP) ) continue;

    const double simP   = simRef->p();
    const double simPt  = simRef->pt();
    const double simEta = doAbsEta_ ? fabs(simRef->eta()) : simRef->eta();
    const double simPhi = simRef->phi();

    const unsigned int nSimHits = simRef->pSimHit_end() - simRef->pSimHit_begin();

    commonME_->hSimP_  ->Fill(simP  );
    commonME_->hSimPt_ ->Fill(simPt );
    commonME_->hSimEta_->Fill(simEta);
    commonME_->hSimPhi_->Fill(simPhi);

    commonME_->hNSimHits_->Fill(nSimHits);

    // Get sim-reco association for a simRef
    vector<pair<RefToBase<Track>, double> > trkMuRefV, staMuRefV, glbMuRefV;
    if ( simToTrkMuColl.find(simRef) != simToTrkMuColl.end() ) {
      trkMuRefV = simToTrkMuColl[simRef];

      trkMuME_->hNSimToReco_->Fill(trkMuRefV.size());
      if ( ! trkMuRefV.empty() ) {
        const Track* trkMuTrack = trkMuRefV.begin()->first.get();
//        const double assocQuality = trkMuRefV.begin()->second;

        trkMuME_->fill(simTP, trkMuTrack);
      }
    }

    if ( simToStaMuColl.find(simRef) != simToStaMuColl.end() ) {
      staMuRefV = simToStaMuColl[simRef];

      staMuME_->hNSimToReco_->Fill(staMuRefV.size());
      if ( ! staMuRefV.empty() ) {
        const Track* staMuTrack = staMuRefV.begin()->first.get();
//        const double assocQuality = staMuRefV.begin().second;

        staMuME_->fill(simTP, staMuTrack);
      }
    }

    if ( simToGlbMuColl.find(simRef) != simToGlbMuColl.end() ) {
      glbMuRefV = simToGlbMuColl[simRef];

      glbMuME_->hNSimToReco_->Fill(glbMuRefV.size());
      if ( ! glbMuRefV.empty() ) {
        const Track* glbMuTrack = glbMuRefV.begin()->first.get();
//        const double assocQuality = glbMuRefV.begin().second;

        glbMuME_->fill(simTP, glbMuTrack);
      }
    }
  }
}

int
RecoMuonValidator::countMuonHits(const reco::Track& track) const {
  TransientTrackingRecHit::ConstRecHitContainer result;
  
  int count = 0;

  for (trackingRecHit_iterator hit = track.recHitsBegin(); hit != track.recHitsEnd(); ++hit) {
    if((*hit)->isValid()) {
      DetId recoid = (*hit)->geographicalId();
      if ( recoid.det() == DetId::Muon ) count++;
    }
  }
  return count;
}

int
RecoMuonValidator::countTrackerHits(const reco::Track& track) const {
  TransientTrackingRecHit::ConstRecHitContainer result;
  
  int count = 0;

  for (trackingRecHit_iterator hit = track.recHitsBegin(); hit != track.recHitsEnd(); ++hit) {
    if((*hit)->isValid()) {
      DetId recoid = (*hit)->geographicalId();
      if ( recoid.det() == DetId::Tracker ) count++;
    }
  }
  return count;
}

/* vim:set ts=2 sts=2 sw=2 expandtab: */
