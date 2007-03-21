#ifndef Validation_RecoMuon_RecoMuonValidator_H
#define Validation_RecoMuon_RecoMuonValidator_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

#include <utility>

#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>

class HResolution
{
 public:
  HResolution(std::string name,
              int nBinErrQPt, double widthErrQPt,
              int nBinPull, double widthPull,
              int nBinEta, double minEta, double maxEta,
              int nBinPhi, double minPhi, double maxPhi)
  {
    hEtaVsErrQPt_  = new TH2F((name+"EtaVsErrQPt").c_str(), (name+" #eta vs #sigma(q/p_{T})").c_str(),
                              nBinEta, minEta, maxEta, nBinErrQPt, -widthErrQPt, widthErrQPt);
    hEtaVsPullPt_  = new TH2F((name+"EtaVsPullPt").c_str(), (name+" #eta vs Pull p_{T}").c_str(),
                              nBinEta, minEta, maxEta, nBinPull, -widthPull, widthPull);
    hPhiVsPullPt_  = new TH2F((name+"PhiVsPullPt").c_str(), (name+" #phi vs Pull p_{T}").c_str(),
                              nBinPhi, minPhi, maxPhi, nBinPull, -widthPull, widthPull);
    hEtaVsPullEta_ = new TH2F((name+"EtaVsPullEta").c_str(), (name+" #eta vs Pull #eta").c_str(),
                              nBinEta, minEta, maxEta, nBinPull, -widthPull, widthPull);
    hPhiVsPullEta_ = new TH2F((name+"PhiVsPullEta").c_str(), (name+" #phi vs Pull #eta").c_str(),
                              nBinPhi, minPhi, maxPhi, nBinPull, -widthPull, widthPull);
    hEtaVsPullPhi_ = new TH2F((name+"EtaVsPullPhi").c_str(), (name+" #eta vs Pull #phi").c_str(),
                              nBinEta, minEta, maxEta, nBinPull, -widthPull, widthPull);
    hPhiVsPullPhi_ = new TH2F((name+"PhiVsPullPhi").c_str(), (name+" #phi vs Pull #phi").c_str(),
                              nBinPhi, minPhi, maxPhi, nBinPull, -widthPull, widthPull);
    hMisQAboutEta_ = new TH1F((name+"MisQAboutEta").c_str(), (name+" mischarge about #eta").c_str(),
                              nBinEta, minEta, maxEta);
  };
  ~HResolution()
  {
//    delete hEtaVsErrQPt_ ;
//    delete hEtaVsPullPt_ ;
//    delete hPhiVsPullPt_ ;
//    delete hEtaVsPullEta_;
//    delete hPhiVsPullEta_;
//    delete hEtaVsPullPhi_;
//    delete hPhiVsPullPhi_;
//    delete hMisQAboutEta_;
  };
  void write()
  {
    hEtaVsErrQPt_ ->Write();
    hEtaVsPullPt_ ->Write();
    hPhiVsPullPt_ ->Write();
    hEtaVsPullEta_->Write();
    hPhiVsPullEta_->Write();
    hEtaVsPullPhi_->Write();
    hPhiVsPullPhi_->Write();
    hMisQAboutEta_->Write();
  };
  void fillInfo(const SimTrack& simTrack, const TrajectoryStateOnSurface& tsos)
  {
    const TrackCharge simQ = static_cast<TrackCharge>(simTrack.charge());
    const double simPt  = simTrack.momentum().perp();
    const double simEta = simTrack.momentum().eta();
    const double simPhi = simTrack.momentum().phi();
    const double simQPt = simQ/simPt;

    const TrackCharge recQ = static_cast<TrackCharge>(tsos.charge());
    const double recPt  = tsos.globalMomentum().perp();
    const double recEta = tsos.globalMomentum().eta();
    const double recPhi = tsos.globalMomentum().phi();
    const double recQPt = recQ/recPt;

    AlgebraicSymMatrix cartErr = tsos.cartesianError().matrix();
    const double px = tsos.globalMomentum().x();
    const double py = tsos.globalMomentum().y();
    //const double pz = tsos.globalMomentum().z();
    const double errPt  = sqrt(cartErr[3][3]*px*px+cartErr[4][4]*py*py)/tsos.globalMomentum().perp();
    //const double errP   = sqrt(cartErr[3][3]*px*px+cartErr[4][4]*py*py+cartErr[5][5]*pz*pz)/tsos.globalMomentum().mag();
    AlgebraicSymMatrix curvErr = tsos.curvilinearError().matrix();
    const double errEta = sqrt(curvErr[1][1])*fabs(sin(tsos.globalMomentum().theta()));
    const double errPhi = sqrt(curvErr[2][2]);
    hEtaVsErrQPt_ ->Fill(simEta, (recQPt-simQPt)/simQPt);
    hEtaVsPullPt_ ->Fill(simEta, (recPt-simPt)/errPt);
    hPhiVsPullPt_ ->Fill(simPhi, (recPt-simPt)/errPt);
    hEtaVsPullEta_->Fill(simEta, (recEta-simEta)/errEta);
    hPhiVsPullEta_->Fill(simPhi, (recEta-simEta)/errEta);
    hEtaVsPullPhi_->Fill(simEta, (recPhi-simPhi)/errPhi);
    hPhiVsPullPhi_->Fill(simPhi, (recPhi-simPhi)/errPhi);
    if ( simQ != recQ ) hMisQAboutEta_->Fill(simEta);
  };
 protected:
  TH2F * hEtaVsErrQPt_ ;
  TH2F * hEtaVsPullPt_ , * hPhiVsPullPt_ ;
  TH2F * hEtaVsPullEta_, * hPhiVsPullEta_;
  TH2F * hEtaVsPullPhi_, * hPhiVsPullPhi_;
  TH1F * hMisQAboutEta_;
};

class RecoMuonValidator : public edm::EDAnalyzer
{
 public:
  RecoMuonValidator(const edm::ParameterSet& pset);
  ~RecoMuonValidator();

  virtual void beginJob(const edm::EventSetup& eventSetup);
  virtual void endJob();
  virtual void analyze(const edm::Event& event, const edm::EventSetup& eventSetup);

 protected:
  typedef TrajectoryStateOnSurface TSOS;
  std::pair<TSOS, reco::TransientTrack> matchTrack(const SimTrack& simTrack,
                                             edm::Handle<reco::TrackCollection> recTrack);
  std::pair<TSOS, TrajectorySeed> matchTrack(const SimTrack& simTrack, 
                                             edm::Handle<TrajectorySeedCollection> seeds);
  TSOS getSeedTSOS(const TrajectorySeed& seed);
  int getNSimHits(const edm::Event& event, std::string simHitLabel, unsigned int trackId);

 protected:
  unsigned int nBinEta_, nBinPhi_;
  unsigned int nBinErrQPt_, nBinPull_;
  unsigned int nHits_;
  double minPt_ , maxPt_ ;
  double minEta_, maxEta_;
  double minPhi_, maxPhi_;
  double widthStaErrQPt_, widthGlbErrQPt_, widthSeedErrQPt_;
  double widthPull_;
  double staMinPt_, staMinRho_, staMinR_;
  double tkMinP_, tkMinPt_;

  edm::InputTag simTrackLabel_;
  edm::InputTag staTrackLabel_;
  edm::InputTag glbTrackLabel_;
  edm::InputTag tkTrackLabel_;
  edm::InputTag seedLabel_    ;

  std::string outputFileName_;
  TFile* outputFile_;

  TH2F * hSimEtaVsPhi_, * hStaEtaVsPhi_, * hGlbEtaVsPhi_, * hTkEtaVsPhi_, * hSeedEtaVsPhi_;
  TH2F * hEtaVsNDtSimHits_, * hEtaVsNCSCSimHits_, * hEtaVsNRPCSimHits_, * hEtaVsNSimHits_;
  TH2F * hSeedEtaVsNHits_, * hStaEtaVsNHits_, * hGlbEtaVsNHits_;
  HResolution * hStaResol_, * hGlbResol_, * hSeedResol_;

  MuonServiceProxy * theMuonService_;
  std::string seedPropagatorName_;
};

#endif

