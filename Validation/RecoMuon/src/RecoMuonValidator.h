#ifndef Validation_RecoMuon_RecoMuonValidator_H
#define Validation_RecoMuon_RecoMuonValidator_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "SimDataFormats/Track/interface/SimTrackContainer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"

//#include "DQMServices/Core/interface/DaqMonitorBEInterface.h"
//#include "DQMServices/Daemon/interface/MonitorDaemon.h"

#include <string>
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"

class RecoMuonValidator : public edm::EDAnalyzer 
{
 public:
  RecoMuonValidator(const edm::ParameterSet& pset);
  ~RecoMuonValidator();

  virtual void beginJob(const edm::EventSetup& eventSetup);
  virtual void endJob();
  virtual void analyze(const edm::Event& event, const edm::EventSetup& eventSetup);

 protected:
  reco::TrackCollection::const_iterator matchTrack(edm::SimTrackContainer::const_iterator simTrack,
                                                   edm::Handle<reco::TrackCollection>& recTracks);
 protected:
  // Histogram range and binning
  unsigned int nBinPt_, nBinEta_, nBinPhi_, nBinDPt_, nBinQPt_;
  //unsigned int nBinStaQPt_, nBinGlbQPt_;
  double minPt_ , maxPt_;
  double minEta_, maxEta_;
  double minPhi_, maxPhi_;
  double maxStaQPt_, maxGlbQPt_;
  double maxStaDPt_, maxGlbDPt_;

  //edm::InputTag recTrackLabel_;
  edm::InputTag staTrackLabel_;
  edm::InputTag glbTrackLabel_;
  edm::InputTag simTrackLabel_;

  //DQM Interface
  //DaqMonitorBEInterface* theDQM_;

  std::string outFileName_;
  TFile* outFile_;
  TH1F * hGenPt_ , * hSimPt_ , * hStaPt_ , * hGlbPt_ ;
  TH2F * hGenPhiVsEta_, * hSimPhiVsEta_, * hStaPhiVsEta_, * hGlbPhiVsEta_;
  TH2F * hStaEtaVsDeltaPt_, * hGlbEtaVsDeltaPt_;
  TH2F * hStaEtaVsResolPt_, * hGlbEtaVsResolPt_;
};

#endif
