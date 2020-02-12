#ifndef Validation_RecoMuon_RecoMuonValidator_H
#define Validation_RecoMuon_RecoMuonValidator_H

#include "DQMServices/Core/interface/DQMOneEDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "SimTracker/Common/interface/TrackingParticleSelector.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"

#include "SimDataFormats/Associations/interface/MuonToTrackingParticleAssociator.h"

// for selection cut
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

class TrackAssociatorBase;

class RecoMuonValidator : public DQMOneEDAnalyzer<> {
public:
  RecoMuonValidator(const edm::ParameterSet& pset);
  ~RecoMuonValidator() override;

  void dqmBeginRun(const edm::Run&, const edm::EventSetup& eventSetup) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;
  void dqmEndRun(edm::Run const&, edm::EventSetup const&) override;
  void analyze(const edm::Event& event, const edm::EventSetup& eventSetup) override;
  virtual int countMuonHits(const reco::Track& track) const;
  virtual int countTrackerHits(const reco::Track& track) const;

protected:
  unsigned int verbose_;

  edm::InputTag simLabel_;
  edm::InputTag muonLabel_;
  std::string muonSelection_;
  edm::EDGetTokenT<TrackingParticleCollection> simToken_;
  edm::EDGetTokenT<edm::View<reco::Muon> > muonToken_;

  edm::InputTag muAssocLabel_;
  edm::EDGetTokenT<reco::MuonToTrackingParticleAssociator> muAssocToken_;

  edm::InputTag beamspotLabel_;
  edm::InputTag primvertexLabel_;
  edm::EDGetTokenT<reco::BeamSpot> beamspotToken_;
  edm::EDGetTokenT<reco::VertexCollection> primvertexToken_;

  std::string outputFileName_;
  std::string subDir_;
  std::string subsystemname_;
  edm::ParameterSet pset;

  DQMStore* dbe_;

  bool doAbsEta_;
  bool doAssoc_;
  bool usePFMuon_;

  TrackingParticleSelector tpSelector_;

  // Track to use
  reco::MuonTrackType trackType_;

  struct MuonME;
  MuonME* muonME_;

  struct CommonME;
  CommonME* commonME_;

  //
  //struct for histogram dimensions
  //
  struct HistoDimensions {
    //p
    unsigned int nBinP;
    double minP, maxP;
    //pt
    unsigned int nBinPt;
    double minPt, maxPt;
    //if abs eta
    bool doAbsEta;
    //eta
    unsigned int nBinEta;
    double minEta, maxEta;
    //phi
    unsigned int nBinPhi;
    double minPhi, maxPhi;
    //dxy
    unsigned int nBinDxy;
    double minDxy, maxDxy;
    //dz
    unsigned int nBinDz;
    double minDz, maxDz;
    //pulls
    unsigned int nBinPull;
    double wPull;
    //resolustions
    unsigned int nBinErr;
    double minErrP, maxErrP;
    double minErrPt, maxErrPt;
    double minErrQPt, maxErrQPt;
    double minErrEta, maxErrEta;
    double minErrPhi, maxErrPhi;
    double minErrDxy, maxErrDxy;
    double minErrDz, maxErrDz;
    //track multiplicities
    unsigned int nTrks, nAssoc;
    unsigned int nDof;
    // for PF muons
    bool usePFMuon;
  };

  HistoDimensions hDim;

private:
  StringCutObjectSelector<reco::Muon> selector_;
  bool wantTightMuon_;
};

#endif
/* vim:set ts=2 sts=2 sw=2 expandtab: */
