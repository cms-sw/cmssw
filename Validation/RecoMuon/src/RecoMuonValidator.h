#ifndef Validation_RecoMuon_RecoMuonValidator_H
#define Validation_RecoMuon_RecoMuonValidator_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "CommonTools/RecoAlgos/interface/TrackingParticleSelector.h"
#include "DataFormats/TrackReco/interface/Track.h"

class DQMStore;
class MonitorElement;
class MuonServiceProxy;
class TrackAssociatorBase;

class RecoMuonValidator : public edm::EDAnalyzer
{
 public:
  RecoMuonValidator(const edm::ParameterSet& pset);
  ~RecoMuonValidator();
  
  virtual void beginRun(const edm::Run&, const edm::EventSetup& eventSetup);
  virtual void endRun();
  virtual void analyze(const edm::Event& event, const edm::EventSetup& eventSetup);
  virtual int countMuonHits(const reco::Track& track) const;
  virtual int countTrackerHits(const reco::Track& track) const;

 protected:
  unsigned int verbose_;

  edm::InputTag simLabel_;
  edm::InputTag trkMuLabel_;
  edm::InputTag staMuLabel_;
  edm::InputTag glbMuLabel_;
  edm::InputTag muonLabel_;

  edm::InputTag trkMuAssocLabel_;
  edm::InputTag staMuAssocLabel_;
  edm::InputTag glbMuAssocLabel_;
  
  std::string outputFileName_;
  std::string subDir_;

  MuonServiceProxy * theMuonService;
  DQMStore * theDQM;
  
  bool doAbsEta_;
  bool doAssoc_;

  TrackingParticleSelector tpSelector_;
  TrackAssociatorBase* trkMuAssociator_, * staMuAssociator_, * glbMuAssociator_;

  struct MuonME;
  MuonME * trkMuME_, * staMuME_, * glbMuME_;

  struct CommonME;
  CommonME * commonME_;
};

#endif
/* vim:set ts=2 sts=2 sw=2 expandtab: */
