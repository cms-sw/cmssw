#ifndef METTESTER_H
#define METTESTER_H

// author: Mike Schmitt (The University of Florida)
// date: 8/24/2006
// modification: Bobby Scurlock
// date: 03.11.2006
// note: added RMS(METx) vs SumET capability
// modification: Rick Cavanaugh
// date: 05.11.2006
// note: added configuration parameters
// modification: Mike Schmitt
// date: 02.28.2007
// note: code rewrite

#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/METCollection.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonMETCorrectionData.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include <string>
#include <map>
#include "DQMServices/Core/interface/MonitorElement.h"

namespace reco {class BeamSpot;}

class METTester: public edm::EDAnalyzer {
public:

  explicit METTester(const edm::ParameterSet&);

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  //virtual void beginJob() ;
  virtual void beginRun(const edm::Run&, const edm::EventSetup&) ;
  //virtual void beginJob() ;
  virtual void endJob() ;
  //  virtual void endRun(const edm::Run&, const edm::EventSetup&);
  void FillpfMETRes();


private:

  // DAQ Tools
  DQMStore* dbe_;
  std::map<std::string, MonitorElement*> me;

  // Inputs from Configuration File
  std::string METType_;
  std::string FolderName_;
  std::string sample_;
  edm::EDGetTokenT<reco::GenMETCollection>  inputGenMETToken_;
  edm::EDGetTokenT<reco::METCollection>     inputMETToken_;
  edm::EDGetTokenT<reco::CaloMETCollection> inputMETToken2_;
  edm::EDGetTokenT<reco::CaloMETCollection> inputCaloMETToken_;
  edm::EDGetTokenT<reco::PFMETCollection>   inputPFMETToken_;
  edm::EDGetTokenT<reco::TrackCollection>   inputTrackToken_;
  edm::EDGetTokenT<reco::MuonCollection>    inputMuonToken_;
  edm::EDGetTokenT<reco::MuonCollection>    inputMuonFixedToken_;
  edm::EDGetTokenT<edm::View<reco::GsfElectron> > inputElectronToken_;
  edm::EDGetTokenT<reco::BeamSpot>          inputBeamSpotToken_;
  edm::EDGetTokenT<reco::VertexCollection> offline_pvToken_;
  edm::EDGetTokenT<reco::GenMETCollection> genTrue_Token_;
  edm::EDGetTokenT<reco::GenMETCollection> genCalo_Token_;
  edm::EDGetTokenT<edm::ValueMap<reco::MuonMETCorrectionData> > tcMet_ValueMap_Token_;
  edm::EDGetTokenT<edm::ValueMap<reco::MuonMETCorrectionData> > muon_ValueMap_Token_;

  bool finebinning_;

  bool isGoodTrack( const reco::TrackRef, float d0corr );

  int minhits_;
  double maxd0_;
  double maxchi2_;
  double maxeta_;
  double maxpt_;
  double maxPtErr_;
  std::vector<int> trkQuality_;
  std::vector<int> trkAlgos_;
};

#endif // METTESTER_H
