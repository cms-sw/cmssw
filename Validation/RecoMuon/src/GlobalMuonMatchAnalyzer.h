#ifndef Validation_RecoMuon_GlobalMuonMatchAnalyzer_H
#define Validation_RecoMuon_GlobalMuonMatchAnalyzer_H

/** \class GlobalMuonMatchAnalyzer 
 *
 *
 *
 *
 *  \author Adam Everett        Purdue University
 */


// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Common/interface/View.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"


namespace reco {class Track;}

class InputTag;
class MonitorElement;
class TrackAssociatorBase;
class  DQMStore;

//
// class decleration
//

class GlobalMuonMatchAnalyzer : public DQMEDAnalyzer {
   public:
      explicit GlobalMuonMatchAnalyzer(const edm::ParameterSet&);
      ~GlobalMuonMatchAnalyzer();


   private:
      virtual void beginJob() ;
      //      virtual void beginRun(const edm::Run&, const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      void bookHistograms(DQMStore::IBooker &, edm::Run const &, edm::EventSetup const &) override;
      virtual void endJob() ;
      virtual void endRun() ;

  void computeEfficiencyEta(MonitorElement*, MonitorElement *recoTH2, MonitorElement *simTH2);
  void computeEfficiencyPt(MonitorElement*, MonitorElement *recoTH2, MonitorElement *simTH2);
      // ----------member data ---------------------------
  std::string out;
  DQMStore* dbe_;
  edm::ParameterSet iConfig;
  std::string subsystemname_;
  MonitorElement *h_shouldMatch, *h_goodMatchSim, *h_tkOnlySim, *h_staOnlySim;
  MonitorElement *h_totReco, *h_goodMatch, *h_fakeMatch;
  MonitorElement *h_effic, *h_efficPt;
  MonitorElement *h_fake, *h_fakePt;

  const TrackAssociatorBase *tkAssociator_, *muAssociator_;
  edm::InputTag tkAssociatorName_, muAssociatorName_;
  edm::InputTag tkName_, tpName_, glbName_, staName_;
  edm::EDGetTokenT<edm::View<reco::Track> >  tkToken_, tpToken_, glbToken_, staToken_;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> tkAssociatorToken_, muAssociatorToken_;

};

#endif
