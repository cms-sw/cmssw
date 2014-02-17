#ifndef Validation_RecoMuon_GlobalMuonMatchAnalyzer_H
#define Validation_RecoMuon_GlobalMuonMatchAnalyzer_H

/** \class GlobalMuonMatchAnalyzer 
 *
 *
 *
 *  $Date: 2009/10/31 05:17:36 $
 *  $Revision: 1.5 $
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

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

namespace reco {class Track;}

class InputTag;
class MonitorElement;
class TrackAssociatorBase;
class  DQMStore;

//
// class decleration
//

class GlobalMuonMatchAnalyzer : public edm::EDAnalyzer {
   public:
      explicit GlobalMuonMatchAnalyzer(const edm::ParameterSet&);
      ~GlobalMuonMatchAnalyzer();


   private:
      virtual void beginJob() ;
      virtual void beginRun(const edm::Run&, const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

  void computeEfficiencyEta(MonitorElement*, MonitorElement *recoTH2, MonitorElement *simTH2);
  void computeEfficiencyPt(MonitorElement*, MonitorElement *recoTH2, MonitorElement *simTH2);
      // ----------member data ---------------------------
  std::string out;
  DQMStore* dbe_;

  MonitorElement *h_shouldMatch, *h_goodMatchSim, *h_tkOnlySim, *h_staOnlySim;
  MonitorElement *h_totReco, *h_goodMatch, *h_fakeMatch;
  MonitorElement *h_effic, *h_efficPt;
  MonitorElement *h_fake, *h_fakePt;

  const TrackAssociatorBase *tkAssociator_, *muAssociator_;
  std::string tkAssociatorName_, muAssociatorName_;
  edm::InputTag tkName_, tpName_, glbName_, staName_;


};

#endif
