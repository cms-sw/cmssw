#ifndef ElectronConversionRejectionValidator_H
#define ElectronConversionRejectionValidator_H
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "DataFormats/Common/interface/RefToBaseVector.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
//
//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"


//
#include <map>
#include <vector>

// forward declarations
class TFile;
class TH1F;
class TH2F;
class TProfile;
class TTree;
class SimVertex;
class SimTrack;
/** \class ElectronConversionRejectionValidator
 **  
 **
 **  $Id: ElectronConversionRejectionValidator
 **  $Date: 2011/07/29 03:00:20 $ 
 **  $Revision: 1.1 $
 **  \author J.Bendavid
 **
 ***/


class ElectronConversionRejectionValidator : public edm::EDAnalyzer
{

 public:
   
  //
  explicit ElectronConversionRejectionValidator( const edm::ParameterSet& ) ;
  virtual ~ElectronConversionRejectionValidator();
                                   
      
  virtual void analyze( const edm::Event&, const edm::EventSetup& ) ;
  virtual void beginJob();
  virtual void beginRun( edm::Run const & r, edm::EventSetup const & theEventSetup) ;
  virtual void endRun (edm::Run& r, edm::EventSetup const & es);
  virtual void endJob() ;
  
 private:
  //


      
  std::string fName_;
  DQMStore *dbe_;

  int verbosity_;
  int nEvt_;
  int nEntry_;


  edm::ParameterSet parameters_;
  std::string conversionCollectionProducer_;       
  std::string conversionCollection_;

  std::string gsfElectronCollectionProducer_;       
  std::string gsfElectronCollection_;

  std::string dqmpath_;

  edm::InputTag label_tp_;



  bool isRunCentrally_;

  float elePtMin_;
  int eleExpectedHitsInnerMax_;
  float eleD0Max_;
  
  //
  MonitorElement* h_elePtAll_;
  MonitorElement* h_eleEtaAll_;
  MonitorElement* h_elePhiAll_; 
  
  MonitorElement* h_elePtPass_;
  MonitorElement* h_eleEtaPass_;
  MonitorElement* h_elePhiPass_; 
  
  MonitorElement* h_elePtFail_;
  MonitorElement* h_eleEtaFail_;  
  MonitorElement* h_elePhiFail_;
  
  MonitorElement* h_elePtEff_;
  MonitorElement* h_eleEtaEff_;
  MonitorElement* h_elePhiEff_;   
  
  MonitorElement* h_convPt_;
  MonitorElement* h_convEta_;  
  MonitorElement* h_convPhi_; 
  MonitorElement* h_convRho_;  
  MonitorElement* h_convZ_;  
  
  MonitorElement* h_convProb_;    

  MonitorElement* h_convLeadTrackpt_;  
  MonitorElement* h_convTrailTrackpt_;  
  MonitorElement* h_convLog10TrailTrackpt_;  
  MonitorElement* h_convLeadTrackAlgo_;    
  MonitorElement* h_convTrailTrackAlgo_;    
  
};




#endif
