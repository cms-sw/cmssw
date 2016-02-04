#ifndef Validation_MuonDTDigis_h
#define Validation_MuonDTDigis_h

/** \class MuonDTDigis
 *  Analyse the the muon-drift-tubes digitizer. 
 *  
 *  $Date: 2009/12/14 22:24:47 $
 *  $Revision: 1.6 $
 *  \authors: R. Bellan
 */
#include <FWCore/Framework/interface/Event.h>
#include "FWCore/Framework/interface/EventSetup.h"
#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
  
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTGeometry/interface/DTGeometry.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
  
#include "DataFormats/DTDigi/interface/DTDigi.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"
  
#include "SimMuon/DTDigitizer/test/Histograms.h"

#include<vector>
#include "DQMServices/Core/interface/MonitorElement.h"

class TH1F;
class TFile;

class PSimHit;

class   hDigis;

namespace edm {
  class ParameterSet; class Event; class EventSetup;}

class MuonDTDigis : public edm::EDAnalyzer{
  
 public:
  // Constructor
  explicit MuonDTDigis(const edm::ParameterSet& pset);

  // Destructor
  virtual ~MuonDTDigis();

 protected:
  // Analysis
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);

  hDigis* WheelHistos(int wheel);
  
  // BeginJob
//void beginJob();

  // EndJob
//void endJob(void);


 private:
  typedef std::map<DTWireId, std::vector<const PSimHit*> > DTWireIdMap; 

  std::string SimHitLabel;
  std::string DigiLabel;
  std::string outputFile_;

  // Switch for debug output
  bool verbose_;

  // DaqMonitor element
  DQMStore* dbe_;

  // Monitor elements
  MonitorElement* meDigiTimeBox_;
  MonitorElement* meDigiTimeBox_wheel2m_;
  MonitorElement* meDigiTimeBox_wheel1m_;
  MonitorElement* meDigiTimeBox_wheel0_;
  MonitorElement* meDigiTimeBox_wheel1p_;
  MonitorElement* meDigiTimeBox_wheel2p_;
  MonitorElement* meDigiEfficiency_;
  MonitorElement* meDigiEfficiencyMu_;
  MonitorElement* meDoubleDigi_;
  MonitorElement* meSimvsDigi_;
  MonitorElement* meWire_DoubleDigi_; 

  MonitorElement* meMB1_sim_occup_;
  MonitorElement* meMB1_digi_occup_;
  MonitorElement* meMB2_sim_occup_;
  MonitorElement* meMB2_digi_occup_;
  MonitorElement* meMB3_sim_occup_;
  MonitorElement* meMB3_digi_occup_;
  MonitorElement* meMB4_sim_occup_;
  MonitorElement* meMB4_digi_occup_;
 
  std::vector<MonitorElement*> meDigiTimeBox_SL_;
  MonitorElement* meDigiHisto_;

  TH1F *hMuonDigis;
//  TH1F *DigiTimeBox;
//  TFile *file_more_plots;

  hDigis *hDigis_global;
  hDigis *hDigis_W0;
  hDigis *hDigis_W1; 
  hDigis *hDigis_W2;
  hHits *hAllHits;
 
};

#endif    
