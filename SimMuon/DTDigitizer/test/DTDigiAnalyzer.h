#ifndef SimMuon_DTDigiAnalyzer_h
#define SimMuon_DTDigiAnalyzer_h

/** \class DTDigiAnalyzer
 *  Analyse the the muon-drift-tubes digitizer. 
 *  
 *  \authors: R. Bellan
 */

#include "FWCore/Framework/interface/ConsumesCollector.h"

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <DataFormats/MuonDetId/interface/DTWireId.h>
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "SimMuon/DTDigitizer/test/Histograms.h"

class TH1F;
class TFile;
class PSimHit;
//class DTMCStatistics;
//class DTMuonDigiStatistics;
//class DTHitsAnalysis;
class   hDigis;

namespace edm {
  class ParameterSet; class Event; class EventSetup;}

class DTDigiAnalyzer : public edm::EDAnalyzer{
  
 public:
  explicit DTDigiAnalyzer(const edm::ParameterSet& pset);
  ~DTDigiAnalyzer() override;
  void endJob() override;
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup) override;
  hDigis* WheelHistos(int wheel);
  
 private:
  typedef std::map<DTWireId, std::vector<const PSimHit*> > DTWireIdMap; 

  TH1F *DigiTimeBox;
  TFile *file;
  std::string label;
  //  DTMCStatistics        *MCStatistics;
  // DTMuonDigiStatistics  *MuonDigiStatistics;
  // DTHitsAnalysis        *HitsAnalysis;

  edm::EDGetTokenT< edm::PSimHitContainer > psim_token;
  edm::EDGetTokenT< DTDigiCollection > DTd_token;

  hDigis hDigis_global;
  hDigis hDigis_W0;
  hDigis hDigis_W1;
  hDigis hDigis_W2;

  hHits hAllHits;
  
};

#endif    
