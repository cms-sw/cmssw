#ifndef SimMuon_DTDigiAnalyzer_h
#define SimMuon_DTDigiAnalyzer_h

/** \class DTDigiAnalyzer
 *  Analyse the the muon-drift-tubes digitizer. 
 *  
 *  $Date: 2007/05/11 14:44:38 $
 *  $Revision: 1.4 $
 *  \authors: R. Bellan
 */

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include <DataFormats/MuonDetId/interface/DTWireId.h>

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
  virtual ~DTDigiAnalyzer();
  void endJob();
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);
  hDigis* WheelHistos(int wheel);
  
 private:
  typedef std::map<DTWireId, std::vector<const PSimHit*> > DTWireIdMap; 

  TH1F *DigiTimeBox;
  TFile *file;
  std::string label;
  //  DTMCStatistics        *MCStatistics;
  // DTMuonDigiStatistics  *MuonDigiStatistics;
  // DTHitsAnalysis        *HitsAnalysis;
  
};

#endif    
