#ifndef SimMuon_DTDigiAnalyzer_h
#define SimMuon_DTDigiAnalyzer_h

/** \class DTDigiAnalyzer
 *  Analyse the the muon-drift-tubes digitizer. 
 *  
 *  $Date: 2006/02/07 19:12:38 $
 *  $Revision: 1.2 $
 *  \authors: R. Bellan
 */

#include <FWCore/Framework/interface/EDAnalyzer.h>
#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"


class TH1F;
class TFile;
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
  void analyze(const edm::Event & event, const edm::EventSetup& eventSetup);
  hDigis* WheelHistos(int wheel);
  
 private:
  TH1F *DigiTimeBox;
  TFile *file;

  //  DTMCStatistics        *MCStatistics;
  // DTMuonDigiStatistics  *MuonDigiStatistics;
  // DTHitsAnalysis        *HitsAnalysis;
  
};

DEFINE_FWK_MODULE(DTDigiAnalyzer)
#endif    
