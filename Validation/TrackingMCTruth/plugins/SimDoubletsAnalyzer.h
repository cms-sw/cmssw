#ifndef Validation_TrackingMCTruth_SimDoubletsAnalyzer_h
#define Validation_TrackingMCTruth_SimDoubletsAnalyzer_h

// Package:    Validation/TrackingMCTruth
// Class:      SimDoubletsAnalyzer
//
/**\class SimDoubletsAnalyzer SimDoubletsAnalyzer.cc Validation/TrackingMCTruth/plugins/SimDoubletsAnalyzer.cc

 Description: DQM analyzer for true RecHit doublets (SimDoublets) of the inner tracker

 Implementation:
    This analyzer takes as input collection SimDoublets produced by the SimDoubletsProducer. Like the producer,
    you have two versions, one for Phase 1 and one for Phase 2.
    The analyzer's main purpose is to study the goodness of the pixel RecHit doublet creation in the patatrack
    pixel track reconstruction. This depends on a bunch of cuts, meaning a doublet of two RecHits is only formed
    if it passes all of those cuts. The SimDoublets represent the true and therefore ideal doublets of 
    simulated TrackingParticles (Note that in the SimDoublet production you may apply selections on the TPs).
    
    For this study, the analyzer produces histograms for the variables that are cut on during the real 
    reconstruction. Each distribution is produced twice:
     1. for all SimDoublets (name = `{variablename}`)
     2. for those SimDoublets which actually pass all selection cuts (name = `pass_{variablename}`)
    The cut values can be passed in the configuration of the analyzer and should default to the ones used in 
    reconstruction. The distributions of the cut variables can be found in the folder:
        SimDoublets/cutParameters/
    In there, you have one subfolder global/ where you find the cut parameters which are set globally for the 
    entire pixel detector. Additionally, you have for each configured layer pair a subfolder with all layer-pair-
    dependent cut parameters. Those are labeled `lp_{innerLayerId}_{outerLayerId}`.

    Besides this, also general distributions can be found in 
        SimDoublets/general/
    These are e.g. histograms for the number of SimDoublets per layer pair or SimDoublet efficiencies depending 
    on the cut values set in the configuration.

*/
//
// Original Author:  Luca Ferragina, Elena Vernazza, Jan Schulz
//         Created:  Thu, 16 Jan 2025 13:46:21 GMT
//
//

// includes
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "SimDataFormats/TrackingAnalysis/interface/SimDoublets.h"

#include <vector>
#include <string>
#include <map>

// -------------------------------------------------------------------------------------------------------------
// class declaration
// -------------------------------------------------------------------------------------------------------------

template <typename TrackerTraits>
class SimDoubletsAnalyzer : public DQMEDAnalyzer {
public:
  explicit SimDoubletsAnalyzer(const edm::ParameterSet&);
  ~SimDoubletsAnalyzer() override;

  void dqmBeginRun(const edm::Run& iRun, const edm::EventSetup& iSetup) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void bookHistograms(DQMStore::IBooker&, edm::Run const&, edm::EventSetup const&) override;

  void analyze(const edm::Event&, const edm::EventSetup&) override;

  // ------------ member data ------------

  const TrackerTopology* trackerTopology_ = nullptr;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> topology_getToken_;
  const edm::EDGetTokenT<SimDoubletsCollection> simDoublets_getToken_;

  // map that takes the layerPairId as defined in the SimDoublets
  // and gives the position of the histogram in the histogram vector
  std::map<int, int> layerPairId2Index_;

  // cutting parameters
  std::vector<double> cellMinz_;
  std::vector<double> cellMaxz_;
  std::vector<int> cellPhiCuts_;
  std::vector<double> cellMaxr_;
  int cellMinYSizeB1_;
  int cellMinYSizeB2_;
  int cellMaxDYSize12_;
  int cellMaxDYSize_;
  int cellMaxDYPred_;
  double cellZ0Cut_;
  double cellPtCut_;

  std::string folder_;  // main folder in the DQM file

  // monitor elements
  // profiles to be filled
  MonitorElement* h_effSimDoubletsPerTPVsPt_;
  MonitorElement* h_effSimDoubletsPerTPVsEta_;
  // histograms to be filled
  MonitorElement* h_layerPairs_;
  MonitorElement* h_numSkippedLayers_;
  MonitorElement* h_numSimDoubletsPerTrackingParticle_;
  MonitorElement* h_numLayersPerTrackingParticle_;
  MonitorElement* h_numTPVsPt_;
  MonitorElement* h_pass_numTPVsPt_;
  MonitorElement* h_numTPVsEta_;
  MonitorElement* h_pass_numTPVsEta_;
  MonitorElement* h_numVsPt_;
  MonitorElement* h_pass_numVsPt_;
  MonitorElement* h_numVsEta_;
  MonitorElement* h_pass_numVsEta_;
  MonitorElement* h_z0_;
  MonitorElement* h_curvatureR_;
  MonitorElement* h_pTFromR_;
  MonitorElement* h_YsizeB1_;
  MonitorElement* h_YsizeB2_;
  MonitorElement* h_DYsize12_;
  MonitorElement* h_DYsize_;
  MonitorElement* h_DYPred_;
  MonitorElement* h_pass_layerPairs_;
  MonitorElement* h_pass_z0_;
  MonitorElement* h_pass_pTFromR_;
  MonitorElement* h_pass_YsizeB1_;
  MonitorElement* h_pass_YsizeB2_;
  MonitorElement* h_pass_DYsize12_;
  MonitorElement* h_pass_DYsize_;
  MonitorElement* h_pass_DYPred_;
  // vectors of histograms (one hist per layer pair)
  std::vector<MonitorElement*> hVector_dr_;
  std::vector<MonitorElement*> hVector_dphi_;
  std::vector<MonitorElement*> hVector_idphi_;
  std::vector<MonitorElement*> hVector_innerZ_;
  std::vector<MonitorElement*> hVector_Ysize_;
  std::vector<MonitorElement*> hVector_DYsize_;
  std::vector<MonitorElement*> hVector_DYPred_;
  std::vector<MonitorElement*> hVector_pass_dr_;
  std::vector<MonitorElement*> hVector_pass_idphi_;
  std::vector<MonitorElement*> hVector_pass_innerZ_;
};

#endif