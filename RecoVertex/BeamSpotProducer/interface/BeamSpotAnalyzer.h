#ifndef BeamSpotProducer_BeamSpotAnalyzer_h
#define BeamSpotProducer_BeamSpotAnalyzer_h

/**_________________________________________________________________
   class:   BeamSpotAnalyzer.h
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)


________________________________________________________________**/


// C++ standard
#include <string>
// CMS
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoVertex/BeamSpotProducer/interface/BeamFitter.h"


class BeamSpotAnalyzer : public edm::EDAnalyzer {
 public:
  explicit BeamSpotAnalyzer(const edm::ParameterSet&);
  ~BeamSpotAnalyzer() override;

 private:
  void beginJob() override ;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override ;
  void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
									const edm::EventSetup& context) override ;
  void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
								  const edm::EventSetup& c) override;

  int    ftotalevents;
  int fitNLumi_;
  int resetFitNLumi_;
  // int countEvt_;       //counter
  int countLumi_;      //counter
  int Org_resetFitNLumi_;
  int previousLumi_;
  int previousRun_;
  int ftmprun0, ftmprun;
  int beginLumiOfBSFit_;
  int endLumiOfBSFit_;
  std::time_t refBStime[2];

  bool write2DB_;
  bool runbeamwidthfit_;
  bool runallfitters_;
  //  double inputBeamWidth_;

  BeamFitter * theBeamFitter;
};

#endif
