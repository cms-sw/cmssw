#ifndef BeamSpotProducer_BeamSpotAnalyzer_h
#define BeamSpotProducer_BeamSpotAnalyzer_h

/**_________________________________________________________________
   class:   BeamSpotAnalyzer.h
   package: RecoVertex/BeamSpotProducer
   


 author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)

 version $Id: BeamSpotAnalyzer.h,v 1.14 2010/06/18 19:35:59 yumiceva Exp $

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
  ~BeamSpotAnalyzer();

 private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  virtual void beginLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
									const edm::EventSetup& context) ;
  virtual void endLuminosityBlock(const edm::LuminosityBlock& lumiSeg, 
								  const edm::EventSetup& c);

  int    ftotalevents;
  int fitNLumi_;
  int resetFitNLumi_;
  int countEvt_;       //counter
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
  double inputBeamWidth_;

  BeamFitter * theBeamFitter;
};

#endif
