#ifndef _RECOMET_METANALYZER_HCALNOISEINFOANALYZER_H_
#define _RECOMET_METANALYZER_HCALNOISEINFOANALYZER_H_

//
// HcalNoiseInfoAnalyzer.h
//
//    description: Skeleton analyzer for the HCAL noise information
//
//    author: J.P. Chou, Brown
//
//

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"


// forward declarations
class TH1D;
class TFile;

namespace reco {

  //
  // class declaration
  //

  class HcalNoiseInfoAnalyzer : public edm::EDAnalyzer {
  public:
    explicit HcalNoiseInfoAnalyzer(const edm::ParameterSet&);
    ~HcalNoiseInfoAnalyzer();
  
  
  private:
    virtual void beginJob(const edm::EventSetup&);
    virtual void analyze(const edm::Event&, const edm::EventSetup&);
    virtual void endJob();

    // root file/histograms
    TFile* rootfile_;
    TH1D* hNHPDHits_;
    TH1D* hNHPDMaxHits_;

    // parameters
    std::string rbxCollName_;      // label for the HcalNoiseRBXCollection
    std::string rootHistFilename_;   // name of the histogram file
  
  };

} // end of namespace

#endif
