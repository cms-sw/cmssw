
// Original Author:  Jan Heyninck
//         Created:  Thu May 18 16:40:24 CEST 2006


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Handle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "TH2.h"
#include "TH1.h"
#include "TF1.h"
#include "TKey.h"
#include "TFile.h"
#include "TGraph.h"
#include <vector>
#include <fstream>
#include "AnalysisDataFormats/TopObjects/interface/TtSemiEvtSolution.h"
#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "TopQuarkAnalysis/TopJetCombination/interface/JetMatchObservables.h"
#include "TopQuarkAnalysis/TopJetCombination/interface/LRHelpFunctions.h"
#include "AnalysisDataFormats/TopObjects/interface/BestMatching.h"


using namespace edm;
using namespace std;

//
// class decleration
//

class JetMatchLR : public edm::EDAnalyzer {
   public:
      explicit JetMatchLR(const edm::ParameterSet&);
      ~JetMatchLR();
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      
   private:
      TFile *outfile[20];
      TH1F  *obsHist[20][10][3], *hLRtotS[20], *hLRtotB[20], *hPurity[20];
      TH2F  *corHist[20][10][10];
      TF1   *fPurity[20], *obsFits[20][10], *combBTagFit;
      TGraph *hEffvsPur[20];
      std::vector<bool>  signEvt[20];
      std::vector<float> obsVals[20][10];
      std::vector<string> *jetLabels;
      std::ofstream *purfile[20];
      JetMatchObservables *JMobs;
      void fillLRtoHistos();
      void makePurityPlots();
};
