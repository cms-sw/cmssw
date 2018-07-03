#include <iostream>
#include <sstream>

#include "include/PlotTypes.h"
#include "include/PlotCompareUtility.h"
#include "include/HistoData.h"
using namespace std;


int main(int argc, char *argv[]) {

  std::string branchRef("DQMData/PFTask/Benchmarks");
  std::string branchNew("DQMData/PFTask/Benchmarks"); 

  std::string branchRefPrefix("ParticleFlow/Gen");
  std::string branchNewPrefix("ParticleFlow/Gen"); 

  // make sure command line arguments were supplied
  cout <<argc <<endl;

 if (argc == 5) {
    // initialize plot comparison tool with 2 arguments
    branchRefPrefix= argv[3];
    branchNewPrefix= argv[4];
  }
 if (argc == 3) {};
 if ((argc != 3) && (argc != 5))
     // initialize plot comparison tool with 4 arguments
    {
      cerr << "Usage: " << argv[0] << " [reference.root] [new-comparison.root} [RefSubDirectory(default=ParticleFlow/Gen)] [NewSubDirectory(default=ParticleFlow/Gen)]\n"; return 1; 
    }
    
     
    
PlotCompareUtility *pc = new PlotCompareUtility(argv[1],argv[2],branchNew.c_str(), branchNewPrefix,branchRef.c_str(),branchRefPrefix );


  // set thresholds for tests (set to zero or negative to ignore results)
  //pc->setKSThreshold(1e-6);
  //pc->setChi2Threshold(0);
  pc->setKSThreshold(1e-6);
  pc->setChi2Threshold(0);

  // add histogram information for comparison here --reverse order
  //PFJetBenchmark
  pc->addHistoData("ERneutvsPt", Plot2D)->setDoAllow2DRebinningY(true);
  pc->addHistoData("ERNEEvsPt", Plot2D)->setDoAllow2DRebinningY(true);
  pc->addHistoData("ERNHEvsPt", Plot2D)->setDoAllow2DRebinningY(true);
  pc->addHistoData("ERCHEvsPt", Plot2D)->setDoAllow2DRebinningY(true);
  pc->addHistoData("ERPtvsPt", Plot2D)->setDoAllow2DRebinningY(true);

  pc->addHistoData("ERneut", Plot1D);
  pc->addHistoData("ERNEE", Plot1D);
  pc->addHistoData("ERNHE", Plot1D);
  pc->addHistoData("ERCHE", Plot1D);
  pc->addHistoData("ERPt", Plot1D);

  pc->addHistoData("BRneutvsPt", Plot2D)->setDoAllow2DRebinningY(true);
  pc->addHistoData("BRNEEvsPt", Plot2D)->setDoAllow2DRebinningY(true);
  pc->addHistoData("BRNHEvsPt", Plot2D)->setDoAllow2DRebinningY(true);
  pc->addHistoData("BRCHEvsPt", Plot2D)->setDoAllow2DRebinningY(true);
  pc->addHistoData("BRPtvsPt", Plot2D)->setDoAllow2DRebinningY(true);

  pc->addHistoData("BRneut", Plot1D);
  pc->addHistoData("BRNEE", Plot1D);
  pc->addHistoData("BRNHE", Plot1D);
  pc->addHistoData("BRCHE", Plot1D);
  pc->addHistoData("BRPt", Plot1D);

  pc->addHistoData("jetsPt", Plot1D);
  pc->addHistoData("jetsEta", Plot1D);
  pc->addHistoData("Njets", Plot1D);
 
  // PFBenchmark Analyzer
  pc->addHistoData("DeltaRvsEta",Plot2D)->setDoAllow2DRebinningY(true);
  pc->addHistoData("DeltaPhivsEta",Plot2D);
  pc->addHistoData("DeltaEtavsEta",Plot2D);
  pc->addHistoData("DeltaEtvsEta",Plot2D)->setDoAllow2DRebinningY(true);
  pc->addHistoData("DeltaRvsEt",Plot2D)->setDoAllow2DRebinningY(true);
  pc->addHistoData("DeltaEtvsEt",Plot2D)->setDoAllow2DRebinningY(true);

  pc->addHistoData("DeltaR",Plot1D);
  pc->addHistoData("DeltaPhi",Plot1D);
  pc->addHistoData("DeltaEta",Plot1D);
  pc->addHistoData("DeltaEt",Plot1D);

  // ...
  // ...

  // check if everything was set up properly
  if (!pc->isValid()) {
    cout << "PlotCompareUtility failed to initialize!!!" << endl << "Final Result: no_data" << endl;
    cerr << "Invalid TFile(s), directory, task, or no histograms specified.\n";
    pc->dump();
    return 1;
  }

  // retrieve the list of added HistoData
  vector<HistoData> *histos = pc->getHistos();

  // loop over HistoData, compare and produce plots
  vector<HistoData>::iterator hd;
  for (hd = histos->begin(); hd != histos->end(); hd++) {
    //    cout << " ## #"<< hd->getName() << endl;
    pc->compare(&(*hd));
    pc->makePlots(&(*hd));
    pc->makeHTML(&(*hd));
    }


  // produce summary plots
  pc->makeDefaultPlots();
  pc->makeSummary("SummaryResults");

  // report integrated result from all studied HistoData
  cout << "Final Result: " << (pc->getFinalResult() ? "pass" : "fail") << endl;
  return 0;

}
