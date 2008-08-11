#include <iostream>
#include <sstream>

#include "include/PlotTypes.h"
#include "include/PlotCompareUtility.h"
#include "include/HistoData.h"
using namespace std;

int main(int argc, char *argv[]) {

  // make sure command line arguments were supplied
  if (argc != 3) { cerr << "Usage: " << argv[0] << " [reference.root] [new-comparison.root]\n"; return 1; }

  // initialize plot comparison tool
  PlotCompareUtility *pc = new PlotCompareUtility(argv[1],argv[2],"DQMData/PFTask/Benchmarks/ParticleFlow/Reco");

  // set thresholds for tests (set to zero or negative to ignore results)
  //pc->setKSThreshold(1e-6);
  //pc->setChi2Threshold(0);
  pc->setKSThreshold(1e-6);
  pc->setChi2Threshold(0);

  // add histogram information for comparison here
  pc->addHistoData("DeltaEt",Plot1D);
  pc->addHistoData("DeltaEta",Plot1D);
  pc->addHistoData("DeltaPhi",Plot1D);
  pc->addHistoData("DeltaR",Plot1D);
  pc->addHistoData("DeltaEtvsEt",Plot2D)->setDoAllow2DRebinningY(true);
  pc->addHistoData("DeltaEtvsEta",Plot2D)->setDoAllow2DRebinningY(true);
  pc->addHistoData("DeltaEtavsEta",Plot2D);
  pc->addHistoData("DeltaPhivsEta",Plot2D);
  //pc->addHistoData("DeltaEtOverEtvsEt",Plot2D);
  //pc->addHistoData("DeltaEta",Plot1D);
  //pc->addHistoData("DeltaPhi",Plot1D);
  //pc->addHistoData("DeltaPhi",Plot1D);
  //pc->addHistoData("DeltaPhi",Plot1D);
  //pc->addHistoData("DeltaEt",Plot1D);
  //pc->addHistoData("DeltaEt",Plot1D);
  //pc->addHistoData("DeltaEt",Plot1D);
  //pc->addHistoData("DeltaPhi",Plot1D);
  //pc->addHistoData("DeltaEta",Plot1D);
  //pc->addHistoData("DeltaEta",Plot1D);
  //pc->addHistoData("DeltaEta",Plot1D);
  // ...
  // ...
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
