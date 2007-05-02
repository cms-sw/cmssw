#ifndef LRHELPFUNCTIONS_H
#define LRHELPFUNCTIONS_H

#include "TH1.h"
#include "TF1.h"
#include "TGraph.h"

using namespace std;

TH1F SoverB(TH1F *hsign, TH1F *hback, int obsNr);
TH1F makePurityPlot(TH1F *hLRtotS, TH1F *hLRtotB);
TGraph makeEffVsPurGraph(TH1F *hLRtotS, TF1 *fPurity);

#endif
