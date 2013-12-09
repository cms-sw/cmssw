#include <iostream>
#include "TGraph.h"
#include "TMath.h"

TGraph* recoVertexModel(const double dz, const double sigma, const int npileup, const double eff) {

  TGraph* nrecograph = new TGraph;

  double frac = TMath::Erf(dz/(sigma*sqrt(2)));

  cout << frac << endl;

  for(unsigned int i=0; i<npileup; ++i) {

    cout << i << endl;

    double vispileup = i*eff;

    cout << vispileup << endl;

    double nreco = 2 +  vispileup - pow((1+frac),vispileup);

    cout << nreco << endl;


    nrecograph->SetPoint(nrecograph->GetN(),i,nreco);

    cout << "done" << endl;


  }

  return nrecograph;

}
