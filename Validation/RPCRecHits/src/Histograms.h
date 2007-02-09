#ifndef RecoLocalMuon_Histograms_H
#define RecoLocalMuon_Histograms_H

/*
 * \class Histograms
 *  Collection of histograms for RPC RecHit
 *
 *  $Date: 2007/01/30 13:59:43 $
 *  $Revision: 1.3 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */


#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"
#include "TString.h"
#include "TTree.h"
#include "TFolder.h"

#include "stdlib.h"
#include <string>
#include <iostream>
using namespace std;



//---------------------------------------------------------------------------------------
// A set of histograms of residuals and pulls for RecHits

  
    TH1F* Res  = new TH1F ("Res", "1D RecHit Residuals", 300, -8, 8);
    TH1F* ResWmin2 = new TH1F ("W-2 Residuals", "Residuals for Wheel -2", 300, -8, 8);
    TH1F* ResWmin1 = new TH1F ("W-1 Residuals", "Residuals for Wheel -1", 300, -8, 8);
    TH1F* ResWzer0 = new TH1F ("W 0 Residuals", "Residuals for Wheel 0", 300, -8, 8);
    TH1F* ResWplu1 = new TH1F ("W+1 Residuals", "Residuals for Wheel +1", 300, -8, 8);
    TH1F* ResWplu2 = new TH1F ("W+2 Residuals", "Residuals for Wheel +2", 300, -8, 8);
    

    TFolder* fres = new TFolder("Residuals","Residuals");
    TFolder* focc = new TFolder("Occupancy","Occupancy");


    TH1F* Rechisto   = new TH1F ("RecHits", "RPC RecHits", 300, -150, 150);
    TH1F* Simhisto   = new TH1F ("SimHits", "Simulated Hits", 300, -150, 150);
    TH1F* Pulls      = new TH1F ("Pulls", "RPC Pulls", 100, -4,4);
    TH1F* ClSize     = new TH1F ("Global ClSize", "Global Cluster Size", 10, 0, 10);
    TH1F* Occupancy  = new TH1F ("Global Occupancy", "Global Occupancy", 100, 0, 100);
   

#endif
