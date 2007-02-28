#ifndef RecoLocalMuon_Histograms_H
#define RecoLocalMuon_Histograms_H

/*
 * \class Histograms
 *  Collection of histograms for RPC RecHit
 *
 *  $Date: 2007/02/09 11:45:24 $
 *  $Revision: 1.4 $
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

  
    TH1F* Res  = new TH1F ("Global Residuals", "Global Residuals", 300, -8, 8);

    TH1F* ResWmin2 = new TH1F ("W-2 Residuals", "Residuals for Wheel -2", 300, -8, 8);
    TH1F* ResWmin1 = new TH1F ("W-1 Residuals", "Residuals for Wheel -1", 300, -8, 8);
    TH1F* ResWzer0 = new TH1F ("W 0 Residuals", "Residuals for Wheel 0", 300, -8, 8);
    TH1F* ResWplu1 = new TH1F ("W+1 Residuals", "Residuals for Wheel +1", 300, -8, 8);
    TH1F* ResWplu2 = new TH1F ("W+2 Residuals", "Residuals for Wheel +2", 300, -8, 8);
    TH1F* ResS1    = new TH1F ("Sector 1 Residuals", "Sector 1 Residuals", 300, -8, 8);
    TH1F* ResS3    = new TH1F ("Sector 3 Residuals", "Sector 3 Residuals", 300, -8, 8);
    

    TFolder* fres = new TFolder("Residuals","Residuals");
    TFolder* focc = new TFolder("Occupancy","Occupancy");


    TH1F* Rechisto   = new TH1F ("RecHits", "RPC RecHits", 300, -150, 150);
    TH1F* Simhisto   = new TH1F ("SimHits", "Simulated Hits", 300, -150, 150);
    TH1F* Pulls      = new TH1F ("Global Pulls", "RPC Global Pulls", 100, -4,4);
    TH1F* ClSize     = new TH1F ("Global ClSize", "Global Cluster Size", 10, 0, 10);

    TH1F* res1cl     = new TH1F ("Residuals CS = 1", "Residuals for ClSize = 1", 300, -8, 8);
    

    TH1F* occRB1IN   = new TH1F ("RB1 IN Occupancy", "RB1 IN Occupancy", 100, 0, 100);
    TH1F* occRB1OUT   = new TH1F ("RB1 OUT Occupancy", "RB1 OUT Occupancy", 100, 0, 100);
   

#endif
