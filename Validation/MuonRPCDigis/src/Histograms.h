#ifndef ValidationMuonRPCDigis_Histograms_H
#define ValidationMuonRPCDigis_Histograms_H


/*
 * \class Histograms
 *  Collection of histograms for RPC Digi
 *
 *  $Date: 2007/042/18 11:36:11 $
 *  $Revision: 1.1 $
 *  \author M. Maggi - INFN Bari
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

    TH2F* xyview = new TH2F("X Vs Y View","X Vs Y View",1000, -700., 700., 
                                  			1000, -700., 700.);

    TH2F* rzview = new TH2F("R Vs Z View","X Vs Y View",1000, -1100., 1100.,
			    1000,0., 700.);

    TH1F* Res  = new TH1F ("Digi SimHit difference", "Digi SimHit Difference", 
			   300, -8, 8);

    TH1F* ResWmin2 = new TH1F ("W-2 Residuals", "Residuals for Wheel -2", 300, -8, 8);
    TH1F* ResWmin1 = new TH1F ("W-1 Residuals", "Residuals for Wheel -1", 300, -8, 8);
    TH1F* ResWzer0 = new TH1F ("W 0 Residuals", "Residuals for Wheel 0", 300, -8, 8);
    TH1F* ResWplu1 = new TH1F ("W+1 Residuals", "Residuals for Wheel +1", 300, -8, 8);
    TH1F* ResWplu2 = new TH1F ("W+2 Residuals", "Residuals for Wheel +2", 300, -8, 8);

    

    TFolder* fres = new TFolder("Residuals","Residuals");


#endif
