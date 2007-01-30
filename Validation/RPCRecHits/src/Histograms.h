#ifndef RecoLocalMuon_Histograms_H
#define RecoLocalMuon_Histograms_H

/*
 * \class Histograms
 *  Collection of histograms for RPC RecHit
 *
 *  $Date: 2007/01/12 12:34:03 $
 *  $Revision: 1.1 $
 *  \author D. Pagano - Dip. Fis. Nucl. e Teo. & INFN Pavia
 */


#include "TH1F.h"
#include "TH2F.h"
#include "TFile.h"
#include "TString.h"

#include <string>
#include <iostream>
using namespace std;



//---------------------------------------------------------------------------------------
// A set of histograms of residuals and pulls for RecHits

  
    TH1F* Residuals  = new TH1F ("Residuals", "RecHit residual", 300, -10, 10);

    TH1F* Rechisto   = new TH1F ("RecHits", "RPC RecHits", 300, -150, 150);
   
    TH1F* Simhisto   = new TH1F ("SimHits", "Simulated Hits", 300, -150, 150);

    TH1F* Pulls      = new TH1F ("Pulls", "Pulls", 100, -5,5);

   
#endif

