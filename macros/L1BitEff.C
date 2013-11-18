#include <iostream>
#include <fstream>
using namespace std;
#include "hist.C"
#include <TCanvas.h>
//#include "geom.C"
#include "L1Ntuple.h"

void L1BitEff(L1Ntuple * ntuple, int ibx=-1, int nevs=-1);
void loop(L1Ntuple * ntuple, int i1=0, int i2=-1);
bool getfirst(int ib, L1Ntuple* ntuple);

//TH1F* evt = new TH1F("evt", "BX Number", 100, -50., 50.);

int nTotalMinBias = 0;
int PhysicsBit[128], TTBit[64];

int bxc;

int run_;
int lumi(L1Ntuple* ntuple);
int run(L1Ntuple* ntuple);
//////////////////////////////////////////////////////////////////////////////

void L1BitEff(L1Ntuple* ntuple, int ibx, int nevs) {

  bxc = ibx;

  for(int i = 0; i < 128; i++){PhysicsBit[i] = 0; }
  for(int i = 0; i < 64; i++){TTBit[i] = 0;}
  
  if(nevs) {
    hreset();

   loop(ntuple, 0,nevs);
 }

  TString filer = "L1BitEff_";
  filer = filer + (Long_t) run_;
  TString file_txt = filer + ".txt";
  ofstream outFile;
  //outFile.open("AlgoEff.txt");
  outFile.open(file_txt);
  outFile<<"Physics Algo bit Number"<<", "<<"Efficiency"<<endl;
  for(int i = 0; i < 128; i++){
    outFile<<i<<"           "<<double(PhysicsBit[i])/double(nTotalMinBias)<<endl;
  }
  outFile<<"TT Algo bit Number"<<", "<<"Efficiency"<<endl;
  for(int i = 0; i < 64; i++){
    outFile<<i<<"           "<<double(TTBit[i])/double(nTotalMinBias)<<endl;
  }

  outFile.close();

  //TCanvas* c1 = new TCanvas("c1","",600,600);
  //evt->Draw();
}

////////////////////////////////////////////////////////////////////////////////////////////

void loop(L1Ntuple* ntuple, int i1, int i2) {

 if(i2==-1 || i2>ntuple->fChain->GetEntries()) i2=ntuple->fChain->GetEntries();

 cout << "Going to run on " << i2 << " events" << endl;

 for(int i=i1; i<i2; i++) {

   if (ntuple->GetEntry(i)) {

     if(!(i%100000) && i) cout << "processing event " << i << "\r" << flush;
     //    if(!(i%1) && i) cout << "processing event " << i << "\r" << endl;
     
     if(bxc!=-1 && abs(ntuple->event_->bx - bxc) > 4) continue;

     bool TTBitPass = false;
     if(getfirst(1000, ntuple) && 
	(getfirst(1040, ntuple) || getfirst(1041, ntuple)) && 
	!(getfirst(1036, ntuple) || getfirst(1037, ntuple) || 
	  getfirst(1038, ntuple) || getfirst(1039, ntuple)))TTBitPass = true;

     if(!TTBitPass)continue;

     bool NoScraping = false;
     bool GoodVertex = false;
     //cout<<"nTracks "<<ntuple->recoTrack_.nTrk<<endl;
     //cout<<"nVtx "<<ntuple->recoVertex_.nVtx<<endl;
     if(ntuple->recoTrack_->nTrk > 10){
       if(ntuple->recoTrack_->fHighPurity > 0.25)NoScraping = true;
     }else{ NoScraping = true;}

     for(unsigned int ivx = 0; ivx < ntuple->recoVertex_->nVtx; ivx++){
       if(ntuple->recoVertex_->NDoF[ivx] > 4 &&
	  fabs(ntuple->recoVertex_->Z[ivx]) <= 15. &&
	  fabs(ntuple->recoVertex_->Rho[ivx]) <= 2.)GoodVertex = true;
     }
     if(!NoScraping)continue;
     if(!GoodVertex)continue;
     
     nTotalMinBias++;

     //evt->Fill(ntuple->event_.bx);

     //get all Physics trigger bits.
     for(int i = 0; i < 128; i++){
       if(getfirst(i,ntuple))PhysicsBit[i] += 1;
     }
     for(int i = 0; i < 64; i++){
       if(getfirst(1000+i,ntuple))TTBit[i] += 1;
     }
     
     run_=run(ntuple);
   }
 }
 cout << "                                                      \r" << flush;
 return;
}
int lumi(L1Ntuple* ntuple){
  return ntuple->event_->lumi;
}
int run(L1Ntuple* ntuple){
  return ntuple->event_->run;
}
bool getfirst(int ib, L1Ntuple* ntuple) {
  bool PassBit = false;
  if(ib<64) {
    if((ntuple->gt_->tw1[2]>>ib)&1) {
      PassBit = true;
     }
   }
  else if(ib<128) {
    if((ntuple->gt_->tw2[2]>>(ib-64))&1) {
       PassBit = true;
    }
  }
  else {
    if((ntuple->gt_->tt[2]>>(ib-1000))&1) {
      PassBit = true;
    }
  }
  
 return PassBit;
}

