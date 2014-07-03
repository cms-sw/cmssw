#include <iostream>
using namespace std;
#include "hist.C"
//#include "geom.C"
#include "L1Ntuple.h"
#include <TCanvas.h>

void L1JetTiming(L1Ntuple * ntuple);
void loop(L1Ntuple * ntuple);

TH2F* H2bb = h2d("H2bb",6,-3.5,2.5,6,-3.5,2.5);

int b1,b2,bxc;

////////////////// //////////////////////////////////////////////////////////////////////////

void L1BitCorr(L1Ntuple * ntuple, int ib1, int ib2, int ibx, int nevs) {

 b1 = ib1;
 b2 = ib2;
 bxc = ibx;

 if(nevs) {
   hreset();

   loop(ntuple, 0,nevs);
 }

 TCanvas* c1 = new TCanvas("c1","",600,600);
 H2bb->Draw("coltext colz");

}

////////////////////////////////////////////////////////////////////////////////////////////

void loop(L1Ntuple * ntuple, int i1, int i2) {

//  cout << "enter1" <<endl;

//  h1->SetBranchStatus("*",0);
//  cout << "enter2" <<endl;
//  h1->SetBranchStatus("Gtt*",1);
//  cout << "enter3" <<endl;
//  h1->SetBranchStatus("Bx",1);
//  cout << "enter4" <<endl;

 if(i2==-1 || i2>ntuple->fChain->GetEntries()) i2=ntuple->fChain->GetEntries();

 cout << "Going to run on " << i2 << " events" << endl;

 for(int i=i1; i<i2; i++) {

   if (ntuple->GetEntry(i)) {

     if(!(i%100000) && i) cout << "processing event " << i << "\r" << flush;
     //    if(!(i%1) && i) cout << "processing event " << i << "\r" << endl;
     
     if(bxc!=-1 && abs(ntuple->ev_.bx - bxc) > 4) continue;
     H2bb->Fill(getfirst(b1,ntuple),getfirst(b2,ntuple));
   }

 }
 cout << "                                                                        \r" << flush;
 return;
}

int getfirst(int ib, L1Ntuple* ntuple) {
 int ir = -3;
 if(ib<64) {
   for(int i=0; i<5; i++) {
     if((ntuple->gttw1[i]>>ib)&1) {
       ir=i-2;
	break;
     }
   }
 } else if(ib<128) {
   for(int i=0; i<5; i++) {
     if((ntuple->gttw2[i]>>(ib-64))&1) {
       ir=i-2;
	break;
     }
   }
 } else {
   for(int i=0; i<5; i++) {
     if((ntuple->gttt[i]>>(ib-1000))&1) {
       ir=i-2;
	break;
     }
   }
 }
 return ir;
}
