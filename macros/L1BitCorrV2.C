#include <iostream>
using namespace std;
#include "hist.C"
//#include "geom.C"
#include "L1Ntuple.h"
#include <TCanvas.h>


class L1BitCorr:public L1Ntuple
{
public :
  L1BitCorr() {initHistos();}
  L1BitCorr(std::string filename):L1Ntuple(filename) {initHistos();}
  void run(int ib1, int ib2, int ibx=-1, int nevs=-1, bool savegif=false);

private:
  void loop(int i1=0, int i2=-1);
  int getfirst(int ib);
  void initHistos() {H2bb = h2d("H2bb",6,-3.5,2.5,6,-3.5,2.5);}
  TString filename(int bit1, int bit2);
  TString axistitle(int bit);

  int b1,b2,bxc;
  bool savegifplot;
  TH2F* H2bb;
};



////////////////// //////////////////////////////////////////////////////////////////////////

void L1BitCorr::run(int ib1, int ib2, int ibx, int nevs, bool savegif)
{
  if (fChain==0) return;

  b1 = ib1;
  b2 = ib2;
  bxc = ibx;
  savegifplot = savegif;

  if(nevs)
  {
   hreset();
   loop(0,nevs);
  }

  TCanvas* c1 = new TCanvas("c1","",600,600);
  TString xtitle = axistitle(b1);
  TString ytitle = axistitle(b2);
  H2bb->GetXaxis()->SetTitle(xtitle);
  H2bb->GetYaxis()->SetTitle(ytitle);

  H2bb->Draw("coltext colz");

  if (savegifplot)
  {
   TString filen = filename(b1,b2);
   c1->Print(filen,"gif");
  }
}

////////////////////////////////////////////////////////////////////////////////////////////

void L1BitCorr::loop(int i1, int i2)
{

 if(i2==-1 || i2>fChain->GetEntries()) i2=fChain->GetEntries();

 cout << "Going to run on " << i2 << " events" << " for synchro of bits "<< b1 << " and " << b2 << endl;

 for(int i=i1; i<i2; i++)
 {

   if (GetEntry(i)) {

     if(!(i%100000) && i) cout << "processing event " << i << "\r" << flush;
     //    if(!(i%1) && i) cout << "processing event " << i << "\r" << endl;
     
     if(bxc!=-1 && abs(event_->bx - bxc) > 4) continue;
     H2bb->Fill(getfirst(b1),getfirst(b2));
   }

 }
 cout << "                                                                        \r" << flush;
 return;
}

////////////////////////////////////////////////////////////////////////////////////////////

int L1BitCorr::getfirst(int ib)
{
 int ir = -3;
 if(ib<64) {
   for(int i=0; i<5; i++) {
     if((gt_->tw1[i]>>ib)&1) {
       ir=i-2;
	break;
     }
   }
 } else if(ib<128) {
   for(int i=0; i<5; i++) {
     if((gt_->tw2[i]>>(ib-64))&1) {
       ir=i-2;
	break;
     }
   }
 } else {
   for(int i=0; i<5; i++) {
     if((gt_->tt[i]>>(ib-1000))&1) {
       ir=i-2;
	break;
     }
   }
 }
 return ir;
}

////////////////////////////////////////////////////////////////////////////////////////////

TString L1BitCorr::axistitle(int bit)
{
  TString title;
  if (bit>127) {
    title = "BX of TT bit ";
    title = title + (Long_t)(bit-1000);
  } else {
    title = "BX of Algo bit ";
    title = title + (Long_t)(bit);
  } 
  return title;
}

////////////////////////////////////////////////////////////////////////////////////////////
TString L1BitCorr::filename(int bit1, int bit2)
{
  TString filename = "L1BitCorr_";  
  if (bit2>127) {
    filename = filename + "TT";
    filename = filename + (Long_t) (bit2-1000);
  } else {
    filename = filename + "Algo";
    filename = filename + (Long_t)(bit2);
  } 
  filename = filename + "vs";
  if (bit1>127) {
    filename = filename + "TT";
    filename = filename + (Long_t) (bit1-1000);
  } else {
    filename = filename + "Algo";
    filename = filename + (Long_t)(bit1);
  } 
  filename = filename + ".gif";
  return filename;
}
 
