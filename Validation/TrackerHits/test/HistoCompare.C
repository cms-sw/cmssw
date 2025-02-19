#include <iostream.h>

class HistoCompare {

 public:

  HistoCompare() { std::cout << "Initializing HistoCompare... " << std::endl; } ;

  Double_t KSCompute(TH1 * oldHisto , TH1 * newHisto , TText * te );
  Int_t KSok(TH1 * oldHisto , TH1 * newHisto);
  void KSdraw(TH1 * oldHisto , TH1 * newHisto); 

 private:
  
  Double_t ks;

  TH1 * myoldHisto1;
  TH1 * mynewHisto1;


  TText * myte;

};

Int_t HistoCompare::KSok(TH1 * oldHisto , TH1 * newHisto) 
{
  myoldHisto1 = oldHisto;
  mynewHisto1 = newHisto;
  int n_test, n_ref;  
  int s_test, s_ref;  
  n_test = mynewHisto1->GetEntries();
  n_ref =  myoldHisto1->GetEntries();
  s_test = mynewHisto1->Integral();
  s_ref =  myoldHisto1->Integral();
//  cout << n_test<<" "<<n_ref<<" "<<s_test<<" "<<s_ref<<" "<< endl;
  int flag = 1;
  if (n_test == 0 || n_ref == 0) flag = 0;
  if (n_test == 0 && n_ref == 0) {
   cout << "[OVAL]: Reference and test histograms are empty for " 
        << myoldHisto1->GetName()<< endl;
   flag = 0;
  } 
  if (s_test < 0.00000001 && s_ref < 0.00000001) {
   cout << "[OVAL]: Reference and test histograms have zero integral for " 
        << myoldHisto1->GetName()<< endl;
   flag = 0;
  } 
  if ((n_test != 0 && n_ref == 0) || (n_test == 0 && n_ref != 0)) {
   cout << "[OVAL]: # of hits in test histogram " << myoldHisto1->GetName() << " " << 
   n_test << " # of hits in reference histogram " << n_ref << endl;
   flag = 0;
  }		   
  if ((s_test > 0.00000001 && s_ref < 0.00000001) || 
      (s_test < 0.00000001 && s_ref > 0.00000001)) {
   cout << "[OVAL]: integral of test histogram "<< myoldHisto1->GetName() << " " << 
   s_test << " integral of reference histogram " << s_ref << endl;
   flag = 0;
  }    
  
  return flag;
}


Double_t HistoCompare::KSCompute(TH1 * oldHisto , TH1 * newHisto , TText * te )
{
  myoldHisto1 = oldHisto;
  mynewHisto1 = newHisto;
  myte = te;
  mynewHisto1->Sumw2();
  myoldHisto1->Sumw2();
  Double_t ks = mynewHisto1->KolmogorovTest(myoldHisto1);
  return ks;

}

void HistoCompare::KSdraw(TH1 * oldHisto , TH1 * newHisto)
{
  float max;
  myoldHisto1 = oldHisto;
  mynewHisto1 = newHisto;
  myoldHisto1->Rebin(100);
  mynewHisto1->Rebin(100);
  float max_r = myoldHisto1->GetMaximum();
  float max_t = mynewHisto1->GetMaximum();
  if (max_r>max_t) max = 1.1*max_r;
  else max = 1.1*max_t;
  myoldHisto1->SetMaximum(max);
  myoldHisto1->SetLineColor(2);
  myoldHisto1->SetLineStyle(1);
  mynewHisto1->SetLineColor(4);
  mynewHisto1->SetLineStyle(2);
  
  return;
}


