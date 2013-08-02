#include <iostream>
#include <cstdlib>
#include "TF1.h"
#include "TFile.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TLegend.h"

#define B 3.8
#define BarrelR 1.3
#define Zhalflength 3.0

using namespace std;
using namespace TMath;

Double_t delta_t( Double_t *eta, Double_t *par )
{
    Double_t m=par[2]/Sqrt(1-Power(par[0],2));
    Double_t Tfinal=ACos(1-0.5*Power(BarrelR*par[3]*B*(1+exp(-2*(*eta)))/(m*(3.0E8)*par[0]*2*exp(-(*eta))),2));
    Double_t neutral_t=BarrelR*(1+exp(-2*(*eta)))/(2*exp(-(*eta))*(3.0E8)*par[0]);
    Double_t neutral_t_e=(Zhalflength-par[1])*(1+exp(-2*(*eta)))/(par[0]*(3.0E8)*(1-exp(-2*(*eta))));
    Double_t value=Tfinal*m/(par[3]*B)-neutral_t;
    Double_t Tz1=(Zhalflength-par[1])*par[3]*B*(1+exp(-2*(*eta)))/(m*(3.0E8)*par[0]*(1-exp(-2*(*eta))));
    if((*eta)>=0.0 && Abs(Tfinal)>Abs(Tz1)) {Tfinal=Tz1;
	if(neutral_t>neutral_t_e) neutral_t=neutral_t_e;
	value=Tfinal*m/(par[3]*B)-neutral_t;
    }
    if((*eta)<0.0 && Abs(Tfinal)>Abs(Tz1)) {Tfinal=Tz1;
	if(neutral_t>neutral_t_e) neutral_t=neutral_t_e;
	value=Tfinal*m/(par[3]*B)-neutral_t;
    }    
    return value;
}

int main()
{
    TFile *outfile=new TFile("1.root", "RECREATE");
    TCanvas *c1=new TCanvas("c1","asdfaksdfj",600,400);
    TLegend *legend=new TLegend(0.4,0.6,0.89,0.89);
    TF1 *f = new TF1( "deltat",delta_t, -3.,3.,4);
    f->SetParameters(0.9999999999999,0.0,1.9E-31,1.6E-19);
    f->Draw();
    f->Write();
    c1->Update();
    TF1 *f2 = new TF1( "deltat2", delta_t, -3., 3., 4 );
    f2->SetParameters(0.99999999,0.0,1.9E-31,1.6E-19);
    f2->Draw("SAME");
    f2->Write();
    TF1 *f3 = new TF1( "deltat3", delta_t, -3., 3., 4 );
    f3->SetParameters(0.99999999,0.0,1.9E-31,1.6E-19);
    f3->Draw("SAME");
    f3->Write();
    legend->AddEntry(f,"Vz=0.","l");
    legend->AddEntry(f2,"Vz=1.","l");
    legend->AddEntry(f3,"Vz=2.","l");
    legend->Draw();
    c1->Update();
    c1->Write();
    outfile->Close();
    return 0;
}
