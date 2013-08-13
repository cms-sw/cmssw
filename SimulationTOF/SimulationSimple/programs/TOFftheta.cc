#include <iostream>
#include <cstdlib>
#include "TF1.h"
#include "TFile.h"
#include "TMath.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TH1F.h"

#define B 3.8
#define BarrelR 1.3
#define Zhalflength 3.0

using namespace std;
using namespace TMath;

Double_t delta_t( Double_t *eta, Double_t *par )
{
    Double_t value=42.0;Double_t tmomentum=par[0];Double_t zcoord=par[1];Double_t natcharge=par[2];Double_t invmass=par[3];Double_t theta=2*ATan(Exp(-(*eta)));
    Double_t relr=tmomentum*(1.0E9)/(C()*natcharge*B);
    Double_t Tfinal=ACos(1-0.5*Power(BarrelR/relr,2));
    Double_t neutral_t=BarrelR/(Sin(theta)*C());
    Double_t neutral_t_e=(Zhalflength-zcoord)/(C()*Cos(theta));
    Double_t tdiff=Tfinal*relr/(C()*Sin(theta))-neutral_t;
    Double_t Tz1=(Zhalflength-zcoord)*natcharge*1.60217657*B*Tan(theta)/(tmomentum*5.344286);
    if((*eta)>=0.0 && Abs(Tfinal)>Abs(Tz1)) {Tfinal=Tz1;
        if(neutral_t>neutral_t_e) neutral_t=neutral_t_e;
        tdiff=Tfinal*relr/(C()*Sin(theta))-neutral_t;
    }
/*  if((*eta)<0.0 && Abs(Tfinal)>Abs(Tz1)) {Tfinal=Tz1;
        if(neutral_t>neutral_t_e) neutral_t=neutral_t_e;
        tdiff=Tfinal*relr/(C()*Sin(theta))-neutral_t;
    }*/
    if(par[4]==0.0) value=tdiff;
    if(par[4]==1.0) value=neutral_t;
    if(par[4]==2.0) value=Tfinal*relr/(C()*Sin(theta));
    return value;
}


int main()
{
    TFile *outfile=new TFile("1.root", "RECREATE");
    TCanvas *c1=new TCanvas("c1","asdfaksdfj",600,400);
    TLegend *legend=new TLegend(0.6,0.7,0.89,0.89);
    TF1 *f = new TF1( "deltat",delta_t, 0.1,3.,5);
    f->SetParameters(5.0,0.0,1.0,9.1E-31,0.0);f->SetLineColor(42);
    f->Draw();
    f->Write();
    c1->Update();
    TF1 *f2 = new TF1( "deltat2", delta_t, 0.1, 3., 5 );
    f2->SetParameters(10.0,0.0,1.,9.1E-31,0.0);f2->SetLineColor(13);
    f2->Draw("SAME");
    f2->Write();
    TF1 *f3 = new TF1( "deltat3", delta_t, 0.1, 3., 5 );
    f3->SetParameters(50.0,0.0,1.,9.1E-31,0.0);
    f3->Draw("SAME");
    f3->Write();
    legend->AddEntry(f,"p=5GeV","l");
    legend->AddEntry(f2,"p=10GeV","l");
    legend->AddEntry(f3,"p=50GeV","l");
    legend->Draw();
    c1->Update();
    c1->Write();
    outfile->Close();
    return 0;
}
