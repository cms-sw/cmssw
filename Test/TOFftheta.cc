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

    Double_t momentum=par[0];
    Double_t zcoord=par[1];
    Double_t natcharge=par[2];
    Double_t invmass=par[3];
    Double_t theta=2*ATan(Exp(-(*eta)));
    Double_t relm=Sqrt(Power(invmass,2.0)+Power(momentum*(1.0E9)*(1.783E-36),2));
    Double_t relr=3.3*momentum*Sin(theta)/(natcharge*B);
    Double_t Tfinal=ACos(1-0.5*Power(BarrelR/relr,2));
    Double_t neutral_t=BarrelR/(Sin(theta)*(3.0E8));
    Double_t neutral_t_e=(Zhalflength-zcoord)/((3.0E8)*Cos(theta));
    Double_t value=Tfinal*relm/(natcharge*(1.6E-19)*B)-neutral_t;
    Double_t Tz1=(Zhalflength-zcoord)*natcharge*(1.6E-19)*B/(momentum*(1.0E9)*(5.344286E-28)*Cos(theta));
    if((*eta)>=0.0 && Abs(Tfinal)>Abs(Tz1)) {Tfinal=Tz1;
        if(neutral_t>neutral_t_e) neutral_t=neutral_t_e;
        value=Tfinal*relm/(natcharge*(1.6E-19)*B)-neutral_t;
    }
/*  if((*eta)<0.0 && Abs(Tfinal)>Abs(Tz1)) {Tfinal=Tz1;
        if(neutral_t>neutral_t_e) neutral_t=neutral_t_e;
        value=Tfinal*relm/(natcharge*(1.6E-19)*B)-neutral_t;
    }*/
    return value;
}


int main()
{
    TFile *outfile=new TFile("1.root", "RECREATE");
    TCanvas *c1=new TCanvas("c1","asdfaksdfj",600,400);
    TLegend *legend=new TLegend(0.4,0.6,0.89,0.89);
    TF1 *f = new TF1( "deltat",delta_t, 0.1,3.,4);
    f->SetParameters(1.0E6,0.0,1.0,9.1E-31);
    f->Draw();
    f->Write();
    c1->Update();
    TF1 *f2 = new TF1( "deltat2", delta_t, 0.1, 3., 4 );
    f2->SetParameters(10.0,0.0,1.,9.1E-31);
    f2->Draw("SAME");
    f2->Write();
    TF1 *f3 = new TF1( "deltat3", delta_t, 0.1, 3., 4 );
    f3->SetParameters(100.0,0.0,1.,9.1E-31);
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
