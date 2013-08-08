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
#define Zhalflength 3.5

using namespace std;
using namespace TMath;

Double_t etad_etap_pt( Double_t *eta, Double_t *par )
{
    Double_t value=42.0;Double_t tmomentum=par[0];Double_t zcoord=par[1];
    Double_t natcharge=par[2];Double_t invmass=par[3]; Double_t theta=2*ATan(Exp(-(*eta)));
    Double_t relr=tmomentum*(1.0E9)/(C()*natcharge*B);
    Double_t Tfinal=ACos(1-0.5*Power(BarrelR/relr,2));
    Double_t neutral_t=BarrelR/(Sin(theta)*C());
    Double_t neutral_t_e=(Zhalflength-zcoord)/(C()*Cos(theta));
    Double_t tdiff=Tfinal*relr/(C()*Sin(theta))-neutral_t;
    Double_t eta_n=-Log(Tan(0.5*ATan(BarrelR/(C()*Cos(theta)*neutral_t+zcoord))));
    Double_t Tz1=(Zhalflength-zcoord)*natcharge*1.60217657*B*Tan(theta)/(tmomentum*5.344286);
    Double_t deltaeta=-Log(Tan(0.5*ATan(BarrelR/(relr*Tfinal/Tan(theta)+zcoord))))-eta_n;
    if((*eta)>=0.0 && Abs(Tfinal)>Abs(Tz1)) {Tfinal=Tz1;
        if(neutral_t>neutral_t_e) {neutral_t=neutral_t_e;}
        tdiff=Tfinal*relr/(C()*Sin(theta))-neutral_t;
	eta_n=-Log(Tan(0.5*ATan(BarrelR/(C()*Cos(theta)*neutral_t+zcoord))));
       	deltaeta=-Log(Tan(0.5*ATan(Sqrt(Power(relr*(1-Cos(Tfinal)),2)+Power(relr*Sin(Tfinal),2))/Zhalflength)))-eta_n;
    }
    if(par[4]==0.0) value=tdiff;
    if(par[4]==1.0) value=Tz1*relr/(C()*Sin(theta));
    if(par[4]==2.0) value=Tfinal*relr/(C()*Sin(theta));
    if(par[4]==3.0) value=relr;
    if(par[4]==4.0) value=-Log(Tan(0.5*ATan(BarrelR*Tan(theta)/(relr*Tfinal))));
    if(par[4]==5.0) value=deltaeta;
    if(par[4]==6.0) value=2*tdiff;
    return value;
}


void etad_etap_pt()
{
    TCanvas *c1=new TCanvas("c1","asdfaksdfj",600,400);
    TLegend *legend=new TLegend(0.6,0.7,0.89,0.89);
    TF1 *f1 = new TF1( "pt1",etad_etap_pt, 0.01,3.,5);
    f1->SetParameters(4.0,0.0,1.0,9.1E-31,5.0);f1->SetLineColor(42);
    f1->Draw();
    f1->GetHistogram()->GetXaxis()->SetTitle("#eta of particle");
    f1->GetHistogram()->GetYaxis()->SetTitle("#Delta#eta of detector");
    TF1 *f2 = new TF1( "pt2",etad_etap_pt, 0.01,3.,5);
    f2->SetParameters(5.0,0.0,1.0,9.1E-31,5.0);f2->SetLineColor(13);
    f2->Draw("SAME");
    TF1 *f3 = new TF1( "pt3",etad_etap_pt, 0.01,3.,5);
    f3->SetParameters(6.0,0.0,1.0,9.1E-31,5.0);f3->SetLineColor(33);
    f3->Draw("SAME");
    gPad->SetLogy();
    c1->Draw();
    legend->AddEntry(f1,"p_t=4GeV","l");
    legend->AddEntry(f2,"p_t=5GeV","l");
    legend->AddEntry(f3,"p_t=6GeV","l");
    legend->Draw();
}

