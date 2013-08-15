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

Double_t tof_etad_pt( Double_t *etad, Double_t *par )
{
    Double_t value=42.0;Double_t tmomentum=par[0];Double_t zcoord=par[1];
    Double_t natcharge=par[2];Double_t invmass=par[3]; Double_t theta_d=2*ATan(Exp(-(*etad)));
    Double_t relr=tmomentum*(1.0E9)/(C()*natcharge*B);
    Double_t Tfinal=ACos(1-0.5*Power(BarrelR/relr,2));
    Double_t z_eta_d=BarrelR/Tan(theta_d);
    Double_t theta_n=ATan(BarrelR/z_eta_d);
    Double_t t_n=BarrelR/(C()*Sin(theta_n));
    Double_t theta_c=ATan(relr*Tfinal/z_eta_d);
    Double_t t_c=Tfinal*relr/(C()*Sin(theta_c));
    Double_t r_d=Tan(theta_d)*Zhalflength;
    Double_t value=t_c-t_n;
    if((*etad)>1.57) {
	t_n=Zhalflength/(C()*Cos(theta_d));
	theta_c=ACos(1-0.5*Power(r_d/relr,2));
	Tfinal=Zhalflength*natcharge*1.60217657*B*Tan(theta_c)/(tmomentum*5.344286);
	t_c=Tfinal*relr/(C()*Sin(theta_c));
	value=t_c-t_n;
    }
    return value;
}


void tof_etad_pt()
{
    TCanvas *c1=new TCanvas("c1","asdfaksdfj",600,400);
    TLegend *legend=new TLegend(0.6,0.7,0.89,0.89);
    TF1 *f1 = new TF1( "pt1",tof_etad_pt, 0.01,3.,4);
    f1->SetParameters(0.75,0.05,1.0,9.1E-31);f1->SetLineColor(42);
    f1->Draw();
    f1->GetHistogram()->GetXaxis()->SetTitle("#eta of particle");
    f1->GetHistogram()->GetYaxis()->SetTitle("#Delta#eta of detector");
    f1->SetMinimum(1.0E-17);f1->SetMaximum(1.0E-4);
    TF1 *f2 = new TF1( "pt2",tof_etad_pt, 0.01,3.,4);
    f2->SetParameters(3.0,0.05,1.0,9.1E-31);f2->SetLineColor(13);
    f2->Draw("SAME");
    TF1 *f3 = new TF1( "pt3",tof_etad_pt, 0.01,3.,4);
    f3->SetParameters(5.0,0.05,1.0,9.1E-31);f3->SetLineColor(33);
    f3->Draw("SAME");
    TF1 *f4 = new TF1( "pt4",tof_etad_pt, 0.01,3.,4);
    f4->SetParameters(10.0,0.05,1.0,9.1E-31);f4->SetLineColor(66);
    f4->Draw("SAME");
    TF1 *f5 = new TF1( "pt5",tof_etad_pt, 0.01,3.,4);f5->SetLineColor(1);
    f5->SetParameters(50.0,0.05,1.0,9.1E-31);
    f5->Draw("SAME");
    TF1 *f6 = new TF1( "pt6",tof_etad_pt, 0.01,3.,4);
    f6->SetParameters(100.0,0.05,1.0,9.1E-31);f6->SetLineColor(99);
    f6->Draw("SAME");
    gPad->SetLogy();
    c1->Draw();
    legend->SetHeader("z_0=0.00m");
    legend->AddEntry(f1,"p_t=0.75GeV","l");
    legend->AddEntry(f2,"p_t=3.00GeV","l");
    legend->AddEntry(f3,"p_t=5.00GeV","l");
    legend->AddEntry(f4,"p_t=10.00GeV","l");
    legend->AddEntry(f5,"p_t=50.00GeV","l");
    legend->AddEntry(f6,"p_t=100.00GeV","l");
    legend->Draw();
}

