#ifndef ROOTLOGON_H
#define ROOTLOGON_H

#include "TROOT.h"
#include "TColor.h"
#include "TStyle.h"

void setColors()
{
    TColor::InitializeColors(); 
    TColor *color = (TColor*)(gROOT->GetListOfColors()->At(40));
    color->SetRGB(0.87, 0.73, 0.53); // light brown    
    color = (TColor*)(gROOT->GetListOfColors()->At(41));
    color->SetRGB(1.0, 0.1, 0.5); // deep roze    
    color = (TColor*)(gROOT->GetListOfColors()->At(42));
    color->SetRGB(0.5, 0.8, 0.1); // light green  
    color = (TColor*)(gROOT->GetListOfColors()->At(43));
    color->SetRGB(0.1, 0.5, 0.3); // dark  green  
    color = (TColor*)(gROOT->GetListOfColors()->At(44));
    color->SetRGB(0.5, 0.2, 0.8); // blue-violet  
    color = (TColor*)(gROOT->GetListOfColors()->At(45));
    color->SetRGB(0.2, 0.6, 0.9); // grey-blue    
    color = (TColor*)(gROOT->GetListOfColors()->At(46));
    color->SetRGB(1.0, 0.5, 0.0); // orange-brick 
    color = (TColor*)(gROOT->GetListOfColors()->At(47));
    color->SetRGB(0.8, 0.0, 0.0); // brick 
//
    color = (TColor*)(gROOT->GetListOfColors()->At(51));
    color->SetRGB(1.0 , 1.0 , 0.8 ); // lightest yellow 
    color = (TColor*)(gROOT->GetListOfColors()->At(52));
    color->SetRGB(0.8 , 1.00, 1.00); // lightest blue-cyan       
    color = (TColor*)(gROOT->GetListOfColors()->At(53));
    color->SetRGB(1.0 , 0.95, 0.95); // lightest rose
    color = (TColor*)(gROOT->GetListOfColors()->At(54));
    color->SetRGB(0.8 , 1.0 , 0.8 ); // lightest green
    color = (TColor*)(gROOT->GetListOfColors()->At(55));
    color->SetRGB(1.00, 1.00, 1.00); // white

// gStyle->SetOptStat(0);   

    gStyle->SetCanvasBorderMode(0);
    gStyle->SetCanvasColor(432-10);//kCyan-10  //formerly 52
    gStyle->SetTitleSize(0.06, "XYZ");
    gStyle->SetTitleXOffset(0.9);
    gStyle->SetTitleYOffset(1.25);

    gStyle->SetLabelOffset(0.007, "XYZ");
    gStyle->SetLabelSize(0.05, "XYZ");

 
    gStyle->SetTitle("");
    gStyle->SetOptTitle(0);
 
    gStyle->SetHistLineColor(0);//45
    gStyle->SetHistLineStyle(1);
    gStyle->SetHistLineWidth(2);

    gStyle->SetPadColor(0);//52  
    gStyle->SetPadBorderSize(1); 
    gStyle->SetPadBottomMargin(0.15);
    gStyle->SetPadTopMargin(0.1);
    gStyle->SetPadLeftMargin(0.15);
    gStyle->SetPadRightMargin(0.15);
    gStyle->SetFrameBorderMode(0);
    gStyle->SetFrameFillColor(10);//55

    Float_t r, g, b;
    Float_t saturation = 1;
    Float_t lightness = 0.5;
    Float_t maxHue = 280;
    Float_t minHue = 0;
    const Int_t maxPretty = 50;
    Float_t hue;
    int colors[maxPretty];

    for (int j = 0; j < maxPretty; j++) 
    {
        hue = maxHue - (j + 1)*((maxHue - minHue) / maxPretty);
        TColor::HLStoRGB(hue, lightness, saturation, r, g, b);
	colors[j] = TColor::GetColor(r, g, b);
    }
    gStyle->SetPalette(maxPretty, colors);
}

#endif
