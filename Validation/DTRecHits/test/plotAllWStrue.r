void plotAllWStrue(TString filename, int sector, int sl){
  

  if (! TString(gSystem->GetLibraries()).Contains("Histograms_h")) {
    gROOT->LoadMacro("$CMSSW_BASE/src/Validation/DTRecHits/test/Histograms.h+"); 
     gROOT->LoadMacro("macros.C");
     gROOT->LoadMacro("ranges.C+");
     gROOT->LoadMacro("summaryPlot.C+");
  }

  
  bool doRes = true;
  bool doResVsDist = true;
  bool doResVsAngle = true;
  bool doResVsX = true;
  bool doResVsY = true;
  bool doAngleDist = true;

  TStyle * style = getStyle("tdr");
  style->cd();
  gStyle->SetCanvasDefW(1375); // Set larger canvas
  gStyle->SetCanvasDefH(800);  
  gStyle->SetTitleSize(0.05,"XYZ"); // Set larger axis titles
  gStyle->SetTitleOffset(1.5,"Y");
  gStyle->SetOptTitle(0); // remove histogram titles
  
  setPalette();
  opt2Dplot = "col";
  
 float nsigma = 2;
 
 TFile *file = new TFile(filename);
  
  TString canvbasename = filename;
  canvbasename = canvbasename.Replace(canvbasename.Length()-5,5,"") + TString("_Se") + (long) sector + "_SL" + (long) sl; 

  HRes1DHit* hRes1D[5][3]; // W, S;  
  HRes4DHit* hRes4D[5][3];
  
  
  for (int wheel = -2; wheel<=2; ++wheel) {
    for (int station = 1; station<=3; ++station) {
      
      int iW = wheel+2;
      int iSt= station-1;
      hRes1D[iW][iSt] = new HRes1DHit(file,wheel,station,sl,"S3");
      hRes4D[iW][iSt] = new HRes4DHit(file,wheel,station,0);
      
    }

  }
  
  
  if(doRes){
    TCanvas* c0= new TCanvas(canvbasename+"_AllWSRes", canvbasename+"_AllWSRes");    
    
    c0->Divide(5,3,0.0005,0.0005);

    for (int wheel = -2; wheel<=2; ++wheel) {
      for (int station = 1; station<=3; ++station) {
	int iW = wheel+2;
	int iSt= station-1;

	int ipad=iW+1 +(2-iSt)*5;
	c0->cd(ipad); ++ipad;
	
	if(wheel<0) continue;

	TH1F* h = hRes1D[iW][iSt]->hRes;
	
	TF1* fres=drawGFit(h, nsigma, -0.4, 0.4);
	
	
      }
    }
  }
 
  if(doResVsDist){
    TCanvas* c0= new TCanvas(canvbasename+"_AllWSResVsDist", canvbasename+"_AllWSResVsDist");    
    c0->Divide(5,3,0.0005,0.0005);
        
    for (int wheel = -2; wheel<=2; ++wheel) {
      for (int station = 1; station<=3; ++station) {
	int iW = wheel+2;
	int iSt= station-1;
	int ipad=iW+1 + (2-iSt)*5;
	
	c0->cd(ipad); ++ipad;
	if(wheel<0) continue;
	
	plotAndProfileX(hRes1D[iW][iSt]->hResVsPos,2,1,1,-.1, .1, 0., 2.1);
	
      }
    }
  }
 
  if(doResVsAngle){
    SummaryPlot hs_p0("p0");
    SummaryPlot hs_p1("p1");

    TCanvas* c0= new TCanvas(canvbasename+"_AllWSResVsAngle", canvbasename+"_AllWSResVsAngle");    
    c0->Divide(5,3,0.0005,0.0005);
    
    for (int wheel = -2; wheel<=2; ++wheel) {
      for (int station = 1; station<=3; ++station) {
	  
	  int iW = wheel+2;
	  int iSt= station-1;
	  int ipad=iW+1 + (2-iSt)*5;
	  
	  c0->cd(ipad); ++ipad;
	  if(wheel<0) continue;
	  
	  float min;
	  float max;
	  
	  rangeAngle(wheel, station, sl, min, max);
	  
	  TH1F* hprof = plotAndProfileX(hRes1D[iW][iSt]->hResVsAngle,1,1,1,-0.04, 0.04, min-0.03, max+0.03);
	  TF1 *angleDep= new TF1("angleDep","[0]*cos(x+[2])+[1]", min, max);
	  //TF1 *angleDep= new TF1("angleDep","[0]*x*x+[2]*x+[1]", min, max);
	  angleDep->SetParameters(0.01,0.001, 0.1);
	  
	  if(sl == 2 || (sl == 1 && (iSt!=0 ||(iSt==0 && iW==2)))) angleDep->FixParameter(2,0.);

	  if(sl == 1 && (iW == 3 ||iW == 0) && iSt == 0)  angleDep->SetParameter(2,-0.12);
	  
	  hprof->Fit(angleDep,"RQN"); 
	  
	  if(sl == 2 && (iW == 0 || iW == 4) && (iSt == 0 || iSt == 1)) continue;
	  
	  angleDep->Draw("same");
	  float p0 = angleDep->GetParameter(0);
	  float p1 = angleDep->GetParameter(1);
	  float p2 = angleDep->GetParameter(2);
	  
       	cout << "Wh:" <<wheel<< " St:"<<station << " SL:" <<sl << " "<< p0 << " " << p1 << " " << p2 << endl;
	
	  hs_p0.Fill(wheel, station, sector, p0);
	  hs_p1.Fill(wheel, station, sector, p1);
	
      }
    }
  // Summary plot (still being developed)
    TCanvas* c6= new TCanvas(canvbasename+"_AllWSAngleCorr_p0", canvbasename+"_AllWSAngleCorr_p0");    
    hs_p0.hsumm->GetYaxis()->SetRangeUser(0.5,0.5);
    hs_p0.hsumm->Draw("p");
    
    
    float p0_m[3][3];
    float p1_m[3][3];
    
    for (int wheel = 0; wheel<=2; ++wheel) {
      for (int station = 1; station<=3; ++station) {
	
	float p0_n = hs_p0.hsumm->GetBinContent(hs_p0.bin(-wheel,station,0));
	float p0_p = hs_p0.hsumm->GetBinContent(hs_p0.bin(wheel,station,0));
	float p1_n = hs_p1.hsumm->GetBinContent(hs_p1.bin(-wheel,station,0));
	float p1_p = hs_p1.hsumm->GetBinContent(hs_p1.bin(wheel,station,0));
	p0_m[wheel][station] = (p0_n + p0_p)/2;
	p1_m[wheel][station] = (p1_n + p1_p)/2;
	//	cout << wheel << "   "   << station << "   " <<  p0_m[wheel][station] <<  "   "<<p1_m[wheel][station] <<endl;
      }
      
    }
    
 }
 

  //  if (doResVsX) {
//     TCanvas* c0=new TCanvas(canvbasename+"_AllWSResVsX", canvbasename+"_AllWSResVsX");    
    
//     c0->Divide(5,3);

//     for (int wheel = 0; wheel<=2; ++wheel) {
//       for (int station = 0; station<=2; ++station) {

// 	int ipad=wheel+3 + (2-station)*5;
// 	c0->cd(ipad); ++ipad;
// 	plotAndProfileX(hRes1D[wheel][station]->hResDistVsX,2,1,1,-0.05, 0.05, -150, 150); // FIXME
//       }
//     }
//   }

//  if (doResVsY) {

//    TCanvas* c0=new TCanvas(canvbasename+"_AllWSResVsY", canvbasename+"_AllWSResVsY");    
     
//     c0->Divide(5,3);

//     for (int wheel = 0; wheel<=2; ++wheel) {
//       for (int station = 0; station<=2; ++station) {

// 	int ipad=wheel+3 + (2-station)*5;
// 	c0->cd(ipad); ++ipad;
// 	plotAndProfileX(hRes1D[wheel][station]->hResDistVsY,2,1,1,-0.05, 0.05, -150, 150); // FIXME
//       }
//     }
//   }

 if(doAngleDist){
   TCanvas* c0= new TCanvas(canvbasename+"_AllWSAngleDist", canvbasename+"_AllWSAngleDist");
   c0->Divide(5,3,0.0005,0.0005);
   
   for (int wheel = -2; wheel<=2; ++wheel) {
     for (int station = 1; station<=3; ++station) {
       int iW = wheel+2;
       int iSt= station-1;
       // DTDetId detId(wheel, station, sector, sl, 0, 0);
       int ipad=iW+1 + (2-iSt)*5;
       
       c0->cd(ipad); ++ipad;
       if(wheel<0) continue;

       float min;
       float max;
              
       rangeAngle(wheel, station, sl, min, max);
      //  cout << "min" << min << endl;
//        cout << "max" << max << endl;

       TH1F* hAngle = hRes4D[iW][iSt]->hSimAlpha;
       hAngle->Sumw2();       

       if(sl==2){
	 TH1F* hAngle = hRes4D[iW][iSt]->hSimBeta; 
	 hAngle->Sumw2();       
       }

       hAngle->GetXaxis()->SetRangeUser(-1.2,1.2);
       hAngle->Draw();

     }
   }
}
 
}

   

