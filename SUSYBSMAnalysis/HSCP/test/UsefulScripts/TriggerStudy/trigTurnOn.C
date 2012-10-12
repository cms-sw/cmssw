{
  vector<TString> EffPlots;
  EffPlots.push_back("METM");
  EffPlots.push_back("METS");
  EffPlots.push_back("METMS");
  EffPlots.push_back("HTM");
  EffPlots.push_back("HTS");
  EffPlots.push_back("HTMS");
  EffPlots.push_back("MuM");
  EffPlots.push_back("MuS");
  EffPlots.push_back("MuMS");

  TCanvas *c1;

  for(unsigned int i=0; i<EffPlots.size(); i++) {
  //Also, you'll need to fix some things at the bottom of the script (x axis, maybe y axis, but little things)
  //But I've marked them all with "DD". 
  //Otherwise, I think things work out of the box (I know, I've said this before)
  //DD: Put in your root file with your plots
  //Initialize your TH1F's from your file....
  TFile *myEffNfile_Data = new TFile("pictures/Efficiency_Histos_Data12.root");
  //DD: Change the name of the plots to be what you want.
  TH1D* htop_data = (TH1D*) myEffNfile_Data->Get(EffPlots[i] + "DeDxTop");
  TH1D* hbot_data = (TH1D*) myEffNfile_Data->Get(EffPlots[i] + "DeDxBot");

  htop_data->Rebin(6);
  hbot_data->Rebin(6);

  //DD: Set your plots x-axis range:
 float x_min = 0;
 float x_max = 6;
 if(i%3==1) x_max=1;

  htop_data->Draw();
  hbot_data->Draw("same");
  TH1D* ratio = htop_data->Clone();
  ratio->Sumw2();
  ratio->Divide(hbot_data);
  //Fitting with the error function with the form:
  //erf((m-m0)/sigma)+1)/2
  //DD: Change the range for the fitting yourself from 20 - 100 to your own (min,max)
  TF1 * f1 = new TF1("f1","(TMath::Erf((x*[0] -[1])/[2])+1.)/2.",0,6.);
  f1->SetParameter(2,1.);
  ratio->Fit(f1,"R");
  //This is all to make things look nice

  c1 = new TCanvas("c1","Canvas1",0,0,500,500);

  c1->SetLineColor(0);
  c1->SetFrameFillColor(0);
  c1->SetFillStyle(4000);
  c1->SetFillColor(0);   
  c1->SetBorderMode(0);
  gStyle->SetOptStat(0);    
  c1->SetFillColor(0);
  c1->SetBorderSize(0);
  c1->SetBorderMode(0);
  c1->SetLeftMargin(0.15);
  c1->SetRightMargin(0.12);
  c1->SetTopMargin(0.12);
  c1->SetBottomMargin(0.15);
  gROOT->SetStyle("Plain");
  gStyle->SetOptStat(0000); //This puts in stats box
  gStyle->SetTitleBorderSize(0);
  gStyle->SetTitleX(0.5); // X position of the title box from left
  gStyle->SetTitleAlign(23);
  gStyle->SetTitleY(.975); // Y position of the title box from bottom
  gStyle->SetLabelSize(0.03,"y");
  gStyle->SetStatX(.9);
  gStyle->SetStatY(.9);
  gStyle->SetStatW(0.20);
  gStyle->SetStatFontSize(0.044);
  gStyle->SetStatColor(0);  

  eff0 = new TGraphAsymmErrors();

  //Really you can divide in any way you'd like, here's the link for the options:
  //http://root.cern.ch/root/html528/TGraphAsymmErrors.html (Wilson, CP, etc)
  eff0->BayesDivide(htop_data,hbot_data);
  //  eff0->BayesDivide(htop_data,hbot_data,"w");

  //
  TH1D *T_Empty = new TH1D("T_Empty", "", 1, x_min, x_max);
  T_Empty->SetMinimum(0.0);
  T_Empty->SetMaximum(1.10);
  T_Empty->SetStats(kTRUE);
  T_Empty->GetXaxis()->SetLabelOffset(0.01);
  T_Empty->GetYaxis()->SetLabelOffset(0.01);
  T_Empty->GetXaxis()->SetLabelSize(0.035);
 T_Empty->GetXaxis()->SetLabelFont(42);
 T_Empty->GetXaxis()->SetTitleSize(0.040);
 T_Empty->GetYaxis()->SetLabelSize(0.035);
 T_Empty->GetYaxis()->SetLabelFont(42);
 T_Empty->GetYaxis()->SetTitleSize(0.040);
 T_Empty->GetXaxis()->SetTitleOffset(1.29);
 T_Empty->GetYaxis()->SetTitleOffset(1.39);
 T_Empty->GetXaxis()->SetTitleColor(1);
 T_Empty->GetYaxis()->SetTitleColor(1);
 T_Empty->GetXaxis()->SetNdivisions(10505);
 T_Empty->GetYaxis()->SetNdivisions(515);
 T_Empty->GetXaxis()->SetTitleFont(42);
 T_Empty->GetYaxis()->SetTitleFont(42);
 //DD:Edit these labels
 T_Empty->GetXaxis()->SetTitle("I_{h} (MeV/cm)");
 T_Empty->GetYaxis()->SetTitle("Efficiency");
 T_Empty->Draw("AXIS");
 
 eff0->SetMarkerStyle(20);
 eff0->SetMarkerSize(1.0);
 eff0->SetMarkerColor(1);
 eff0->SetLineWidth(2);

 eff0->Draw("e1pZ");

 TLegend *leg = new TLegend(0.40,0.42,0.82,0.57,NULL,"brNDC");
 leg->SetTextFont(42);
 leg->SetTextSize(0.030);
 leg->SetLineColor(1);
 leg->SetLineStyle(1);
 leg->SetLineWidth(1);
 leg->SetFillStyle(1001);
 //DD:Edit these labels and positions
 leg->AddEntry(eff0,"Any labels you'd like to add to make it nice","p");
 leg->SetBorderSize(0);
 leg->SetFillColor(0);
 //leg->Draw();

 TPaveText* T = new TPaveText(0.4,0.995,0.82,0.945, "NDC");
 T->SetFillColor(0);
 T->SetTextAlign(22);
 char tmp[2048];
 sprintf(tmp,"CMS Preliminary   #sqrt{s} = %1.0f TeV",8.0);
 T->AddText(tmp);
 T->Draw("same");

 tex = new TLatex();
 tex->SetTextColor(1);
 tex->SetTextSize(0.030);
 tex->SetLineWidth(2);
 tex->SetTextFont(42);
 //DD:Edit these labels and positions
 //tex->DrawLatex(1000., 0.65, "More labels!");
 //tex->Draw();

 c1->SaveAs("pictures/" + EffPlots[i] + "Eff.png");
 c1->SaveAs("pictures/" + EffPlots[i] + "Eff.pdf");
 //c1->SaveAs("pictures/" + EffPlots[i] + "Eff_3to5.png");
 //c1->SaveAs("pictures/" + EffPlots[i] + "Eff_3to5.pdf");

 //DD:Edit these labels
 //99% efficiency line.
 float y_min = 0.9;
 float y_max = 0.9;
 TLine *line = new TLine(x_min, y_min, x_max, y_max);
 line->SetLineColor(kRed);
 line->SetLineWidth(2);
 line->SetLineStyle(3);
 //line->Draw("same");
 //I use this line to show on the x-axis at what x I become 99% efficient
 //but this I leave up to you.
 TLine *line2 = new TLine(900., 0.5, 900., 1.0);
 line2->SetLineColor(kBlue);
 line2->SetLineWidth(3);
 line2->SetLineStyle(2);
 //line2->Draw("same");
 f1->SetLineColor(4);
 f1->Draw("same");

 c1->SaveAs("pictures/" + EffPlots[i] + "Eff_Fit.png");
 c1->SaveAs("pictures/" + EffPlots[i] + "Eff_Fit.pdf");
 //c1->SaveAs("pictures/" + EffPlots[i] + "Eff_3to5_Fit.png");
 //c1->SaveAs("pictures/" + EffPlots[i] + "Eff_3to5_Fit.pdf");


 delete c1;
  }
}
