{ int plot=0;
  TCanvas MyCanvas("MyCanvas", "New Geometry");
    gStyle->SetCanvasColor(10); gStyle->SetCanvasBorderSize(0); gStyle->SetCanvasBorderMode(0);gStyle->SetTitleFillColor(0);
    gStyle->SetFillColor(0);MyCanvas->UseCurrentStyle();MyCanvas->Clear();
  MyCanvas->Divide(2,1);
  MyCanvas->cd(1);MyCanvas_1->Divide(1,2);MyCanvas_1->cd(1);
  if (
     StripNtuple->Draw("strip_recHit.gy:strip_recHit.gx", "strip_recHit.subid == 3 || strip_recHit.subid == 5"))
     {plot=1;}
  if (plot==0){
     PixelNtuple->Draw("pixel_recHit.gy:pixel_recHit.gx", "abs(pixel_recHit.gz) < 30");
     } 
     htemp->SetYTitle("Y (cm)");htemp->SetXTitle("X (cm)");htemp->SetTitle("Tracker hits |z|<30 (cm)");
  if (plot==1){
     PixelNtuple->Draw("pixel_recHit.gy:pixel_recHit.gx", "abs(pixel_recHit.gz) < 30","same");
     }
  MyCanvas_1->cd(2);
  plot=0;
  if (
     PixelNtuple->Draw("pixel_recHit.gy:pixel_recHit.gx", "pixel_recHit.gx < 30 && pixel_recHit.gx>15 && pixel_recHit.gy<30 && pixel_recHit.gy>15"))
     {plot=1;}
  if (plot==0){
  StripNtuple->Draw("strip_recHit.gy:strip_recHit.gx", "strip_recHit.gx < 30 && strip_recHit.gx>15 && strip_recHit.gy<30 && strip_recHit.gy>15 && abs(strip_recHit.gz)<30");
  }
     htemp->SetYTitle("Y (cm)");htemp->SetXTitle("X (cm)");htemp->SetTitle("");
     MyCanvas_1->cd(2)->RedrawAxis();

     int n_bins;  
     double x_min,x_max,r_min,r_max;
     n_bins=htemp->GetNbinsX();
     x_min=htemp->GetBinLowEdge(0);
     x_max=(htemp->GetBinLowEdge(n_bins))+(htemp->GetBinWidth(n_bins));
     r_min=x_min*sqrt(2.0);r_max=x_max*sqrt(2.0);
     MyCanvas_1->cd(1);
     TLine l1=TLine(x_min,x_min,x_min,x_max);l1.SetLineColor(4);
     TLine l2=TLine(x_min,x_min,x_max,x_min);l2.SetLineColor(4);
     TLine l3=TLine(x_max,x_min,x_max,x_max);l3.SetLineColor(4);
     TLine l4=TLine(x_min,x_max,x_max,x_max);l4.SetLineColor(4);
     l1->Draw("Same"); l2->Draw("Same"); l3->Draw("Same"); l4->Draw("Same");

  MyCanvas->cd(2);
  StripNtuple->Draw("sqrt((strip_recHit.gy*strip_recHit.gy)+(strip_recHit.gx*strip_recHit.gx)):strip_recHit.gz");  
  htemp->GetYaxis()->SetLimits(0.0,115.0);htemp->GetXaxis()->SetLimits(-300.0,300.0);
  PixelNtuple->Draw("sqrt((pixel_recHit.gy*pixel_recHit.gy)+(pixel_recHit.gx*pixel_recHit.gx)):pixel_recHit.gz","","same");
     htemp->SetYTitle("R (cm)");
     htemp->SetXTitle("Z (cm)");
     htemp->SetTitle("Tracker Hits");
     MyCanvas_2->RedrawAxis();
     TLine l5=TLine(-30.0,r_min, 30.0,r_min);l5.SetLineColor(4);
     TLine l6=TLine(-30.0,r_max, 30.0,r_max);l6.SetLineColor(4);
     TLine l7=TLine(-30.0,r_min,-30.0,r_max);l7.SetLineColor(4);
     TLine l8=TLine( 30.0,r_min, 30.0,r_max);l8.SetLineColor(4);
     l5->Draw("Same"); l6->Draw("Same"); l7->Draw("Same"); l8->Draw("Same");

     TLine l9 =TLine(0.0,0.0, 300.0  , 49.585); l9.SetLineColor(2); l9->Draw("Same");
     TLine l10=TLine(0.0,0.0,-300.0  , 49.585);l10.SetLineColor(2);l10->Draw("Same");
     TLine l11=TLine(0.0,0.0, 300.0  , 82.716);l11.SetLineColor(2);l11->Draw("Same");
     TLine l12=TLine(0.0,0.0,-300.0  , 82.716);l12.SetLineColor(2);l12->Draw("Same");
     TLine l13=TLine(0.0,0.0, 244.867,115.0  );l13.SetLineColor(2);l13->Draw("Same");
     TLine l14=TLine(0.0,0.0,-244.867,115.0  );l14.SetLineColor(2);l14->Draw("Same");
     TLine l15=TLine(0.0,0.0, 135.149,115.0  );l15.SetLineColor(2);l15->Draw("Same");
     TLine l16=TLine(0.0,0.0,-135.149,115.0  );l16.SetLineColor(2);l16->Draw("Same");
     TLine l17=TLine(0.0,0.0,   0.0  ,115.0  );l17.SetLineColor(2);l17->Draw("Same");

     TPaveText T1(  0.0,115.0, 70.0,125.0);TText *Text1=T1->AddText("#eta = 0  ");T1.SetBorderSize(0);T1->Draw(""); 
     TPaveText T2(135.0,115.0,205.0,125.0);TText *Text2=T2->AddText("#eta = 1  ");T2.SetBorderSize(0);T2->Draw("");
     TPaveText T3(245.0,115.0,315.0,125.0);TText *Text3=T3->AddText("#eta = 1.5");T3.SetBorderSize(0);T3->Draw("");
     TPaveText T4(305.0, 78.0,375.0, 88.0);TText *Text4=T4->AddText("#eta = 2.0");T4.SetBorderSize(0);T4->Draw("");
     TPaveText T5(305.0, 45.0,375.0, 55.0);TText *Text5=T5->AddText("#eta = 2.5");T5.SetBorderSize(0);T5->Draw("");
}

