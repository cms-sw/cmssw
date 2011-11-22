{
  TCanvas MyCanvas("MyCanvas", "New Geometry");
  MyCanvas->Divide(2,1);
  MyCanvas->cd(1);MyCanvas_1->Divide(1,2);MyCanvas_1->cd(1);
  StripNtuple->Draw("strip_recHit.gy:strip_recHit.gx", "strip_recHit.subid == 3 || strip_recHit.subid == 5");
  PixelNtuple->Draw("pixel_recHit.gy:pixel_recHit.gx", "abs(pixel_recHit.gz) < 29","same");
     htemp->SetYTitle("Y (cm)");htemp->SetXTitle("X (cm)");htemp->SetTitle("Tracker hits |z|<30 (cm)");
  MyCanvas_1->cd(2);
//  StripNtuple->Draw("strip_recHit.gy:strip_recHit.gx", "strip_recHit.gx < 30 && strip_recHit.gx>10 && strip_recHit.gy<30 && strip_recHit.gy>10");
//  PixelNtuple->Draw("pixel_recHit.gy:pixel_recHit.gx", "pixel_recHit.gx < 30 && pixel_recHit.gx>10 && pixel_recHit.gy<30 && pixel_recHit.gy>10","same");
  PixelNtuple->Draw("pixel_recHit.gy:pixel_recHit.gx", "pixel_recHit.gx < 30 && pixel_recHit.gx>10 && pixel_recHit.gy<30 && pixel_recHit.gy>10");
     // second use of htemp does not seem to work without getting it
     TH2D *htemp = (TH2D*)gPad->GetPrimitive("htemp");
     htemp->SetYTitle("Y (cm)");htemp->SetXTitle("X (cm)");htemp->SetTitle("");
     MyCanvas_1->cd(2)->RedrawAxis();

     int n_bins;  
     double x_min,x_max,r_min,r_max;
     n_bins=htemp->GetNbinsX();
     x_min=htemp->GetBinLowEdge(0);
     x_max=(htemp->GetBinLowEdge(n_bins))+(htemp->GetBinWidth(n_bins));
     r_min=x_min*sqrt(2.0);r_max=x_max*sqrt(2.0);
     MyCanvas_1->cd(1);
     TLine l1=TLine(x_min,x_min,x_min,x_max);l1.SetLineColor(2);
     TLine l2=TLine(x_min,x_min,x_max,x_min);l2.SetLineColor(2);
     TLine l3=TLine(x_max,x_min,x_max,x_max);l3.SetLineColor(2);
     TLine l4=TLine(x_min,x_max,x_max,x_max);l4.SetLineColor(2);
     l1->Draw("Same"); l2->Draw("Same"); l3->Draw("Same"); l4->Draw("Same");

  MyCanvas->cd(2);
  StripNtuple->Draw("sqrt((strip_recHit.gy*strip_recHit.gy)+(strip_recHit.gx*strip_recHit.gx)):strip_recHit.gz");
  PixelNtuple->Draw("sqrt((pixel_recHit.gy*pixel_recHit.gy)+(pixel_recHit.gx*pixel_recHit.gx)):pixel_recHit.gz","","same");
     TH2D *htemp = (TH2D*)gPad->GetPrimitive("htemp");
     htemp->SetYTitle("R (cm)");
     htemp->SetXTitle("Z (cm)");
     htemp->SetTitle("Tracker Hits");
     TAxis *axis = htemp->GetYaxis();
     axis->SetLimits(0., 115.);
     TAxis *xaxis = htemp->GetXaxis();
     xaxis->SetLimits(-320., 320.);
     MyCanvas_2->RedrawAxis();
     TLine l5=TLine(-30.0,r_min, 30.0,r_min);l5.SetLineColor(2);
     TLine l6=TLine(-30.0,r_max, 30.0,r_max);l6.SetLineColor(2);
     TLine l7=TLine(-30.0,r_min,-30.0,r_max);l7.SetLineColor(2);
     TLine l8=TLine( 30.0,r_min, 30.0,r_max);l8.SetLineColor(2);
     l5->Draw("Same"); l6->Draw("Same"); l7->Draw("Same"); l8->Draw("Same");

     TLine l0= TLine(0.0, 0.0,   0.00, 115.0); l0.SetLineColor(2);
     TLine n1= TLine(0.0, 0.0, 135.12, 115.0); n1.SetLineColor(2);
     TLine l15=TLine(0.0, 0.0, 244.83, 115.0);l15.SetLineColor(2);
     TLine n2= TLine(0.0, 0.0, 320.00, 88.20); n2.SetLineColor(2);
     TLine l25=TLine(0.0, 0.0, 320.00, 52.92);l25.SetLineColor(2);
     TLine m1= TLine(0.0, 0.0, -135.12, 115.0); m1.SetLineColor(2);
     TLine m15=TLine(0.0, 0.0, -244.83, 115.0);m15.SetLineColor(2);
     TLine m2= TLine(0.0, 0.0, -320.00, 88.20); m2.SetLineColor(2);
     TLine m25=TLine(0.0, 0.0, -320.00, 52.92);m25.SetLineColor(2);
     l0->Draw("Same");
     n1->Draw("Same"); l15->Draw("Same"); n2->Draw("Same"); l25->Draw("Same");
     m1->Draw("Same"); m15->Draw("Same"); m2->Draw("Same"); m25->Draw("Same");
     TLatex l; l.SetTextAlign(12); l.SetTextSize(0.04); l.SetTextColor(1);
     l.DrawLatex(5,110,"#eta = 0");
     l.DrawLatex(137,110,"#eta = 1");
     l.DrawLatex(247,110,"#eta = 1.5");
     l.DrawLatex(285,90,"#eta = 2");
     l.DrawLatex(285,43,"#eta = 2.5");
}

