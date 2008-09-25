{
  TCanvas MyCanvas("MyCanvas", "New Geometry");
  MyCanvas->Divide(2,1);
  MyCanvas->cd(1);MyCanvas_1->Divide(1,2);MyCanvas_1->cd(1);
  PixelNtuple->Draw("pixel_recHit.gy:pixel_recHit.gx", "abs(pixel_recHit.gz) < 30");
     htemp->SetYTitle("Y (cm)");htemp->SetXTitle("X (cm)");htemp->SetTitle("Tracker hits |z|<30 (cm)");
  MyCanvas_1->cd(2);
  PixelNtuple->Draw("pixel_recHit.gy:pixel_recHit.gx", "pixel_recHit.gx < 30 && pixel_recHit.gx>20 && pixel_recHit.gy<30 && pixel_recHit.gy>20");
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
     htemp->SetYTitle("R (cm)");
     htemp->SetXTitle("Z (cm)");
     htemp->SetTitle("Tracker Hits");
     TAxis *axis = htemp->GetYaxis();
     axis->SetLimits(0., 115.);
     MyCanvas_2->RedrawAxis();
     TLine l5=TLine(-30.0,r_min, 30.0,r_min);l5.SetLineColor(2);
     TLine l6=TLine(-30.0,r_max, 30.0,r_max);l6.SetLineColor(2);
     TLine l7=TLine(-30.0,r_min,-30.0,r_max);l7.SetLineColor(2);
     TLine l8=TLine( 30.0,r_min, 30.0,r_max);l8.SetLineColor(2);
     l5->Draw("Same"); l6->Draw("Same"); l7->Draw("Same"); l8->Draw("Same");
}

