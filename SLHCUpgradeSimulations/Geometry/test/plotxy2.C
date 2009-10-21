{
  TCanvas MyCanvas("MyCanvas", "New Geometry");
  PixelNtuple->Draw("sqrt((pixel_recHit.gy*pixel_recHit.gy)+(pixel_recHit.gx*pixel_recHit.gx)):pixel_recHit.gz");
     htemp->GetYaxis()->SetLimits(0.0,17.0);htemp->GetXaxis()->SetLimits(-65.0,65.0);
     htemp->SetYTitle("R (cm)");
     htemp->SetXTitle("Z (cm)");
     htemp->SetTitle("Tracker Hits");
     MyCanvas->RedrawAxis();
     TLine l9 =TLine(0.0,0.0,  65.0  , 10.743); l9.SetLineColor(2); l9->Draw("Same");
     TLine l10=TLine(0.0,0.0, -65.0  , 10.743);l10.SetLineColor(2);l10->Draw("Same");
     TLine l11=TLine(0.0,0.0, 61.657 , 17.000);l11.SetLineColor(2);l11->Draw("Same");
     TLine l12=TLine(0.0,0.0,-61.657 , 17.000);l12.SetLineColor(2);l12->Draw("Same");
     TLine l13=TLine(0.0,0.0, 36.198 , 17.0  );l13.SetLineColor(2);l13->Draw("Same");
     TLine l14=TLine(0.0,0.0,-36.198 , 17.0  );l14.SetLineColor(2);l14->Draw("Same");
     TLine l15=TLine(0.0,0.0, 19.979 , 17.0  );l15.SetLineColor(2);l15->Draw("Same");
     TLine l16=TLine(0.0,0.0,-19.979 , 17.0  );l16.SetLineColor(2);l16->Draw("Same");
     TLine l17=TLine(0.0,0.0,   0.0  , 17.0  );l17.SetLineColor(2);l17->Draw("Same");

     TLatex l; l.SetTextAlign(12); l.SetTextSize(0.04); l.SetTextColor(1);
     l.DrawLatex(  0 , 18.0,"#eta = 0");
     l.DrawLatex( 20 , 18.0,"#eta = 1");
     l.DrawLatex( 36 , 18.0,"#eta = 1.5");
     l.DrawLatex(61.7, 18.0,"#eta = 2");
     l.DrawLatex(66.0, 10.5,"#eta = 2.5");
}

