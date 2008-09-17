{
  TCanvas MyCanvas("MyCanvas", "New Geometry");
  MyCanvas->Divide(2,2);
  MyCanvas->cd(1);
  StripNtuple->Draw("strip_recHit.gx:strip_recHit.gy", "strip_recHit.subid == 3 || strip_recHit.subid == 5");
  PixelNtuple->Draw("pixel_recHit.gx:pixel_recHit.gy", "abs(pixel_recHit.gz) < 30","same");
  MyCanvas->cd(2);
  PixelNtuple->Draw("pixel_recHit.gy:pixel_recHit.gx", "pixel_recHit.gx < 30 && pixel_recHit.gx>15 && pixel_recHit.gy<30 && pixel_recHit.gy>15");
  MyCanvas->cd(3);
  StripNtuple->Draw("sqrt((strip_recHit.gy*strip_recHit.gy)+(strip_recHit.gx*strip_recHit.gx)):strip_recHit.gz");
  PixelNtuple->Draw("sqrt((pixel_recHit.gy*pixel_recHit.gy)+(pixel_recHit.gx*pixel_recHit.gx)):pixel_recHit.gz","","same");
}

