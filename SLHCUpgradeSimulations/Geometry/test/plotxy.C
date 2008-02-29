{
  TCanvas MyCanvas("MyCanvas", "New Geometry");
  MyCanvas->Divide(2,2);
  MyCanvas->cd(1);
  PixelNtuple->Draw("pixel_recHit.gy:pixel_recHit.gx", "abs(pixel_recHit.gz) < 30");
  MyCanvas->cd(2);
  PixelNtuple->Draw("pixel_recHit.gy:pixel_recHit.gx", "pixel_recHit.gx < 30 && pixel_recHit.gx>20 && pixel_recHit.gy<30 && pixel_recHit.gy>20");
  MyCanvas->cd(3);
  PixelNtuple->Draw("sqrt((pixel_recHit.gy*pixel_recHit.gy)+(pixel_recHit.gx*pixel_recHit.gx)):pixel_recHit.gz");
}

