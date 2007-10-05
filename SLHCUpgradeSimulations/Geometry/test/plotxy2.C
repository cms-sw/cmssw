{
  StripNtuple->Draw("strip_recHit.gx:strip_recHit.gy", "strip_recHit.subid == 3 || strip_recHit.subid == 5");
  PixelNtuple->Draw("pixel_recHit.gx:pixel_recHit.gy", "abs(pixel_recHit.gz) < 30","same");
}

