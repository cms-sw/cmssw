{
  TCanvas MyCanvas("MyCanvas", "New Geometry");
    gStyle->SetCanvasColor(10); gStyle->SetCanvasBorderSize(0); gStyle->SetCanvasBorderMode(0);gStyle->SetTitleFillColor(0);
    gStyle->SetFillColor(0);MyCanvas->UseCurrentStyle();MyCanvas->Clear();
  // XY View -- SiStrip Hits
  StripNtuple->Draw("strip_recHit.gy:strip_recHit.gx", "abs(strip_recHit.gz)<=65.0");
  htemp->SetYTitle("Y (cm)");htemp->SetXTitle("X (cm)");htemp->SetTitle("SiStrip recHits");
  MyCanvas->Print("recHits_FPix_XY_View.eps");

  // XY View -- Pixel Hits
  PixelNtuple->Draw("pixel_recHit.gy:pixel_recHit.gx", "pixel_recHit.subid==1");
  htemp->SetYTitle("Y (cm)");htemp->SetXTitle("X (cm)");htemp->SetTitle("BPix recHits");
  MyCanvas->Print("recHits_BPix_XY_View.eps");

  // RZ View
  StripNtuple->Draw("sqrt((strip_recHit.gy*strip_recHit.gy)+(strip_recHit.gx*strip_recHit.gx)):strip_recHit.gz");  
  htemp->GetYaxis()->SetLimits(0.0,115.0);htemp->GetXaxis()->SetLimits(-300.0,300.0);
  PixelNtuple->Draw("sqrt((pixel_recHit.gy*pixel_recHit.gy)+(pixel_recHit.gx*pixel_recHit.gx)):pixel_recHit.gz","","same");
     htemp->SetYTitle("R (cm)");
     htemp->SetXTitle("Z (cm)");
     htemp->SetTitle("Tracker Hits");

     TLine l9 =TLine(0.0,0.0, 300.0  , 49.585); l9.SetLineColor(2); l9->Draw("Same");
     TLine l10=TLine(0.0,0.0,-300.0  , 49.585);l10.SetLineColor(2);l10->Draw("Same");
     TLine l11=TLine(0.0,0.0, 300.0  , 82.716);l11.SetLineColor(2);l11->Draw("Same");
     TLine l12=TLine(0.0,0.0,-300.0  , 82.716);l12.SetLineColor(2);l12->Draw("Same");
     TLine l13=TLine(0.0,0.0, 244.867,115.0  );l13.SetLineColor(2);l13->Draw("Same");
     TLine l14=TLine(0.0,0.0,-244.867,115.0  );l14.SetLineColor(2);l14->Draw("Same");
     TLine l15=TLine(0.0,0.0, 135.149,115.0  );l15.SetLineColor(2);l15->Draw("Same");
     TLine l16=TLine(0.0,0.0,-135.149,115.0  );l16.SetLineColor(2);l16->Draw("Same");
     TLine l17=TLine(0.0,0.0,   0.0  ,115.0  );l17.SetLineColor(2);l17->Draw("Same");

     TPaveText T1(  0.0,116.0, 70.0,125.0);TText *Text1=T1->AddText("#eta = 0  ");T1.SetBorderSize(0);T1->Draw(""); 
     TPaveText T2(135.0,116.0,205.0,125.0);TText *Text2=T2->AddText("#eta = 1  ");T2.SetBorderSize(0);T2->Draw("");
     TPaveText T3(245.0,116.0,315.0,125.0);TText *Text3=T3->AddText("#eta = 1.5");T3.SetBorderSize(0);T3->Draw("");
     TPaveText T4(305.0, 78.0,375.0, 88.0);TText *Text4=T4->AddText("#eta = 2.0");T4.SetBorderSize(0);T4->Draw("");
     TPaveText T5(305.0, 45.0,375.0, 55.0);TText *Text5=T5->AddText("#eta = 2.5");T5.SetBorderSize(0);T5->Draw("");
  MyCanvas->Print("recHits_Tracker_RZ_View.eps");

}

/*
Create your RE_DIGI_L1_DIGI2RAW_RAW2DIGI_L1Reco_RECO.py from the instructions on the twiki
Edit it to include the Ntupleizer (See below)
// 
  process.ReadLocalMeasurement = cms.EDAnalyzer("StdHitNtuplizer",
     src = cms.InputTag("siPixelRecHits"),
     stereoRecHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
     rphiRecHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
     matchedRecHits = cms.InputTag("siStripMatchedRecHits","matchedRecHit"),
     ### if using simple (non-iterative) or old (as in 1_8_4) tracking
     trackProducer = cms.InputTag("generalTracks"),
     OutputFile = cms.string("stdgrechitfullph1g_ntuple.root"),
     ### for using track hit association
     associatePixel = cms.bool(True),
     associateStrip = cms.bool(False),
     associateRecoTracks = cms.bool(False),
     ROUList = cms.vstring('g4SimHitsTrackerHitsPixelBarrelLowTof',
                         'g4SimHitsTrackerHitsPixelBarrelHighTof',
                         'g4SimHitsTrackerHitsPixelEndcapLowTof',
                         'g4SimHitsTrackerHitsPixelEndcapHighTof')
  )
  process.reconstruction_step = cms.Path(process.reconstruction*process.ReadLocalMeasurement)
//
Run your RE_DIGI_L1_DIGI2RAW_RAW2DIGI_L1Reco_RECO.py workflow
Then plot the recHits.
$ root -l stdgrechitfullph1g_ntuple.root 
root [0] 
Attaching file stdgrechitfullph1g_ntuple.root as _file0...
root [1] .X plotxy.C

*/
