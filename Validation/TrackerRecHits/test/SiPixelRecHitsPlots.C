void SiPixelRecHitsPlots()
{
   gROOT->Reset();
   gStyle->SetOptStat(1111111);
   char * rfilename = "pixelrechitshisto.root";

   delete gROOT->GetListOfFiles()->FindObject(rfilename);

   TText* te = new TText();
   TFile * rfile = new TFile(rfilename);
   Char_t histo[200];

   rfile->cd("DQMData/TrackerRecHits/Pixel");
  ////////////////////////// 
  // Barrel Clusters
  //////////////////////////

   // Cluster y-size by module
   TCanvas * Pixel = new TCanvas("Pixel", "Pixel",800,1200);
   Pixel->Divide(2,4);
   TH1* clustYSizeModule_[8];

   for (Int_t i=0; i<8; i++) {
      sprintf(histo, "DQMData/TrackerRecHits/Pixel/clustBPIX/Clust_y_size_Module%d;1", i+1);
      Pixel->cd(i+1);
      rfile->GetObject(histo, clustYSizeModule_[i]);
      clustYSizeModule_[i]->Draw();
   }

   Pixel->Print("Clust_y_size_by_module.eps");

   // Cluster x-size by layer
   TH1* clustXSizeLayer_[3];
   Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(1,3);
 
   for (Int_t i=0; i<3; i++) {  
      sprintf(histo, "DQMData/TrackerRecHits/Pixel/clustBPIX/Clust_x_size_Layer%d;1", i+1);
      Pixel->cd(i+1);
      rfile->GetObject(histo, clustXSizeLayer_[i]);
      clustXSizeLayer_[i]->Draw();
   }

   Pixel->Print("Clust_x_size_by_layer.eps");

   // Cluster charge by module for layer1
   TH1* clustChargeLayer1Modules_[8];
   Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   for (Int_t i=0; i<8; i++) {
	sprintf(histo, "DQMData/TrackerRecHits/Pixel/clustBPIX/Clust_charge_Layer1_Module%d;1", i+1);
        Pixel->cd(i+1);
	rfile->GetObject(histo, clustChargeLayer1Modules_[i]);
	clustChargeLayer1Modules_[i]->Draw();
   }

   Pixel->Print("Clust_charge_layer1_modules.eps");

   // Cluster charge by module for layer2
   TH1* clustChargeLayer2Modules_[8];
   Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   for (Int_t i=0; i<8; i++) {
	sprintf(histo, "DQMData/TrackerRecHits/Pixel/clustBPIX/Clust_charge_Layer2_Module%d;1", i+1);
        Pixel->cd(i+1);
	rfile->GetObject(histo, clustChargeLayer2Modules_[i]);
	clustChargeLayer2Modules_[i]->Draw();
   }

   Pixel->Print("Clust_charge_layer2_modules.eps");

   // Cluster charge by module for layer3
   TH1* clustChargeLayer3Modules_[8];
   Pixel = new TCanvas("Pixel","Pixel",800,1200);
   Pixel->Divide(2,4);

   for (Int_t i=0; i<8; i++) {
	sprintf(histo, "DQMData/TrackerRecHits/Pixel/clustBPIX/Clust_charge_Layer3_Module%d;1", i+1);
        Pixel->cd(i+1);
	rfile->GetObject(histo, clustChargeLayer3Modules_[i]);
	clustChargeLayer3Modules_[i]->Draw();
   }

   Pixel->Print("Clust_charge_layer3_modules.eps");

   //////////////////////////////////
   // Forward Clusters
   /////////////////////////////////

   // Cluster xsize for Disk1 by plaquette
   Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
   Pixel->Divide(2,4);
   TH1* clustXSizeDisk1Plaquettes_[7];

   for (Int_t i=0; i<7; i++) {
	sprintf(histo, "DQMData/TrackerRecHits/Pixel/clustFPIX/Clust_x_size_Disk1_Plaquette%d;1", i+1);
	Pixel->cd(i+1);
	rfile->GetObject(histo,clustXSizeDisk1Plaquettes_[i]);
	clustXSizeDisk1Plaquettes_[i]->Draw();
   }

   Pixel->Print("Clust_xsize_disk1_plaquettes.eps");

   // Cluster xsize for Disk2 by plaquette
   Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
   Pixel->Divide(2,4);
   TH1* clustXSizeDisk2Plaquettes_[7];

   for (Int_t i=0; i<7; i++) {
	sprintf(histo, "DQMData/TrackerRecHits/Pixel/clustFPIX/Clust_x_size_Disk2_Plaquette%d;1", i+1);
	Pixel->cd(i+1);
	rfile->GetObject(histo,clustXSizeDisk2Plaquettes_[i]);
	clustXSizeDisk2Plaquettes_[i]->Draw();
   }

   Pixel->Print("Clust_xsize_disk2_plaquettes.eps");

   // Cluster ysize for Disk1 by plaquette
   Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
   Pixel->Divide(2,4);
   TH1* clustYSizeDisk1Plaquettes_[7];

   for (Int_t i=0; i<7; i++) {
	sprintf(histo, "DQMData/TrackerRecHits/Pixel/clustFPIX/Clust_y_size_Disk1_Plaquette%d;1", i+1);
	Pixel->cd(i+1);
	rfile->GetObject(histo,clustYSizeDisk1Plaquettes_[i]);
	clustYSizeDisk1Plaquettes_[i]->Draw();
   }

   Pixel->Print("Clust_ysize_disk1_plaquettes.eps");

   // Cluster ysize for Disk2 by plaquette
   Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
   Pixel->Divide(2,4);
   TH1* clustYSizeDisk2Plaquettes_[7];

   for (Int_t i=0; i<7; i++) {
	sprintf(histo, "DQMData/TrackerRecHits/Pixel/clustFPIX/Clust_y_size_Disk2_Plaquette%d;1", i+1);
	Pixel->cd(i+1);
	rfile->GetObject(histo,clustYSizeDisk2Plaquettes_[i]);
	clustYSizeDisk2Plaquettes_[i]->Draw();
   }

   Pixel->Print("Clust_ysize_disk2_plaquettes.eps");

   //Cluster charge for Disk1 by plaquette
   Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
   Pixel->Divide(2,4);
   TH1* clustChargeDisk1Plaquettes_[7];

   for (Int_t i=0; i<7; i++) {
	sprintf(histo, "DQMData/TrackerRecHits/Pixel/clustFPIX/Clust_charge_Disk1_Plaquette%d;1", i+1);
	Pixel->cd(i+1);
	rfile->GetObject(histo,clustChargeDisk1Plaquettes_[i]);
	clustChargeDisk1Plaquettes_[i]->Draw();
   }

   Pixel->Print("Clust_charge_disk1_plaquettes.eps");

   //Cluster charge for Disk2 by plaquette
   Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
   Pixel->Divide(2,4);
   TH1* clustChargeDisk2Plaquettes_[7];

   for (Int_t i=0; i<7; i++) {
	sprintf(histo, "DQMData/TrackerRecHits/Pixel/clustFPIX/Clust_charge_Disk2_Plaquette%d;1", i+1);
	Pixel->cd(i+1);
	rfile->GetObject(histo,clustChargeDisk2Plaquettes_[i]);
	clustChargeDisk2Plaquettes_[i]->Draw();
   }

   Pixel->Print("Clust_charge_disk2_plaquettes.eps");

   //////////////////////////
   // RecHit Barrel
   ///////////////////////////

   // RecHit xres all
   Pixel = new TCanvas("Pixel", "Pixel", 200, 300);
   TH1* recHitXResAll_;

   sprintf(histo, "DQMData/TrackerRecHits/Pixel/recHitBPIX/RecHit_xres_b_All;1");
   rfile->GetObject(histo, recHitXResAll_);
   recHitXResAll_->Draw();

   Pixel->Print("RecHit_XRes_b_All.eps");

   // RecHit yres all
   Pixel = new TCanvas("Pixel", "Pixel", 200, 300);
   TH1* recHitYResAll_;

   sprintf(histo, "DQMData/TrackerRecHits/Pixel/recHitBPIX/RecHit_yres_b_All;1");
   rfile->GetObject(histo, recHitYResAll_);
   recHitYResAll_->Draw();

   Pixel->Print("RecHit_YRes_b_All.eps");

   // RecHit x distribution for full modules
   Pixel = new TCanvas("Pixel", "Pixel", 200, 300);
   TH1* recHitXFullModules_;

   sprintf(histo, "DQMData/TrackerRecHits/Pixel/recHitBPIX/RecHit_x_FullModules;1");
   rfile->GetObject(histo, recHitXFullModules_);
   recHitXFullModules_->Draw();

   Pixel->Print("RecHit_XDist_FullModules.eps");

   // RecHit x distribution for half modules
   Pixel = new TCanvas("Pixel", "Pixel", 200, 300);
   TH1* recHitXHalfModules_;

   sprintf(histo, "DQMData/TrackerRecHits/Pixel/recHitBPIX/RecHit_x_HalfModules;1");
   rfile->GetObject(histo, recHitXHalfModules_);
   recHitXHalfModules_->Draw();

   Pixel->Print("RecHit_XDist_HalfModules.eps");

   // RecHit y distribution for all modules
   Pixel = new TCanvas("Pixel", "Pixel", 200, 300);
   TH1* recHitYAllModules_;

   sprintf(histo, "DQMData/TrackerRecHits/Pixel/recHitBPIX/RecHit_y_AllModules;1");
   rfile->GetObject(histo, recHitYAllModules_);
   recHitYAllModules_->Draw();

   Pixel->Print("RecHit_YDist_AllModules.eps");

   //RecHit XRes Flipped ladders by layer
   Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
   Pixel->Divide(1,3);
   TH1* recHitXResFlippedLadderLayers_[3];

   for (Int_t i=0; i<3; i++) {
	sprintf (histo, "DQMData/TrackerRecHits/Pixel/recHitBPIX/RecHit_XRes_FlippedLadder_Layer%d;1", i+1);
	Pixel->cd(i+1);
	rfile->GetObject(histo, recHitXResFlippedLadderLayers_[i]);
	recHitXResFlippedLadderLayers_[i]->Draw();
   }

   Pixel->Print("RecHit_XRes_FlippedLadder_Layers.eps");

   //RecHit XRes UnFlipped ladders by layer
   Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
   Pixel->Divide(1,3);
   TH1* recHitXResUnFlippedLadderLayers_[3];

   for (Int_t i=0; i<3; i++) {
	sprintf (histo, "DQMData/TrackerRecHits/Pixel/recHitBPIX/RecHit_XRes_UnFlippedLadder_Layer%d;1", i+1);
	Pixel->cd(i+1);
	rfile->GetObject(histo, recHitXResUnFlippedLadderLayers_[i]);
	recHitXResUnFlippedLadderLayers_[i]->Draw();
   }

   Pixel->Print("RecHit_XRes_UnFlippedLadder_Layers.eps");

   //RecHit Y resolution by module for layer1
   Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
   Pixel->Divide(2,4);
   TH1* recHitYResLayer1Modules_[8];

   for (Int_t i=0; i<8; i++) {
	sprintf(histo, "DQMData/TrackerRecHits/Pixel/recHitBPIX/RecHit_YRes_Layer1_Module%d;1", i+1);
	Pixel->cd(i+1);
	rfile->GetObject(histo, recHitYResLayer1Modules_[i]);
	recHitYResLayer1Modules_[i]->Draw();
   }

   Pixel->Print("RecHit_YRes_Layer1_Modules.eps");

   //RecHit Y resolution by module for layer2
   Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
   Pixel->Divide(2,4);
   TH1* recHitYResLayer2Modules_[8];

   for (Int_t i=0; i<8; i++) {
	sprintf(histo, "DQMData/TrackerRecHits/Pixel/recHitBPIX/RecHit_YRes_Layer2_Module%d;1", i+1);
	Pixel->cd(i+1);
	rfile->GetObject(histo, recHitYResLayer2Modules_[i]);
	recHitYResLayer2Modules_[i]->Draw();
   }

   Pixel->Print("RecHit_YRes_Layer2_Modules.eps");

   //RecHit Y resolution by module for layer3
   Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
   Pixel->Divide(2,4);
   TH1* recHitYResLayer3Modules_[8];

   for (Int_t i=0; i<8; i++) {
	sprintf(histo, "DQMData/TrackerRecHits/Pixel/recHitBPIX/RecHit_YRes_Layer3_Module%d;1", i+1);
	Pixel->cd(i+1);
	rfile->GetObject(histo, recHitYResLayer3Modules_[i]);
	recHitYResLayer3Modules_[i]->Draw();
   }

   Pixel->Print("RecHit_YRes_Layer3_Modules.eps");

   /////////////////////
   // RecHit forward
   /////////////////////

   //RecHit xres forward
   Pixel = new TCanvas("Pixel", "Pixel", 200, 300);
   TH1* recHitXResAll_;

   sprintf(histo, "DQMData/TrackerRecHits/Pixel/recHitFPIX/RecHit_xres_f_All;1");
   rfile->GetObject(histo, recHitXResAll_);
   recHitXResAll_->Draw();

   Pixel->Print("RecHit_XRes_f_All.eps");

   //RecHit yres forward
   Pixel = new TCanvas("Pixel", "Pixel", 200, 300);
   TH1* recHitYResAll_;

   sprintf(histo, "DQMData/TrackerRecHits/Pixel/recHitFPIX/RecHit_yres_f_All;1");
   rfile->GetObject(histo, recHitYResAll_);
   recHitYResAll_->Draw();

   Pixel->Print("RecHit_YRes_f_All.eps");

   // RecHit x distribution for plaquettes x-size 1
   Pixel = new TCanvas("Pixel", "Pixel", 200, 300);

   TH1* recHitXPlaquetteXSize1_;

   sprintf(histo, "DQMData/TrackerRecHits/Pixel/recHitFPIX/RecHit_x_Plaquette_xsize1;1");
   rfile->GetObject(histo, recHitXPlaquetteXSize1_);
   recHitXPlaquetteXSize1_->Draw();

   Pixel->Print("RecHit_X_Plaquette_xsize1.eps");

   // RecHit x distribution for plaquettes x-size 2
   Pixel = new TCanvas("Pixel", "Pixel", 200, 300);

   TH1* recHitXPlaquetteXSize2_;

   sprintf(histo, "DQMData/TrackerRecHits/Pixel/recHitFPIX/RecHit_x_Plaquette_xsize2;1");
   rfile->GetObject(histo, recHitXPlaquetteXSize2_);
   recHitXPlaquetteXSize2_->Draw();

   Pixel->Print("RecHit_X_Plaquette_xsize2.eps");

   // RecHit y distribution for plaquettes y-size 2
   Pixel = new TCanvas("Pixel", "Pixel", 200, 300);

   TH1* recHitYPlaquetteYSize2_;

   sprintf(histo, "DQMData/TrackerRecHits/Pixel/recHitFPIX/RecHit_y_Plaquette_ysize2;1");
   rfile->GetObject(histo, recHitYPlaquetteYSize2_);
   recHitYPlaquetteYSize2_->Draw();

   Pixel->Print("RecHit_Y_Plaquette_ysize2.eps");

   // RecHit y distribution for plaquettes y-size 3
   Pixel = new TCanvas("Pixel", "Pixel", 200, 300);

   TH1* recHitYPlaquetteYSize3_;

   sprintf(histo, "DQMData/TrackerRecHits/Pixel/recHitFPIX/RecHit_y_Plaquette_ysize3;1");
   rfile->GetObject(histo, recHitYPlaquetteYSize3_);
   recHitYPlaquetteYSize3_->Draw();

   Pixel->Print("RecHit_Y_Plaquette_ysize3.eps");

   // RecHit y distribution for plaquettes y-size 4
   Pixel = new TCanvas("Pixel", "Pixel", 200, 300);

   TH1* recHitYPlaquetteYSize4_;

   sprintf(histo, "DQMData/TrackerRecHits/Pixel/recHitFPIX/RecHit_y_Plaquette_ysize4;1");
   rfile->GetObject(histo, recHitYPlaquetteYSize4_);
   recHitYPlaquetteYSize4_->Draw();

   Pixel->Print("RecHit_Y_Plaquette_ysize4.eps");

   // RecHit y distribution for plaquettes y-size 5
   Pixel = new TCanvas("Pixel", "Pixel", 200, 300);

   TH1* recHitYPlaquetteYSize5_;

   sprintf(histo, "DQMData/TrackerRecHits/Pixel/recHitFPIX/RecHit_y_Plaquette_ysize5;1");
   rfile->GetObject(histo, recHitYPlaquetteYSize5_);
   recHitYPlaquetteYSize5_->Draw();

   Pixel->Print("RecHit_Y_Plaquette_ysize5.eps");

   //RecHit x resolution by plaquette for Disk1
   Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
   Pixel->Divide(2,4);
   TH1* recHitXResDisk1Plaquettes_[7];

   for (Int_t i=0; i<7; i++) {
	Pixel->cd(i+1);
	sprintf(histo, "DQMData/TrackerRecHits/Pixel/recHitFPIX/RecHit_XRes_Disk1_Plaquette%d;1", i+1);
	rfile->GetObject(histo, recHitXResDisk1Plaquettes_[i]);
	recHitXResDisk1Plaquettes_[i]->Draw();
   }

   Pixel->Print("RecHit_XRes_disk1_plaquettes.eps");

   //RecHit x resolution by plaquette for Disk2
   Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
   Pixel->Divide(2,4);
   TH1* recHitXResDisk2Plaquettes_[7];

   for (Int_t i=0; i<7; i++) {
	Pixel->cd(i+1);
	sprintf(histo, "DQMData/TrackerRecHits/Pixel/recHitFPIX/RecHit_XRes_Disk2_Plaquette%d;1", i+1);
	rfile->GetObject(histo, recHitXResDisk2Plaquettes_[i]);
	recHitXResDisk2Plaquettes_[i]->Draw();
   }

   Pixel->Print("RecHit_XRes_disk2_plaquettes.eps");

   //RecHit y resolution by plaquette for Disk1
   Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
   Pixel->Divide(2,4);
   TH1* recHitYResDisk1Plaquettes_[7];

   for (Int_t i=0; i<7; i++) {
	Pixel->cd(i+1);
	sprintf(histo, "DQMData/TrackerRecHits/Pixel/recHitFPIX/RecHit_YRes_Disk1_Plaquette%d;1", i+1);
	rfile->GetObject(histo, recHitYResDisk1Plaquettes_[i]);
	recHitYResDisk1Plaquettes_[i]->Draw();
   }

   Pixel->Print("RecHit_YRes_disk1_plaquettes.eps");

   //RecHit y resolution by plaquette for Disk2
   Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
   Pixel->Divide(2,4);
   TH1* recHitYResDisk2Plaquettes_[7];

   for (Int_t i=0; i<7; i++) {
	Pixel->cd(i+1);
	sprintf(histo, "DQMData/TrackerRecHits/Pixel/recHitFPIX/RecHit_YRes_Disk2_Plaquette%d;1", i+1);
	rfile->GetObject(histo, recHitYResDisk2Plaquettes_[i]);
	recHitYResDisk2Plaquettes_[i]->Draw();
   }

   Pixel->Print("RecHit_YRes_disk2_plaquettes.eps");
} // end
