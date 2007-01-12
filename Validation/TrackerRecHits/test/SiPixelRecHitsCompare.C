void SiPixelRecHitsCompare()
{
   gROOT->Reset();
   char* rfilename = "pixelrechitshisto.root";
   char* sfilename = "../data/pixelrechitshisto.root";

   delete gROOT->GetListOfFiles()->FindObject(rfilename);
   delete gROOT->GetListOfFiles()->FindObject(sfilename);

   TText * te = new TText();
   TFile * rfile = new TFile(rfilename);
   TFile * sfile = new TFile(sfilename);

   rfile->cd("DQMData");
   sfile->cd("DQMData");

   gROOT->ProcessLine(".x HistoCompare.C");    
   HistoCompare * myPV = new HistoCompare();

   Char_t histo[200];

   TCanvas * pixel;

   ////////////////////////////
   // Barrel Clusters
   ///////////////////////////

   if (1) {
	//Cluster y size by module

	Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
	Pixel->Divide(2,4);

	TH1* clustYSizeModule_[8];
	TH1* newclustYSizeModule_[8];

	for (Int_t i=0; i<8; i++) {
	   sprintf(histo, "DQMData/clustBPIX/Clust_y_size_Module%d;1", i+1);
	   rfile->GetObject(histo, clustYSizeModule_[i]);
	   sfile->GetObject(histo, newclustYSizeModule_[i]);
	   Pixel->cd(i+1);
	   gPad->SetLogy();
	   clustYSizeModule_[i]->SetLineColor(2);
	   clustYSizeModule_[i]->Draw();
	   newclustYSizeModule_[i]->SetLineColor(4);
	   newclustYSizeModule_[i]->SetLineStyle(2);
	   newclustYSizeModule_[i]->Draw("Sames");
	   myPV->PVCompute(clustYSizeModule_[i], newclustYSizeModule_[i], te);
	}
	Pixel->Print("Clust_y_size_by_module.eps");   
	Pixel->Print("Clust_y_size_by_module.gif");   
   }

   if (1) {
	//Cluster x size by layer

	Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
	Pixel->Divide(1,3);

	TH1* clustXSizeLayer_[3];
	TH1* newclustXSizeLayer_[3];

	for (Int_t i=0; i<3; i++) {
	   sprintf(histo, "DQMData/clustBPIX/Clust_x_size_Layer%d;1", i+1);
	   rfile->GetObject(histo, clustXSizeLayer_[i]);
	   sfile->GetObject(histo, newclustXSizeLayer_[i]);
	   Pixel->cd(i+1);
	   gPad->SetLogy();
	   clustXSizeLayer_[i]->SetLineColor(2);
	   clustXSizeLayer_[i]->Draw();
	   newclustXSizeLayer_[i]->SetLineColor(4);
	   newclustXSizeLayer_[i]->SetLineStyle(2);
	   newclustXSizeLayer_[i]->Draw("Sames");
	   myPV->PVCompute(clustXSizeLayer_[i], newclustXSizeLayer_[i], te);
	}
	Pixel->Print("Clust_x_size_by_layer.eps");   
	Pixel->Print("Clust_x_size_by_layer.gif");   
   }

   if (1) {
	//Cluster charge for Layer 1 by module

	Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
	Pixel->Divide(2,4);

	TH1* clustChargeLayer1Modules_[8];
	TH1* newclustChargeLayer1Modules_[8];

	for (Int_t i=0; i<8; i++) {
	   sprintf(histo, "DQMData/clustBPIX/Clust_charge_Layer1_Module%d;1", i+1);
	   rfile->GetObject(histo, clustChargeLayer1Modules_[i]);
	   sfile->GetObject(histo, newclustChargeLayer1Modules_[i]);
	   Pixel->cd(i+1);
	   gPad->SetLogy();
	   clustChargeLayer1Modules_[i]->SetLineColor(2);
	   clustChargeLayer1Modules_[i]->Draw();
	   newclustChargeLayer1Modules_[i]->SetLineColor(4);
	   newclustChargeLayer1Modules_[i]->SetLineStyle(2);
	   newclustChargeLayer1Modules_[i]->Draw("Sames");
	   myPV->PVCompute(clustChargeLayer1Modules_[i], newclustChargeLayer1Modules_[i], te);
	}
	Pixel->Print("Clust_charge_layer1_modules.eps");   
	Pixel->Print("Clust_charge_layer1_modules.gif");   
   }

   if (1) {
	//Cluster charge for Layer 2 by module

	Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
	Pixel->Divide(2,4);

	TH1* clustChargeLayer2Modules_[8];
	TH1* newclustChargeLayer2Modules_[8];

	for (Int_t i=0; i<8; i++) {
	   sprintf(histo, "DQMData/clustBPIX/Clust_charge_Layer2_Module%d;1", i+1);
	   rfile->GetObject(histo, clustChargeLayer2Modules_[i]);
	   sfile->GetObject(histo, newclustChargeLayer2Modules_[i]);
	   Pixel->cd(i+1);
	   gPad->SetLogy();
	   clustChargeLayer2Modules_[i]->SetLineColor(2);
	   clustChargeLayer2Modules_[i]->Draw();
	   newclustChargeLayer2Modules_[i]->SetLineColor(4);
	   newclustChargeLayer2Modules_[i]->SetLineStyle(2);
	   newclustChargeLayer2Modules_[i]->Draw("Sames");
	   myPV->PVCompute(clustChargeLayer2Modules_[i], newclustChargeLayer2Modules_[i], te);
	}
	Pixel->Print("Clust_charge_layer2_modules.eps");   
	Pixel->Print("Clust_charge_layer2_modules.gif");   
   }

   if (1) {
	//Cluster charge for Layer 3 by module

	Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
	Pixel->Divide(2,4);

	TH1* clustChargeLayer3Modules_[8];
	TH1* newclustChargeLayer3Modules_[8];

	for (Int_t i=0; i<8; i++) {
	   sprintf(histo, "DQMData/clustBPIX/Clust_charge_Layer3_Module%d;1", i+1);
	   rfile->GetObject(histo, clustChargeLayer3Modules_[i]);
	   sfile->GetObject(histo, newclustChargeLayer3Modules_[i]);
	   Pixel->cd(i+1);
	   gPad->SetLogy();
	   clustChargeLayer3Modules_[i]->SetLineColor(2);
	   clustChargeLayer3Modules_[i]->Draw();
	   newclustChargeLayer3Modules_[i]->SetLineColor(4);
	   newclustChargeLayer3Modules_[i]->SetLineStyle(2);
	   newclustChargeLayer3Modules_[i]->Draw("Sames");
	   myPV->PVCompute(clustChargeLayer3Modules_[i], newclustChargeLayer3Modules_[i], te);
	}
	Pixel->Print("Clust_charge_layer3_modules.eps");   
	Pixel->Print("Clust_charge_layer3_modules.gif");   
   }

   ///////////////////////////////////////////
   // Forward clusters
   //////////////////////////////////////////

   if (1) {
	// Cluster xsize for Disk1 by plaquette

	Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
	Pixel->Divide(2,4);

	TH1* clustXSizeDisk1Plaquettes_[7];
	TH1* newclustXSizeDisk1Plaquettes_[7];

	for (Int_t i=0; i<7; i++) {
	   sprintf(histo, "DQMData/clustFPIX/Clust_x_size_Disk1_Plaquette%d;1", i+1);
	   rfile->GetObject(histo, clustXSizeDisk1Plaquettes_[i]);
	   sfile->GetObject(histo, newclustXSizeDisk1Plaquettes_[i]);
	   Pixel->cd(i+1);
	   gPad->SetLogy();
	   clustXSizeDisk1Plaquettes_[i]->SetLineColor(2);
	   clustXSizeDisk1Plaquettes_[i]->Draw();
	   newclustXSizeDisk1Plaquettes_[i]->SetLineColor(4);
	   newclustXSizeDisk1Plaquettes_[i]->SetLineStyle(2);
	   newclustXSizeDisk1Plaquettes_[i]->Draw("Sames");
	   myPV->PVCompute(clustXSizeDisk1Plaquettes_[i], newclustXSizeDisk1Plaquettes_[i], te);
	}
	Pixel->Print("Clust_xsize_disk1_plaquettes.eps");   
	Pixel->Print("Clust_xsize_disk1_plaquettes.gif");   
   }

   if (1) {
	//Cluster xsize for Disk2 by plaquette

	Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
	Pixel->Divide(2,4);

	TH1* clustXSizeDisk2Plaquettes_[7];
	TH1* newclustXSizeDisk2Plaquettes_[7];

	for (Int_t i=0; i<7; i++) {
	   sprintf(histo, "DQMData/clustFPIX/Clust_x_size_Disk2_Plaquette%d;1", i+1);
	   rfile->GetObject(histo, clustXSizeDisk2Plaquettes_[i]);
	   sfile->GetObject(histo, newclustXSizeDisk2Plaquettes_[i]);
	   Pixel->cd(i+1);
	   gPad->SetLogy();
	   clustXSizeDisk2Plaquettes_[i]->SetLineColor(2);
	   clustXSizeDisk2Plaquettes_[i]->Draw();
	   newclustXSizeDisk2Plaquettes_[i]->SetLineColor(4);
	   newclustXSizeDisk2Plaquettes_[i]->SetLineStyle(2);
	   newclustXSizeDisk2Plaquettes_[i]->Draw("Sames");
	   myPV->PVCompute(clustXSizeDisk2Plaquettes_[i], newclustXSizeDisk2Plaquettes_[i], te);
	}
	Pixel->Print("Clust_xsize_disk2_plaquettes.eps");   
	Pixel->Print("Clust_xsize_disk2_plaquettes.gif");   
   }


   if (1) {
	// Cluster ysize for Disk1 by plaquette

	Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
	Pixel->Divide(2,4);

	TH1* clustYSizeDisk1Plaquettes_[7];
	TH1* newclustYSizeDisk1Plaquettes_[7];

	for (Int_t i=0; i<7; i++) {
	   sprintf(histo, "DQMData/clustFPIX/Clust_y_size_Disk1_Plaquette%d;1", i+1);
	   rfile->GetObject(histo, clustYSizeDisk1Plaquettes_[i]);
	   sfile->GetObject(histo, newclustYSizeDisk1Plaquettes_[i]);
	   Pixel->cd(i+1);
	   gPad->SetLogy();
	   clustYSizeDisk1Plaquettes_[i]->SetLineColor(2);
	   clustYSizeDisk1Plaquettes_[i]->Draw();
	   newclustYSizeDisk1Plaquettes_[i]->SetLineColor(4);
	   newclustYSizeDisk1Plaquettes_[i]->SetLineStyle(2);
	   newclustYSizeDisk1Plaquettes_[i]->Draw("Sames");
	   myPV->PVCompute(clustYSizeDisk1Plaquettes_[i], newclustYSizeDisk1Plaquettes_[i], te);
	}
	Pixel->Print("Clust_ysize_disk1_plaquettes.eps");   
	Pixel->Print("Clust_ysize_disk1_plaquettes.gif");   
   }

   if (1) {
	//Cluster ysize for Disk2 by plaquette

	Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
	Pixel->Divide(2,4);

	TH1* clustYSizeDisk2Plaquettes_[7];
	TH1* newclustYSizeDisk2Plaquettes_[7];

	for (Int_t i=0; i<7; i++) {
	   sprintf(histo, "DQMData/clustFPIX/Clust_y_size_Disk2_Plaquette%d;1", i+1);
	   rfile->GetObject(histo, clustYSizeDisk2Plaquettes_[i]);
	   sfile->GetObject(histo, newclustYSizeDisk2Plaquettes_[i]);
	   Pixel->cd(i+1);
	   gPad->SetLogy();
	   clustYSizeDisk2Plaquettes_[i]->SetLineColor(2);
	   clustYSizeDisk2Plaquettes_[i]->Draw();
	   newclustYSizeDisk2Plaquettes_[i]->SetLineColor(4);
	   newclustYSizeDisk2Plaquettes_[i]->SetLineStyle(2);
	   newclustYSizeDisk2Plaquettes_[i]->Draw("Sames");
	   myPV->PVCompute(clustYSizeDisk2Plaquettes_[i], newclustYSizeDisk2Plaquettes_[i], te);
	}
	Pixel->Print("Clust_ysize_disk2_plaquettes.eps");   
	Pixel->Print("Clust_ysize_disk2_plaquettes.gif");   
   }


   if (1) {
	//Cluster charge for Disk1 by plaquette

	Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
	Pixel->Divide(2,4);

	TH1* clustChargeDisk1Plaquettes_[7];
	TH1* newclustChargeDisk1Plaquettes_[7];

	for (Int_t i=0; i<7; i++) {
	   sprintf(histo, "DQMData/clustFPIX/Clust_charge_Disk1_Plaquette%d;1", i+1);
	   rfile->GetObject(histo, clustChargeDisk1Plaquettes_[i]);
	   sfile->GetObject(histo, newclustChargeDisk1Plaquettes_[i]);
	   Pixel->cd(i+1);
	   gPad->SetLogy();
	   clustChargeDisk1Plaquettes_[i]->SetLineColor(2);
	   clustChargeDisk1Plaquettes_[i]->Draw();
	   newclustChargeDisk1Plaquettes_[i]->SetLineColor(4);
	   newclustChargeDisk1Plaquettes_[i]->SetLineStyle(2);
	   newclustChargeDisk1Plaquettes_[i]->Draw("Sames");
	   myPV->PVCompute(clustChargeDisk1Plaquettes_[i], newclustChargeDisk1Plaquettes_[i], te);
	}
	Pixel->Print("Clust_charge_disk1_plaquettes.eps");   
	Pixel->Print("Clust_charge_disk1_plaquettes.gif");   
   }

   if (1) {
	//Cluster charge for Disk2 by plaquette

	Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
	Pixel->Divide(2,4);

	TH1* clustChargeDisk2Plaquettes_[7];
	TH1* newclustChargeDisk2Plaquettes_[7];

	for (Int_t i=0; i<7; i++) {
	   sprintf(histo, "DQMData/clustFPIX/Clust_charge_Disk2_Plaquette%d;1", i+1);
	   rfile->GetObject(histo, clustChargeDisk2Plaquettes_[i]);
	   sfile->GetObject(histo, newclustChargeDisk2Plaquettes_[i]);
	   Pixel->cd(i+1);
	   gPad->SetLogy();
	   clustChargeDisk2Plaquettes_[i]->SetLineColor(2);
	   clustChargeDisk2Plaquettes_[i]->Draw();
	   newclustChargeDisk2Plaquettes_[i]->SetLineColor(4);
	   newclustChargeDisk2Plaquettes_[i]->SetLineStyle(2);
	   newclustChargeDisk2Plaquettes_[i]->Draw("Sames");
	   myPV->PVCompute(clustChargeDisk2Plaquettes_[i], newclustChargeDisk2Plaquettes_[i], te);
	}
	Pixel->Print("Clust_charge_disk2_plaquettes.eps");   
	Pixel->Print("Clust_charge_disk2_plaquettes.gif");   
   }
   
   /////////////////////////
   // RecHit Barrel
   /////////////////////////

   if (1) {
	//RecHit x distribution for Full Modules

	Pixel = new TCanvas("Pixel", "Pixel", 200, 300);
	TH1* recHitXFullModules_;
	TH1* newrecHitXFullModules_;

	sprintf (histo, "DQMData/recHitBPIX/RecHit_x_FullModules;1");
	rfile->GetObject(histo, recHitXFullModules_);
	sfile->GetObject(histo, newrecHitXFullModules_);
   
	gPad->SetLogy();
	recHitXFullModules_->SetLineColor(2);
	recHitXFullModules_->Draw();
	newrecHitXFullModules_->SetLineColor(4);
	newrecHitXFullModules_->SetLineStyle(2);
	newrecHitXFullModules_->Draw("Sames");
	myPV->PVCompute(recHitXFullModules_, newrecHitXFullModules_, te);

	Pixel->Print("RecHit_XDist_FullModules.eps");
	Pixel->Print("RecHit_XDist_FullModules.gif");

	//RecHit x distribution half modules

	Pixel = new TCanvas("Pixel", "Pixel", 200, 300);
	TH1* recHitXHalfModules_;
	TH1* newrecHitXHalfModules_;
  
	sprintf (histo, "DQMData/recHitBPIX/RecHit_x_HalfModules;1");
	rfile->GetObject(histo, recHitXHalfModules_);
	sfile->GetObject(histo, newrecHitXHalfModules_);

	gPad->SetLogy();
	recHitXHalfModules_->SetLineColor(2);
	recHitXHalfModules_->Draw();
	newrecHitXHalfModules_->SetLineColor(4);
	newrecHitXHalfModules_->SetLineStyle(2);
	newrecHitXHalfModules_->Draw("Sames");
	myPV->PVCompute(recHitXHalfModules_, newrecHitXHalfModules_, te);

	Pixel->Print("RecHit_XDist_HalfModules.eps");
	Pixel->Print("RecHit_XDist_HalfModules.gif");

	//RecHit y distribution all modules

	Pixel = new TCanvas("Pixel", "Pixel", 200, 300);
	TH1* recHitYAllModules_;
	TH1* newrecHitYAllModules_;
  
	sprintf (histo, "DQMData/recHitBPIX/RecHit_y_AllModules;1");
	rfile->GetObject(histo, recHitYAllModules_);
	sfile->GetObject(histo, newrecHitYAllModules_);

	gPad->SetLogy();
	recHitYAllModules_->SetLineColor(2);
	recHitYAllModules_->Draw();
	newrecHitYAllModules_->SetLineColor(4);
	newrecHitYAllModules_->SetLineStyle(2);
	newrecHitYAllModules_->Draw("Sames");
	myPV->PVCompute(recHitYAllModules_, newrecHitYAllModules_, te);

	Pixel->Print("RecHit_YDist_AllModules.eps");
	Pixel->Print("RecHit_YDist_AllModules.gif");
   }

   if (1) {
	TH1* recHitXResFlippedLadderLayers_[3];
	TH1* newrecHitXResFlippedLadderLayers_[3];
	Pixel = new TCanvas("Pixel", "Pixel", 400, 600);
	Pixel->Divide(1,3);

	for (Int_t i=0; i<3; i++) {
	   //RecHit XRes Flipped ladders by layer

	   sprintf(histo, "DQMData/recHitBPIX/RecHit_XRes_FlippedLadder_Layer%d;1", i+1);
	   rfile->GetObject(histo, recHitXResFlippedLadderLayers_[i]);
	   sfile->GetObject(histo, newrecHitXResFlippedLadderLayers_[i]);

	   Pixel->cd(i+1);
	   gPad->SetLogy();
	   recHitXResFlippedLadderLayers_[i]->SetLineColor(2);
	   recHitXResFlippedLadderLayers_[i]->Draw();
	   newrecHitXResFlippedLadderLayers_[i]->SetLineColor(4);
	   newrecHitXResFlippedLadderLayers_[i]->SetLineStyle(2);
	   newrecHitXResFlippedLadderLayers_[i]->Draw("Sames");
	   myPV->PVCompute(recHitXResFlippedLadderLayers_[i], newrecHitXResFlippedLadderLayers_[i], te);
	   
	}
	Pixel->Print("RecHit_XRes_FlippedLadder_Layers.eps");
	Pixel->Print("RecHit_XRes_FlippedLadder_Layers.gif");
   }

   if (1) {
	TH1* recHitXResUnFlippedLadderLayers_[3];
	TH1* newrecHitXResUnFlippedLadderLayers_[3];
	Pixel = new TCanvas("Pixel", "Pixel", 400, 600);
	Pixel->Divide(1,3);

	for (Int_t i=0; i<3; i++) {
	   //RecHit XRes unflipped ladders by layer

	   sprintf(histo, "DQMData/recHitBPIX/RecHit_XRes_UnFlippedLadder_Layer%d;1", i+1);
	   rfile->GetObject(histo, recHitXResUnFlippedLadderLayers_[i]);
	   sfile->GetObject(histo, newrecHitXResUnFlippedLadderLayers_[i]);

	   Pixel->cd(i+1);
	   gPad->SetLogy();
	   recHitXResUnFlippedLadderLayers_[i]->SetLineColor(2);
	   recHitXResUnFlippedLadderLayers_[i]->Draw();
	   newrecHitXResUnFlippedLadderLayers_[i]->SetLineColor(4);
	   newrecHitXResUnFlippedLadderLayers_[i]->SetLineStyle(2);
	   newrecHitXResUnFlippedLadderLayers_[i]->Draw("Sames");
	   myPV->PVCompute(recHitXResUnFlippedLadderLayers_[i], newrecHitXResUnFlippedLadderLayers_[i], te);
	   
	}
	Pixel->Print("RecHit_XRes_UnFlippedLadder_Layers.eps");
	Pixel->Print("RecHit_XRes_UnFlippedLadder_Layers.gif");
   }

   if (1) {
	TH1* recHitYResLayer1Modules_[8];
	TH1* newrecHitYResLayer1Modules_[8];
	Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
	Pixel->Divide(2,4);

	for (Int_t i=0; i<8; i++) {
	   //RecHit Y resolution by module for layer 1

	   sprintf(histo, "DQMData/recHitBPIX/RecHit_YRes_Layer1_Module%d;1", i+1);
	   rfile->GetObject(histo, recHitYResLayer1Modules_[i]);
	   sfile->GetObject(histo, newrecHitYResLayer1Modules_[i]);

	   Pixel->cd(i+1);
	   gPad->SetLogy();
	   recHitYResLayer1Modules_[i]->SetLineColor(2);
	   recHitYResLayer1Modules_[i]->Draw();
	   newrecHitYResLayer1Modules_[i]->SetLineColor(4);
	   newrecHitYResLayer1Modules_[i]->SetLineStyle(2);
	   newrecHitYResLayer1Modules_[i]->Draw("Sames");
	   myPV->PVCompute(recHitYResLayer1Modules_[i], newrecHitYResLayer1Modules_[i], te);
	   
	}
	Pixel->Print("RecHit_YRes_Layer1_Modules.eps");
	Pixel->Print("RecHit_YRes_Layer1_Modules.gif");
   }

   if (1) {
	TH1* recHitYResLayer2Modules_[8];
	TH1* newrecHitYResLayer2Modules_[8];
	Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
	Pixel->Divide(2,4);

	for (Int_t i=0; i<8; i++) {
	   //RecHit Y resolution by module for layer 2

	   sprintf(histo, "DQMData/recHitBPIX/RecHit_YRes_Layer2_Module%d;1", i+1);
	   rfile->GetObject(histo, recHitYResLayer2Modules_[i]);
	   sfile->GetObject(histo, newrecHitYResLayer2Modules_[i]);

	   Pixel->cd(i+1);
	   gPad->SetLogy();
	   recHitYResLayer2Modules_[i]->SetLineColor(2);
	   recHitYResLayer2Modules_[i]->Draw();
	   newrecHitYResLayer2Modules_[i]->SetLineColor(4);
	   newrecHitYResLayer2Modules_[i]->SetLineStyle(2);
	   newrecHitYResLayer2Modules_[i]->Draw("Sames");
	   myPV->PVCompute(recHitYResLayer2Modules_[i], newrecHitYResLayer2Modules_[i], te);
	   
	}
	Pixel->Print("RecHit_YRes_Layer2_Modules.eps");
	Pixel->Print("RecHit_YRes_Layer2_Modules.gif");
   }

   if (1) {
	TH1* recHitYResLayer3Modules_[8];
	TH1* newrecHitYResLayer3Modules_[8];
	Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
	Pixel->Divide(2,4);

	for (Int_t i=0; i<8; i++) {
	   //RecHit Y resolution by module for layer 3

	   sprintf(histo, "DQMData/recHitBPIX/RecHit_YRes_Layer3_Module%d;1", i+1);
	   rfile->GetObject(histo, recHitYResLayer3Modules_[i]);
	   sfile->GetObject(histo, newrecHitYResLayer3Modules_[i]);

	   Pixel->cd(i+1);
	   gPad->SetLogy();
	   recHitYResLayer3Modules_[i]->SetLineColor(2);
	   recHitYResLayer3Modules_[i]->Draw();
	   newrecHitYResLayer3Modules_[i]->SetLineColor(4);
	   newrecHitYResLayer3Modules_[i]->SetLineStyle(2);
	   newrecHitYResLayer3Modules_[i]->Draw("Sames");
	   myPV->PVCompute(recHitYResLayer3Modules_[i], newrecHitYResLayer3Modules_[i], te);
	   
	}
	Pixel->Print("RecHit_YRes_Layer3_Modules.eps");
	Pixel->Print("RecHit_YRes_Layer3_Modules.gif");
   }

   ////////////////////////////////
   //RecHit forward
   ////////////////////////////////

   if (1) {
	//RecHit x distribution for plaquettes x-size 1

	Pixel = new TCanvas("Pixel", "Pixel", 200, 300);
	
	TH1* recHitXPlaquetteXSize1_;
	TH1* newrecHitXPlaquetteXSize1_;

	sprintf(histo, "DQMData/recHitFPIX/RecHit_x_Plaquette_xsize1;1");
	rfile->GetObject(histo, recHitXPlaquetteXSize1_);
	sfile->GetObject(histo, newrecHitXPlaquetteXSize1_);

	gPad->SetLogy();
	recHitXPlaquetteXSize1_->SetLineColor(2);
	recHitXPlaquetteXSize1_->Draw();
	newrecHitXPlaquetteXSize1_->SetLineColor(4);
	newrecHitXPlaquetteXSize1_->SetLineStyle(2);
	newrecHitXPlaquetteXSize1_->Draw("Sames");
	myPV->PVCompute(recHitXPlaquetteXSize1_, newrecHitXPlaquetteXSize1_, te);

	Pixel->Print("RecHit_X_Plaquette_xsize1.eps");
	Pixel->Print("RecHit_X_Plaquette_xsize1.gif");

	//RecHit x distribution for plaquettes x-size 2

	Pixel = new TCanvas("Pixel", "Pixel", 200, 300);
	
	TH1* recHitXPlaquetteXSize2_;
	TH1* newrecHitXPlaquetteXSize2_;

	sprintf(histo, "DQMData/recHitFPIX/RecHit_x_Plaquette_xsize2;1");
	rfile->GetObject(histo, recHitXPlaquetteXSize2_);
	sfile->GetObject(histo, newrecHitXPlaquetteXSize2_);

	gPad->SetLogy();
	recHitXPlaquetteXSize2_->SetLineColor(2);
	recHitXPlaquetteXSize2_->Draw();
	newrecHitXPlaquetteXSize2_->SetLineColor(4);
	newrecHitXPlaquetteXSize2_->SetLineStyle(2);
	newrecHitXPlaquetteXSize2_->Draw("Sames");
	myPV->PVCompute(recHitXPlaquetteXSize2_, newrecHitXPlaquetteXSize2_, te);

	Pixel->Print("RecHit_X_Plaquette_xsize2.eps");
	Pixel->Print("RecHit_X_Plaquette_xsize2.gif");

	//RecHit y distribution for plaquettes y-size 2

	Pixel = new TCanvas("Pixel", "Pixel", 200, 300);
	
	TH1* recHitYPlaquetteYSize2_;
	TH1* newrecHitYPlaquetteYSize2_;

	sprintf(histo, "DQMData/recHitFPIX/RecHit_y_Plaquette_ysize2;1");
	rfile->GetObject(histo, recHitYPlaquetteYSize2_);
	sfile->GetObject(histo, newrecHitYPlaquetteYSize2_);

	gPad->SetLogy();
	recHitYPlaquetteYSize2_->SetLineColor(2);
	recHitYPlaquetteYSize2_->Draw();
	newrecHitYPlaquetteYSize2_->SetLineColor(4);
	newrecHitYPlaquetteYSize2_->SetLineStyle(2);
	newrecHitYPlaquetteYSize2_->Draw("Sames");
	myPV->PVCompute(recHitYPlaquetteYSize2_, newrecHitYPlaquetteYSize2_, te);

	Pixel->Print("RecHit_Y_Plaquette_ysize2.eps");
	Pixel->Print("RecHit_Y_Plaquette_ysize2.gif");

	//RecHit y distribution for plaquettes y-size 3

	Pixel = new TCanvas("Pixel", "Pixel", 200, 300);
	
	TH1* recHitYPlaquetteYSize3_;
	TH1* newrecHitYPlaquetteYSize3_;

	sprintf(histo, "DQMData/recHitFPIX/RecHit_y_Plaquette_ysize3;1");
	rfile->GetObject(histo, recHitYPlaquetteYSize3_);
	sfile->GetObject(histo, newrecHitYPlaquetteYSize3_);

	gPad->SetLogy();
	recHitYPlaquetteYSize3_->SetLineColor(2);
	recHitYPlaquetteYSize3_->Draw();
	newrecHitYPlaquetteYSize3_->SetLineColor(4);
	newrecHitYPlaquetteYSize3_->SetLineStyle(2);
	newrecHitYPlaquetteYSize3_->Draw("Sames");
	myPV->PVCompute(recHitYPlaquetteYSize3_, newrecHitYPlaquetteYSize3_, te);

	Pixel->Print("RecHit_Y_Plaquette_ysize3.eps");
	Pixel->Print("RecHit_Y_Plaquette_ysize3.gif");

	//RecHit y distribution for plaquettes y-size 4

	Pixel = new TCanvas("Pixel", "Pixel", 200, 300);
	
	TH1* recHitYPlaquetteYSize4_;
	TH1* newrecHitYPlaquetteYSize4_;

	sprintf(histo, "DQMData/recHitFPIX/RecHit_y_Plaquette_ysize4;1");
	rfile->GetObject(histo, recHitYPlaquetteYSize4_);
	sfile->GetObject(histo, newrecHitYPlaquetteYSize4_);

	gPad->SetLogy();
	recHitYPlaquetteYSize4_->SetLineColor(2);
	recHitYPlaquetteYSize4_->Draw();
	newrecHitYPlaquetteYSize4_->SetLineColor(4);
	newrecHitYPlaquetteYSize4_->SetLineStyle(2);
	newrecHitYPlaquetteYSize4_->Draw("Sames");
	myPV->PVCompute(recHitYPlaquetteYSize4_, newrecHitYPlaquetteYSize4_, te);

	Pixel->Print("RecHit_Y_Plaquette_ysize4.eps");
	Pixel->Print("RecHit_Y_Plaquette_ysize4.gif");

	//RecHit y distribution for plaquettes y-size 5

	Pixel = new TCanvas("Pixel", "Pixel", 200, 300);
	
	TH1* recHitYPlaquetteYSize5_;
	TH1* newrecHitYPlaquetteYSize5_;

	sprintf(histo, "DQMData/recHitFPIX/RecHit_y_Plaquette_ysize5;1");
	rfile->GetObject(histo, recHitYPlaquetteYSize5_);
	sfile->GetObject(histo, newrecHitYPlaquetteYSize5_);

	gPad->SetLogy();
	recHitYPlaquetteYSize5_->SetLineColor(2);
	recHitYPlaquetteYSize5_->Draw();
	newrecHitYPlaquetteYSize5_->SetLineColor(4);
	newrecHitYPlaquetteYSize5_->SetLineStyle(2);
	newrecHitYPlaquetteYSize5_->Draw("Sames");
	myPV->PVCompute(recHitYPlaquetteYSize5_, newrecHitYPlaquetteYSize5_, te);

	Pixel->Print("RecHit_Y_Plaquette_ysize5.eps");
	Pixel->Print("RecHit_Y_Plaquette_ysize5.gif");
   }

   if (1) {
	TH1* recHitXResDisk1Plaquettes_[7];
	TH1* newrecHitXResDisk1Plaquettes_[7];
        Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
	Pixel->Divide(2,4);

	for (Int_t i=0; i<7; i++) {
	   //RecHit X resolution by plaquette for Disk1

	   sprintf(histo, "DQMData/recHitFPIX/RecHit_XRes_Disk1_Plaquette%d;1", i+1);
	   rfile->GetObject(histo, recHitXResDisk1Plaquettes_[i]);
	   sfile->GetObject(histo, newrecHitXResDisk1Plaquettes_[i]);

	   Pixel->cd(i+1);
	   gPad->SetLogy();
	   recHitXResDisk1Plaquettes_[i]->SetLineColor(2);
	   recHitXResDisk1Plaquettes_[i]->Draw();
	   newrecHitXResDisk1Plaquettes_[i]->SetLineColor(4);
	   newrecHitXResDisk1Plaquettes_[i]->SetLineStyle(2);
	   newrecHitXResDisk1Plaquettes_[i]->Draw("Sames");
	   myPV->PVCompute(recHitXResDisk1Plaquettes_[i], newrecHitXResDisk1Plaquettes_[i], te);
	   
	}
	Pixel->Print("RecHit_XRes_disk1_plaquettes.eps");
	Pixel->Print("RecHit_XRes_disk1_plaquettes.gif");
   }

   if (1) {
	TH1* recHitXResDisk2Plaquettes_[7];
	TH1* newrecHitXResDisk2Plaquettes_[7];
	Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
	Pixel->Divide(2,4);

	for (Int_t i=0; i<7; i++) {
	   //RecHit X resolution by plaquette for Disk2

	   sprintf(histo, "DQMData/recHitFPIX/RecHit_XRes_Disk2_Plaquette%d;1", i+1);
	   rfile->GetObject(histo, recHitXResDisk2Plaquettes_[i]);
	   sfile->GetObject(histo, newrecHitXResDisk2Plaquettes_[i]);

	   Pixel->cd(i+1);
	   gPad->SetLogy();
	   recHitXResDisk2Plaquettes_[i]->SetLineColor(2);
	   recHitXResDisk2Plaquettes_[i]->Draw();
	   newrecHitXResDisk2Plaquettes_[i]->SetLineColor(4);
	   newrecHitXResDisk2Plaquettes_[i]->SetLineStyle(2);
	   newrecHitXResDisk2Plaquettes_[i]->Draw("Sames");
	   myPV->PVCompute(recHitXResDisk2Plaquettes_[i], newrecHitXResDisk2Plaquettes_[i], te);
	   
	}
	Pixel->Print("RecHit_XRes_disk2_plaquettes.eps");
	Pixel->Print("RecHit_XRes_disk2_plaquettes.gif");
   }

   if (1) {
	TH1* recHitYResDisk1Plaquettes_[7];
	TH1* newrecHitYResDisk1Plaquettes_[7];
	Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
	Pixel->Divide(2,4);

	for (Int_t i=0; i<7; i++) {
	   //RecHit Y resolution by plaquette for Disk1

	   sprintf(histo, "DQMData/recHitFPIX/RecHit_YRes_Disk1_Plaquette%d;1", i+1);
	   rfile->GetObject(histo, recHitYResDisk1Plaquettes_[i]);
	   sfile->GetObject(histo, newrecHitYResDisk1Plaquettes_[i]);

	   Pixel->cd(i+1);
	   gPad->SetLogy();
	   recHitYResDisk1Plaquettes_[i]->SetLineColor(2);
	   recHitYResDisk1Plaquettes_[i]->Draw();
	   newrecHitYResDisk1Plaquettes_[i]->SetLineColor(4);
	   newrecHitYResDisk1Plaquettes_[i]->SetLineStyle(2);
	   newrecHitYResDisk1Plaquettes_[i]->Draw("Sames");
	   myPV->PVCompute(recHitYResDisk1Plaquettes_[i], newrecHitYResDisk1Plaquettes_[i], te);
	   
	}
	Pixel->Print("RecHit_YRes_disk1_plaquettes.eps");
	Pixel->Print("RecHit_YRes_disk1_plaquettes.gif");
   }

   if (1) {
	TH1* recHitYResDisk2Plaquettes_[7];
	TH1* newrecHitYResDisk2Plaquettes_[7];
	Pixel = new TCanvas("Pixel", "Pixel", 800, 1200);
	Pixel->Divide(2,4);

	for (Int_t i=0; i<7; i++) {
	   //RecHit X resolution by plaquette for Disk2

	   sprintf(histo, "DQMData/recHitFPIX/RecHit_YRes_Disk2_Plaquette%d;1", i+1);
	   rfile->GetObject(histo, recHitYResDisk2Plaquettes_[i]);
	   sfile->GetObject(histo, newrecHitYResDisk2Plaquettes_[i]);

	   Pixel->cd(i+1);
	   gPad->SetLogy();
	   recHitYResDisk2Plaquettes_[i]->SetLineColor(2);
	   recHitYResDisk2Plaquettes_[i]->Draw();
	   newrecHitYResDisk2Plaquettes_[i]->SetLineColor(4);
	   newrecHitYResDisk2Plaquettes_[i]->SetLineStyle(2);
	   newrecHitYResDisk2Plaquettes_[i]->Draw("Sames");
	   myPV->PVCompute(recHitYResDisk2Plaquettes_[i], newrecHitYResDisk2Plaquettes_[i], te);
	   
	}
	Pixel->Print("RecHit_YRes_disk2_plaquettes.eps");
	Pixel->Print("RecHit_YRes_disk2_plaquettes.gif");
   }
} // end
