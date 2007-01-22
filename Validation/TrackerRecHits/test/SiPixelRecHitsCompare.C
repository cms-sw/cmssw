void SiPixelRecHitsCompare()
{
   gROOT->Reset();
   char* rfilename = "pixelrechitshisto.root";
   char* sfilename = "../data/pixelrechitshisto.root";
   
   //char* rfilename = "/uscms/home/ggiurgiu/work/CMSSW_1_2_0_pre4/src/Validation/TrackerRecHits/test/pixelrechitshisto_test34.root";
   //char* sfilename = "/uscms/home/ggiurgiu/work/CMSSW_1_2_0_pre4/src/Validation/TrackerRecHits/test/pixelrechitshisto_test56.root";
   

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

   TCanvas* Pixel0;
   TCanvas* Pixel1;
   TCanvas* Pixel2;
   TCanvas* Pixel3;
   TCanvas* Pixel4;
   TCanvas* Pixel5;
   TCanvas* Pixel6;
   TCanvas* Pixel7;
   TCanvas* Pixel8;
   TCanvas* Pixel9;
   TCanvas* Pixel10;
   TCanvas* Pixel11;
   TCanvas* Pixel12;
   TCanvas* Pixel13;
   TCanvas* Pixel14;
   TCanvas* Pixel15;
   TCanvas* Pixel16;
   TCanvas* Pixel17;
   TCanvas* Pixel18;
   TCanvas* Pixel19;
   TCanvas* Pixel20;
   TCanvas* Pixel21;
   TCanvas* Pixel22;
   TCanvas* Pixel23;
   TCanvas* Pixel24;
   TCanvas* Pixel25;
   TCanvas* Pixel26;
   TCanvas* Pixel27;
   TCanvas* Pixel28;
   TCanvas* Pixel29;
   TCanvas* Pixel30;
   TCanvas* Pixel31;
   TCanvas* Pixel32;
   TCanvas* Pixel33;
   TCanvas* Pixel34;
   TCanvas* Pixel35;
   TCanvas* Pixel36;
   TCanvas* Pixel37;

   float scale = -99999.9;

   ////////////////////////////
   // Barrel Clusters
   ///////////////////////////

   if (1) 
     {
       //Cluster y size by module
       
       Pixel0 = new TCanvas("Pixel0", "Pixel0", 800, 1200);
       Pixel0->Divide(2,4);
       
       TH1* clustYSizeModule_[8];
       TH1* newclustYSizeModule_[8];
       
       for (Int_t i=0; i<8; i++) 
	 {
	   sprintf(histo, "DQMData/clustBPIX/Clust_y_size_Module%d", i+1);
	   rfile->GetObject(histo, clustYSizeModule_[i]);
	   sfile->GetObject(histo, newclustYSizeModule_[i]);
	   Pixel0->cd(i+1);
	   gPad->SetLogy();
	   clustYSizeModule_[i]->SetLineColor(2);
 	   clustYSizeModule_[i]->Sumw2();
	   newclustYSizeModule_[i]->Sumw2();
	   scale = 1.0/clustYSizeModule_[i]->Integral();
	   clustYSizeModule_[i]->Scale(scale);
	   scale = 1.0/newclustYSizeModule_[i]->Integral();
	   newclustYSizeModule_[i]->Scale(scale);
	   clustYSizeModule_[i]->Draw("h");
	   newclustYSizeModule_[i]->SetLineColor(4);
	   newclustYSizeModule_[i]->SetLineStyle(2);
	   newclustYSizeModule_[i]->Draw("sameh");
	   myPV->PVCompute(clustYSizeModule_[i], newclustYSizeModule_[i], te);
	 }
       Pixel0->Print("Clust_y_size_by_module.eps");   
     }
   
   if (1) 
     {
       //Cluster x size by layer
       
       Pixel1 = new TCanvas("Pixel1", "Pixel1", 800, 1200);
       Pixel1->Divide(1,3);
       
       TH1* clustXSizeLayer_[3];
       TH1* newclustXSizeLayer_[3];
       
       for (Int_t i=0; i<3; i++) 
	 {
	   sprintf(histo, "DQMData/clustBPIX/Clust_x_size_Layer%d", i+1);
	   rfile->GetObject(histo, clustXSizeLayer_[i]);
	   sfile->GetObject(histo, newclustXSizeLayer_[i]);
	   Pixel1->cd(i+1);
	   gPad->SetLogy();
	   clustXSizeLayer_[i]->SetLineColor(2);
	   clustXSizeLayer_[i]->Sumw2();
	   newclustXSizeLayer_[i]->Sumw2();
	   scale = 1.0/clustXSizeLayer_[i]->Integral();
	   clustXSizeLayer_[i]->Scale(scale);
	   scale = 1.0/newclustXSizeLayer_[i]->Integral();
	   newclustXSizeLayer_[i]->Scale(scale);
	   clustXSizeLayer_[i]->Draw("h");
	   newclustXSizeLayer_[i]->SetLineColor(4);
	   newclustXSizeLayer_[i]->SetLineStyle(2);
	   newclustXSizeLayer_[i]->Draw("sameh");
	   myPV->PVCompute(clustXSizeLayer_[i], newclustXSizeLayer_[i], te);
	 }
       Pixel1->Print("Clust_x_size_by_layer.eps");   
     }
   
   if (1) 
     {
       //Cluster charge for Layer 1 by module
       
       Pixel2 = new TCanvas("Pixel2", "Pixel2", 800, 1200);
       Pixel2->Divide(2,4);
       
       TH1* clustChargeLayer1Modules_[8];
       TH1* newclustChargeLayer1Modules_[8];
       
       for (Int_t i=0; i<8; i++) 
	 {
	   sprintf(histo, "DQMData/clustBPIX/Clust_charge_Layer1_Module%d", i+1);
	   rfile->GetObject(histo, clustChargeLayer1Modules_[i]);
	   sfile->GetObject(histo, newclustChargeLayer1Modules_[i]);
	   Pixel2->cd(i+1);
	   gPad->SetLogy();
	   clustChargeLayer1Modules_[i]->SetLineColor(2);
	   clustChargeLayer1Modules_[i]->Sumw2();
	   newclustChargeLayer1Modules_[i]->Sumw2();
	   scale = 1.0/clustChargeLayer1Modules_[i]->Integral();
	   clustChargeLayer1Modules_[i]->Scale(scale);
	   scale = 1.0/newclustChargeLayer1Modules_[i]->Integral();
	   newclustChargeLayer1Modules_[i]->Scale(scale);
	   clustChargeLayer1Modules_[i]->Draw("h");
	   newclustChargeLayer1Modules_[i]->SetLineColor(4);
	   newclustChargeLayer1Modules_[i]->SetLineStyle(2);
	   newclustChargeLayer1Modules_[i]->Draw("sameh");
	   myPV->PVCompute(clustChargeLayer1Modules_[i], newclustChargeLayer1Modules_[i], te);
	 }
       Pixel2->Print("Clust_charge_layer1_modules.eps");   
     }
   
   if (1) 
     {
       //Cluster charge for Layer 2 by module
       
       Pixel3 = new TCanvas("Pixel3", "Pixel3", 800, 1200);
       Pixel3->Divide(2,4);
       
       TH1* clustChargeLayer2Modules_[8];
       TH1* newclustChargeLayer2Modules_[8];
       
       for (Int_t i=0; i<8; i++) 
	 {
	   sprintf(histo, "DQMData/clustBPIX/Clust_charge_Layer2_Module%d", i+1);
	   rfile->GetObject(histo, clustChargeLayer2Modules_[i]);
	   sfile->GetObject(histo, newclustChargeLayer2Modules_[i]);
	   Pixel3->cd(i+1);
	   gPad->SetLogy();
	   clustChargeLayer2Modules_[i]->SetLineColor(2);
	   clustChargeLayer2Modules_[i]->Sumw2();
	   newclustChargeLayer2Modules_[i]->Sumw2();
	   scale = 1.0/clustChargeLayer2Modules_[i]->Integral();
	   clustChargeLayer2Modules_[i]->Scale(scale);
	   scale = 1.0/newclustChargeLayer2Modules_[i]->Integral();
	   newclustChargeLayer2Modules_[i]->Scale(scale);
	   clustChargeLayer2Modules_[i]->Draw("h");
	   newclustChargeLayer2Modules_[i]->SetLineColor(4);
	   newclustChargeLayer2Modules_[i]->SetLineStyle(2);
	   newclustChargeLayer2Modules_[i]->Draw("sameh");
	   myPV->PVCompute(clustChargeLayer2Modules_[i], newclustChargeLayer2Modules_[i], te);
	 }
       Pixel3->Print("Clust_charge_layer2_modules.eps");   
     }
   
   if (1) 
     {
       //Cluster charge for Layer 3 by module
       
       Pixel4 = new TCanvas("Pixel4", "Pixel4", 800, 1200);
       Pixel4->Divide(2,4);
       
       TH1* clustChargeLayer3Modules_[8];
       TH1* newclustChargeLayer3Modules_[8];
       
       for (Int_t i=0; i<8; i++) 
	 {
	   sprintf(histo, "DQMData/clustBPIX/Clust_charge_Layer3_Module%d", i+1);
	   rfile->GetObject(histo, clustChargeLayer3Modules_[i]);
	   sfile->GetObject(histo, newclustChargeLayer3Modules_[i]);
	   Pixel4->cd(i+1);
	   gPad->SetLogy();
	   clustChargeLayer3Modules_[i]->SetLineColor(2);
	   clustChargeLayer3Modules_[i]->Sumw2();
	   newclustChargeLayer3Modules_[i]->Sumw2();
	   scale = 1.0/clustChargeLayer3Modules_[i]->Integral();
	   clustChargeLayer3Modules_[i]->Scale(scale);
	   scale = 1.0/newclustChargeLayer3Modules_[i]->Integral();
	   newclustChargeLayer3Modules_[i]->Scale(scale);
	   clustChargeLayer3Modules_[i]->Draw("h");
	   newclustChargeLayer3Modules_[i]->SetLineColor(4);
	   newclustChargeLayer3Modules_[i]->SetLineStyle(2);
	   newclustChargeLayer3Modules_[i]->Draw("sameh");
	   myPV->PVCompute(clustChargeLayer3Modules_[i], newclustChargeLayer3Modules_[i], te);
	 }
       Pixel4->Print("Clust_charge_layer3_modules.eps");   
     }
   
   //----------------------------
   // Forward clusters
   //----------------------------
     
   if (1) 
     {
       // Cluster xsize for Disk1 by plaquette
       
       Pixel5 = new TCanvas("Pixel5", "Pixel5", 800, 1200);
       Pixel5->Divide(2,4);
       
       TH1* clustXSizeDisk1Plaquettes_[7];
       TH1* newclustXSizeDisk1Plaquettes_[7];
       
       for (Int_t i=0; i<7; i++) 
	 {
	   sprintf(histo, "DQMData/clustFPIX/Clust_x_size_Disk1_Plaquette%d", i+1);
	   rfile->GetObject(histo, clustXSizeDisk1Plaquettes_[i]);
	   sfile->GetObject(histo, newclustXSizeDisk1Plaquettes_[i]);
	   Pixel5->cd(i+1);
	   gPad->SetLogy();
	   clustXSizeDisk1Plaquettes_[i]->SetLineColor(2);
	   clustXSizeDisk1Plaquettes_[i]->Sumw2();
	   newclustXSizeDisk1Plaquettes_[i]->Sumw2();
	   scale = 1.0/clustXSizeDisk1Plaquettes_[i]->Integral();
	   clustXSizeDisk1Plaquettes_[i]->Scale(scale);
	   scale = 1.0/ newclustXSizeDisk1Plaquettes_[i]->Integral();
	   newclustXSizeDisk1Plaquettes_[i]->Scale(scale);
	   clustXSizeDisk1Plaquettes_[i]->Draw("h");
	   newclustXSizeDisk1Plaquettes_[i]->SetLineColor(4);
	   newclustXSizeDisk1Plaquettes_[i]->SetLineStyle(2);
	   newclustXSizeDisk1Plaquettes_[i]->Draw("sameh");
	   myPV->PVCompute(clustXSizeDisk1Plaquettes_[i], newclustXSizeDisk1Plaquettes_[i], te);
	 }
       Pixel5->Print("Clust_xsize_disk1_plaquettes.eps");   
     }
   
   if (1) 
     {
       //Cluster xsize for Disk2 by plaquette
       
       Pixel6 = new TCanvas("Pixel6", "Pixel6", 800, 1200);
       Pixel6->Divide(2,4);
       
       TH1* clustXSizeDisk2Plaquettes_[7];
       TH1* newclustXSizeDisk2Plaquettes_[7];
       
       for (Int_t i=0; i<7; i++) 
	 {
	   sprintf(histo, "DQMData/clustFPIX/Clust_x_size_Disk2_Plaquette%d", i+1);
	   rfile->GetObject(histo, clustXSizeDisk2Plaquettes_[i]);
	   sfile->GetObject(histo, newclustXSizeDisk2Plaquettes_[i]);
	   Pixel6->cd(i+1);
	   gPad->SetLogy();
	   clustXSizeDisk2Plaquettes_[i]->SetLineColor(2);
	   clustXSizeDisk2Plaquettes_[i]->Sumw2();
	   newclustXSizeDisk2Plaquettes_[i]->Sumw2();
	   scale = 1.0/clustXSizeDisk2Plaquettes_[i]->Integral();
	   clustXSizeDisk2Plaquettes_[i]->Scale(scale);
	   scale = 1.0/newclustXSizeDisk2Plaquettes_[i]->Integral();
	   newclustXSizeDisk2Plaquettes_[i]->Scale(scale);
	   clustXSizeDisk2Plaquettes_[i]->Draw("h");
	   newclustXSizeDisk2Plaquettes_[i]->SetLineColor(4);
	   newclustXSizeDisk2Plaquettes_[i]->SetLineStyle(2);
	   newclustXSizeDisk2Plaquettes_[i]->Draw("sameh");
	   myPV->PVCompute(clustXSizeDisk2Plaquettes_[i], newclustXSizeDisk2Plaquettes_[i], te);
	 }
       Pixel6->Print("Clust_xsize_disk2_plaquettes.eps");   
     }
   
   
   if (1) 
     {
       // Cluster ysize for Disk1 by plaquette
       
       Pixel7 = new TCanvas("Pixel7", "Pixel7", 800, 1200);
       Pixel7->Divide(2,4);
       
       TH1* clustYSizeDisk1Plaquettes_[7];
       TH1* newclustYSizeDisk1Plaquettes_[7];
       
       for (Int_t i=0; i<7; i++) 
	 {
	   sprintf(histo, "DQMData/clustFPIX/Clust_y_size_Disk1_Plaquette%d", i+1);
	   rfile->GetObject(histo, clustYSizeDisk1Plaquettes_[i]);
	   sfile->GetObject(histo, newclustYSizeDisk1Plaquettes_[i]);
	   Pixel7->cd(i+1);
	   gPad->SetLogy();
	   clustYSizeDisk1Plaquettes_[i]->SetLineColor(2);
	   clustYSizeDisk1Plaquettes_[i]->Sumw2();
	   newclustYSizeDisk1Plaquettes_[i]->Sumw2();
	   scale = 1.0/clustYSizeDisk1Plaquettes_[i]->Integral();
	   clustYSizeDisk1Plaquettes_[i]->Scale(scale);
	   scale = 1.0/newclustYSizeDisk1Plaquettes_[i]->Integral();
	   newclustYSizeDisk1Plaquettes_[i]->Scale(scale);
	   clustYSizeDisk1Plaquettes_[i]->Draw("h");
	   newclustYSizeDisk1Plaquettes_[i]->SetLineColor(4);
	   newclustYSizeDisk1Plaquettes_[i]->SetLineStyle(2);
	   newclustYSizeDisk1Plaquettes_[i]->Draw("sameh");
	   myPV->PVCompute(clustYSizeDisk1Plaquettes_[i], newclustYSizeDisk1Plaquettes_[i], te);
	 }
       Pixel7->Print("Clust_ysize_disk1_plaquettes.eps");   
     }
   
   if (1) 
     {
       //Cluster ysize for Disk2 by plaquette
       
       Pixel8 = new TCanvas("Pixel8", "Pixel8", 800, 1200);
       Pixel8->Divide(2,4);
       
       TH1* clustYSizeDisk2Plaquettes_[7];
       TH1* newclustYSizeDisk2Plaquettes_[7];
       
       for (Int_t i=0; i<7; i++) 
	 {
	   sprintf(histo, "DQMData/clustFPIX/Clust_y_size_Disk2_Plaquette%d", i+1);
	   rfile->GetObject(histo, clustYSizeDisk2Plaquettes_[i]);
	   sfile->GetObject(histo, newclustYSizeDisk2Plaquettes_[i]);
	   Pixel8->cd(i+1);
	   gPad->SetLogy();
	   clustYSizeDisk2Plaquettes_[i]->SetLineColor(2);
	   clustYSizeDisk2Plaquettes_[i]->Sumw2();
	   newclustYSizeDisk2Plaquettes_[i]->Sumw2();
	   scale = 1.0/clustYSizeDisk2Plaquettes_[i]->Integral();
	   clustYSizeDisk2Plaquettes_[i]->Scale(scale);
	   scale = 1.0/newclustYSizeDisk2Plaquettes_[i]->Integral();
	   newclustYSizeDisk2Plaquettes_[i]->Scale(scale);
	   clustYSizeDisk2Plaquettes_[i]->Draw("h");
	   newclustYSizeDisk2Plaquettes_[i]->SetLineColor(4);
	   newclustYSizeDisk2Plaquettes_[i]->SetLineStyle(2);
	   newclustYSizeDisk2Plaquettes_[i]->Draw("sameh");
	   myPV->PVCompute(clustYSizeDisk2Plaquettes_[i], newclustYSizeDisk2Plaquettes_[i], te);
	}
       Pixel8->Print("Clust_ysize_disk2_plaquettes.eps");   
     }
   
   
   if (1) 
     {
       //Cluster charge for Disk1 by plaquette
       
       Pixel9 = new TCanvas("Pixel9", "Pixel9", 800, 1200);
       Pixel9->Divide(2,4);
       
       TH1* clustChargeDisk1Plaquettes_[7];
       TH1* newclustChargeDisk1Plaquettes_[7];
       
       for (Int_t i=0; i<7; i++) 
	 {
	   sprintf(histo, "DQMData/clustFPIX/Clust_charge_Disk1_Plaquette%d", i+1);
	   rfile->GetObject(histo, clustChargeDisk1Plaquettes_[i]);
	   sfile->GetObject(histo, newclustChargeDisk1Plaquettes_[i]);
	   Pixel9->cd(i+1);
	   gPad->SetLogy();
	   clustChargeDisk1Plaquettes_[i]->SetLineColor(2);
	   clustChargeDisk1Plaquettes_[i]->Sumw2();
	   newclustChargeDisk1Plaquettes_[i]->Sumw2();
	   scale = 1.0/clustChargeDisk1Plaquettes_[i]->Integral();
	   clustChargeDisk1Plaquettes_[i]->Scale(scale);
	   scale = 1.0/newclustChargeDisk1Plaquettes_[i]->Integral();
	   newclustChargeDisk1Plaquettes_[i]->Scale(scale);
	   clustChargeDisk1Plaquettes_[i]->Draw("h");
	   newclustChargeDisk1Plaquettes_[i]->SetLineColor(4);
	   newclustChargeDisk1Plaquettes_[i]->SetLineStyle(2);
	   newclustChargeDisk1Plaquettes_[i]->Draw("sameh");
	   myPV->PVCompute(clustChargeDisk1Plaquettes_[i], newclustChargeDisk1Plaquettes_[i], te);
	 }
       Pixel9->Print("Clust_charge_disk1_plaquettes.eps");   
     }
   
   if (1) 
     {
       //Cluster charge for Disk2 by plaquette
       
       Pixel10 = new TCanvas("Pixel10", "Pixel10", 800, 1200);
       Pixel10->Divide(2,4);
       
       TH1* clustChargeDisk2Plaquettes_[7];
       TH1* newclustChargeDisk2Plaquettes_[7];
       
       for (Int_t i=0; i<7; i++) 
	 {
	   sprintf(histo, "DQMData/clustFPIX/Clust_charge_Disk2_Plaquette%d", i+1);
	   rfile->GetObject(histo, clustChargeDisk2Plaquettes_[i]);
	   sfile->GetObject(histo, newclustChargeDisk2Plaquettes_[i]);
	   Pixel10->cd(i+1);
	   gPad->SetLogy();
	   clustChargeDisk2Plaquettes_[i]->SetLineColor(2);
	   clustChargeDisk2Plaquettes_[i]->Sumw2();
	   newclustChargeDisk2Plaquettes_[i]->Sumw2();
	   scale = 1.0/clustChargeDisk2Plaquettes_[i]->Integral();
	   clustChargeDisk2Plaquettes_[i]->Scale(scale);
	   scale = 1.0/newclustChargeDisk2Plaquettes_[i]->Integral();
	   newclustChargeDisk2Plaquettes_[i]->Scale(scale);
	   clustChargeDisk2Plaquettes_[i]->Draw("h");
	   newclustChargeDisk2Plaquettes_[i]->SetLineColor(4);
	   newclustChargeDisk2Plaquettes_[i]->SetLineStyle(2);
	   newclustChargeDisk2Plaquettes_[i]->Draw("sameh");
	   myPV->PVCompute(clustChargeDisk2Plaquettes_[i], newclustChargeDisk2Plaquettes_[i], te);
	}
       Pixel10->Print("Clust_charge_disk2_plaquettes.eps");   
     }

   
   //----------------------
   // RecHit Barrel
   //----------------------
   
   if (1) 
     {
       //RecHit x distribution for Full Modules
       
       Pixel11 = new TCanvas("Pixel11", "Pixel11", 200, 300);
       TH1* recHitXFullModules_;
       TH1* newrecHitXFullModules_;
       
       sprintf (histo, "DQMData/recHitBPIX/RecHit_x_FullModules");
       rfile->GetObject(histo, recHitXFullModules_);
       sfile->GetObject(histo, newrecHitXFullModules_);
       
       gPad->SetLogy();
       recHitXFullModules_->SetLineColor(2);
       recHitXFullModules_->Sumw2();
       newrecHitXFullModules_->Sumw2();
       scale = 1.0/recHitXFullModules_->Integral();
       recHitXFullModules_->Scale(scale);
       scale = 1.0/newrecHitXFullModules_->Integral();
       newrecHitXFullModules_->Scale(scale);
       recHitXFullModules_->Draw("h");
       newrecHitXFullModules_->SetLineColor(4);
       newrecHitXFullModules_->SetLineStyle(2);
       newrecHitXFullModules_->Draw("sameh");
       myPV->PVCompute(recHitXFullModules_, newrecHitXFullModules_, te);
       
       Pixel11->Print("RecHit_XDist_FullModules.eps");
       
       //RecHit x distribution half modules
       
       Pixel12 = new TCanvas("Pixel12", "Pixel12", 200, 300);
       TH1* recHitXHalfModules_;
       TH1* newrecHitXHalfModules_;
       
       sprintf (histo, "DQMData/recHitBPIX/RecHit_x_HalfModules");
       rfile->GetObject(histo, recHitXHalfModules_);
       sfile->GetObject(histo, newrecHitXHalfModules_);
       
       gPad->SetLogy();
       recHitXHalfModules_->SetLineColor(2);
       recHitXHalfModules_->Sumw2();
       newrecHitXHalfModules_->Sumw2();
       scale = 1.0/recHitXHalfModules_->Integral();
       recHitXHalfModules_->Scale(scale);
       scale = 1.0/newrecHitXHalfModules_->Integral();
       newrecHitXHalfModules_->Scale(scale);
       recHitXHalfModules_->Draw("h");
       newrecHitXHalfModules_->SetLineColor(4);
       newrecHitXHalfModules_->SetLineStyle(2);
       newrecHitXHalfModules_->Draw("sameh");
       myPV->PVCompute(recHitXHalfModules_, newrecHitXHalfModules_, te);
       
       Pixel12->Print("RecHit_XDist_HalfModules.eps");
       
       //RecHit y distribution all modules
       
       Pixel13 = new TCanvas("Pixel13", "Pixel13", 200, 300);
       TH1* recHitYAllModules_;
       TH1* newrecHitYAllModules_;
       
       sprintf (histo, "DQMData/recHitBPIX/RecHit_y_AllModules");
       rfile->GetObject(histo, recHitYAllModules_);
       sfile->GetObject(histo, newrecHitYAllModules_);
       
       gPad->SetLogy();
       recHitYAllModules_->SetLineColor(2);
       recHitYAllModules_->Sumw2();
       newrecHitYAllModules_->Sumw2();
       scale = 1.0/recHitYAllModules_->Integral();
       recHitYAllModules_->Scale(scale);
       scale = 1.0/newrecHitYAllModules_->Integral();
       newrecHitYAllModules_->Scale(scale);
       recHitYAllModules_->Draw("h");
       newrecHitYAllModules_->SetLineColor(4);
       newrecHitYAllModules_->SetLineStyle(2);
       newrecHitYAllModules_->Draw("sameh");
       myPV->PVCompute(recHitYAllModules_, newrecHitYAllModules_, te);
       
       Pixel13->Print("RecHit_YDist_AllModules.eps");
     }
   
   if (1) 
     {
       TH1* recHitXResFlippedLadderLayers_[3];
       TH1* newrecHitXResFlippedLadderLayers_[3];
       Pixel14 = new TCanvas("Pixel14", "Pixel14", 400, 600);
       Pixel14->Divide(1,3);
       
       for (Int_t i=0; i<3; i++) 
	 {
	   //RecHit XRes Flipped ladders by layer
	   
	   sprintf(histo, "DQMData/recHitBPIX/RecHit_XRes_FlippedLadder_Layer%d", i+1);
	   rfile->GetObject(histo, recHitXResFlippedLadderLayers_[i]);
	   sfile->GetObject(histo, newrecHitXResFlippedLadderLayers_[i]);
	   
	   Pixel14->cd(i+1);
	   gPad->SetLogy();
	   recHitXResFlippedLadderLayers_[i]->SetLineColor(2);
	   recHitXResFlippedLadderLayers_[i]->Sumw2();
	   newrecHitXResFlippedLadderLayers_[i]->Sumw2();
	   scale = 1.0/recHitXResFlippedLadderLayers_[i]->Integral();
	   recHitXResFlippedLadderLayers_[i]->Scale(scale);
	   scale = 1.0/newrecHitXResFlippedLadderLayers_[i]->Integral();
	   newrecHitXResFlippedLadderLayers_[i]->Scale(scale);
	   recHitXResFlippedLadderLayers_[i]->Draw("h");
	   newrecHitXResFlippedLadderLayers_[i]->SetLineColor(4);
	   newrecHitXResFlippedLadderLayers_[i]->SetLineStyle(2);
	   newrecHitXResFlippedLadderLayers_[i]->Draw("sameh");
	   myPV->PVCompute(recHitXResFlippedLadderLayers_[i], newrecHitXResFlippedLadderLayers_[i], te);
	   
	 }
       Pixel14->Print("RecHit_XRes_FlippedLadder_Layers.eps");
     }
   
   if (1) 
     {
       TH1* recHitXResUnFlippedLadderLayers_[3];
       TH1* newrecHitXResUnFlippedLadderLayers_[3];
       Pixel15 = new TCanvas("Pixel15", "Pixel15", 400, 600);
       Pixel15->Divide(1,3);
       
       for (Int_t i=0; i<3; i++) 
	 {
	   //RecHit XRes unflipped ladders by layer
	   
	   sprintf(histo, "DQMData/recHitBPIX/RecHit_XRes_UnFlippedLadder_Layer%d", i+1);
	   rfile->GetObject(histo, recHitXResUnFlippedLadderLayers_[i]);
	   sfile->GetObject(histo, newrecHitXResUnFlippedLadderLayers_[i]);
	   
	   Pixel15->cd(i+1);
	   gPad->SetLogy();
	   recHitXResUnFlippedLadderLayers_[i]->SetLineColor(2);
	   recHitXResUnFlippedLadderLayers_[i]->Sumw2();
	   newrecHitXResUnFlippedLadderLayers_[i]->Sumw2();
	   scale = 1.0/recHitXResUnFlippedLadderLayers_[i]->Integral();
	   recHitXResUnFlippedLadderLayers_[i]->Scale(scale);
	   scale = 1.0/newrecHitXResUnFlippedLadderLayers_[i]->Integral();
	   newrecHitXResUnFlippedLadderLayers_[i]->Scale(scale);
	   recHitXResUnFlippedLadderLayers_[i]->Draw("h");
	   newrecHitXResUnFlippedLadderLayers_[i]->SetLineColor(4);
	   newrecHitXResUnFlippedLadderLayers_[i]->SetLineStyle(2);
	   newrecHitXResUnFlippedLadderLayers_[i]->Draw("sameh");
	   myPV->PVCompute(recHitXResUnFlippedLadderLayers_[i], newrecHitXResUnFlippedLadderLayers_[i], te);
	   
	 }
       Pixel15->Print("RecHit_XRes_UnFlippedLadder_Layers.eps");
     }
   
   if (1) 
     {
       TH1* recHitYResLayer1Modules_[8];
       TH1* newrecHitYResLayer1Modules_[8];
       Pixel16 = new TCanvas("Pixel16", "Pixel16", 800, 1200);
       Pixel16->Divide(2,4);
       
       for (Int_t i=0; i<8; i++) 
	 {
	   //RecHit Y resolution by module for layer 1
	   
	   sprintf(histo, "DQMData/recHitBPIX/RecHit_YRes_Layer1_Module%d", i+1);
	   rfile->GetObject(histo, recHitYResLayer1Modules_[i]);
	   sfile->GetObject(histo, newrecHitYResLayer1Modules_[i]);

	   Pixel16->cd(i+1);
	   gPad->SetLogy();
	   recHitYResLayer1Modules_[i]->SetLineColor(2);
	   recHitYResLayer1Modules_[i]->Sumw2();
	   newrecHitYResLayer1Modules_[i]->Sumw2();
	   scale = 1.0/recHitYResLayer1Modules_[i]->Integral();
	   recHitYResLayer1Modules_[i]->Scale(scale);
	   scale = 1.0/newrecHitYResLayer1Modules_[i]->Integral();
	   newrecHitYResLayer1Modules_[i]->Scale(scale);
	   recHitYResLayer1Modules_[i]->Draw("h");
	   newrecHitYResLayer1Modules_[i]->SetLineColor(4);
	   newrecHitYResLayer1Modules_[i]->SetLineStyle(2);
	   newrecHitYResLayer1Modules_[i]->Draw("sameh");
	   myPV->PVCompute(recHitYResLayer1Modules_[i], newrecHitYResLayer1Modules_[i], te);
	   
	 }
       Pixel16->Print("RecHit_YRes_Layer1_Modules.eps");
     }
   
   if (1) 
     {
       TH1* recHitYResLayer2Modules_[8];
       TH1* newrecHitYResLayer2Modules_[8];
       Pixel17 = new TCanvas("Pixel17", "Pixel17", 800, 1200);
       Pixel17->Divide(2,4);
       
       for (Int_t i=0; i<8; i++)
	 {
	   //RecHit Y resolution by module for layer 2
	   
	   sprintf(histo, "DQMData/recHitBPIX/RecHit_YRes_Layer2_Module%d", i+1);
	   rfile->GetObject(histo, recHitYResLayer2Modules_[i]);
	   sfile->GetObject(histo, newrecHitYResLayer2Modules_[i]);

	   Pixel17->cd(i+1);
	   gPad->SetLogy();
	   recHitYResLayer2Modules_[i]->SetLineColor(2);
	   recHitYResLayer2Modules_[i]->Sumw2();
	   newrecHitYResLayer2Modules_[i]->Sumw2();
	   scale = 1.0/recHitYResLayer2Modules_[i]->Integral();
	   recHitYResLayer2Modules_[i]->Scale(scale);
	   scale = 1.0/newrecHitYResLayer2Modules_[i]->Integral();
	   newrecHitYResLayer2Modules_[i]->Scale(scale);
	   recHitYResLayer2Modules_[i]->Draw("h");
	   newrecHitYResLayer2Modules_[i]->SetLineColor(4);
	   newrecHitYResLayer2Modules_[i]->SetLineStyle(2);
	   newrecHitYResLayer2Modules_[i]->Draw("sameh");
	   myPV->PVCompute(recHitYResLayer2Modules_[i], newrecHitYResLayer2Modules_[i], te);
	   
	}
       Pixel17->Print("RecHit_YRes_Layer2_Modules.eps");
     }
   
   if (1) 
     {
       TH1* recHitYResLayer3Modules_[8];
       TH1* newrecHitYResLayer3Modules_[8];
       Pixel18 = new TCanvas("Pixel18", "Pixel18", 800, 1200);
       Pixel18->Divide(2,4);
       
       for (Int_t i=0; i<8; i++) 
	 {
	   //RecHit Y resolution by module for layer 3
	   
	   sprintf(histo, "DQMData/recHitBPIX/RecHit_YRes_Layer3_Module%d", i+1);
	   rfile->GetObject(histo, recHitYResLayer3Modules_[i]);
	   sfile->GetObject(histo, newrecHitYResLayer3Modules_[i]);

	   Pixel18->cd(i+1);
	   gPad->SetLogy();
	   recHitYResLayer3Modules_[i]->SetLineColor(2);
	   recHitYResLayer3Modules_[i]->Sumw2();
	   newrecHitYResLayer3Modules_[i]->Sumw2();
	   scale = 1.0/recHitYResLayer3Modules_[i]->Integral();
	   recHitYResLayer3Modules_[i]->Scale(scale);
	   scale = 1.0/newrecHitYResLayer3Modules_[i]->Integral();
	   newrecHitYResLayer3Modules_[i]->Scale(scale);
	   recHitYResLayer3Modules_[i]->Draw("h");
	   newrecHitYResLayer3Modules_[i]->SetLineColor(4);
	   newrecHitYResLayer3Modules_[i]->SetLineStyle(2);
	   newrecHitYResLayer3Modules_[i]->Draw("sameh");
	   myPV->PVCompute(recHitYResLayer3Modules_[i], newrecHitYResLayer3Modules_[i], te);
	   
	 }
       Pixel18->Print("RecHit_YRes_Layer3_Modules.eps");
     }
   
   ////////////////////////////////
   //RecHit forward
   ////////////////////////////////
   
   if (1) 
     {
       //RecHit x distribution for plaquettes x-size 1
       
       Pixel19 = new TCanvas("Pixel19", "Pixel19", 200, 300);
       
       TH1* recHitXPlaquetteXSize1_;
       TH1* newrecHitXPlaquetteXSize1_;
       
       sprintf(histo, "DQMData/recHitFPIX/RecHit_x_Plaquette_xsize1");
       rfile->GetObject(histo, recHitXPlaquetteXSize1_);
       sfile->GetObject(histo, newrecHitXPlaquetteXSize1_);
       
       gPad->SetLogy();
       recHitXPlaquetteXSize1_->SetLineColor(2);
       recHitXPlaquetteXSize1_->Sumw2();
       newrecHitXPlaquetteXSize1_->Sumw2();
       scale = 1.0/recHitXPlaquetteXSize1_->Integral();
       recHitXPlaquetteXSize1_->Scale(scale);
       scale = 1.0/newrecHitXPlaquetteXSize1_->Integral();
       newrecHitXPlaquetteXSize1_->Scale(scale);
       recHitXPlaquetteXSize1_->Draw("h");
       newrecHitXPlaquetteXSize1_->SetLineColor(4);
       newrecHitXPlaquetteXSize1_->SetLineStyle(2);
       newrecHitXPlaquetteXSize1_->Draw("sameh");
       myPV->PVCompute(recHitXPlaquetteXSize1_, newrecHitXPlaquetteXSize1_, te);
       
       Pixel19->Print("RecHit_X_Plaquette_xsize1.eps");
       
       //RecHit x distribution for plaquettes x-size 2
       
       Pixel20 = new TCanvas("Pixel20", "Pixel20", 200, 300);
       
       TH1* recHitXPlaquetteXSize2_;
       TH1* newrecHitXPlaquetteXSize2_;
       
       sprintf(histo, "DQMData/recHitFPIX/RecHit_x_Plaquette_xsize2");
       rfile->GetObject(histo, recHitXPlaquetteXSize2_);
       sfile->GetObject(histo, newrecHitXPlaquetteXSize2_);
       
       gPad->SetLogy();
       recHitXPlaquetteXSize2_->SetLineColor(2);
       recHitXPlaquetteXSize2_->Sumw2();
       newrecHitXPlaquetteXSize2_->Sumw2();
       scale = 1.0/recHitXPlaquetteXSize2_->Integral();
       recHitXPlaquetteXSize2_->Scale(scale);
       scale = 1.0/newrecHitXPlaquetteXSize2_->Integral();
       newrecHitXPlaquetteXSize2_->Scale(scale);
       recHitXPlaquetteXSize2_->Draw("h");
       newrecHitXPlaquetteXSize2_->SetLineColor(4);
       newrecHitXPlaquetteXSize2_->SetLineStyle(2);
       newrecHitXPlaquetteXSize2_->Draw("sameh");
       myPV->PVCompute(recHitXPlaquetteXSize2_, newrecHitXPlaquetteXSize2_, te);
       
       Pixel20->Print("RecHit_X_Plaquette_xsize2.eps");
       
       //RecHit y distribution for plaquettes y-size 2
       
       Pixel21 = new TCanvas("Pixel21", "Pixel21", 200, 300);
       
       TH1* recHitYPlaquetteYSize2_;
       TH1* newrecHitYPlaquetteYSize2_;
       
       sprintf(histo, "DQMData/recHitFPIX/RecHit_y_Plaquette_ysize2");
       rfile->GetObject(histo, recHitYPlaquetteYSize2_);
       sfile->GetObject(histo, newrecHitYPlaquetteYSize2_);
       
       gPad->SetLogy();
       recHitYPlaquetteYSize2_->SetLineColor(2);
       recHitYPlaquetteYSize2_->Sumw2();
       newrecHitYPlaquetteYSize2_->Sumw2();
       scale = 1.0/recHitYPlaquetteYSize2_->Integral();
       recHitYPlaquetteYSize2_->Scale(scale);
       scale = 1.0/newrecHitYPlaquetteYSize2_->Integral();
       newrecHitYPlaquetteYSize2_->Scale(scale);
       recHitYPlaquetteYSize2_->Draw("h");
       newrecHitYPlaquetteYSize2_->SetLineColor(4);
       newrecHitYPlaquetteYSize2_->SetLineStyle(2);
       newrecHitYPlaquetteYSize2_->Draw("sameh");
       myPV->PVCompute(recHitYPlaquetteYSize2_, newrecHitYPlaquetteYSize2_, te);
       
       Pixel21->Print("RecHit_Y_Plaquette_ysize2.eps");
       
       //RecHit y distribution for plaquettes y-size 3
       
       Pixel22 = new TCanvas("Pixel22", "Pixel22", 200, 300);
       
       TH1* recHitYPlaquetteYSize3_;
       TH1* newrecHitYPlaquetteYSize3_;
       
       sprintf(histo, "DQMData/recHitFPIX/RecHit_y_Plaquette_ysize3");
       rfile->GetObject(histo, recHitYPlaquetteYSize3_);
       sfile->GetObject(histo, newrecHitYPlaquetteYSize3_);
       
       gPad->SetLogy();
       recHitYPlaquetteYSize3_->SetLineColor(2);
       recHitYPlaquetteYSize3_->Sumw2();
       newrecHitYPlaquetteYSize3_->Sumw2();
       scale = 1.0/recHitYPlaquetteYSize3_->Integral();
       recHitYPlaquetteYSize3_->Scale(scale);
       scale = 1.0/newrecHitYPlaquetteYSize3_->Integral();
       newrecHitYPlaquetteYSize3_->Scale(scale);
       recHitYPlaquetteYSize3_->Draw("h");
       newrecHitYPlaquetteYSize3_->SetLineColor(4);
       newrecHitYPlaquetteYSize3_->SetLineStyle(2);
       newrecHitYPlaquetteYSize3_->Draw("sameh");
       myPV->PVCompute(recHitYPlaquetteYSize3_, newrecHitYPlaquetteYSize3_, te);
       
       Pixel22->Print("RecHit_Y_Plaquette_ysize3.eps");
       
       //RecHit y distribution for plaquettes y-size 4
       
       Pixel23 = new TCanvas("Pixel23", "Pixel23", 200, 300);
       
       TH1* recHitYPlaquetteYSize4_;
       TH1* newrecHitYPlaquetteYSize4_;
       
       sprintf(histo, "DQMData/recHitFPIX/RecHit_y_Plaquette_ysize4");
       rfile->GetObject(histo, recHitYPlaquetteYSize4_);
       sfile->GetObject(histo, newrecHitYPlaquetteYSize4_);
       
       gPad->SetLogy();
       recHitYPlaquetteYSize4_->SetLineColor(2);
       recHitYPlaquetteYSize4_->Sumw2();
       newrecHitYPlaquetteYSize4_->Sumw2();
       scale = 1.0/recHitYPlaquetteYSize4_->Integral();
       recHitYPlaquetteYSize4_->Scale(scale);
       scale = 1.0/newrecHitYPlaquetteYSize4_->Integral();
       newrecHitYPlaquetteYSize4_->Scale(scale);
       recHitYPlaquetteYSize4_->Draw("h");
       newrecHitYPlaquetteYSize4_->SetLineColor(4);
       newrecHitYPlaquetteYSize4_->SetLineStyle(2);
       newrecHitYPlaquetteYSize4_->Draw("sameh");
       myPV->PVCompute(recHitYPlaquetteYSize4_, newrecHitYPlaquetteYSize4_, te);
       
       Pixel23->Print("RecHit_Y_Plaquette_ysize4.eps");
       
       //RecHit y distribution for plaquettes y-size 5
       
       Pixel24 = new TCanvas("Pixel24", "Pixel24", 200, 300);
       
       TH1* recHitYPlaquetteYSize5_;
       TH1* newrecHitYPlaquetteYSize5_;
       
       sprintf(histo, "DQMData/recHitFPIX/RecHit_y_Plaquette_ysize5");
       rfile->GetObject(histo, recHitYPlaquetteYSize5_);
       sfile->GetObject(histo, newrecHitYPlaquetteYSize5_);
       
       gPad->SetLogy();
       recHitYPlaquetteYSize5_->SetLineColor(2);
       recHitYPlaquetteYSize5_->Sumw2();
       newrecHitYPlaquetteYSize5_->Sumw2();
       scale = 1.0/recHitYPlaquetteYSize5_->Integral();
       recHitYPlaquetteYSize5_->Scale(scale);
       scale = 1.0/newrecHitYPlaquetteYSize5_->Integral();
       newrecHitYPlaquetteYSize5_->Scale(scale);
       recHitYPlaquetteYSize5_->Draw("h");
       newrecHitYPlaquetteYSize5_->SetLineColor(4);
       newrecHitYPlaquetteYSize5_->SetLineStyle(2);
       newrecHitYPlaquetteYSize5_->Draw("sameh");
       myPV->PVCompute(recHitYPlaquetteYSize5_, newrecHitYPlaquetteYSize5_, te);
       
       Pixel24->Print("RecHit_Y_Plaquette_ysize5.eps");
     }
   
   if (1) 
     {
       TH1* recHitXResDisk1Plaquettes_[7];
       TH1* newrecHitXResDisk1Plaquettes_[7];
       Pixel25 = new TCanvas("Pixel25", "Pixel25", 800, 1200);
       Pixel25->Divide(2,4);
       
       for (Int_t i=0; i<7; i++) 
	 {
	   //RecHit X resolution by plaquette for Disk1
	   
	   sprintf(histo, "DQMData/recHitFPIX/RecHit_XRes_Disk1_Plaquette%d", i+1);
	   rfile->GetObject(histo, recHitXResDisk1Plaquettes_[i]);
	   sfile->GetObject(histo, newrecHitXResDisk1Plaquettes_[i]);

	   Pixel25->cd(i+1);
	   gPad->SetLogy();
	   recHitXResDisk1Plaquettes_[i]->SetLineColor(2);
	   recHitXResDisk1Plaquettes_[i]->Sumw2();
	   newrecHitXResDisk1Plaquettes_[i]->Sumw2();
	   scale = 1.0/recHitXResDisk1Plaquettes_[i]->Integral();
	   recHitXResDisk1Plaquettes_[i]->Scale(scale);
	   scale = 1.0/newrecHitXResDisk1Plaquettes_[i]->Integral();
	   newrecHitXResDisk1Plaquettes_[i]->Scale(scale);
	   recHitXResDisk1Plaquettes_[i]->Draw("h");
	   newrecHitXResDisk1Plaquettes_[i]->SetLineColor(4);
	   newrecHitXResDisk1Plaquettes_[i]->SetLineStyle(2);
	   newrecHitXResDisk1Plaquettes_[i]->Draw("sameh");
	   myPV->PVCompute(recHitXResDisk1Plaquettes_[i], newrecHitXResDisk1Plaquettes_[i], te);
	   
	}
       Pixel25->Print("RecHit_XRes_disk1_plaquettes.eps");
     }
   
   if (1) 
     {
       TH1* recHitXResDisk2Plaquettes_[7];
       TH1* newrecHitXResDisk2Plaquettes_[7];
       Pixel26 = new TCanvas("Pixel26", "Pixel26", 800, 1200);
       Pixel26->Divide(2,4);
       
       for (Int_t i=0; i<7; i++) 
	 {
	   //RecHit X resolution by plaquette for Disk2
	   
	   sprintf(histo, "DQMData/recHitFPIX/RecHit_XRes_Disk2_Plaquette%d", i+1);
	   rfile->GetObject(histo, recHitXResDisk2Plaquettes_[i]);
	   sfile->GetObject(histo, newrecHitXResDisk2Plaquettes_[i]);

	   Pixel26->cd(i+1);
	   gPad->SetLogy();
	   recHitXResDisk2Plaquettes_[i]->SetLineColor(2);
	   recHitXResDisk2Plaquettes_[i]->Sumw2();
	   newrecHitXResDisk2Plaquettes_[i]->Sumw2();
	   scale = 1.0/recHitXResDisk2Plaquettes_[i]->Integral();
	   recHitXResDisk2Plaquettes_[i]->Scale(scale);
	   scale = 1.0/newrecHitXResDisk2Plaquettes_[i]->Integral();
	   newrecHitXResDisk2Plaquettes_[i]->Scale(scale);
	   recHitXResDisk2Plaquettes_[i]->Draw("h");
	   newrecHitXResDisk2Plaquettes_[i]->SetLineColor(4);
	   newrecHitXResDisk2Plaquettes_[i]->SetLineStyle(2);
	   newrecHitXResDisk2Plaquettes_[i]->Draw("sameh");
	   myPV->PVCompute(recHitXResDisk2Plaquettes_[i], newrecHitXResDisk2Plaquettes_[i], te);
	   
	 }
       Pixel26->Print("RecHit_XRes_disk2_plaquettes.eps");
     }
   
   if (1) 
     {
       TH1* recHitYResDisk1Plaquettes_[7];
       TH1* newrecHitYResDisk1Plaquettes_[7];
       Pixel27 = new TCanvas("Pixel27", "Pixel27", 800, 1200);
       Pixel27->Divide(2,4);
       
       for (Int_t i=0; i<7; i++)
	 {
	   //RecHit Y resolution by plaquette for Disk1
	   
	   sprintf(histo, "DQMData/recHitFPIX/RecHit_YRes_Disk1_Plaquette%d", i+1);
	   rfile->GetObject(histo, recHitYResDisk1Plaquettes_[i]);
	   sfile->GetObject(histo, newrecHitYResDisk1Plaquettes_[i]);

	   Pixel27->cd(i+1);
	   gPad->SetLogy();
	   recHitYResDisk1Plaquettes_[i]->SetLineColor(2);
	   recHitYResDisk1Plaquettes_[i]->Sumw2();
	   newrecHitYResDisk1Plaquettes_[i]->Sumw2();
	   scale = 1.0/recHitYResDisk1Plaquettes_[i]->Integral();
	   recHitYResDisk1Plaquettes_[i]->Scale(scale);
	   scale = 1.0/newrecHitYResDisk1Plaquettes_[i]->Integral();
	   newrecHitYResDisk1Plaquettes_[i]->Scale(scale);
	   recHitYResDisk1Plaquettes_[i]->Draw("h");
	   newrecHitYResDisk1Plaquettes_[i]->SetLineColor(4);
	   newrecHitYResDisk1Plaquettes_[i]->SetLineStyle(2);
	   newrecHitYResDisk1Plaquettes_[i]->Draw("sameh");
	   myPV->PVCompute(recHitYResDisk1Plaquettes_[i], newrecHitYResDisk1Plaquettes_[i], te);
	   
	 }
       Pixel27->Print("RecHit_YRes_disk1_plaquettes.eps");
     }
   
   if (1) 
     {
       TH1* recHitYResDisk2Plaquettes_[7];
       TH1* newrecHitYResDisk2Plaquettes_[7];
       Pixel28 = new TCanvas("Pixel28", "Pixel28", 800, 1200);
       Pixel28->Divide(2,4);
       
       for (Int_t i=0; i<7; i++) 
	 {
	   //RecHit X resolution by plaquette for Disk2
	   
	   sprintf(histo, "DQMData/recHitFPIX/RecHit_YRes_Disk2_Plaquette%d", i+1);
	   rfile->GetObject(histo, recHitYResDisk2Plaquettes_[i]);
	   sfile->GetObject(histo, newrecHitYResDisk2Plaquettes_[i]);

	   Pixel28->cd(i+1);
	   gPad->SetLogy();
	   recHitYResDisk2Plaquettes_[i]->SetLineColor(2);
	   recHitYResDisk2Plaquettes_[i]->Sumw2();
	   newrecHitYResDisk2Plaquettes_[i]->Sumw2();
	   scale = 1.0/recHitYResDisk2Plaquettes_[i]->Integral();
	   recHitYResDisk2Plaquettes_[i]->Scale(scale);
	   scale = 1.0/newrecHitYResDisk2Plaquettes_[i]->Integral();
	   newrecHitYResDisk2Plaquettes_[i]->Scale(scale);
	   recHitYResDisk2Plaquettes_[i]->Draw("h");
	   newrecHitYResDisk2Plaquettes_[i]->SetLineColor(4);
	   newrecHitYResDisk2Plaquettes_[i]->SetLineStyle(2);
	   newrecHitYResDisk2Plaquettes_[i]->Draw("sameh");
	   myPV->PVCompute(recHitYResDisk2Plaquettes_[i], newrecHitYResDisk2Plaquettes_[i], te);
	   
	 }
       Pixel28->Print("RecHit_YRes_disk2_plaquettes.eps");
     }
   
      
   //------------------------
   // RecHit Pull Distributions Barrel
   //------------------------
   
   if (1) 
     {
       TH1* recHitXPullFlippedLadderLayers_[3];
       TH1* newrecHitXPullFlippedLadderLayers_[3];
       Pixel29 = new TCanvas("Pixel29", "Pixel29", 400, 600);
       Pixel29->Divide(1,3);
       
       for (Int_t i=0; i<3; i++) 
	 {
	   //RecHit XPull Flipped ladders by layer
	   
	   sprintf(histo, "DQMData/recHitPullsBPIX/RecHit_XPull_FlippedLadder_Layer%d", i+1);
	   rfile->GetObject(histo, recHitXPullFlippedLadderLayers_[i]);
	   sfile->GetObject(histo, newrecHitXPullFlippedLadderLayers_[i]);

	   Pixel29->cd(i+1);
	   //gPad->SetLogy();
	   recHitXPullFlippedLadderLayers_[i]->SetLineColor(2);
	   recHitXPullFlippedLadderLayers_[i]->Sumw2();
	   newrecHitXPullFlippedLadderLayers_[i]->Sumw2();
	   scale = 1.0/recHitXPullFlippedLadderLayers_[i]->Integral();
	   recHitXPullFlippedLadderLayers_[i]->Scale(scale);
	   scale = 1.0/newrecHitXPullFlippedLadderLayers_[i]->Integral();
	   newrecHitXPullFlippedLadderLayers_[i]->Scale(scale);
	   recHitXPullFlippedLadderLayers_[i]->Draw("h");
	   newrecHitXPullFlippedLadderLayers_[i]->SetLineColor(4);
	   newrecHitXPullFlippedLadderLayers_[i]->SetLineStyle(2);
	   newrecHitXPullFlippedLadderLayers_[i]->Draw("sameh");
	   myPV->PVCompute(recHitXPullFlippedLadderLayers_[i], newrecHitXPullFlippedLadderLayers_[i], te);
	   
	}
       Pixel29->Print("RecHit_XPull_FlippedLadder_Layers.eps");
     }
   
   if (1) 
     {
       TH1* recHitXPullUnFlippedLadderLayers_[3];
       TH1* newrecHitXPullUnFlippedLadderLayers_[3];
       Pixel30 = new TCanvas("Pixel30", "Pixel30", 400, 600);
       Pixel30->Divide(1,3);
       
       for (Int_t i=0; i<3; i++) 
	 {
	   //RecHit XPull unflipped ladders by layer
	   
	   sprintf(histo, "DQMData/recHitPullsBPIX/RecHit_XPull_UnFlippedLadder_Layer%d", i+1);
	   rfile->GetObject(histo, recHitXPullUnFlippedLadderLayers_[i]);
	   sfile->GetObject(histo, newrecHitXPullUnFlippedLadderLayers_[i]);

	   Pixel30->cd(i+1);
	   //gPad->SetLogy();
	   recHitXPullUnFlippedLadderLayers_[i]->SetLineColor(2);
	   recHitXPullUnFlippedLadderLayers_[i]->Sumw2();
	   newrecHitXPullUnFlippedLadderLayers_[i]->Sumw2();
	   scale = 1.0/recHitXPullUnFlippedLadderLayers_[i]->Integral();
	   recHitXPullUnFlippedLadderLayers_[i]->Scale(scale);
	   scale = 1.0/newrecHitXPullUnFlippedLadderLayers_[i]->Integral();
	   newrecHitXPullUnFlippedLadderLayers_[i]->Scale(scale);
	   recHitXPullUnFlippedLadderLayers_[i]->Draw("h");
	   newrecHitXPullUnFlippedLadderLayers_[i]->SetLineColor(4);
	   newrecHitXPullUnFlippedLadderLayers_[i]->SetLineStyle(2);
	   newrecHitXPullUnFlippedLadderLayers_[i]->Draw("sameh");
	   myPV->PVCompute(recHitXPullUnFlippedLadderLayers_[i], newrecHitXPullUnFlippedLadderLayers_[i], te);
	   
	 }
       Pixel30->Print("RecHit_XPull_UnFlippedLadder_Layers.eps");
     }
   
   if (1) 
     {
       TH1* recHitPullYPullLayer1Modules_[8];
       TH1* newrecHitPullYPullLayer1Modules_[8];
       Pixel31 = new TCanvas("Pixel31", "Pixel31", 800, 1200);
       Pixel31->Divide(2,4);
       
       for (Int_t i=0; i<8; i++) 
	 {
	   //RecHit Y pullolution by module for layer 1
	   
	   sprintf(histo, "DQMData/recHitPullsBPIX/RecHit_YPull_Layer1_Module%d", i+1);
	   rfile->GetObject(histo, recHitPullYPullLayer1Modules_[i]);
	   sfile->GetObject(histo, newrecHitPullYPullLayer1Modules_[i]);

	   Pixel31->cd(i+1);
	   //gPad->SetLogy();
	   recHitPullYPullLayer1Modules_[i]->SetLineColor(2);
	   recHitPullYPullLayer1Modules_[i]->Sumw2();
	   newrecHitPullYPullLayer1Modules_[i]->Sumw2();
	   scale = 1.0/recHitPullYPullLayer1Modules_[i]->Integral();
	   recHitPullYPullLayer1Modules_[i]->Scale(scale);
	   scale = 1.0/newrecHitPullYPullLayer1Modules_[i]->Integral();
	   newrecHitPullYPullLayer1Modules_[i]->Scale(scale);
	   recHitPullYPullLayer1Modules_[i]->Draw("h");
	   newrecHitPullYPullLayer1Modules_[i]->SetLineColor(4);
	   newrecHitPullYPullLayer1Modules_[i]->SetLineStyle(2);
	   newrecHitPullYPullLayer1Modules_[i]->Draw("sameh");
	   myPV->PVCompute(recHitPullYPullLayer1Modules_[i], newrecHitPullYPullLayer1Modules_[i], te);
	   
	 }
       Pixel31->Print("RecHit_YPull_Layer1_Modules.eps");
     }
   
   if (1) 
     {
       TH1* recHitPullYPullLayer2Modules_[8];
       TH1* newrecHitPullYPullLayer2Modules_[8];
       Pixel32 = new TCanvas("Pixel32", "Pixel32", 800, 1200);
       Pixel32->Divide(2,4);
       
       for (Int_t i=0; i<8; i++) 
	 {
	   //RecHit Y pullolution by module for layer 2
	   
	   sprintf(histo, "DQMData/recHitPullsBPIX/RecHit_YPull_Layer2_Module%d", i+1);
	   rfile->GetObject(histo, recHitPullYPullLayer2Modules_[i]);
	   sfile->GetObject(histo, newrecHitPullYPullLayer2Modules_[i]);
	   
	   Pixel32->cd(i+1);
	   // gPad->SetLogy();
	   recHitPullYPullLayer2Modules_[i]->SetLineColor(2);
	   recHitPullYPullLayer2Modules_[i]->Sumw2();
	   newrecHitPullYPullLayer2Modules_[i]->Sumw2();
	   scale = 1.0/recHitPullYPullLayer2Modules_[i]->Integral();
	   recHitPullYPullLayer2Modules_[i]->Scale(scale);
	   scale = 1.0/newrecHitPullYPullLayer2Modules_[i]->Integral();
	   newrecHitPullYPullLayer2Modules_[i]->Scale(scale);
	   recHitPullYPullLayer2Modules_[i]->Draw("h");
	   newrecHitPullYPullLayer2Modules_[i]->SetLineColor(4);
	   newrecHitPullYPullLayer2Modules_[i]->SetLineStyle(2);
	   newrecHitPullYPullLayer2Modules_[i]->Draw("sameh");
	   myPV->PVCompute(recHitPullYPullLayer2Modules_[i], newrecHitPullYPullLayer2Modules_[i], te);
	   
	 }
       Pixel32->Print("RecHit_YPull_Layer2_Modules.eps");
     }

   if (1) 
     {
       TH1* recHitPullYPullLayer3Modules_[8];
       TH1* newrecHitPullYPullLayer3Modules_[8];
       Pixel33 = new TCanvas("Pixel33", "Pixel33", 800, 1200);
       Pixel33->Divide(2,4);
       
       for (Int_t i=0; i<8; i++)
	 {
	   //RecHit Y pullolution by module for layer 3
	   
	   sprintf(histo, "DQMData/recHitPullsBPIX/RecHit_YPull_Layer3_Module%d", i+1);
	   rfile->GetObject(histo, recHitPullYPullLayer3Modules_[i]);
	   sfile->GetObject(histo, newrecHitPullYPullLayer3Modules_[i]);

	   Pixel33->cd(i+1);
	   // gPad->SetLogy();
	   recHitPullYPullLayer3Modules_[i]->SetLineColor(2);
	   recHitPullYPullLayer3Modules_[i]->Sumw2();
	   newrecHitPullYPullLayer3Modules_[i]->Sumw2();
	   scale = 1.0/recHitPullYPullLayer3Modules_[i]->Integral();
	   recHitPullYPullLayer3Modules_[i]->Scale(scale);
	   scale = 1.0/newrecHitPullYPullLayer3Modules_[i]->Integral();
	   newrecHitPullYPullLayer3Modules_[i]->Scale(scale);
	   recHitPullYPullLayer3Modules_[i]->Draw("h");
	   newrecHitPullYPullLayer3Modules_[i]->SetLineColor(4);
	   newrecHitPullYPullLayer3Modules_[i]->SetLineStyle(2);
	   newrecHitPullYPullLayer3Modules_[i]->Draw("sameh");
	   myPV->PVCompute(recHitPullYPullLayer3Modules_[i], newrecHitPullYPullLayer3Modules_[i], te);
	   
	 }
       Pixel33->Print("RecHit_YPull_Layer3_Modules.eps");
     }
   
   ////////////////////////////////
   //RecHit forward
   ////////////////////////////////
   
   
   if (1) 
     {
       TH1* recHitXPullDisk1Plaquettes_[7];
       TH1* newrecHitXPullDisk1Plaquettes_[7];
       Pixel34 = new TCanvas("Pixel34", "Pixel34", 800, 1200);
       Pixel34->Divide(2,4);
       
       for (Int_t i=0; i<7; i++) 
	 {
	   //RecHit X pullolution by plaquette for Disk1
	   
	   sprintf(histo, "DQMData/recHitPullsFPIX/RecHit_XPull_Disk1_Plaquette%d", i+1);
	   rfile->GetObject(histo, recHitXPullDisk1Plaquettes_[i]);
	   sfile->GetObject(histo, newrecHitXPullDisk1Plaquettes_[i]);

	   Pixel34->cd(i+1);
	   //gPad->SetLogy();
	   recHitXPullDisk1Plaquettes_[i]->SetLineColor(2);
	   recHitXPullDisk1Plaquettes_[i]->Sumw2();
	   newrecHitXPullDisk1Plaquettes_[i]->Sumw2();
	   scale = 1.0/recHitXPullDisk1Plaquettes_[i]->Integral();
	   recHitXPullDisk1Plaquettes_[i]->Scale(scale);
	   scale = 1.0/newrecHitXPullDisk1Plaquettes_[i]->Integral();
	   newrecHitXPullDisk1Plaquettes_[i]->Scale(scale);
	   recHitXPullDisk1Plaquettes_[i]->Draw("h");
	   newrecHitXPullDisk1Plaquettes_[i]->SetLineColor(4);
	   newrecHitXPullDisk1Plaquettes_[i]->SetLineStyle(2);
	   newrecHitXPullDisk1Plaquettes_[i]->Draw("sameh");
	   myPV->PVCompute(recHitXPullDisk1Plaquettes_[i], newrecHitXPullDisk1Plaquettes_[i], te);
	   
	 }
       Pixel34->Print("RecHit_XPull_disk1_plaquettes.eps");
     }
   
   if (1) 
     {
       TH1* recHitXPullDisk2Plaquettes_[7];
       TH1* newrecHitXPullDisk2Plaquettes_[7];
       Pixel35 = new TCanvas("Pixel35", "Pixel35", 800, 1200);
       Pixel35->Divide(2,4);
       
       for (Int_t i=0; i<7; i++) 
	 {
	   //RecHit X pullolution by plaquette for Disk2
	   
	   sprintf(histo, "DQMData/recHitPullsFPIX/RecHit_XPull_Disk2_Plaquette%d", i+1);
	   rfile->GetObject(histo, recHitXPullDisk2Plaquettes_[i]);
	   sfile->GetObject(histo, newrecHitXPullDisk2Plaquettes_[i]);

	   Pixel35->cd(i+1);
	   //gPad->SetLogy();
	   recHitXPullDisk2Plaquettes_[i]->SetLineColor(2);
	   recHitXPullDisk2Plaquettes_[i]->Sumw2();
	   newrecHitXPullDisk2Plaquettes_[i]->Sumw2();
	   scale = 1.0/recHitXPullDisk2Plaquettes_[i]->Integral();
	   recHitXPullDisk2Plaquettes_[i]->Scale(scale);
	   scale = 1.0/newrecHitXPullDisk2Plaquettes_[i]->Integral();
	   newrecHitXPullDisk2Plaquettes_[i]->Scale(scale);
	   recHitXPullDisk2Plaquettes_[i]->Draw("h");
	   newrecHitXPullDisk2Plaquettes_[i]->SetLineColor(4);
	   newrecHitXPullDisk2Plaquettes_[i]->SetLineStyle(2);
	   newrecHitXPullDisk2Plaquettes_[i]->Draw("sameh");
	   myPV->PVCompute(recHitXPullDisk2Plaquettes_[i], newrecHitXPullDisk2Plaquettes_[i], te);
	   
	 }
       Pixel35->Print("RecHit_XPull_disk2_plaquettes.eps");
     }
   
   if (1) 
     {
       TH1* recHitPullYPullDisk1Plaquettes_[7];
       TH1* newrecHitPullYPullDisk1Plaquettes_[7];
       Pixel36 = new TCanvas("Pixel36", "Pixel36", 800, 1200);
       Pixel36->Divide(2,4);
       
       for (Int_t i=0; i<7; i++) 
	 {
	   //RecHit Y pullolution by plaquette for Disk1
	   
	   sprintf(histo, "DQMData/recHitPullsFPIX/RecHit_YPull_Disk1_Plaquette%d", i+1);
	   rfile->GetObject(histo, recHitPullYPullDisk1Plaquettes_[i]);
	   sfile->GetObject(histo, newrecHitPullYPullDisk1Plaquettes_[i]);
	   
	   Pixel36->cd(i+1);
	   //gPad->SetLogy();
	   recHitPullYPullDisk1Plaquettes_[i]->SetLineColor(2);
	   recHitPullYPullDisk1Plaquettes_[i]->Sumw2();
	   newrecHitPullYPullDisk1Plaquettes_[i]->Sumw2();
	   scale = 1.0/recHitPullYPullDisk1Plaquettes_[i]->Integral();
	   recHitPullYPullDisk1Plaquettes_[i]->Scale(scale);
	   scale = 1.0/newrecHitPullYPullDisk1Plaquettes_[i]->Integral();
	   newrecHitPullYPullDisk1Plaquettes_[i]->Scale(scale);
	   recHitPullYPullDisk1Plaquettes_[i]->Draw("h");
	   newrecHitPullYPullDisk1Plaquettes_[i]->SetLineColor(4);
	   newrecHitPullYPullDisk1Plaquettes_[i]->SetLineStyle(2);
	   newrecHitPullYPullDisk1Plaquettes_[i]->Draw("sameh");
	   myPV->PVCompute(recHitPullYPullDisk1Plaquettes_[i], newrecHitPullYPullDisk1Plaquettes_[i], te);
	   
	 }
       Pixel36->Print("RecHit_YPull_disk1_plaquettes.eps");
     }
   
   if (1) 
     {
       TH1* recHitPullYPullDisk2Plaquettes_[7];
       TH1* newrecHitPullYPullDisk2Plaquettes_[7];
       Pixel37 = new TCanvas("Pixel37", "Pixel37", 800, 1200);
       Pixel37->Divide(2,4);
       
       for (Int_t i=0; i<7; i++) 
	 {
	   //RecHit X pullolution by plaquette for Disk2
	   
	   sprintf(histo, "DQMData/recHitPullsFPIX/RecHit_YPull_Disk2_Plaquette%d", i+1);
	   rfile->GetObject(histo, recHitPullYPullDisk2Plaquettes_[i]);
	   sfile->GetObject(histo, newrecHitPullYPullDisk2Plaquettes_[i]);

	   Pixel37->cd(i+1);
	   //gPad->SetLogy();
	   recHitPullYPullDisk2Plaquettes_[i]->SetLineColor(2);
	   recHitPullYPullDisk2Plaquettes_[i]->Sumw2();
	   newrecHitPullYPullDisk2Plaquettes_[i]->Sumw2();
	   scale = 1.0/recHitPullYPullDisk2Plaquettes_[i]->Integral();
	   recHitPullYPullDisk2Plaquettes_[i]->Scale(scale);
	   scale = 1.0/newrecHitPullYPullDisk2Plaquettes_[i]->Integral();
	   newrecHitPullYPullDisk2Plaquettes_[i]->Scale(scale);
	   recHitPullYPullDisk2Plaquettes_[i]->Draw("h");
	   newrecHitPullYPullDisk2Plaquettes_[i]->SetLineColor(4);
	   newrecHitPullYPullDisk2Plaquettes_[i]->SetLineStyle(2);
	   newrecHitPullYPullDisk2Plaquettes_[i]->Draw("sameh");
	   myPV->PVCompute(recHitPullYPullDisk2Plaquettes_[i], newrecHitPullYPullDisk2Plaquettes_[i], te);
	   
	}
       Pixel37->Print("RecHit_YPull_disk2_plaquettes.eps");
     }

} // end
