void TrackingTruthCompare()
{
  
  gROOT ->Reset();
  char*  sfilename = "trackingtruthhisto.root";
  char*  rfilename = "../data/trackingtruthhisto.root";
  
  delete gROOT->GetListOfFiles()->FindObject(rfilename);
  delete gROOT->GetListOfFiles()->FindObject(sfilename); 
  
  TText* te = new TText();
  TFile * rfile = new TFile(rfilename);
  TFile * sfile = new TFile(sfilename);
  
  rfile->cd("DQMData/TrackingMCTruth/TrackingParticle");
  sfile->cd("DQMData/TrackingMCTruth/TrackingParticle");
  TLegend leg(0.3, 0.83, 0.55, 0.90);
  //Get list of Keys from the Reference file.
  TList* ref_list = rfile->GetListOfKeys() ;
  if (!ref_list) {
    std::cout<<"=========>> AutoComaprison:: There is no Keys available in the Reference file."<<std::endl;
    exit(1) ;
  }
  
  //Get list of Keys from the New file.
  TList* new_list = sfile->GetListOfKeys() ;
  if (!new_list) {
    std::cout<<"=========>> AutoComaprison:: There is no Keys available in New file."<<std::endl;
    exit(1) ;
  }
  
  
  //Iterate on the List of Keys of the  Reference file.
  TIter     refkey_iter( ref_list) ;
  TKey*     ref_key ;
  TObject*  ref_obj ;
  
  char rver[50];
  char cver[50];
  while ( ref_key = (TKey*) refkey_iter() ) {
    ref_obj = ref_key->ReadObj() ;
    if (strcmp(ref_obj->IsA()->GetName(),"TObjString")==0) {
      
      TObjString * rversion = dynamic_cast< TObjString*> (ref_obj);
      sprintf(rver, "%s", rversion->GetName());
      std::cout<<" Ref. version =" << rver<<std::endl;
      break;
    }
  }
  
  //Iterate on the List of Keys of the  Reference file.
  TIter     newkey_iter( new_list) ;
  TKey*     new_key ;
  TObject*  new_obj ;
  while ( new_key = (TKey*) newkey_iter() ) {
    new_obj = new_key->ReadObj() ;
    if (strcmp(new_obj->IsA()->GetName(),"TObjString")==0) {
      
      TObjString * cversion = dynamic_cast< TObjString*> (new_obj);
      sprintf(cver, "%s", cversion->GetName());
      std::cout<<" Cur version =" << cver<<std::endl;
      break;
      
    }
  }
  
  //gDirectory->ls();
  
  Char_t histo[200];
  
  gROOT->ProcessLine(".x HistoCompare.C");
  HistoCompare * myPV = new HistoCompare();
  
  
  if (1) {
    TCanvas * TrackingParticleGV = new TCanvas("TrackingParticleGV","TrackingParticleGV",1000,1000);
    TrackingParticleGV->Divide(3,2);
    
    TH1* meTPmass;
    TH1* meTPcharge;
    TH1* meTPid;
    TH1* meTPhits;
    TH1* meTPmhits;
    
    TH1* newmeTPmass;
    TH1* newmeTPcharge;
    TH1* newmeTPid;
    TH1* newmeTPhits;
    TH1* newmeTPmhits;
    
    sprintf(histo,"DQMData/TrackingMCTruth/TrackingParticle/TPMass");
    rfile->GetObject( histo,meTPmass);
    sfile->GetObject(histo,newmeTPmass);
    meTPmass;
    newmeTPmass;
    TrackingParticleGV->cd(1);
    meTPmass->SetLineColor(2);
    newmeTPmass->SetLineColor(4);
    newmeTPmass->SetLineStyle(2);
    meTPmass->Draw();
    newmeTPmass->Draw("sames");
    myPV->PVCompute(meTPmass , newmeTPmass , te );
    leg.Clear();
    leg.AddEntry(meTPmass,rver , "l");
    leg.AddEntry(newmeTPmass,cver , "l");
    leg.Draw();

    sprintf(histo,"DQMData/TrackingMCTruth/TrackingParticle/TPCharge");
    rfile->GetObject( histo,meTPcharge);
    sfile->GetObject(histo,newmeTPcharge);
    meTPcharge;
    newmeTPcharge;
    TrackingParticleGV->cd(2);
    meTPcharge->SetLineColor(2);
    newmeTPcharge->SetLineColor(4);
    newmeTPcharge->SetLineStyle(2);
    meTPcharge->Draw();
    newmeTPcharge->Draw("sames");
    myPV->PVCompute(meTPcharge , newmeTPcharge , te );
    leg.Clear();
    leg.AddEntry(meTPcharge,rver , "l");
    leg.AddEntry(newmeTPcharge,cver , "l");
    leg.Draw();
    
    sprintf(histo,"DQMData/TrackingMCTruth/TrackingParticle/TPId");
    rfile->GetObject( histo,meTPid);
    sfile->GetObject(histo,newmeTPid);
    meTPid;
    newmeTPid;
    TrackingParticleGV->cd(3);
    meTPid->SetLineColor(2);
    newmeTPid->SetLineColor(4);
    newmeTPid->SetLineStyle(2);
    meTPid->Draw();
    newmeTPid->Draw("sames");
    myPV->PVCompute(meTPid , newmeTPid , te );
    leg.Clear();
    leg.AddEntry(meTPid,rver , "l");
    leg.AddEntry(newmeTPid,cver , "l");
    leg.Draw();

    sprintf(histo,"DQMData/TrackingMCTruth/TrackingParticle/TPAllHits");
    rfile->GetObject( histo,meTPhits);
    sfile->GetObject(histo,newmeTPhits);
    meTPhits;
    newmeTPhits;
    TrackingParticleGV->cd(4);
    meTPhits->SetLineColor(2);
    newmeTPhits->SetLineColor(4);
    newmeTPhits->SetLineStyle(2);
    meTPhits->Draw();
    newmeTPhits->Draw("sames");
    myPV->PVCompute(meTPhits , newmeTPhits , te );
    leg.Clear();
    leg.AddEntry(meTPhits,rver , "l");
    leg.AddEntry(newmeTPhits,cver , "l");
    leg.Draw();

    sprintf(histo,"DQMData/TrackingMCTruth/TrackingParticle/TPMatchedHits");
    rfile->GetObject( histo,meTPmhits);
    sfile->GetObject(histo,newmeTPmhits);
    meTPmhits;
    newmeTPmhits;
    TrackingParticleGV->cd(5);
    meTPmhits->SetLineColor(2);
    newmeTPmhits->SetLineColor(4);
    newmeTPmhits->SetLineStyle(2);
    meTPmhits->Draw();
    newmeTPmhits->Draw("sames");
    myPV->PVCompute(meTPmhits , newmeTPmhits , te );
    leg.Clear();
    leg.AddEntry(meTPmhits,rver , "l");
    leg.AddEntry(newmeTPmhits,cver , "l");
    leg.Draw();
    
    TrackingParticleGV->Print("TPGeneralVariables.eps");
    TrackingParticleGV->Print("TPGeneralVariables.gif");
 }

 if (1) {
  TCanvas * TrackingParticleTV = new TCanvas("TrackingParticleTV","TrackingParticleTV",1000,1000);
   TrackingParticleTV->Divide(4,2);

   TH1* meTPpt;
   TH1* meTPeta;
   TH1* meTPphi;
   TH1* meTPvtxx;
   TH1* meTPvtxy;
   TH1* meTPvtxz;
   TH1* meTPlip;
   TH1* meTPtip;

   TH1* newmeTPpt;
   TH1* newmeTPeta;
   TH1* newmeTPphi;
   TH1* newmeTPvtxx;
   TH1* newmeTPvtxy;
   TH1* newmeTPvtxz;
   TH1* newmeTPlip;
   TH1* newmeTPtip;

   sprintf(histo,"DQMData/TrackingMCTruth/TrackingParticle/TPPt");
   rfile->GetObject(histo ,meTPpt);
   sfile->GetObject(histo ,newmeTPpt);
   meTPpt;
   newmeTPpt;
   TrackingParticleTV->cd(1);
   meTPpt->SetLineColor(2);
   newmeTPpt->SetLineColor(4);
   newmeTPpt->SetLineStyle(2);
   meTPpt->Draw();
   newmeTPpt->Draw("sames");
   myPV->PVCompute(meTPpt , newmeTPpt , te );
   leg.Clear();
   leg.AddEntry(meTPpt,rver , "l");
   leg.AddEntry(newmeTPpt,cver , "l");
   leg.Draw();

   sprintf(histo,"DQMData/TrackingMCTruth/TrackingParticle/TPEta");
   rfile->GetObject(histo ,meTPeta);
   sfile->GetObject(histo ,newmeTPeta);
   meTPeta;
   newmeTPeta;
   TrackingParticleTV->cd(2);
   meTPeta->SetLineColor(2);
   newmeTPeta->SetLineColor(4);
   newmeTPeta->SetLineStyle(2);
   meTPeta->Draw();
   newmeTPeta->Draw("sames");
   myPV->PVCompute(meTPeta , newmeTPeta , te );
   leg.Clear();
   leg.AddEntry(meTPeta,rver , "l");
   leg.AddEntry(newmeTPeta,cver , "l");
   leg.Draw();

   sprintf(histo,"DQMData/TrackingMCTruth/TrackingParticle/TPPhi");
   rfile->GetObject(histo ,meTPphi);
   sfile->GetObject(histo ,newmeTPphi);
   meTPphi;
   newmeTPphi;
   TrackingParticleTV->cd(3);
   meTPphi->SetLineColor(2);
   newmeTPphi->SetLineColor(4);
   newmeTPphi->SetLineStyle(2);
   meTPphi->Draw();
   newmeTPphi->Draw("sames");
   myPV->PVCompute(meTPphi , newmeTPphi , te );
   leg.Clear();
   leg.AddEntry(meTPphi,rver , "l");
   leg.AddEntry(newmeTPphi,cver , "l");
   leg.Draw();
   
   sprintf(histo,"DQMData/TrackingMCTruth/TrackingParticle/TPVtxX");
   rfile->GetObject(histo ,meTPvtxx);
   sfile->GetObject(histo ,newmeTPvtxx);
   meTPvtxx;
   newmeTPvtxx;
   TrackingParticleTV->cd(4);
   meTPvtxx->SetLineColor(2);
   newmeTPvtxx->SetLineColor(4);
   newmeTPvtxx->SetLineStyle(2);
   meTPvtxx->Draw();
   newmeTPvtxx->Draw("sames");
   myPV->PVCompute(meTPvtxx , newmeTPvtxx , te );
   leg.Clear();
   leg.AddEntry(meTPvtxx,rver , "l");
   leg.AddEntry(newmeTPvtxx,cver , "l");
   leg.Draw();

   sprintf(histo,"DQMData/TrackingMCTruth/TrackingParticle/TPVtxY");
   rfile->GetObject(histo ,meTPvtxy);
   sfile->GetObject(histo ,newmeTPvtxy);
   meTPvtxy;
   newmeTPvtxy;
   TrackingParticleTV->cd(5);
   meTPvtxy->SetLineColor(2);
   newmeTPvtxy->SetLineColor(4);
   newmeTPvtxy->SetLineStyle(2);
   meTPvtxy->Draw();
   newmeTPvtxy->Draw("sames");
   myPV->PVCompute(meTPvtxy , newmeTPvtxy , te );
   leg.Clear();
   leg.AddEntry(meTPvtxy,rver , "l");
   leg.AddEntry(newmeTPvtxy,cver , "l");
   leg.Draw();

   sprintf(histo,"DQMData/TrackingMCTruth/TrackingParticle/TPVtxZ");
   rfile->GetObject(histo ,meTPvtxz);
   sfile->GetObject(histo ,newmeTPvtxz);
   meTPvtxz;
   newmeTPvtxz;
   TrackingParticleTV->cd(6);
   meTPvtxz->SetLineColor(2);
   newmeTPvtxz->SetLineColor(4);
   newmeTPvtxz->SetLineStyle(2);
   meTPvtxz->Draw();
   newmeTPvtxz->Draw("sames");
   myPV->PVCompute(meTPvtxz , newmeTPvtxz , te );
   leg.Clear();
   leg.AddEntry(meTPvtxz,rver , "l");
   leg.AddEntry(newmeTPvtxz,cver , "l");
   leg.Draw();

   sprintf(histo,"DQMData/TrackingMCTruth/TrackingParticle/TPtip");
   rfile->GetObject(histo ,meTPtip);
   sfile->GetObject(histo ,newmeTPtip);
   meTPtip;
   newmeTPtip;
   TrackingParticleTV->cd(7);
   meTPtip->SetLineColor(2);
   newmeTPtip->SetLineColor(4);
   newmeTPtip->SetLineStyle(2);
   meTPtip->Draw();
   newmeTPtip->Draw("sames");
   myPV->PVCompute(meTPtip , newmeTPtip , te );
   leg.Clear();
   leg.AddEntry(meTPtip,rver , "l");
   leg.AddEntry(newmeTPtip,cver , "l");
   leg.Draw();

   sprintf(histo,"DQMData/TrackingMCTruth/TrackingParticle/TPlip");
   rfile->GetObject(histo ,meTPlip);
   sfile->GetObject(histo ,newmeTPlip);
   meTPlip;
   newmeTPlip;
   TrackingParticleTV->cd(8);
   meTPlip->SetLineColor(2);
   newmeTPlip->SetLineColor(4);
   newmeTPlip->SetLineStyle(2);
   meTPlip->Draw();
   newmeTPlip->Draw("sames");
   myPV->PVCompute(meTPlip , newmeTPlip , te );
   leg.Clear();
   leg.AddEntry(meTPlip,rver , "l");
   leg.AddEntry(newmeTPlip,cver , "l");
   leg.Draw();

   TrackingParticleTV->Print("TPGeneralVariable.eps");
   TrackingParticleTV->Print("TPGeneralVariable.gif");
 }
}

