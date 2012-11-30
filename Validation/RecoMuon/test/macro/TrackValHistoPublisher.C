void TrackValHistoPublisher(char* newFile="NEW_FILE",char* refFile="REF_FILE")
{
  //gROOT->ProcessLine(".x HistoCompare_Tracks.C");
 gROOT ->Reset();
 gROOT ->SetBatch();

 //=========  settings ====================

 char* dataType = "DATATYPE";

 gROOT->SetStyle("Plain");
 gStyle->SetPadGridX(kTRUE);
 gStyle->SetPadGridY(kTRUE);
 gStyle->SetPadRightMargin(0.07);
 gStyle->SetPadLeftMargin(0.13);
 //gStyle->SetTitleXSize(0.07); 
 //gStyle->SetTitleXOffset(0.6); 
 //tyle->SetTitleYSize(0.3);
 //gStyle->SetLabelSize(0.6) 
 //gStyle->SetTextSize(0.5);
 char* refLabel("REF_LABEL, REF_RELEASE REFSELECTION");
 char* newLabel("NEW_LABEL, NEW_RELEASE NEWSELECTION");

 Float_t maxPT=1500.;


 //=============================================


 delete gROOT->GetListOfFiles()->FindObject(refFile);
 delete gROOT->GetListOfFiles()->FindObject(newFile); 

 TText* te = new TText();
 TFile * sfile = new TFile(newFile);
 TDirectory * sdir=gDirectory;
 TFile * rfile = new TFile(refFile);
 TDirectory * rdir=gDirectory;

 if (dataType == "HLT") {
   if(sfile->cd("DQMData/Run 1/HLT")) {sfile->cd("DQMData/Run 1/HLT/Run summary/Muon/MultiTrack");}
   else {sfile->cd("DQMData/HLT/Muon/MultiTrack");}
 }
 else if (dataType == "RECO") {
   if(sfile->cd("DQMData/Run 1/RecoMuonV")) {sfile->cd("DQMData/Run 1/RecoMuonV/Run summary/MultiTrack");}
   else if(sfile->cd("DQMData/Run 1/Muons/Run summary/RecoMuonV")) {sfile->cd("DQMData/Run 1/Muons/Run summary/RecoMuonV/MultiTrack");}
   else {sfile->cd("DQMData/RecoMuonV/MultiTrack");}
 }
 else {
   cout << " Data type " << dataType << " not allowed: only RECO and HLT are considered" << endl;
   return;
 }
 sdir=gDirectory;
 //TList *sl= sdir->GetListOfKeys();
 TIter nextkey( sdir->GetListOfKeys() );
 TList *sl = new TList();
 TKey *key, *oldkey=0;
 cout << "- New collections: " << endl;
 while ( (key = (TKey*)nextkey())) {
   TObject *obj = key->ReadObj();
   if ( obj->IsA()->InheritsFrom( "TDirectory" ) ) {
     cout << obj->GetName() << endl;
     sl->Add(obj);
   }
 }
 TString collname2 =sl->At(0)->GetName(); 
 
 if (dataType == "HLT") {
   if(rfile->cd("DQMData/Run 1/HLT")) rfile->cd("DQMData/Run 1/HLT/Run summary/Muon/MultiTrack");
   else rfile->cd("DQMData/HLT/Muon/MultiTrack");
 }
 else if (dataType == "RECO") {
   if(rfile->cd("DQMData/Run 1/RecoMuonV")) rfile->cd("DQMData/Run 1/RecoMuonV/Run summary/MultiTrack");
   else if(rfile->cd("DQMData/Run 1/Muons/Run summary/RecoMuonV")) {rfile->cd("DQMData/Run 1/Muons/Run summary/RecoMuonV/MultiTrack");}
   else rfile->cd("DQMData/RecoMuonV/MultiTrack");
 }
 rdir=gDirectory;
 //TList *rl= rdir->GetListOfKeys();
 TIter nextkeyr( rdir->GetListOfKeys() );
 TList *rl = new TList();
 TKey *keyr, *oldkeyr=0;
 cout << "- Ref collections: " << endl;
 while ( (keyr = (TKey*)nextkeyr())) {
   TObject *obj = keyr->ReadObj();
   if ( obj->IsA()->InheritsFrom( "TDirectory" ) ) {
     cout << obj->GetName() << endl;
     rl->Add(obj);
   }
 }
 TString collname1=rl->At(0)->GetName();
 //HistoCompare_Tracks * myPV = new HistoCompare_Tracks();
 
 TCanvas *canvas;
 
 TH1F *sh1,*rh1;
 TH1F *sh2,*rh2;
 TH1F *sh3,*rh3;
 TH1F *sh4,*rh4;
 TH1F *sh5,*rh5;
 TH1F *sh6,*rh6;
 
 TH1F *sc1,*rc1;
 TH1F *sc2,*rc2;
 TH1F *sc3,*rc3;
 
 bool hit=1;
 bool chi2=1;
 bool ctf=1;
 bool rs=0;

 bool hasOnlyMuonAssociatorInRef = true, hasOnlyMuonAssociatorInSig = true;
 bool hasOnlyTrackAssociatorInRef = true, hasOnlyTrackAssociatorInSig = true;
 TIter iter_r0( rl );
 TIter iter_s0( sl );
 TKey * myKey;
 while ( (myKey = (TKey*)iter_r0()) ) {
   TString myName = myKey->GetName();
   if ( !(myName.Contains("tpToTkmu")) && !(myName.Contains("MuonAssociation")) ) {
     hasOnlyMuonAssociatorInRef = false;
   }
   if ( !(myName.Contains("tpToTkmu")) && (myName.Contains("MuonAssociation")) ) {
     hasOnlyTrackAssociatorInRef = false;
   }
 }
 while ( (myKey = (TKey*)iter_s0()) ) {
   TString myName = myKey->GetName();
   if ( !(myName.Contains("tpToTkmu")) && !(myName.Contains("MuonAssociation")) ) {
     hasOnlyMuonAssociatorInSig = false;
   }
   if ( !(myName.Contains("tpToTkmu")) && (myName.Contains("MuonAssociation")) ) {
     hasOnlyTrackAssociatorInSig = false;
   }
 }
 bool considerOnlyMuonAssociator = hasOnlyMuonAssociatorInRef || hasOnlyMuonAssociatorInSig;


 TIter iter_r( rl );
 TIter iter_s( sl );
 TKey * myKey1, *myKey2, *myNext2=0;
 while ( (myKey1 = (TKey*)iter_r()) ) {
   TString myName = myKey1->GetName();
   if (!(myName.Contains("tpToTkmu")) && considerOnlyMuonAssociator && hasOnlyTrackAssociatorInRef) {
     if (myName.Contains("TrackAssociation")) myName.ReplaceAll("TrackAssociation","MuonAssociation");
     else myName.ReplaceAll("Association","MuonAssociation");
   }
   while (considerOnlyMuonAssociator && !(myName.Contains("tpToTkmu")) && !(myName.Contains("MuonAssociation")) ) {
     myKey1 = (TKey*)iter_r();
     myName = myKey1->GetName();
   }
   collname1 = myName;
   // ==> extractedGlobalMuons are in a different position wrt globalMuons:
   if (myNext2) {
     myKey2 = myNext2;
     myNext2 = 0;
   }
   else {
     myKey2 = (TKey*)iter_s();
   }
   if (!myKey2) continue;
   TString myName2 = myKey2->GetName();
   if (myName2.Contains("extractedGlobalMuons") && !myName.Contains("lobalMuons")) {
     myNext2 =  (TKey*)iter_s();
     TKey* myTemp = myKey2; 
     myKey2 = myNext2;
     myName2 = myKey2->GetName();
     myNext2 = myTemp;
   }
   // ==> end(extractedGlobalMuons) -> Only consider them in the "signal" samples
   if (!(myName2.Contains("tpToTkmu")) && considerOnlyMuonAssociator && hasOnlyTrackAssociatorInSig) {
     if (myName2.Contains("TrackAssociation")) myName2.ReplaceAll("TrackAssociation","MuonAssociation");
     else myName2.ReplaceAll("Association","MuonAssociation");
   }
   while (considerOnlyMuonAssociator && !(myName2.Contains("tpToTkmu")) && !(myName2.Contains("MuonAssociation")) ) {
     myKey2 = (TKey*)iter_s();
     if (!myKey2) continue;
     myName2 = myKey2->GetName();
   }
   collname2 = myName2;

   cout << " Comparing " << collname1 << " and " << collname2 << endl;
   if ( 
       (myName == myName2) || (myName+"FS" == myName2) || (myName == myName2+"FS" )
       || (myName.Contains("extractedGlobalMuons") && myName2.Contains("globalMuons") )
       || (myName.Contains("globalMuons") && myName2.Contains("extractedGlobalMuons") )
       ) {
     collname1 = myKey1->GetName();
     collname2 = myKey2->GetName();
   }
   else if ( (collname1 != collname2) && (collname1+"FS" != collname2) && (collname1 != collname2+"FS") ) {
     bool goodAsWell = false;
     if (collname1.BeginsWith("standAloneMuons_UpdatedAtVtx") && collname2.BeginsWith("standAloneMuons_UpdatedAtVtx")) {
       if (collname1.Contains("MuonAssociation")==collname2.Contains("MuonAssociation"));
       goodAsWell = true;
     }
     if (collname1.BeginsWith("hltL2Muons_UpdatedAtVtx") && collname2.BeginsWith("hltL2Muons_UpdatedAtVtx")) {
       if (collname1.Contains("MuonAssociation")==collname2.Contains("MuonAssociation"));
       goodAsWell = true;
     }
     if (collname1.BeginsWith("hltL3TkFromL2") && collname2.BeginsWith("hltL3TkFromL2")) {
       if (collname1.Contains("MuonAssociation")==collname2.Contains("MuonAssociation"));
       goodAsWell = true;
     }
     //     TString isGood = (goodAsWell? "good": "NOT good");
     //     cout << " -- The two collections: " << collname1 << " : " << collname2 << " -> " << isGood << endl;
     if (! goodAsWell) {

       if (collname1.Contains("SET") && !collname2.Contains("SET")) {
	 while (collname1.Contains("SET")) {
	   if (myKey1 = (TKey*)iter_r())  collname1 = myKey1->GetName();
	 }
       }
       else if (collname2.Contains("SET") && !collname1.Contains("SET")) {
	 while (collname2.Contains("SET")) {
	   if (myKey2 = (TKey*)iter_s())  collname2 = myKey2->GetName();
	 }
       }

       if (collname1.Contains("dyt") && !collname2.Contains("dyt")) {
	 while (collname1.Contains("dyt")) {
	   if (myKey1 = (TKey*)iter_r())  collname1 = myKey1->GetName();
	 }
       }
       else if (collname2.Contains("dyt") && !collname1.Contains("dyt")) {
	 while (collname2.Contains("dyt")) {
	   if (myKey2 = (TKey*)iter_s())  collname2 = myKey2->GetName();
	 }
       }

       if (collname1.Contains("refitted") && !collname2.Contains("refitted")) {
	 while (collname1.Contains("refitted")) {
	   if (myKey1 = (TKey*)iter_r())  collname1 = myKey1->GetName();
	 }
       }
       else if (collname2.Contains("refitted") && !collname1.Contains("refitted")) {
	 while (collname2.Contains("refitted")) {
	   if (myKey2 = (TKey*)iter_s())  collname2 = myKey2->GetName();
	 }
       }

       if ( (collname1 != collname2) && (collname1+"FS" != collname2) && (collname1 != collname2+"FS") ) {
	 cout << " Different collection names, please check: " << collname1 << " : " << collname2 << endl;
	 continue;
       }
       else {
	 //	 cout << "    The NEW collections: " << collname1 << " : " << collname2 << endl;
	 myName = myKey1->GetName();
       }
     }
   }

   TString newDir("NEW_RELEASE/NEWSELECTION/NEW_LABEL/");
   newDir+=myName;
   gSystem->mkdir(newDir,kTRUE);
 
   rh1 = 0; rh2 = 0; rh3 = 0; rh4 = 0; rh5 = 0; rh6 = 0;
   sh1 = 0; sh2 = 0; sh3 = 0; sh4 = 0; sh5 = 0; sh6 = 0;
 
 //////////////////////////////////////
 /////////// CTF //////////////////////
 //////////////////////////////////////
 if (ctf){
   //===== building
   rdir->GetObject(collname1+"/effic",rh1);
   sdir->GetObject(collname2+"/effic",sh1);
   if(! rh1 && sh1) continue;
   rh1->GetYaxis()->SetRangeUser(0.5,1.0125);
   sh1->GetYaxis()->SetRangeUser(0.5,1.0125);
   rh1->GetYaxis()->SetTitle("efficiency vs #eta");
   rdir->GetObject(collname1+"/fakerate",rh2);
   sdir->GetObject(collname2+"/fakerate",sh2);
   rh2->GetYaxis()->SetTitle("fakerate vs #eta");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);
   //   rh2->GetYaxis()->SetRangeUser(0.,.70);
   //   sh2->GetYaxis()->SetRangeUser(0.,.70);

   rdir->GetObject(collname1+"/efficPt",rh3);
   sdir->GetObject(collname2+"/efficPt",sh3);
   rh3->GetXaxis()->SetRangeUser(0,maxPT);
   sh3->GetXaxis()->SetRangeUser(0,maxPT);
   rh3->GetYaxis()->SetRangeUser(0.,1.025);
   sh3->GetYaxis()->SetRangeUser(0.,1.025);
   rh3->GetYaxis()->SetTitle("efficiency vs p_{t}");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);
   rh3->SetTitle("");

   rdir->GetObject(collname1+"/fakeratePt",rh4);
   sdir->GetObject(collname2+"/fakeratePt",sh4);
   rh4->SetTitle("");
   rh4->GetYaxis()->SetTitle("fakerate vs p_{t}");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);
   //   rh4->GetYaxis()->SetRangeUser(0.,.80);
   //   sh4->GetYaxis()->SetRangeUser(0.,.80);
   rh4->GetXaxis()->SetRangeUser(0.2,maxPT);
   sh4->GetXaxis()->SetRangeUser(0.2,maxPT);

   rdir->GetObject(collname1+"/effic_vs_phi",rh5);
   sdir->GetObject(collname2+"/effic_vs_phi",sh5);
   rh5->GetYaxis()->SetTitle("efficiency vs #phi");
   rh5->GetYaxis()->SetTitleSize(0.05);
   rh5->GetYaxis()->SetTitleOffset(1.2);
   rh5->GetYaxis()->SetRangeUser(0.5,1.0125);
   sh5->GetYaxis()->SetRangeUser(0.5,1.0125);
   rdir->GetObject(collname1+"/fakerate_vs_phi",rh6);
   sdir->GetObject(collname2+"/fakerate_vs_phi",sh6);
   rh6->GetYaxis()->SetTitle("fakerate vs #phi");
   rh6->GetYaxis()->SetTitleSize(0.05);
   rh6->GetYaxis()->SetTitleOffset(1.2);


   canvas = new TCanvas("Tracks1","Tracks: efficiency & fakerate",1000,1400);


   //NormalizeHistograms(rh2,sh2);
   //NormalizeHistograms(rh6,sh6);
   //rh1->GetYaxis()->SetRangeUser(8,24);
   //sh1->GetYaxis()->SetRangeUser(8,24);

   //rh6->GetXaxis()->SetRangeUser(0,10);
   //sh6->GetXaxis()->SetRangeUser(0,10);


   plotBuilding(canvas,
		sh1,rh1,sh2,rh2,
		sh3,rh3,sh4,rh4,
		sh5,rh5,sh6,rh6,
		te,"UU",-1);

   canvas->cd();
   //TPaveText* text = new TPaveText(0.25,0.72,0.75,0.77,"prova");
   //text->SetFillColor(0);
   //text->SetTextColor(1);
   //text->Draw();
   l = new TLegend(0.10,0.655,0.90,0.69);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print(newDir+"/building.pdf");   
   delete l;
   delete canvas;

   // ====== hits and pt

   rh1 = 0; rh2 = 0; rh3 = 0; rh4 = 0; rh5 = 0; rh6 = 0;
   sh1 = 0; sh2 = 0; sh3 = 0; sh4 = 0; sh5 = 0; sh6 = 0;

   rdir->GetObject(collname1+"/effic_vs_hit",rh1);
   if (!rh1) rdir->GetObject(collname1+"/effic_vs_hit",(TProfile*)rh1);
   sdir->GetObject(collname2+"/effic_vs_hit",sh1);
   if (!sh1) sdir->GetObject(collname2+"/effic_vs_hit",(TProfile*)sh1);
   //rh1->GetXaxis()->SetRangeUser(0,30);
   //sh1->GetXaxis()->SetRangeUser(0,30);
   rh1->GetYaxis()->SetTitle("efficiency vs hit");
   rh1->GetYaxis()->SetRangeUser(0.,1.025);
   sh1->GetYaxis()->SetRangeUser(0.,1.025);
   rdir->GetObject(collname1+"/fakerate_vs_hit",rh2);
   if (!rh2) rdir->GetObject(collname1+"/fakerate_vs_hit",(TProfile*)rh2);
   sdir->GetObject(collname2+"/fakerate_vs_hit",sh2);
   if (!sh2) sdir->GetObject(collname2+"/fakerate_vs_hit",(TProfile*)sh2);
   rh2->GetYaxis()->SetTitle("fakerate vs hit");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);

   rdir->GetObject(collname1+"/nhits_vs_eta_pfx",(TProfile*)rh3);
   if (!rh3) {
     TH2F* h2tmp;
     rdir->GetObject(collname1+"/nhits_vs_eta",h2tmp);
     rh3 = (TH1F*) h2tmp->ProfileX();
   }
   sdir->GetObject(collname2+"/nhits_vs_eta_pfx",(TProfile*)sh3);
   if (!sh3) {
     TH2F* h2tmp;
     sdir->GetObject(collname2+"/nhits_vs_eta",h2tmp);
     sh3 = (TH1F*) h2tmp->ProfileX();
   }
   rdir->GetObject(collname1+"/hits",rh4);
   sdir->GetObject(collname2+"/hits",sh4);
   
   rdir->GetObject(collname1+"/num_simul_pT",rh5);
   sdir->GetObject(collname2+"/num_simul_pT",sh5);
   rdir->GetObject(collname1+"/num_reco_pT",rh6);
   sdir->GetObject(collname2+"/num_reco_pT",sh6);
   
   rh3->GetYaxis()->SetRangeUser(0,74);
   sh3->GetYaxis()->SetRangeUser(0,74);
   rh4->GetXaxis()->SetRangeUser(0,74);
   sh4->GetXaxis()->SetRangeUser(0,74);
   
   rh5->GetXaxis()->SetRangeUser(0,maxPT);
   sh5->GetXaxis()->SetRangeUser(0,maxPT);
   rh6->GetXaxis()->SetRangeUser(0,maxPT);
   sh6->GetXaxis()->SetRangeUser(0,maxPT);
   NormalizeHistograms(rh4,sh4);
   NormalizeHistograms(rh5,sh5);
   NormalizeHistograms(rh6,sh6);
   
   canvas = new TCanvas("Tracks2","Tracks: efficiency & fakerate",1000,1400);
   
   plot6histos(canvas,
	      sh1,rh1,sh2,rh2,
	      sh3,rh3,sh4,rh4,
	      sh5,rh5,sh6,rh6,
	      te,"UU",-1);
   
   canvas->cd();
   
   l = new TLegend(0.10,0.655,0.90,0.69);
   //   l = new TLegend(0.10,0.64,0.90,0.69);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print(newDir+"/hitsAndPt.pdf");
   delete l;
   delete canvas;


   //===== tuning
   rh1 = 0; rh2 = 0; rh3 = 0; rh4 = 0; rh5 = 0; rh6 = 0;
   sh1 = 0; sh2 = 0; sh3 = 0; sh4 = 0; sh5 = 0; sh6 = 0;

   rdir->GetObject(collname1+"/chi2",rh1);
   sdir->GetObject(collname2+"/chi2",sh1);
   rdir->GetObject(collname1+"/chi2_prob",rh2);
   sdir->GetObject(collname2+"/chi2_prob",sh2);
   rdir->GetObject(collname1+"/chi2_vs_eta_pfx",(TProfile*)rh3);
   if (!rh3) {
     TH2F* h2tmp;
     rdir->GetObject(collname1+"/chi2_vs_eta",h2tmp);
     rh3 = (TH1F*) h2tmp->ProfileX();
   }
   sdir->GetObject(collname2+"/chi2_vs_eta_pfx",(TProfile*)sh3);
   if (!sh3) {
     TH2F* h2tmp;
     sdir->GetObject(collname2+"/chi2_vs_eta",h2tmp);
     sh3 = (TH1F*) h2tmp->ProfileX();
   }

   rdir->GetObject(collname1+"/ptres_vs_eta_Mean",rh4);
   if (!rh4) rdir->GetObject(collname1+"/ptres_vs_eta_Mean",(TProfile*)rh4);
   sdir->GetObject(collname2+"/ptres_vs_eta_Mean",sh4);
   if (!sh4) sdir->GetObject(collname2+"/ptres_vs_eta_Mean",(TProfile*)sh4);


   NormalizeHistograms(rh1,sh1);
   NormalizeHistograms(rh2,sh2);
   fixRangeY(rh1,sh1);
   fixRangeY(rh2,sh2);
   rh3->GetYaxis()->SetRangeUser(0,5.5);
   sh3->GetYaxis()->SetRangeUser(0,5.5);
 
   rh3->SetTitle("");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);
   rh3->GetYaxis()->SetTitle("<  #chi^{2} / ndf >");
   rh3->GetXaxis()->SetTitleSize(0.07);
   rh3->GetXaxis()->SetTitleOffset(0.6);
   rh3->GetXaxis()->SetTitle("#eta");

   rh4->Scale(100.);
   sh4->Scale(100.);
   rh4->GetYaxis()->SetRangeUser(-1.5,1.5);
   sh4->GetYaxis()->SetRangeUser(-1.5,1.5);
   rh4->SetTitle("");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);
   rh4->GetYaxis()->SetTitle("< #delta p_{t} / p_{t} > [%]");
   rh4->GetXaxis()->SetTitleSize(0.07);
   rh4->GetXaxis()->SetTitleOffset(0.6);
   rh4->GetXaxis()->SetTitle("#eta");

     
   canvas = new TCanvas("Tracks3","Tracks: chi2 & chi2 probability",1000,1050);

   plot4histos(canvas,
	       sh1,rh1,sh2,rh2,
	       sh3,rh3,sh4,rh4,    
	       te,"UU",-1);
   
   canvas->cd();   
   l = new TLegend(0.20,0.48,0.90,0.53);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print(newDir+"/tuning.pdf");
   delete l;
   delete canvas;
   

   //===== pulls
   rdir->GetObject(collname1+"/pullPt",rh1);
   sdir->GetObject(collname2+"/pullPt",sh1);

   rdir->GetObject(collname1+"/pullQoverp",rh2);
   sdir->GetObject(collname2+"/pullQoverp",sh2);

   rdir->GetObject(collname1+"/pullPhi",rh3);
   sdir->GetObject(collname2+"/pullPhi",sh3);

   rdir->GetObject(collname1+"/pullTheta",rh4);
   sdir->GetObject(collname2+"/pullTheta",sh4);

   rdir->GetObject(collname1+"/pullDxy",rh5);
   sdir->GetObject(collname2+"/pullDxy",sh5);

   rdir->GetObject(collname1+"/pullDz",rh6);
   sdir->GetObject(collname2+"/pullDz",sh6);


   fixRangeY(rh1,sh1);
   fixRangeY(rh2,sh2);
   fixRangeY(rh3,sh3);
   fixRangeY(rh4,sh4);
   fixRangeY(rh5,sh5);
   fixRangeY(rh6,sh6);
   NormalizeHistograms(rh1,sh1);
   NormalizeHistograms(rh2,sh2);
   NormalizeHistograms(rh3,sh3);
   NormalizeHistograms(rh4,sh4);
   NormalizeHistograms(rh5,sh5);
   NormalizeHistograms(rh6,sh6);

   rh1->GetXaxis()->SetRangeUser(-10,10);
   sh1->GetXaxis()->SetRangeUser(-10,10);
   rh2->GetXaxis()->SetRangeUser(-10,10);
   sh2->GetXaxis()->SetRangeUser(-10,10);
   rh3->GetXaxis()->SetRangeUser(-10,10);
   sh3->GetXaxis()->SetRangeUser(-10,10);
   rh4->GetXaxis()->SetRangeUser(-10,10);
   sh4->GetXaxis()->SetRangeUser(-10,10);
   rh5->GetXaxis()->SetRangeUser(-10,10);
   sh5->GetXaxis()->SetRangeUser(-10,10);
   rh6->GetXaxis()->SetRangeUser(-10,10);
   sh6->GetXaxis()->SetRangeUser(-10,10);


   canvas = new TCanvas("Tracks4","Tracks: pull of Pt, Qoverp and Phi",1000,1400);

   plotPulls(canvas,
	     sh1,rh1,sh2,rh2,
	     sh3,rh3,sh4,rh4,
	     sh5,rh5,sh6,rh6,
	     te,"UU",-1);

   canvas->cd();

   l = new TLegend(0.10,0.655,0.90,0.69);
   //   l = new TLegend(0.20,0.655,0.80,0.69);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print(newDir+"/pulls.pdf");
   delete l;
   delete canvas;



   //===== residuals

   TH2F *rtemp, *stemp;
   TH1F *rproj, *sproj;

   rdir->GetObject(collname1+"/ptres_vs_eta",rtemp);
   sdir->GetObject(collname2+"/ptres_vs_eta",stemp);
   rproj = (TH1F*) rtemp->ProjectionY();
   TH1F *rhres1 = new TH1F(*rproj);
   sproj = (TH1F*) stemp->ProjectionY();
   TH1F *shres1 = new TH1F(*sproj);

   rdir->GetObject(collname1+"/etares_vs_eta",rtemp);
   sdir->GetObject(collname2+"/etares_vs_eta",stemp);
   rproj = (TH1F*) rtemp->ProjectionY();
   TH1F *rhres2 = new TH1F(*rproj);
   sproj = (TH1F*) stemp->ProjectionY();
   TH1F* shres2 = new TH1F(*sproj);

   rdir->GetObject(collname1+"/phires_vs_eta",rtemp);
   sdir->GetObject(collname2+"/phires_vs_eta",stemp);
   rproj = (TH1F*) rtemp->ProjectionY();
   TH1F *rhres3 = new TH1F(*rproj);
   sproj = (TH1F*) stemp->ProjectionY();
   TH1F* shres3 = new TH1F(*sproj);

   rdir->GetObject(collname1+"/cotThetares_vs_eta",rtemp);
   sdir->GetObject(collname2+"/cotThetares_vs_eta",stemp);
   rproj = (TH1F*) rtemp->ProjectionY();
   TH1F *rhres4 = new TH1F(*rproj);
   sproj = (TH1F*) stemp->ProjectionY();
   TH1F* shres4 = new TH1F(*sproj);

   rdir->GetObject(collname1+"/dxyres_vs_eta",rtemp);
   sdir->GetObject(collname2+"/dxyres_vs_eta",stemp);
   rproj = (TH1F*) rtemp->ProjectionY();
   TH1F *rhres5 = new TH1F(*rproj);
   sproj = (TH1F*) stemp->ProjectionY();
   TH1F* shres5 = new TH1F(*sproj);

   rdir->GetObject(collname1+"/dzres_vs_eta",rtemp);
   sdir->GetObject(collname2+"/dzres_vs_eta",stemp);
   rproj = (TH1F*) rtemp->ProjectionY();
   TH1F *rhres6 = new TH1F(*rproj);
   sproj = (TH1F*) stemp->ProjectionY();
   TH1F* shres6 = new TH1F(*sproj);

   NormalizeHistograms(rhres1,shres1);
   NormalizeHistograms(rhres2,shres2);
   NormalizeHistograms(rhres3,shres3);
   NormalizeHistograms(rhres4,shres4);
   NormalizeHistograms(rhres5,shres5);
   NormalizeHistograms(rhres6,shres6);
   fixRangeY(rhres1,shres1);
   fixRangeY(rhres2,shres2);
   fixRangeY(rhres3,shres3);
   fixRangeY(rhres4,shres4);
   fixRangeY(rhres5,shres5);
   fixRangeY(rhres6,shres6);

   rhres1->SetTitle("p_{t} resolution"); 
   //   rh1->GetXaxis()->SetTitleSize(0.07);
//   rh1->GetXaxis()->SetTitleOffset(0.6);
//   rh1->GetXaxis()->SetTitle("(p_{t}(rec)-p_{t}(sim))/p_{t}(sim)");
   shres1->SetTitle("p_{t} resolution"); 
//   sh1->GetXaxis()->SetTitleSize(0.07);
//   sh1->GetXaxis()->SetTitleOffset(0.6);
//   sh1->GetXaxis()->SetTitle("(p_{t}(rec)-p_{t}(sim))/p_{t}(sim)");
   rhres2->SetTitle("#eta resolution"); 
//   rh2->GetXaxis()->SetTitleSize(0.07);
//   rh2->GetXaxis()->SetTitleOffset(0.6);
//   rh2->GetXaxis()->SetTitle("#eta(rec)-#eta(sim)");
   shres2->SetTitle("#eta resolution"); 
//   sh2->GetXaxis()->SetTitleSize(0.07);
//   sh2->GetXaxis()->SetTitleOffset(0.6);
//   sh2->GetXaxis()->SetTitle("#eta(rec)-#eta(sim)");
   rhres3->SetTitle("#phi resolution"); 
//   rh3->GetXaxis()->SetTitleSize(0.07);
//   rh3->GetXaxis()->SetTitleOffset(0.6);
//   rh3->GetXaxis()->SetTitle("#phi(rec)-#phi(sim)");
   shres3->SetTitle("#phi resolution"); 
//   sh3->GetXaxis()->SetTitleSize(0.07);
//   sh3->GetXaxis()->SetTitleOffset(0.6);
//   sh3->GetXaxis()->SetTitle("#phi(rec)-#phi(sim)");
   rhres4->SetTitle("cot(#Theta) resolution"); 
//   rh4->GetXaxis()->SetTitleSize(0.07);
//   rh4->GetXaxis()->SetTitleOffset(0.6);
//   rh4->GetXaxis()->SetTitle("cotTheta(rec)-cotTheta(sim)");
   shres4->SetTitle("cot(#Theta) resolution"); 
//   sh4->GetXaxis()->SetTitleSize(0.07);
//   sh4->GetXaxis()->SetTitleOffset(0.6);
//   sh4->GetXaxis()->SetTitle("cotTheta(rec)-cotTheta(sim)");
   rhres5->SetTitle("dxy resolution"); 
//   rh5->GetXaxis()->SetTitleSize(0.07);
//   rh5->GetXaxis()->SetTitleOffset(0.6);
//   rh5->GetXaxis()->SetTitle("dxy(rec)-dxy(sim)");
   shres5->SetTitle("dxy resolution"); 
//   sh5->GetXaxis()->SetTitleSize(0.07);
//   sh5->GetXaxis()->SetTitleOffset(0.6);
//   sh5->GetXaxis()->SetTitle("dxy(rec)-dxy(sim)");
   rhres6->SetTitle("dz resolution"); 
//   rh6->GetXaxis()->SetTitleSize(0.07);
//   rh6->GetXaxis()->SetTitleOffset(0.6);
//   rh6->GetXaxis()->SetTitle("dz(rec)-dz(sim)");
   shres6->SetTitle("dz resolution"); 
//   sh6->GetXaxis()->SetTitleSize(0.07);
//   sh6->GetXaxis()->SetTitleOffset(0.6);
//   sh6->GetXaxis()->SetTitle("dz(rec)-dz(sim)");

   canvas = new TCanvas("Tracks5","Track residuals",1000,1400);

   plotPulls(canvas,
	     shres1,rhres1,shres2,rhres2,
	     shres3,rhres3,shres4,rhres4,
	     shres5,rhres5,shres6,rhres6,
	     te,"UU",-1);

   canvas->cd();

   l = new TLegend(0.10,0.655,0.90,0.69);
   //   l = new TLegend(0.20,0.655,0.80,0.69);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print(newDir+"/residuals.pdf");
   delete l;
   delete canvas;

   delete rhres1;
   delete shres1;
   delete rhres2;
   delete shres2;
   delete rhres3;
   delete shres3;
   delete rhres4;
   delete shres4;
   delete rhres5;
   delete shres5;
   delete rhres6;
   delete shres6;

   

   //===== resolutions vs eta
   rh1 = 0; rh2 = 0; rh3 = 0; rh4 = 0; rh5 = 0; rh6 = 0;
   sh1 = 0; sh2 = 0; sh3 = 0; sh4 = 0; sh5 = 0; sh6 = 0;

   rdir->GetObject(collname1+"/phires_vs_eta_Sigma",rh1);
   if (!rh1) rdir->GetObject(collname1+"/phires_vs_eta_Sigma",(TProfile*)rh1);
   sdir->GetObject(collname2+"/phires_vs_eta_Sigma",sh1);
   if (!sh1) sdir->GetObject(collname2+"/phires_vs_eta_Sigma",(TProfile*)sh1);

   rdir->GetObject(collname1+"/cotThetares_vs_eta_Sigma",rh2);
   if (!rh2) rdir->GetObject(collname1+"/cotThetares_vs_eta_Sigma",(TProfile*)rh2);
   sdir->GetObject(collname2+"/cotThetares_vs_eta_Sigma",sh2);
   if (!sh2) sdir->GetObject(collname2+"/cotThetares_vs_eta_Sigma",(TProfile*)sh2);

   rdir->GetObject(collname1+"/dxyres_vs_eta_Sigma",rh3);
   if (!rh3) rdir->GetObject(collname1+"/dxyres_vs_eta_Sigma",(TProfile*)rh3);
   sdir->GetObject(collname2+"/dxyres_vs_eta_Sigma",sh3);
   if (!sh3) sdir->GetObject(collname2+"/dxyres_vs_eta_Sigma",(TProfile*)sh3);

   rdir->GetObject(collname1+"/dzres_vs_eta_Sigma",rh4);
   if (!rh4) rdir->GetObject(collname1+"/dzres_vs_eta_Sigma",(TProfile*)rh4);
   sdir->GetObject(collname2+"/dzres_vs_eta_Sigma",sh4);
   if (!sh4) sdir->GetObject(collname2+"/dzres_vs_eta_Sigma",(TProfile*)sh4);

   rdir->GetObject(collname1+"/ptres_vs_eta_Sigma",rh5);
   if (!rh5) rdir->GetObject(collname1+"/ptres_vs_eta_Sigma",(TProfile*)rh5);
   sdir->GetObject(collname2+"/ptres_vs_eta_Sigma",sh5);
   if (!sh5) sdir->GetObject(collname2+"/ptres_vs_eta_Sigma",(TProfile*)sh5);



   canvas = new TCanvas("Tracks6","Tracks: Dxy, Dz, Theta resolution",1000,1400);

   plotResolutions(canvas,
		   sh1,rh1,sh2,rh2,
		   sh3,rh3,sh4,rh4,
		   sh5,rh5,sh6,rh6,
		   te,"UU",-1);
   
   // new general range
   //rh1->GetYaxis()->SetRangeUser(0.000009,0.01);
   //sh1->GetYaxis()->SetRangeUser(0.000009,0.01);
   // for multi-track samples
   //rh1->GetYaxis()->SetRangeUser(0.0008,0.005);
   //sh1->GetYaxis()->SetRangeUser(0.0008,0.005);
   // for single particle pt 100
   //rh1->GetYaxis()->SetRangeUser(0.000009,0.0005);
   //sh1->GetYaxis()->SetRangeUser(0.000009,0.0005);
   // for single particle pt 10
   //rh1->GetYaxis()->SetRangeUser(0.00009,0.001);  
   //sh1->GetYaxis()->SetRangeUser(0.00009,0.001); 
   // for single particle pt 1
   //rh1->GetYaxis()->SetRangeUser(0.0008,0.005);
   //sh1->GetYaxis()->SetRangeUser(0.0008,0.005);
   rh1->SetTitle(""); 
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);
   //   rh1->GetYaxis()->SetTitleColor(2);
   rh1->GetYaxis()->SetTitle("#sigma(#delta #phi) [rad]");
   rh1->GetXaxis()->SetTitleSize(0.07);
   rh1->GetXaxis()->SetTitleOffset(0.6);
   rh1->GetXaxis()->SetTitle("#eta");


   // new general range
   //rh2->GetYaxis()->SetRangeUser(0.00009,0.03);
   //sh2->GetYaxis()->SetRangeUser(0.00009,0.03);
   // for multi-track samples
   //rh2->GetYaxis()->SetRangeUser(0.0009,0.01);
   //sh2->GetYaxis()->SetRangeUser(0.0009,0.01);
   // for single particle pt 10
   //rh2->GetYaxis()->SetRangeUser(0.00009,0.01);
   //sh2->GetYaxis()->SetRangeUser(0.00009,0.01);
   // for single particle pt 1
   //rh2->GetYaxis()->SetRangeUser(0.0009,0.01);
   //sh2->GetYaxis()->SetRangeUser(0.0009,0.01);
   rh2->SetTitle("");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);
   rh2->GetYaxis()->SetTitle("#sigma(#delta cot(#theta)) ");
   rh2->GetXaxis()->SetTitleSize(0.07);
   rh2->GetXaxis()->SetTitleOffset(0.6);
   rh2->GetXaxis()->SetTitle("#eta");


   // new general range
   //rh3->GetYaxis()->SetRangeUser(0.00009,0.05);  
   //sh3->GetYaxis()->SetRangeUser(0.00009,0.05);
   // for multi-track samples
   //rh3->GetYaxis()->SetRangeUser(0.0009,0.02);  
   //sh3->GetYaxis()->SetRangeUser(0.0009,0.02);
   //rh3->GetYaxis()->SetRangeUser(0.0009,0.02);
   //sh3->GetYaxis()->SetRangeUser(0.0009,0.02);
   // for single particle pt 100    
   //rh3->GetYaxis()->SetRangeUser(0.00009,0.002);
   //sh3->GetYaxis()->SetRangeUser(0.00009,0.002);
   rh3->SetTitle("");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);
   rh3->GetYaxis()->SetTitle("#sigma(#delta d_{0}) [cm]");
   rh3->GetXaxis()->SetTitleSize(0.07);
   rh3->GetXaxis()->SetTitleOffset(0.6);
   rh3->GetXaxis()->SetTitle("#eta"); 


   // new general range
   //rh4->GetYaxis()->SetRangeUser(0.0009,0.1);  
   //sh4->GetYaxis()->SetRangeUser(0.0009,0.1);
   // for multi-track samples
   //rh4->GetYaxis()->SetRangeUser(0.0009,0.08);
   //sh4->GetYaxis()->SetRangeUser(0.0009,0.08);

   rh4->SetTitle("");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);
   rh4->GetYaxis()->SetTitle("#sigma(#delta z_{0}) [cm]");
   rh4->GetXaxis()->SetTitleSize(0.07);
   rh4->GetXaxis()->SetTitleOffset(0.6);
   rh4->GetXaxis()->SetTitle("#eta");

   rh5->SetTitle("");
   rh5->GetYaxis()->SetTitleSize(0.05);
   rh5->GetYaxis()->SetTitleOffset(1.2);
   rh5->GetYaxis()->SetTitle("#sigma(#delta p_{t}/p_{t}) ");
   rh5->GetXaxis()->SetTitleSize(0.07);
   rh5->GetXaxis()->SetTitleOffset(0.6);
   rh5->GetXaxis()->SetTitle("#eta");

   canvas->cd();

   l = new TLegend(0.10,0.655,0.90,0.69);
   //   l = new TLegend(0.10,0.63,0.90,0.67);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print(newDir+"/resolutionsEta.pdf");
   delete l;
   delete canvas;

   //
   //===== mean values vs eta
   //
   rh1 = 0; rh2 = 0; rh3 = 0; rh4 = 0; rh5 = 0; rh6 = 0;
   sh1 = 0; sh2 = 0; sh3 = 0; sh4 = 0; sh5 = 0; sh6 = 0;

   rdir->GetObject(collname1+"/phires_vs_eta_Mean",rh1);
   if (!rh1) rdir->GetObject(collname1+"/phires_vs_eta_Mean",(TProfile*)rh1);
   sdir->GetObject(collname2+"/phires_vs_eta_Mean",sh1);
   if (!sh1) sdir->GetObject(collname2+"/phires_vs_eta_Mean",(TProfile*)sh1);

   rdir->GetObject(collname1+"/cotThetares_vs_eta_Mean",rh2);
   if (!rh2) rdir->GetObject(collname1+"/cotThetares_vs_eta_Mean",(TProfile*)rh2);
   sdir->GetObject(collname2+"/cotThetares_vs_eta_Mean",sh2);
   if (!sh2) sdir->GetObject(collname2+"/cotThetares_vs_eta_Mean",(TProfile*)sh2);

   rdir->GetObject(collname1+"/dxyres_vs_eta_Mean",rh3);
   if (!rh3) rdir->GetObject(collname1+"/dxyres_vs_eta_Mean",(TProfile*)rh3);
   sdir->GetObject(collname2+"/dxyres_vs_eta_Mean",sh3);
   if (!sh3) sdir->GetObject(collname2+"/dxyres_vs_eta_Mean",(TProfile*)sh3);

   rdir->GetObject(collname1+"/dzres_vs_eta_Mean",rh4);
   if (!rh4) rdir->GetObject(collname1+"/dzres_vs_eta_Mean",(TProfile*)rh4);
   sdir->GetObject(collname2+"/dzres_vs_eta_Mean",sh4);
   if (!sh4) sdir->GetObject(collname2+"/dzres_vs_eta_Mean",(TProfile*)sh4);

   rdir->GetObject(collname1+"/ptres_vs_eta_Mean",rh5);
   if (!rh5) rdir->GetObject(collname1+"/ptres_vs_eta_Mean",(TProfile*)rh5);
   sdir->GetObject(collname2+"/ptres_vs_eta_Mean",sh5);
   if (!sh5) sdir->GetObject(collname2+"/ptres_vs_eta_Mean",(TProfile*)sh5);



   canvas = new TCanvas("Tracks7","Tracks: Dxy, Dz, Theta mean values",1000,1400);

   plotMeanValues(canvas,
		   sh1,rh1,sh2,rh2,
		   sh3,rh3,sh4,rh4,
		   sh5,rh5,sh6,rh6,
		   te,"UU",-1);
   
   rh1->SetTitle(""); 
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);
   rh1->GetYaxis()->SetTitle("#delta #phi [rad]");
   rh1->GetXaxis()->SetTitleSize(0.07);
   rh1->GetXaxis()->SetTitleOffset(0.6);
   rh1->GetXaxis()->SetTitle("#eta");

   rh2->SetTitle("");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);
   rh2->GetYaxis()->SetTitle("#delta cot(#theta)");
   rh2->GetXaxis()->SetTitleSize(0.07);
   rh2->GetXaxis()->SetTitleOffset(0.6);
   rh2->GetXaxis()->SetTitle("#eta");

   rh3->SetTitle("");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);
   rh3->GetYaxis()->SetTitle("#delta d_{0} [cm]");
   rh3->GetXaxis()->SetTitleSize(0.07);
   rh3->GetXaxis()->SetTitleOffset(0.6);
   rh3->GetXaxis()->SetTitle("#eta"); 

   rh4->SetTitle("");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);
   rh4->GetYaxis()->SetTitle("#delta z_{0} [cm]");
   rh4->GetXaxis()->SetTitleSize(0.07);
   rh4->GetXaxis()->SetTitleOffset(0.6);
   rh4->GetXaxis()->SetTitle("#eta");

   rh5->SetTitle("");
   rh5->GetYaxis()->SetTitleSize(0.05);
   rh5->GetYaxis()->SetTitleOffset(1.2);
   rh5->GetYaxis()->SetTitle("#delta p_{t}/p_{t} ");
   rh5->GetXaxis()->SetTitleSize(0.07);
   rh5->GetXaxis()->SetTitleOffset(0.6);
   rh5->GetXaxis()->SetTitle("#eta");

   canvas->cd();

   l = new TLegend(0.10,0.655,0.90,0.69);
   //   l = new TLegend(0.10,0.63,0.90,0.67);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print(newDir+"/meanvaluesEta.pdf");
   delete l;
   delete canvas;


   //
   //===== resolutions vs pt
   //
   rh1 = 0; rh2 = 0; rh3 = 0; rh4 = 0; rh5 = 0; rh6 = 0;
   sh1 = 0; sh2 = 0; sh3 = 0; sh4 = 0; sh5 = 0; sh6 = 0;

   rdir->GetObject(collname1+"/phires_vs_pt_Sigma",rh1);
   if (!rh1) rdir->GetObject(collname1+"/phires_vs_pt_Sigma",(TProfile*)rh1);
   sdir->GetObject(collname2+"/phires_vs_pt_Sigma",sh1);
   if (!sh1) sdir->GetObject(collname2+"/phires_vs_pt_Sigma",(TProfile*)sh1);

   rdir->GetObject(collname1+"/cotThetares_vs_pt_Sigma",rh2);
   if (!rh2) rdir->GetObject(collname1+"/cotThetares_vs_pt_Sigma",(TProfile*)rh2);
   sdir->GetObject(collname2+"/cotThetares_vs_pt_Sigma",sh2);
   if (!sh2) sdir->GetObject(collname2+"/cotThetares_vs_pt_Sigma",(TProfile*)sh2);

   rdir->GetObject(collname1+"/dxyres_vs_pt_Sigma",rh3);
   if (!rh3) rdir->GetObject(collname1+"/dxyres_vs_pt_Sigma",(TProfile*)rh3);
   sdir->GetObject(collname2+"/dxyres_vs_pt_Sigma",sh3);
   if (!sh3) sdir->GetObject(collname2+"/dxyres_vs_pt_Sigma",(TProfile*)sh3);

   rdir->GetObject(collname1+"/dzres_vs_pt_Sigma",rh4);
   if (!rh4) rdir->GetObject(collname1+"/dzres_vs_pt_Sigma",(TProfile*)rh4);
   sdir->GetObject(collname2+"/dzres_vs_pt_Sigma",sh4);
   if (!sh4) sdir->GetObject(collname2+"/dzres_vs_pt_Sigma",(TProfile*)sh4);

   rdir->GetObject(collname1+"/ptres_vs_pt_Sigma",rh5);
   if (!rh5) rdir->GetObject(collname1+"/ptres_vs_pt_Sigma",(TProfile*)rh5);
   sdir->GetObject(collname2+"/ptres_vs_pt_Sigma",sh5);
   if (!sh5) sdir->GetObject(collname2+"/ptres_vs_pt_Sigma",(TProfile*)sh5);

   rh1->SetTitle("");
   rh1->GetYaxis()->SetTitleSize(0.05);
   rh1->GetYaxis()->SetTitleOffset(1.2);
   rh1->GetYaxis()->SetTitle("#sigma(#delta #phi) [rad]");
   rh1->GetXaxis()->SetTitleSize(0.055);
   rh1->GetXaxis()->SetTitleOffset(0.8);
   rh1->GetXaxis()->SetTitle("p_{t}");
   rh1->GetXaxis()->SetRangeUser(0,maxPT);
   sh1->GetXaxis()->SetRangeUser(0,maxPT);
 

   rh2->SetTitle("");
   rh2->GetYaxis()->SetTitleSize(0.05);
   rh2->GetYaxis()->SetTitleOffset(1.2);
   rh2->GetYaxis()->SetTitle("#sigma(#delta cot(#theta)) ");
   rh2->GetXaxis()->SetTitleSize(0.055);
   rh2->GetXaxis()->SetTitleOffset(0.8);
   rh2->GetXaxis()->SetTitle("p_{t}");
   rh2->GetXaxis()->SetRangeUser(0,maxPT);
   sh2->GetXaxis()->SetRangeUser(0,maxPT);

   rh3->SetTitle("");
   rh3->GetYaxis()->SetTitleSize(0.05);
   rh3->GetYaxis()->SetTitleOffset(1.2);
   rh3->GetYaxis()->SetTitle("#sigma(#delta d_{0}) [cm]");
   rh3->GetXaxis()->SetTitleSize(0.055);
   rh3->GetXaxis()->SetTitleOffset(0.8);
   rh3->GetXaxis()->SetTitle("p_{t}");
   rh3->GetXaxis()->SetRangeUser(0,maxPT);
   sh3->GetXaxis()->SetRangeUser(0,maxPT);


   rh4->SetTitle("");
   rh4->GetYaxis()->SetTitleSize(0.05);
   rh4->GetYaxis()->SetTitleOffset(1.2);
   rh4->GetYaxis()->SetTitle("#sigma(#delta z_{0}) [cm]");
   rh4->GetXaxis()->SetTitleSize(0.055);
   rh4->GetXaxis()->SetTitleOffset(0.8);
   rh4->GetXaxis()->SetTitle("p_{t}");
   rh4->GetXaxis()->SetRangeUser(0,maxPT);
   sh4->GetXaxis()->SetRangeUser(0,maxPT);


   rh5->SetTitle("");
   rh5->GetYaxis()->SetTitleSize(0.05);
   rh5->GetYaxis()->SetTitleOffset(1.2);
   rh5->GetYaxis()->SetTitle("#sigma(#delta p_{t}/p_{t}) ");
   rh5->GetXaxis()->SetTitleSize(0.055);
   rh5->GetXaxis()->SetTitleOffset(0.8);
   rh5->GetXaxis()->SetTitle("p_{t}");
   rh5->GetXaxis()->SetRangeUser(0,maxPT);
   sh5->GetXaxis()->SetRangeUser(0,maxPT);


   //   rh6->GetXaxis()->SetRangeUser(0,maxPT);
   //   sh6->GetXaxis()->SetRangeUser(0,maxPT);


   canvas = new TCanvas("Tracks8","Tracks: Dxy, Dz, Theta resolution",1000,1400);

   plotResolutions(canvas,
	     sh1,rh1,sh2,rh2,
	     sh3,rh3,sh4,rh4,
	     sh5,rh5,sh6,rh6,
	     te,"UU",-1);

   canvas->cd();

   l = new TLegend(0.10,0.655,0.90,0.69);
   //   l = new TLegend(0.10,0.63,0.90,0.67);
   l->SetTextSize(0.016);
   l->SetLineColor(1);
   l->SetLineWidth(1);
   l->SetLineStyle(1);
   l->SetFillColor(0);
   l->SetBorderSize(3);
   l->AddEntry(rh1,refLabel,"LPF");
   l->AddEntry(sh1,newLabel,"LPF");
   l->Draw();
   canvas->Print(newDir+"/resolutionsPt.pdf");
   delete l;
   delete canvas;


 }  // end of "if CTF"
 
 //// Merge pdf histograms together into larger files, and name them based on the collection names
 gSystem->Exec("gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=merged.pdf "
	       +newDir+"/building.pdf "
	       +newDir+"/hitsAndPt.pdf "
	       +newDir+"/tuning.pdf "
	       +newDir+"/pulls.pdf "
	       +newDir+"/residuals.pdf "
	       +newDir+"/resolutionsEta.pdf "
	       +newDir+"/meanvaluesEta.pdf "
	       +newDir+"/resolutionsPt.pdf ");
 // gSystem->Exec("cp merged.pdf "+newDir+"/../"+myName+".pdf");
 gSystem->Exec("mv merged.pdf "+newDir+"/../"+myName+".pdf");
 gSystem->Exec("rm -r "+newDir);
 
 }  // end of "while loop"
 
}


void NormalizeHistograms(TH1F* h1, TH1F* h2)
{
  if (h1==0 || h2==0) return;
  float scale1 = -9999.9;
  float scale2 = -9999.9;

  if ( h1->Integral() != 0 && h2->Integral() != 0 ){
      scale1 = 1.0/(float)h1->Integral();
      scale2 = 1.0/(float)h2->Integral();
    
      //h1->Sumw2();
      //h2->Sumw2();
      h1->Scale(scale1);
      h2->Scale(scale2);
    }
}



void plot4histos(TCanvas *canvas, 
		TH1F *s1,TH1F *r1, TH1F *s2,TH1F *r2, 
		TH1F *s3,TH1F *r3, TH1F *s4,TH1F *r4,
		TText* te,
	       char * option, double startingY, double startingX = .1,bool fit = false){
  canvas->Divide(2,2);

  s1->SetMarkerStyle(20);
  r1->SetMarkerStyle(21);
  s1->SetMarkerColor(2);
  r1->SetMarkerColor(4);
  s1->SetMarkerSize(0.7);
  r1->SetMarkerSize(0.7);
  s1->SetLineColor(2);
  r1->SetLineColor(4);
  s1->SetLineWidth(2);
  r1->SetLineWidth(2);

  s2->SetMarkerStyle(20);
  r2->SetMarkerStyle(21);
  s2->SetMarkerColor(2);
  r2->SetMarkerColor(4);
  s2->SetMarkerSize(0.1);
  r2->SetMarkerSize(0.1);
  s2->SetLineColor(2);
  r2->SetLineColor(4);
  s2->SetLineWidth(2);
  r2->SetLineWidth(2);

  s3->SetMarkerStyle(20);
  r3->SetMarkerStyle(21);
  s3->SetMarkerColor(2);
  r3->SetMarkerColor(4);
  s3->SetMarkerSize(0.7);
  r3->SetMarkerSize(0.7);
  s3->SetLineColor(2);
  r3->SetLineColor(4);
  r3->SetLineWidth(2);
  s3->SetLineWidth(2);

  s4->SetMarkerStyle(20);
  r4->SetMarkerStyle(21);
  s4->SetMarkerColor(2);
  r4->SetMarkerColor(4);
  s4->SetMarkerSize(0.7);
  r4->SetMarkerSize(0.7);
  s4->SetLineColor(2);
  r4->SetLineColor(4);
  r4->SetLineWidth(2);
  s4->SetLineWidth(2);


  //setStats(r1,s1, startingY, startingX, fit);
  canvas->cd(1);
  setStats(s1,r1, 0.6, 0.65, false);
  r1->Draw();
  s1->Draw("sames");

  canvas->cd(2);
  setStats(s2,r2, 0.6, 0.65, false);
  s2->Draw();
  r2->Draw("sames");

  canvas->cd(3);
  setStats(s3,r3, -1, 0, false);
  r3->Draw();
  s3->Draw("sames");

  canvas->cd(4);
  setStats(s4,r4, -1, 0, false);
  s4->Draw();
  r4->Draw("sames");

}

void plot6histos(TCanvas *canvas, 
		TH1F *s1,TH1F *r1, TH1F *s2,TH1F *r2, 
		TH1F *s3,TH1F *r3, TH1F *s4,TH1F *r4,
		TH1F *s5,TH1F *r5, TH1F *s6,TH1F *r6,
		TText* te,
	       char * option, double startingY, double startingX = .1,bool fit = false){
  canvas->Divide(2,3);

  s1->SetMarkerStyle(20);
  r1->SetMarkerStyle(21);
  s1->SetMarkerColor(2);
  r1->SetMarkerColor(4);
  s1->SetMarkerSize(0.7);
  r1->SetMarkerSize(0.7);
  s1->SetLineColor(2);
  r1->SetLineColor(4);
  s1->SetLineWidth(2);
  r1->SetLineWidth(2);

  s2->SetMarkerStyle(20);
  r2->SetMarkerStyle(21);
  s2->SetMarkerColor(2);
  r2->SetMarkerColor(4);
  s2->SetMarkerSize(0.1);
  r2->SetMarkerSize(0.1);
  s2->SetLineColor(2);
  r2->SetLineColor(4);
  s2->SetLineWidth(2);
  r2->SetLineWidth(2);

  s3->SetMarkerStyle(20);
  r3->SetMarkerStyle(21);
  s3->SetMarkerColor(2);
  r3->SetMarkerColor(4);
  s3->SetMarkerSize(0.7);
  r3->SetMarkerSize(0.7);
  s3->SetLineColor(2);
  r3->SetLineColor(4);
  r3->SetLineWidth(2);
  s3->SetLineWidth(2);

  s4->SetMarkerStyle(20);
  r4->SetMarkerStyle(21);
  s4->SetMarkerColor(2);
  r4->SetMarkerColor(4);
  s4->SetMarkerSize(0.7);
  r4->SetMarkerSize(0.7);
  s4->SetLineColor(2);
  r4->SetLineColor(4);
  r4->SetLineWidth(2);
  s4->SetLineWidth(2);

  s5->SetMarkerStyle(20);
  r5->SetMarkerStyle(21);
  s5->SetMarkerColor(2);
  r5->SetMarkerColor(4);
  s5->SetMarkerSize(0.7);
  r5->SetMarkerSize(0.7);
  s5->SetLineColor(2);
  r5->SetLineColor(4);
  r5->SetLineWidth(2);
  s5->SetLineWidth(2);

  s6->SetMarkerStyle(20);
  r6->SetMarkerStyle(21);
  s6->SetMarkerColor(2);
  r6->SetMarkerColor(4);
  s6->SetMarkerSize(0.7);
  r6->SetMarkerSize(0.7);
  s6->SetLineColor(2);
  r6->SetLineColor(4);
  r6->SetLineWidth(2);
  s6->SetLineWidth(2);


  //setStats(r1,s1, startingY, startingX, fit);
  canvas->cd(1);
  setStats(s1,r1, -1, 0, false);
  r1->Draw();
  s1->Draw("sames");

  canvas->cd(2);
  setStats(s2,r2, -1, 0, false);
  s2->Draw();
  r2->Draw("sames");

  canvas->cd(3);
  setStats(s3,r3, -1, 0, false);
  r3->Draw();
  s3->Draw("sames");

  canvas->cd(4);
  setStats(s4,r4, 0.6, 0.65, false);
  s4->Draw();
  r4->Draw("sames");

  canvas->cd(5);
  setStats(s5,r5, 0.6, 0.65, false);
  r5->Draw();
  s5->Draw("sames");

  canvas->cd(6);
  setStats(s6,r6, 0.6, 0.65, false);
  r6->Draw();
  s6->Draw("sames");

}

void plotBuilding(TCanvas *canvas, 
		  TH1F *s1,TH1F *r1, TH1F *s2,TH1F *r2, 
		  TH1F *s3,TH1F *r3, TH1F *s4,TH1F *r4,
		  TH1F *s5,TH1F *r5,TH1F *s6,TH1F *r6,
		  TText* te,
	       char * option, double startingY, double startingX = .1,bool fit = false){
  canvas->Divide(2,3);

  s1->SetMarkerStyle(20);
  r1->SetMarkerStyle(21);
  s1->SetMarkerColor(2);
  r1->SetMarkerColor(4);
  s1->SetMarkerSize(0.7);
  r1->SetMarkerSize(0.7);
  s1->SetLineColor(1);
  r1->SetLineColor(1);

  s2->SetMarkerStyle(20);
  r2->SetMarkerStyle(21);
  s2->SetMarkerColor(2);
  r2->SetMarkerColor(4);
  s2->SetMarkerSize(0.7);
  r2->SetMarkerSize(0.7);
  s2->SetLineColor(1);
  r2->SetLineColor(1);

  s3->SetMarkerStyle(20);
  r3->SetMarkerStyle(21);
  s3->SetMarkerColor(2);
  r3->SetMarkerColor(4);
  s3->SetMarkerSize(0.7);
  r3->SetMarkerSize(0.7);
  s3->SetLineColor(1);
  r3->SetLineColor(1);
  s3->SetLineWidth(1);
  r3->SetLineWidth(1);

  s4->SetMarkerStyle(20);
  r4->SetMarkerStyle(21);
  s4->SetMarkerColor(2);
  r4->SetMarkerColor(4);
  s4->SetMarkerSize(0.7);
  r4->SetMarkerSize(0.7);
  s4->SetLineColor(2);
  r4->SetLineColor(4);
  s4->SetLineWidth(2);
  r4->SetLineWidth(2);

  s5->SetMarkerStyle(20);
  r5->SetMarkerStyle(21);
  s5->SetMarkerColor(2);
  r5->SetMarkerColor(4);
  s5->SetMarkerSize(0.7);
  r5->SetMarkerSize(0.7);
  s5->SetLineColor(1);
  r5->SetLineColor(1);

  s6->SetMarkerStyle(20);
  r6->SetMarkerStyle(21);
  s6->SetMarkerColor(2);
  r6->SetMarkerColor(4);
  s6->SetMarkerSize(0.7);
  r6->SetMarkerSize(0.7);
  s6->SetLineColor(2);
  r6->SetLineColor(4);
  s6->SetLineWidth(2);
  r6->SetLineWidth(2);



  //setStats(r1,s1, startingY, startingX, fit);
  canvas->cd(1);
  setStats(s1,r1, -1, 0, false);
  r1->Draw();
  s1->Draw("sames");

  canvas->cd(2);
  setStats(s2,r2, -1, 0, false);
  r2->Draw();
  s2->Draw("sames");

  canvas->cd(3);
  gPad->SetLogx();
  setStats(s3,r3, -1, 0, false);
  r3->Draw();
  s3->Draw("sames");

  canvas->cd(4);
  gPad->SetLogx();
  setStats(s4,r4, -1, 0, false);
  //  setStats(s4,r4, 0.6, 0.65, false);
  r4->Draw();
  s4->Draw("sames");

  canvas->cd(5);
  setStats(s5,r5, -1, 0, false);
  //  setStats(s5,r5, -1, 0, false);
  r5->Draw();
  s5->Draw("sames");


  canvas->cd(6);
  setStats(s6,r6, -1, 0, false);
  //  setStats(s6,r6, 0.6, 0.65, false);
  r6->Draw();
  s6->Draw("sames");
}

void plotPulls(TCanvas *canvas, 
	       TH1F *s1,TH1F *r1, TH1F *s2,TH1F *r2, 
	       TH1F *s3,TH1F *r3, TH1F *s4,TH1F *r4,
	       TH1F *s5,TH1F *r5,TH1F *s6,TH1F *r6,
	       TText* te,
	       char * option, double startingY, double startingX = .1,bool fit = false){
  canvas->Divide(2,3);

  s1->SetMarkerStyle(20);
  r1->SetMarkerStyle(21);
  s1->SetMarkerColor(2);
  r1->SetMarkerColor(4);
  s1->SetMarkerSize(0.7);
  r1->SetMarkerSize(0.7);
  s1->SetLineColor(2);
  r1->SetLineColor(4);
  s1->SetLineWidth(2);
  r1->SetLineWidth(2);

  s2->SetMarkerStyle(20);
  r2->SetMarkerStyle(21);
  s2->SetMarkerColor(2);
  r2->SetMarkerColor(4);
  s2->SetMarkerSize(0.7);
  r2->SetMarkerSize(0.7);
  s2->SetLineColor(2);
  r2->SetLineColor(4);
  s2->SetLineWidth(2);
  r2->SetLineWidth(2);

  s3->SetMarkerStyle(20);
  r3->SetMarkerStyle(21);
  s3->SetMarkerColor(2);
  r3->SetMarkerColor(4);
  s3->SetMarkerSize(0.7);
  r3->SetMarkerSize(0.7);
  s3->SetLineColor(2);
  r3->SetLineColor(4);
  s3->SetLineWidth(2);
  r3->SetLineWidth(2);

  s4->SetMarkerStyle(20);
  r4->SetMarkerStyle(21);
  s4->SetMarkerColor(2);
  r4->SetMarkerColor(4);
  s4->SetMarkerSize(0.7);
  r4->SetMarkerSize(0.7);
  s4->SetLineColor(2);
  r4->SetLineColor(4);
  s4->SetLineWidth(2);
  r4->SetLineWidth(2);

  s5->SetMarkerStyle(20);
  r5->SetMarkerStyle(21);
  s5->SetMarkerColor(2);
  r5->SetMarkerColor(4);
  s5->SetMarkerSize(0.7);
  r5->SetMarkerSize(0.7);
  s5->SetLineColor(2);
  r5->SetLineColor(4);
  s5->SetLineWidth(2);
  r5->SetLineWidth(2);

  s6->SetMarkerStyle(20);
  r6->SetMarkerStyle(21);
  s6->SetMarkerColor(2);
  r6->SetMarkerColor(4);
  s6->SetMarkerSize(0.7);
  r6->SetMarkerSize(0.7);
  s6->SetLineColor(2);
  r6->SetLineColor(4);
  s6->SetLineWidth(2);
  r6->SetLineWidth(2);


  //setStats(r1,s1, startingY, startingX, fit);

  float small = 1e-4;

  canvas->cd(1);
  setStats(s1,r1, 0.6, 0.65, true);
  r1->Draw();
  s1->Draw("sames");

  canvas->cd(2);
  setStats(s2,r2, 0.6, 0.65, true);
  r2->Draw();
  s2->Draw("sames");

  canvas->cd(3);
  setStats(s3,r3, 0.6, 0.65, true);
  r3->Draw();
  s3->Draw("sames");

  canvas->cd(4);
  setStats(s4,r4, 0.6, 0.65, true);
  r4->Draw();
  s4->Draw("sames");

  canvas->cd(5);
  setStats(s5,r5, 0.6, 0.65, true);
  r5->Draw();
  s5->Draw("sames");

  canvas->cd(6);
  setStats(s6,r6, 0.6, 0.65, true);
  r6->Draw();
  s6->Draw("sames");
}

void plotResolutions(TCanvas *canvas, 
		     TH1F *s1,TH1F *r1, TH1F *s2,TH1F *r2, 
		     TH1F *s3,TH1F *r3, TH1F *s4,TH1F *r4,
		     TH1F *s5,TH1F *r5,TH1F *s6,TH1F *r6,
		     TText* te,
		     char * option, double startingY, double startingX = .1,bool fit = false){

  canvas->Divide(2,3);

  s1->SetMarkerStyle(20);
  r1->SetMarkerStyle(21);
  s1->SetMarkerColor(2);
  r1->SetMarkerColor(4);
  s1->SetMarkerSize(0.7);
  r1->SetMarkerSize(0.7);
  s1->SetLineColor(1);
  r1->SetLineColor(1);

  s2->SetMarkerStyle(20);
  r2->SetMarkerStyle(21);
  s2->SetMarkerColor(2);
  r2->SetMarkerColor(4);
  s2->SetMarkerSize(0.7);
  r2->SetMarkerSize(0.7);
  s2->SetLineColor(1);
  r2->SetLineColor(1);

  s3->SetMarkerStyle(20);
  r3->SetMarkerStyle(21);
  s3->SetMarkerColor(2);
  r3->SetMarkerColor(4);
  s3->SetMarkerSize(0.7);
  r3->SetMarkerSize(0.7);
  s3->SetLineColor(1);
  r3->SetLineColor(1);

  s4->SetMarkerStyle(20);
  r4->SetMarkerStyle(21);
  s4->SetMarkerColor(2);
  r4->SetMarkerColor(4);
  s4->SetMarkerSize(0.7);
  r4->SetMarkerSize(0.7);
  s4->SetLineColor(1);
  r4->SetLineColor(1);

  s5->SetMarkerStyle(20);
  r5->SetMarkerStyle(21);
  s5->SetMarkerColor(2);
  r5->SetMarkerColor(4);
  s5->SetMarkerSize(0.7);
  r5->SetMarkerSize(0.7);
  s5->SetLineColor(1);
  r5->SetLineColor(1);

  if (s6&&r6) {
    s6->SetMarkerStyle(20);
    r6->SetMarkerStyle(21);
    s6->SetMarkerColor(2);
    r6->SetMarkerColor(4);
    s6->SetMarkerSize(0.7);
    r6->SetMarkerSize(0.7);
    s6->SetLineColor(1);
    r6->SetLineColor(1);
    s6->SetLineWidth(2);
    r6->SetLineWidth(2);
  }


  //setStats(r1,s1, startingY, startingX, fit);
  canvas->cd(1);
  gPad->SetLogy(); 
  setStats(s1,r1, -1, 0, false);
  r1->Draw();
  s1->Draw("sames");

  canvas->cd(2);
  gPad->SetLogy(); 
  setStats(s2,r2, -1, 0, false);
  r2->Draw();
  s2->Draw("sames");

  canvas->cd(3);
  gPad->SetLogy(); 
  setStats(s3,r3, -1, 0, false);
  r3->Draw();
  s3->Draw("sames");

  canvas->cd(4);
  gPad->SetLogy(); 
  setStats(s4,r4, -1, 0, false);
  r4->Draw();
  s4->Draw("sames");

  canvas->cd(5);
  gPad->SetLogy(); 
  setStats(s5,r5, -1, 0, false);
  r5->Draw();
  s5->Draw("sames");

  if (s6&&r6) {
    canvas->cd(6);
    r6->Draw();
    s6->Draw("sames");
  }
}

void plotMeanValues(TCanvas *canvas, 
		     TH1F *s1,TH1F *r1, TH1F *s2,TH1F *r2, 
		     TH1F *s3,TH1F *r3, TH1F *s4,TH1F *r4,
		     TH1F *s5,TH1F *r5,TH1F *s6,TH1F *r6,
		     TText* te,
		     char * option, double startingY, double startingX = .1,bool fit = false){
  canvas->Divide(2,3);

  s1->SetMarkerStyle(20);
  r1->SetMarkerStyle(21);
  s1->SetMarkerColor(2);
  r1->SetMarkerColor(4);
  s1->SetMarkerSize(0.7);
  r1->SetMarkerSize(0.7);
  s1->SetLineColor(1);
  r1->SetLineColor(1);

  s2->SetMarkerStyle(20);
  r2->SetMarkerStyle(21);
  s2->SetMarkerColor(2);
  r2->SetMarkerColor(4);
  s2->SetMarkerSize(0.7);
  r2->SetMarkerSize(0.7);
  s2->SetLineColor(1);
  r2->SetLineColor(1);

  s3->SetMarkerStyle(20);
  r3->SetMarkerStyle(21);
  s3->SetMarkerColor(2);
  r3->SetMarkerColor(4);
  s3->SetMarkerSize(0.7);
  r3->SetMarkerSize(0.7);
  s3->SetLineColor(1);
  r3->SetLineColor(1);

  s4->SetMarkerStyle(20);
  r4->SetMarkerStyle(21);
  s4->SetMarkerColor(2);
  r4->SetMarkerColor(4);
  s4->SetMarkerSize(0.7);
  r4->SetMarkerSize(0.7);
  s4->SetLineColor(1);
  r4->SetLineColor(1);


  s5->SetMarkerStyle(20);
  r5->SetMarkerStyle(21);
  s5->SetMarkerColor(2);
  r5->SetMarkerColor(4);
  s5->SetMarkerSize(0.7);
  r5->SetMarkerSize(0.7);
  s5->SetLineColor(1);
  r5->SetLineColor(1);

  if (s6&&r6) {
    s6->SetMarkerStyle(20);
    r6->SetMarkerStyle(21);
    s6->SetMarkerColor(2);
    r6->SetMarkerColor(4);
    s6->SetMarkerSize(0.7);
    r6->SetMarkerSize(0.7);
    s6->SetLineColor(1);
    r6->SetLineColor(1);
    s6->SetLineWidth(2);
    r6->SetLineWidth(2);
  }

  //setStats(r1,s1, startingY, startingX, fit);
  canvas->cd(1);
  setStats(s1,r1, -1, 0, false);
  r1->Draw();
  s1->Draw("sames");

  canvas->cd(2);
  setStats(s2,r2, -1, 0, false);
  r2->Draw();
  s2->Draw("sames");

  canvas->cd(3);
  setStats(s3,r3, -1, 0, false);
  r3->Draw();
  s3->Draw("sames");

  canvas->cd(4);
  setStats(s4,r4, -1, 0, false);
  r4->Draw();
  s4->Draw("sames");

  canvas->cd(5);
  setStats(s5,r5, -1, 0, false);
  r5->Draw();
  s5->Draw("sames");

  if (s6&&r6) {
    canvas->cd(6);
    setStats(s6,r6, -1, 0, false);
    r6->Draw();
    s6->Draw("sames");
  }
}


void setStats(TH1* s,TH1* r, double startingY, double startingX = .1,bool fit){
  if (startingY<0){
    s->SetStats(0);
    r->SetStats(0);
  } else {
    //gStyle->SetOptStat(1001);

    if (fit){
      s->Fit("gaus");
      TF1* f1 = (TF1*) s->GetListOfFunctions()->FindObject("gaus");
      if (f1) {
	f1->SetLineColor(2);
	f1->SetLineWidth(1);
      }
    }
    s->Draw();
    gPad->Update(); 
    TPaveStats* st1 = (TPaveStats*) s->GetListOfFunctions()->FindObject("stats");
    if (st1) {
      //      if (fit) {st1->SetOptFit(0010);    st1->SetOptStat(1001);}
      if (fit) {st1->SetOptFit(0010);    st1->SetOptStat(111110);}
      st1->SetX1NDC(startingX);
      st1->SetX2NDC(startingX+0.30);
      st1->SetY1NDC(startingY+0.20);
      st1->SetY2NDC(startingY+0.35);
      st1->SetTextColor(2);
    }
    else s->SetStats(0);
    if (fit) {
      r->Fit("gaus");
      TF1* f2 = (TF1*) r->GetListOfFunctions()->FindObject("gaus");
      if (f2) {
	f2->SetLineColor(4);
	f2->SetLineWidth(1);
      }
    }
    r->Draw();
    gPad->Update(); 
    TPaveStats* st2 = (TPaveStats*) r->GetListOfFunctions()->FindObject("stats");
    if (st2) {
      //      if (fit) {st2->SetOptFit(0010);    st2->SetOptStat(1001);}
      if (fit) {st2->SetOptFit(0010);    st2->SetOptStat(111110);}
      st2->SetX1NDC(startingX);
      st2->SetX2NDC(startingX+0.30);
      st2->SetY1NDC(startingY);
      st2->SetY2NDC(startingY+0.15);
      st2->SetTextColor(4);
    }
    else r->SetStats(0);
  }
}

void fixRangeY(TH1* r,TH1* s){
  double ymin = (r->GetBinContent(r->GetMinimumBin()) < s->GetBinContent(s->GetMinimumBin())) ? 
    r->GetBinContent(r->GetMinimumBin()) : s->GetBinContent(s->GetMinimumBin());
  double ymax = (r->GetBinContent(r->GetMaximumBin()) > s->GetBinContent(s->GetMaximumBin())) ?
    r->GetBinContent(r->GetMaximumBin()) : s->GetBinContent(s->GetMaximumBin());
  r->GetYaxis()->SetRangeUser(ymin*0.9,ymax*1.1);
  s->GetYaxis()->SetRangeUser(ymin*0.9,ymax*1.1);
}
