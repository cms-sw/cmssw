#include <vector>
// Commands executed in a GLOBAL scope, e.g. created hitograms aren't erased...
void plot_HB(TString inputfile="simevent_HB.root",
	     TString outputfile="HB_histo.root",
	     Int_t drawmode = 2, 
	     TString    reffile="../data/HB_ref.root")
{

  // Option to no-action(0)/draw(1)/save(2) (default = 0) histograms in gif.
  //int doDraw = 0; 
  int doDraw = drawmode;

  char * treename = "Events";        //The Title of Tree.
  
  delete gROOT->GetListOfFiles()->FindObject(inputfile);

  TFile * myf  = new TFile(inputfile);
  
  TTree * tree = dynamic_cast<TTree*>(myf->Get("Events"));
  assert(tree != 0);

  TBranch * branchLayer = tree->GetBranch("PHcalValidInfoLayer_g4SimHits_HcalInfoLayer_CaloTest.obj");
  assert(branchLayer != 0);

  TBranch * branchNxN = tree->GetBranch("PHcalValidInfoNxN_g4SimHits_HcalInfoNxN_CaloTest.obj");
  assert(branchNxN != 0);

  TBranch * branchJets = tree->GetBranch( "PHcalValidInfoJets_g4SimHits_HcalInfoJets_CaloTest.obj");  assert(branchJets != 0);

  // Just number of entries (same for all branches)
  int  nent = branchLayer->GetEntries();
  cout << "Entries branchLayer : " << nent << endl;
  nent = branchJets->GetEntries();
  cout << "Entries branchJets  : " << nent << endl;
  nent = branchNxN->GetEntries();
  cout << "Entries branchNxN   : " << nent << endl;

  // Variables from branches
  PHcalValidInfoJets infoJets;
  branchJets->SetAddress( &infoJets); 
  PHcalValidInfoLayer infoLayer;
  branchLayer->SetAddress( &infoLayer); 
  PHcalValidInfoNxN infoNxN;
  branchNxN->SetAddress( &infoNxN); 
  
  //***************************************************************************
  // Histo titles-labels
  const int Nhist1     = 47, Nhist2 = 1;  // N simple and N combined histos
  const int Nhist1spec =  8;              // N special out of Nsimple total 
  const int nLayersMAX = 20;
  const int nDepthsMAX =  5;

  TH1F *h;                              // just a pointer
  TH1F *h1[Nhist1];
  TH1F *h1l[nLayersMAX];                // + all scint. layers separately
  TH1F *h1d[nDepthsMAX];                // + all depths  

  TH2F *h2[Nhist2];
  TH2F *h2g[5];         // +  eta-phi grid -related for all depthes
  
  char *label1[Nhist1], *label2[Nhist2], *label1l[nLayersMAX ];
  char *label1d[nDepthsMAX], *label2g[5];
  

  // simple histos  
  label1[0]  = &"rJetHits.gif";
  label1[1]  = &"tJetHits.gif";
  label1[2]  = &"eJetHits.gif";
  label1[3]  = &"ecalJet.gif";
  label1[4]  = &"hcalJet.gif";
  label1[5]  = &"hoJet.gif";
  label1[6]  = &"etotJet.gif";
  label1[7]  = &"detaJet.gif";
  label1[8]  = &"dphiJet.gif";
  label1[9]  = &"drJet.gif";
  label1[10] = &"jetE.gif";
  label1[11] = &"jetEta.gif";
  label1[12] = &"jetPhi.gif";
  label1[13] = &"dijetM.gif";
  label1[14] = &"ecalNxNr.gif";
  label1[15] = &"hcalNxNr.gif";
  label1[16] = &"hoNxNr.gif";
  label1[17] = &"etotNxNr.gif";
  label1[18] = &"ecalNxN.gif";
  label1[19] = &"hcalNxN.gif";
  label1[20] = &"hoNxN.gif";
  label1[21] = &"etotNxN.gif";
  label1[22] = &"layerHits.gif";
  label1[23] = &"etaHits.gif";          
  label1[24] = &"phiHits.gif";
  label1[25] = &"eHits.gif";
  label1[26] = &"tHits.gif";
  label1[27] = &"idHits.gif";
  label1[28] = &"jitterHits.gif";
  label1[29] = &"eIxI.gif";
  label1[30] = &"tIxI.gif";
  label1[31] = &"eLayer.gif";
  label1[32] = &"eDepth.gif";
  label1[33] = &"eHO.gif";
  label1[34] = &"eHBHE.gif";
  label1[35] = &"elongHF.gif";
  label1[36] = &"eshortHF.gif";
  label1[37] = &"eEcalHF.gif";
  label1[38] = &"eHcalHF.gif";
  label1[46] = &"E_hcal.gif";

  // special
  label1[39] = &"NxN_trans_fraction.gif"; 
  label1[40] = &"tHist_50ns.gif";   
  label1[41] = &"tHist_eweighted.gif"; 
  label1[42] = &"nHits_ECAL.gif";
  label1[43] = &"nHits_HCAL.gif";
  label1[44] = &"nHits.gif";
  label1[45] = &"longProf_eweighted.gif";

  label1l[0] = &"layer_0.gif"; 
  label1l[1] = &"layer_1.gif"; 
  label1l[2] = &"layer_2.gif"; 
  label1l[3] = &"layer_3.gif"; 
  label1l[4] = &"layer_4.gif"; 
  label1l[5] = &"layer_5.gif"; 
  label1l[6] = &"layer_6.gif"; 
  label1l[7] = &"layer_7.gif"; 
  label1l[8] = &"layer_8.gif"; 
  label1l[9] = &"layer_9.gif"; 
  label1l[10] = &"layer_10.gif"; 
  label1l[11] = &"layer_11.gif"; 
  label1l[12] = &"layer_12.gif"; 
  label1l[13] = &"layer_13.gif"; 
  label1l[14] = &"layer_14.gif"; 
  label1l[15] = &"layer_15.gif"; 
  label1l[16] = &"layer_16.gif"; 
  label1l[17] = &"layer_17.gif"; 
  label1l[18] = &"layer_18.gif"; 
  label1l[19] = &"layer_19.gif"; 

  label1d[0] = &"depth_0.gif"; 
  label1d[1] = &"depth_1.gif"; 
  label1d[2] = &"depth_2.gif"; 
  label1d[3] = &"depth_3.gif"; 
  label1d[4] = &"depth_4.gif"; 
 
  // more complicated histos and profiles
 
  label2[0] = &"JetHCALvsECAL.gif";

  label2g[0] = &"Eta-phi_grid_depth_0.gif";
  label2g[1] = &"Eta-phi_grid_depth_1.gif";
  label2g[2] = &"Eta-phi_grid_depth_2.gif";
  label2g[3] = &"Eta-phi_grid_depth_3.gif";
  label2g[4] = &"Eta-phi_grid_all_depths.gif";

  // Some constants

  const float fact = 117.0; // sampling factor which corresponds
                                   //  to those 
                                   // for layer = 0,1 in SimG4HcalValidation.cc

  //***************************************************************************
  //...Book histograms 


  for (int i = 0; i < Nhist1-Nhist1spec; i++) {
    char hname[3]; 
    sprintf(hname,"h%d",i);

    if(i == 4 || i == 7 || i == 8 || i == 11 || i == 12 || i == 6) {
      if(i == 11) h1[i] = new TH1F(hname,label1[i],100,-5.,5.);   
      if(i == 12)  
	          h1[i] = new TH1F(hname,label1[i],72,-3.1415926,3.1415926);   
      if(i == 7 || i == 8) h1[i] = new TH1F(hname,label1[i],100,-0.1,0.1);  
      if( i == 4)  h1[i] = new TH1F(hname,label1[i],50,0.,100.);  
      if( i == 6)  h1[i] = new TH1F(hname,label1[i],50,0.,100.);
    }
    else { 
      h1[i] = new TH1F(hname,label1[i],100,1.,0.);  
    }
  }

  // Special : transverse profile 
  h1[39] = new TH1F("h39",label1[39],4,0.,4.);  

  // Special : global timing < 50 ns 
  h1[40] = new TH1F("h40",label1[40],50,0.,50.);  
  // Special : timing in the cluster (7x7) enery-weighted
  h1[41] = new TH1F("h41",label1[41],30,0.,30.);  
  // Special : number of ECAL&HCAL hits
  h1[42] = new TH1F("h42",label1[42],300,0.,3000.);  
  h1[43] = new TH1F("h43",label1[43],300,0.,3000.);  
  h1[44] = new TH1F("h44",label1[44],300,0.,3000.);  

  // Special : Longitudinal profile
  h1[45] = new TH1F("h45",label1[45],20,0.,20.);
  // Etot HCAL
  TH1F *h1[46] = new TH1F("h46",label1[46],50,0.,1.0);

  for (int i = 0;  i < Nhist1; i++) {
    if(i != 39)  h1[i]->Sumw2();
  }

  for (int i = 0; i < Nhist2; i++) {
    char hname[3]; 
    sprintf(hname,"D%d",i);
    h2[i] = new TH2F(hname,label2[i],100,0.,100.,100,0.,100.);
  }

  // scint. layers
  for (int i = 0; i < nLayersMAX; i++) {
    char hname[4]; 
    sprintf(hname,"hl%d",i);
    h1l[i] = new TH1F(hname,label1l[i],40,0.,0.4);  
  }
  // depths
  Float_t max[5] = {30000, 500, 500, 200, 200.};
  for (int i = 0; i < nDepthsMAX; i++) {
    char hname[3]; 
    sprintf(hname,"hd%d",i);
    h1d[i] = new TH1F(hname,label1d[i],100,0.,max[i]);  
  }

  // eta-phi grid (for muon samples)
  for (int i = 0; i < 5; i++) {
    char hname[3]; 
    sprintf(hname,"Dg%d",i);
    h2g[i] = new TH2F(hname,label2g[i],1000,-5.,5.,576,-3.1415927,3.1415927);
  }

  //***************************************************************************
  //***************************************************************************
  //...Fetch the data and fill the histogram 

  // branches separately - 

  for (int i = 0; i<nent; i++) { 

    //    cout << "Ev. " << i << endl;

    // -- get entries
    branchLayer ->GetEntry(i);
    branchNxN   ->GetEntry(i);
    branchJets  ->GetEntry(i);

    // -- Leading Jet
    const int nJetHits =  infoJets.njethit();

    //    cout << "Ev. " << i <<  "  " << "  nJetHits " << nJetHits <<  endl;

    //    std::vector<float>  rJetHits(nJetHits);
    std::vector<float>  rJetHits(nJetHits);

    //    cout << "pass 1" << endl;

    rJetHits = infoJets.jethitr();

    //    cout << "pass 2" << endl;


    std::vector<float> tJetHits(nJetHits);
    tJetHits = infoJets.jethitt(); 
    std::vector<float> eJetHits(nJetHits);
    eJetHits = infoJets.jethite();

    float ecalJet = infoJets.ecaljet();
    float hcalJet = infoJets.hcaljet();
    float   hoJet = infoJets.hojet();
    float etotJet = infoJets.etotjet();

    float detaJet = infoJets.detajet();
    float dphiJet = infoJets.dphijet();
    float   drJet = infoJets.drjet();
    float  dijetM = infoJets.dijetm();

    
    for (int j = 0; j < nJetHits; j++) {
      h1[0]->Fill(rJetHits[j]);
      h1[1]->Fill(tJetHits[j]); 
      h1[2]->Fill(eJetHits[j]);
    }
   
    h1[3]->Fill(ecalJet); //
    h1[4]->Fill(hcalJet); //

    h1[5]->Fill(hoJet);
    h1[6]->Fill(etotJet);

    h2[0]->Fill(ecalJet,hcalJet); //

    h1[7]->Fill(detaJet);
    h1[8]->Fill(dphiJet);
    h1[9]->Fill(drJet);

    h1[13]->Fill(dijetM);

    // All Jets 

    int                nJets  = infoJets.njet();
    std::vector<float> jetE(nJets);
    jetE  = infoJets.jete();
    std::vector<float> jetEta(nJets);
    jetEta = infoJets.jeteta();
    std::vector<float> jetPhi(nJets);
    jetPhi = infoJets.jetphi();

    for (int j = 0; j < nJets; j++) {
      h1[10]->Fill(jetE[j]);
      h1[11]->Fill(jetEta[j]);
      h1[12]->Fill(jetPhi[j]);
    }

  
    // NxN quantities
    float ecalNxNr = infoNxN.ecalnxnr();
    float hcalNxNr = infoNxN.hcalnxnr();
    float   hoNxNr = infoNxN.honxnr();
    float etotNxNr = infoNxN.etotnxnr();

    float ecalNxN  = infoNxN.ecalnxn();
    float hcalNxN  = infoNxN.hcalnxn();
    float   hoNxN  = infoNxN.honxn();
    float etotNxN  = infoNxN.etotnxn();

    h1[14]->Fill(ecalNxNr);
    h1[15]->Fill(hcalNxNr);
    h1[16]->Fill(hoNxNr);
    h1[17]->Fill(etotNxNr);
   
    h1[18]->Fill(ecalNxN);
    h1[19]->Fill(hcalNxN);
    h1[20]->Fill(hoNxN);
    h1[21]->Fill(etotNxN);


    // CaloHits from PHcalValidInfoLayer  
    
    int                    nHits = infoLayer.nHit();
    std::vector<float>    idHits (nHits);
    idHits = infoLayer.idHit();
    std::vector<float>   phiHits (nHits);
    phiHits = infoLayer.phiHit();
    std::vector<float>   etaHits (nHits);
    etaHits = infoLayer.etaHit();
    std::vector<float> layerHits (nHits);
    layerHits = infoLayer.layerHit();
    std::vector<float>     eHits (nHits);
    eHits = infoLayer.eHit();
    std::vector<float>     tHits (nHits);
    tHits  = infoLayer.tHit();

    int ne = 0, nh = 0; 
    for (int j = 0; j < nHits; j++) {
      int layer = layerHits[j]-1;
      int id    = (int)idHits[j];

      if(id >= 10) {ne++;}
      else {nh++;}

      //      cout << "Hit subdet = " << id  << "  lay = " << layer << endl;

      h1[22]->Fill(Float_t(layer));
      h1[23]->Fill(etaHits[j]);
      h1[24]->Fill(phiHits[j]);
      h1[25]->Fill(eHits[j]);
      h1[26]->Fill(tHits[j]);
      h1[27]->Fill(idHits[j]);
      //      h1[28]->Fill(jitterHits[j]);   // no jitter anymore

      h1[40]->Fill(tHits[j]);
      h1[41]->Fill(tHits[j],eHits[j]);
      
      if(id < 6) { // HCAL only. Depth is needed, not layer !!!
	//if(layer == 0)               h2g[0]->Fill(etaHits[j],phiHits[j]);
	//if(layer == 1)               h2g[1]->Fill(etaHits[j],phiHits[j]);
	//if(layer == 2)               h2g[2]->Fill(etaHits[j],phiHits[j]);
	//if(layer == 3)               h2g[3]->Fill(etaHits[j],phiHits[j]);
	h2g[4]->Fill(etaHits[j],phiHits[j]);
      }
      

    }
      
    h1[42]->Fill(Float_t(ne));
    h1[43]->Fill(Float_t(nh));
    h1[44]->Fill(Float_t(nHits));

    // NxN  PHcalValidInfoNxN 
    int                    nIxI = infoNxN.nnxn();
    std::vector<float>    idIxI (nIxI);
    idIxI = infoNxN.idnxn();
    std::vector<float>     eIxI (nIxI);
    eIxI  = infoNxN.enxn();
    std::vector<float>     tIxI (nIxI);
    tIxI = infoNxN.tnxn();
 
    for (int j = 0; j < nIxI ; j++) {   // NB !!! j < nIxI
      h1[29]->Fill(eIxI[j]);
      h1[30]->Fill(tIxI[j]);

      h1[39]->Fill(idIxI[j],eIxI[j]);  // transverse profile
      

    }

    // Layers and depths PHcalValidInfoLayer
    
    std::vector<float> eLayer (nLayersMAX);
    eLayer = infoLayer.elayer();
    std::vector<float> eDepth (nDepthsMAX);
    eDepth = infoLayer.edepth();
    
    float eTot = 0.;

    for (int j = 0; j < nLayersMAX ; j++) {
      h1[31]->Fill(eLayer[j]);
      h1l[j]->Fill(eLayer[j]);

	h1[45]->Fill((Float_t)(j),eLayer[j]);  // HCAL SimHits only 
        eTot += eLayer[j];
    }
    for (int j = 0; j < nDepthsMAX; j++) {
      h1[32]->Fill(eDepth[j]);
      h1d[j]->Fill(eDepth[j]);
    }

    h1[46]->Fill(eTot);               

       
    // The rest  PHcalValidInfoLayer
    float eHO      =  infoLayer.eho(); 
    float eHBHE    =  infoLayer.ehbhe(); 
    float elongHF  =  infoLayer.elonghf(); 
    float eshortHF =  infoLayer.eshorthf(); 
    float eEcalHF  =  infoLayer.eecalhf(); 
    float eHcalHF  =  infoLayer.ehcalhf(); 

    h1[33]->Fill(eHO);
    h1[34]->Fill(eHBHE);
 
    h1[35]->Fill(elongHF);
    h1[36]->Fill(eshortHF);
    h1[37]->Fill(eEcalHF);
    h1[38]->Fill(eHcalHF);

  }

  // cout << "After event cycle " << i << endl;

  // Transverse size histo integration
    
 
  h = h1[39];

  //  h->Draw();

  if(h->Integral() > 1.e-30 && h->Integral() < 1.e30 ) {
   
    int size = h->GetNbinsX();                  
    Float_t sum = 0.;

    for (int i = 1; i <= size; i++) { 
      Float_t y = h->GetBinContent(i);
      //      cout << " 1) h[39] bin " << i << " content = " << y << endl;
      sum +=  y;
      h->SetBinContent((Int_t)i, (Float_t)sum);

    }

    for (int i = 1; i <= size; i++) { 
      Float_t y = h->GetBinContent(i);
      h->SetBinContent((Int_t)i, y/sum);
      //      cout << " 2) h[39] bin " << i << " content = " << y/sum << endl;
    }
  }

  //...Prepare the main canva
    TCanvas *myc = new TCanvas("myc","",800,600);
    gStyle->SetOptStat(1111);   // set stat         :0 - nothing 


 
  // Cycle for 1D distributions
  for (int ihist = 0; ihist < Nhist1 ; ihist++) {
    if(h1[ihist]->Integral() > 1.e-30 && h1[ihist]->Integral() < 1.e30 ) { 
      
      h1[ihist]->SetLineColor(45);
      h1[ihist]->SetLineWidth(2); 
      
      if(doDraw == 1) {
	h1[ihist]->Draw("h");
	myc->SaveAs(label1[ihist]);
      }
    }
  }

  // Cycle for energy in all layers
  for (int ihist = 0; ihist < nLayersMAX; ihist++) {
    if(h1l[ihist]->Integral() > 1.e-30 && h1l[ihist]->Integral() < 1.e30 ) { 
      
      h1l[ihist]->SetLineColor(45);
      h1l[ihist]->SetLineWidth(2); 

      
      if(doDraw == 1) {
	h1l[ihist]->Draw("h");
	myc->SaveAs(label1l[ihist]);
      }
    }
  }


  // Cycle for 2D distributions 
  //  for (int ihist = 0; ihist < 1 ; ihist++) {
  for (int ihist = 0; ihist < Nhist2 ; ihist++) {
    if(h2[ihist]->Integral() > 1.e-30 && h2[ihist]->Integral() < 1.e30 ) { 
      
      h2[ihist]->SetMarkerColor(45);
      h2[ihist]->SetMarkerStyle(20);
      h2[ihist]->SetMarkerSize(0.7);  // marker size !
      
      h2[ihist]->SetLineColor(45);
      h2[ihist]->SetLineWidth(2); 
      
      if(doDraw == 1) {
	h2[ihist]->Draw();
	myc->SaveAs(label2[ihist]);
      }
    }
  }
  
 
  // Cycle for eta-phi grids 
  //  for (int ihist = 0; ihist < 5 ; ihist++) {
  for (int ihist = 4; ihist < 5 ; ihist++) {
    if(h2g[ihist]->Integral() > 1.e-30 && h2g[ihist]->Integral() < 1.e30 ) { 
      
      h2g[ihist]->SetMarkerColor(41);
      h2g[ihist]->SetMarkerStyle(20);
      h2g[ihist]->SetMarkerSize(0.2); 
      
      h2g[ihist]->SetLineColor(41);
      h2g[ihist]->SetLineWidth(2); 
      

      if(doDraw == 1) {
	h2g[ihist]->Draw();
	myc->SaveAs(label2g[ihist]);
      }
    }
  }
 

  // added by Julia Yarba
  //-----------------------   
  // this is a temporary stuff that I've made
  // to create a reference ROOT histogram file


  if (doDraw == 2) {
    TFile OutFile(outputfile,"RECREATE") ;
    int ih = 0 ;
    for ( ih=0; ih<nLayersMAX; ih++ )
      {
	h1l[ih]->Write() ;
      }
    for ( ih=0; ih<Nhist1; ih++ )
      { 
	h1[ih]->Write() ;
      }

    OutFile.Write() ;
    OutFile.Close() ;
    cout << outputfile << " histogram file created" << endl ; 
    
    return ;
    
  }

   // now perform Chi2 test for histograms that hold
   // energy deposition in the Hcal layers 1-6, using
   // "reference" and "current" histograms 
   
   
   // open up ref. ROOT file
   //
   TFile RefFile(reffile) ;
   
   // service variables
   //
   TH1F* ref_hist = 0 ;
   int ih = 0 ;
   
   // loop over layers 1-10
   //
   for ( ih=1; ih<11; ih++ )
   {
      // service - name of the ref histo
      //
      char ref_hname[4] ;
      sprintf( ref_hname, "hl%d", ih ) ;
      
      // retrive ref.histos one by one
      //
      ref_hist = (TH1F*)RefFile.Get( ref_hname ) ;
      
      // check if valid (no-NULL)
      //
      if ( ref_hist == NULL )
      {
         // print warning in case of trouble
	 //
	 cout << "No such ref. histogram" << *ref_hname << endl ; 
      }
      else
      {
	// everything OK - perform Chi2 test
	//
	Double_t *res;
	Double_t pval = h1l[ih]->Chi2Test( ref_hist, "UU", res) ;
	 
	// output Chi2 comparison results
	//
	cout << "[OVAL] : Edep in Layer " << ih << ", p-value= " << pval << endl ;
      }
   }


   // loop over specials : timing,  nhits(ECAL and HCAL) 
   //
   for ( ih=39; ih<47; ih++ )
   {
      // service - name of the ref histo
      //
      char ref_hname[4] ;
      sprintf( ref_hname, "h%d", ih ) ;
      
      // retrive ref.histos one by one
      //
      ref_hist = (TH1F*)RefFile.Get( ref_hname ) ;
      
      // check if valid (no-NULL)
      //
      if ( ref_hist == NULL )
      {
         // print warning in case of trouble
	 //
	 cout << "No such ref. histogram" << *ref_hname << endl ; 
      }
      else
      {
	// everything OK - perform Chi2 test
	 //
	Double_t *res;
	Double_t pval = h1[ih]->Chi2Test( ref_hist, "UU", res ) ;
	
	// output Chi2 comparison results
	//
	cout << "[OVAL] : histo " << ih << ", p-value= " << pval << endl ;
      }
   }

   
   // close ref. ROOT file
   //
   RefFile.Close() ;
   
  // at the end, close "current" ROOT tree file
  //
  myf->Close();

  // COMMENT OUT THE REST ================================================ 

}
