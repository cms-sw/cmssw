// Commands executed in a GLOBAL scope, e.g. created hitograms aren't erased...
void plot_HF(TString  inputfile="HF_ref.root",
	     TString outputfile="HF_histo.root",
	     Int_t drawmode = 0,
             TString    reffile="../data/HF_ref.root")
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

  const int Nhist1     = 20, Nhist2 = 1;  // N simple and N combined histos
  const int Nhist1spec =  5;              // N special out of Nsimple total 

  TH1F *h;                              // just a pointer
  TH1F *h1[Nhist1];
  TH2F *h2g;                    // +  eta-phi grid -related for all depthes
  
  char *label1[Nhist1];
  char *label2g;
  

  // simple histos  
  label1[0]  = &"rJetHits.gif";
  label1[1]  = &"tJetHits.gif";
  label1[2]  = &"eJetHits.gif";
  label1[3]  = &"ecalJet.gif";
  label1[4]  = &"hcalJet.gif";
  label1[5]  = &"etotJet.gif";
  label1[6]  = &"jetE.gif";
  label1[7]  = &"jetEta.gif";
  label1[8]  = &"jetPhi.gif";
  label1[9]  = &"etaHits.gif";          
  label1[10] = &"phiHits.gif";
  label1[11] = &"eHits.gif";
  label1[12] = &"tHits.gif";
  label1[13] = &"eEcalHF.gif";
  label1[14] = &"eHcalHF.gif";

  // special
  label1[15] = &"tHits_60ns.gif";   
  label1[16] = &"tHits_eweighted.gif"; 
  label1[17] = &"nHits_HF.gif";
  label1[18] = &"elongHF.gif";
  label1[19] = &"eshortHF.gif";

  label2g    = &"Eta-phi_grid_all_depths.gif";


  //***************************************************************************
  //...Book histograms 

  for (Int_t i = 0; i < Nhist1-Nhist1spec; i++) {
    char hname[3]; 
    sprintf(hname,"h%d",i);

    if(i == 4 || i == 7 || i == 8 ) {
      if(i == 7) h1[i] = new TH1F(hname,label1[i],100,-5.,5.);   
      if(i == 8) h1[i] = new TH1F(hname,label1[i],72,-3.1415926,3.1415926);   
      if(i == 4) h1[i] = new TH1F(hname,label1[i],30,0.,150.);  
    }
    else { 
      h1[i] = new TH1F(hname,label1[i],100,1.,0.);  
    }

  }

  // Special : global timing < 60 ns 
  h1[15] = new TH1F("h15",label1[15],60,0.,60.);  
  // Special : timing in the cluster (7x7) enery-weighted
  h1[16] = new TH1F("h16",label1[16],60,0.,60.);  
  // Special : number of HCAL hits
  h1[17] = new TH1F("h17",label1[17],20,0.,20.);  
  // Special : signal in long fibers
  h1[18] = new TH1F("h18",label1[18],50,0.,50.);
  // Special : signal in short fibers
  h1[19] = new TH1F("h19",label1[19],50,0.,50.);

  for (int i = 0;  i < Nhist1; i++) {
     h1[i]->Sumw2();
  }

  // eta-phi grid (for muon samples)
  h2g = new TH2F("Grid",label2g,1000,-5.,5.,576,-3.1415927,3.1415927);

  //***************************************************************************
  //***************************************************************************
  //...Fetch the data and fill the histogram 

  // branches separately - 

  for (int i = 0; i<nent; i++) { 

    // cout << "Ev. " << i << endl;

    // -- get entries
    branchLayer ->GetEntry(i);
    branchNxN   ->GetEntry(i);
    branchJets  ->GetEntry(i);

    // -- Leading Jet
    int nJetHits =  infoJets.njethit();
    //cout << "nJetHits = " << nJetHits << endl; 

    std::vector<float> rJetHits(nJetHits);
    rJetHits = infoJets.jethitr();
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
    h1[5]->Fill(etotJet);


    // All Jets 

    int                nJets  = infoJets.njet();
    std::vector<float> jetE(nJets);
    jetE  = infoJets.jete();
    std::vector<float> jetEta(nJets);
    jetEta = infoJets.jeteta();
    std::vector<float> jetPhi(nJets);
    jetPhi = infoJets.jetphi();

    for (int j = 0; j < nJets; j++) {
      h1[6]->Fill(jetE[j]);
      h1[7]->Fill(jetEta[j]);
      h1[8]->Fill(jetPhi[j]);
    }
 
  
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
 
      h1[9]->Fill(etaHits[j]);
      h1[10]->Fill(phiHits[j]);
      h1[11]->Fill(eHits[j]);
      h1[12]->Fill(tHits[j]);

      h1[15]->Fill(tHits[j]);
      h1[16]->Fill(tHits[j],eHits[j]);
      
      if(id < 6) { // HCAL only. Depth is needed, not layer !!!
	h2g->Fill(etaHits[j],phiHits[j]);
      }
      

    }
      
    h1[17]->Fill(Float_t(nh));


    // The rest  PHcalValidInfoLayer
    float elongHF  =  infoLayer.elonghf(); 
    float eshortHF =  infoLayer.eshorthf(); 
    float eEcalHF  =  infoLayer.eecalhf(); 
    float eHcalHF  =  infoLayer.ehcalhf(); 

    h1[13]->Fill(eEcalHF);
    h1[14]->Fill(eHcalHF);
 
    h1[18]->Fill(elongHF);
    h1[19]->Fill(eshortHF);

  }

  // cout << "After event cycle " << i << endl;

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


  // eta-phi grid 
  if(h2g->Integral() > 1.e-30 && h2g->Integral() < 1.e30 ) { 
    
    h2g->SetMarkerColor(41);
    h2g->SetMarkerStyle(20);
    h2g->SetMarkerSize(0.2); 
    h2g->SetLineColor(41);
    h2g->SetLineWidth(2); 
    
    if(doDraw == 1) {	
      h2g->Draw();
      myc->SaveAs(label2g);      
    }
  }
  

  // added by Julia Yarba
  //-----------------------   
  // this is a temporary stuff that I've made
  // to create a reference ROOT histogram file

  if (doDraw == 2) {
    TFile OutFile(outputfile,"RECREATE") ;
    int ih = 0 ;

    for ( ih=0; ih<Nhist1; ih++ )
      { 
	h1[ih]->Write() ;
      }

    OutFile.Write() ;
    OutFile.Close() ;
    cout << outputfile << " histogram file created" << endl ; 
    
    return;
    
  }

  /*
  return;
  */
 

   // now perform Chi2 test for histograms using
   // "reference" and "current" histograms 
   
   
   // open up ref. ROOT file
   //
   TFile RefFile(reffile) ;
   
   // service variables
   //
   TH1F* ref_hist = 0 ;
   int ih = 0 ;
   

   // loop over specials : timing,  nhits, simhits-E  
   //
   for ( ih=15; ih<20; ih++ )
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
	Double_t pval = h1[ih]->Chi2Test( ref_hist, "UU",res ) ;
	
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


  return ;  
}
