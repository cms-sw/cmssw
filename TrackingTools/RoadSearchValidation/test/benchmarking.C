{
  /*
    This macro will create various plots that can help with benchmarking and validation 
    of the track reconstruction both for RoadSearch Algorithm and Combinatorial TrackFinder
    in Package : RecoTracker.
    
    It is intended for checking the track validation using single muon input
    pT files. It can also be modified to work with any other MC files, but
    in that case, some of the histograms will not be meaningful.

    You need to change the input files, and input pT values of the files.
   
    Output Histogram : track_validation_histos.root

    Original authors: Carl Hinchey(chinchey@ku.edu), Tania Moulik(tmoulik@fnal.gov) 
                      (Univ. of Kansas)              (Univ. of Kansas)
  */

  gSystem->Load("libFWCoreFWLite.so");
  AutoLibraryLoader::enable();
  gStyle->SetLabelSize(0.05);

  // Configuration variables

  const int npt = 8; // number of pt points
  float ptval[npt] = {2.0,3.0,5.0,7.0,10.0,15.0,20.0,25.0};
  float xminpt[npt] = {1.9,2.8,4.6,6.5,9.2,14.0,18.5,23.0};
  float xmaxpt[npt] = {2.1,3.2,5.4,7.5,10.8,16.0,21.5,27.0};
  int usept[npt]   = {1,1,1,1,1,1,1,1};
  char *myfile[npt] = {"single_mum_2_final_fit.root",
			 "single_mum_3_final_fit.root",
			 "single_mum_5_final_fit.root",
			 "single_mum_7_final_fit.root",
			 "single_mum_10_final_fit.root",
			 "single_mum_15_final_fit.root",
			 "single_mum_20_final_fit.root",
			 "single_mum_25_final_fit.root"};


  

  TFile *fil1 = new TFile("track_validation_histos.root","RECREATE");
  fil1->cd();

  // creating histograms for each pt point
  TH1F *h_pttr_pt[npt];
  TH1F *h_pttr_eta[npt];
  TH1F *h_pttr_normchi[npt];
  TH1F *h_pttr_d0[npt];
  TH1F *h_pttr_nhit[npt];
  TH1F *h_pttr_phi[npt];
  TH1F *h_pttr_chi[npt];
  TH1F *h_pttr_etares[npt];
  TH1F *h_pttr_phires[npt];
  TH1F *h_pttr_ptres[npt];
  TH1F *h_pttr_z0[npt];
  TProfile *h_pttr_hitseta[npt];
  
  TH1F *h_pttr_pt_ctf[npt];
  TH1F *h_pttr_eta_ctf[npt];
  TH1F *h_pttr_normchi_ctf[npt];
  TH1F *h_pttr_d0_ctf[npt];
  TH1F *h_pttr_nhit_ctf[npt];
  TH1F *h_pttr_phi_ctf[npt];
  TH1F *h_pttr_chi_ctf[npt];
  TH1F *h_pttr_etares_ctf[npt];
  TH1F *h_pttr_phires_ctf[npt];
  TH1F *h_pttr_ptres_ctf[npt];
  TH1F *h_pttr_z0_ctf[npt];
  TProfile *h_pttr_hitseta_ctf[npt];
  TH1F *h_pttr_tracksfound[npt];
  TH1F *h_pttr_tracksfound_ctf[npt];

  float eta_resolutions[npt];
  float eta_resolutions_error[npt];

  float eta_resolutions_ctf[npt];
  float eta_resolutions_error_ctf[npt];

  float phi_resolutions[npt]; 
  float phi_resolutions_error[npt];

  float phi_resolutions_ctf[npt];
  float phi_resolutions_error_ctf[npt];
  
  float pt_resolutions[npt];
  float pt_resolutions_error[npt];

  float pt_resolutions_ctf[npt];
  float pt_resolutions_error_ctf[npt];
  
  float d0_resolutions[npt];
  float d0_resolutions_error[npt];
  
  float z0_resolutions[npt];
  float z0_resolutions_error[npt];
  
  float d0_resolutions_ctf[npt];
  float d0_resolutions_error_ctf[npt];
  
  float z0_resolutions_ctf[npt];
  float z0_resolutions_error_ctf[npt];
  
  TProfile *h_pttr_ptres_eta[npt];
  TProfile *h_pttr_ptres_eta_ctf[npt];

  TGraphErrors *h_pttr_sigptres_eta[npt];
  TGraphErrors *h_pttr_sigptres_eta_ctf[npt];

  //  TProfile *h_pttr_ptres_eta[npt];
  //  TProfile *h_pttr_ptres_eta_ctf[npt];
  
  for (unsigned int i=0; i<npt; ++i) { //RS histograms
    
    if (!usept[i]) continue;
    
    TString hname = Form("pt_vs_eta_%d",i);
    h_pttr_ptres_eta[i] = new TProfile(hname,"pT vs eta",25,0.0,2.5,0.0,100.0,"S");

    //    TString hname = Form("ptres_vs_eta_%d",i);
    //    h_pttr_ptres_eta[i] = new TProfile(hname,"pTres vs eta",25,0.0,2.5,0.0,100.0);
    h_pttr_sigptres_eta[i] = new TGraphErrors(25);
    
    TString hname = Form("pttr_pt_%d",i);
    TString htitle = Form("pT for %.1f GeV Single Muons",ptval[i]);
    h_pttr_pt[i] = new TH1F(hname, htitle, 100, xminpt[i], xmaxpt[i]);
    TString xtit = Form("pT %.1f GeV",ptval[i]);
    h_pttr_pt[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_pt[i]->GetYaxis()->SetTitle("events");
    
    hname = Form("pttr_eta_%d",i);
    htitle = Form("eta for %.1f GeV Single Muons",ptval[i]);
    h_pttr_eta[i] = new TH1F(hname, htitle, 100, -3.0, 3.0);
    xtit = Form("eta %.1f GeV",ptval[i]);
    h_pttr_eta[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_eta[i]->GetYaxis()->SetTitle("events");
    
    hname = Form("pttr_normchi_%d",i);
    htitle = Form("normalized chi2 for %.1f GeV Single Muons",ptval[i]);
    h_pttr_normchi[i] = new TH1F(hname, htitle, 100, 0.0, 5.0);
    xtit = Form("normalized chi2 %.1f GeV",ptval[i]);
    h_pttr_normchi[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_normchi[i]->GetYaxis()->SetTitle("events");
    
    hname = Form("pttr_d0_%d",i);
    htitle = Form("Impact parameter %.1f GeV Single Muons",ptval[i]);
    h_pttr_d0[i] = new TH1F(hname, htitle, 100, -0.02, 0.02);
    xtit = Form("d0 %.1f GeV",ptval[i]);
    h_pttr_d0[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_d0[i]->GetYaxis()->SetTitle("events");
    
    hname = Form("pttr_nhit_%d",i);
    htitle = Form("Number of hits for %.1f GeV Single Muons",ptval[i]);
    h_pttr_nhit[i] = new TH1F(hname, htitle, 25, 0.0, 25.0);
    xtit = Form("Hits %.1f GeV",ptval[i]);
    h_pttr_nhit[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_nhit[i]->GetYaxis()->SetTitle("events");
   
    hname = Form("pttr_phi_%d",i);
    htitle = Form("phi0 for %.1f GeV Single Muons",ptval[i]);
    h_pttr_phi[i] = new TH1F(hname, htitle, 100, -4.0, 4.0);
    xtit = Form("phi %.1f GeV",ptval[i]);
    h_pttr_phi[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_phi[i]->GetYaxis()->SetTitle("events");
    
    hname = Form("pttr_chi_%d",i);
    htitle = Form("chi2 for %.1f GeV Single Muons",ptval[i]);
    h_pttr_chi[i] = new TH1F(hname, htitle, 100, 0.0, 100.0);
    xtit = Form("chi %.1f GeV",ptval[i]);
    h_pttr_chi[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_chi[i]->GetYaxis()->SetTitle("events");
    
    hname = Form("pttr_etares_%d",i);
    htitle = Form("eta(rec)-eta(gen) for %.1f GeV Single Muons",ptval[i]);
    h_pttr_etares[i] = new TH1F(hname, htitle, 100, -0.005, 0.005);
    xtit = Form("eta resolution %.1f GeV",ptval[i]);
    h_pttr_etares[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_etares[i]->GetYaxis()->SetTitle("events");

    hname = Form("pttr_phires_%d",i);
    htitle = Form("phi(rec)-phi(gen) for %.1f GeV Single Muons",ptval[i]);
    h_pttr_phires[i] = new TH1F(hname, htitle, 100, -0.005, 0.005);
    xtit = Form("phi resolution %.1f GeV",ptval[i]);
    h_pttr_phires[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_phires[i]->GetYaxis()->SetTitle("events");
    
    hname = Form("pttr_ptres_%d",i);
    htitle = Form("pt(rec)-pt(gen) for %.1f GeV Single Muons",ptval[i]);
    h_pttr_ptres[i] = new TH1F(hname, htitle, 100, -1.0, 1.0);
    xtit = Form("pt resolution %.1f GeV",ptval[i]);
    h_pttr_ptres[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_ptres[i]->GetYaxis()->SetTitle("events");
    
    hname = Form("pttr_z0_%d",i);
    htitle = Form("z0 for %.1f GeV Single Muons",ptval[i]);
    h_pttr_z0[i] = new TH1F(hname,htitle,100,-0.05,0.05);
    xtit = Form("z0 for %.1f GeV Single Muons",ptval[i]);
    h_pttr_z0[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_z0[i]->GetYaxis()->SetTitle("events");
    
    hname = Form("pttr_hitseta_%d",i);
    htitle = Form("hits vs eta for %.1f GeV Single Muons",ptval[i]);
    h_pttr_hitseta[i] = new TProfile(hname, htitle, -54, -2.7, 2.7,0.0,20.0);
    xtit = Form("hits vs eta %.1f GeV",ptval[i]);
    h_pttr_hitseta[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_hitseta[i]->GetYaxis()->SetTitle("events");
  
    hname = Form("pttr_tracksfound_%d",i);
    htitle = Form("tracks/event for %.1f GeV Single Muons",ptval[i]);
    h_pttr_tracksfound[i] = new TH1F(hname, htitle, 10, 0.0, 10.0);
    xtit = Form("tracks/event for %.1f GeV",ptval[i]);
    h_pttr_tracksfound[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_tracksfound[i]->GetYaxis()->SetTitle("events");
  } // End Loop RS Histograms
  

  // Start LOOP CTF Histograms
  for (unsigned int i=0; i<npt; ++i) { // ctf histograms

    if (!usept[i]) continue;

    TString hname = Form("pt_vs_eta_%d_ctf",i);
    h_pttr_ptres_eta_ctf[i] = new TProfile(hname,"pT vs eta",25,0.0,2.5,0.0,100.0,"S");

    //    TString hname = Form("ptres_vs_eta_%d_ctf",i);
    //    h_pttr_ptres_eta_ctf[i] = new TProfile(hname,"pTres vs eta",25,0.0,2.5,0.0,100.0);
    h_pttr_sigptres_eta_ctf[i] = new TGraphErrors(25);

    TString hname = Form("pttr_pt_%d_ctf",i);
    TString htitle = Form("pT for %.1f GeV Single Muons",ptval[i]);
    h_pttr_pt_ctf[i] = new TH1F(hname, htitle, 100, xminpt[i], xmaxpt[i]);
    TString xtit = Form("pT %.1f GeV",ptval[i]);
    h_pttr_pt_ctf[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_pt_ctf[i]->GetYaxis()->SetTitle("events");
    
    hname = Form("pttr_eta_%d_ctf",i);
    htitle = Form("eta for %.1f GeV Single Muons",ptval[i]);
    h_pttr_eta_ctf[i] = new TH1F(hname, htitle, 100, -3.0, 3.0);
    xtit = Form("eta %.1f GeV",ptval[i]);
    h_pttr_eta_ctf[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_eta_ctf[i]->GetYaxis()->SetTitle("events");
    
    hname = Form("pttr_normchi_%d_ctf",i);
    htitle = Form("normalized chi2 for %.1f GeV Single Muons",ptval[i]);
    h_pttr_normchi_ctf[i] = new TH1F(hname, htitle, 100, 0.0, 5.0);
    xtit = Form("normalized chi2 %.1f GeV",ptval[i]);
    h_pttr_normchi_ctf[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_normchi_ctf[i]->GetYaxis()->SetTitle("events");

    hname = Form("pttr_d0_%d_ctf",i);
    htitle = Form("Impact parameter %.1f GeV Single Muons",ptval[i]);
    h_pttr_d0_ctf[i] = new TH1F(hname, htitle, 100, -0.02, 0.02);
    xtit = Form("d0 %.1f GeV",ptval[i]);
    h_pttr_d0_ctf[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_d0_ctf[i]->GetYaxis()->SetTitle("events");

    hname = Form("pttr_nhit_%d_ctf",i);
    htitle = Form("Number of hits for %.1f GeV Single Muons",ptval[i]);
    h_pttr_nhit_ctf[i] = new TH1F(hname, htitle, 25, 0.0, 25.0);
    xtit = Form("Hits %.1f GeV",ptval[i]);
    h_pttr_nhit_ctf[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_nhit_ctf[i]->GetYaxis()->SetTitle("events");

    hname = Form("pttr_phi_%d_ctf",i);
    htitle = Form("phi0 for %.1f GeV Single Muons",ptval[i]);
    h_pttr_phi_ctf[i] = new TH1F(hname, htitle, 100, -4.0, 4.0);
    xtit = Form("phi %.1f GeV",ptval[i]);
    h_pttr_phi_ctf[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_phi_ctf[i]->GetYaxis()->SetTitle("events");
    
    hname = Form("pttr_chi_%d_ctf",i);
    htitle = Form("chi2 for %.1f GeV Single Muons",ptval[i]);
    h_pttr_chi_ctf[i] = new TH1F(hname, htitle, 100, 0.0, 100.0);
    xtit = Form("chi %.1f GeV",ptval[i]);
    h_pttr_chi_ctf[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_chi_ctf[i]->GetYaxis()->SetTitle("events");
    
    hname = Form("pttr_etares_%d_ctf",i);
    htitle = Form("eta(rec)-eta(gen) for %.1f GeV Single Muons",ptval[i]);
    h_pttr_etares_ctf[i] = new TH1F(hname, htitle, 100, -0.005, 0.005);
    xtit = Form("eta resolution %.1f GeV",ptval[i]);
    h_pttr_etares_ctf[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_etares_ctf[i]->GetYaxis()->SetTitle("events");
    
    hname = Form("pttr_phires_%d_ctf",i);
    htitle = Form("phi(rec)-phi(gen) for %.1f GeV Single Muons",ptval[i]);
    h_pttr_phires_ctf[i] = new TH1F(hname, htitle, 100, -0.005, 0.005);
    xtit = Form("phi resolution %.1f GeV",ptval[i]);
    h_pttr_phires_ctf[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_phires_ctf[i]->GetYaxis()->SetTitle("events");
    
    hname = Form("pttr_ptres_%d_ctf",i);
    htitle = Form("pt(rec)-pt(gen) for %.1f GeV Single Muons",ptval[i]);
    h_pttr_ptres_ctf[i] = new TH1F(hname, htitle, 100, -1.0, 1.0);
    xtit = Form("pt resolution %.1f GeV",ptval[i]);
    h_pttr_ptres_ctf[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_ptres_ctf[i]->GetYaxis()->SetTitle("events");
    
    hname = Form("pttr_z0_%d_ctf",i);
    htitle = Form("z0 for %.1f GeV Single Muons",ptval[i]);
    h_pttr_z0_ctf[i] = new TH1F(hname,htitle,100,-0.05,0.05);
    xtit = Form("z0 for %.1f GeV Single Muons",ptval[i]);
    h_pttr_z0_ctf[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_z0_ctf[i]->GetYaxis()->SetTitle("events");
    
    hname = Form("pttr_hitseta_%d_ctf",i);
    htitle = Form("hits vs eta for %.1f GeV Single Muons",ptval[i]);
    h_pttr_hitseta_ctf[i] = new TProfile(hname, htitle, -54, -2.7, 2.7,0.0,20.0);
    xtit = Form("hits vs eta %.1f GeV",ptval[i]);
    h_pttr_hitseta_ctf[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_hitseta_ctf[i]->GetYaxis()->SetTitle("events");
    
    hname = Form("pttr_tracksfound_%d_ctf",i);
    htitle = Form("tracks/event for %.1f GeV Single Muons",ptval[i]);
    h_pttr_tracksfound_ctf[i] = new TH1F(hname, htitle, 10, 0.0, 10.0);
    xtit = Form("traks/event for %.1f GeV",ptval[i]);
    h_pttr_tracksfound_ctf[i]->GetXaxis()->SetTitle(xtit);
    h_pttr_tracksfound_ctf[i]->GetYaxis()->SetTitle("events");
  } // End Loop CTF Histograms

  //  histograms and variables that deal with all samples
  
  // smalleta and smallphi are holders for data used in efficiency calculations.
  // They are the eta and phi of tracks that were good matches with their corresponding generated tracks
  TH1F* h_pttr_smallphi = new TH1F("pttr_smallphi","Small phi bins",62,-3.14,3.14); 
  TH1F* h_pttr_smallphi_ctf = new TH1F("pttr_smallphi_ctf","Small phi bins",62,-3.14,3.14); 
  TH1F* h_pttr_smalleta = new TH1F("pttr_smalleta","Small eta bins",54,-2.7,2.7);
  TH1F* h_pttr_smalleta_ctf = new TH1F("pttr_smalleta_ctf","Small eta bins",54,-2.7,2.7);
 
  // genphi and geneta hold this generator level data
  TH1F* h_pttr_genphi = new TH1F("pttr_genphi","generated phi",62,-3.14.0,3.14);
  TH1F* h_pttr_geneta = new TH1F("pttr_geneta","generated eta",54,-2.7,2.7);
  TProfile* h_pttr_etahits = new TProfile("pttr_etahits","<N_{hits}> vs #eta",50,-2.5,2.5,0.0,20.0);
  TProfile* h_pttr_etahits_ctf = new TProfile("pttr_etahits_ctf","<N_{hits}> vs #eta",50,-2.5,2.5,0.0,20.0);
  
  TH1F* h_pttr_found_tracks = new TH1F("pttr_found_tracks","Number of tracks found",30,0.0,30.0);
  TH1F* h_pttr_found_tracks_ctf = new TH1F("pttr_found_tracks_ctf","Number of tracks found",30, 0.0,30.0);
  

  
  int found[8];
  int found_ctf[8];

  // Start filling things  

  float match_cut = 0.05; // change this to loosen or tighten the cut for efficiency calculations
  
  int tracks_found[npt];
  int tracks_found_ctf[npt];
  for(unsigned int j = 0; j < npt; ++j)  {
    tracks_found[j] = 0;
    tracks_found_ctf[j] = 0; 
  }


  // Loop Over pT points
  for  (unsigned int i = 0; i < npt; ++i)      {

    if (!usept[i]) continue;

    cout << myfile[i] << endl;

    TFile file(myfile[i]);
    TTree *tree = (TTree*)file.Get("Events");

    
    // CMSSW_0_9_0_pre3
    //      TBranch *branch     = tree->GetBranch("recoTracks_rsWithMaterialTracks__OneMu.obj");
    //      TBranch *branch_ctf = tree->GetBranch("recoTracks_ctfWithMaterialTracks__OneMu.obj");
    //      TBranch *branch_gen = tree->GetBranch("edmHepMCProduct_source__OneMu.obj");
    
    // CMSSW_1_0_0_pre4
    TBranch *branch_gen = tree->GetBranch("edmHepMCProduct_source__l.obj");
    TBranch *branch     = tree->GetBranch("recoTracks_rsWithMaterialTracks__e.obj");
    TBranch *branch_ctf = tree->GetBranch("recoTracks_ctfWithMaterialTracks__d.obj");

    
    std::vector<reco::Track> trackCollection;
    std::vector<reco::Track> trackCollection_ctf;
    edm::HepMCProduct evtProd;

    branch_gen->SetAddress(&evtProd);
    branch->SetAddress(&trackCollection);
    branch_ctf->SetAddress(&trackCollection_ctf);

    
    float eta_temp;
    float phi_temp;
    float gen_pt;
    
    cout << i << " " << tree->GetEntries() << endl;

    // Loop over the entries 
    for (unsigned int index = 0; index < tree->GetEntries(); ++index) {
      
      branch->GetEntry(index);
      branch_ctf->GetEntry(index);
      branch_gen->GetEntry(index);
      
      HepMC::GenEvent* event = evtProd->GetEvent();
      HepMC::GenEvent::particle_iterator pit;
      
      // Loop over generator level particles
      for( pit = event->particles_begin(); pit != event->particles_end(); ++pit )   {
	HepMC::GenParticle* Part = (*pit);
	CLHEP::HepLorentzVector h = Part->Momentum();
	h_pttr_genphi->Fill(h->phi());
	h_pttr_geneta->Fill(h->eta());
	gen_pt   = sqrt(h->x()*h->x() + h->y()*h->y());
      }


      float best_eta;
      float best_phi;
      float best_pt;
      float tracks = 0;
    

      // Loop over tracks in the event
      for ( unsigned int bindex = 0; bindex < trackCollection.size(); ++bindex )      { //RS
	
	reco::Track* track = (reco::Track*)trackCollection[bindex];
	h_pttr_hitseta[i]->Fill(track->eta(),track->found()); 
	// Hits should be on x axis
	h_pttr_normchi[i]->Fill(track->normalizedChi2());
	h_pttr_d0[i]->Fill(track->d0());
	h_pttr_nhit[i]->Fill(track->found());
	h_pttr_phi[i]->Fill(track->phi());
	h_pttr_eta[i]->Fill(track->eta());
	h_pttr_pt[i]->Fill(track->pt());
	h_pttr_chi[i]->Fill(track->chi2());
	h_pttr_z0[i]->Fill(track->dz());
	++tracks;
	
	// Get the Best Matched MC particle in eta/phi
	float best_match = 9999.0;
	for( pit = event->particles_begin(); pit != event->particles_end(); ++pit )   {
	  HepMC::GenParticle* Part = (*pit);
	  CLHEP::HepLorentzVector h = Part->Momentum();
	  float deta = track->eta()-h->eta();
	  float dphi = track->phi()-h->phi();
	  float match = deta*deta + dphi*dphi;
	  if(match < best_match)	  {
	    best_eta = h->eta();
	    best_phi = h->phi();
	    best_match = match;
	  }
	}

	if( best_match < match_cut ) {
	  h_pttr_smalleta->Fill(best_eta);
	  h_pttr_smallphi->Fill(best_phi);
	  ++tracks_found[i];
	}
	
	h_pttr_etares[i]->Fill((track->eta()-best_eta)/best_eta);
	h_pttr_phires[i]->Fill((track->phi()-best_phi)/best_phi);
	h_pttr_ptres[i]->Fill(track->pt()-gen_pt);
	h_pttr_found_tracks->Fill(ptval[i]);
	h_pttr_ptres_eta[i]->Fill(fabs(track->eta()),(track->pt()-ptval[i])/ptval[i]);

      } // End Loop over RS Tracks
      h_pttr_tracksfound[i]->Fill(tracks);

      tracks_found[i]=0;
      tracks = 0;
      // Loop over CTF Tracks
      for( unsigned int bindex = 0; bindex < trackCollection_ctf.size(); ++bindex )   { //CTF

	reco::Track* track = (reco::Track*)trackCollection_ctf[bindex];

	h_pttr_hitseta_ctf[i]->Fill(track->eta(),track->found());
	h_pttr_normchi_ctf[i]->Fill(track->normalizedChi2());
	h_pttr_d0_ctf[i]->Fill(track->d0());
	h_pttr_nhit_ctf[i]->Fill(track->found());
	h_pttr_phi_ctf[i]->Fill(track->phi());
	h_pttr_eta_ctf[i]->Fill(track->eta());
	h_pttr_pt_ctf[i]->Fill(track->pt());
	h_pttr_chi_ctf[i]->Fill(track->chi2());
	h_pttr_z0_ctf[i]->Fill(track->dz());

	// Get the Best Matched MC particle in eta/phi
	float best_match = 9999.0;
	for( pit = event->particles_begin(); pit != event->particles_end(); ++pit )   {
	  HepMC::GenParticle* Part = (*pit);
	  CLHEP::HepLorentzVector h = Part->Momentum();
	  float deta = track->eta()-h->eta();
	  float dphi = track->phi()-h->phi();
	  float match = deta*deta + dphi*dphi;
	  if(match < best_match)	  {
	    best_eta = h->eta();
	    best_phi = h->phi();
	    best_match = match;
	  }
	}

	if( best_match < match_cut ) {
	  h_pttr_smalleta->Fill(best_eta);
	  h_pttr_smallphi->Fill(best_phi);
	  ++tracks_found[i];
	}

	h_pttr_etares_ctf[i]->Fill((track->eta()-best_eta)/best_eta);
	h_pttr_phires_ctf[i]->Fill((track->phi()-best_phi)/best_phi);
	h_pttr_ptres_ctf[i]->Fill(track->pt()-gen_pt);
	h_pttr_found_tracks_ctf->Fill(ptval[i]);
	h_pttr_ptres_eta_ctf[i]->Fill(fabs(track->eta()),(track->pt()-ptval[i])/ptval[i]);
      }
      h_pttr_tracksfound_ctf[i]->Fill(tracks);

    }

    found[i] = h_pttr_found_tracks->GetBinContent(ptval[i]);
    found_ctf[i] = h_pttr_found_tracks_ctf->GetBinContent(ptval[i]);

  //  TGraph *found_tracks = new TGraph(npt,ptval,found); 
  //  TGraph *found_tracks_ctf = new TGraph(npt,ptval,found_ctf);

    //eta, RS
    TFormula *f1 = new TFormula("fitgaus", "gaus(0)");
    TF1 *f2 = new TF1("f2","fitgaus");
    float ymax = h_pttr_etares[i]->GetMaximum();
    float mean = h_pttr_etares[i]->GetMean();
    float rms  = h_pttr_etares[i]->GetRMS();
    f2->SetParameters(ymax/3.0,mean,rms);
    f2->SetParLimits(0,0.0,0.0);
    h_pttr_etares[i]->Fit("f2","0");  
    eta_resolutions[i] = f2->GetParameter(2);
    eta_resolutions_error[i] = f2->GetParError(2);
    
    //eta, CTF
    
    ymax = h_pttr_etares_ctf[i]->GetMaximum();
    mean = h_pttr_etares_ctf[i]->GetMean();
    rms  = h_pttr_etares_ctf[i]->GetRMS();
    f2->SetParameters(ymax/3.0,mean,rms);
    f2->SetParLimits(0,0.0,0.0);
    h_pttr_etares_ctf[i]->Fit("f2","0"); 
    eta_resolutions_ctf[i] = f2->GetParameter(2);
    eta_resolutions_error_ctf[i] = f2->GetParError(2);
 
    //phi, RS

    ymax = h_pttr_phires[i]->GetMaximum();
    mean = h_pttr_phires[i]->GetMean();
    rms  = h_pttr_phires[i]->GetRMS();
    f2->SetParameters(ymax/3.0,mean,rms);
    f2->SetParLimits(0,0.0,0.0);
    h_pttr_phires[i]->Fit("f2","0");
    phi_resolutions[i] = f2->GetParameter(2);
    phi_resolutions_error[i] = f2->GetParError(2);
  
    //phi, CTF

    ymax = h_pttr_phires_ctf[i]->GetMaximum();
    mean = h_pttr_phires_ctf[i]->GetMean();
    rms  = h_pttr_phires_ctf[i]->GetRMS();
    f2->SetParameters(ymax/3.0,mean,rms);
    f2->SetParLimits(0,0.0,0.0 );
    h_pttr_phires_ctf[i]->Fit("f2","0");
    phi_resolutions_ctf[i] = f2->GetParameter(2);
    phi_resolutions_error_ctf[i] = f2->GetParError(2);

    //pT, RS

    ymax = h_pttr_pt[i]->GetMaximum();
    mean = h_pttr_pt[i]->GetMean();
    rms  = h_pttr_pt[i]->GetRMS();
    f2->SetParameters(ymax/3.0,mean,rms); 
    f2->SetParLimits(0,0.0,0.0);
    h_pttr_pt[i]->Fit("f2","0");
    pt_resolutions[i] = f2->GetParameter(2)/ptval[i];
    pt_resolutions_error[i] = f2->GetParError(2)/ptval[i];

    //pT, CTF

    ymax = h_pttr_pt_ctf[i]->GetMaximum();
    mean = h_pttr_pt_ctf[i]->GetMean();
    rms  = h_pttr_pt_ctf[i]->GetRMS();
    f2->SetParameters(ymax/3.0,mean,rms);
    f2->SetParLimits(0,0.0,0.0);
    h_pttr_pt_ctf[i]->Fit("f2","0");
    pt_resolutions_ctf[i] = f2->GetParameter(2)/ptval[i];
    pt_resolutions_error_ctf[i] = f2->GetParError(2)/ptval[i];
    
    //d0, RS
    
    ymax = h_pttr_d0[i]->GetMaximum();
    mean = h_pttr_d0[i]->GetMean();
    rms  = h_pttr_d0[i]->GetRMS();
    f2->SetParameters(ymax/3.0,mean,rms); 
    f2->SetParLimits(0,0.0,0.0);
    h_pttr_d0[i]->Fit("f2","0");
    d0_resolutions[i] = f2->GetParameter(2)/ptval[i];
    d0_resolutions_error[i] = f2->GetParError(2)/ptval[i];
    
    //d0, CTF
    
    ymax = h_pttr_d0_ctf[i]->GetMaximum();
    mean = h_pttr_d0_ctf[i]->GetMean();
    rms  = h_pttr_d0_ctf[i]->GetRMS();
    f2->SetParameters(ymax/3.0,mean,rms); 
    f2->SetParLimits(0,0.0,0.0);
    h_pttr_d0_ctf[i]->Fit("f2","0");
    d0_resolutions_ctf[i] = f2->GetParameter(2)/ptval[i];
    d0_resolutions_error_ctf[i] = f2->GetParError(2)/ptval[i];
    
    //z0, RS
    
    ymax = h_pttr_z0[i]->GetMaximum();
    mean = h_pttr_z0[i]->GetMean();
    rms  = h_pttr_z0[i]->GetRMS();
    f2->SetParameters(ymax/3.0,mean,rms); 
    f2->SetParLimits(0,0.0,0.0);
    h_pttr_z0[i]->Fit("f2","0");
    z0_resolutions[i] = f2->GetParameter(2)/ptval[i];
    z0_resolutions_error[i] = f2->GetParError(2)/ptval[i];
    
    //z0, CTF
    
    ymax = h_pttr_z0_ctf[i]->GetMaximum();
    mean = h_pttr_z0_ctf[i]->GetMean();
    rms  = h_pttr_z0_ctf[i]->GetRMS();
    f2->SetParameters(ymax/3.0,mean,rms); 
    f2->SetParLimits(0,0.0,0.0);
    h_pttr_z0_ctf[i]->Fit("f2","0");
    z0_resolutions_ctf[i] = f2->GetParameter(2)/ptval[i];
    z0_resolutions_error_ctf[i] = f2->GetParError(2)/ptval[i];
  }


  fil1->cd();

  for (unsigned int i=0; i<npt; i++) {
    if (!usept[i]) continue;
    for (int k=0; k<25; k++) {
      int nentries = h_pttr_ptres_eta[i]->GetBinEntries(k);
      Double_t xpoint = h_pttr_ptres_eta[i]->GetBinCenter(k);
      Double_t mean1 = h_pttr_ptres_eta[i]->GetBinContent(k);
      Double_t sigma = h_pttr_ptres_eta[i]->GetBinError(k);
      Double_t sigerr = nentries>0?sigma/sqrt(nentries):0;
      cout << k << " " << xpoint << " " <<  mean1 << " " << sigma << endl;
      double y1 = sigma;
      TString hname = Form("sigptres_vs_eta_%d",i);
      h_pttr_sigptres_eta[i]->SetPoint(k,xpoint,y1);
      h_pttr_sigptres_eta[i]->SetPointError(k,0.0,sigerr);
      h_pttr_sigptres_eta[i]->SetName(hname);
    }
    h_pttr_sigptres_eta[i]->Write();
  }


  for (unsigned int i=0; i<npt; i++) {
    if (!usept[i]) continue;
    for (int k=0; k<25; k++) {
      int nentries = h_pttr_ptres_eta_ctf[i]->GetBinEntries(k);
      Double_t xpoint = h_pttr_ptres_eta_ctf[i]->GetBinCenter(k);
      Double_t mean1 = h_pttr_ptres_eta_ctf[i]->GetBinContent(k);
      Double_t sigma = h_pttr_ptres_eta_ctf[i]->GetBinError(k);
      Double_t sigerr = nentries>0?sigma/sqrt(nentries):0;
      cout << k << " " << xpoint << " " <<  mean1 << " " << sigma << endl;
      double y1 = sigma;
      TString hname = Form("sigptres_vs_eta_ctf_%d",i);
      h_pttr_sigptres_eta_ctf[i]->SetPoint(k,xpoint,y1);
      h_pttr_sigptres_eta_ctf[i]->SetPointError(k,0.0,sigerr);
      h_pttr_sigptres_eta_ctf[i]->SetName(hname);
    }
    h_pttr_sigptres_eta_ctf[i]->Write();
  }


  TGraphErrors* etares_vs_pt      = new TGraphErrors(npt, ptval, eta_resolutions, 0, eta_resolutions_error);
  TGraphErrors* etares_vs_pt_ctf  = new TGraphErrors(npt, ptval, eta_resolutions_ctf, 0, eta_resolutions_error_ctf);
  TGraphErrors* phires_vs_pt      = new TGraphErrors(npt, ptval, phi_resolutions, 0, phi_resolutions_error);
  TGraphErrors* phires_vs_pt_ctf  = new TGraphErrors(npt, ptval, phi_resolutions_ctf, 0, phi_resolutions_error_ctf);
  TGraphErrors* ptres_vs_pt       = new TGraphErrors(npt, ptval, pt_resolutions, 0, pt_resolutions_error);
  TGraphErrors* ptres_vs_pt_ctf   = new TGraphErrors(npt, ptval, pt_resolutions_ctf, 0, pt_resolutions_error_ctf);
  TGraphErrors* d0res_vs_pt       = new TGraphErrors(npt,ptval,d0_resolutions,0,d0_resolutions_error);
  TGraphErrors* d0res_vs_pt_ctf   = new TGraphErrors(npt,ptval,d0_resolutions_ctf,0,d0_resolutions_error_ctf);
  TGraphErrors* z0res_vs_pt       = new TGraphErrors(npt,ptval,z0_resolutions,0,z0_resolutions_error);
  TGraphErrors* z0res_vs_pt_ctf   = new TGraphErrors(npt,ptval,z0_resolutions_ctf,0,z0_resolutions_error_ctf);


  etares_vs_pt->SetName("etaresvspt");
  etares_vs_pt->Write();
  etares_vs_pt_ctf->SetName("etaresvspt_ctf");
  etares_vs_pt_ctf->Write();

  phires_vs_pt->SetName("phiresvspt");
  phires_vs_pt->Write();
  phires_vs_pt_ctf->SetName("phiresvspt_ctf");
  phires_vs_pt_ctf->Write();


  ptres_vs_pt->SetName("ptresvspt");
  ptres_vs_pt->Write();
  ptres_vs_pt_ctf->SetName("ptresvspt_ctf");
  ptres_vs_pt_ctf->Write();

  d0res_vs_pt->SetName("d0resvspt");
  d0res_vs_pt->Write();
  d0res_vs_pt_ctf->SetName("d0resvspt_ctf");
  d0res_vs_pt_ctf->Write();

  z0res_vs_pt->SetName("z0resvspt");
  z0res_vs_pt->Write();
  z0res_vs_pt_ctf->SetName("z0resvspt_ctf");
  z0res_vs_pt_ctf->Write();

  fil1->Write();
  fil1->Close();

}
