{
  // book histograms
  TFile histofile("track_hists.root","RECREATE");
  TH1F* h_chisqtr = new TH1F("chisqtr","Track chisq",100,0.,10.);
  TH1F* h_pttr = new TH1F("pttr","Track pT (GeV)",100,8.0,12.0);
  TH1F* h_nhittr = new TH1F("nhittr","Number of Hits",31,-0.5,30.5);

  TTree *tree = (TTree*)file.Get("Events");

  std::vector<reco::Track> trackCollection;

  TBranch *branch = tree->GetBranch("recoTracks_generalTracks__RECO.obj");
  branch->SetAddress(&trackCollection);

  for ( unsigned int index = 0; index < tree->GetEntries(); ++index ) {
    std::cout << "Event: " << index << std::endl;
    branch->GetEntry(index);
    for ( unsigned int bindex = 0; bindex < trackCollection.size(); ++bindex ) {
      reco::Track* track = (reco::Track*)trackCollection[bindex];
      h_chisqtr->Fill(track->normalizedChi2());
      double pT = sqrt(track->px()*track->px()+track->py()*track->py());
      h_pttr->Fill(pT);

      h_nhittr->Fill(track->found());
    }
  }
  
  // save histograms
  histofile.Write();
  histofile.Close();
}
