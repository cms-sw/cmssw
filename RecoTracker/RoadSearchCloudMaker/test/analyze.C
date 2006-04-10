{
  // book histograms
  TFile histofile("track_hists.root","RECREATE");
  TH1F* h_chisqtr = new TH1F("chisqtr","Track chisq",100,0.,40.);
  TH1F* h_pttr = new TH1F("pttr","Track pT (GeV)",100,8.0,12.0);

  TTree *tree = (TTree*)file.Get("Events");

  std::vector<reco::Track> trackCollection;

  TBranch *branch = tree->GetBranch("recoTracks_RoadSearchDebugTracks__RoadSearch.obj");
//  TBranch *branch = tree->GetBranch("recoTracks_trackp__RoadSearch.obj");
  branch->SetAddress(&trackCollection);

  for ( unsigned int index = 0; index < tree->GetEntries(); ++index ) {
    std::cout << "index: " << index << std::endl;
    branch->GetEntry(index);
    std::cout << "content: " << trackCollection.size() << std::endl;
    for ( unsigned int bindex = 0; bindex < trackCollection.size(); ++bindex ) {
      std::cout << "track: " << bindex << std::endl;
      reco::Track* track = (reco::Track*)trackCollection[bindex];
      h_chisqtr->Fill(track->chi2());
      double pT = sqrt(track->px()*track->px()+track->py()*track->py());
      std::cout << "pT: " << pT << std::endl;
      h_pttr->Fill(pT);
    }
  }
  

  // save histograms
  histofile.Write();
  histofile.Close();
}
