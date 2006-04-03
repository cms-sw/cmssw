{

  TTree *tree = (TTree*)file.Get("Events");

  std::vector<reco::Track> trackCollection;

  TBranch *branch = tree->GetBranch("recoTracks_RoadSearchDebugTracks__RoadSearch.obj");
  branch->SetAddress(&trackCollection);

  for ( unsigned int index = 0; index < tree->GetEntries(); ++index ) {
    std::cout << "index: " << index << std::endl;
    branch->GetEntry(index);
    std::cout << "content: " << trackCollection.size() << std::endl;
  }
  

}
