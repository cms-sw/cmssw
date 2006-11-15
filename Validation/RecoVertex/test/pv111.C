gSystem->Load("libFWCoreFWLite");

AutoLibraryLoader::enable();

// reco file with vertices
TFile f("pv_reco.root");
TBrowser a;

Events->SetAlias("pv", "recoVertexs_offlinePrimaryVerticesFromCTFTracks__Demo.obj");

Events->Draw("pv.x()");
Events->Draw("pv.z()");

// error on x
Events->Draw("sqrt(pv.covariance(0, 0))");

// pull on z
Events->Draw("pv.z()/sqrt(pv.covariance(2, 2))");
