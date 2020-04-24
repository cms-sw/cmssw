gSystem->Load("libFWCoreFWLite");

FWLiteEnabler::enable();

// reco file with vertices
TFile f("pv_reco.root");
TBrowser a;

Events->SetAlias("pv", "recoVertexs_offlinePrimaryVerticesFromCTFTracks__Demo.obj");

// position distributions
Events->Draw("pv.x()");
Events->Draw("pv.z()");

// error on x
Events->Draw("sqrt(pv.covariance(0, 0))");

// pull on x
Events->Draw("pv.x()/sqrt(pv.covariance(0, 0))");
