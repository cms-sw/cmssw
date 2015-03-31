{
gSystem->Load("libFWCoreFWLite.so");
gSystem->Load("libValidationRecoParticleFlow.so");

TF2 gaus2("gaus2", "[0]*exp(-0.5*((x-[1])/[2])**2)*exp(-0.5*((y-[3])/[4])**2)",0,10,0,10);
gaus2.SetParameters( 100, 5, 5, 2, 2);
gaus2.Draw("colz");

TH2D h("h", "", 10,0,10,100,0,10);
h.FillRandom("gaus2");
h.Draw("col");

TH2Analyzer hana(&h);

}
