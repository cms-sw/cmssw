{
  // book histograms
  TFile histofile("HZZ.root","RECREATE");
  TH1F* h_chisqtr = new TH1F("chisqtr","Track chisq",100,0.,40.);
  TH1F* h_pttr = new TH1F("pttr","Track pT (GeV)",100,0.0,200.0);
  TH1F* h_pxtr = new TH1F("pxtr","Track px (GeV)",100,0.0,200.0);
  TH1F* h_pytr = new TH1F("pytr","Track py (GeV)",100,0.0,200.0);
  TH1F* h_pztr = new TH1F("pztr","Track pz (GeV)",100,0.0,200.0);
  TH1F* h_etatr = new TH1F("etatr","Track#eta",60,-3.0,3.0);
  TH1F* h_chargetr = new TH1F("chargetr","Track charge",6,-3.0,3.0);
  TH1F* h_mmumu = new TH1F("mmumu","mu+mu- mass (GeV)",100,0.0,200.0);
  TH1F* h_noz = new TH1F("noz","num of reco Z",10,0.0,10.0);
  TH1F* h_mzz = new TH1F("mzz","Z0Z0 mass (GeV)",100,100.0,300.0);
  TH1F* h_m4mu = new TH1F("m4mu","2mu+2mu- mass (GeV)",100,100.0,300.0);

  TTree *tree = (TTree*)file.Get("Events");

  std::vector<reco::Track> trackCollection;

  TBranch *branch = tree->GetBranch("recoTracks_generalTracks__RECO.obj");
  branch->SetAddress(&trackCollection);

  for ( unsigned int index = 0; index < tree->GetEntries(); ++index ) {
    std::cout << "Event: " << index << std::endl;
    branch->GetEntry(index);
    double px4=0.0, py4=0.0, pz4=0.0, e4=0.0;
    int q4=0, n4=0;
    for ( unsigned int bindex = 0; bindex < trackCollection.size(); ++bindex ) {
      reco::Track* track = (reco::Track*)trackCollection[bindex];
      h_chisqtr->Fill(track->normalizedChi2());
      double pT = sqrt(track->px()*track->px()+track->py()*track->py());
      h_pttr->Fill(pT);
      h_pxtr->Fill(track->px());
      h_pytr->Fill(track->py());
      h_pztr->Fill(track->pz());
      h_etatr->Fill(track->eta());
      h_chargetr->Fill(track->charge());
      n4++; 
      if (track->charge()>0.0){q4++;}else{q4--;}
      px4+=track->px(); 
      py4+=track->py(); 
      pz4+=track->pz(); 
      e4+=sqrt(track->px()*track->px()+track->py()*track->py()
              +track->pz()*track->pz()+0.011163691);
    }
    double ptot = sqrt( px4*px4 + py4*py4 + pz4*pz4 );
    double mz = sqrt((e4+ptot)*(e4-ptot));
    if ((4==n4)&&(0==q4))h_m4mu->Fill(mz);
    std::vector<double> Zpx;
    std::vector<double> Zpy;
    std::vector<double> Zpz;
    std::vector<double> Ze;
    std::vector<int> Zpart1;
    std::vector<int> Zpart2;
    if (trackCollection.size() >1){
      for ( unsigned int bindex = 0; bindex < trackCollection.size()-1; ++bindex ) {
        reco::Track* track1 = (reco::Track*)trackCollection[bindex];
        for ( unsigned int cindex = bindex+1; cindex < trackCollection.size(); ++cindex ) {
          reco::Track* track2 = (reco::Track*)trackCollection[cindex];
          if (track1->charge()*track2->charge() < 0.0){
            double e1 = sqrt((track1->px()*track1->px())
                            +(track1->py()*track1->py())
                            +(track1->pz()*track1->pz())+0.011163691);
            double e2 = sqrt((track2->px()*track2->px())
                            +(track2->py()*track2->py())
                            +(track2->pz()*track2->pz())+0.011163691);
            double etot = e1+e2;
            double pxtot = track1->px()+track2->px();
            double pytot = track1->py()+track2->py();
            double pztot = track1->pz()+track2->pz();
            double ptot = sqrt( pxtot*pxtot + pytot*pytot + pztot*pztot );
            double mz = sqrt((etot+ptot)*(etot-ptot));
            h_mmumu->Fill(mz);
            if ((mz>80.0)&&(mz<100.0)){
              Zpx.push_back(pxtot); Zpy.push_back(pytot); Zpz.push_back(pztot); 
              Ze.push_back(etot); Zpart1.push_back(bindex); Zpart2.push_back(cindex); 
            }
          }//tracks opposite charge
        }//end track2
      }//end track1
    }//end got >1 trk
    h_noz->Fill( Zpx.size() );
    if (Zpx.size() >1){
      for ( unsigned int bindex = 0; bindex < Zpx.size()-1; ++bindex ) {
        for ( unsigned int cindex = bindex+1; cindex < Zpx.size(); ++cindex ) {
          if ((Zpart1[bindex]!=Zpart1[cindex])&&(Zpart1[bindex]!=Zpart2[cindex])
            &&(Zpart2[bindex]!=Zpart1[cindex])&&(Zpart2[bindex]!=Zpart2[cindex])){
              double etot = Ze[bindex]+Ze[cindex];
              double pxtot = Zpx[bindex]+Zpx[cindex];
              double pytot = Zpy[bindex]+Zpy[cindex];
              double pztot = Zpz[bindex]+Zpz[cindex];
              double ptot = sqrt( pxtot*pxtot + pytot*pytot + pztot*pztot );
              double mh = sqrt((etot+ptot)*(etot-ptot));
              h_mzz->Fill(mh);
          }
        }
      }
    }
  }//end evt loop
  

  // save histograms
  histofile.Write();
  histofile.Close();
}
