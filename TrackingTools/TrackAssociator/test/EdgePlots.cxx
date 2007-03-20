{
   gStyle->SetPalette(1,0);
   TFile f("/afs/cern.ch/user/d/dmytro/scratch0/muons.root");
   TTree* tree = (TTree*)f.Get("Events");
   TString name = "edges";

   TCanvas* c1 = new TCanvas(name,name,600,900);
   c1->Divide(2,4);
   TH1F* hRecoXCSC = new TH1F(name+"_CSC_recoX",name+"_CSC_recoX",110,-100,10);
   TH1F* hTrueXCSC = new TH1F(name+"_CSC_trueX",name+"_CSC_trueX",110,-100,10);
   TH1F* hRecoYCSC = new TH1F(name+"_CSC_recoY",name+"_CSC_recoY",110,-100,10);
   TH1F* hTrueYCSC = new TH1F(name+"_CSC_trueY",name+"_CSC_trueY",110,-100,10);
   TH1F* hRecoXDT = new TH1F(name+"_DT_recoX",name+"_DT_recoX",110,-210,10);
   TH1F* hTrueXDT = new TH1F(name+"_DT_trueX",name+"_DT_trueX",110,-210,10);
   TH1F* hRecoYDT = new TH1F(name+"_DT_recoY",name+"_DT_recoY",140,-270,10);
   TH1F* hTrueYDT = new TH1F(name+"_DT_trueY",name+"_DT_trueY",140,-270,10);

   // create and connect muon collection branch 

   tree->SetBranchStatus("*",0);
   tree->SetBranchStatus("recoMuonWithMatchInfos_test_muons_TEST.obj*",1);
   std::vector<reco::MuonWithMatchInfo> muons;
   
   TString branchName = tree->GetAlias("muons");
   tree->SetBranchAddress(branchName,&muons);

   for ( unsigned int index = 0; index < tree->GetEntries(); ++index ) {
      tree->GetEntry(index);
      tree->SetBranchAddress(branchName,&muons);
      if (index%1000==0) std::cout << "Event " << index << " has " << muons.size() << " muons" << std::endl;
      for(unsigned int i=0; i<muons.size(); i++)
	{
	   const std::vector<reco::MuonWithMatchInfo::MuonChamberMatch>& matches = muons[i].matches();
	   if ( matches.empty() || matches.front().segmentMatches.empty() ) continue;
	   
	   bool fillReco = false;
	   bool fillTrue = false;
	   // std::cout << "#segments: " << matches.front().segmentMatches.front().x << std::endl;
	   // std::cout << "#segments: " << muons.front().matches().front().segmentMatches.size() << std::endl;
	   for ( unsigned int j = 0; j <  matches.front().segmentMatches.size(); j++)
	     {
		// std::cout << "\txErr: " << matches.front().segmentMatches[j].xErr << std::endl;
		if (matches.front().segmentMatches[j].xErr < -100)
		  fillTrue = true;
		else
		  fillReco = true;
	     }
	   
	   if (fillReco) {
	      if ( matches.front().id.subdetId() == 2 ) { //CSC
		 hRecoXCSC->Fill(matches.front().edgeX);
		 hRecoYCSC->Fill(matches.front().edgeY);
		 //  std::cout << matches.front().edgeX << std::endl;
	      }else{
		 hRecoXDT->Fill(matches.front().edgeX);
		 hRecoYDT->Fill(matches.front().edgeY);
	      }
	   }
	   
	   if (fillTrue) {
	      if ( matches.front().id.subdetId() == 2 ) { //CSC
		 hTrueXCSC->Fill(matches.front().edgeX);
		 hTrueYCSC->Fill(matches.front().edgeY);
		 //  std::cout << matches.front().edgeX << std::endl;
	      }else{
		 hTrueXDT->Fill(matches.front().edgeX);
		 hTrueYDT->Fill(matches.front().edgeY);
	      }
	   }
	}
   }
   
   c1->cd(1);
   hRecoXDT->Draw();
   c1->cd(2);
   hRecoYDT->Draw();
   c1->cd(3);
   hTrueXDT->Draw();
   c1->cd(4);
   hTrueYDT->Draw();
   c1->cd(5);
   hRecoXCSC->Draw();
   c1->cd(6);
   hRecoYCSC->Draw();
   c1->cd(7);
   hTrueXCSC->Draw();
   c1->cd(8);
   hTrueYCSC->Draw();
   
   //return c1;
}
