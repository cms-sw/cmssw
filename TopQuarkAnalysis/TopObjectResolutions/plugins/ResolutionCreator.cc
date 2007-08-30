#include "TopQuarkAnalysis/TopObjectResolutions/interface/ResolutionCreator.h"

//
// constants, enums and typedefs
//

//
// static data member definitions
//

    

//
// constructors and destructor
//
ResolutionCreator::ResolutionCreator(const edm::ParameterSet& iConfig)
{
  // input parameters
  objectType_  	= iConfig.getParameter< std::string >    	("object");
  labelName_  = iConfig.getParameter< std::string > 	 	("label");
  if(objectType_ != "met"){
    etabinVals_	= iConfig.getParameter< std::vector<double> > 	("etabinValues");
  }
  eTbinVals_	= iConfig.getParameter< std::vector<double> > 	("eTbinValues");
  minDR_	= iConfig.getParameter< double >         	("minMatchingDR");


  // input constants  
  TString  	  resObsName[8] 	= {"_ares","_bres","_cres","_dres","_thres","_phres","_etres","_etares"};
  int      	  resObsNrBins  	= 120;
  if( (objectType_ == "muon") || (objectType_ == "electron") ) resObsNrBins = 80;
  std::vector<double>  resObsMin, resObsMax;
  if(objectType_ == "electron"){ 
    resObsMin.push_back(-0.15);  resObsMin.push_back(-0.2);  resObsMin.push_back(-0.1);  resObsMin.push_back(-0.15);  resObsMin.push_back(-0.0012); resObsMin.push_back(-0.009);  resObsMin.push_back(-16);   resObsMin.push_back(-0.0012);   
    resObsMax.push_back( 0.15);  resObsMax.push_back( 0.2);  resObsMax.push_back( 0.1);  resObsMax.push_back( 0.15);  resObsMax.push_back( 0.0012); resObsMax.push_back( 0.009);  resObsMax.push_back( 16);   resObsMax.push_back( 0.0012);
  } else if(objectType_ == "muon"){
    resObsMin.push_back(-0.15);  resObsMin.push_back(-0.1);  resObsMin.push_back(-0.05);  resObsMin.push_back(-0.15);  resObsMin.push_back(-0.004);  resObsMin.push_back(-0.003);  resObsMin.push_back(-8);    resObsMin.push_back(-0.004);   
    resObsMax.push_back( 0.15);  resObsMax.push_back( 0.1);  resObsMax.push_back( 0.05);  resObsMax.push_back( 0.15);  resObsMax.push_back( 0.004);  resObsMax.push_back( 0.003);  resObsMax.push_back( 8);    resObsMax.push_back( 0.004);
  } else if(objectType_ == "tau"){ 
    resObsMin.push_back(-1.);    resObsMin.push_back(-10.);  resObsMin.push_back(-10);   resObsMin.push_back(-1.);   resObsMin.push_back(-0.1);    resObsMin.push_back(-0.1);    resObsMin.push_back(-80);   resObsMin.push_back(-0.1);   
    resObsMax.push_back( 1.);    resObsMax.push_back( 10.);  resObsMax.push_back( 10);   resObsMax.push_back( 1.);   resObsMax.push_back( 0.1);    resObsMax.push_back( 0.1);    resObsMax.push_back( 50);   resObsMax.push_back( 0.1);
  } else if(objectType_ == "lJets" || objectType_ == "bJets"){
    resObsMin.push_back(-1.);    resObsMin.push_back(-10.);  resObsMin.push_back(-10.);  resObsMin.push_back(-1.);   resObsMin.push_back(-0.4);    resObsMin.push_back(-0.6);    resObsMin.push_back( -80);  resObsMin.push_back(-0.6);   
    resObsMax.push_back( 1.);    resObsMax.push_back( 10.);  resObsMax.push_back( 10.);  resObsMax.push_back( 1.);   resObsMax.push_back( 0.4);    resObsMax.push_back( 0.6);    resObsMax.push_back( 80);   resObsMax.push_back( 0.6);
  } else{
    resObsMin.push_back(-2.);   resObsMin.push_back(-150.); resObsMin.push_back(-150.); resObsMin.push_back(-2.);   resObsMin.push_back(-6);      resObsMin.push_back(-6);      resObsMin.push_back( -180); resObsMin.push_back(-6);   
    resObsMax.push_back( 3.);   resObsMax.push_back( 150.); resObsMax.push_back( 150.); resObsMax.push_back( 3.);   resObsMax.push_back( 6);      resObsMax.push_back( 6);      resObsMax.push_back(  180); resObsMax.push_back( 6);
  }
  
  const char*   resObsVsEtFit[8]    	= {"[0]+[1]*exp(-[2]*x)",
                                           "[0]+[1]*exp(-[2]*x)",
                                           "[0]+[1]*exp(-[2]*x)",
                                           "[0]+[1]*exp(-[2]*x)",
					   "[0]+[1]*exp(-[2]*x)",
					   "[0]+[1]*exp(-[2]*x)",
					   "[0]+[1]*exp(-[2]*x)",
					   "[0]+[1]*exp(-[2]*x)"
					  };
 
  etnrbins        = eTbinVals_.size()-1;
  double *etbins  = new double[eTbinVals_.size()];
  for(unsigned int b=0; b<eTbinVals_.size(); b++)  etbins[b]  = eTbinVals_[b];
  double *etabins;
  if(objectType_ != "met"){
    etanrbins  = etabinVals_.size()-1;
    etabins    = new double[etabinVals_.size()];
    for(unsigned int b=0; b<etabinVals_.size(); b++) etabins[b] = etabinVals_[b];
  }
  else
  {
    etanrbins = 1;
    etabins    = new double[2];
    etabins[0] = 0; etabins[1] = 5.;
  }
  TString outputFileName = "Resolutions_"; outputFileName += objectType_;
  if(objectType_ == "lJets" || objectType_ == "bJets") { outputFileName += "_"; outputFileName += labelName_; }; 
  outputFileName += ".root"; 
  outfile = new TFile(outputFileName, "RECREATE");
  
  for(Int_t ro=0; ro<8; ro++) {
    for(Int_t etab=0; etab<etanrbins; etab++) {	
      for(Int_t etb=0; etb<etnrbins; etb++) {
        TString obsName = objectType_; obsName += resObsName[ro]; obsName += "_etabin"; obsName += etab; obsName += "_etbin"; obsName += etb;
	hResEtEtaBin[ro][etab][etb] = new TH1F(obsName,obsName,resObsNrBins,resObsMin[ro],resObsMax[ro]);
        fResEtEtaBin[ro][etab][etb] = new TF1("F_"+obsName,"gaus");
      }
      TString obsName2 = objectType_; obsName2 += resObsName[ro]; obsName2 += "_etabin"; obsName2 += etab;
      hResEtaBin[ro][etab] = new TH1F(obsName2,obsName2,etnrbins,etbins);
      fResEtaBin[ro][etab] = new TF1("F_"+obsName2,resObsVsEtFit[ro],eTbinVals_[0],eTbinVals_[eTbinVals_.size()-1]);
    }
  }
  hEtaBins = new TH1F("hEtaBins","hEtaBins",etanrbins,etabins);
  delete [] etabins; 
  delete [] etbins; 
  
  nrFilled = 0;

}


ResolutionCreator::~ResolutionCreator()
{
  outfile->cd();
  Int_t ro=0;
  Double_t et=0.;
  Double_t eta=0.;
  Double_t value,error;
  // CD: create the output tree
  TTree* tResVar = new TTree("tResVar","Resolution tree");
  tResVar->Branch("Et",&et,"Et/D");
  tResVar->Branch("Eta",&eta,"Eta/D");
  tResVar->Branch("ro",&ro,"ro/I");
  tResVar->Branch("value",&value,"value/D");
  tResVar->Branch("error",&error,"error/D");
  
  for(ro=0; ro<8; ro++) {
    for(int etab=0; etab<etanrbins; etab++) {	
      //CD set eta at the center of the bin
      eta = etanrbins > 1 ? (etabinVals_[etab]+etabinVals_[etab+1])/2. : 2.5 ; 
      for(int etb=0; etb<etnrbins; etb++) {
	//CD set et at the center of the bin
	et = (eTbinVals_[etb]+eTbinVals_[etb+1])/2.; 
        double maxcontent = 0.;
	int maxbin = 0;
	for(int nb=1; nb<hResEtEtaBin[ro][etab][etb]->GetNbinsX(); nb ++){
	  if (hResEtEtaBin[ro][etab][etb]->GetBinContent(nb)>maxcontent) {
	    maxcontent = hResEtEtaBin[ro][etab][etb]->GetBinContent(nb);
	    maxbin = nb;
	  }
	}
	int range = (int)(hResEtEtaBin[ro][etab][etb]->GetNbinsX()/6); //in order that ~1/3 of X-axis range is fitted
        fResEtEtaBin[ro][etab][etb] -> SetRange(hResEtEtaBin[ro][etab][etb]->GetBinCenter(maxbin-range),
	 					hResEtEtaBin[ro][etab][etb]->GetBinCenter(maxbin+range));
	fResEtEtaBin[ro][etab][etb] -> SetParameters(hResEtEtaBin[ro][etab][etb] -> GetMaximum(),
	                                             hResEtEtaBin[ro][etab][etb] -> GetMean(),
	 					     hResEtEtaBin[ro][etab][etb] -> GetRMS());
	hResEtEtaBin[ro][etab][etb] -> Fit(fResEtEtaBin[ro][etab][etb]->GetName(),"RQ");
        //hResEtEtaBin[ro][etab][etb] -> Fit(fResEtEtaBin[ro][etab][etb]->GetName(),"QL");
        hResEtEtaBin[ro][etab][etb] -> Write();
        hResEtaBin[ro][etab]        -> SetBinContent(etb+1,fResEtEtaBin[ro][etab][etb]->GetParameter(2));
        hResEtaBin[ro][etab]        -> SetBinError(etb+1,fResEtEtaBin[ro][etab][etb]->GetParError(2));
	//CD: Fill the tree
	value = fResEtEtaBin[ro][etab][etb]->GetParameter(2); //parameter value
	error = fResEtEtaBin[ro][etab][etb]->GetParError(2);  //parameter error
	tResVar->Fill();
      }
      //CD: add a fake entry in et=0 for the NN training
      // for that, use a linear extrapolation.
      et = 0.;
      value = ((eTbinVals_[0]+eTbinVals_[1])/2.)*(fResEtEtaBin[ro][etab][0]->GetParameter(2)-fResEtEtaBin[ro][etab][1]->GetParameter(2))/((eTbinVals_[2]-eTbinVals_[0])/2.)+fResEtEtaBin[ro][etab][0]->GetParameter(2);
      error = fResEtEtaBin[ro][etab][0]->GetParError(2)+fResEtEtaBin[ro][etab][1]->GetParError(2);
      tResVar->Fill();
      // standard fit
      hResEtaBin[ro][etab] -> Fit(fResEtaBin[ro][etab]->GetName(),"RQ");
      hResEtaBin[ro][etab] -> Write();
    }
  } 
  hEtaBins -> Write();
  outfile->cd();
  outfile->Write();
  outfile->Close();
  std::cout<<"nr. of "<<objectType_<<" filled: "<<nrFilled<<std::endl;
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void ResolutionCreator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   // Get the gen and cal object fourvector
   std::vector<reco::Particle *> p4gen, p4rec;
   
   edm::Handle<TtGenEvent> genEvt;
   iEvent.getByLabel ("genEvt",genEvt);
   
   if(genEvt->particles().size()<10) return;
   
      if(objectType_ == "electron"){ 
     edm::Handle<std::vector<TopElectron> >  electrons; //to calculate the resolutions for the electrons, i used the TopElectron instead of the AOD information
     iEvent.getByLabel(labelName_,electrons);
     for(size_t e=0; e<electrons->size(); e++) { 
       for(size_t p=0; p<genEvt->particles().size(); p++){
         if( (abs(genEvt->particles()[p].pdgId()) == 11) && (ROOT::Math::VectorUtil::DeltaR(genEvt->particles()[p].p4(), (*electrons)[e].p4()) < minDR_) ) {
           p4gen.push_back(new reco::Particle(genEvt->particles()[p]));
	   p4rec.push_back(new reco::Particle((TopElectron)((*electrons)[e])));
	 }
       }
     }
   }
   else if(objectType_ == "muon"){
     edm::Handle<std::vector<TopMuon> >  muons;
     iEvent.getByLabel(labelName_,muons);
     for(size_t m=0; m<muons->size(); m++) {      
       for(size_t p=0; p<genEvt->particles().size(); p++){
         if( (abs(genEvt->particles()[p].pdgId()) == 13) && (ROOT::Math::VectorUtil::DeltaR(genEvt->particles()[p].p4(), (*muons)[m].p4()) < minDR_) ) {
           p4gen.push_back(new reco::Particle(genEvt->particles()[p]));
           p4rec.push_back(new reco::Particle((muonType)((*muons)[m])));
	 }
       }
     }
   }
   else if(objectType_ == "lJets"){
     edm::Handle<std::vector<jetType> >  ljets;
     iEvent.getByLabel(labelName_,ljets);
     if(ljets->size()>=4) { 
       for(unsigned int j = 0; j<4; j++){      
         for(size_t p=0; p<genEvt->particles().size(); p++){
           if( (abs(genEvt->particles()[p].pdgId()) < 5) && (ROOT::Math::VectorUtil::DeltaR(genEvt->particles()[p].p4(), (*ljets)[j].p4())< minDR_) ){
	     p4gen.push_back(new reco::Particle(genEvt->particles()[p]));
	     p4rec.push_back(new reco::Particle((jetType)(*ljets)[j]));
	   }
	 }
       }
     }
   }
   else if(objectType_ == "bJets"){
     edm::Handle<std::vector<jetType> >  bjets;
     iEvent.getByLabel(labelName_,bjets);
     if(bjets->size()>=4) { 
       for(unsigned int j = 0; j<4; j++){     
         for(size_t p=0; p<genEvt->particles().size(); p++){
           if( (abs(genEvt->particles()[p].pdgId()) == 5) && (ROOT::Math::VectorUtil::DeltaR(genEvt->particles()[p].p4(), (*bjets)[j].p4())< minDR_) ) {
	     p4gen.push_back(new reco::Particle(genEvt->particles()[p]));
	     p4rec.push_back(new reco::Particle((jetType)(*bjets)[j]));
	   }
	 }
       }
     }
   }
   else if(objectType_ == "met"){
     edm::Handle<std::vector<metType> >  mets;
     iEvent.getByLabel(labelName_,mets);
     if(mets->size()>=1) { 
       if( genEvt->isSemiLeptonic() && ROOT::Math::VectorUtil::DeltaR(genEvt->singleNeutrino()->p4(), (*mets)[0].p4()) < minDR_) {
         p4gen.push_back(new reco::Particle(0,genEvt->singleNeutrino()->p4(),math::XYZPoint()));
         p4rec.push_back(new reco::Particle((metType)((*mets)[0])));
       }
     }
   } 
   else if(objectType_ == "tau"){
     edm::Handle<std::vector<TopTau> > taus; 
     iEvent.getByLabel(labelName_,taus);
     for(std::vector<TopTau>::const_iterator tau = taus->begin(); tau != taus->end(); ++tau) {
       // find the tau (if any) that matches a MC tau from W
       reco::GenParticleCandidate genLepton = tau->getGenLepton();
       if( abs(genLepton.pdgId())==15 && genLepton.status()==2 &&
           genLepton.numberOfMothers()>0 &&
           abs(genLepton.mother(0)->pdgId())==15 &&
           genLepton.mother(0)->numberOfMothers()>0 &&
           abs(genLepton.mother(0)->mother(0)->pdgId())==24 &&
	   ROOT::Math::VectorUtil::DeltaR(genLepton.p4(), tau->p4()) < minDR_  ) {
       }
       p4gen.push_back(new reco::Particle(genLepton));
       p4rec.push_back(new reco::Particle(*tau));
	   
     }
   }
   // Fill the object's value
     for(unsigned m=0; m<p4gen.size(); m++){ 
       double Egen     = p4gen[m]->energy(); 
       double Thetagen = p4gen[m]->theta(); 
       double Phigen   = p4gen[m]->phi();
       double Etgen    = p4gen[m]->et();
       double Etagen   = p4gen[m]->eta();
       double Ecal     = p4rec[m]->energy(); 
       double Thetacal = p4rec[m]->theta();
       double Phical   = p4rec[m]->phi();
       double Etcal    = p4rec[m]->et();
       double Etacal   = p4rec[m]->eta();
       double phidiff  = Phical- Phigen;
       if(phidiff>3.14159)  phidiff = 2.*3.14159 - phidiff;
       if(phidiff<-3.14159) phidiff = -phidiff - 2.*3.14159;
   
       // find eta and et bin
       int etabin  =  0;
       if(etanrbins > 1){
         for(unsigned int b=0; b<etabinVals_.size()-1; b++) {
           if(fabs(Etacal) > etabinVals_[b]) etabin = b;
         }
       }
     
       int etbin  =  0;
       for(unsigned int b=0; b<eTbinVals_.size()-1; b++) {
         if(Etcal > eTbinVals_[b]) etbin = b;
       }
     
       // calculate the resolution on "a", "b", "c" & "d" according to the definition (CMS-NOTE-2006-023):
       // p = a*|p_meas|*u_1 + b*u_2 + c*u_3
       // E(fit) = E_meas * d
       //
       // with u_1 = p/|p_meas|
       //      u_3 = (u_z x u_1)/|u_z x u_1|
       //      u_2 = (u_1 x u_3)/|u_1 x u_3|
       //
       // The initial parameters values are chosen like (a, b, c, d) = (1., 0., 0., 1.)

       // 1/ calculate the unitary vectors of the basis u_1, u_2, u_3
       ROOT::Math::SVector<double,3> pcalvec(p4rec[m]->px(),p4rec[m]->py(),p4rec[m]->pz());
       ROOT::Math::SVector<double,3> pgenvec(p4gen[m]->px(),p4gen[m]->py(),p4gen[m]->pz());
       
       ROOT::Math::SVector<double,3> u_z(0,0,1);
       ROOT::Math::SVector<double,3> u_1 = ROOT::Math::Unit(pcalvec);
       ROOT::Math::SVector<double,3> u_3 = ROOT::Math::Cross(u_z,u_1)/ROOT::Math::Mag(ROOT::Math::Cross(u_z,u_1));
       ROOT::Math::SVector<double,3> u_2 = ROOT::Math::Cross(u_1,u_3)/ROOT::Math::Mag(ROOT::Math::Cross(u_1,u_3));
       double acal = 1.;
       double bcal = 0.;
       double ccal = 0.;
       double dcal = 1.;
       double agen = ROOT::Math::Dot(pgenvec,u_1)/ROOT::Math::Mag(pcalvec);
       double bgen = ROOT::Math::Dot(pgenvec,u_2);
       double cgen = ROOT::Math::Dot(pgenvec,u_3);
       double dgen = Egen/Ecal;
       //cout<<objectType_<<": "<<agen<<" "<<bgen<<" "<<cgen<<" "<<dgen<<endl;
        
       
       //fill histograms    
       ++nrFilled; 
       hResEtEtaBin[0][etabin][etbin] -> Fill(acal-agen);
       hResEtEtaBin[1][etabin][etbin] -> Fill(bcal-bgen);
       hResEtEtaBin[2][etabin][etbin] -> Fill(ccal-cgen);
       hResEtEtaBin[3][etabin][etbin] -> Fill(dcal-dgen);
       hResEtEtaBin[4][etabin][etbin] -> Fill(Thetacal-Thetagen);
       hResEtEtaBin[5][etabin][etbin] -> Fill(phidiff);
       hResEtEtaBin[6][etabin][etbin] -> Fill(Etcal-Etgen);
       hResEtEtaBin[7][etabin][etbin] -> Fill(Etacal-Etagen);

       delete p4gen[m];
       delete p4rec[m];
     } 
    
    
}
