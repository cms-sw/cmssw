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
  objectType_  	= iConfig.getParameter< std::string >    ("object");
  jetmetLabel_  = iConfig.getParameter< std::string > 	 ("jetmetLabel");
  if(objectType_ != "met"){
    etabinVals_	= iConfig.getParameter< std::vector<double> > ("etabinValues");
  }
  eTbinVals_	= iConfig.getParameter< std::vector<double> > ("eTbinValues");
  minDR_	= iConfig.getParameter< double >         ("minMatchingDR");


  // input constants  
  TString  	  resObsName[6] 	= {"pres","eres","thres","phres","etres","etares"};
  TString  	  def[2] 	 	= {"_abs","_rel"};
  int      	  resObsNrBins  	= 240;
  std::vector<double>  resObsMinAbs, resObsMaxAbs, resObsMinRel, resObsMaxRel;
  if(objectType_ == "electron"){ 
    resObsMinAbs.push_back(-0.004);  resObsMinAbs.push_back(0.45);  resObsMinAbs.push_back(-0.0012);  resObsMinAbs.push_back(-0.009);  resObsMinAbs.push_back(-16); resObsMinAbs.push_back(-0.0012);   
    resObsMaxAbs.push_back( 0.004);  resObsMaxAbs.push_back(2.2);  resObsMaxAbs.push_back( 0.0012);  resObsMaxAbs.push_back( 0.009);  resObsMaxAbs.push_back( 16); resObsMaxAbs.push_back( 0.0012);
    resObsMinRel.push_back(-0.0001); resObsMinRel.push_back(0.00);  resObsMinRel.push_back( -0.004); resObsMinRel.push_back(-0.004); resObsMinRel.push_back( -0.4); resObsMinRel.push_back( -0.01);
    resObsMaxRel.push_back( 0.0001); resObsMaxRel.push_back( 0.05); resObsMaxRel.push_back(  0.004); resObsMaxRel.push_back( 0.004); resObsMaxRel.push_back(  0.4); resObsMaxRel.push_back(  0.01);
  } else if(objectType_ == "muon"){
    resObsMinAbs.push_back(-0.004);  resObsMinAbs.push_back(0.90);  resObsMinAbs.push_back(-0.004);  resObsMinAbs.push_back(-0.003);  resObsMinAbs.push_back(-8);  resObsMinAbs.push_back(-0.004);   
    resObsMaxAbs.push_back( 0.004);  resObsMaxAbs.push_back(1.1);  resObsMaxAbs.push_back( 0.004);  resObsMaxAbs.push_back( 0.003);  resObsMaxAbs.push_back( 8);  resObsMaxAbs.push_back( 0.004);
    resObsMinRel.push_back(-0.0001); resObsMinRel.push_back(0.00);  resObsMinRel.push_back( -0.004); resObsMinRel.push_back(-0.004); resObsMinRel.push_back( -0.4); resObsMinRel.push_back( -0.01);
    resObsMaxRel.push_back( 0.0001); resObsMaxRel.push_back( 0.05); resObsMaxRel.push_back(  0.004); resObsMaxRel.push_back( 0.004); resObsMaxRel.push_back(  0.4); resObsMaxRel.push_back(  0.01);
  } else if(objectType_ == "lJets" || objectType_ == "bJets"){
    resObsMinAbs.push_back(-0.04);  resObsMinAbs.push_back(0.);   resObsMinAbs.push_back(-0.4); resObsMinAbs.push_back(-0.6); resObsMinAbs.push_back( -80); resObsMinAbs.push_back(-0.6);   
    resObsMaxAbs.push_back( 0.04);  resObsMaxAbs.push_back(5);   resObsMaxAbs.push_back( 0.4); resObsMaxAbs.push_back( 0.6); resObsMaxAbs.push_back( 80);  resObsMaxAbs.push_back( 0.6);
    resObsMinRel.push_back(-0.002); resObsMinRel.push_back(0.);   resObsMinRel.push_back(-0.4); resObsMinRel.push_back(-0.2); resObsMinRel.push_back( -2);  resObsMinRel.push_back( -0.5);
    resObsMaxRel.push_back( 0.002); resObsMaxRel.push_back(0.1); resObsMaxRel.push_back( 0.4); resObsMaxRel.push_back( 0.2); resObsMaxRel.push_back(  00); resObsMaxRel.push_back( 0.5);
  } else{
    resObsMinAbs.push_back(-0.06);   resObsMinAbs.push_back(0.);   resObsMinAbs.push_back(-6);   resObsMinAbs.push_back(-6);   resObsMinAbs.push_back( -180); resObsMinAbs.push_back(-6);   
    resObsMaxAbs.push_back( 0.18);   resObsMaxAbs.push_back(20.);  resObsMaxAbs.push_back( 6);   resObsMaxAbs.push_back( 6);   resObsMaxAbs.push_back(  180); resObsMaxAbs.push_back( 6);
    resObsMinRel.push_back(-0.008); resObsMinRel.push_back(0.);   resObsMinRel.push_back(-1); resObsMinRel.push_back(-1); resObsMinRel.push_back( -2);  resObsMinRel.push_back( -1.);
    resObsMaxRel.push_back( 0.002);  resObsMaxRel.push_back(0.1); resObsMaxRel.push_back( 1); resObsMaxRel.push_back( 1); resObsMaxRel.push_back(  2);  resObsMaxRel.push_back( 1.);
  
  }
  
  const char*   resObsVsEtFit[6]    	= {"[0]+[1]*exp(-[2]*pow(x,-1))",
                                           "[0]+[1]*exp(-[2]*x)",
					   "[0]+[1]*exp(-[2]*x)",
					   "[0]+[1]*exp(-[2]*x)",
					   "[0]+[1]*exp(-[2]*x)",
					   "[0]+[1]*exp(-[2]*x)"
					  };
 
  const char*   resObsVsEtaFit[18]  	= {
                                           "[0]+[1]*exp(-[2]*x)","[0]+[1]*exp(-[2]*x)","[0]+[1]*exp(-[2]*x)",
					   "[0]+[1]*exp(-[2]*x)","[0]+[1]*exp(-[2]*x)","[0]+[1]*exp(-[2]*x)",
					   "[0]+[1]*exp(-[2]*x)","[0]+[1]*exp(-[2]*x)","[0]+[1]*exp(-[2]*x)",
					   "[0]+[1]*exp(-[2]*x)","[0]+[1]*exp(-[2]*x)","[0]+[1]*exp(-[2]*x)",
					   "[0]+[1]*exp(-[2]*x)","[0]+[1]*exp(-[2]*x)","[0]+[1]*exp(-[2]*x)",
					   "[0]+[1]*exp(-[2]*x)","[0]+[1]*exp(-[2]*x)","[0]+[1]*exp(-[2]*x)"
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
  if(objectType_ == "lJets" || objectType_ == "bJets") { outputFileName += "_"; outputFileName += jetmetLabel_; }; 
  outputFileName += ".root"; 
  outfile = new TFile(outputFileName, "RECREATE");
  
  for(Int_t ro=0; ro<6; ro++) {
    for(Int_t aor=0; aor<1; aor++) {
      for(Int_t etab=0; etab<etanrbins; etab++) {	
        for(Int_t etb=0; etb<etnrbins; etb++) {
          TString obsName = objectType_; obsName += resObsName[ro]; obsName += "_etabin"; obsName += etab; obsName += "_etbin"; obsName += etb; obsName += def[aor];
	  if(aor==0) hResEtEtaBin[ro][etab][etb][aor] = new TH1F(obsName,obsName,resObsNrBins,resObsMinAbs[ro],resObsMaxAbs[ro]);
	  if(aor==1) hResEtEtaBin[ro][etab][etb][aor] = new TH1F(obsName,obsName,resObsNrBins,resObsMinRel[ro],resObsMaxRel[ro]);
          fResEtEtaBin[ro][etab][etb][aor] = new TF1("F_"+obsName,"gaus");
        }
        TString obsName2 = objectType_; obsName2 += resObsName[ro]; obsName2 += "_etabin"; obsName2 += etab; obsName2 += def[aor];
	hResEtaBin[ro][etab][aor] = new TH1F(obsName2,obsName2,etnrbins,etbins);
        fResEtaBin[ro][etab][aor] = new TF1("F_"+obsName2,resObsVsEtFit[ro],eTbinVals_[0],eTbinVals_[eTbinVals_.size()-1]);
      }
      for(Int_t par=0; par<3; par++) {
        TString obsName3 = objectType_; obsName3 += resObsName[ro]; obsName3 += "_par"; obsName3 += par; obsName3 += def[aor];
        hResPar[ro][aor][par] = new TH1F(obsName3,obsName3,etanrbins,etabins);
        fResPar[ro][aor][par] = new TF1(obsName3,resObsVsEtaFit[ro*3+par],etabins[0],etabins[etabinVals_.size()-1]);
      }
    }
  }
  delete [] etabins; 
  delete [] etbins; 
  
  nrFilled = 0;
}


ResolutionCreator::~ResolutionCreator()
{
  outfile->cd();
  for(Int_t ro=0; ro<6; ro++) {
    for(Int_t aor=0; aor<1; aor++) {
      for(int etab=0; etab<etanrbins; etab++) {	
        for(int etb=0; etb<etnrbins; etb++) {
          double maxcontent = 0.;
	  int maxbin = 0;
	  for(int nb=1; nb<hResEtEtaBin[ro][etab][etb][aor]->GetNbinsX(); nb ++){
	    if (hResEtEtaBin[ro][etab][etb][aor]->GetBinContent(nb)>maxcontent) {
	      maxcontent = hResEtEtaBin[ro][etab][etb][aor]->GetBinContent(nb);
	      maxbin = nb;
	    }
	  }
          fResEtEtaBin[ro][etab][etb][aor] -> SetRange(hResEtEtaBin[ro][etab][etb][aor]->GetBinCenter(maxbin-8),
	  					       hResEtEtaBin[ro][etab][etb][aor]->GetBinCenter(maxbin+8));
          hResEtEtaBin[ro][etab][etb][aor] -> Fit(fResEtEtaBin[ro][etab][etb][aor]->GetName(),"Q");
          //fResEtEtaBin[ro][etab][etb][aor] -> SetRange(fResEtEtaBin[ro][etab][etb][aor]->GetParameter(1)-1.5*fResEtEtaBin[ro][etab][etb][aor]->GetParameter(2),
	  //                                            fResEtEtaBin[ro][etab][etb][aor]->GetParameter(1)+1.5*fResEtEtaBin[ro][etab][etb][aor]->GetParameter(2));
          //hResEtEtaBin[ro][etab][etb][aor] -> Fit(fResEtEtaBin[ro][etab][etb][aor]->GetName(),"RQ");
          hResEtEtaBin[ro][etab][etb][aor] -> Write();
          hResEtaBin[ro][etab][aor]        -> SetBinContent(etb+1,fResEtEtaBin[ro][etab][etb][aor]->GetParameter(2));
          hResEtaBin[ro][etab][aor]        -> SetBinError(etb+1,fResEtEtaBin[ro][etab][etb][aor]->GetParError(2));
	}
        hResEtaBin[ro][etab][aor] -> Fit(fResEtaBin[ro][etab][aor]->GetName(),"Q");
        hResEtaBin[ro][etab][aor] -> Write();
        if(objectType_ != "met"){
	  for(Int_t par=0; par<3; par++) {
            hResPar[ro][aor][par]   -> SetBinContent(etab+1,fResEtaBin[ro][etab][aor]->GetParameter(par)); 
            hResPar[ro][aor][par]   -> SetBinError(etab+1,fResEtaBin[ro][etab][aor]->GetParError(par)); 
          }
	}     
      }
      if(etanrbins > 1){
        for(Int_t par=0; par<3; par++) {
          hResPar[ro][aor][par] -> Fit(hResPar[ro][aor][par] -> GetName(),"Q"); 
          hResPar[ro][aor][par] -> Write(); 
        }
      }  
    }
  } 
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
     iEvent.getByType(electrons);
     if(electrons->size()>=1) {	 
       if( ROOT::Math::VectorUtil::DeltaR( genEvt->particles()[4]->p4(),(*electrons)[0].p4()) < minDR_)  {
         p4gen.push_back(new reco::Particle(*(genEvt->particles()[4]))); 
	 p4rec.push_back(new reco::Particle((TopElectron)((*electrons)[0]))); //changed this too	 
       }
     }
   }else if(objectType_ == "muon"){
     edm::Handle<std::vector<muonType> >  muons;
     iEvent.getByType(muons);
     if(muons->size()>=1) { 
       if( ROOT::Math::VectorUtil::DeltaR(genEvt->particles()[4]->p4(), (*muons)[0].p4()) < minDR_){
         p4gen.push_back(new reco::Particle(*(genEvt->particles()[4]))); 
         p4rec.push_back(new reco::Particle((muonType)((*muons)[0]))); 
       }
     }
   }else if(objectType_ == "lJets"){
     edm::Handle<std::vector<jetType> >  ljets;
     iEvent.getByLabel(jetmetLabel_,ljets);
     if(ljets->size()>=4) { 
       for(unsigned int p = 0; p<2; p++){
         for(unsigned int j = 0; j<4; j++){
           if( ROOT::Math::VectorUtil::DeltaR(genEvt->particles()[p]->p4(), (*ljets)[j].p4())< minDR_)  {
	      p4gen.push_back(new reco::Particle(*(genEvt->particles()[p]))); 
	      p4rec.push_back(new reco::Particle((jetType)(*ljets)[j])); 
	   }
	 }
       }
     }
   }else if(objectType_ == "bJets"){
     edm::Handle<std::vector<jetType> >  bjets;
     iEvent.getByLabel(jetmetLabel_,bjets);
     if(bjets->size()>=4) { 
       for(unsigned int p = 2; p<4; p++){
         for(unsigned int j = 0; j<4; j++){
           if( ROOT::Math::VectorUtil::DeltaR(genEvt->particles()[p]->p4(), (*bjets)[j].p4())< minDR_)  {
	      p4gen.push_back(new reco::Particle(*(genEvt->particles()[p]))); 
	      p4rec.push_back(new reco::Particle((jetType)(*bjets)[j])); 
	   }
	 }
       }
     }
   }else if(objectType_ == "met"){

     edm::Handle<std::vector<metType> >  mets;
     iEvent.getByLabel(jetmetLabel_,mets);
     if(mets->size()>=1) { 
       if( ROOT::Math::VectorUtil::DeltaR(genEvt->particles()[5]->p4(), (*mets)[0].p4()) < minDR_){
         p4gen.push_back(new reco::Particle(*(genEvt->particles()[5]))); 
         p4rec.push_back(new reco::Particle((metType)((*mets)[0]))); 
       }
     }
   }
   // Fill the object's value
     for(unsigned m=0; m<p4gen.size(); m++){ 
       double Egen     = p4gen[m]->energy(); 
       double Pgen     = p4gen[m]->p(); 
       double Thetagen = p4gen[m]->theta(); 
       double Phigen   = p4gen[m]->phi();
       double Etgen    = p4gen[m]->et();
       double Etagen   = p4gen[m]->eta();
       double Ecal     = p4rec[m]->energy(); 
       double Pcal     = p4rec[m]->p();
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
     
       //fill histograms    
       ++nrFilled; 
       //abs
       hResEtEtaBin[0][etabin][etbin][0] -> Fill(1./Pcal - 1./Pgen);
       hResEtEtaBin[1][etabin][etbin][0] -> Fill(Egen/Ecal);
       hResEtEtaBin[2][etabin][etbin][0] -> Fill(Thetacal-Thetagen);
       hResEtEtaBin[3][etabin][etbin][0] -> Fill(phidiff);
       hResEtEtaBin[4][etabin][etbin][0] -> Fill(Etcal-Etgen);
       hResEtEtaBin[5][etabin][etbin][0] -> Fill(Etacal-Etagen);
       //rel
      /* hResEtEtaBin[0][etabin][etbin][1] -> Fill((1./Pcal - 1./Pgen)/Pgen);
       hResEtEtaBin[1][etabin][etbin][1] -> Fill(1./Ecal);
       hResEtEtaBin[2][etabin][etbin][1] -> Fill((Thetacal-Thetagen)/Thetagen);
       hResEtEtaBin[3][etabin][etbin][1] -> Fill(phidiff/Phigen);
       hResEtEtaBin[4][etabin][etbin][1] -> Fill((Etcal-Etgen)/Etgen);
       hResEtEtaBin[5][etabin][etbin][1] -> Fill((Etacal-Etagen)/Etagen);*/

       delete p4gen[m];
       delete p4rec[m];
     } 
    
    
}
