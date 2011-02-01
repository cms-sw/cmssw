// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
//needed for TFileService
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
//needed for MessageLogger
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Electron.h"

#include "TDirectory.h"
#include "TH1F.h"
#include "TF1.h"
#include "TTree.h"

#include <Math/VectorUtil.h>

//
// class declaration
//

class ResolutionCreator : public edm::EDAnalyzer {
   public:
      explicit ResolutionCreator(const edm::ParameterSet&);
      ~ResolutionCreator();

   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
		  std::string objectType_, labelName_;
		  std::vector<double> etabinVals_, pTbinVals_;
  		double minDR_;
      int matchingAlgo_;
      bool useMaxDist_;
      bool useDeltaR_;
      double maxDist_;
  		int ptnrbins, etanrbins;
  		int nrFilled;

			//Histograms are booked in the beginJob() method
		  TF1 *fResPtEtaBin[10][20][20];
  		TF1 *fResEtaBin[10][20];
  		TH1F *hResPtEtaBin[10][20][20];
  		TH1F *hResEtaBin[10][20];
      TTree* tResVar;
};


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
  pTbinVals_	= iConfig.getParameter< std::vector<double> > 	("pTbinValues");
  minDR_	= iConfig.getParameter< double > ("minMatchingDR");

	nrFilled = 0;

}


ResolutionCreator::~ResolutionCreator()
{
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
ResolutionCreator::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   
 // Get the gen and cal object fourvector
   std::vector<reco::Particle *> p4gen, p4rec;
	    
   edm::Handle<TtGenEvent> genEvt;
   iEvent.getByLabel ("genEvt",genEvt);

   if(genEvt->particles().size()<10) return;

   if(objectType_ == "electron"){ 
     edm::Handle<std::vector<pat::Electron> >  electrons; //to calculate the ResolutionCreator for the electrons, i used the TopElectron instead of the AOD information
     iEvent.getByLabel(labelName_,electrons);
     for(size_t e=0; e<electrons->size(); e++) { 
       for(size_t p=0; p<genEvt->particles().size(); p++){
         if( (std::abs(genEvt->particles()[p].pdgId()) == 11) && (ROOT::Math::VectorUtil::DeltaR(genEvt->particles()[p].p4(), (*electrons)[e].p4()) < minDR_) ) {
           //p4gen.push_back(new reco::Particle(genEvt->particles()[p]));
	   //p4rec.push_back(new reco::Particle((pat::Electron)((*electrons)[e])));
	 }
       }
     }
   }
   else if(objectType_ == "muon"){
     edm::Handle<std::vector<pat::Muon> >  muons;
     iEvent.getByLabel(labelName_,muons);
     for(size_t m=0; m<muons->size(); m++) {      
       for(size_t p=0; p<genEvt->particles().size(); p++){
         if( (std::abs(genEvt->particles()[p].pdgId()) == 13) && (ROOT::Math::VectorUtil::DeltaR(genEvt->particles()[p].p4(), (*muons)[m].p4()) < minDR_) ) {
           //p4gen.push_back(new reco::Particle(genEvt->particles()[p]));
           //p4rec.push_back(new reco::Particle((pat::Muon)((*muons)[m])));
	 }
       }
     }
   }
   else if(objectType_ == "lJets" ){
     edm::Handle<std::vector<pat::Jet> > jets;
     iEvent.getByLabel(labelName_,jets);	 
     if(jets->size()>=4) { 
       for(unsigned int j = 0; j<4; j++){      
         for(size_t p=0; p<genEvt->particles().size(); p++){
           if( (std::abs(genEvt->particles()[p].pdgId()) < 5) && (ROOT::Math::VectorUtil::DeltaR(genEvt->particles()[p].p4(), (*jets)[j].p4())< minDR_) ){
	     //p4gen.push_back(new reco::Particle(genEvt->particles()[p]));
	     //p4rec.push_back(new reco::Particle((pat::Jet)(*jets)[j]));
	   }
	 }
       }
     }
   }
   else if(objectType_ == "bJets" ){
     edm::Handle<std::vector<pat::Jet> > jets;
     iEvent.getByLabel(labelName_,jets);
     if(jets->size()>=4) { 
       for(unsigned int j = 0; j<4; j++){      
         for(size_t p=0; p<genEvt->particles().size(); p++){
	   if( (std::abs(genEvt->particles()[p].pdgId()) == 5) && (ROOT::Math::VectorUtil::DeltaR(genEvt->particles()[p].p4(), (*jets)[j].p4())< minDR_) ) {
	     //p4gen.push_back(new reco::Particle(genEvt->particles()[p]));
	     //p4rec.push_back(new reco::Particle((pat::Jet)(*jets)[j]));
	   }
	 }
       }
     }
   }
   else if(objectType_ == "met"){
     edm::Handle<std::vector<pat::MET> >  mets;
     iEvent.getByLabel(labelName_,mets);
     if(mets->size()>=1) { 
       if( genEvt->isSemiLeptonic() && genEvt->singleNeutrino() != 0 && ROOT::Math::VectorUtil::DeltaR(genEvt->singleNeutrino()->p4(), (*mets)[0].p4()) < minDR_) {
         //p4gen.push_back(new reco::Particle(0,genEvt->singleNeutrino()->p4(),math::XYZPoint()));
         //p4rec.push_back(new reco::Particle((pat::MET)((*mets)[0])));
       }
     }
   } 
   else if(objectType_ == "tau"){
     edm::Handle<std::vector<pat::Tau> > taus; 
     iEvent.getByLabel(labelName_,taus);
     for(std::vector<pat::Tau>::const_iterator tau = taus->begin(); tau != taus->end(); ++tau) {
       // find the tau (if any) that matches a MC tau from W
       reco::GenParticle genLepton = *(tau->genLepton());
       if( std::abs(genLepton.pdgId())==15 && genLepton.status()==2 &&
           genLepton.numberOfMothers()>0 &&
           std::abs(genLepton.mother(0)->pdgId())==15 &&
           genLepton.mother(0)->numberOfMothers()>0 &&
           std::abs(genLepton.mother(0)->mother(0)->pdgId())==24 &&
	   ROOT::Math::VectorUtil::DeltaR(genLepton.p4(), tau->p4()) < minDR_  ) {
       }
       //p4gen.push_back(new reco::Particle(genLepton));
       //p4rec.push_back(new reco::Particle(*tau));
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
     
       int ptbin  =  0;
       for(unsigned int b=0; b<pTbinVals_.size()-1; b++) {
         if(p4rec[m]->pt() > pTbinVals_[b]) ptbin = b;
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
			        
       //fill histograms    
       ++nrFilled; 
       hResPtEtaBin[0][etabin][ptbin] -> Fill(acal-agen);
       hResPtEtaBin[1][etabin][ptbin] -> Fill(bcal-bgen);
       hResPtEtaBin[2][etabin][ptbin] -> Fill(ccal-cgen);
       hResPtEtaBin[3][etabin][ptbin] -> Fill(dcal-dgen);
       hResPtEtaBin[4][etabin][ptbin] -> Fill(Thetacal-Thetagen);
       hResPtEtaBin[5][etabin][ptbin] -> Fill(phidiff);
       hResPtEtaBin[6][etabin][ptbin] -> Fill(Etcal-Etgen);
       hResPtEtaBin[7][etabin][ptbin] -> Fill(Etacal-Etagen);

       delete p4gen[m];
       delete p4rec[m];
     }
		 
}


// ------------ method called once each job just before starting event loop  ------------
void 
ResolutionCreator::beginJob()
{
  edm::Service<TFileService> fs;
  if (!fs) throw edm::Exception(edm::errors::Configuration, "TFileService missing from configuration!");

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
  
  const char*   resObsVsPtFit[8]    	= {	"[0]+[1]*exp(-[2]*x)",
                                          "[0]+[1]*exp(-[2]*x)",
                                          "[0]+[1]*exp(-[2]*x)",
                                          "[0]+[1]*exp(-[2]*x)",
					   															"[0]+[1]*exp(-[2]*x)",
					   															"[0]+[1]*exp(-[2]*x)",
					   															"pol1",
					   															"[0]+[1]*exp(-[2]*x)"
					  														};
 
  ptnrbins        = pTbinVals_.size()-1;
  double *ptbins  = new double[pTbinVals_.size()];
  for(unsigned int b=0; b<pTbinVals_.size(); b++)  ptbins[b]  = pTbinVals_[b];
  double *etabins;
  if(objectType_ != "met"){
    etanrbins  = etabinVals_.size()-1;
    etabins    = new double[etabinVals_.size()];
    for(unsigned int b=0; b<etabinVals_.size(); b++) etabins[b] = etabinVals_[b];
  }else{
    etanrbins = 1;
    etabins    = new double[2];
    etabins[0] = 0; etabins[1] = 5.;
  }
	

  //define the histograms booked
  for(Int_t ro=0; ro<8; ro++) {
    for(Int_t etab=0; etab<etanrbins; etab++) {	
      for(Int_t ptb=0; ptb<ptnrbins; ptb++) {
        TString obsName = objectType_; obsName += resObsName[ro]; obsName += "_etabin"; obsName += etab; obsName += "_ptbin";
				obsName += ptb;
				hResPtEtaBin[ro][etab][ptb] = fs->make<TH1F>(obsName,obsName,resObsNrBins,resObsMin[ro],resObsMax[ro]);
        fResPtEtaBin[ro][etab][ptb] = fs->make<TF1>("F_"+obsName,"gaus");
      }
      TString obsName2 = objectType_; obsName2 += resObsName[ro]; obsName2 += "_etabin"; obsName2 += etab;
      hResEtaBin[ro][etab] = fs->make<TH1F>(obsName2,obsName2,ptnrbins,ptbins);
      fResEtaBin[ro][etab] = fs->make<TF1>("F_"+obsName2,resObsVsPtFit[ro],pTbinVals_[0],pTbinVals_[pTbinVals_.size()-1]);
    }
  }
	tResVar = fs->make< TTree >("tResVar","Resolution tree");

  delete [] etabins; 
  delete [] ptbins; 

}

// ------------ method called once each job just after ending the event loop  ------------
void 
ResolutionCreator::endJob() {
  TString  	  resObsName2[8] 	= {"a","b","c","d","theta","phi","et","eta"};
  Int_t ro=0;
  Double_t pt=0.;
  Double_t eta=0.;
  Double_t value,error;

  tResVar->Branch("Pt",&pt,"Pt/D");
  tResVar->Branch("Eta",&eta,"Eta/D");
  tResVar->Branch("ro",&ro,"ro/I");
  tResVar->Branch("value",&value,"value/D");
  tResVar->Branch("error",&error,"error/D");
  
  for(ro=0; ro<8; ro++) {
    for(int etab=0; etab<etanrbins; etab++) {	
      //CD set eta at the center of the bin
      eta = etanrbins > 1 ? (etabinVals_[etab]+etabinVals_[etab+1])/2. : 2.5 ; 
      for(int ptb=0; ptb<ptnrbins; ptb++) {
				//CD set pt at the center of the bin
				pt = (pTbinVals_[ptb]+pTbinVals_[ptb+1])/2.; 
        double maxcontent = 0.;
				int maxbin = 0;
				for(int nb=1; nb<hResPtEtaBin[ro][etab][ptb]->GetNbinsX(); nb ++){
	  			if (hResPtEtaBin[ro][etab][ptb]->GetBinContent(nb)>maxcontent) {
	    			maxcontent = hResPtEtaBin[ro][etab][ptb]->GetBinContent(nb);
	    			maxbin = nb;
	  			}
				}
				int range = (int)(hResPtEtaBin[ro][etab][ptb]->GetNbinsX()/6); //in order that ~1/3 of X-axis range is fitted
  			fResPtEtaBin[ro][etab][ptb] -> SetRange(hResPtEtaBin[ro][etab][ptb]->GetBinCenter(maxbin-range),
				hResPtEtaBin[ro][etab][ptb]->GetBinCenter(maxbin+range));
				fResPtEtaBin[ro][etab][ptb] -> SetParameters(hResPtEtaBin[ro][etab][ptb] -> GetMaximum(),
	                                             				hResPtEtaBin[ro][etab][ptb] -> GetMean(),
	 					     																			hResPtEtaBin[ro][etab][ptb] -> GetRMS());
				hResPtEtaBin[ro][etab][ptb] -> Fit(fResPtEtaBin[ro][etab][ptb]->GetName(),"RQ");
  			hResEtaBin[ro][etab]        -> SetBinContent(ptb+1,fResPtEtaBin[ro][etab][ptb]->GetParameter(2));
  			hResEtaBin[ro][etab]        -> SetBinError(ptb+1,fResPtEtaBin[ro][etab][ptb]->GetParError(2));
				//CD: Fill the tree
				value = fResPtEtaBin[ro][etab][ptb]->GetParameter(2); //parameter value
				error = fResPtEtaBin[ro][etab][ptb]->GetParError(2);  //parameter error
				tResVar->Fill();
      }
      //CD: add a fake entry in pt=0 for the NN training
      // for that, use a linear extrapolation.
      pt = 0.;
      value =	((pTbinVals_[0]+pTbinVals_[1])/2.)*(fResPtEtaBin[ro][etab][0]->GetParameter(2)-fResPtEtaBin[ro][etab][1]->GetParameter(2))/((pTbinVals_[2]-pTbinVals_[0])/2.)+fResPtEtaBin[ro][etab][0]->GetParameter(2);
      error = fResPtEtaBin[ro][etab][0]->GetParError(2)+fResPtEtaBin[ro][etab][1]->GetParError(2);
      tResVar->Fill();
      // standard fit
      hResEtaBin[ro][etab] -> Fit(fResEtaBin[ro][etab]->GetName(),"RQ");
    }
  } 
  if(objectType_ == "lJets" && nrFilled == 0) edm::LogProblem  ("SummaryError") << "No plots filled for light jets \n";    
  if(objectType_ == "bJets" && nrFilled == 0) edm::LogProblem  ("SummaryError") << "No plots filled for bjets \n";    
  if(objectType_ == "muon" && nrFilled == 0) edm::LogProblem  ("SummaryError") << "No plots filled for muons \n";    
  if(objectType_ == "electron" && nrFilled == 0) edm::LogProblem  ("SummaryError") << "No plots filled for electrons \n";    
  if(objectType_ == "tau" && nrFilled == 0) edm::LogProblem  ("SummaryError") << "No plots filled for taus \n";    
  if(objectType_ == "met" && nrFilled == 0) edm::LogProblem  ("SummaryError") << "No plots filled for met \n";    
	
  edm::LogVerbatim ("MainResults") << " \n\n";	
  edm::LogVerbatim ("MainResults") << " ----------------------------------------------";
  edm::LogVerbatim ("MainResults") << " ----------------------------------------------";
  edm::LogVerbatim ("MainResults") << " Resolutions on "<< objectType_ << " with nrfilled: "<<nrFilled;
  edm::LogVerbatim ("MainResults") << " ----------------------------------------------";
  edm::LogVerbatim ("MainResults") << " ----------------------------------------------";
  if(nrFilled != 0 && objectType_ != "met") {
  	for(ro=0; ro<8; ro++) {
 			edm::LogVerbatim ("MainResults") << "-------------------- ";
    	edm::LogVerbatim ("MainResults") << "\n Resolutions on " << resObsName2[ro] << "\n";
  		edm::LogVerbatim ("MainResults") << "-------------------- ";
			for(int etab=0; etab<etanrbins; etab++) {	
  			if(nrFilled != 0 && ro != 6) {
					if(etab == 0){
						edm::LogVerbatim   ("MainResults") << "if(fabs(eta)<"<<etabinVals_[etab+1] <<") res = " << 
						fResEtaBin[ro][etab]->GetParameter(0) << "+" << fResEtaBin[ro][etab]->GetParameter(1) 
						<< "*exp(-(" <<fResEtaBin[ro][etab]->GetParameter(2) << "*pt));";  
					}else{ 
						edm::LogVerbatim   ("MainResults") << "else if(fabs(eta)<"<<etabinVals_[etab+1] <<") res = " << 
						fResEtaBin[ro][etab]->GetParameter(0) << "+" << fResEtaBin[ro][etab]->GetParameter(1) 
						<< "*exp(-(" <<fResEtaBin[ro][etab]->GetParameter(2) << "*pt));";  					
					}
				}else if(nrFilled != 0 && ro == 6){
					if(etab == 0){
						edm::LogVerbatim   ("MainResults") << "if(fabs(eta)<"<<etabinVals_[etab+1] <<") res = " << 
						fResEtaBin[ro][etab]->GetParameter(0) << "+" << fResEtaBin[ro][etab]->GetParameter(1) 
						<< "*pt;";
					}else{  					
						edm::LogVerbatim   ("MainResults") << "else if(fabs(eta)<"<<etabinVals_[etab+1] <<") res = " << 
						fResEtaBin[ro][etab]->GetParameter(0) << "+" << fResEtaBin[ro][etab]->GetParameter(1) 
						<< "*pt;";

					}
				}
			}
		}
	}else if(nrFilled != 0 && objectType_ == "met"){
 	for(ro=0; ro<8; ro++) {
 			edm::LogVerbatim ("MainResults") << "-------------------- ";
    	edm::LogVerbatim ("MainResults") << "\n Resolutions on " << resObsName2[ro] << "\n";
  		edm::LogVerbatim ("MainResults") << "-------------------- ";
			if(nrFilled != 0 && ro != 6) {
					edm::LogVerbatim   ("MainResults") << "res = " <<
					fResEtaBin[ro][0]->GetParameter(0) << "+" << fResEtaBin[ro][0]->GetParameter(1) 
					<< "*exp(-(" <<fResEtaBin[ro][0]->GetParameter(2) << "*pt));";  			
			}else if(nrFilled != 0 && ro == 6){
					edm::LogVerbatim   ("MainResults") << "res = " << 
					fResEtaBin[ro][0]->GetParameter(0) << "+" << fResEtaBin[ro][0]->GetParameter(1) << "*pt;";
			}
		}
	}
}

//define this as a plug-in
DEFINE_FWK_MODULE(ResolutionCreator);
