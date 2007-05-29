#include "Validation/RecoMuon/src/HTrack.h" 
#include "Validation/RecoMuon/src/Histograms.h" 
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "SimDataFormats/Track/interface/SimTrack.h"

#include "TFile.h"
#include "TDirectory.h"


using namespace std;

HTrack::HTrack(string name,string whereIs):
  theName(name.c_str()),where(whereIs.c_str()){

  hVariables = new HTrackVariables(name,whereIs);
  hResolution = new HResolution(name+"_Res",whereIs);
  hPull = new HResolution(name+"_Pull",whereIs);
  hTDRResolution = new HResolution(name+"_TDRRes",whereIs);
  hTDRPull = new HResolution(name+"_TDRPull",whereIs);
  
  
  doSubHisto = false;

  if(doSubHisto){

    // [5-10] GeV range
    hResolution_5_10 = new HResolution(name+"_Res_Pt_5_10",whereIs);
    hTDRResolution_5_10 = new HResolution(name+"_TDRRes_Pt_5_10",whereIs);
    hPull_5_10 = new HResolution(name+"_Pull_Pt_5_10",whereIs);
    
    
    hResolution_10_40 = new HResolution(name+"_Res_Pt_10_40",whereIs);
    hTDRResolution_10_40 = new HResolution(name+"_TDRRes_Pt_10_40",whereIs);
    hPull_10_40 = new HResolution(name+"_Pull_Pt_10_40",whereIs);
    
    
    hResolution_40_70 = new HResolution(name+"_Res_Pt_40_70",whereIs);
    hTDRResolution_40_70 = new HResolution(name+"_TDRRes_Pt_40_70",whereIs);
    hPull_40_70 = new HResolution(name+"_Pull_Pt_40_70",whereIs);
    
    
    hResolution_70_100 = new HResolution(name+"_Res_Pt_70_100",whereIs);
    hTDRResolution_70_100 = new HResolution(name+"_TDRRes_Pt_70_100",whereIs);
    hPull_70_100 = new HResolution(name+"_Pull_Pt_70_100",whereIs);
    
    
    hResolution_08 = new HResolution(name+"_Res_Eta_08",whereIs);
    hTDRResolution_08 = new HResolution(name+"_TDRRes_Eta_08",whereIs);
    hPull_08 = new HResolution(name+"_Pull_Eta_08",whereIs);
    
   	
    hResolution_08_12 = new HResolution(name+"_Res_Eta_08_12",whereIs);
    hTDRResolution_08_12 = new HResolution(name+"_TDRRes_Eta_08_12",whereIs);
    hPull_08_12 = new HResolution(name+"_Pull_Eta_08_12",whereIs);
    
    
    hResolution_12_21 = new HResolution(name+"_Res_Eta_12_21",whereIs);
    hTDRResolution_12_21 = new HResolution(name+"_TDRRes_Eta_12_21",whereIs);
    hPull_12_21 = new HResolution(name+"_Pull_Eta_12_21",whereIs);
    
    
    hResolution_12_24 = new HResolution(name+"_Res_Eta_12_24",whereIs);
    hTDRResolution_12_24 = new HResolution(name+"_TDRRes_Eta_12_24",whereIs);
    hPull_12_24 = new HResolution(name+"_Pull_Eta_12_24",whereIs);

    
    hResolution_12_21_plus = new HResolution(name+"_Res_Eta_12_21_plus",whereIs);
    hTDRResolution_12_21_plus = new HResolution(name+"_TDRRes_Eta_12_21_plus",whereIs);
    hPull_12_21_plus = new HResolution(name+"_Pull_Eta_12_21_plus",whereIs);
    
    
    hResolution_12_24_plus = new HResolution(name+"_Res_Eta_12_24_plus",whereIs);
    hTDRResolution_12_24_plus = new HResolution(name+"_TDRRes_Eta_12_24_plus",whereIs);
    hPull_12_24_plus = new HResolution(name+"_Pull_Eta_12_24_plus",whereIs);
    
    
    hResolution_12_21_minus = new HResolution(name+"_Res_Eta_12_21_minus",whereIs);
    hTDRResolution_12_21_minus = new HResolution(name+"_TDRRes_Eta_12_21_minus",whereIs);
    hPull_12_21_minus = new HResolution(name+"_Pull_Eta_12_21_minus",whereIs);
    
    
    hResolution_12_24_minus = new HResolution(name+"_Res_Eta_12_24_minus",whereIs);
    hTDRResolution_12_24_minus = new HResolution(name+"_TDRRes_Eta_12_24_minus",whereIs);
    hPull_12_24_minus = new HResolution(name+"_Pull_Eta_12_24_minus",whereIs);
  }
}


void HTrack::Write(TFile *file){
  
  TDirectory *rootDir = file->mkdir(theName+"_"+where);
  rootDir->cd();
  
  hVariables->Write();

  TDirectory * res = rootDir->mkdir("Resolution");
  res->cd();
      
  hResolution->Write();
  hTDRResolution->Write(); 
  
  TDirectory * pull = rootDir->mkdir("Pull");
  pull->cd();
      
  hPull->Write();
  hTDRPull->Write();

  if(doSubHisto){
    //
    TDirectory * res5_10 = rootDir->mkdir("[5-10]GeV");
    res5_10->cd();

    hResolution_5_10->Write();
    hTDRResolution_5_10->Write();
    hPull_5_10->Write();

    TDirectory * res10_40 = rootDir->mkdir("[10-40]GeV");
    res10_40->cd();

    hResolution_10_40->Write();
    hTDRResolution_10_40->Write();
    hPull_10_40->Write();

    TDirectory * res40_70 = rootDir->mkdir("[40-70]GeV");
    res40_70->cd();

    hResolution_40_70->Write();
    hTDRResolution_40_70->Write();
    hPull_40_70->Write();

    TDirectory * res70_100 = rootDir->mkdir("[70-100]GeV");
    res70_100->cd();

    hResolution_70_100->Write();
    hTDRResolution_70_100->Write();
    hPull_70_100->Write();

    // eta range |eta|<0.8
    TDirectory * res08 = rootDir->mkdir("|eta|<0.8");
    res08->cd();

    hResolution_08->Write();
    hTDRResolution_08->Write();
    hPull_08->Write();
  
    // eta range 0.8<|eta|<1.2
    TDirectory * res08_12 = rootDir->mkdir("0.8<|eta|<1.2");
    res08_12->cd();

    hResolution_08_12->Write(); 
    hTDRResolution_08_12->Write(); 
    hPull_08_12->Write(); 
  
    // eta range 1.2<|eta|<2.1
    TDirectory * res12_21 = rootDir->mkdir("1.2<|eta|<2.1");
    res12_21->cd();

    hResolution_12_21->Write(); 
    hTDRResolution_12_21->Write();
    hPull_12_21->Write(); 

    // eta range 1.2<|eta|<2.4
    TDirectory * res12_24 = rootDir->mkdir("1.2<|eta|<2.4");
    res12_24->cd();

    hResolution_12_24->Write(); 
    hTDRResolution_12_24->Write(); 
    hPull_12_24->Write(); 

    // Endacap Plus
    TDirectory * endcapP = rootDir->mkdir("EndCap+");
    endcapP->cd();

    // eta range 1.2<eta<2.1
    hResolution_12_21_plus->Write(); 
    hTDRResolution_12_21_plus->Write(); 
    hPull_12_21_plus->Write();  

    // eta range 1.2<eta<2.4
    hResolution_12_24_plus->Write();  
    hTDRResolution_12_24_plus->Write(); 
    hPull_12_24_plus->Write();  

    // Endacap Plus
    TDirectory * endcapM = rootDir->mkdir("EndCap-");
    endcapM->cd();
  
    // eta range -2.1<eta<-1.2
    hResolution_12_21_minus->Write();  
    hTDRResolution_12_21_minus->Write();  
    hPull_12_21_minus->Write();  

    // eta range -2.4<eta<-1.2
    hResolution_12_24_minus->Write();  
    hTDRResolution_12_24_minus->Write();  
    hPull_12_24_minus->Write();  
  }
} 

double HTrack::pull(double rec,double sim, double sigmarec){
  return (rec-sim)/sigmarec;
}

double HTrack::resolution(double rec,double sim){
  return (rec-sim)/sim;
}


void HTrack::Fill(TrajectoryStateOnSurface &tsos){ 
  Fill(*tsos.freeState());
}

void HTrack::Fill(FreeTrajectoryState &fts){
  
  hVariables->Fill(fts.momentum().mag(),
		   fts.momentum().perp(),
		   fts.momentum().eta(),
		   fts.momentum().phi(),
		   fts.charge());  
}

void HTrack::FillDeltaR(double deltaR){
  hVariables->FillDeltaR(deltaR);
}


double HTrack::computeEfficiency(HTrackVariables *sim){
  return hVariables->computeEfficiency(sim);
}


void HTrack::computeResolutionAndPull(TrajectoryStateOnSurface& tsos, SimTrack& simTrack){
  computeResolutionAndPull(*tsos.freeState(),simTrack);
}


void HTrack::computeResolutionAndPull(FreeTrajectoryState& fts, SimTrack& simTrack){
  

  // Global Resolution
  computeResolution(fts,simTrack,hResolution); 
  computePull(fts,simTrack,hPull); 

  // TDR Resolution
  computeTDRResolution(fts,simTrack,hTDRResolution);
  // computeTDRPull(fts,simTracks,hTDRPull); 


  hVariables->Fill(simTrack.momentum().perp(),
		   simTrack.momentum().eta(),
		   simTrack.momentum().phi()); //FIXME


  if(doSubHisto){
    // [5-10] GeV range
    if(simTrack.momentum().perp() <10 ){  
      computeResolution(fts,simTrack,hResolution_5_10);
      computeTDRResolution(fts,simTrack,hTDRResolution_5_10);
      computePull(fts,simTrack,hPull_5_10);
    }

    // [10-40] GeV range
    if(simTrack.momentum().perp() >=10 && simTrack.momentum().perp() < 40){
      computeResolution(fts,simTrack,hResolution_10_40); 
      computeTDRResolution(fts,simTrack,  hTDRResolution_10_40);
      computePull(fts,simTrack,hPull_10_40);
    }
  
    // [40-70] GeV range
    if(simTrack.momentum().perp() >=40 && simTrack.momentum().perp() < 70){
      computeResolution(fts,simTrack,hResolution_40_70); 
      computeTDRResolution(fts,simTrack,hTDRResolution_40_70);
      computePull(fts,simTrack,hPull_40_70); 
    }

    // [70-100] GeV range
    if(simTrack.momentum().perp() >= 70 && simTrack.momentum().perp() < 100){	       
      computeResolution(fts,simTrack,hResolution_70_100); 
      computeTDRResolution(fts,simTrack,hTDRResolution_70_100);
      computePull(fts,simTrack,hPull_70_100); 
    }
  
    // eta range |eta|<0.8
    if(abs(simTrack.momentum().eta()) <= 0.8){
      computeResolution(fts,simTrack,hResolution_08); 
      computeTDRResolution(fts,simTrack,  hTDRResolution_08);
      computePull(fts,simTrack,hPull_08);
    }

    // eta range 0.8<|eta|<1.2
    if( abs(simTrack.momentum().eta()) > 0.8 && abs(simTrack.momentum().eta()) <= 1.2 ){
      computeResolution(fts,simTrack,hResolution_08_12); 
      computeTDRResolution(fts,simTrack,  hTDRResolution_08_12);
      computePull(fts,simTrack,hPull_08_12);
    }

    // eta range 1.2<|eta|<2.1
    if( abs(simTrack.momentum().eta()) > 1.2 && abs(simTrack.momentum().eta()) <= 2.1 ){
      computeResolution(fts,simTrack,hResolution_12_21); 
      computeTDRResolution(fts,simTrack,  hTDRResolution_12_21);
      computePull(fts,simTrack,hPull_12_21);
    
      if(simTrack.momentum().eta() > 0){
	computeResolution(fts,simTrack,hResolution_12_21_plus); 
	computeTDRResolution(fts,simTrack,  hTDRResolution_12_21_plus);
	computePull(fts,simTrack,hPull_12_21_plus);
      }
      else{
	computeResolution(fts,simTrack,hResolution_12_21_minus); 
	computeTDRResolution(fts,simTrack,  hTDRResolution_12_21_minus);
	computePull(fts,simTrack,hPull_12_21_minus);
      }  
    }

    // eta range 1.2<|eta|<2.4
    if( abs(simTrack.momentum().eta()) > 1.2 && abs(simTrack.momentum().eta()) <= 2.4 ){
      computeResolution(fts,simTrack,hResolution_12_24); 
      computeTDRResolution(fts,simTrack,  hTDRResolution_12_24);
      computePull(fts,simTrack,hPull_12_24);

      if(simTrack.momentum().eta() > 0){
	computeResolution(fts,simTrack,hResolution_12_24_plus); 
	computeTDRResolution(fts,simTrack,  hTDRResolution_12_24_plus);
	computePull(fts,simTrack,hPull_12_24_plus);
      }
      else{
	computeResolution(fts,simTrack,hResolution_12_24_minus); 
	computeTDRResolution(fts,simTrack,  hTDRResolution_12_24_minus);
	computePull(fts,simTrack,hPull_12_24_minus);
      }
    }
  }
}

void HTrack::computeResolution(FreeTrajectoryState &fts,
			       SimTrack& simTrack,
			       HResolution* hReso){


  hReso->Fill(simTrack.momentum().mag(),
	      simTrack.momentum().perp(),
	      simTrack.momentum().eta(),
	      simTrack.momentum().phi(),
	      resolution(fts.momentum().mag(),simTrack.momentum().mag()),
	      resolution(fts.momentum().perp(),simTrack.momentum().perp()),
	      fts.momentum().eta()-simTrack.momentum().eta(),
	      fts.momentum().phi()-simTrack.momentum().phi(),
	      fts.charge()+simTrack.type()/abs(simTrack.type())); // FIXME
}

void HTrack::computeTDRResolution(FreeTrajectoryState &fts,
				  SimTrack& simTrack,
				  HResolution* hReso){

  int simCharge = -simTrack.type()/ abs(simTrack.type());

  double invSimP =  (simTrack.momentum().mag() == 0 ) ? 0 : simTrack.momentum().mag();
  double signedInverseRecMom = (simTrack.momentum().mag() == 0 ) ? 0 : fts.signedInverseMomentum();

  hReso->Fill(simTrack.momentum().mag(),
	      simTrack.momentum().perp(),
	      simTrack.momentum().eta(),
	      simTrack.momentum().phi(),
	      resolution(signedInverseRecMom , simCharge*invSimP ),
	      resolution(fts.charge()/fts.momentum().perp(),simCharge/simTrack.momentum().perp()));
}

void HTrack::computePull(FreeTrajectoryState &fts,
			 SimTrack& simTrack,
			 HResolution* hReso){

  // x,y,z, px,py,pz
  AlgebraicSymMatrix errors = fts.cartesianError().matrix_old();
  
  double partialPterror = errors[3][3]*pow(fts.momentum().x(),2) + errors[4][4]*pow(fts.momentum().y(),2);
  
  // sqrt( (px*spx)^2 + (py*spy)^2 ) / pt
  double pterror = sqrt(partialPterror)/fts.momentum().perp();
  
  // sqrt( (px*spx)^2 + (py*spy)^2 + (pz*spz)^2 ) / p
  double perror = sqrt(partialPterror+errors[5][5]*pow(fts.momentum().z(),2))/fts.momentum().mag();

  double phierror = sqrt(fts.curvilinearError().matrix_old()[2][2]);

  double etaerror = sqrt(fts.curvilinearError().matrix_old()[1][1])*abs(sin(fts.momentum().theta()));


  hReso->Fill(simTrack.momentum().mag(),
	      simTrack.momentum().perp(),
	      simTrack.momentum().eta(),
	      simTrack.momentum().phi(),
	      pull(fts.momentum().mag(),simTrack.momentum().mag(),perror),
	      pull(fts.momentum().perp(),simTrack.momentum().perp(),pterror),
	      pull(fts.momentum().eta(),simTrack.momentum().eta(),etaerror),
	      pull(fts.momentum().phi(),simTrack.momentum().phi(),phierror),
	      pull(fts.charge() , -simTrack.type()/ abs(simTrack.type()), 1.)); // FIXME
}
