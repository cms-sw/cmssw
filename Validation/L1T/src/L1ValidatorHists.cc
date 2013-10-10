#include "Validation/L1T/interface/L1ValidatorHists.h"

//#include <DataFormats/HepMCCandidate/interface/GenParticle.h>

#include "DataFormats/Math/interface/deltaR.h"

/*#define BOOKHISTS(TYPE) \
TYPE ## _N_Pt = new TH2F(#TYPE "_N_Pt", #TYPE " Number", 20, 0, 200); \
TYPE ## _N_Eta = new TH2F(#TYPE "_N_Eta", #TYPE " Number", 20, -4, 4); \
TYPE ## _Eff_Pt = new TH2F(#TYPE "_Eff_Pt", #TYPE " Number", 20, 0, 200); \
TYPE ## _Eff_Eta = new TH2F(#TYPE "_Eff_Eta", #TYPE " Number", 20, -4, 4); \
TYPE ## _dR = new TH2F(#TYPE "_dR", #TYPE " Number", 20, 0, 1); \
TYPE ## _dPt = new TH2F(#TYPE "_dPt", #TYPE " Number", 20, -1, 1);
*/
L1ValidatorHists::L1ValidatorHists(/*DQMStore *dbe*/){
  /*_dbe=dbe;

  IsoEG_Eff_Pt_Eta = _dbe->book2D("IsoEG_Eff_Pt_Eta", "IsoEG Efficiency", 100, 0, 200, 100, -4, 4);
  IsoEG_Fake_Pt_Eta = _dbe->book2D("IsoEG_Fake_Pt_Eta", "IsoEG Fake Rate", 100, 0, 200, 100, -4, 4);
  IsoEG_dR_Pt_Eta = _dbe->book2D("IsoEG_dR_Pt", "IsoEG dR", 100, 0, 200, 100, -1, 1);

  CenJet_Eff_Pt_Eta = _dbe->book2D("CenJet_Eff_Pt_Eta", "CenJet Efficiency", 100, 0, 200, 100, -4, 4);
  CenJet_Fake_Pt_Eta = _dbe->book2D("CenJet_Fake_Pt_Eta", "CenJet Fake Rate", 100, 0, 200, 100, -4, 4);
  CenJet_dR_Pt_Eta = _dbe->book2D("CenJet_dR_Pt", "CenJet dR", 100, 0, 200, 100, -1, 1);*/

  Name[0]="IsoEG";
  Name[1]="NonIsoEG";
  Name[2]="CenJet";
  Name[3]="ForJet";
  Name[4]="TauJet";
  Name[5]="Muon";

  NEvents=0;

  for(int i=0; i<Type::Number; i++){
    N[i] = new TH1F( (Name[i]+"_N").c_str(), (Name[i]+" Number").c_str(), 5, -0.5, 4.5);
    N_Pt[i] = new TH1F( (Name[i]+"_N_Pt").c_str(), (Name[i]+" Number").c_str(), 20, 0, 100);
    N_Eta[i] = new TH1F( (Name[i]+"_N_Eta").c_str(), (Name[i]+" Number").c_str(), 20, -4, 4);
    Eff_Pt[i] = new TH1F( (Name[i]+"_Eff_Pt").c_str(), (Name[i]+" Efficiency").c_str(), 20, 0, 100);
    Eff_Eta[i] = new TH1F( (Name[i]+"_Eff_Eta").c_str(), (Name[i]+" Efficiency").c_str(), 20, -4, 4);
    TurnOn_15[i] = new TH1F( (Name[i]+"_TurnOn_15").c_str(), (Name[i]+" Turn On (15 GeV)").c_str(), 20, 0, 100);
    TurnOn_30[i] = new TH1F( (Name[i]+"_TurnOn_30").c_str(), (Name[i]+" Turn On (30 GeV)").c_str(), 20, 0, 100);
    dR[i] = new TH1F( (Name[i]+"_dR").c_str(), (Name[i]+" dR").c_str(), 20, 0, 1);
    dPt[i] = new TH1F( (Name[i]+"_dPt").c_str(), (Name[i]+" dPt").c_str(), 20, -1, 1);
  }

  /*BOOKHISTS(IsoEG)
  BOOKHISTS(NonIsoEG)
  BOOKHISTS(TauJet)
  BOOKHISTS(ForJet)
  BOOKHISTS(TauJet)
  BOOKHISTS(Muon)*/


  /*IsoEG_N_Pt_Eta = new TH2F("IsoEG_N_Pt_Eta", "IsoEG Number", 20, 0, 200, 20, -4, 4);
  IsoEG_Eff_Pt = new TH2F("IsoEG_Eff_Pt", "IsoEG Efficiency", 20, 0, 200);
  IsoEG_Eff_Eta = new TH2F("IsoEG_Eff_Eta", "IsoEG Efficiency", 20, -4, 4);
  IsoEG_dR_Pt = new TH2F("IsoEG_dR_Pt", "IsoEG dR", 20, 0, 200, 20, 0, 1);
  IsoEG_dPt_Pt = new TH2F("IsoEG_dPt_Pt", "IsoEG Pt Resolution", 20, 0, 200, 20, -.5, .5);

  NonIsoEG_N_Pt_Eta = new TH2F("NonIsoEG_N_Pt_Eta", "NonIsoEG Number", 20, 0, 200, 20, -4, 4);
  NonIsoEG_Eff_Pt_Eta = new TH2F("NonIsoEG_Eff_Pt_Eta", "NonIsoEG Efficiency", 20, 0, 200, 20, -4, 4);
  NonIsoEG_Fake_Pt_Eta = new TH2F("NonIsoEG_Fake_Pt_Eta", "NonIsoEG Fake Rate", 20, 0, 200, 20, -4, 4);
  NonIsoEG_dR_Pt = new TH2F("NonIsoEG_dR_Pt", "NonIsoEG dR", 20, 0, 200, 20, 0, 1);
  NonIsoEG_dPt_Pt = new TH2F("NonIsoEG_dPt_Pt", "NonIsoEG Pt Resolution", 20, 0, 200, 20, -.5, .5);

  CenJet_N_Pt_Eta = new TH2F("CenJet_N_Pt_Eta", "CenJet Number", 20, 0, 200, 20, -4, 4);
  CenJet_Eff_Pt_Eta = new TH2F("CenJet_Eff_Pt_Eta", "CenJet Efficiency", 20, 0, 200, 20, -4, 4);
  CenJet_Fake_Pt_Eta = new TH2F("CenJet_Fake_Pt_Eta", "CenJet Fake Rate", 20, 0, 200, 20, -4, 4);
  CenJet_dR_Pt = new TH2F("CenJet_dR_Pt", "CenJet dR", 20, 0, 200, 20, 0, 1);
  CenJet_dPt_Pt = new TH2F("CenJet_dPt_Pt", "CenJet Pt Resolution", 20, 0, 200, 20, -.5, .5);

  Muon_N_Pt_Eta = new TH2F("Muon_N_Pt_Eta", "Muon Number", 20, 0, 200, 20, -4, 4);
  Muon_Eff_Pt_Eta = new TH2F("Muon_Eff_Pt_Eta", "Muon Efficiency", 20, 0, 200, 20, -4, 4);
  Muon_Fake_Pt_Eta = new TH2F("Muon_Fake_Pt_Eta", "Muon Fake Rate", 20, 0, 200, 20, -4, 4);
  Muon_dR_Pt = new TH2F("Muon_dR_Pt", "Muon dR", 20, 0, 200, 20, 0, 1);
  Muon_dPt_Pt = new TH2F("Muon_dPt_Pt", "Muon Pt Resolution", 20, 0, 200, 20, -.5, .5);
  */
}

void L1ValidatorHists::Fill(int i, const reco::LeafCandidate *GenPart, const reco::LeafCandidate *RecoPart){
  N_Pt[i]->Fill(GenPart->pt());
  N_Eta[i]->Fill(GenPart->eta());

  if(RecoPart==NULL) return;

  Eff_Pt[i]->Fill(GenPart->pt());
  if(RecoPart->pt()>15) TurnOn_15[i]->Fill(GenPart->pt());
  if(RecoPart->pt()>30) TurnOn_30[i]->Fill(GenPart->pt());
  Eff_Eta[i]->Fill(GenPart->eta());
  dR[i]->Fill(reco::deltaR(GenPart->eta(), GenPart->phi(), RecoPart->eta(), RecoPart->phi()));
  dPt[i]->Fill( (RecoPart->pt()-GenPart->pt()) / GenPart->pt() );
}

void L1ValidatorHists::FillNumber(int i, int Number){
  N[i]->Fill(Number);
}

void L1ValidatorHists::Normalize(){
  for(int i=0; i<Type::Number; i++){
    Eff_Pt[i]->Divide(N_Pt[i]);
    TurnOn_15[i]->Divide(N_Pt[i]);
    TurnOn_30[i]->Divide(N_Pt[i]);
    Eff_Eta[i]->Divide(N_Eta[i]);
    dR[i]->Scale(1./dR[i]->Integral(0, -1));
    dPt[i]->Scale(1./dPt[i]->Integral(0, -1));
  }
}

void L1ValidatorHists::Write(){
  for(int i=0; i<Type::Number; i++){
    N[i]->Write();
    Eff_Pt[i]->Write();
    Eff_Eta[i]->Write();
    TurnOn_15[i]->Write();
    TurnOn_30[i]->Write();
    dR[i]->Write();
    dPt[i]->Write();
  }
}

/*void L1ValidatorHists::NormalizeSlices(TH2F *Hist){
  int NBinsX = Hist->GetNbinsX();
  int NBinsY = Hist->GetNbinsY();
  for(int i=0; i<NBinsX+2; i++){
    float Total = Hist->Integral(i, i, 0, -1);
    if(Total == 0) continue;
    for(int j=0; j<NBinsY+2; j++){
      Hist->SetBinContent(i,j, Hist->GetBinContent(i,j)/Total);
    }
  }
}
*/
