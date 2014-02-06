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
L1ValidatorHists::L1ValidatorHists(DQMStore *dbe){
  _dbe=dbe;
  //_dbe->setCurrentFolder("L1T/L1Extra");

  Name[0]="IsoEG";
  Name[1]="NonIsoEG";
  Name[2]="CenJet";
  Name[3]="ForJet";
  Name[4]="TauJet";
  Name[5]="Muon";
}

void L1ValidatorHists::Book(){
  NEvents=0;

  for(int i=0; i<Type::Number; i++){
    N[i] = _dbe->book1D( (Name[i]+"_N").c_str(), (Name[i]+" Number").c_str(), 5, -0.5, 4.5);
    N_Pt[i] = new TH1F( (Name[i]+"_N_Pt").c_str(), (Name[i]+" Number").c_str(), 20, 0, 100);
    N_Eta[i] = new TH1F( (Name[i]+"_N_Eta").c_str(), (Name[i]+" Number").c_str(), 20, -4, 4);
    Eff_Pt[i] = _dbe->book1D( (Name[i]+"_Eff_Pt").c_str(), (Name[i]+" Efficiency").c_str(), 20, 0, 100);
    Eff_Eta[i] = _dbe->book1D( (Name[i]+"_Eff_Eta").c_str(), (Name[i]+" Efficiency").c_str(), 20, -4, 4);
    TurnOn_15[i] = _dbe->book1D( (Name[i]+"_TurnOn_15").c_str(), (Name[i]+" Turn On (15 GeV)").c_str(), 20, 0, 100);
    TurnOn_30[i] = _dbe->book1D( (Name[i]+"_TurnOn_30").c_str(), (Name[i]+" Turn On (30 GeV)").c_str(), 20, 0, 100);
    dR[i] = _dbe->book1D( (Name[i]+"_dR").c_str(), (Name[i]+" dR").c_str(), 20, 0, 1);
    dPt[i] = _dbe->book1D( (Name[i]+"_dPt").c_str(), (Name[i]+" dPt").c_str(), 20, -1, 1);

    N[i]->getTH1()->Sumw2();
    N_Pt[i]->Sumw2();
    N_Eta[i]->Sumw2();
    Eff_Pt[i]->getTH1()->Sumw2();
    Eff_Eta[i]->getTH1()->Sumw2();
    TurnOn_15[i]->getTH1()->Sumw2();
    TurnOn_30[i]->getTH1()->Sumw2();
    dR[i]->getTH1()->Sumw2();
    dPt[i]->getTH1()->Sumw2();
  }

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
    Eff_Pt[i]->getTH1F()->Divide(N_Pt[i]);
    TurnOn_15[i]->getTH1F()->Divide(N_Pt[i]);
    TurnOn_30[i]->getTH1F()->Divide(N_Pt[i]);
    Eff_Eta[i]->getTH1F()->Divide(N_Eta[i]);
    N[i]->getTH1()->Scale(1./N[i]->getEntries());
    dR[i]->getTH1()->Scale(1./dR[i]->getEntries());
    dPt[i]->getTH1()->Scale(1./dPt[i]->getEntries());
  }
}

void L1ValidatorHists::Write(){
  for(int i=0; i<Type::Number; i++){
    N[i]->getTH1()->Write();
    Eff_Pt[i]->getTH1()->Write();
    Eff_Eta[i]->getTH1()->Write();
    TurnOn_15[i]->getTH1()->Write();
    TurnOn_30[i]->getTH1()->Write();
    dR[i]->getTH1()->Write();
    dPt[i]->getTH1()->Write();
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
