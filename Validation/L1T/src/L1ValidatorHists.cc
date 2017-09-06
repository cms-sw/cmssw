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
L1ValidatorHists::L1ValidatorHists(){
//  Name[0]="IsoEG"; // Run I legacy
//  Name[1]="NonIsoEG";
//  Name[2]="CenJet";
//  Name[3]="ForJet";
//  Name[4]="TauJet";
//  Name[5]="Muon";
  Name[0]="Egamma";
  Name[1]="Jet";
  Name[2]="Tau";
  Name[3]="Muon";

}
L1ValidatorHists::~L1ValidatorHists(){}

void L1ValidatorHists::Book(DQMStore::IBooker &iBooker){
  NEvents=0;

  float ptbins[14] = { 0,5,10,15,20,25,30,35, 40, 50, 60, 80, 120, 160}; 
  int Nptbin = 13;

  for(int i=0; i<Type::Number; i++){
    N[i] = iBooker.book1D( (Name[i]+"_N").c_str(), ("L1 " + Name[i]+" Number with BX=0").c_str(), 16, -0.5, 15.5);

    Eff_Pt[i] = iBooker.book1D( (Name[i]+"_Eff_Pt").c_str(), (Name[i]+" Efficiency vs Pt; Gen p_{T} [GeV]; L1T Efficiency").c_str(), Nptbin, ptbins);
    Eff_Pt_Denom[i] = iBooker.book1D( (Name[i]+"_Eff_Pt_Denom").c_str(), (Name[i]+" Efficiency vs Pt Denom; Gen p_{T} [GeV]; Entries").c_str(), Nptbin, ptbins);
    Eff_Pt_Nomin[i] = iBooker.book1D( (Name[i]+"_Eff_Pt_Nomin").c_str(), (Name[i]+" Efficiency vs Pt Nomin; Gen p_{T} [GeV]; Entries").c_str(), Nptbin, ptbins);
    Eff_Eta[i] = iBooker.book1D( (Name[i]+"_Eff_Eta").c_str(), (Name[i]+" Efficiency vs #eta (Gen p_{T} > 10GeV); Gen #eta; L1T Efficiency").c_str(), 80, -4, 4);
    Eff_Eta_Denom[i] = iBooker.book1D( (Name[i]+"_Eff_Eta_Denom").c_str(), (Name[i]+" Efficiency vs #eta Denom; Gen #eta; Entries").c_str(), 80, -4, 4);
    Eff_Eta_Nomin[i] = iBooker.book1D( (Name[i]+"_Eff_Eta_Nomin").c_str(), (Name[i]+" Efficiency vs #eta Nomin; Gen #eta; Entries").c_str(), 80, -4, 4);
    TurnOn_15[i] = iBooker.book1D( (Name[i]+"_TurnOn_15").c_str(), (Name[i]+" Turn On (15 GeV); Gen p_{T} [GeV]; L1T Efficiency").c_str(), Nptbin, ptbins);
    TurnOn_15_Denom[i] = iBooker.book1D( (Name[i]+"_TurnOn_15_Denom").c_str(), (Name[i]+" Turn On (15 GeV) Denom; Gen p_{T} [GeV]; Entries").c_str(), Nptbin, ptbins);
    TurnOn_15_Nomin[i] = iBooker.book1D( (Name[i]+"_TurnOn_15_Nomin").c_str(), (Name[i]+" Turn On (15 GeV) Nomin; Gen p_{T} [GeV]; Entries").c_str(), Nptbin, ptbins);
    TurnOn_30[i] = iBooker.book1D( (Name[i]+"_TurnOn_30").c_str(), (Name[i]+" Turn On (30 GeV); Gen p_{T} [GeV]; L1T Efficiency").c_str(), Nptbin, ptbins);
    TurnOn_30_Denom[i] = iBooker.book1D( (Name[i]+"_TurnOn_30_Denom").c_str(), (Name[i]+" Turn On (30 GeV) Denom; Gen p_{T} [GeV]; Entries").c_str(), Nptbin, ptbins);
    TurnOn_30_Nomin[i] = iBooker.book1D( (Name[i]+"_TurnOn_30_Nomin").c_str(), (Name[i]+" Turn On (30 GeV) Nomin; Gen p_{T} [GeV]; Entries").c_str(), Nptbin, ptbins);
    dR[i] = iBooker.book1D( (Name[i]+"_dR").c_str(), (Name[i]+" #DeltaR; #DeltaR(L1 object, Gen object); Entries").c_str(), 40, 0, 1);
    dR_vs_Pt[i] = iBooker.book2D( (Name[i]+"_dR_vs_Pt").c_str(), (Name[i]+" #DeltaR vs p_{T}; Gen p_{T} [GeV]; #DeltaR(L1 object, Gen object); Entries").c_str(), 12, 0, 120, 40, 0, 1);
    dPt[i] = iBooker.book1D( (Name[i]+"_dPt").c_str(), (Name[i]+" #Deltap_{T}; (p_{T}^{L1}-p_{T}^{Gen})/p_{T}^{Gen}; Entries").c_str(), 100, -2, 2);
    dPt_vs_Pt[i] = iBooker.book2D( (Name[i]+"_dPt_vs_Pt").c_str(), (Name[i]+" #Deltap_{T} vs p_{T}; Gen p_{T} [GeV]; (p_{T}^{L1}-p_{T}^{Gen})/p_{T}^{Gen}; Entries").c_str(), 12, 0, 120, 40, -2, 2);
  }

}

void L1ValidatorHists::Fill(int i, const reco::LeafCandidate *GenPart, const reco::LeafCandidate *L1Part){
  double GenPartPt = GenPart->pt();
  // fill the overflow in the last bin
  if(GenPart->pt()>=160.0) GenPartPt = 159.0;
  if(L1Part==nullptr) {
     Eff_Pt_Denom[i]->Fill(GenPartPt);
     if(GenPart->pt()>10)Eff_Eta_Denom[i]->Fill(GenPart->eta());
     TurnOn_15_Denom[i]->Fill(GenPartPt);
     TurnOn_30_Denom[i]->Fill(GenPartPt);
  } else {
     double idR = reco::deltaR(GenPart->eta(), GenPart->phi(), L1Part->eta(), L1Part->phi());
     bool matched  = idR < 0.15;
     Eff_Pt_Denom[i]->Fill(GenPartPt);
     if(GenPart->pt()>10)Eff_Eta_Denom[i]->Fill(GenPart->eta());
     if(matched)Eff_Pt_Nomin[i]->Fill(GenPartPt);
     if(matched && GenPart->pt()>10)Eff_Eta_Nomin[i]->Fill(GenPart->eta());
     TurnOn_15_Denom[i]->Fill(GenPartPt);
     TurnOn_30_Denom[i]->Fill(GenPartPt);
     if(L1Part->pt()>15 && matched) TurnOn_15_Nomin[i]->Fill(GenPartPt);
     if(L1Part->pt()>30 && matched) TurnOn_30_Nomin[i]->Fill(GenPartPt);
     dR[i]->Fill(idR);
     dPt[i]->Fill( (L1Part->pt()-GenPart->pt()) / GenPart->pt() );
     dR_vs_Pt[i]->Fill(GenPart->pt(), idR);
     dPt_vs_Pt[i]->Fill( GenPart->pt(),  (L1Part->pt()-GenPart->pt()) / GenPart->pt() );
  }
}

void L1ValidatorHists::FillNumber(int i, int Number){
  N[i]->Fill(Number);
}

void L1ValidatorHists::Write(){
  for(int i=0; i<Type::Number; i++){
    N[i]->getTH1()->Write();
    Eff_Pt[i]->getTH1()->Write();
    Eff_Pt_Denom[i]->getTH1()->Write();
    Eff_Pt_Nomin[i]->getTH1()->Write();
    Eff_Eta[i]->getTH1()->Write();
    Eff_Eta_Denom[i]->getTH1()->Write();
    Eff_Eta_Nomin[i]->getTH1()->Write();
    TurnOn_15[i]->getTH1()->Write();
    TurnOn_15_Denom[i]->getTH1()->Write();
    TurnOn_15_Nomin[i]->getTH1()->Write();
    TurnOn_30[i]->getTH1()->Write();
    TurnOn_30_Denom[i]->getTH1()->Write();
    TurnOn_30_Nomin[i]->getTH1()->Write();
    dR[i]->getTH1()->Write();
    dPt[i]->getTH1()->Write();
    dR_vs_Pt[i]->getTH2F()->Write();
    dPt_vs_Pt[i]->getTH2F()->Write();
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
