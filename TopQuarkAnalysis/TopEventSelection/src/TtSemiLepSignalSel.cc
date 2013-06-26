#include "TopQuarkAnalysis/TopEventSelection/interface/TtSemiLepSignalSel.h"
#include "TVector3.h"

TtSemiLepSignalSel::TtSemiLepSignalSel(){}

TtSemiLepSignalSel::TtSemiLepSignalSel(const std::vector<pat::Jet>& topJets, const math::XYZTLorentzVector& lepton, 
                                 const edm::View<pat::MET>& MET)
{ //function

  unsigned int nJetsMax = topJets.size();
  
  var_MET = MET.begin()->et();
  var_sumEt = 0.;
  
  math::XYZTLorentzVector Jetsum(0.,0.,0.,0.);
  
  for(unsigned int i=0; i<nJetsMax; i++) {
    math::XYZTLorentzVector aJet = topJets[i].p4();
    Jetsum += aJet;
    var_sumEt += topJets[i].et();
  }
  massalljets = Jetsum.M();
  
  var_lepeta = lepton.Eta();

  math::XYZTLorentzVector Met = MET.begin()->p4();
  math::XYZTLorentzVector Lep = lepton;
  double Etjet[4];
  double Jetjet[6];
  double dijetmass;
  var_mindijetmass = 99999.;
  var_maxdijetmass = -1.;
  int counter = 0;
  for(int i=0; i<4; i++) {
    math::XYZTLorentzVector aJet = topJets[i].p4();
    Etjet[i] = aJet.Et();
    for(int j=i+1; j<4; j++) {
      math::XYZTLorentzVector asecJet = topJets[j].p4();
      dijetmass = (aJet+asecJet).M();
      if(dijetmass<var_mindijetmass) var_mindijetmass = dijetmass;
      if(dijetmass>var_maxdijetmass) var_maxdijetmass = dijetmass;
      counter++;
    }
  }

  var_Et1 = Etjet[0];
   
  var_dphiMETlepton = DeltaPhi(Met,Lep);



  counter=0;
  for(int i=0; i<4; i++) {
    math::XYZTLorentzVector aJet = topJets[i].p4();
    for(int j=i+1; j<4; j++) {
      math::XYZTLorentzVector asecJet = topJets[j].p4();
      Jetjet[counter] = fabs(aJet.Eta()-asecJet.Eta());
      counter++;
    }
  }

  var_detajet2jet3 = Jetjet[3];
  var_detajet3jet4 = Jetjet[5];

 
  double Lepjet[4];
  var_mindRjetlepton = 99999.;
  for(int i=0; i<4; i++) {
    math::XYZTLorentzVector aJet = topJets[i].p4();
    Lepjet[i] = DeltaR(Lep,aJet);
    if(Lepjet[i]<var_mindRjetlepton) var_mindRjetlepton = Lepjet[i];
  }
  
}

double TtSemiLepSignalSel::DeltaPhi(const math::XYZTLorentzVector& v1, const math::XYZTLorentzVector& v2)
{
  double dPhi = fabs(v1.Phi() - v2.Phi());
  if (dPhi > TMath::Pi()) dPhi =  2*TMath::Pi() - dPhi;
  return dPhi;
}

double TtSemiLepSignalSel::DeltaR(const math::XYZTLorentzVector& v1, const math::XYZTLorentzVector& v2)
{
  double dPhi = DeltaPhi(v1,v2);
  double dR = TMath::Sqrt((v1.Eta()-v2.Eta())*(v1.Eta()-v2.Eta())+dPhi*dPhi);
  return dR;
}

TtSemiLepSignalSel::~TtSemiLepSignalSel() 
{
}
