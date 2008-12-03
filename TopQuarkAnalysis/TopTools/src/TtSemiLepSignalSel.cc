#include "TopQuarkAnalysis/TopTools/interface/TtSemiLepSignalSel.h"
#include "TopQuarkAnalysis/TopTools/interface/EventShapeVariables.h"
#include "TVector3.h"

TtSemiLepSignalSel::TtSemiLepSignalSel(){}

TtSemiLepSignalSel::TtSemiLepSignalSel(const std::vector<pat::Jet>& topJets, math::XYZTLorentzVector lepton, 
                                 const edm::View<pat::MET>& MET, unsigned int maxNJets)
{ 

  var_dphiMETlepton = fabs(MET.begin()->phi()-lepton.phi());
  if (var_dphiMETlepton > 3.1415927)  var_dphiMETlepton =  2*3.1415927 - var_dphiMETlepton;
  if (var_dphiMETlepton < -3.1415927) var_dphiMETlepton = -2*3.1415927 - var_dphiMETlepton;


  var_dphiMETleadingjet = fabs(MET.begin()->phi()-topJets[0].phi());
  if (var_dphiMETleadingjet > 3.1415927)  var_dphiMETleadingjet =  2*3.1415927 - var_dphiMETleadingjet;
  if (var_dphiMETleadingjet < -3.1415927) var_dphiMETleadingjet = -2*3.1415927 - var_dphiMETleadingjet;

  if(topJets.size()>=5) var_ETratiojet5jet4 = (topJets[4].et()/topJets[3].et());
  else var_ETratiojet5jet4 = 0.;

  std::vector<TVector3> p;

  TVector3 lep(lepton.px(),lepton.py(),lepton.pz());
  
  //std::cout<<"lepton -- px: "<<lepton.px()<<"  py: "<<lepton.py()<<"  pz: "<<lepton.pz()<<std::endl;
  //std::cout<<"lepton aus lep -- px: "<<lep.x()<<"  py: "<<lep.y()<<"  pz: "<<lep.z()<<std::endl;
 
  p.push_back(lep);
  //std::cout<<"lepton aus p -- px: "<<p[0].x()<<"  py: "<<p[0].y()<<"  pz: "<<p[0].z()<<std::endl;

  if(topJets.size()<maxNJets) maxNJets = topJets.size();

  //std::cout<<"maxNJets: "<<maxNJets<<std::endl;

  for(unsigned int i=0; i<maxNJets; i++) {
    TVector3 jet(topJets[i].px(),topJets[i].py(),topJets[i].pz());
    //std::cout<<"jet"<<i<<":  px: "<<topJets[i].px()<<"  py: "<<topJets[i].py()<<"  pz: "<<topJets[i].pz()<<std::endl;
    p.push_back(jet);
  }

  EventShapeVariables eventshape;

  var_aplanarity = eventshape.aplanarity(p);
  var_sphericity = eventshape.sphericity(p);
  var_circularity = eventshape.circularity(p);
  var_isotropy = eventshape.isotropy(p);


  var_sumEt = 0.;
  var_maxEta = 0.;
  for(unsigned int i=0; i<maxNJets; i++) {
    var_sumEt += topJets[i].et();
    if(i==0) var_maxEta = fabs(topJets[i].eta());
    else if(fabs(topJets[i].eta())>var_maxEta) var_maxEta = fabs(topJets[i].eta());
  }

  var_Et1 = topJets[0].et();
  var_Et2 = topJets[1].et();
  var_Et3 = topJets[2].et();
  var_Et4 = topJets[3].et();
  
  var_lepPt = TMath::Sqrt(lep.x()*lep.x()+lep.y()*lep.y());

}

TtSemiLepSignalSel::~TtSemiLepSignalSel() 
{
}
