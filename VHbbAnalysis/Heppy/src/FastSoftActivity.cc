//#include "PhysicsTools/Heppy/interface/ReclusterJets.h"
#include "VHbbAnalysis/Heppy/interface/FastSoftActivity.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "fastjet/tools/Pruner.hh"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
using namespace std;

//using namespace std;
using namespace fastjet;

namespace heppy{

FastSoftActivity::FastSoftActivity(const std::vector<LorentzVector> & objects, double ktpower, double rparam,
     const LorentzVector &j1,const LorentzVector &j2,double sumDeltaRMin) : 
    ktpower_(ktpower), rparam_(rparam), j1_(j1),j2_(j2),sumDeltaRMin_(sumDeltaRMin)
{
  // define jet inputs
  fjInputs_.clear();
  int index=0;
  for (const LorentzVector &o : objects) {
    double dr1=Geom::deltaR(o,j1);
    double dr2=Geom::deltaR(o,j2);
    if(dr1+dr2>sumDeltaRMin) {
     fastjet::PseudoJet j(o.Px(),o.Py(),o.Pz(),o.E());
     j.set_user_index(index);  // in case we want to know which piece ended where
     fjInputs_.push_back(j);
    }
    index++;
  }

  // choose a jet definition
  fastjet::JetDefinition jet_def;

  // prepare jet def 
  if (ktpower_ == 1.0) {
    jet_def = JetDefinition(kt_algorithm, rparam_);
  }  else if (ktpower_ == 0.0) {
    jet_def = JetDefinition(cambridge_algorithm, rparam_);
  }  else if (ktpower_ == -1.0) {
    jet_def = JetDefinition(antikt_algorithm, rparam_);
  }  else {
    throw cms::Exception("InvalidArgument", "Unsupported ktpower value");
  }
  
  // print out some infos
  //  cout << "Clustering with " << jet_def.description() << endl;
  ///
  // define jet clustering sequence
  fjClusterSeq_ = ClusterSequencePtr( new fastjet::ClusterSequence( fjInputs_, jet_def)); 
}

std::vector<math::XYZTLorentzVector> FastSoftActivity::makeP4s(const std::vector<fastjet::PseudoJet> &jets) {
  std::vector<math::XYZTLorentzVector> JetObjectsAll;
  for (const fastjet::PseudoJet & pj : jets) {
/*    std::vector<fastjet::PseudoJet> constituents = pj.constituents();
    std::cout << "Constituents for " << pj.pt() << " " ;
    for (unsigned j = 0; j < constituents.size(); j++) {
		std::cout << constituents[j].user_index() << " ";
    }
    std::cout << std::endl;*/
    JetObjectsAll.push_back( LorentzVector( pj.px(), pj.py(), pj.pz(), pj.e() ) );
  }
  return JetObjectsAll;
}
std::vector<math::XYZTLorentzVector> FastSoftActivity::getGrouping(double ptMin) {
  // recluster jet
  inclusiveJets_ = fastjet::sorted_by_pt(fjClusterSeq_->inclusive_jets(ptMin));
  // return
  return makeP4s(inclusiveJets_);
}

}
