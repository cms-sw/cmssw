#include "SUSYBSMAnalysis/HSCP/interface/CandidateSelector.h"

using namespace edm;
using namespace reco;
using namespace susybsm;

CandidateSelector::CandidateSelector(const edm::ParameterSet& iConfig) {
  isTrack = iConfig.getParameter<bool>("onlyConsiderTrack");
  isMuon = iConfig.getParameter<bool>("onlyConsiderMuon");
  isMuonSTA = iConfig.getParameter<bool>("onlyConsiderMuonSTA");
  isMuonGB = iConfig.getParameter<bool>("onlyConsiderMuonGB");
  isMuonTK = iConfig.getParameter<bool>("onlyConsiderMuonTK");
  isMTMuon = iConfig.getParameter<bool>("onlyConsiderMTMuon");
  isRpc = iConfig.getParameter<bool>("onlyConsiderRpc");
  isEcal = iConfig.getParameter<bool>("onlyConsiderEcal");

  minTrackHits = iConfig.getParameter<int>("minTrackHits");
  minTrackP = iConfig.getParameter<double>("minTrackP");
  minTrackPt = iConfig.getParameter<double>("minTrackPt");

  minDedx = iConfig.getParameter<double>("minDedx");

  minMuonP = iConfig.getParameter<double>("minMuonP");
  minMuonPt = iConfig.getParameter<double>("minMuonPt");
  minSAMuonPt = iConfig.getParameter<double>("minMTMuonPt");
  minMTMuonPt = iConfig.getParameter<double>("minMTMuonPt");

  maxMuTimeDtBeta = iConfig.getParameter<double>("maxMuTimeDtBeta");
  minMuTimeDtNdof = iConfig.getParameter<double>("minMuTimeDtNdof");
  maxMuTimeCscBeta = iConfig.getParameter<double>("maxMuTimeCscBeta");
  minMuTimeCscNdof = iConfig.getParameter<double>("minMuTimeCscNdof");
  maxMuTimeCombinedBeta = iConfig.getParameter<double>("maxMuTimeCombinedBeta");
  minMuTimeCombinedNdof = iConfig.getParameter<double>("minMuTimeCombinedNdof");

  maxBetaRpc = iConfig.getParameter<double>("maxBetaRpc");
  maxBetaEcal = iConfig.getParameter<double>("maxBetaEcal");
}

bool CandidateSelector::isSelected(HSCParticle& candidate) {
  if (isTrack && !candidate.hasTrackRef()) {
    return false;
  }
  if (isMuon && !candidate.hasMuonRef()) {
    return false;
  }
  if (isMuonSTA && (!candidate.hasMuonRef() || candidate.muonRef()->standAloneMuon().isNull())) {
    return false;
  }
  if (isMuonGB && (!candidate.hasMuonRef() || candidate.muonRef()->combinedMuon().isNull())) {
    return false;
  }
  if (isMuonTK && (!candidate.hasMuonRef() || candidate.muonRef()->innerTrack().isNull())) {
    return false;
  }
  if (isMTMuon && !candidate.hasMTMuonRef()) {
    return false;
  }
  if (isRpc && !candidate.hasRpcInfo()) {
    return false;
  }
  if (isEcal && !candidate.hasCaloInfo()) {
    return false;
  }

  if (candidate.hasTrackRef()) {
    if (candidate.trackRef()->found() < minTrackHits) {
      return false;
    }
    if (candidate.trackRef()->p() < minTrackP) {
      return false;
    }
    if (candidate.trackRef()->pt() < minTrackPt) {
      return false;
    }

    //      Need to be implemented using external dE/dx object
    //      if(candidate.hasDedxEstim1()   && minDedxEstimator1>=0     && candidate.dedxEstimator1    ().dEdx()<minDedxEstimator1)    {return false;}
    //      if(candidate.hasDedxDiscrim1() && minDedxDiscriminator1>=0 && candidate.dedxDiscriminator1().dEdx()<minDedxDiscriminator1){return false;}
  }

  if (candidate.hasMuonRef()) {
    if (candidate.muonRef()->p() < minMuonP) {
      return false;
    }
    if (candidate.muonRef()->pt() < minMuonPt) {
      return false;
    }

    //      Need to be implemented using external timing object
    //      if(maxMuTimeDtBeta      >=0 && 1.0/candidate.muonTimeDt().inverseBeta()       > maxMuTimeDtBeta      ){return false;}
    //      if(minMuTimeDtNdof      >=0 && 1.0/candidate.muonTimeDt().nDof()              < minMuTimeDtNdof      ){return false;}
    //      if(maxMuTimeCscBeta     >=0 && 1.0/candidate.muonTimeCsc().inverseBeta()      > maxMuTimeCscBeta     ){return false;}
    //      if(minMuTimeCscNdof     >=0 && 1.0/candidate.muonTimeCsc().nDof()             < minMuTimeCscNdof     ){return false;}
    //      if(maxMuTimeCombinedBeta>=0 && 1.0/candidate.muonTimeCombined().inverseBeta() > maxMuTimeCombinedBeta){return false;}
    //      if(minMuTimeCombinedNdof>=0 && 1.0/candidate.muonTimeCombined().nDof()        < minMuTimeCombinedNdof){return false;}
  }

  if (candidate.hasRpcInfo() && maxBetaRpc >= 0 && candidate.rpc().beta > maxBetaRpc) {
    return false;
  }

  if (candidate.hasMuonRef() && candidate.muonRef()->isStandAloneMuon()) {
    if (candidate.muonRef()->standAloneMuon()->pt() < minSAMuonPt) {
      return false;
    }
  }

  if (candidate.hasMTMuonRef()) {
    if (!candidate.MTMuonRef()->standAloneMuon().isNull()) {
      if (candidate.MTMuonRef()->standAloneMuon()->pt() < minMTMuonPt) {
        return false;
      }
    }
  }

  //      Need to be implemented using external dE/dx object
  //   if(candidate.hasCaloInfo() && maxBetaEcal>=0 && candidate.calo().ecalBeta > maxBetaEcal){return false;}

  return true;
}
