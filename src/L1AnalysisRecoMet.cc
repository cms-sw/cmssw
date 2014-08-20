#include "L1TriggerDPG/L1Ntuples/interface/L1AnalysisRecoMet.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalSeverityLevelAlgo.h"

#include <TVector2.h>


L1Analysis::L1AnalysisRecoMet::L1AnalysisRecoMet() 
{
}

L1Analysis::L1AnalysisRecoMet::~L1AnalysisRecoMet()
{
}

void L1Analysis::L1AnalysisRecoMet::SetMet(const edm::Handle<reco::CaloMETCollection> recoMet)
{ 
  const reco::CaloMETCollection *metCol = recoMet.product();
  const reco::CaloMET theMet = metCol->front();

  recoMet_.met    = theMet.et();
  recoMet_.metPhi = theMet.phi();
  recoMet_.sumEt  = theMet.sumEt();

}

void L1Analysis::L1AnalysisRecoMet::SetHtMht(const edm::Handle<reco::CaloJetCollection> caloJets, float jetptThreshold)
{  
  float mHx = 0.;
  float mHy = 0.;

  recoMet_.Ht     = 0;
  recoMet_.mHt    = -999;
  recoMet_.mHtPhi = -999;
  
  for (reco::CaloJetCollection::const_iterator calojet = caloJets->begin(); calojet!=caloJets->end(); ++calojet)
  {
    if (calojet->pt()>jetptThreshold){
      mHx += -1.*calojet->px();
      mHy += -1.*calojet->py();
      recoMet_.Ht  += calojet->pt();
    }
  }

  TVector2 *tv2 = new TVector2(mHx,mHy);
  
  recoMet_.mHt	= tv2->Mod();
  recoMet_.mHtPhi= tv2->Phi();

}

void L1Analysis::L1AnalysisRecoMet::SetECALFlags(const edm::ESHandle<EcalChannelStatus> chStatus,
						 const edm::Handle<EcalRecHitCollection> ebRecHits,
						 const edm::Handle<EcalRecHitCollection> eeRecHits,
                                                 const EcalSeverityLevelAlgo* sevlv)
{
  int ecalFlag=0;
  
  // loop over EB rechits
  for(EcalRecHitCollection::const_iterator  rechit = ebRecHits->begin();
      rechit != ebRecHits->end();
      ++rechit){

    EBDetId eid(rechit->id());
    //rechit->recoFlag();
    //rechit->chi2();
    //rechit->outOfTimeChi2();

    //int flag = EcalSeverityLevelAlgo::severityLevel( eid, *ebRecHits, *chStatus );
    int flag = sevlv->severityLevel( eid, *ebRecHits);
    if (flag>ecalFlag) ecalFlag = flag;
  }
  
  // not clear what flags are in EE....  don't use them yet

  recoMet_.ecalFlag = ecalFlag;

}



