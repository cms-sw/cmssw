//
//
#include "DataFormats/Common/interface/Ref.h"

#include "TrackAssociatorByHitsImpl.h"

#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

//reco track
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackingRecHit/interface/TrackingRecHit.h"
//TrackingParticle
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticleFwd.h"
#include "SimGeneral/TrackingAnalysis/interface/SimHitTPAssociationProducer.h"
#include "SimDataFormats/EncodedEventId/interface/EncodedEventId.h"
//##---new stuff
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

using namespace reco;
using namespace std;

/* Constructor */

TrackAssociatorByHitsImpl::TrackAssociatorByHitsImpl(std::unique_ptr<TrackerHitAssociator> iAssociate,
                                                     TrackerTopology const* iTopo,
                                                     SimHitTPAssociationList const* iSimHitsTPAssoc,
                                                     SimToRecoDenomType iSimToRecoDenominator,
                                                     double iQuality_SimToReco,
                                                     double iPurity_SimToReco,
                                                     double iCut_RecoToSim,
                                                     bool iUsePixels,
                                                     bool iUseGrouped,
                                                     bool iUseSplitting,
                                                     bool iThreeHitTracksAreSpecial,
                                                     bool iAbsoluteNumberOfHits) :
  associate(std::move(iAssociate)),
  tTopo(iTopo),
  simHitsTPAssoc(iSimHitsTPAssoc),
  SimToRecoDenominator(iSimToRecoDenominator),
  quality_SimToReco(iQuality_SimToReco),
  purity_SimToReco(iPurity_SimToReco),
  cut_RecoToSim(iCut_RecoToSim),
  UsePixels(iUsePixels),
  UseGrouped(iUseGrouped),
  UseSplitting(iUseSplitting),
  ThreeHitTracksAreSpecial(iThreeHitTracksAreSpecial),
  AbsoluteNumberOfHits(iAbsoluteNumberOfHits) {}

/*
TrackAssociatorByHitsImpl::TrackAssociatorByHitsImpl (const edm::ParameterSet& conf) :  
  conf_(conf),
  AbsoluteNumberOfHits(conf_.getParameter<bool>("AbsoluteNumberOfHits")),
  SimToRecoDenominator(denomnone),
  quality_SimToReco(conf_.getParameter<double>("Quality_SimToReco")),
  purity_SimToReco(conf_.getParameter<double>("Purity_SimToReco")),
  cut_RecoToSim(conf_.getParameter<double>("Cut_RecoToSim")),
  UsePixels(conf_.getParameter<bool>("UsePixels")),
  UseGrouped(conf_.getParameter<bool>("UseGrouped")),
  UseSplitting(conf_.getParameter<bool>("UseSplitting")),
  ThreeHitTracksAreSpecial(conf_.getParameter<bool>("ThreeHitTracksAreSpecial")),
  _simHitTpMapTag(conf_.getParameter<edm::InputTag>("simHitTpMapTag"))
{
  std::string tmp = conf_.getParameter<string>("SimToRecoDenominator");
  if (tmp=="sim") {
    SimToRecoDenominator = denomsim;
  } else if (tmp == "reco") {
    SimToRecoDenominator = denomreco;
  } 

  if (SimToRecoDenominator == denomnone) {
    throw cms::Exception("TrackAssociatorByHitsImpl") << "SimToRecoDenominator not specified as sim or reco";
  }

}
*/

//
//---member functions
//

RecoToSimCollection  
TrackAssociatorByHitsImpl::associateRecoToSim(const edm::RefToBaseVector<reco::Track>& tC, 
					  const edm::RefVector<TrackingParticleCollection>& TPCollectionH) const {

  //edm::LogVerbatim("TrackAssociator") << "Starting TrackAssociatorByHitsImpl::associateRecoToSim - #tracks="<<tC.size()<<" #TPs="<<TPCollectionH.size();
  int nshared = 0;
  float quality=0;//fraction or absolute number of shared hits
  std::vector< SimHitIdpr> SimTrackIds;
  std::vector< SimHitIdpr> matchedIds; 
  RecoToSimCollection  outputCollection;
  
  //dereference the edm::Refs only once
  std::vector<TrackingParticle const*> tPC;
  tPC.reserve(tPC.size());
  for(auto const& ref: TPCollectionH) {
    tPC.push_back(&(*ref));
  }

  //get the ID of the recotrack  by hits 
  int tindex=0;
  for (edm::RefToBaseVector<reco::Track>::const_iterator track=tC.begin(); track!=tC.end(); track++, tindex++){
    matchedIds.clear();
    int ri=0;//valid rechits
    //LogTrace("TrackAssociator") << "\nNEW TRACK - track number " << tindex <<" with pt =" << (*track)->pt() << " # valid=" << (*track)->found(); 
    getMatchedIds<trackingRecHit_iterator>(matchedIds, SimTrackIds, ri, (*track)->recHitsBegin(), (*track)->recHitsEnd(), associate.get());

    //LogTrace("TrackAssociator") << "MATCHED IDS LIST BEGIN" ;
    //for(size_t j=0; j<matchedIds.size(); j++){
    //  LogTrace("TrackAssociator") << "matchedIds[j].first=" << matchedIds[j].first;
    //}
    //LogTrace("TrackAssociator") << "MATCHED IDS LIST END" ;
    //LogTrace("TrackAssociator") << "#matched ids=" << matchedIds.size() << " #tps=" << tPC.size();

    //save id for the track
    std::vector<SimHitIdpr> idcachev;
    if(!matchedIds.empty()){

      int tpindex =0;
      for (auto t = tPC.begin(); t != tPC.end(); ++t, ++tpindex) {
        //int nsimhit = (*t)->trackPSimHit(DetId::Tracker).size(); 
	//LogTrace("TrackAssociator") << "TP number " << tpindex << " pdgId=" << (*t)->pdgId() << " with number of PSimHits: "  << nsimhit;
	idcachev.clear();
	nshared = getShared(matchedIds, idcachev, **t);

	//if electron subtract double counting
	if (abs((*t)->pdgId())==11&&((*t)->g4Track_end()-(*t)->g4Track_begin())>1){
	  nshared-=getDoubleCount<trackingRecHit_iterator>((*track)->recHitsBegin(), (*track)->recHitsEnd(), associate.get(), **t);
	}

	if (AbsoluteNumberOfHits) quality = static_cast<double>(nshared);
	else if(ri!=0) quality = (static_cast<double>(nshared)/static_cast<double>(ri));
	else quality = 0;
	//cut on the fraction
       	//float purity = 1.0*nshared/ri;
	if(quality > cut_RecoToSim && !(ThreeHitTracksAreSpecial && ri==3 && nshared<3)){
	  //if a track has just 3 hits we require that all 3 hits are shared
	  outputCollection.insert(tC[tindex],
				  std::make_pair(TPCollectionH[tpindex],
						 quality));
//          LogTrace("TrackAssociator") << "reco::Track number " << tindex  << " with #hits=" << ri <<" pt=" << (*track)->pt() 
//                                      << " associated to TP (pdgId, nb segments, p) = " 
//                                      << (*t).pdgId() << " " << (*t).g4Tracks().size() 
//                                      << " " << (*t).momentum() << " #hits=" << nsimhit
//                                      << " TP index=" << tpindex<< " with quality =" << quality;
	} else {
//          LogTrace("TrackAssociator") <<"reco::Track number " << tindex << " with #hits=" << ri 
//                                      << " NOT associated with quality =" << quality;
	}
      }//TP loop
    }
  }
  //LogTrace("TrackAssociator") << "% of Assoc Tracks=" << ((double)outputCollection.size())/((double)tC.size());
  outputCollection.post_insert();
  return outputCollection;
}


SimToRecoCollection  
TrackAssociatorByHitsImpl::associateSimToReco(const edm::RefToBaseVector<reco::Track>& tC, 
					  const edm::RefVector<TrackingParticleCollection>& TPCollectionH) const {

  //edm::ESHandle<TrackerTopology> tTopoHand;
  //setup->get<IdealGeometryRecord>().get(tTopoHand);

//  edm::LogVerbatim("TrackAssociator") << "Starting TrackAssociatorByHitsImpl::associateSimToReco - #tracks="<<tC.size()<<" #TPs="<<TPCollectionH.size();
  float quality=0;//fraction or absolute number of shared hits
  int nshared = 0;
  std::vector< SimHitIdpr> SimTrackIds;
  std::vector< SimHitIdpr> matchedIds; 
  SimToRecoCollection  outputCollection;

  //dereferene the edm::Refs only once
  std::vector<TrackingParticle const*> tPC;
  tPC.reserve(tPC.size());
  for(auto const& ref: TPCollectionH) {
    tPC.push_back(&(*ref));
  }

  //for (TrackingParticleCollection::const_iterator t = tPC.begin(); t != tPC.end(); ++t) {
  //  LogTrace("TrackAssociator") << "NEW TP DUMP";
  //  for (TrackingParticle::g4t_iterator g4T = t -> g4Track_begin();g4T !=  t -> g4Track_end(); ++g4T) {
  //    LogTrace("TrackAssociator") << "(*g4T).trackId()=" <<(*g4T).trackId() ;
  //  }
  //}

  //cdj edm::Handle<SimHitTPAssociationProducer::SimHitTPAssociationList> simHitsTPAssoc;
  //warning: make sure the TP collection used in the map is the same used in the associator!
  //e->getByLabel(_simHitTpMapTag,simHitsTPAssoc);
  
  //get the ID of the recotrack  by hits 
  int tindex=0;
  for (edm::RefToBaseVector<reco::Track>::const_iterator track=tC.begin(); track!=tC.end(); track++, tindex++){
    //LogTrace("TrackAssociator") << "\nNEW TRACK - hits of track number " << tindex <<" with pt =" << (*track)->pt() << " # valid=" << (*track)->found(); 
    int ri=0;//valid rechits
    getMatchedIds<trackingRecHit_iterator>(matchedIds, SimTrackIds, ri, (*track)->recHitsBegin(), (*track)->recHitsEnd(), associate.get());

    //save id for the track
    std::vector<SimHitIdpr> idcachev;
    if(!matchedIds.empty()){
	
      int tpindex =0;
      for (auto t = tPC.begin(); t != tPC.end(); ++t, ++tpindex) {
	idcachev.clear();
	float totsimhit = 0; 

        //int nsimhit = trackerPSimHit.size();
	std::vector<PSimHit> tphits;
	//LogTrace("TrackAssociator") << "TP number " << tpindex << " pdgId=" << (*t)->pdgId() << " with number of PSimHits: "  << nsimhit;

	nshared = getShared(matchedIds, idcachev, **t);

	//for(std::vector<PSimHit>::const_iterator TPhit = (*t)->trackerPSimHit_begin(); TPhit != (*t)->trackerPSimHit_end(); TPhit++){
	//  unsigned int detid = TPhit->detUnitId();
	//  DetId detId = DetId(TPhit->detUnitId());
	//  LogTrace("TrackAssociator") <<  " hit trackId= " << TPhit->trackId() << " det ID = " << detid 
	//				      << " SUBDET = " << detId.subdetId() << " layer = " << LayerFromDetid(detId); 
	//}

	if (nshared!=0) {//do not waste time recounting when it is not needed!!!!

	  //count the TP simhit
	  //LogTrace("TrackAssociator") << "recounting of tp hits";

	  std::pair<TrackingParticleRef, TrackPSimHitRef> 
	    clusterTPpairWithDummyTP(TPCollectionH[tpindex],TrackPSimHitRef());//SimHit is dummy: for simHitTPAssociationListGreater 
	                                                                                                 // sorting only the cluster is needed
	  auto range = std::equal_range(simHitsTPAssoc->begin(), simHitsTPAssoc->end(), 
					clusterTPpairWithDummyTP, SimHitTPAssociationProducer::simHitTPAssociationListGreater);
	  for(auto ip = range.first; ip != range.second; ++ip) {
	    TrackPSimHitRef TPhit = ip->second;
	    DetId dId = DetId(TPhit->detUnitId());
	  
	    unsigned int subdetId = static_cast<unsigned int>(dId.subdetId());
	    if (!UsePixels && (subdetId==PixelSubdetector::PixelBarrel || subdetId==PixelSubdetector::PixelEndcap) )
	      continue;

	    //unsigned int dRawId = dId.rawId();
	    SiStripDetId* stripDetId = 0;
	    if (subdetId==SiStripDetId::TIB||subdetId==SiStripDetId::TOB||
		subdetId==SiStripDetId::TID||subdetId==SiStripDetId::TEC)
	      stripDetId= new SiStripDetId(dId);
            //LogTrace("TrackAssociator") << "consider hit SUBDET = " << subdetId
            //                            << " layer = " << LayerFromDetid(dId) 
            //                            << " id = " << dId.rawId();
	    bool newhit = true;
	    for(std::vector<PSimHit>::const_iterator TPhitOK = tphits.begin(); TPhitOK != tphits.end(); TPhitOK++){
	      DetId dIdOK = DetId(TPhitOK->detUnitId());
	      //unsigned int dRawIdOK = dIdOK.rawId();
              //LogTrace("TrackAssociator") << "\t\tcompare with SUBDET = " << dIdOK.subdetId()
              //                            << " layer = " << LayerFromDetid(dIdOK)
              //                            << " id = " << dIdOK.rawId();
	      //no grouped, no splitting
	      if (!UseGrouped && !UseSplitting)
		if (tTopo->layer(dId)==tTopo->layer(dIdOK) &&
		    dId.subdetId()==dIdOK.subdetId()) newhit = false;
	      //no grouped, splitting
	      if (!UseGrouped && UseSplitting)
		if (tTopo->layer(dId)==tTopo->layer(dIdOK) &&
		    dId.subdetId()==dIdOK.subdetId() &&
		    (stripDetId==0 || stripDetId->partnerDetId()!=dIdOK.rawId()))
		  newhit = false;
	      //grouped, no splitting
	      if (UseGrouped && !UseSplitting)
		if (tTopo->layer(dId)==tTopo->layer(dIdOK) &&
		    dId.subdetId()==dIdOK.subdetId() &&
		    stripDetId!=0 && stripDetId->partnerDetId()==dIdOK.rawId())
		  newhit = false;
	      //grouped, splitting
	      if (UseGrouped && UseSplitting)
		newhit = true;
	    }
	    if (newhit) {
	      //LogTrace("TrackAssociator") << "\t\tok";
	      tphits.push_back(*TPhit);
	    }
	    //else LogTrace("TrackAssociator") << "\t\tno";
	    delete stripDetId;
	  }
	  totsimhit = tphits.size();
	}

	if (AbsoluteNumberOfHits) quality = static_cast<double>(nshared);
	else if(SimToRecoDenominator == denomsim && totsimhit!=0) quality = ((double) nshared)/((double)totsimhit);
	else if(SimToRecoDenominator == denomreco && ri!=0) quality = ((double) nshared)/((double)ri);
	else quality = 0;
	//LogTrace("TrackAssociator") << "Final count: nhit(TP) = " << nsimhit << " re-counted = " << totsimhit 
	//<< " nshared = " << nshared << " nrechit = " << ri;
	
	float purity = 1.0*nshared/ri;
	if (quality>quality_SimToReco && !(ThreeHitTracksAreSpecial && totsimhit==3 && nshared<3) && (AbsoluteNumberOfHits||(purity>purity_SimToReco))) {
	  //if a track has just 3 hits we require that all 3 hits are shared
	  outputCollection.insert(TPCollectionH[tpindex], 
				  std::make_pair(tC[tindex],quality));
//          LogTrace("TrackAssociator") << "TrackingParticle number " << tpindex << " with #hits=" << nsimhit 
//                                      << " re-counted = "  << totsimhit << " nshared = " << nshared 
//                                      << " associated to track number " << tindex << " with pt=" << (*track)->pt() 
//                                      << " with hit quality =" << quality ;
	} else {
//          LogTrace("TrackAssociator") << "TrackingParticle number " << tpindex << " with #hits=" << nsimhit 
//                                      << " re-counted = "  << totsimhit << " nshared = " << nshared 
//                                      << " NOT associated with quality =" << quality;
	}
      }
    }
  }
  //LogTrace("TrackAssociator") << "% of Assoc TPs=" << ((double)outputCollection.size())/((double)TPCollectionH.size());
  outputCollection.post_insert();
  return outputCollection;
}


RecoToSimCollectionSeed  
TrackAssociatorByHitsImpl::associateRecoToSim(const edm::Handle<edm::View<TrajectorySeed> >& seedCollectionH,
                                              const edm::Handle<TrackingParticleCollection>&  TPCollectionH) const {
  edm::LogVerbatim("TrackAssociator") << "Starting TrackAssociatorByHitsImpl::associateRecoToSim - #seeds="
				      <<seedCollectionH->size()<<" #TPs="<<TPCollectionH->size();
  int nshared = 0;
  float quality=0;//fraction or absolute number of shared hits
  std::vector< SimHitIdpr> SimTrackIds;
  std::vector< SimHitIdpr> matchedIds; 
  RecoToSimCollectionSeed  outputCollection;
  
  const TrackingParticleCollection& tPC   = *(TPCollectionH.product());

  const edm::View<TrajectorySeed> sC = *(seedCollectionH.product()); 
  
  //get the ID of the recotrack  by hits 
  int tindex=0;
  for (edm::View<TrajectorySeed>::const_iterator seed=sC.begin(); seed!=sC.end(); seed++, tindex++) {
    matchedIds.clear();
    int ri=0;//valid rechits
    int nsimhit = seed->recHits().second-seed->recHits().first;
    LogTrace("TrackAssociator") << "\nNEW SEED - seed number " << tindex << " # valid=" << nsimhit;
    getMatchedIds<edm::OwnVector<TrackingRecHit>::const_iterator>(matchedIds, SimTrackIds, ri, seed->recHits().first, seed->recHits().second, associate.get() );

    //save id for the track
    std::vector<SimHitIdpr> idcachev;
    if(!matchedIds.empty()){

      int tpindex =0;
      for (TrackingParticleCollection::const_iterator t = tPC.begin(); t != tPC.end(); ++t, ++tpindex) {
	LogTrace("TrackAssociator") << "TP number " << tpindex << " pdgId=" << t->pdgId() << " with number of PSimHits: "  << nsimhit;
	idcachev.clear();
	nshared = getShared(matchedIds, idcachev, *t);

	//if electron subtract double counting
	if (abs(t->pdgId())==11&&(t->g4Track_end()-t->g4Track_begin())>1){
	  nshared-=getDoubleCount<edm::OwnVector<TrackingRecHit>::const_iterator>(seed->recHits().first, seed->recHits().second, associate.get(), *t);
	}
	
	if (AbsoluteNumberOfHits) quality = static_cast<double>(nshared);
	else if(ri!=0) quality = (static_cast<double>(nshared)/static_cast<double>(ri));
	else quality = 0;
	//cut on the fraction
	if(quality > cut_RecoToSim && !(ThreeHitTracksAreSpecial && ri==3 && nshared<3) ){
	  //if a track has just 3 hits we require that all 3 hits are shared
	  outputCollection.insert(edm::RefToBase<TrajectorySeed>(seedCollectionH,tindex), 
				  std::make_pair(edm::Ref<TrackingParticleCollection>(TPCollectionH, tpindex),quality));
	  LogTrace("TrackAssociator") << "Seed number " << tindex << " with #hits=" << ri
				      << "associated to TP (pdgId, nb segments, p) = " 
				      << (*t).pdgId() << " " << (*t).g4Tracks().size() 
				      << " " << (*t).momentum() <<" number " << tpindex << " with quality =" << quality;
	} else {
	  LogTrace("TrackAssociator") <<"Seed number " << tindex << " with #hits=" << ri << " NOT associated with quality =" << quality;
	}
      }//TP loop
    }
  }
  LogTrace("TrackAssociator") << "% of Assoc Seeds=" << ((double)outputCollection.size())/((double)seedCollectionH->size());

  outputCollection.post_insert();
  return outputCollection;
}


SimToRecoCollectionSeed
TrackAssociatorByHitsImpl::associateSimToReco(const edm::Handle<edm::View<TrajectorySeed> >& seedCollectionH,
                                              const edm::Handle<TrackingParticleCollection>& TPCollectionH) const {

  edm::LogVerbatim("TrackAssociator") << "Starting TrackAssociatorByHitsImpl::associateSimToReco - #seeds="
				      <<seedCollectionH->size()<<" #TPs="<<TPCollectionH->size();
  float quality=0;//fraction or absolute number of shared hits
  int nshared = 0;
  std::vector< SimHitIdpr> SimTrackIds;
  std::vector< SimHitIdpr> matchedIds; 
  SimToRecoCollectionSeed  outputCollection;

  const TrackingParticleCollection& tPC =*TPCollectionH.product();

  const edm::View<TrajectorySeed> sC = *(seedCollectionH.product()); 

  //get the ID of the recotrack  by hits 
  int tindex=0;
  for (edm::View<TrajectorySeed>::const_iterator seed=sC.begin(); seed!=sC.end(); seed++, tindex++) {
    int ri=0;//valid rechits
    LogTrace("TrackAssociator") << "\nNEW SEED - seed number " << tindex << " # valid=" << seed->recHits().second-seed->recHits().first;
    getMatchedIds<edm::OwnVector<TrackingRecHit>::const_iterator>(matchedIds, SimTrackIds, ri, seed->recHits().first, seed->recHits().second, associate.get() );

    //save id for the track
    std::vector<SimHitIdpr> idcachev;
    if(!matchedIds.empty()){
      int tpindex =0;
      for (TrackingParticleCollection::const_iterator t = tPC.begin(); t != tPC.end(); ++t, ++tpindex) {
	idcachev.clear();
        int nsimhit = t->numberOfTrackerHits();
	LogTrace("TrackAssociator") << "TP number " << tpindex << " pdgId=" << t->pdgId() << " with number of PSimHits: "  << nsimhit;
	nshared = getShared(matchedIds, idcachev, *t);
	
	if (AbsoluteNumberOfHits) quality = static_cast<double>(nshared);
	else if(ri!=0) quality = ((double) nshared)/((double)ri);
	else quality = 0;
	//LogTrace("TrackAssociator") << "Final count: nhit(TP) = " << nsimhit 
	//<< " nshared = " << nshared 
	//<< " nrechit = " << ri;
	if(quality > quality_SimToReco && !(ThreeHitTracksAreSpecial && ri==3 && nshared<3) ){
	  outputCollection.insert(edm::Ref<TrackingParticleCollection>(TPCollectionH, tpindex), 
				  std::make_pair(edm::RefToBase<TrajectorySeed>(seedCollectionH,tindex), quality));
	  LogTrace("TrackAssociator") << "TrackingParticle number " << tpindex << " with #hits=" << nsimhit
				      << " associated to seed number " << tindex << " with #hits=" << ri 
				      << " with hit quality =" << quality ;
	}
	else {
	  LogTrace("TrackAssociator") << "TrackingParticle number " << tpindex << " with #hits=" << nsimhit << " NOT associated with quality =" << quality;
	}
      }
    }
  }
  LogTrace("TrackAssociator") << "% of Assoc TPs=" << ((double)outputCollection.size())/((double)TPCollectionH->size());
  outputCollection.post_insert();
  return outputCollection;
}

template<typename iter>
void TrackAssociatorByHitsImpl::getMatchedIds(std::vector<SimHitIdpr>& matchedIds, 
					  std::vector<SimHitIdpr>& SimTrackIds, 
					  int& ri, 
					  iter begin,
					  iter end,
					  TrackerHitAssociator* associate ) const {
    matchedIds.clear();
    ri=0;//valid rechits
    for (iter it = begin;  it != end; it++){
      const TrackingRecHit *hit=getHitPtr(it);
      if (hit->isValid()){
	ri++;
	uint32_t t_detID=  hit->geographicalId().rawId();
	SimTrackIds.clear();	  
 	associate->associateHitId(*hit, SimTrackIds);
	//save all the id of matched simtracks
	if(!SimTrackIds.empty()){
	 for(size_t j=0; j<SimTrackIds.size(); j++){
	   LogTrace("TrackAssociator") << " hit # " << ri << " valid=" << hit->isValid() 
				       << " det id = " << t_detID << " SimId " << SimTrackIds[j].first 
				       << " evt=" << SimTrackIds[j].second.event() 
				       << " bc=" << SimTrackIds[j].second.bunchCrossing();  
 	   matchedIds.push_back(SimTrackIds[j]);			
	 }
	}
	////debugging....****
	//****to be used when the crossing frame is in the event and with flag TrackAssociatorByHitsImplESProducer.associateRecoTracks = false
	//std::vector<PSimHit> tmpSimHits = associate->associateHit(*getHitPtr(it));
	////cout << "tmpSimHits size=" << tmpSimHits.size() << endl;
	//for(size_t j=0; j<tmpSimHits.size(); j++) {
	//  LogTrace("TrackAssociator") << " hit # " << ri << " valid=" << getHitPtr(it)->isValid()
	//                              << " det id = " << t_detID << " SimId " << SimTrackIds[j].first
	//                              << " evt=" << SimTrackIds[j].second.event()
	//                              << " bc=" << SimTrackIds[j].second.bunchCrossing()
	//                              << " process=" << tmpSimHits[j].processType()
	//                              << " eloss=" << tmpSimHits[j].energyLoss()
	//                              << " part type=" << tmpSimHits[j].particleType()
	//                              << " pabs=" << tmpSimHits[j].pabs()
	//                              << " trId=" << tmpSimHits[j].trackId();
	//}
	////******************
      }else{
	LogTrace("TrackAssociator") <<"\t\t Invalid Hit On "<<hit->geographicalId().rawId();
      }
    }//trackingRecHit loop
}


int TrackAssociatorByHitsImpl::getShared(std::vector<SimHitIdpr>& matchedIds, 
				     std::vector<SimHitIdpr>& idcachev,
				     TrackingParticle const& t) const {
  int nshared = 0;
  if (t.numberOfHits()==0) return nshared;//should use trackerPSimHit but is not const

  for(size_t j=0; j<matchedIds.size(); j++){
    //LogTrace("TrackAssociator") << "now matchedId=" << matchedIds[j].first;
    if(find(idcachev.begin(), idcachev.end(),matchedIds[j]) == idcachev.end() ){
      //only the first time we see this ID 
      idcachev.push_back(matchedIds[j]);
      
      for (TrackingParticle::g4t_iterator g4T = t . g4Track_begin(); g4T !=  t . g4Track_end(); ++g4T) {
//	LogTrace("TrackAssociator") << " TP   (ID, Ev, BC) = " << (*g4T).trackId() 
//				    << ", " << t.eventId().event() << ", "<< t.eventId().bunchCrossing()
//				    << " Match(ID, Ev, BC) = " <<  matchedIds[j].first
//				    << ", " << matchedIds[j].second.event() << ", "
//				    << matchedIds[j].second.bunchCrossing() ;
	                            //<< "\t G4  Track Momentum " << (*g4T).momentum() 
	                            //<< " \t reco Track Momentum " << track->momentum();  	      
	if((*g4T).trackId() == matchedIds[j].first && t.eventId() == matchedIds[j].second){
		int countedhits = std::count(matchedIds.begin(), matchedIds.end(), matchedIds[j]);
		nshared += countedhits;

//		LogTrace("TrackAssociator") << "hits shared by this segment : " << countedhits;
//		LogTrace("TrackAssociator") << "hits shared so far : " << nshared;
	}
      }//g4Tracks loop
    }
  }
  return nshared;
}


template<typename iter>
int TrackAssociatorByHitsImpl::getDoubleCount(iter begin,
					  iter end,
					  TrackerHitAssociator* associate,
					  TrackingParticle const& t) const {
  int doublecount = 0 ;
  std::vector<SimHitIdpr> SimTrackIdsDC;
  //  cout<<begin-end<<endl;
  for (iter it = begin;  it != end; it++){
    int idcount = 0;
    SimTrackIdsDC.clear();
    associate->associateHitId(*getHitPtr(it), SimTrackIdsDC);
    //    cout<<SimTrackIdsDC.size()<<endl;
    if(SimTrackIdsDC.size()>1){
      //     cout<<(t.g4Track_end()-t.g4Track_begin())<<endl;
      for (TrackingParticle::g4t_iterator g4T = t . g4Track_begin(); g4T !=  t . g4Track_end(); ++g4T) {
	if(find(SimTrackIdsDC.begin(), SimTrackIdsDC.end(),SimHitIdpr((*g4T).trackId(), SimTrackIdsDC.begin()->second)) != SimTrackIdsDC.end() ){
	  idcount++;
	}
      }
    }
    if (idcount>1) doublecount+=(idcount-1);
  }
  return doublecount;
}
