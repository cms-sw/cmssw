// File: TrackerHitAssociator.cc

#include <memory>
#include <string>
#include <vector>

#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h"
//--- for Geometry:
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"

//for accumulate
#include <numeric>
#include <iostream>

using namespace std;
using namespace edm;

//
// Constructor 
//
TrackerHitAssociator::TrackerHitAssociator(const edm::Event& e)  : 
  myEvent_(e), 
  doPixel_( true ),
  doStrip_( true ) {
  
  trackerContainers.clear();
  //
  // Take by default all tracker SimHits
  //
  trackerContainers.push_back("TrackerHitsTIBLowTof");
  trackerContainers.push_back("TrackerHitsTIBHighTof");
  trackerContainers.push_back("TrackerHitsTIDLowTof");
  trackerContainers.push_back("TrackerHitsTIDHighTof");
  trackerContainers.push_back("TrackerHitsTOBLowTof");
  trackerContainers.push_back("TrackerHitsTOBHighTof");
  trackerContainers.push_back("TrackerHitsTECLowTof");
  trackerContainers.push_back("TrackerHitsTECHighTof");
  trackerContainers.push_back("TrackerHitsPixelBarrelLowTof");
  trackerContainers.push_back("TrackerHitsPixelBarrelHighTof");
  trackerContainers.push_back("TrackerHitsPixelEndcapLowTof");
  trackerContainers.push_back("TrackerHitsPixelEndcapHighTof");

  // Step A: Get Inputs
  edm::Handle<CrossingFrame<PSimHit> > cf_simhit;
  std::vector<const CrossingFrame<PSimHit> *> cf_simhitvec;
  for(uint32_t i = 0; i< trackerContainers.size();i++){
    e.getByLabel("mix",trackerContainers[i],cf_simhit);
    cf_simhitvec.push_back(cf_simhit.product());
  }
  
  std::auto_ptr<MixCollection<PSimHit> > allTrackerHits(new MixCollection<PSimHit>(cf_simhitvec));
  

 //Loop on PSimHit
  SimHitMap.clear();
  
  MixCollection<PSimHit>::iterator isim;
  for (isim=allTrackerHits->begin(); isim!= allTrackerHits->end();isim++) {
    SimHitMap[(*isim).detUnitId()].push_back((*isim));
  }
  
  if(doStrip_) e.getByLabel("siStripDigis", stripdigisimlink);
  if(doPixel_) e.getByLabel("siPixelDigis", pixeldigisimlink);
  
}

//
// Constructor with configurables
//
TrackerHitAssociator::TrackerHitAssociator(const edm::Event& e, const edm::ParameterSet& conf)  : 
  myEvent_(e), 
  doPixel_( conf.getParameter<bool>("associatePixel") ),
  doStrip_( conf.getParameter<bool>("associateStrip") ),
  doTrackAssoc_( conf.getParameter<bool>("associateRecoTracks") ) {
  
  trackerContainers.clear();
  trackerContainers = conf.getParameter<std::vector<std::string> >("ROUList");

  //if track association there is no need to acces the CrossingFrame
  if(!doTrackAssoc_) {

    // Step A: Get Inputs
    edm::Handle<CrossingFrame<PSimHit> > cf_simhit;
    std::vector<const CrossingFrame<PSimHit> *> cf_simhitvec;
    for(uint32_t i = 0; i< trackerContainers.size();i++){
      e.getByLabel("mix",trackerContainers[i],cf_simhit);
      cf_simhitvec.push_back(cf_simhit.product());
    }
    
    std::auto_ptr<MixCollection<PSimHit> > allTrackerHits(new MixCollection<PSimHit>(cf_simhitvec));
    
    //Loop on PSimHit
    SimHitMap.clear();
    
    MixCollection<PSimHit>::iterator isim;
    for (isim=allTrackerHits->begin(); isim!= allTrackerHits->end();isim++) {
      SimHitMap[(*isim).detUnitId()].push_back((*isim));
    }
    
  }

  if(doStrip_) e.getByLabel("siStripDigis", stripdigisimlink);
  if(doPixel_) e.getByLabel("siPixelDigis", pixeldigisimlink);
  
}

std::vector<PSimHit> TrackerHitAssociator::associateHit(const TrackingRecHit & thit) 
{
  
  //check in case of TTRH
  if(const TransientTrackingRecHit * ttrh = dynamic_cast<const TransientTrackingRecHit *>(&thit)) {
      return associateHit(*ttrh->hit());
  }
 
  //vector with the matched SimHit
  std::vector<PSimHit> result; 
  simtrackid.clear();

  //get the Detector type of the rechit
  DetId detid=  thit.geographicalId();
  uint32_t detID = detid.rawId();
  //cout << "Associator ---> get Detid " << detID << endl;
  //check we are in the strip tracker
  if(detid.subdetId() == StripSubdetector::TIB ||
     detid.subdetId() == StripSubdetector::TOB || 
     detid.subdetId() == StripSubdetector::TID ||
     detid.subdetId() == StripSubdetector::TEC) 
    {
      //check if it is a simple SiStripRecHit2D
      if(const SiStripRecHit2D * rechit = 
	 dynamic_cast<const SiStripRecHit2D *>(&thit))
	{	  
	  simtrackid = associateSimpleRecHit(rechit);
	}
      //check if it is a matched SiStripMatchedRecHit2D
      if(const SiStripMatchedRecHit2D * rechit = 
	 dynamic_cast<const SiStripMatchedRecHit2D *>(&thit))
	{	  
	  simtrackid = associateMatchedRecHit(rechit);
	}
      //check if it is a  ProjectedSiStripRecHit2D
      if(const ProjectedSiStripRecHit2D * rechit = 
	 dynamic_cast<const ProjectedSiStripRecHit2D *>(&thit))
	{	  
	  simtrackid = associateProjectedRecHit(rechit);
	  detid = rechit->originalHit().geographicalId();
	  detID = detid.rawId();
	}
    }
  //check we are in the pixel tracker
  if( detid.subdetId() == PixelSubdetector::PixelBarrel || 
      detid.subdetId() == PixelSubdetector::PixelEndcap) 
    {
      if(const SiPixelRecHit * rechit = dynamic_cast<const SiPixelRecHit *>(&thit))
	{	  
	  simtrackid = associatePixelRecHit(rechit);
	}
    }
  //check if these are GSRecHits (from FastSim)

  if(const SiTrackerGSRecHit2D * rechit = dynamic_cast<const SiTrackerGSRecHit2D *>(&thit))
    {
      simtrackid = associateGSRecHit(rechit);
    }
  if (const SiTrackerMultiRecHit * rechit = dynamic_cast<const SiTrackerMultiRecHit *>(&thit)){
    return associateMultiRecHit(rechit);
  }
  
  //check if these are GSMatchedRecHits (from FastSim)
  if(const SiTrackerGSMatchedRecHit2D * rechit = dynamic_cast<const SiTrackerGSMatchedRecHit2D *>(&thit))
    {
      simtrackid = associateGSMatchedRecHit(rechit);
    }
  
  
  //now get the SimHit from the trackid
  vector<PSimHit> simHit; 
  std::map<unsigned int, std::vector<PSimHit> >::const_iterator it = SimHitMap.find(detID);
  simHit.clear();
  if (it!= SimHitMap.end()){
    simHit = it->second;
    vector<PSimHit>::const_iterator simHitIter = simHit.begin();
    vector<PSimHit>::const_iterator simHitIterEnd = simHit.end();
    for (;simHitIter != simHitIterEnd; ++simHitIter) {
      const PSimHit ihit = *simHitIter;
      unsigned int simHitid = ihit.trackId();
      EncodedEventId simHiteid = ihit.eventId();
      
      for(size_t i=0; i<simtrackid.size();i++){
	if(simHitid == simtrackid[i].first && simHiteid == simtrackid[i].second){ 
	  //	  cout << "Associator ---> ID" << ihit.trackId() << " Simhit x= " << ihit.localPosition().x() 
	  //	       << " y= " <<  ihit.localPosition().y() << " z= " <<  ihit.localPosition().x() << endl; 
	  result.push_back(ihit);
	}
      }
    }
  }else{
    /// Check if it's the gluedDet   
    std::map<unsigned int, std::vector<PSimHit> >::const_iterator itrphi = 
      SimHitMap.find(detID+2);//iterator to the simhit in the rphi module
    std::map<unsigned int, std::vector<PSimHit> >::const_iterator itster = 
      SimHitMap.find(detID+1);//iterator to the simhit in the stereo module
    if (itrphi!= SimHitMap.end()&&itster!=SimHitMap.end()){
      simHit = itrphi->second;
      simHit.insert(simHit.end(),(itster->second).begin(),(itster->second).end());
      vector<PSimHit>::const_iterator simHitIter = simHit.begin();
      vector<PSimHit>::const_iterator simHitIterEnd = simHit.end();
      for (;simHitIter != simHitIterEnd; ++simHitIter) {
	const PSimHit ihit = *simHitIter;
	unsigned int simHitid = ihit.trackId();
	EncodedEventId simHiteid = ihit.eventId();

	//===>>>>>change here!!!!
	//	for(size_t i=0; i<simtrackid.size();i++){
	//  cout << " GluedDet Associator -->  check sihit id's = " << simHitid <<"; compared id's = "<< simtrackid[i] <<endl;
	// if(simHitid == simtrackid[i]){ //exclude the geant particles. they all have the same id
	  for(size_t i=0; i<simtrackid.size();i++){
	    if(simHitid == simtrackid[i].first && simHiteid == simtrackid[i].second){ 
	      //	  cout << "GluedDet Associator ---> ID" << ihit.trackId() << " Simhit x= " << ihit.localPosition().x() 
	      //	       << " y= " <<  ihit.localPosition().y() << " z= " <<  ihit.localPosition().x() << endl; 
	      result.push_back(ihit);
	    }
	  }
      }
    }
  }
  return result;  
}

//std::vector<unsigned int> TrackerHitAssociator::associateHitId(const TrackingRecHit & thit) 
std::vector< SimHitIdpr > TrackerHitAssociator::associateHitId(const TrackingRecHit & thit) 
{
  
  //check in case of TTRH
  if(const TransientTrackingRecHit * ttrh = dynamic_cast<const TransientTrackingRecHit *>(&thit)) {
      return associateHitId(*ttrh->hit());
  }
   
  //vector with the matched SimTrackID 
  simtrackid.clear();
  
  //get the Detector type of the rechit
  DetId detid=  thit.geographicalId();
  //apparently not used
  //  uint32_t detID = detid.rawId();
  if (const SiTrackerMultiRecHit * rechit = dynamic_cast<const SiTrackerMultiRecHit *>(&thit)){
        return associateMultiRecHitId(rechit);
  }

  //cout << "Associator ---> get Detid " << detID << endl;
  //check we are in the strip tracker
  if(detid.subdetId() == StripSubdetector::TIB ||
     detid.subdetId() == StripSubdetector::TOB || 
     detid.subdetId() == StripSubdetector::TID ||
     detid.subdetId() == StripSubdetector::TEC) 
    {
      //check if it is a simple SiStripRecHit2D
      if(const SiStripRecHit2D * rechit = 
	 dynamic_cast<const SiStripRecHit2D *>(&thit))
	{	  
	  simtrackid = associateSimpleRecHit(rechit);
	}
      //check if it is a matched SiStripMatchedRecHit2D
      else  if(const SiStripMatchedRecHit2D * rechit = 
	 dynamic_cast<const SiStripMatchedRecHit2D *>(&thit))
	{	  
	  simtrackid = associateMatchedRecHit(rechit);
	}
      //check if it is a  ProjectedSiStripRecHit2D
      else if(const ProjectedSiStripRecHit2D * rechit = 
	 dynamic_cast<const ProjectedSiStripRecHit2D *>(&thit))
	{	  
	  simtrackid = associateProjectedRecHit(rechit);
	}
    }
  //check we are in the pixel tracker
  else if( detid.subdetId() == PixelSubdetector::PixelBarrel || 
	   detid.subdetId() == PixelSubdetector::PixelEndcap) 
    {
      if(const SiPixelRecHit * rechit = dynamic_cast<const SiPixelRecHit *>(&thit))
	{	  
	  simtrackid = associatePixelRecHit(rechit);
	}
    }
  //check if these are GSRecHits (from FastSim)
  if(const SiTrackerGSRecHit2D * rechit = dynamic_cast<const SiTrackerGSRecHit2D *>(&thit))
    {
      simtrackid = associateGSRecHit(rechit);
    }  
  if(const SiTrackerGSMatchedRecHit2D * rechit = dynamic_cast<const SiTrackerGSMatchedRecHit2D *>(&thit))
    {
      simtrackid = associateGSMatchedRecHit(rechit);
    }

 
  return simtrackid;  
}


//std::vector<unsigned int>  TrackerHitAssociator::associateSimpleRecHit(const SiStripRecHit2D * simplerechit)
std::vector<SimHitIdpr>  TrackerHitAssociator::associateSimpleRecHit(const SiStripRecHit2D * simplerechit)
{
  DetId detid=  simplerechit->geographicalId();
  uint32_t detID = detid.rawId();

  //to store temporary charge information
  //  std::vector<unsigned int> cache_simtrackid; 
  std::vector<SimHitIdpr> cache_simtrackid; 
  cache_simtrackid.clear();
  //  std::map<unsigned int, vector<float> > temp_simtrackid;
  std::map<SimHitIdpr, vector<float> > temp_simtrackid;
  temp_simtrackid.clear();

  edm::DetSetVector<StripDigiSimLink>::const_iterator isearch = stripdigisimlink->find(detID); 
  if(isearch != stripdigisimlink->end()) {  //if it is not empty
    //link_detset is a structure, link_detset.data is a std::vector<StripDigiSimLink>
    edm::DetSet<StripDigiSimLink> link_detset = (*stripdigisimlink)[detID];
    
    //Modification for regional clustering from Jean-Roch Vlimant
    const SiStripCluster* clust; 
    if(simplerechit->cluster().isNonnull())
    {
      clust=&(*simplerechit->cluster());
    }else if(simplerechit->cluster_regional().isNonnull())
       {
	 clust=&(*simplerechit->cluster_regional());
       } 
    else 
      {
	edm::LogError("TrackerHitAssociator")<<"no cluster reference attached";
      }
  //    const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > clust=simplerechit->cluster();
  
    //float chg;
    int clusiz = clust->amplitudes().size();
    int first  = clust->firstStrip();     
    int last   = first + clusiz;
    //float cluchg = std::accumulate(clust->amplitudes().begin(), clust->amplitudes().end(),0);
    // cout << "Associator ---> Clus size = " << clusiz << " first = " << first << "  last = " << last << "  tot charge = " << cluchg << endl;

    //use a vector
    //    std::vector<unsigned int> idcachev;
    std::vector<SimHitIdpr> idcachev;
    for(edm::DetSet<StripDigiSimLink>::const_iterator linkiter = link_detset.data.begin(); linkiter != link_detset.data.end(); linkiter++){
      StripDigiSimLink link = *linkiter;
      if( link.channel() >= first  && link.channel() < last ){
	SimHitIdpr currentId(link.SimTrackId(), link.eventId());
	//write only once the id
	if(find(idcachev.begin(),idcachev.end(),currentId ) == idcachev.end()){
	  /*
	  std::cout << " Adding track id  = " << currentId.first  
		    << " Event id = " << currentId.second.event() 
		    << " Bunch Xing = " << currentId.second.bunchCrossing() << std::endl;
	  */
	  //cache_simtrackid.push_back(currentId);
	  idcachev.push_back(currentId);
	  simtrackid.push_back(currentId);
	}

	/*
	//get the charge released in the cluster by the simtrack strip by strip
	chg = 0;
        int mychan = link.channel()-first;
        chg = (clust->amplitudes()[mychan])*link.fraction();
        temp_simtrackid[currentId].push_back(chg);
	//std::cout << " Track ID = " <<  link.SimTrackId() << " charge by ch = " << chg << std::endl;
	*/
	
      }
    }
    
    /*
    vector<float> tmpchg;
    float simchg;
    float simfraction;
    std::map<float, SimHitIdpr, greater<float> > temp_map;
    simchg=0;
    //loop over the unique ID's
    //    vector<unsigned int>::iterator new_end = unique(cache_simtrackid.begin(),cache_simtrackid.end());
    vector<SimHitIdpr>::iterator new_end = unique(cache_simtrackid.begin(),cache_simtrackid.end());
    for(vector<SimHitIdpr>::iterator i=cache_simtrackid.begin(); i != new_end; i++){
      std::map<SimHitIdpr, vector<float> >::const_iterator it = temp_simtrackid.find(*i);
      if(it != temp_simtrackid.end()){
	tmpchg = it->second;
	for(size_t ii=0; ii<tmpchg.size(); ii++){
	  simchg +=tmpchg[ii];
	}
	simfraction = simchg/cluchg;
	temp_map.insert(std::pair<float, SimHitIdpr > (simfraction,*i));
      }
    }	
    //sort the map ordered on the charge fraction 
    
    //copy the list of ID's ordered in the charge fraction 
    for(std::map<float , SimHitIdpr>::const_iterator it = temp_map.begin(); it!=temp_map.end(); it++){
      //std::cout << "Track id = " << it->second << " fraction = " << it->first << std::endl;
      simtrackid.push_back(it->second);
    }
    */

  }    
  
  
  return simtrackid;
  
}


//std::vector<unsigned int>  TrackerHitAssociator::associateMatchedRecHit(const SiStripMatchedRecHit2D * matchedrechit)
std::vector<SimHitIdpr>  TrackerHitAssociator::associateMatchedRecHit(const SiStripMatchedRecHit2D * matchedrechit)
{

  //  vector<unsigned int> matched_mono;
  //  vector<unsigned int> matched_st;
  vector<SimHitIdpr> matched_mono;
  vector<SimHitIdpr> matched_st;
  matched_mono.clear();
  matched_st.clear();
  
  const SiStripRecHit2D *mono = matchedrechit->monoHit();
  const SiStripRecHit2D *st = matchedrechit->stereoHit();
  //associate the two simple hits separately
  matched_mono = associateSimpleRecHit(mono);
  matched_st   = associateSimpleRecHit(st);
  
  //save in a vector all the simtrack-id's that are common to mono and stereo hits
  if(!matched_mono.empty() && !matched_st.empty()){
    simtrackid.clear(); //final result vector
    //    std::vector<unsigned int> idcachev;
    std::vector<SimHitIdpr> idcachev;
    //for(vector<unsigned int>::iterator mhit=matched_mono.begin(); mhit != matched_mono.end(); mhit++){
    for(vector<SimHitIdpr>::iterator mhit=matched_mono.begin(); mhit != matched_mono.end(); mhit++){
      //save only once the ID
      if(find(idcachev.begin(), idcachev.end(),(*mhit)) == idcachev.end()) {
	idcachev.push_back(*mhit);
	//save if the stereoID matched the monoID
	if(find(matched_st.begin(), matched_st.end(),(*mhit))!=matched_st.end()){
	  simtrackid.push_back(*mhit);
	  //std::cout << "matched case: saved ID " << (*mhit) << std::endl; 
	}
      }
    }
  }
  return simtrackid;
}


std::vector<SimHitIdpr>  TrackerHitAssociator::associateProjectedRecHit(const ProjectedSiStripRecHit2D * projectedrechit)
{
  //projectedRecHit is a "matched" rechit with only one component

  vector<SimHitIdpr> matched_mono;
  matched_mono.clear();
 
  const SiStripRecHit2D mono = projectedrechit->originalHit();
  matched_mono = associateSimpleRecHit(&mono);
  return matched_mono;
}

//std::vector<unsigned int>  TrackerHitAssociator::associatePixelRecHit(const SiPixelRecHit * pixelrechit)
std::vector<SimHitIdpr>  TrackerHitAssociator::associatePixelRecHit(const SiPixelRecHit * pixelrechit)
{
  //
  // Pixel associator
  //
  DetId detid=  pixelrechit->geographicalId();
  uint32_t detID = detid.rawId();
  edm::DetSetVector<PixelDigiSimLink>::const_iterator isearch = pixeldigisimlink->find(detID); 
  if(isearch != pixeldigisimlink->end()) {  //if it is not empty
    edm::DetSet<PixelDigiSimLink> link_detset = (*pixeldigisimlink)[detID];
    SiPixelRecHit::ClusterRef const& cluster = pixelrechit->cluster();
    int minPixelRow = (*cluster).minPixelRow();
    int maxPixelRow = (*cluster).maxPixelRow();
    int minPixelCol = (*cluster).minPixelCol();
    int maxPixelCol = (*cluster).maxPixelCol();    
    //std::cout << "    Cluster minRow " << minPixelRow << " maxRow " << maxPixelRow << std::endl;
    //std::cout << "    Cluster minCol " << minPixelCol << " maxCol " << maxPixelCol << std::endl;
    edm::DetSet<PixelDigiSimLink>::const_iterator linkiter = link_detset.data.begin();
    int dsl = 0;
    //    std::vector<unsigned int> idcachev;
    std::vector<SimHitIdpr> idcachev;
    for( ; linkiter != link_detset.data.end(); linkiter++) {
      dsl++;
      std::pair<int,int> pixel_coord = PixelDigi::channelToPixel(linkiter->channel());
      //std::cout << "    " << dsl << ") Digi link: row " << pixel_coord.first << " col " << pixel_coord.second << std::endl;      
      if(  pixel_coord.first  <= maxPixelRow && 
	   pixel_coord.first  >= minPixelRow &&
	   pixel_coord.second <= maxPixelCol &&
	   pixel_coord.second >= minPixelCol ) {
	//std::cout << "      !-> trackid   " << linkiter->SimTrackId() << endl;
	//std::cout << "          fraction  " << linkiter->fraction()   << endl;
	SimHitIdpr currentId(linkiter->SimTrackId(), linkiter->eventId());
	//	if(find(idcachev.begin(),idcachev.end(),linkiter->SimTrackId()) == idcachev.end()){
	if(find(idcachev.begin(),idcachev.end(),currentId) == idcachev.end()){
	  //	  simtrackid.push_back(linkiter->SimTrackId());
	  //idcachev.push_back(linkiter->SimTrackId());
	  simtrackid.push_back(currentId);
	  idcachev.push_back(currentId);
	}
      } 
    }
  }
  
  return simtrackid;
}

std::vector<SimHitIdpr>  TrackerHitAssociator::associateGSRecHit(const SiTrackerGSRecHit2D * gsrechit)
{
  //GSRecHit is the FastSimulation RecHit that contains the TrackId already

  vector<SimHitIdpr> simtrackid;
  simtrackid.clear();
  SimHitIdpr currentId(gsrechit->simtrackId(), EncodedEventId(gsrechit->eeId()));
  simtrackid.push_back(currentId);
  return simtrackid;
}

std::vector<PSimHit> TrackerHitAssociator::associateMultiRecHit(const SiTrackerMultiRecHit * multirechit){
        std::vector<const TrackingRecHit*> componenthits = multirechit->recHits();
        std::vector<PSimHit> assimhits;
        std::vector<const TrackingRecHit*>::const_iterator iter;
        for (iter = componenthits.begin(); iter != componenthits.end(); iter ++){
                std::vector<PSimHit> asstocurrent = associateHit(**iter);
                assimhits.insert(assimhits.end(), asstocurrent.begin(), asstocurrent.end());
        }
        //std::cout << "Returning " << assimhits.size() << " simhits" << std::endl;
        return assimhits;
}

std::vector<SimHitIdpr> TrackerHitAssociator::associateMultiRecHitId(const SiTrackerMultiRecHit * multirechit){
        std::vector<const TrackingRecHit*> componenthits = multirechit->recHits();
        std::vector<SimHitIdpr> assimhits;
        std::vector<const TrackingRecHit*>::const_iterator iter;
        for (iter = componenthits.begin(); iter != componenthits.end(); iter ++){
	  std::vector<SimHitIdpr> asstocurrent = associateHitId(**iter);
	  assimhits.insert(assimhits.end(), asstocurrent.begin(), asstocurrent.end());
        }
        //std::cout << "Returning " << assimhits.size() << " simhits" << std::endl;
        return assimhits;
}

std::vector<SimHitIdpr>  TrackerHitAssociator::associateGSMatchedRecHit(const SiTrackerGSMatchedRecHit2D * gsmrechit)
{
  //GSRecHit is the FastSimulation RecHit that contains the TrackId already
  
  vector<SimHitIdpr> simtrackid;
  simtrackid.clear();
  SimHitIdpr currentId(gsmrechit->simtrackId(), EncodedEventId(gsmrechit->eeId()));
  simtrackid.push_back(currentId);
  return simtrackid;
}

