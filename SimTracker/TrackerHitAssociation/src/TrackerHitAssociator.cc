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
  doStrip_( true ), 
  doTrackAssoc_( false ) {
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
  //MAKE THIS PRIVATE MEMBERS 
  //  edm::Handle<CrossingFrame<PSimHit> > cf_simhit;
  //  std::vector<const CrossingFrame<PSimHit> *> cf_simhitvec;

  for(uint32_t i = 0; i< trackerContainers.size();i++){
    e.getByLabel("mix",trackerContainers[i],cf_simhit);
    cf_simhitvec.push_back(cf_simhit.product());
  }
  
  std::auto_ptr<MixCollection<PSimHit> > allTrackerHits(new MixCollection<PSimHit>(cf_simhitvec));
  TrackerHits = (*allTrackerHits);

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
  doTrackAssoc_( conf.getParameter<bool>("associateRecoTracks") ){
  
  trackerContainers.clear();
  trackerContainers = conf.getParameter<std::vector<std::string> >("ROUList");

  //if track association there is no need to acces the CrossingFrame
  if(!doTrackAssoc_) {
    
    // Step A: Get Inputs
    //    edm::Handle<CrossingFrame<PSimHit> > cf_simhit;
    //    std::vector<const CrossingFrame<PSimHit> *> cf_simhitvec;
    for(uint32_t i = 0; i< trackerContainers.size();i++){
      e.getByLabel("mix",trackerContainers[i],cf_simhit);
      cf_simhitvec.push_back(cf_simhit.product());
    }

    //  std::cout << "SIMHITVEC SIZE = " <<  cf_simhitvec.size() << std::endl;
    
    //    TrackerHits = new MixCollection<PSimHit>(cf_simhitvec);
    //std::auto_ptr<MixCollection<PSimHit> > allTrackerHits(TrackerHits);
    //   std::auto_ptr<MixCollection<PSimHit> > allTrackerHits(new MixCollection<PSimHit>(cf_simhitvec));
    std::auto_ptr<MixCollection<PSimHit> > allTrackerHits(new MixCollection<PSimHit>(cf_simhitvec));
    TrackerHits = (*allTrackerHits);
    
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
  //  std::vector<PSimHit> result_old; 
  //initialize vectors!
  simtrackid.clear();
  simhitCFPos.clear();

  //get the Detector type of the rechit
  DetId detid=  thit.geographicalId();
  uint32_t detID = detid.rawId();

  //  cout << "Associator ---> get Detid " << detID << endl;
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
  if( (unsigned int)(detid.subdetId()) == PixelSubdetector::PixelBarrel || 
      (unsigned int)(detid.subdetId()) == PixelSubdetector::PixelEndcap) 
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
  
  //
  //Save the SimHits in a vector. for the macthed hits both the rphi and stereo simhits are saved. 
  //
  //  std::cout << "NEW SIZE =  " << simhitCFPos.size() << std::endl;
  for(int i=0; i<simhitCFPos.size(); i++){
    //std::cout << "NEW CFPOS " << simhitCFPos[i] << endl;
    //std::cout << "NEW LOCALPOS " <<  TrackerHits.getObject(simhitCFPos[i]).localPosition()  << endl;
    result.push_back( TrackerHits.getObject(simhitCFPos[i]));
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
   
  //vector with the associated SimTrackID 
  simtrackid.clear();
  //vector with the CF position of the associated simhits
  simhitCFPos.clear();

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
  else if( (unsigned int)(detid.subdetId()) == PixelSubdetector::PixelBarrel || 
	   (unsigned int)(detid.subdetId()) == PixelSubdetector::PixelEndcap) 
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


std::vector<SimHitIdpr>  TrackerHitAssociator::associateSimpleRecHit(const SiStripRecHit2D * simplerechit)
{
  //  std::cout <<"ASSOCIATE SIMPLE RECHIT" << std::endl;	    

  DetId detid=  simplerechit->geographicalId();
  uint32_t detID = detid.rawId();

  //to store temporary charge information
  std::vector<SimHitIdpr> cache_simtrackid; 
  cache_simtrackid.clear();

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

    //    std::cout << "CLUSTERSIZE " << clusiz << " first strip = " << first << " last strip = " << last << std::endl;
    //   std::cout << " DETSET size = " << link_detset.data.size() << std::endl;
    //use a vector
    std::vector<SimHitIdpr> idcachev;
    std::vector<int> CFposcachev;
    for(edm::DetSet<StripDigiSimLink>::const_iterator linkiter = link_detset.data.begin(); linkiter != link_detset.data.end(); linkiter++){
      StripDigiSimLink link = *linkiter;

      if( (int)(link.channel()) >= first  && (int)(link.channel()) < last ){
	
	//check this digisimlink
	/*
	std::cout << "CHECKING CHANNEL  = " << link.channel()   << std::endl;
	std::cout << "TrackID  = " << link.SimTrackId()  << std::endl;
	std::cout << "Position = " << link.CFposition()  << std::endl;
    	std::cout << " POS -1 = " << TrackerHits.getObject(link.CFposition()-1).localPosition() << std::endl;
  	std::cout << " Process = " << TrackerHits.getObject(link.CFposition()-1).processType() << std::endl;
	*/

	SimHitIdpr currentId(link.SimTrackId(), link.eventId());

	//create a vector with the list of SimTrack ID's of the tracks that contributed to the RecHit
	//write the id only once in the vector

	if(find(idcachev.begin(),idcachev.end(),currentId ) == idcachev.end()){
	  /*
	  std::cout << " Adding track id  = " << currentId.first  
		    << " Event id = " << currentId.second.event() 
		    << " Bunch Xing = " << currentId.second.bunchCrossing() 
		    << std::endl;
	  */
	  idcachev.push_back(currentId);
	  simtrackid.push_back(currentId);
	}

	//create a vector that contains all the position (in the MixCollection) of the SimHits that contributed to the RecHit
	//write position only once
	int currentCFPos = link.CFposition()-1;
	if(find(CFposcachev.begin(),CFposcachev.end(),currentCFPos ) == CFposcachev.end()){
	  /*
	  std::cout << "CHECKING CHANNEL  = " << link.channel()   << std::endl;
	  std::cout << "\tTrackID  = " << link.SimTrackId()  << "\tCFPos = " << currentCFPos  << std::endl;
	  std::cout << "\tLocal Pos = " << TrackerHits.getObject(currentCFPos).localPosition() 
		    << "\tProcess = " << TrackerHits.getObject(currentCFPos).processType() << std::endl;
	  */
	  CFposcachev.push_back(currentCFPos);
	  simhitCFPos.push_back(currentCFPos);
	  //	  simhitassoc.push_back( TrackerHits.getObject(currentCFPos));
	}
      }
    }    
  }
  return simtrackid;
  
}

std::vector<SimHitIdpr>  TrackerHitAssociator::associateMatchedRecHit(const SiStripMatchedRecHit2D * matchedrechit)
{

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
    //    edm::Ref< edm::DetSetVector<SiPixelCluster>, SiPixelCluster> const& cluster = pixelrechit->cluster();
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

