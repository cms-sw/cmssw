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
  doTrackAssoc_( false ),
  useCFpos_( false ) {
  trackerContainers.clear();
  //
  // Take by default all tracker SimHits
  //
  trackerContainers.push_back("g4SimHitsTrackerHitsTIBLowTof");
  trackerContainers.push_back("g4SimHitsTrackerHitsTIBHighTof");
  trackerContainers.push_back("g4SimHitsTrackerHitsTIDLowTof");
  trackerContainers.push_back("g4SimHitsTrackerHitsTIDHighTof");
  trackerContainers.push_back("g4SimHitsTrackerHitsTOBLowTof");
  trackerContainers.push_back("g4SimHitsTrackerHitsTOBHighTof");
  trackerContainers.push_back("g4SimHitsTrackerHitsTECLowTof");
  trackerContainers.push_back("g4SimHitsTrackerHitsTECHighTof");
  trackerContainers.push_back("g4SimHitsTrackerHitsPixelBarrelLowTof");
  trackerContainers.push_back("g4SimHitsTrackerHitsPixelBarrelHighTof");
  trackerContainers.push_back("g4SimHitsTrackerHitsPixelEndcapLowTof");
  trackerContainers.push_back("g4SimHitsTrackerHitsPixelEndcapHighTof");

  // Step A: Get Inputs
  //MAKE THIS PRIVATE MEMBERS 
  //  edm::Handle<CrossingFrame<PSimHit> > cf_simhit;
  //  std::vector<const CrossingFrame<PSimHit> *> cf_simhitvec;

  for(uint32_t i = 0; i< trackerContainers.size();i++){
    e.getByLabel("mix",trackerContainers[i],cf_simhit);
    cf_simhitvec.push_back(cf_simhit.product());
  }
  
  std::auto_ptr<MixCollection<PSimHit> > allTrackerHits(new MixCollection<PSimHit>(cf_simhitvec));
  // TrackerHits = (*allTrackerHits);

 //Loop on PSimHit
  SimHitMap.clear();
  SimHitSubdetMap.clear();
  
  MixCollection<PSimHit>::iterator isim;
  for (isim=allTrackerHits->begin(); isim!= allTrackerHits->end();isim++) {
    SimHitMap[(*isim).detUnitId()].push_back((*isim));
    if (useCFpos_) {
      DetId theDet((*isim).detUnitId());
      SimHitSubdetMap[theDet.subdetId()].push_back((*isim));
    }
  }
  
  if(doStrip_) e.getByLabel("simSiStripDigis", stripdigisimlink);
  if(doPixel_) e.getByLabel("simSiPixelDigis", pixeldigisimlink);
  
}

//
// Constructor with configurables
//
TrackerHitAssociator::TrackerHitAssociator(const edm::Event& e, const edm::ParameterSet& conf)  : 
  myEvent_(e), 
  doPixel_( conf.getParameter<bool>("associatePixel") ),
  doStrip_( conf.getParameter<bool>("associateStrip") ),
  doTrackAssoc_( conf.getParameter<bool>("associateRecoTracks") ),
  useCFpos_( false ) {
  
  //if track association there is no need to acces the CrossingFrame
  if(!doTrackAssoc_) {
    
  trackerContainers.clear();
  trackerContainers = conf.getParameter<std::vector<std::string> >("ROUList");

    // Step A: Get Inputs
    //    edm::Handle<CrossingFrame<PSimHit> > cf_simhit;
    //    std::vector<const CrossingFrame<PSimHit> *> cf_simhitvec;
    for(uint32_t i = 0; i< trackerContainers.size();i++){
      e.getByLabel("mix",trackerContainers[i],cf_simhit);
      cf_simhitvec.push_back(cf_simhit.product());
    }

//      std::cout << "SIMHITVEC SIZE = " <<  cf_simhitvec.size() << std::endl;
    
    //    TrackerHits = new MixCollection<PSimHit>(cf_simhitvec);
    //std::auto_ptr<MixCollection<PSimHit> > allTrackerHits(TrackerHits);
    //   std::auto_ptr<MixCollection<PSimHit> > allTrackerHits(new MixCollection<PSimHit>(cf_simhitvec));
    std::auto_ptr<MixCollection<PSimHit> > allTrackerHits(new MixCollection<PSimHit>(cf_simhitvec));
    // TrackerHits = (*allTrackerHits);
    
    //Loop on PSimHit
    SimHitMap.clear();
    SimHitSubdetMap.clear();
    
    MixCollection<PSimHit>::iterator isim;
    for (isim=allTrackerHits->begin(); isim!= allTrackerHits->end();isim++) {
      SimHitMap[(*isim).detUnitId()].push_back((*isim));
      if (useCFpos_) {
	DetId theDet((*isim).detUnitId());
	SimHitSubdetMap[theDet.subdetId()].push_back((*isim));
      }
    }
    
  }

  if(doStrip_) e.getByLabel("simSiStripDigis", stripdigisimlink);
  if(doPixel_) e.getByLabel("simSiPixelDigis", pixeldigisimlink);
  
}

std::vector<PSimHit> TrackerHitAssociator::associateHit(const TrackingRecHit & thit) 
{
  
  //check in case of TTRH
  if(const TransientTrackingRecHit * ttrh = dynamic_cast<const TransientTrackingRecHit *>(&thit)) {
    std::cout << "calling associateHit for TransientTRH" << std::endl;
    return associateHit(*ttrh->hit());
  }
 
  //vector with the matched SimHit
  std::vector<PSimHit> result; 
  //  std::vector<PSimHit> result_old; 
  //initialize vectors!
  simtrackid.clear();
  simhitCFPos.clear();
  StripHits = false;
  
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
	  //std::cout << "associate to hit2D" << std::endl;
	  associateSimpleRecHit(rechit, simtrackid);
	}
      else if(const SiStripRecHit1D * rechit = 
	      dynamic_cast<const SiStripRecHit1D *>(&thit)) //check if it is a SiStripRecHit1D
	{	  
	  //std::cout << "associate to hit1D" << std::endl;
	  associateSiStripRecHit1D(rechit,simtrackid);
	}
      else if(const SiStripMatchedRecHit2D * rechit = 
	      dynamic_cast<const SiStripMatchedRecHit2D *>(&thit)) //check if it is a matched SiStripMatchedRecHit2D
	{	  
	  //std::cout << "associate to matched" << std::endl;
	  simtrackid = associateMatchedRecHit(rechit);
	}
      else if(const ProjectedSiStripRecHit2D * rechit = 
	      dynamic_cast<const ProjectedSiStripRecHit2D *>(&thit)) //check if it is a  ProjectedSiStripRecHit2D
	{	  
	  //std::cout << "associate to projectedHit" << std::endl;
	  simtrackid = associateProjectedRecHit(rechit);
	  detid = rechit->originalHit().geographicalId();
	  detID = detid.rawId();
	}
      else {
	//std::cout << "associate to Invalid strip hit??" << std::endl;
	//throw cms::Exception("Unknown RecHit Type") << "TrackerHitAssociator failed first casting of " << typeid(thit).name() << " type ";
      }

    }
  //check we are in the pixel tracker
  if( (unsigned int)(detid.subdetId()) == PixelSubdetector::PixelBarrel || 
      (unsigned int)(detid.subdetId()) == PixelSubdetector::PixelEndcap) 
    {
      if(const SiPixelRecHit * rechit = dynamic_cast<const SiPixelRecHit *>(&thit))
	{	  
// 	  std::cout << "associate to pixelHit" << std::endl;
	  associatePixelRecHit(rechit,simtrackid );
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
  
  // From CMSSW_6_2_0_pre8 simhitCFPos is ill-defined.
  // It points relative to the PSimHit collection, but we don't know whether High- or LowTof.
  // Before CMSSW_6_2_0_pre8 it pointed relative to base of the combined TrackerHits.
  if(useCFpos_ && StripHits){
    //USE THIS FOR STRIPS
//     std::cout << "NEW SIZE =  " << simhitCFPos.size() << std::endl;
    
//     for(size_t i=0; i<simhitCFPos.size(); i++){
//       //std::cout << "NEW CFPOS " << simhitCFPos[i] << endl;
//       //std::cout << "NEW LOCALPOS " <<  TrackerHits.getObject(simhitCFPos[i]).localPosition()  << endl;
//       result.push_back( TrackerHits.getObject(simhitCFPos[i]));
//     }
    std::vector<PSimHit> simHit;
    std::map<unsigned int, std::vector<PSimHit> >::const_iterator it = SimHitSubdetMap.find(detid.subdetId());
    simHit.clear();
    if (it!= SimHitSubdetMap.end()){
      simHit = it->second;

      // std::cout << "NEW SIZE = " << simhitCFPos.size() << std::endl; 
      for(size_t i=0; i<simhitCFPos.size(); i++){
//         std::cout << "NEW CFPOS " << simhitCFPos[i] << endl;
        result.push_back(simHit[simhitCFPos[i]]);
      }
    }
  }else {
    
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
// 	    	  cout << "Associator ---> ID" << ihit.trackId() << " Simhit x= " << ihit.localPosition().x() 
// 	    	       << " y= " <<  ihit.localPosition().y() << " z= " <<  ihit.localPosition().x() << endl; 
	    result.push_back(ihit);
	    continue;
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
	      continue;
	    }
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
  std::vector< SimHitIdpr > simhitid;
  associateHitId(thit, simhitid);
  return simhitid;
}

void TrackerHitAssociator::associateHitId(const TrackingRecHit & thit, std::vector< SimHitIdpr > & simtkid) 
{
  
  //check in case of TTRH
  if(const TransientTrackingRecHit * ttrh = dynamic_cast<const TransientTrackingRecHit *>(&thit)) {
    associateHitId(*ttrh->hit(), simtkid);
  }
  else{
    simtkid.clear();
    simhitCFPos.clear();
    //vector with the associated Simtkid 
    //vector with the CF position of the associated simhits
    
  //get the Detector type of the rechit
    DetId detid=  thit.geographicalId();
    //apparently not used
    //  uint32_t detID = detid.rawId();
    if (const SiTrackerMultiRecHit * rechit = dynamic_cast<const SiTrackerMultiRecHit *>(&thit)){
       simtkid=associateMultiRecHitId(rechit);
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
	    associateSimpleRecHit(rechit, simtkid);
	  }
	//check if it is a matched SiStripMatchedRecHit2D
	else  if(const SiStripRecHit1D * rechit = 
		 dynamic_cast<const SiStripRecHit1D *>(&thit))
	  {	  
	    associateSiStripRecHit1D(rechit,simtkid);
	  }
	//check if it is a matched SiStripMatchedRecHit2D
	else  if(const SiStripMatchedRecHit2D * rechit = 
		 dynamic_cast<const SiStripMatchedRecHit2D *>(&thit))
	  {	  
	    simtkid = associateMatchedRecHit(rechit);
	  }
	//check if it is a  ProjectedSiStripRecHit2D
	else if(const ProjectedSiStripRecHit2D * rechit = 
		dynamic_cast<const ProjectedSiStripRecHit2D *>(&thit))
	  {	  
	    simtkid = associateProjectedRecHit(rechit);
	  }
	else{
	  //std::cout << "associate to invalid" << std::endl;
	  //throw cms::Exception("Unknown RecHit Type") << "TrackerHitAssociator failed second casting of " << typeid(thit).name() << " type ";
	}
      }
    //check we are in the pixel tracker
    else if( (unsigned int)(detid.subdetId()) == PixelSubdetector::PixelBarrel || 
	     (unsigned int)(detid.subdetId()) == PixelSubdetector::PixelEndcap) 
      {
      if(const SiPixelRecHit * rechit = dynamic_cast<const SiPixelRecHit *>(&thit))
	{	  
	  associatePixelRecHit(rechit,simtkid );
	}
      }
    //check if these are GSRecHits (from FastSim)
    if(const SiTrackerGSRecHit2D * rechit = dynamic_cast<const SiTrackerGSRecHit2D *>(&thit))
      {
	simtkid = associateGSRecHit(rechit);
      }  
    if(const SiTrackerGSMatchedRecHit2D * rechit = dynamic_cast<const SiTrackerGSMatchedRecHit2D *>(&thit))
      {
	simtkid = associateGSMatchedRecHit(rechit);
      }

  }
}


void TrackerHitAssociator::associateSimpleRecHit(const SiStripRecHit2D * simplerechit, std::vector<SimHitIdpr>& simtrackid)
{
  const SiStripCluster* clust = 0; 
  if(simplerechit->cluster().isNonnull())
    {
      clust=&(*simplerechit->cluster());
    }
  else if(simplerechit->cluster_regional().isNonnull())
    {
      clust=&(*simplerechit->cluster_regional());
    } 
  
  associateSimpleRecHitCluster(clust,simplerechit->geographicalId(),simtrackid);
}


//This function could be templated to avoid to repeat the same code twice??
void TrackerHitAssociator::associateSiStripRecHit1D(const SiStripRecHit1D * simplerechit, std::vector<SimHitIdpr>& simtrackid)
{
  const SiStripCluster* clust = 0; 
  if(simplerechit->cluster().isNonnull())
    {
      clust=&(*simplerechit->cluster());
    }
  else if(simplerechit->cluster_regional().isNonnull())
    {
      clust=&(*simplerechit->cluster_regional());
    } 
  associateSimpleRecHitCluster(clust,simplerechit->geographicalId(),simtrackid);
}

/*
void TrackerHitAssociator::associateSimpleRecHitCluster(const SiStripCluster* clust,
							const uint32_t& detID,
							std::vector<SimHitIdpr>& theSimtrackid, std::vector<PSimHit>& clusterSimHits)
{
// Caller needs to clear clusterSimHits before calling this function

  //initialize vector
  theSimtrackid.clear();
  //initialize class vector
  simhitCFPos.clear();

  associateSimpleRecHitCluster(clust, detID, theSimtrackid);

  DetId theDet(detID);
  std::vector<PSimHit> subDetSimHits;
  std::map<unsigned int, std::vector<PSimHit> >::const_iterator it = SimHitSubdetMap.find(theDet.subdetId());
  subDetSimHits.clear();
  if (it!= SimHitSubdetMap.end()){
    subDetSimHits = it->second;
    for(size_t i=0; i<simhitCFPos.size(); i++){
//     clusterSimHits.push_back(TrackerHits.getObject(simhitCFPos[i]));
      clusterSimHits.push_back(subDetSimHits[simhitCFPos[i]]);
    }
  }
}

void TrackerHitAssociator::associateSimpleRecHitCluster(const SiStripCluster* clust,
							const uint32_t& detID,
							std::vector<PSimHit>& clusterSimHits)
{
// Caller needs to clear clusterSimHits before calling this function

  //initialize class vectors
  simtrackid.clear();
  simhitCFPos.clear();

  associateSimpleRecHitCluster(clust, detID, simtrackid);


  DetId theDet(detID);
  std::vector<PSimHit> subDetSimHits;
  std::map<unsigned int, std::vector<PSimHit> >::const_iterator it = SimHitSubdetMap.find(theDet.subdetId());
  subDetSimHits.clear();
  if (it!= SimHitSubdetMap.end()){
    subDetSimHits = it->second;
    for(size_t i=0; i<simhitCFPos.size(); i++){
//     clusterSimHits.push_back(TrackerHits.getObject(simhitCFPos[i]));
      clusterSimHits.push_back(subDetSimHits[simhitCFPos[i]]);
    }
  }
}
*/

void TrackerHitAssociator::associateSimpleRecHitCluster(const SiStripCluster* clust,
							const uint32_t& detID,
							std::vector<SimHitIdpr>& simtrackid){
  //  std::cout <<"ASSOCIATE SIMPLE RECHIT" << std::endl;	    
  StripHits =true;	  
  
  //DetId detid=  clust->geographicalId();
  //  uint32_t detID = detid.rawId();
  
  //to store temporary charge information
  std::vector<SimHitIdpr> cache_simtrackid; 
  cache_simtrackid.clear();
  
  std::map<SimHitIdpr, vector<float> > temp_simtrackid;
  temp_simtrackid.clear();
  
  edm::DetSetVector<StripDigiSimLink>::const_iterator isearch = stripdigisimlink->find(detID); 
  if(isearch != stripdigisimlink->end()) {  //if it is not empty
    //link_detset is a structure, link_detset.data is a std::vector<StripDigiSimLink>
    //edm::DetSet<StripDigiSimLink> link_detset = (*stripdigisimlink)[detID];
    edm::DetSet<StripDigiSimLink> link_detset = (*isearch);
    
    if(clust!=0){//the cluster is valid

      //    const edm::Ref<edm::DetSetVector<SiStripCluster>, SiStripCluster, edm::refhelper::FindForDetSetVector<SiStripCluster> > clust=simplerechit->cluster();
      
      //float chg;
      int clusiz = clust->amplitudes().size();
      int first  = clust->firstStrip();     
      int last   = first + clusiz;
      
//       std::cout << "CLUSTERSIZE " << clusiz << " first strip = " << first << " last strip = " << last-1 << std::endl;
//       std::cout << " detID = " << detID << " DETSET size = " << link_detset.data.size() << std::endl;
      //use a vector
      std::vector<SimHitIdpr> idcachev;
      std::vector<int> CFposcachev;
      for(edm::DetSet<StripDigiSimLink>::const_iterator linkiter = link_detset.data.begin(); linkiter != link_detset.data.end(); linkiter++){
	//StripDigiSimLink link = *linkiter;
	
	if( (int)(linkiter->channel()) >= first  && (int)(linkiter->channel()) < last ){
	  
	  //check this digisimlink
// 	  printf("%s%4d%s%8d%s%3d%s%8.4f\n", "CHANNEL = ", linkiter->channel(), " TrackID = ", linkiter->SimTrackId(),
// 		 " Process = ", TrackerHits.getObject(linkiter->CFposition()-1).processType(), " fraction = ", linkiter->fraction());
	  /*
	    std::cout << "CHECKING CHANNEL  = " << linkiter->channel()   << std::endl;
	    std::cout << "TrackID  = " << linkiter->SimTrackId()  << std::endl;
	    std::cout << "Position = " << linkiter->CFposition()  << std::endl;
	    std::cout << " POS -1 = " << TrackerHits.getObject(linkiter->CFposition()-1).localPosition() << std::endl;
	    std::cout << " Process = " << TrackerHits.getObject(linkiter->CFposition()-1).processType() << std::endl;
	    std::cout << " fraction = " << linkiter->fraction() << std::endl;
	  */
	  
	  SimHitIdpr currentId(linkiter->SimTrackId(), linkiter->eventId());
	  
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
	  int currentCFPos = linkiter->CFposition()-1;
	  if(useCFpos_ && find(CFposcachev.begin(),CFposcachev.end(),currentCFPos ) == CFposcachev.end()){
// 	    std::cout << "CHECKING CHANNEL  = " << linkiter->channel()   << std::endl;
// 	    std::cout << "\tTrackID  = " << linkiter->SimTrackId()  << "\tCFPos = " << currentCFPos  << std::endl;
// 	    std::cout << "\tLocal Pos = " << TrackerHits.getObject(currentCFPos).localPosition()
// 		      << "\tProcess = " << TrackerHits.getObject(currentCFPos).processType() << std::endl;
	    CFposcachev.push_back(currentCFPos);
	    simhitCFPos.push_back(currentCFPos);
	    //	  simhitassoc.push_back( TrackerHits.getObject(currentCFPos));
	  }
	}
      }    
    }
    else {
      edm::LogError("TrackerHitAssociator")<<"no cluster reference attached";
    }
  }
}

std::vector<SimHitIdpr>  TrackerHitAssociator::associateMatchedRecHit(const SiStripMatchedRecHit2D * matchedrechit)
{

  StripHits = true;


  vector<SimHitIdpr> matched_mono;
  vector<SimHitIdpr> matched_st;
  matched_mono.clear();
  matched_st.clear();

  const SiStripRecHit2D mono = matchedrechit->monoHit();
  const SiStripRecHit2D st = matchedrechit->stereoHit();
  //associate the two simple hits separately
  associateSimpleRecHit(&mono,matched_mono );
  associateSimpleRecHit(&st, matched_st );
  
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
  StripHits = true;


  //projectedRecHit is a "matched" rechit with only one component

  vector<SimHitIdpr> matched_mono;
  matched_mono.clear();
 
  const SiStripRecHit2D mono = projectedrechit->originalHit();
  associateSimpleRecHit(&mono, matched_mono);
  return matched_mono;
}

//std::vector<unsigned int>  TrackerHitAssociator::associatePixelRecHit(const SiPixelRecHit * pixelrechit)
void  TrackerHitAssociator::associatePixelRecHit(const SiPixelRecHit * pixelrechit, std::vector<SimHitIdpr> & simtrackid)
{
  StripHits = false;

  //
  // Pixel associator
  //
  DetId detid=  pixelrechit->geographicalId();
  uint32_t detID = detid.rawId();

  edm::DetSetVector<PixelDigiSimLink>::const_iterator isearch = pixeldigisimlink->find(detID); 
  if(isearch != pixeldigisimlink->end()) {  //if it is not empty
    //    edm::DetSet<PixelDigiSimLink> link_detset = (*pixeldigisimlink)[detID];
    edm::DetSet<PixelDigiSimLink> link_detset = (*isearch);
    //    edm::Ref< edm::DetSetVector<SiPixelCluster>, SiPixelCluster> const& cluster = pixelrechit->cluster();
    SiPixelRecHit::ClusterRef const& cluster = pixelrechit->cluster();
    
    //check the reference is valid
    
    if(!(cluster.isNull())){//if the cluster is valid
      
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
    else{      
      edm::LogError("TrackerHitAssociator")<<"no Pixel cluster reference attached";
      
    }
  }
  

}

std::vector<SimHitIdpr>  TrackerHitAssociator::associateGSRecHit(const SiTrackerGSRecHit2D * gsrechit)
{
  StripHits = false;
  //GSRecHit is the FastSimulation RecHit that contains the TrackId already

  vector<SimHitIdpr> simtrackid;
  simtrackid.clear();
  SimHitIdpr currentId(gsrechit->simtrackId(), EncodedEventId(gsrechit->eeId()));
  simtrackid.push_back(currentId);
  return simtrackid;
}

std::vector<PSimHit> TrackerHitAssociator::associateMultiRecHit(const SiTrackerMultiRecHit * multirechit){
  std::vector<const TrackingRecHit*> componenthits = multirechit->recHits();
  //        std::vector<PSimHit> assimhits;
  int size=multirechit->weights().size(), idmostprobable=0;
  
  for (int i=0; i<size; i++){
    if(multirechit->weight(i)>multirechit->weight(idmostprobable)) idmostprobable=i;
  }
  
  //	std::vector<PSimHit> assimhits = associateHit(**mostpropable);
  //	assimhits.insert(assimhits.end(), asstocurrent.begin(), asstocurrent.end());
  //std::cout << "Returning " << assimhits.size() << " simhits" << std::endl;
  return associateHit(*componenthits[idmostprobable]);
}

std::vector<SimHitIdpr> TrackerHitAssociator::associateMultiRecHitId(const SiTrackerMultiRecHit * multirechit){
  std::vector<const TrackingRecHit*> componenthits = multirechit->recHits();
  int size=multirechit->weights().size(), idmostprobable=0;
  
  for (int i=0; i<size; i++){
    if(multirechit->weight(i)>multirechit->weight(idmostprobable)) idmostprobable=i;
  }
  
  return associateHitId(*componenthits[idmostprobable]);
}

std::vector<SimHitIdpr>  TrackerHitAssociator::associateGSMatchedRecHit(const SiTrackerGSMatchedRecHit2D * gsmrechit)
{
  StripHits = false;
  //GSRecHit is the FastSimulation RecHit that contains the TrackId already
  
  vector<SimHitIdpr> simtrackid;
  simtrackid.clear();
  SimHitIdpr currentId(gsmrechit->simtrackId(), EncodedEventId(gsmrechit->eeId()));
  simtrackid.push_back(currentId);
  return simtrackid;
}

