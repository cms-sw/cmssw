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
// Constructor for Config helper class, using default parameters
//
TrackerHitAssociator::Config::Config(edm::ConsumesCollector && iC) :
  doPixel_(true),
  doStrip_(true),
  doTrackAssoc_(false),
  assocHitbySimTrack_(false) {

  if(doStrip_) stripToken_ = iC.consumes<edm::DetSetVector<StripDigiSimLink> >(edm::InputTag("simSiStripDigis"));
  if(doPixel_) pixelToken_ = iC.consumes<edm::DetSetVector<PixelDigiSimLink> >(edm::InputTag("simSiPixelDigis"));
  if(!doTrackAssoc_) {
    std::vector<std::string> trackerContainers;
    trackerContainers.reserve(12);
    trackerContainers.emplace_back("g4SimHitsTrackerHitsTIBLowTof");
    trackerContainers.emplace_back("g4SimHitsTrackerHitsTIBHighTof");
    trackerContainers.emplace_back("g4SimHitsTrackerHitsTIDLowTof");
    trackerContainers.emplace_back("g4SimHitsTrackerHitsTIDHighTof");
    trackerContainers.emplace_back("g4SimHitsTrackerHitsTOBLowTof");
    trackerContainers.emplace_back("g4SimHitsTrackerHitsTOBHighTof");
    trackerContainers.emplace_back("g4SimHitsTrackerHitsTECLowTof");
    trackerContainers.emplace_back("g4SimHitsTrackerHitsTECHighTof");
    trackerContainers.emplace_back("g4SimHitsTrackerHitsPixelBarrelLowTof");
    trackerContainers.emplace_back("g4SimHitsTrackerHitsPixelBarrelHighTof");
    trackerContainers.emplace_back("g4SimHitsTrackerHitsPixelEndcapLowTof");
    trackerContainers.emplace_back("g4SimHitsTrackerHitsPixelEndcapHighTof");
    cfTokens_.reserve(trackerContainers.size());
    simHitTokens_.reserve(trackerContainers.size());
    for(auto const& trackerContainer : trackerContainers) {
      cfTokens_.push_back(iC.consumes<CrossingFrame<PSimHit> >(edm::InputTag("mix", trackerContainer)));
      simHitTokens_.push_back(iC.consumes<std::vector<PSimHit> >(edm::InputTag("g4SimHits", trackerContainer)));
    }
  }
}

//
// Constructor for Config helper class, using configured parameters
//
TrackerHitAssociator::Config::Config(const edm::ParameterSet& conf, edm::ConsumesCollector && iC) :
  doPixel_( conf.getParameter<bool>("associatePixel") ),
  doStrip_( conf.getParameter<bool>("associateStrip") ),
  doTrackAssoc_( conf.getParameter<bool>("associateRecoTracks") ),
  assocHitbySimTrack_(conf.existsAs<bool>("associateHitbySimTrack") ? conf.getParameter<bool>("associateHitbySimTrack") : false) {

  if(doStrip_) stripToken_ = iC.consumes<edm::DetSetVector<StripDigiSimLink> >(conf.getParameter<edm::InputTag>("stripSimLinkSrc"));
  if(doPixel_) pixelToken_ = iC.consumes<edm::DetSetVector<PixelDigiSimLink> >(conf.getParameter<edm::InputTag>("pixelSimLinkSrc"));
  if(!doTrackAssoc_) {
    std::vector<std::string> trackerContainers(conf.getParameter<std::vector<std::string> >("ROUList"));
    cfTokens_.reserve(trackerContainers.size());
    simHitTokens_.reserve(trackerContainers.size());
    for(auto const& trackerContainer : trackerContainers) {
      cfTokens_.push_back(iC.consumes<CrossingFrame<PSimHit> >(edm::InputTag("mix", trackerContainer)));
      simHitTokens_.push_back(iC.consumes<std::vector<PSimHit> >(edm::InputTag("g4SimHits", trackerContainer)));
    }
  }
 }

//
// Constructor supporting consumes interface
//
TrackerHitAssociator::TrackerHitAssociator(const edm::Event& e, const TrackerHitAssociator::Config& config) :
  doPixel_(config.doPixel_),
  doStrip_(config.doStrip_),
  doTrackAssoc_(config.doTrackAssoc_),
  assocHitbySimTrack_(config.assocHitbySimTrack_) {
  //if track association there is no need to access the input collections
  if(!doTrackAssoc_) {
    makeMaps(e, config);
  }

  if(doStrip_) e.getByToken(config.stripToken_, stripdigisimlink);
  if(doPixel_) e.getByToken(config.pixelToken_, pixeldigisimlink);
}

void TrackerHitAssociator::makeMaps(const edm::Event& theEvent, const TrackerHitAssociator::Config& config) {
  // Step A: Get Inputs
  //  The collections are specified via ROUList in the configuration, and can
  //  be either crossing frames (e.g., mix/g4SimHitsTrackerHitsTIBLowTof)
  //  or just PSimHits (e.g., g4SimHits/TrackerHitsTIBLowTof)
  if (assocHitbySimTrack_) {
    for(auto const& cfToken : config.cfTokens_) {
      edm::Handle<CrossingFrame<PSimHit> > cf_simhit;
      //int Nhits = 0;
      if (theEvent.getByToken(cfToken, cf_simhit)) {
        std::unique_ptr<MixCollection<PSimHit> > thisContainerHits(new MixCollection<PSimHit>(cf_simhit.product())); 
        for (auto const& isim : *thisContainerHits) {
          DetId theDet(isim.detUnitId());
          SimHitMap[theDet].push_back(isim);
        //  ++Nhits;
        }
      // std::cout << "simHits from crossing frames; map size = " << SimHitCollMap.size() << ", Hit count = " << Nhits << std::endl;
      }
    }
    for(auto const& simHitToken : config.simHitTokens_) {
      edm::Handle<std::vector<PSimHit> > simHits;
      //int Nhits = 0;
      if(theEvent.getByToken(simHitToken, simHits)) {
        for (auto const& isim : *simHits) {
          DetId theDet(isim.detUnitId());
          SimHitMap[theDet].push_back(isim);
        //++Nhits;
        }
      // std::cout << "simHits from prompt collections; map size = " << SimHitCollMap.size() << ", Hit count = " << Nhits << std::endl;
      }
    }
  } else {
    const char* const highTag = "HighTof";
    unsigned int tofBin;
    edm::EDConsumerBase::Labels labels;
    simHitCollectionID theSimHitCollID;
    for(auto const& cfToken : config.cfTokens_) {
      edm::Handle<CrossingFrame<PSimHit> > cf_simhit;
      //int Nhits = 0;
      if (theEvent.getByToken(cfToken, cf_simhit)) {
        std::unique_ptr<MixCollection<PSimHit> > thisContainerHits(new MixCollection<PSimHit>(cf_simhit.product())); 
        theEvent.labelsForToken(cfToken, labels);
        if(std::strstr(labels.productInstance, highTag) != NULL) {
          tofBin = StripDigiSimLink::HighTof;
        } else {
          tofBin = StripDigiSimLink::LowTof; 
        }    
        for (auto const& isim : *thisContainerHits) {
          DetId theDet(isim.detUnitId());
          theSimHitCollID = std::make_pair(theDet.subdetId(), tofBin);
          SimHitCollMap[theSimHitCollID].push_back(isim);
        //++Nhits;
        }
      // std::cout << "simHits from crossing frames; map size = " << SimHitCollMap.size() << ", Hit count = " << Nhits << std::endl;
      }
    }
    for(auto const& simHitToken : config.simHitTokens_) {
      edm::Handle<std::vector<PSimHit> > simHits;
      //int Nhits = 0;
      if(theEvent.getByToken(simHitToken, simHits)) {
        theEvent.labelsForToken(simHitToken, labels);
        if(std::strstr(labels.productInstance, highTag) != NULL) {
          tofBin = StripDigiSimLink::HighTof;
        } else {
          tofBin = StripDigiSimLink::LowTof; 
        }
        for (auto const& isim : *simHits) {
          DetId theDet(isim.detUnitId());
          theSimHitCollID = std::make_pair(theDet.subdetId(), tofBin);
          SimHitCollMap[theSimHitCollID].push_back(isim);
        //++Nhits;
        }
      // std::cout << "simHits from prompt collections; map size = " << SimHitCollMap.size() << ", Hit count = " << Nhits << std::endl;
      }
    }
  }
}

std::vector<PSimHit> TrackerHitAssociator::associateHit(const TrackingRecHit & thit) const
{

  if (const SiTrackerMultiRecHit * rechit = dynamic_cast<const SiTrackerMultiRecHit *>(&thit)){
    return associateMultiRecHit(rechit);
  }

  //vector with the matched SimHit
  std::vector<PSimHit> result;

  if(doTrackAssoc_) return result;

  // Vectors to contain lists of matched simTracks, simHits
  std::vector<SimHitIdpr> simtrackid;
  std::vector<simhitAddr> simhitCFPos;
  
  //get the Detector type of the rechit
  DetId detid=  thit.geographicalId();
  uint32_t detID = detid.rawId();

  // Get the vectors of simtrackIDs and simHit indices associated with this rechit
  associateHitId(thit, simtrackid, &simhitCFPos);
  // std::cout << "recHit subdet, detID = " << detid.subdetId() << ", " << detID << ", (bnch, evt, trk) = ";
  // for (size_t i=0; i<simtrackid.size(); ++i)
  //   std::cout << ", (" << simtrackid[i].second.bunchCrossing() << ", "
  // 	      << simtrackid[i].second.event() << ", " << simtrackid[i].first << ")";
  // std::cout << std::endl; 

  // Get the vector of simHits associated with this rechit

  if (!assocHitbySimTrack_ && simhitCFPos.size() > 0) {
    // We use the indices to the simHit collections taken
    //  from the DigiSimLinks and returned in simhitCFPos.
    //  simhitCFPos[i] contains the full address of the ith simhit:
    //   <collection, index> = <<subdet, tofBin>, index>

    //check if the recHit is a SiStripMatchedRecHit2D
    if(dynamic_cast<const SiStripMatchedRecHit2D *>(&thit)) {
      for(auto const& theSimHitAddr : simhitCFPos) {
        simHitCollectionID theSimHitCollID = theSimHitAddr.first;
        simhit_collectionMap::const_iterator it = SimHitCollMap.find(theSimHitCollID);
        if (it!= SimHitCollMap.end()) {
          unsigned int theSimHitIndex = theSimHitAddr.second;
          if (theSimHitIndex < (it->second).size()) {
            const PSimHit& theSimHit = (it->second)[theSimHitIndex];
            // Try to remove ghosts by requiring a match to the simTrack also
            unsigned int simHitid = theSimHit.trackId();
            EncodedEventId simHiteid = theSimHit.eventId();
            for(auto const& id : simtrackid) {
              if(simHitid == id.first && simHiteid == id.second) {
                result.push_back(theSimHit);
              }
            } 
          // std::cout << "by CFpos, simHit detId =  " << theSimHit.detUnitId() << " address = (" << (theSimHitAddr.first).first
	  // 	    << ", " << (theSimHitAddr.first).second << ", " << theSimHitIndex
	  // 	    << "), process = " << theSimHit.processType() << " (" << theSimHit.eventId().bunchCrossing()
	  // 	    << ", " << theSimHit.eventId().event() << ", " << theSimHit.trackId() << ")" << std::endl;
	  }
        }
      }
    } else {
      for(auto const& theSimHitAddr : simhitCFPos) {
        simHitCollectionID theSimHitCollID = theSimHitAddr.first;
        simhit_collectionMap::const_iterator it = SimHitCollMap.find(theSimHitCollID);
        if (it!= SimHitCollMap.end()) {
          unsigned int theSimHitIndex = theSimHitAddr.second;
          if (theSimHitIndex < (it->second).size()) {
             result.push_back((it->second)[theSimHitIndex]);
          // std::cout << "by CFpos, simHit detId =  " << theSimHit.detUnitId() << " address = (" << (theSimHitAddr.first).first
	  // 	    << ", " << (theSimHitAddr.first).second << ", " << theSimHitIndex
	  // 	    << "), process = " << theSimHit.processType() << " (" << theSimHit.eventId().bunchCrossing()
	  // 	    << ", " << theSimHit.eventId().event() << ", " << theSimHit.trackId() << ")" << std::endl;
          }
        }
      }
    }
    return result;
  }

  // Get the SimHit from the trackid instead
  std::map<unsigned int, std::vector<PSimHit> >::const_iterator it = SimHitMap.find(detID);
  if (it!= SimHitMap.end()) {
    vector<PSimHit>::const_iterator simHitIter = (it->second).begin();
    vector<PSimHit>::const_iterator simHitIterEnd = (it->second).end();
    for (;simHitIter != simHitIterEnd; ++simHitIter) {
      const PSimHit& ihit = *simHitIter;
      unsigned int simHitid = ihit.trackId();
      EncodedEventId simHiteid = ihit.eventId();
      // std::cout << "by simTk, simHit, process = " << ihit.processType() << " (" << ihit.eventId().bunchCrossing()
      // 		<< ", " << ihit.eventId().event() << ", " << ihit.trackId() << ")";
      for(auto id : simtrackid) {
        if(simHitid == id.first && simHiteid == id.second) {
// 	    	cout << "Associator ---> ID" << ihit.trackId() << " Simhit x= " << ihit.localPosition().x() 
//                   << " y= " <<  ihit.localPosition().y() << " z= " <<  ihit.localPosition().x() << endl;
	  // std::cout << " matches";
	  result.push_back(ihit);
	  break;
        }
      } 
      // std::cout << std::endl;
    }

  }else{

    /// Check if it's the gluedDet.
    std::map<unsigned int, std::vector<PSimHit> >::const_iterator itrphi =
      SimHitMap.find(detID+2);  //iterator to the simhit in the rphi module
    std::map<unsigned int, std::vector<PSimHit> >::const_iterator itster = 
      SimHitMap.find(detID+1);//iterator to the simhit in the stereo module
    if (itrphi!= SimHitMap.end()&&itster!=SimHitMap.end()) {
      std::vector<PSimHit> simHitVector = itrphi->second;
      simHitVector.insert(simHitVector.end(),(itster->second).begin(),(itster->second).end());
      vector<PSimHit>::const_iterator simHitIter = simHitVector.begin();
      vector<PSimHit>::const_iterator simHitIterEnd = simHitVector.end();
      for (;simHitIter != simHitIterEnd; ++simHitIter) {
	const PSimHit& ihit = *simHitIter;
	unsigned int simHitid = ihit.trackId();
	EncodedEventId simHiteid = ihit.eventId();
	for(auto const& id : simtrackid) {
	  if(simHitid == id.first && simHiteid == id.second) { 
// 	  if(simHitid == simtrackid[i].first && simHiteid.bunchCrossing() == simtrackid[i].second.bunchCrossing()) {
	    //	  cout << "GluedDet Associator ---> ID" << ihit.trackId() << " Simhit x= " << ihit.localPosition().x() 
            //         << " y= " <<  ihit.localPosition().y() << " z= " <<  ihit.localPosition().x() << endl; 
	    result.push_back(ihit);
	    break;
	  }
	}
      }
    }
  }

  return result;
}

std::vector< SimHitIdpr > TrackerHitAssociator::associateHitId(const TrackingRecHit & thit) const
{
  std::vector< SimHitIdpr > simhitid;
  associateHitId(thit, simhitid);
  return simhitid;
}

void TrackerHitAssociator::associateHitId(const TrackingRecHit & thit, std::vector< SimHitIdpr > & simtkid,
                                          std::vector<simhitAddr>* simhitCFPos) const
{

    simtkid.clear();
    
  //get the Detector type of the rechit
    DetId detid=  thit.geographicalId();
    if (const SiTrackerMultiRecHit * rechit = dynamic_cast<const SiTrackerMultiRecHit *>(&thit)){
       simtkid=associateMultiRecHitId(rechit, simhitCFPos);
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
	    associateSiStripRecHit(rechit, simtkid, simhitCFPos);
	  }
	//check if it is a SiStripRecHit1D
	else  if(const SiStripRecHit1D * rechit = 
		 dynamic_cast<const SiStripRecHit1D *>(&thit))
	  {	  
	    associateSiStripRecHit(rechit, simtkid, simhitCFPos);
	  }
	//check if it is a SiStripMatchedRecHit2D
	else  if(const SiStripMatchedRecHit2D * rechit = 
		 dynamic_cast<const SiStripMatchedRecHit2D *>(&thit))
	  {	  
	    simtkid = associateMatchedRecHit(rechit, simhitCFPos);
	  }
	//check if it is a  ProjectedSiStripRecHit2D
	else if(const ProjectedSiStripRecHit2D * rechit = 
		dynamic_cast<const ProjectedSiStripRecHit2D *>(&thit))
	  {	  
	    simtkid = associateProjectedRecHit(rechit, simhitCFPos);
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
	  associatePixelRecHit(rechit, simtkid, simhitCFPos);
	}
      }
    //check if these are GSRecHits (from FastSim)
    if(trackerHitRTTI::isFast(thit))
      {
	  simtkid = associateFastRecHit(static_cast<const FastTrackerRecHit *>(&thit));
      }
}

template<typename T>
inline void TrackerHitAssociator::associateSiStripRecHit(const T *simplerechit, std::vector<SimHitIdpr>& simtrackid, std::vector<simhitAddr>* simhitCFPos) const
{
  const SiStripCluster* clust = &(*simplerechit->cluster());
  associateSimpleRecHitCluster(clust, simplerechit->geographicalId(), simtrackid, simhitCFPos);
}

//
//  Method for obtaining simTracks and simHits from a cluster
//
void TrackerHitAssociator::associateCluster(const SiStripCluster* clust,
					    const DetId& detid,
					    std::vector<SimHitIdpr>& simtrackid,
					    std::vector<PSimHit>& simhit) const {
  std::vector<simhitAddr> simhitCFPos;
  associateSimpleRecHitCluster(clust, detid, simtrackid, &simhitCFPos);

  for(auto const& theSimHitAddr : simhitCFPos ) {
    simHitCollectionID theSimHitCollID = theSimHitAddr.first;
    simhit_collectionMap::const_iterator it = SimHitCollMap.find(theSimHitCollID);
    if (it!= SimHitCollMap.end()) {
      unsigned int theSimHitIndex = theSimHitAddr.second;
      if (theSimHitIndex < (it->second).size()) simhit.push_back((it->second)[theSimHitIndex]);
      // const PSimHit& theSimHit = (it->second)[theSimHitIndex];
      // std::cout << "For cluster, simHit detId =  " << theSimHit.detUnitId() << " address = (" << (theSimHitAddr.first).first
      // 		<< ", " << (theSimHitAddr.first).second << ", " << theSimHitIndex
      // 		<< "), process = " << theSimHit.processType() << " (" << theSimHit.eventId().bunchCrossing()
      // 		<< ", " << theSimHit.eventId().event() << ", " << theSimHit.trackId() << ")" << std::endl;
    }
  }
}

void TrackerHitAssociator::associateSimpleRecHitCluster(const SiStripCluster* clust,
							const DetId& detid,
							std::vector<SimHitIdpr>& simtrackid,
							std::vector<simhitAddr>* simhitCFPos) const {
  
  uint32_t detID = detid.rawId();
  edm::DetSetVector<StripDigiSimLink>::const_iterator isearch = stripdigisimlink->find(detID); 
  if(isearch != stripdigisimlink->end()) {  //if it is not empty
    edm::DetSet<StripDigiSimLink> link_detset = (*isearch);
    
    if(clust!=0){//the cluster is valid
      int clusiz = clust->amplitudes().size();
      int first  = clust->firstStrip();     
      int last   = first + clusiz;
      
//       std::cout << "CLUSTERSIZE " << clusiz << " first strip = " << first << " last strip = " << last-1 << std::endl;
//       std::cout << " detID = " << detID << " DETSET size = " << link_detset.data.size() << std::endl;
      //use a vector
      std::vector<SimHitIdpr> idcachev;
      std::vector<simhitAddr> CFposcachev;
      int channel;
//      for(auto linkiter : link_detset.data){
      for(edm::DetSet<StripDigiSimLink>::const_iterator linkiter = link_detset.data.begin(), linkerEnd = link_detset.data.end(); linkiter != linkerEnd; ++linkiter){      
        channel = (int)(linkiter->channel()); 
	if( channel >= first  && channel < last ){
	  
	  //check this digisimlink
// 	  printf("%s%4d%s%8d%s%3d%s%8.4f\n", "CHANNEL = ", linkiter->channel(), " TrackID = ", linkiter->SimTrackId(),
// 		 " tofBin = ", linkiter->TofBin(), " fraction = ", linkiter->fraction());
	  /*
	    std::cout << "CHECKING CHANNEL  = " << linkiter->channel()   << std::endl;
	    std::cout << "TrackID  = " << linkiter->SimTrackId()  << std::endl;
	    std::cout << "Position = " << linkiter->CFposition()  << std::endl;
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
	  	  
	  if (simhitCFPos != 0) {
	  //create a vector that contains all the position (in the MixCollection) of the SimHits that contributed to the RecHit
	  //write position only once
	    unsigned int currentCFPos = linkiter->CFposition();
	    unsigned int tofBin = linkiter->TofBin();
	    simHitCollectionID theSimHitCollID = std::make_pair(detid.subdetId(), tofBin);
	    simhitAddr currentAddr = std::make_pair(theSimHitCollID, currentCFPos);

	    if(find(CFposcachev.begin(), CFposcachev.end(), currentAddr ) == CFposcachev.end()) {
//   	      std::cout << "CHECKING CHANNEL  = " << linkiter->channel()   << std::endl;
// 	      std::cout << "\tTrackID  = " << linkiter->SimTrackId()  << "\tCFPos = " << currentCFPos <<"\ttofBin = " << tofBin << std::endl;
// 	      simhit_collectionMap::const_iterator it = SimHitCollMap.find(theSimHitCollID);
// 	      if (it!= SimHitCollMap.end()) {
// 		PSimHit theSimHit = it->second[currentCFPos];
// 		std::cout << "\tLocal Pos = " << theSimHit.localPosition()
// 			  << "\tProcess = " << theSimHit.processType() << std::endl;
// 	      }
	      CFposcachev.push_back(currentAddr);
	      simhitCFPos->push_back(currentAddr);
	    }
	  }
	}
      }    
    }
    else {
      edm::LogError("TrackerHitAssociator")<<"no cluster reference attached";
    }
  }
}

std::vector<SimHitIdpr>  TrackerHitAssociator::associateMatchedRecHit(const SiStripMatchedRecHit2D* matchedrechit, std::vector<simhitAddr>* simhitCFPos) const
{
  std::vector<SimHitIdpr> matched_mono;
  std::vector<SimHitIdpr> matched_st;

  const SiStripRecHit2D mono = matchedrechit->monoHit();
  const SiStripRecHit2D st = matchedrechit->stereoHit();
  //associate the two simple hits separately
  associateSiStripRecHit(&mono, matched_mono, simhitCFPos);
  associateSiStripRecHit(&st, matched_st, simhitCFPos);
  
  //save in a vector all the simtrack-id's that are common to mono and stereo hits
  std::vector<SimHitIdpr> simtrackid;
  if(!(matched_mono.empty() || matched_st.empty())){
    std::vector<SimHitIdpr> idcachev;
    for(auto const& mhit: matched_mono){
      //save only once the ID
      if(find(idcachev.begin(), idcachev.end(), mhit) == idcachev.end()) {
	idcachev.push_back(mhit);
	//save if the stereoID matched the monoID
	if(find(matched_st.begin(), matched_st.end(), mhit) != matched_st.end()) {
	  simtrackid.push_back(mhit);
	}
      }
    }
  }
  
  return simtrackid;
}


std::vector<SimHitIdpr>  TrackerHitAssociator::associateProjectedRecHit(const ProjectedSiStripRecHit2D * projectedrechit,
									std::vector<simhitAddr>* simhitCFPos) const
{
  //projectedRecHit is a "matched" rechit with only one component

  std::vector<SimHitIdpr> matched_mono;
 
  const SiStripRecHit2D mono = projectedrechit->originalHit();
  associateSiStripRecHit(&mono, matched_mono, simhitCFPos);
  return matched_mono;
}

void  TrackerHitAssociator::associatePixelRecHit(const SiPixelRecHit * pixelrechit,
						 std::vector<SimHitIdpr> & simtrackid,
						 std::vector<simhitAddr>* simhitCFPos) const
{
  //
  // Pixel associator
  //
  DetId detid=  pixelrechit->geographicalId();
  uint32_t detID = detid.rawId();

  edm::DetSetVector<PixelDigiSimLink>::const_iterator isearch = pixeldigisimlink->find(detID); 
  if(isearch != pixeldigisimlink->end()) {  //if it is not empty
    edm::DetSet<PixelDigiSimLink> link_detset = (*isearch);
    SiPixelRecHit::ClusterRef const& cluster = pixelrechit->cluster();
    
    //check the reference is valid
    
    if(!(cluster.isNull())){//if the cluster is valid
      
      int minPixelRow = (*cluster).minPixelRow();
      int maxPixelRow = (*cluster).maxPixelRow();
      int minPixelCol = (*cluster).minPixelCol();
      int maxPixelCol = (*cluster).maxPixelCol();    
      //std::cout << "    Cluster minRow " << minPixelRow << " maxRow " << maxPixelRow << std::endl;
      //std::cout << "    Cluster minCol " << minPixelCol << " maxCol " << maxPixelCol << std::endl;
      edm::DetSet<PixelDigiSimLink>::const_iterator linkiter = link_detset.data.begin(), linkEnd = link_detset.data.end();
      int dsl = 0;
      std::vector<SimHitIdpr> idcachev;
      std::vector<simhitAddr> CFposcachev;
      for( ; linkiter != linkEnd; ++linkiter) {
	++dsl;
	std::pair<int,int> pixel_coord = PixelDigi::channelToPixel(linkiter->channel());
	//std::cout << "    " << dsl << ") Digi link: row " << pixel_coord.first << " col " << pixel_coord.second << std::endl;      
	if(  pixel_coord.first  <= maxPixelRow && 
	     pixel_coord.first  >= minPixelRow &&
	     pixel_coord.second <= maxPixelCol &&
	     pixel_coord.second >= minPixelCol ) {
	  //std::cout << "      !-> trackid   " << linkiter->SimTrackId() << endl;
	  //std::cout << "          fraction  " << linkiter->fraction()   << endl;
	  SimHitIdpr currentId(linkiter->SimTrackId(), linkiter->eventId());
	  if(find(idcachev.begin(),idcachev.end(),currentId) == idcachev.end()){
	    simtrackid.push_back(currentId);
	    idcachev.push_back(currentId);
	  }

	  if (simhitCFPos != 0) {
	  //create a vector that contains all the position (in the MixCollection) of the SimHits that contributed to the RecHit
	  //write position only once
	    unsigned int currentCFPos = linkiter->CFposition();
	    unsigned int tofBin = linkiter->TofBin();
	    simHitCollectionID theSimHitCollID = std::make_pair(detid.subdetId(), tofBin);
	    simhitAddr currentAddr = std::make_pair(theSimHitCollID, currentCFPos);

	    if(find(CFposcachev.begin(), CFposcachev.end(), currentAddr) == CFposcachev.end()) {
	      CFposcachev.push_back(currentAddr);
	      simhitCFPos->push_back(currentAddr);
	    }
	  }

	} 
      }
    }
    else{      
      edm::LogError("TrackerHitAssociator")<<"no Pixel cluster reference attached";
      
    }
  }
}

std::vector<PSimHit> TrackerHitAssociator::associateMultiRecHit(const SiTrackerMultiRecHit * multirechit) const{
  std::vector<const TrackingRecHit*> componenthits = multirechit->recHits();
  //        std::vector<PSimHit> assimhits;
  int size=multirechit->weights().size(), idmostprobable=0;
  
  for (int i=0; i<size; ++i){
    if(multirechit->weight(i)>multirechit->weight(idmostprobable)) idmostprobable=i;
  }
  
  return associateHit(*componenthits[idmostprobable]);
}

std::vector<SimHitIdpr> TrackerHitAssociator::associateMultiRecHitId(const SiTrackerMultiRecHit * multirechit, std::vector<simhitAddr>* simhitCFPos) const{
  std::vector<const TrackingRecHit*> componenthits = multirechit->recHits();
  int size=multirechit->weights().size(), idmostprobable=0;
  
  for (int i=0; i<size; ++i){
    if(multirechit->weight(i)>multirechit->weight(idmostprobable)) idmostprobable=i;
  }
  
  std::vector< SimHitIdpr > simhitid;
  associateHitId(*componenthits[idmostprobable], simhitid, simhitCFPos);
  return simhitid;
}

 // fastsim
std::vector<SimHitIdpr>  TrackerHitAssociator::associateFastRecHit(const FastTrackerRecHit * rechit) const
{
  vector<SimHitIdpr> simtrackid;
  simtrackid.clear();
  for(size_t index =0, indexEnd = rechit->nSimTrackIds();index<indexEnd;++index){
      SimHitIdpr currentId(rechit->simTrackId(index), EncodedEventId(rechit->simTrackEventId(index)));
      simtrackid.push_back(currentId);
  }
  return simtrackid;
}
