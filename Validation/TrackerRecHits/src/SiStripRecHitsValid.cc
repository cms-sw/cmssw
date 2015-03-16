#include "Validation/TrackerRecHits/interface/SiStripRecHitsValid.h"
#include "SimTracker/TrackerHitAssociation/interface/TrackerHitAssociator.h" 

//needed for the geometry: 
#include "CalibTracker/Records/interface/SiStripDetCablingRcd.h"
#include "CalibFormats/SiStripObjects/interface/SiStripDetCabling.h"
#include "DataFormats/DetId/interface/DetId.h" 
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h" 
#include "DataFormats/SiStripDetId/interface/SiStripSubStructure.h"
#include "DQM/SiStripCommon/interface/SiStripFolderOrganizer.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/GeometryVector/interface/LocalPoint.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DQM/SiStripCommon/interface/SiStripHistoId.h"


//--- for RecHit
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h" 
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h" 
#include "DataFormats/Common/interface/OwnVector.h" 
#include "DQMServices/Core/interface/DQMStore.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

using namespace std;
using namespace edm;

namespace helper { 
    struct GetDetId { 
        template<typename X> 
        DetId operator()(const X &x) { return DetId(x.detId()); }
    };

    template<typename T>
    std::pair<typename T::DetSet::const_iterator, typename T::DetSet::const_iterator> 
    getRange(const T &detset, const DetId &id) {
        typedef std::pair<typename T::DetSet::const_iterator, typename T::DetSet::const_iterator> return_type;
        typename T::const_iterator match = detset.find(id);
        if (match == detset.end()) return return_type();
        typename T::DetSet hits = *match;
        return return_type(hits.begin(), hits.end());
    } 
}



//Constructor
SiStripRecHitsValid::SiStripRecHitsValid(const ParameterSet& ps) :
  conf_(ps),
  m_cacheID_(0)
  // matchedRecHits_( ps.getParameter<edm::InputTag>("matchedRecHits") ),
  // rphiRecHits_( ps.getParameter<edm::InputTag>("rphiRecHits") ),
  // stereoRecHits_( ps.getParameter<edm::InputTag>("stereoRecHits") ) 
{
  matchedRecHitsToken_ = consumes<SiStripMatchedRecHit2DCollection>( ps.getParameter<edm::InputTag>("matchedRecHits") );
    
  rphiRecHitsToken_ = consumes<SiStripRecHit2DCollection>( ps.getParameter<edm::InputTag>("rphiRecHits") );
    
  stereoRecHitsToken_ = consumes<SiStripRecHit2DCollection>( ps.getParameter<edm::InputTag>("stereoRecHits") ); 

  topFolderName_ = conf_.getParameter<std::string>("TopFolderName");

  SubDetList_ = conf_.getParameter<std::vector<std::string> >("SubDetList");

  edm::ParameterSet ParametersNumTotrphi =  conf_.getParameter<edm::ParameterSet>("TH1NumTotrphi");
  switchNumTotrphi = ParametersNumTotrphi.getParameter<bool>("switchon");

  edm::ParameterSet ParametersNumTotStereo =  conf_.getParameter<edm::ParameterSet>("TH1NumTotStereo");
  switchNumTotStereo = ParametersNumTotStereo.getParameter<bool>("switchon");

  edm::ParameterSet ParametersNumTotMatched =  conf_.getParameter<edm::ParameterSet>("TH1NumTotMatched");
  switchNumTotMatched = ParametersNumTotMatched.getParameter<bool>("switchon");

  edm::ParameterSet ParametersNumrphi =  conf_.getParameter<edm::ParameterSet>("TH1Numrphi");
  switchNumrphi = ParametersNumrphi.getParameter<bool>("switchon");

  edm::ParameterSet ParametersBunchrphi =  conf_.getParameter<edm::ParameterSet>("TH1Bunchrphi");
  switchBunchrphi = ParametersBunchrphi.getParameter<bool>("switchon");

  edm::ParameterSet ParametersEventrphi =  conf_.getParameter<edm::ParameterSet>("TH1Eventrphi");
  switchEventrphi = ParametersEventrphi.getParameter<bool>("switchon");

  edm::ParameterSet ParametersNumStereo =  conf_.getParameter<edm::ParameterSet>("TH1NumStereo");
  switchNumStereo = ParametersNumStereo.getParameter<bool>("switchon");

  edm::ParameterSet ParametersBunchStereo =  conf_.getParameter<edm::ParameterSet>("TH1BunchStereo");
  switchBunchStereo = ParametersBunchStereo.getParameter<bool>("switchon");

  edm::ParameterSet ParametersEventStereo =  conf_.getParameter<edm::ParameterSet>("TH1EventStereo");
  switchEventStereo = ParametersEventStereo.getParameter<bool>("switchon");

  edm::ParameterSet ParametersNumMatched =  conf_.getParameter<edm::ParameterSet>("TH1NumMatched");
  switchNumMatched = ParametersNumMatched.getParameter<bool>("switchon");

  edm::ParameterSet ParametersBunchMatched =  conf_.getParameter<edm::ParameterSet>("TH1BunchMatched");
  switchBunchMatched = ParametersBunchMatched.getParameter<bool>("switchon");

  edm::ParameterSet ParametersEventMatched =  conf_.getParameter<edm::ParameterSet>("TH1EventMatched");
  switchEventMatched = ParametersEventMatched.getParameter<bool>("switchon");

  edm::ParameterSet ParametersWclusrphi =  conf_.getParameter<edm::ParameterSet>("TH1Wclusrphi");
  switchWclusrphi = ParametersWclusrphi.getParameter<bool>("switchon");

  edm::ParameterSet ParametersAdcrphi =  conf_.getParameter<edm::ParameterSet>("TH1Adcrphi");
  switchAdcrphi = ParametersAdcrphi.getParameter<bool>("switchon");

  edm::ParameterSet ParametersPosxrphi =  conf_.getParameter<edm::ParameterSet>("TH1Posxrphi");
  switchPosxrphi = ParametersPosxrphi.getParameter<bool>("switchon");

  edm::ParameterSet ParametersResolxrphi =  conf_.getParameter<edm::ParameterSet>("TH1Resolxrphi");
  switchResolxrphi = ParametersResolxrphi.getParameter<bool>("switchon");

  edm::ParameterSet ParametersResrphi =  conf_.getParameter<edm::ParameterSet>("TH1Resrphi");
  switchResrphi = ParametersResrphi.getParameter<bool>("switchon");

  edm::ParameterSet ParametersPullLFrphi =  conf_.getParameter<edm::ParameterSet>("TH1PullLFrphi");
  switchPullLFrphi = ParametersPullLFrphi.getParameter<bool>("switchon");

  edm::ParameterSet ParametersPullMFrphi =  conf_.getParameter<edm::ParameterSet>("TH1PullMFrphi");
  switchPullMFrphi = ParametersPullMFrphi.getParameter<bool>("switchon");

  edm::ParameterSet ParametersChi2rphi =  conf_.getParameter<edm::ParameterSet>("TH1Chi2rphi");
  switchChi2rphi = ParametersChi2rphi.getParameter<bool>("switchon");

  edm::ParameterSet ParametersNsimHitrphi =  conf_.getParameter<edm::ParameterSet>("TH1NsimHitrphi");
  switchNsimHitrphi = ParametersNsimHitrphi.getParameter<bool>("switchon");

  edm::ParameterSet ParametersWclusStereo =  conf_.getParameter<edm::ParameterSet>("TH1WclusStereo");
  switchWclusStereo = ParametersWclusStereo.getParameter<bool>("switchon");

  edm::ParameterSet ParametersAdcStereo =  conf_.getParameter<edm::ParameterSet>("TH1AdcStereo");
  switchAdcStereo = ParametersAdcStereo.getParameter<bool>("switchon");

  edm::ParameterSet ParametersPosxStereo =  conf_.getParameter<edm::ParameterSet>("TH1PosxStereo");
  switchPosxStereo = ParametersPosxStereo.getParameter<bool>("switchon");

  edm::ParameterSet ParametersResolxStereo =  conf_.getParameter<edm::ParameterSet>("TH1ResolxStereo");
  switchResolxStereo = ParametersResolxStereo.getParameter<bool>("switchon");

  edm::ParameterSet ParametersResStereo =  conf_.getParameter<edm::ParameterSet>("TH1ResStereo");
  switchResStereo = ParametersResStereo.getParameter<bool>("switchon");

  edm::ParameterSet ParametersPullLFStereo =  conf_.getParameter<edm::ParameterSet>("TH1PullLFStereo");
  switchPullLFStereo = ParametersPullLFStereo.getParameter<bool>("switchon");

  edm::ParameterSet ParametersPullMFStereo =  conf_.getParameter<edm::ParameterSet>("TH1PullMFStereo");
  switchPullMFStereo = ParametersPullMFStereo.getParameter<bool>("switchon");

  edm::ParameterSet ParametersChi2Stereo =  conf_.getParameter<edm::ParameterSet>("TH1Chi2Stereo");
  switchChi2Stereo = ParametersChi2Stereo.getParameter<bool>("switchon");

  edm::ParameterSet ParametersNsimHitStereo =  conf_.getParameter<edm::ParameterSet>("TH1NsimHitStereo");
  switchNsimHitStereo = ParametersNsimHitStereo.getParameter<bool>("switchon");

  edm::ParameterSet ParametersPosxMatched =  conf_.getParameter<edm::ParameterSet>("TH1PosxMatched");
  switchPosxMatched = ParametersPosxMatched.getParameter<bool>("switchon");

  edm::ParameterSet ParametersPosyMatched =  conf_.getParameter<edm::ParameterSet>("TH1PosyMatched");
  switchPosyMatched = ParametersPosyMatched.getParameter<bool>("switchon");

  edm::ParameterSet ParametersResolxMatched =  conf_.getParameter<edm::ParameterSet>("TH1ResolxMatched");
  switchResolxMatched = ParametersResolxMatched.getParameter<bool>("switchon");

  edm::ParameterSet ParametersResolyMatched =  conf_.getParameter<edm::ParameterSet>("TH1ResolyMatched");
  switchResolyMatched = ParametersResolyMatched.getParameter<bool>("switchon");

  edm::ParameterSet ParametersResxMatched =  conf_.getParameter<edm::ParameterSet>("TH1ResxMatched");
  switchResxMatched = ParametersResxMatched.getParameter<bool>("switchon");

  edm::ParameterSet ParametersResyMatched =  conf_.getParameter<edm::ParameterSet>("TH1ResyMatched");
  switchResyMatched = ParametersResyMatched.getParameter<bool>("switchon");

  edm::ParameterSet ParametersChi2Matched =  conf_.getParameter<edm::ParameterSet>("TH1Chi2Matched");
  switchChi2Matched = ParametersChi2Matched.getParameter<bool>("switchon");

  edm::ParameterSet ParametersNsimHitMatched =  conf_.getParameter<edm::ParameterSet>("TH1NsimHitMatched");
  switchNsimHitMatched = ParametersNsimHitMatched.getParameter<bool>("switchon");

}

SiStripRecHitsValid::~SiStripRecHitsValid(){
}

//--------------------------------------------------------------------------------------------
void SiStripRecHitsValid::bookHistograms(DQMStore::IBooker & ibooker,const edm::Run& run, const edm::EventSetup& es){

  unsigned long long cacheID = es.get<SiStripDetCablingRcd>().cacheIdentifier();
  if (m_cacheID_ != cacheID) {
    m_cacheID_ = cacheID;       
    edm::LogInfo("SiStripRecHitsValid") <<"SiStripRecHitsValid::beginRun: " 
					  << " Creating MEs for new Cabling ";     
    
    createMEs(ibooker,es);
  }
}

void SiStripRecHitsValid::beginJob(const edm::EventSetup& es){
}

void SiStripRecHitsValid::analyze(const edm::Event& e, const edm::EventSetup& es) {

  LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();  
  //cout  << " Run = " << e.id().run() << " Event = " << e.id().event() << endl;  
  
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();
    
  // Step A: Get Inputs 
  edm::Handle<SiStripMatchedRecHit2DCollection> rechitsmatched;
  edm::Handle<SiStripRecHit2DCollection> rechitsrphi;
  edm::Handle<SiStripRecHit2DCollection> rechitsstereo;
  e.getByToken(matchedRecHitsToken_, rechitsmatched);
  e.getByToken(rphiRecHitsToken_, rechitsrphi);
  e.getByToken(stereoRecHitsToken_, rechitsstereo);

  int numrechitrphi   =0;
  int numrechitstereo    =0;
  int numrechitmatched=0;

  //Variables in order to count total num of rechitrphi,rechitstereo, rechitmatched in subdetectors
  std::map<std::string, std::vector<int> > totnumrechitrphi;
  std::map<std::string, std::vector<int> > totnumrechitstereo;
  std::map<std::string, std::vector<int> > totnumrechitmatched;
  totnumrechitrphi.clear();
  totnumrechitstereo.clear();
  totnumrechitmatched.clear();
  int totrechitrphi =0;
  int totrechitstereo =0;
  int totrechitmatched =0;
   
  TrackerHitAssociator associate(e, conf_);
  
  edm::ESHandle<TrackerGeometry> pDD;
  es.get<TrackerDigiGeometryRecord> ().get (pDD);
  const TrackerGeometry &tracker(*pDD);
  // FIXME: this using of vector<DetId> is suboptimal, but I don't want to re-write the full class now
  std::vector<DetId> IDs; 
  IDs.reserve(rechitsrphi->size() + rechitsmatched->size() + rechitsstereo->size());
  std::transform(rechitsrphi->begin(), rechitsrphi->end(), std::back_inserter(IDs), helper::GetDetId() );
  std::transform(rechitsstereo->begin(), rechitsstereo->end(), std::back_inserter(IDs), helper::GetDetId() );
  std::transform(rechitsmatched->begin(), rechitsmatched->end(), std::back_inserter(IDs), helper::GetDetId() );
  // loop over DetIds
  //  for(TrackerGeometry::DetContainer::const_iterator it = pDD->dets().begin(); it != pDD->dets().end(); it++){
  for(std::vector<DetId>::const_iterator it = IDs.begin(); it != IDs.end(); ++it ){//loop on detector with hits
    uint32_t myid=((*it).rawId());       
    DetId detid = ((*it));
    
    rechitrphi.clear();
    rechitstereo.clear();
    rechitmatched.clear();
    
    numrechitrphi =0;
    //loop over rechits-rphi in the same subdetector
    std::pair<SiStripRecHit2DCollection::DetSet::const_iterator,SiStripRecHit2DCollection::DetSet::const_iterator> rechitrphiRange = helper::getRange(*rechitsrphi, detid);
    SiStripRecHit2DCollection::DetSet::const_iterator rechitrphiRangeIteratorBegin = rechitrphiRange.first;
    SiStripRecHit2DCollection::DetSet::const_iterator rechitrphiRangeIteratorEnd   = rechitrphiRange.second;
    SiStripRecHit2DCollection::DetSet::const_iterator iterrphi=rechitrphiRangeIteratorBegin;
    
    numrechitrphi = rechitrphiRangeIteratorEnd - rechitrphiRangeIteratorBegin;   
         
    if(numrechitrphi > 0 ){
      totrechitrphi+=numrechitrphi;
      for(iterrphi=rechitrphiRangeIteratorBegin; iterrphi!=rechitrphiRangeIteratorEnd;++iterrphi){
	
	const GeomDetUnit *  det = tracker.idToDetUnit(detid);
	const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)(det);
	const StripTopology &topol=(StripTopology&)stripdet->topology();
	SiStripRecHit2D const rechit=*iterrphi;
	//analyze RecHits 
	rechitanalysis(rechit,topol,associate);
	rechitrphi.push_back(rechitpro);
      }
    }
    
    //loop over rechits-stereo in the same subdetector
    numrechitstereo=0;
    std::pair<SiStripRecHit2DCollection::DetSet::const_iterator,SiStripRecHit2DCollection::DetSet::const_iterator> rechitstereoRange = helper::getRange(*rechitsstereo, detid);
    SiStripRecHit2DCollection::DetSet::const_iterator rechitstereoRangeIteratorBegin = rechitstereoRange.first;
    SiStripRecHit2DCollection::DetSet::const_iterator rechitstereoRangeIteratorEnd   = rechitstereoRange.second;
    SiStripRecHit2DCollection::DetSet::const_iterator iterstereo=rechitstereoRangeIteratorBegin;
    numrechitstereo = rechitstereoRangeIteratorEnd - rechitstereoRangeIteratorBegin;   
    if(numrechitstereo > 0){
      totrechitstereo+=numrechitstereo;
      for(iterstereo=rechitstereoRangeIteratorBegin; iterstereo!=rechitstereoRangeIteratorEnd;++iterstereo){
	const GeomDetUnit *  det = tracker.idToDetUnit(detid);
	const StripGeomDetUnit * stripdet=(const StripGeomDetUnit*)(det);
	const StripTopology &topol=(StripTopology&)stripdet->topology();
	SiStripRecHit2D const rechit=*iterstereo;
	//analyze RecHits 
	rechitanalysis(rechit,topol,associate);
	rechitstereo.push_back(rechitpro);
      }
    }
    
    //now matched hits
    //loop over rechits-matched in the same subdetector
    numrechitmatched=0;
    std::pair<SiStripMatchedRecHit2DCollection::DetSet::const_iterator,SiStripMatchedRecHit2DCollection::DetSet::const_iterator> rechitmatchedRange = helper::getRange(*rechitsmatched, detid);
    SiStripMatchedRecHit2DCollection::DetSet::const_iterator rechitmatchedRangeIteratorBegin = rechitmatchedRange.first;
    SiStripMatchedRecHit2DCollection::DetSet::const_iterator rechitmatchedRangeIteratorEnd   = rechitmatchedRange.second;
    SiStripMatchedRecHit2DCollection::DetSet::const_iterator itermatched=rechitmatchedRangeIteratorBegin;
    numrechitmatched = rechitmatchedRangeIteratorEnd - rechitmatchedRangeIteratorBegin;   
    if(numrechitmatched > 0){
      totrechitmatched +=numrechitmatched;

      for(itermatched=rechitmatchedRangeIteratorBegin; itermatched!=rechitmatchedRangeIteratorEnd;++itermatched){
	SiStripMatchedRecHit2D const rechit=*itermatched;
	const GluedGeomDet* gluedDet = (const GluedGeomDet*)tracker.idToDet(rechit.geographicalId());
	//analyze RecHits 
	rechitanalysis_matched(rechit, gluedDet, associate);
	rechitmatched.push_back(rechitpro);
      }
    }
    
    SiStripHistoId hidmanager;
    std::string label = hidmanager.getSubdetid(myid,tTopo,true);
    std::map<std::string, LayerMEs>::iterator iLayerME  = LayerMEsMap.find(label);
    std::map<std::string, StereoAndMatchedMEs>::iterator iStereoAndMatchedME  = StereoAndMatchedMEsMap.find(label);
    //std::cout << "label " << label << endl;
    SiStripFolderOrganizer fold_organ;
    std::pair<std::string,int32_t> det_lay_pair = fold_organ.GetSubDetAndLayer(myid,tTopo,true);
  
    if(numrechitrphi>0 || numrechitstereo>0 || numrechitmatched){

      totnumrechitrphi[det_lay_pair.first].push_back(numrechitrphi);
      totnumrechitstereo[det_lay_pair.first].push_back(numrechitstereo);
      totnumrechitmatched[det_lay_pair.first].push_back(numrechitmatched);

      for (std::vector<RecHitProperties>::iterator irh=rechitrphi.begin(); irh!=rechitrphi.end(); ++irh) {
	if(iLayerME != LayerMEsMap.end()){
	  fillME(iLayerME->second.meWclusrphi,(*irh).clusiz);
	  fillME(iLayerME->second.meAdcrphi,(*irh).cluchg);
	  fillME(iLayerME->second.mePosxrphi,(*irh).x);
	  fillME(iLayerME->second.meResolxrphi,(*irh).resolxx);
	  fillME(iLayerME->second.meNsimHitrphi,(*irh).NsimHit);
	  if ((*irh).NsimHit > 0) {
	    std::map<std::string, SubDetMEs>::iterator iSubDetME = SubDetMEsMap.find(det_lay_pair.first);
	    fillME(iSubDetME->second.meBunchrphi, (*irh).bunch);
	    if ((*irh).bunch == 0) fillME(iSubDetME->second.meEventrphi, (*irh).event);
	  }
	  if ( (*irh).resx != -999999. || (*irh).pullMF != -999999. || (*irh).chi2 != -999999. ){
	    fillME(iLayerME->second.meResrphi,(*irh).resx);
	    fillME(iLayerME->second.mePullLFrphi,(*irh).resx/sqrt((*irh).resolxx));
	    fillME(iLayerME->second.mePullMFrphi,(*irh).pullMF);
	    fillME(iLayerME->second.meChi2rphi,(*irh).chi2);
	  }
	}
      }
      
      
      for (std::vector<RecHitProperties>::iterator irh=rechitstereo.begin(); irh!=rechitstereo.end(); ++irh) {
	if(iStereoAndMatchedME != StereoAndMatchedMEsMap.end()){
	  fillME(iStereoAndMatchedME->second.meWclusStereo,(*irh).clusiz);
	  fillME(iStereoAndMatchedME->second.meAdcStereo,(*irh).cluchg);
	  fillME(iStereoAndMatchedME->second.mePosxStereo,(*irh).x);
	  fillME(iStereoAndMatchedME->second.meResolxStereo,sqrt((*irh).resolxx));
	  fillME(iStereoAndMatchedME->second.meNsimHitStereo,(*irh).NsimHit);
	  if ((*irh).NsimHit > 0) {
	    std::map<std::string, SubDetMEs>::iterator iSubDetME = SubDetMEsMap.find(det_lay_pair.first);
	    fillME(iSubDetME->second.meBunchStereo, (*irh).bunch);
	    if ((*irh).bunch == 0) fillME(iSubDetME->second.meEventStereo, (*irh).event);
	  }
	  if ( (*irh).resx != -999999. || (*irh).pullMF != -999999. || (*irh).chi2 != -999999. ){
	    fillME(iStereoAndMatchedME->second.meResStereo,(*irh).resx);
	    fillME(iStereoAndMatchedME->second.mePullLFStereo,(*irh).resx/sqrt((*irh).resolxx));
	    fillME(iStereoAndMatchedME->second.mePullMFStereo,(*irh).pullMF);
	    fillME(iStereoAndMatchedME->second.meChi2Stereo,(*irh).chi2);
	  }
	}
      }

      for (std::vector<RecHitProperties>::iterator irh=rechitmatched.begin(); irh!=rechitmatched.end(); ++irh) {
	if(iStereoAndMatchedME != StereoAndMatchedMEsMap.end()){
	  fillME(iStereoAndMatchedME->second.mePosxMatched,(*irh).x);
	  fillME(iStereoAndMatchedME->second.mePosyMatched,(*irh).y);
	  fillME(iStereoAndMatchedME->second.meResolxMatched,sqrt((*irh).resolxx));
	  fillME(iStereoAndMatchedME->second.meResolyMatched,sqrt((*irh).resolyy));
	  fillME(iStereoAndMatchedME->second.meNsimHitMatched,(*irh).NsimHit);
	  if ((*irh).NsimHit > 0) {
	    std::map<std::string, SubDetMEs>::iterator iSubDetME = SubDetMEsMap.find(det_lay_pair.first);
	    fillME(iSubDetME->second.meBunchMatched, (*irh).bunch);
	    if ((*irh).bunch == 0) fillME(iSubDetME->second.meEventMatched, (*irh).event);
	  }
	  if ( (*irh).resx != -999999. || (*irh).resy != -999999. || (*irh).chi2 != -999999. ){
	    fillME(iStereoAndMatchedME->second.meResxMatched,(*irh).resx);
	    fillME(iStereoAndMatchedME->second.meResyMatched,(*irh).resy);
	    fillME(iStereoAndMatchedME->second.meChi2Matched,(*irh).chi2);
	  }
	}
      }

    }
    
  }//End of loop over detectors
  

  //now fill the cumulative histograms of the hits
  for (std::vector<std::string>::iterator iSubdet  = SubDetList_.begin(); iSubdet != SubDetList_.end(); iSubdet++){
    std::map<std::string, SubDetMEs>::iterator iSubDetME  = SubDetMEsMap.find((*iSubdet));
    fillME(iSubDetME->second.meNumrphi,std::accumulate(totnumrechitrphi[(*iSubdet)].rbegin(), totnumrechitrphi[(*iSubdet)].rend(), 0));
    fillME(iSubDetME->second.meNumStereo,std::accumulate(totnumrechitstereo[(*iSubdet)].rbegin(), totnumrechitstereo[(*iSubdet)].rend(), 0));
    fillME(iSubDetME->second.meNumMatched,std::accumulate(totnumrechitmatched[(*iSubdet)].rbegin(), totnumrechitmatched[(*iSubdet)].rend(), 0));
  }

  fillME(totalMEs.meNumTotrphi,totrechitrphi);
  fillME(totalMEs.meNumTotStereo,totrechitstereo);
  fillME(totalMEs.meNumTotMatched,totrechitmatched);
 
}

  
//needed by to do the residual for matched hits
std::pair<LocalPoint,LocalVector> SiStripRecHitsValid::projectHit( const PSimHit& hit, const StripGeomDetUnit* stripDet,
								   const BoundPlane& plane) 
{
  //  const StripGeomDetUnit* stripDet = dynamic_cast<const StripGeomDetUnit*>(hit.det());
  //if (stripDet == 0) throw MeasurementDetException("HitMatcher hit is not on StripGeomDetUnit");
  
  const StripTopology& topol = stripDet->specificTopology();
  GlobalPoint globalpos= stripDet->surface().toGlobal(hit.localPosition());
  LocalPoint localHit = plane.toLocal(globalpos);
  //track direction
  LocalVector locdir=hit.localDirection();
  //rotate track in new frame
  
  GlobalVector globaldir= stripDet->surface().toGlobal(locdir);
  LocalVector dir=plane.toLocal(globaldir);
  double scale = -localHit.z() / dir.z();
  
  LocalPoint projectedPos = localHit + scale*dir;
  
  //  std::cout << "projectedPos " << projectedPos << std::endl;
  
  double selfAngle = topol.stripAngle( topol.strip( hit.localPosition()));
  
  LocalVector stripDir( sin(selfAngle), cos(selfAngle), 0); // vector along strip in hit frame
  
  LocalVector localStripDir( plane.toLocal(stripDet->surface().toGlobal( stripDir)));
  
  return std::pair<LocalPoint,LocalVector>( projectedPos, localStripDir);
}
//--------------------------------------------------------------------------------------------
void SiStripRecHitsValid::rechitanalysis(SiStripRecHit2D const rechit,const StripTopology &topol,TrackerHitAssociator& associate){
  
  rechitpro.x = -999999.; rechitpro.y = -999999.; rechitpro.z = -999999.; rechitpro.resolxx = -999999.; rechitpro.resolxy = -999999.; 
  rechitpro.resolyy = -999999.; rechitpro.resx = -999999.; rechitpro.resy = -999999.;rechitpro.pullMF = -999999.; 
  rechitpro.clusiz = -999999.; rechitpro.cluchg = -999999.; rechitpro.chi2 = -999999.; rechitpro.NsimHit = -999999.;
  rechitpro.bunch = -999999.; rechitpro.event = -999999.;

  LocalPoint position=rechit.localPosition();
  LocalError error=rechit.localPositionError();
  MeasurementPoint Mposition;
  MeasurementError Merror;
  Mposition = topol.measurementPosition(position);
  Merror = topol.measurementError(position,error);
  SiStripRecHit2D::ClusterRef clust=rechit.cluster();
  int clusiz=0;
  int totcharge=0;
  clusiz = clust->amplitudes().size();
  const auto & amplitudes=clust->amplitudes();
  for(size_t ia=0; ia<amplitudes.size();ia++){
    totcharge+=amplitudes[ia];
  }
  rechitpro.x = position.x();
  rechitpro.y = position.y();
  rechitpro.z = position.z();
  rechitpro.resolxx = error.xx();
  rechitpro.resolxy = error.xy();
  rechitpro.resolyy = error.yy();
  rechitpro.clusiz = clusiz;
  rechitpro.cluchg = totcharge;


  matched.clear();
  matched = associate.associateHit(rechit);
  rechitpro.NsimHit = matched.size();

  double mindist = 999999;
  double dist = 999999;
  PSimHit closest;
  
  if(!matched.empty()){

    for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
      dist = fabs(rechitpro.x - (*m).localPosition().x());
      if(dist<mindist){
	mindist = dist;
	closest = (*m);
      }
    }  
    rechitpro.bunch = closest.eventId().bunchCrossing();
    rechitpro.event = closest.eventId().event();
    rechitpro.resx = rechitpro.x - closest.localPosition().x();
    rechitpro.pullMF = (Mposition.x() - (topol.measurementPosition(closest.localPosition())).x())/sqrt(Merror.uu());
    
    //chi2test compare rechit errors with the simhit position ( using null matrix for the simhit). 
    //Can spot problems in the geometry better than a simple residual. (thanks to BorisM)
    AlgebraicVector  rhparameters(2);//= rechit.parameters();
    rhparameters[0] = position.x(); 
    rhparameters[1] = position.y();
    AlgebraicVector shparameters(2);
    shparameters[0] = closest.localPosition().x();
    shparameters[1] = closest.localPosition().y();
    AlgebraicVector r(rhparameters - shparameters);
    AlgebraicSymMatrix R(2);//  = rechit.parametersError();
    R[0][0] = error.xx();
    R[0][1] = error.xy();
    R[1][1] = error.yy();
    int ierr; 
    R.invert(ierr); // if (ierr != 0) throw exception;
    double est = R.similarity(r);
    // 	  std::cout << " ====== Chi2 test rphi hits ====== " << std::endl;
    // 	  std::cout << "RecHit param. = " << rhparameters << std::endl;
    // 	  std::cout << "RecHit errors = " << R << std::endl;
    //	  std::cout << "SimHit param. = " << shparameters << std::endl;
    //	  std::cout << " chi2  = " << est << std::endl;
    //	  std::cout << "DEBUG BORIS,filling chi2rphi[i],i: " << i << std::endl;
    rechitpro.chi2 = est;
  }

}


//--------------------------------------------------------------------------------------------
void SiStripRecHitsValid::rechitanalysis_matched(SiStripMatchedRecHit2D const rechit, const GluedGeomDet* gluedDet, TrackerHitAssociator& associate){
  
  rechitpro.x = -999999.; rechitpro.y = -999999.; rechitpro.z = -999999.; rechitpro.resolxx = -999999.; rechitpro.resolxy = -999999.; 
  rechitpro.resolyy = -999999.; rechitpro.resx = -999999.; rechitpro.resy = -999999.;rechitpro.pullMF = -999999.; 
  rechitpro.clusiz = -999999.; rechitpro.cluchg = -999999.; rechitpro.chi2 = -999999.; rechitpro.NsimHit = -999999.;
  rechitpro.bunch = -999999.; rechitpro.event = -999999.;

  LocalPoint position=rechit.localPosition();
  LocalError error=rechit.localPositionError();

  rechitpro.x = position.x();
  rechitpro.y = position.y();
  rechitpro.z = position.z();
  rechitpro.resolxx = error.xx();
  rechitpro.resolxy = error.xy();
  rechitpro.resolyy = error.yy();

  matched.clear();
  matched = associate.associateHit(rechit);
  rechitpro.NsimHit = matched.size();

  double mindist = 999999;
  double dist = 999999;
  double distx = 999999;
  double disty = 999999;
  PSimHit closest;
  std::pair<LocalPoint,LocalVector> closestPair;

  if(!matched.empty()){

    const StripGeomDetUnit* partnerstripdet =(StripGeomDetUnit*) gluedDet->stereoDet();
    std::pair<LocalPoint,LocalVector> hitPair;
    

    for(vector<PSimHit>::const_iterator m=matched.begin(); m<matched.end(); m++){
      SiStripDetId hitDetId(m->detUnitId());
      if (hitDetId.stereo()) {  // project from the stereo sensor
      //project simhit;
	hitPair= projectHit((*m),partnerstripdet,gluedDet->surface());
	distx = fabs(rechitpro.x - hitPair.first.x());
	disty = fabs(rechitpro.y - hitPair.first.y());
	dist = sqrt(distx*distx+disty*disty);
	// std::cout << " Simhit position x = " << hitPair.first.x() 
	//      << " y = " << hitPair.first.y() << " dist = " << dist << std::endl;
	if(dist<mindist){
	  mindist = dist;
	  closestPair = hitPair;
	  closest = (*m);
	}
      }
    }  
    rechitpro.bunch = closest.eventId().bunchCrossing();
    rechitpro.event = closest.eventId().event();
    rechitpro.resx = rechitpro.x - closestPair.first.x();
    rechitpro.resy = rechitpro.y - closestPair.first.y();
    //std::cout << " Closest position x = " << closestPair.first.x() 
    //      << " y = " << closestPair.first.y() << " dist = " << dist << std::endl;
  
    //chi2test compare rechit errors with the simhit position ( using null matrix for the simhit). 
    //Can spot problems in the geometry better than a simple residual. (thanks to BorisM)
    AlgebraicVector  rhparameters(2);//= rechit.parameters();
    rhparameters[0] = position.x(); 
    rhparameters[1] = position.y();
    LocalPoint sh = closestPair.first;
    AlgebraicVector shparameters(2);
    shparameters[0] = sh.x();
    shparameters[1] = sh.y();
    AlgebraicVector r(rhparameters - shparameters);
    AlgebraicSymMatrix R(2);//  = rechit.parametersError();
    R[0][0] = error.xx();
    R[0][1] = error.xy();
    R[1][1] = error.yy();
    int ierr; 
    R.invert(ierr); // if (ierr != 0) throw exception;
    double est = R.similarity(r);
    // 	  std::cout << " ====== Chi2 test rphi hits ====== " << std::endl;
    // 	  std::cout << "RecHit param. = " << rhparameters << std::endl;
    // 	  std::cout << "RecHit errors = " << R << std::endl;
    //	  std::cout << "SimHit param. = " << shparameters << std::endl;
    //	  std::cout << " chi2  = " << est << std::endl;
    //	  std::cout << "DEBUG BORIS,filling chi2rphi[i],i: " << i << std::endl;
    rechitpro.chi2 = est;
  }


}

//--------------------------------------------------------------------------------------------
void SiStripRecHitsValid::createMEs(DQMStore::IBooker & ibooker,const edm::EventSetup& es){

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<IdealGeometryRecord>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();
  
  // take from eventSetup the SiStripDetCabling object - here will use SiStripDetControl later on
  es.get<SiStripDetCablingRcd>().get(SiStripDetCabling_);
    
  // get list of active detectors from SiStripDetCabling 
  std::vector<uint32_t> activeDets;
  SiStripDetCabling_->addActiveDetectorsRawIds(activeDets);
    
  SiStripSubStructure substructure;

  SiStripFolderOrganizer folder_organizer;
  // folder_organizer.setSiStripFolderName(topFolderName_);
  std::string curfold = topFolderName_;
  folder_organizer.setSiStripFolderName(curfold);
  folder_organizer.setSiStripFolder();

  // std::cout << "curfold " << curfold << std::endl;

  createTotalMEs(ibooker);

  // loop over detectors and book MEs
  edm::LogInfo("SiStripTkRecHits|SiStripRecHitsValid")<<"nr. of activeDets:  "<<activeDets.size();
  for(std::vector<uint32_t>::iterator detid_iterator = activeDets.begin(); detid_iterator!=activeDets.end(); detid_iterator++){
    uint32_t detid = (*detid_iterator);
    // remove any eventual zero elements - there should be none, but just in case
    if(detid == 0) {
      activeDets.erase(detid_iterator);
      continue;
    }
    
    // Create Layer Level MEs
    std::pair<std::string,int32_t> det_layer_pair = folder_organizer.GetSubDetAndLayer(detid,tTopo,true);
    SiStripHistoId hidmanager;
    std::string label = hidmanager.getSubdetid(detid,tTopo,true);
    // std::cout << "label " << label << endl;
      
    std::map<std::string, LayerMEs>::iterator iLayerME  = LayerMEsMap.find(label);
    if(iLayerME==LayerMEsMap.end()) {
	
      // get detids for the layer
      // Keep in mind that when we are on the TID or TEC we deal with rings not wheel 
      int32_t lnumber = det_layer_pair.second;
      std::vector<uint32_t> layerDetIds;        
      if (det_layer_pair.first == "TIB") {
	substructure.getTIBDetectors(activeDets,layerDetIds,lnumber,0,0,0);
      } else if (det_layer_pair.first == "TOB") {
	substructure.getTOBDetectors(activeDets,layerDetIds,lnumber,0,0);
      } else if (det_layer_pair.first == "TID" && lnumber > 0) {
	substructure.getTIDDetectors(activeDets,layerDetIds,2,0,abs(lnumber),0);
      } else if (det_layer_pair.first == "TID" && lnumber < 0) {
	substructure.getTIDDetectors(activeDets,layerDetIds,1,0,abs(lnumber),0);
      } else if (det_layer_pair.first == "TEC" && lnumber > 0) {
	substructure.getTECDetectors(activeDets,layerDetIds,2,0,0,0,abs(lnumber),0);
      } else if (det_layer_pair.first == "TEC" && lnumber < 0) {
	substructure.getTECDetectors(activeDets,layerDetIds,1,0,0,0,abs(lnumber),0);
      }
      LayerDetMap[label] = layerDetIds;

      // book Layer MEs 
      folder_organizer.setLayerFolder(detid,tTopo,det_layer_pair.second,true);
      // std::stringstream ss;
      // folder_organizer.getLayerFolderName(ss, detid, tTopo, true); 
      // std::cout << "Folder Name " << ss.str().c_str() << std::endl;
      createLayerMEs(ibooker,label);
    }
    // book sub-detector plots 
    auto sdet_pair = folder_organizer.getSubDetFolderAndTag(detid, tTopo);
    // std::cout << "sdet_pair " << sdet_pair.first << " " << sdet_pair.second << std::endl;
    if (SubDetMEsMap.find(det_layer_pair.first) == SubDetMEsMap.end()){
      ibooker.setCurrentFolder(sdet_pair.first);
      createSubDetMEs(ibooker,det_layer_pair.first);        
    }
    //Create StereoAndMatchedMEs
    std::map<std::string, StereoAndMatchedMEs>::iterator iStereoAndMatchedME  = StereoAndMatchedMEsMap.find(label);
    if(iStereoAndMatchedME==StereoAndMatchedMEsMap.end()) {
	
      // get detids for the stereo and matched layer. We are going to need a bool for these layers
      bool isStereo = false;
      // Keep in mind that when we are on the TID or TEC we deal with rings not wheel 
      int32_t stereolnumber = det_layer_pair.second;
      std::vector<uint32_t> stereoandmatchedDetIds;        
      if ( (det_layer_pair.first == "TIB") &&  (tTopo->tibIsStereo(detid) == 1) ) {
	substructure.getTIBDetectors(activeDets,stereoandmatchedDetIds,stereolnumber,0,0,0);
	isStereo = true;
      } else if ( (det_layer_pair.first == "TOB") &&  (tTopo->tobIsStereo(detid) == 1) ) {
	substructure.getTOBDetectors(activeDets,stereoandmatchedDetIds,stereolnumber,0,0);
	isStereo = true;
      } else if ( (det_layer_pair.first == "TID") && (stereolnumber > 0) && (tTopo->tidIsStereo(detid) == 1) ) {
	substructure.getTIDDetectors(activeDets,stereoandmatchedDetIds,2,0,abs(stereolnumber),1);
	isStereo = true;
      } else if ( (det_layer_pair.first == "TID") && (stereolnumber < 0) && (tTopo->tidIsStereo(detid) == 1) ) {
	substructure.getTIDDetectors(activeDets,stereoandmatchedDetIds,1,0,abs(stereolnumber),1);
	isStereo = true;
      } else if ( (det_layer_pair.first == "TEC") && (stereolnumber > 0) && (tTopo->tecIsStereo(detid) == 1) ) {
	substructure.getTECDetectors(activeDets,stereoandmatchedDetIds,2,0,0,0,abs(stereolnumber),1);
	isStereo = true;
      } else if ( (det_layer_pair.first == "TEC") && (stereolnumber < 0) && (tTopo->tecIsStereo(detid) == 1) ) {
	substructure.getTECDetectors(activeDets,stereoandmatchedDetIds,1,0,0,0,abs(stereolnumber),1);
	isStereo = true;
      }
      StereoAndMatchedDetMap[label] = stereoandmatchedDetIds;

      if(isStereo){
	//book StereoAndMatched MEs 
	folder_organizer.setLayerFolder(detid,tTopo,det_layer_pair.second,true);
	// std::stringstream ss1;
	// folder_organizer.getLayerFolderName(ss1, detid, tTopo, true);  
	// std::cout << "Folder Name stereo " <<  ss1.str().c_str() << std::endl;
	//Create the Monitor Elements only when we have a stereo module
	createStereoAndMatchedMEs(ibooker,label);
      }
    }
 

  }//end of loop over detectors
}
//------------------------------------------------------------------------------------------
void SiStripRecHitsValid::createTotalMEs(DQMStore::IBooker & ibooker) 
{
  totalMEs.meNumTotrphi = 0;
  totalMEs.meNumTotStereo = 0;
  totalMEs.meNumTotMatched = 0;

  //NumTotrphi
  if(switchNumTotrphi) {
    totalMEs.meNumTotrphi = bookME1D(ibooker,"TH1NumTotrphi", "TH1NumTotrphi" ,"Num of RecHits rphi");
    totalMEs.meNumTotrphi->setAxisTitle("Total number of RecHits");
  }
  //NumTotStereo
  if(switchNumTotStereo) {
    totalMEs.meNumTotStereo = bookME1D(ibooker,"TH1NumTotStereo", "TH1NumTotStereo","Num of RecHits stereo");
    totalMEs.meNumTotStereo ->setAxisTitle("Total number of RecHits (stereo)");
  }
  //NumTotMatched
  if(switchNumTotMatched) {
    totalMEs.meNumTotMatched = bookME1D(ibooker,"TH1NumTotMatched","TH1NumTotMatched","Num of RecHits matched"); 
    totalMEs.meNumTotMatched->setAxisTitle("Total number of matched RecHits");
  }
       
}
//------------------------------------------------------------------------------------------
void SiStripRecHitsValid::createLayerMEs(DQMStore::IBooker & ibooker,std::string label) 
{
  SiStripHistoId hidmanager;
  LayerMEs layerMEs; 

  layerMEs.meWclusrphi = 0;
  layerMEs.meAdcrphi = 0;
  layerMEs.mePosxrphi = 0;
  layerMEs.meResolxrphi = 0;
  layerMEs.meResrphi = 0;
  layerMEs.mePullLFrphi = 0;
  layerMEs.mePullMFrphi = 0;
  layerMEs.meChi2rphi = 0;
  layerMEs.meNsimHitrphi = 0;

  //Wclusrphi
  if(switchWclusrphi) {
    layerMEs.meWclusrphi = bookME1D(ibooker,"TH1Wclusrphi", hidmanager.createHistoLayer("Wclus_rphi","layer",label,"").c_str() ,"Cluster Width - Number of strips that belong to the RecHit cluster"); 
    layerMEs.meWclusrphi->setAxisTitle(("Cluster Width [nr strips] in "+ label).c_str());
  }
  //Adcrphi
  if(switchAdcrphi) {
    layerMEs.meAdcrphi = bookME1D(ibooker,"TH1Adcrphi", hidmanager.createHistoLayer("Adc_rphi","layer",label,"").c_str() ,"RecHit Cluster Charge");
    layerMEs.meAdcrphi->setAxisTitle(("cluster charge [ADC] in " + label).c_str());
  }
  //Posxrphi
  if(switchPosxrphi) {
    layerMEs.mePosxrphi = bookME1D(ibooker,"TH1Posxrphi", hidmanager.createHistoLayer("Posx_rphi","layer",label,"").c_str() ,"RecHit x coord."); 
    layerMEs.mePosxrphi->setAxisTitle(("x RecHit coord. (local frame) in " + label).c_str());
  }
  //Resolxrphi
  if(switchResolxrphi) {
    layerMEs.meResolxrphi = bookME1D(ibooker,"TH1Resolxrphi", hidmanager.createHistoLayer("Resolx_rphi","layer",label,"").c_str() ,"RecHit resol(x) coord.");   //<resolor>~20micron  
    layerMEs.meResolxrphi->setAxisTitle(("resol(x) RecHit coord. (local frame) in " + label).c_str());
  }
  //Resrphi
  if(switchResrphi) {
    layerMEs.meResrphi = bookME1D(ibooker,"TH1Resrphi", hidmanager.createHistoLayer("Res_rphi","layer",label,"").c_str() ,"Residuals of the hit x coordinate"); 
    layerMEs.meResrphi->setAxisTitle(("RecHit Res(x) in " + label).c_str());
  }
  //PullLFrphi
  if(switchPullLFrphi) {
    layerMEs.mePullLFrphi = bookME1D(ibooker,"TH1PullLFrphi", hidmanager.createHistoLayer("Pull_LF_rphi","layer",label,"").c_str() ,"Pull distribution");  
    layerMEs.mePullLFrphi->setAxisTitle(("Pull distribution (local frame) in " + label).c_str());
  }
  //PullMFrphi
  if(switchPullMFrphi) {
    layerMEs.mePullMFrphi = bookME1D(ibooker,"TH1PullMFrphi", hidmanager.createHistoLayer("Pull_MF_rphi","layer",label,"").c_str() ,"Pull distribution");  
    layerMEs.mePullMFrphi->setAxisTitle(("Pull distribution (measurement frame) in " + label).c_str());
  }
  //Chi2rphi
  if(switchChi2rphi) {
    layerMEs.meChi2rphi = bookME1D(ibooker,"TH1Chi2rphi", hidmanager.createHistoLayer("Chi2_rphi","layer",label,"").c_str() ,"RecHit Chi2 test"); 
    layerMEs.meChi2rphi->setAxisTitle(("RecHit Chi2 test in " + label).c_str()); 
  }
  //NsimHitrphi
  if(switchNsimHitrphi) {
    layerMEs.meNsimHitrphi = bookME1D(ibooker,"TH1NsimHitrphi", hidmanager.createHistoLayer("NsimHit_rphi","layer",label,"").c_str() ,"No. of assoc. simHits"); 
    layerMEs.meNsimHitrphi->setAxisTitle(("Number of assoc. simHits in " + label).c_str()); 
  }

  LayerMEsMap[label]=layerMEs;
 
}
//------------------------------------------------------------------------------------------
void SiStripRecHitsValid::createStereoAndMatchedMEs(DQMStore::IBooker & ibooker,std::string label) 
{
  SiStripHistoId hidmanager;
  StereoAndMatchedMEs stereoandmatchedMEs; 

  stereoandmatchedMEs.meWclusStereo = 0;
  stereoandmatchedMEs.meAdcStereo = 0;
  stereoandmatchedMEs.mePosxStereo = 0;
  stereoandmatchedMEs.meResolxStereo = 0;
  stereoandmatchedMEs.meResStereo = 0;
  stereoandmatchedMEs.mePullLFStereo = 0;
  stereoandmatchedMEs.mePullMFStereo = 0;
  stereoandmatchedMEs.meChi2Stereo = 0;
  stereoandmatchedMEs.meNsimHitStereo = 0;
  stereoandmatchedMEs.mePosxMatched = 0;
  stereoandmatchedMEs.mePosyMatched = 0;
  stereoandmatchedMEs.meResolxMatched = 0;
  stereoandmatchedMEs.meResolyMatched = 0;
  stereoandmatchedMEs.meResxMatched = 0;
  stereoandmatchedMEs.meResyMatched = 0;
  stereoandmatchedMEs.meChi2Matched = 0;
  stereoandmatchedMEs.meNsimHitMatched = 0;

  //WclusStereo
  if(switchWclusStereo) {
    stereoandmatchedMEs.meWclusStereo = bookME1D(ibooker,"TH1WclusStereo", hidmanager.createHistoLayer("Wclus_stereo","layer",label,"").c_str() ,"Cluster Width - Number of strips that belong to the RecHit cluster");  
    stereoandmatchedMEs.meWclusStereo->setAxisTitle(("Cluster Width [nr strips] in stereo modules in "+ label).c_str());
  }
  //AdcStereo
  if(switchAdcStereo) {
    stereoandmatchedMEs.meAdcStereo = bookME1D(ibooker,"TH1AdcStereo", hidmanager.createHistoLayer("Adc_stereo","layer",label,"").c_str() ,"RecHit Cluster Charge"); 
    stereoandmatchedMEs.meAdcStereo->setAxisTitle(("cluster charge [ADC] in stereo modules in " + label).c_str());
  }
  //PosxStereo
  if(switchPosxStereo) {
    stereoandmatchedMEs.mePosxStereo = bookME1D(ibooker,"TH1PosxStereo", hidmanager.createHistoLayer("Posx_stereo","layer",label,"").c_str() ,"RecHit x coord."); 
    stereoandmatchedMEs.mePosxStereo->setAxisTitle(("x RecHit coord. (local frame) in stereo modules in " + label).c_str());
  }
  //ResolxStereo
  if(switchResolxStereo) {
    stereoandmatchedMEs.meResolxStereo = bookME1D(ibooker,"TH1ResolxStereo", hidmanager.createHistoLayer("Resolx_stereo","layer",label,"").c_str() ,"RecHit resol(x) coord.");  
    stereoandmatchedMEs.meResolxStereo->setAxisTitle(("resol(x) RecHit coord. (local frame) in stereo modules in " + label).c_str());
  }
  //ResStereo
  if(switchResStereo) {
    stereoandmatchedMEs.meResStereo = bookME1D(ibooker,"TH1ResStereo", hidmanager.createHistoLayer("Res_stereo","layer",label,"").c_str() ,"Residuals of the hit x coordinate"); 
    stereoandmatchedMEs.meResStereo->setAxisTitle(("RecHit Res(x) in stereo modules in " + label).c_str());
  }
  //PullLFStereo
  if(switchPullLFStereo) {
    stereoandmatchedMEs.mePullLFStereo = bookME1D(ibooker,"TH1PullLFStereo", hidmanager.createHistoLayer("Pull_LF_stereo","layer",label,"").c_str() ,"Pull distribution");  
    stereoandmatchedMEs.mePullLFStereo->setAxisTitle(("Pull distribution (local frame) in stereo modules in " + label).c_str());
  }
  //PullMFStereo
  if(switchPullMFStereo) {
    stereoandmatchedMEs.mePullMFStereo = bookME1D(ibooker,"TH1PullMFStereo", hidmanager.createHistoLayer("Pull_MF_stereo","layer",label,"").c_str() ,"Pull distribution");  
    stereoandmatchedMEs.mePullMFStereo->setAxisTitle(("Pull distribution (measurement frame) in stereo modules in " + label).c_str());
  }
  //Chi2Stereo
  if(switchChi2Stereo) {
    stereoandmatchedMEs.meChi2Stereo = bookME1D(ibooker,"TH1Chi2Stereo", hidmanager.createHistoLayer("Chi2_stereo","layer",label,"").c_str() ,"RecHit Chi2 test");  
    stereoandmatchedMEs.meChi2Stereo->setAxisTitle(("RecHit Chi2 test in stereo modules in " + label).c_str()); 
  }
  //NsimHitStereo
  if(switchNsimHitStereo) {
    stereoandmatchedMEs.meNsimHitStereo = bookME1D(ibooker,"TH1NsimHitStereo", hidmanager.createHistoLayer("NsimHit_stereo","layer",label,"").c_str() ,"No. of assoc. simHits");  
    stereoandmatchedMEs.meNsimHitStereo->setAxisTitle(("Number of assoc. simHits in stereo modules in " + label).c_str()); 
  }
  //PosxMatched
  if(switchPosxMatched) {
    stereoandmatchedMEs.mePosxMatched = bookME1D(ibooker,"TH1PosxMatched", hidmanager.createHistoLayer("Posx_matched","layer",label,"").c_str() ,"RecHit x coord.");  
    stereoandmatchedMEs.mePosxMatched->setAxisTitle(("x coord. matched RecHit (local frame) in " + label).c_str());
  }
  //PosyMatched
  if(switchPosyMatched) {
    stereoandmatchedMEs.mePosyMatched = bookME1D(ibooker,"TH1PosyMatched", hidmanager.createHistoLayer("Posy_matched","layer",label,"").c_str() ,"RecHit y coord."); 
    stereoandmatchedMEs.mePosyMatched->setAxisTitle(("y coord. matched RecHit (local frame) in " + label).c_str());
  }
  //ResolxMatched
  if(switchResolxMatched) {
    stereoandmatchedMEs.meResolxMatched = bookME1D(ibooker,"TH1ResolxMatched", hidmanager.createHistoLayer("Resolx_matched","layer",label,"").c_str() ,"RecHit resol(x) coord.");  
    stereoandmatchedMEs.meResolxMatched->setAxisTitle(("resol(x) coord. matched RecHit (local frame) in " + label).c_str());
  }
  //ResolyMatched
  if(switchResolyMatched) {
    stereoandmatchedMEs.meResolyMatched = bookME1D(ibooker,"TH1ResolyMatched", hidmanager.createHistoLayer("Resoly_matched","layer",label,"").c_str() ,"RecHit resol(y) coord."); 
    stereoandmatchedMEs.meResolyMatched->setAxisTitle(("resol(y) coord. matched RecHit (local frame) in " + label).c_str());
  }
  //ResxMatched
  if(switchResxMatched) {
    stereoandmatchedMEs.meResxMatched = bookME1D(ibooker,"TH1ResxMatched", hidmanager.createHistoLayer("Resx_matched","layer",label,"").c_str() ,"Residuals of the hit x coord."); 
    stereoandmatchedMEs.meResxMatched->setAxisTitle(("Res(x) in matched RecHit in " + label).c_str());
  }
  //ResyMatched
  if(switchResyMatched) {
    stereoandmatchedMEs.meResyMatched = bookME1D(ibooker,"TH1ResyMatched", hidmanager.createHistoLayer("Resy_matched","layer",label,"").c_str() ,"Residuals of the hit y coord."); 
    stereoandmatchedMEs.meResyMatched->setAxisTitle(("Res(y) in matched RecHit in " + label).c_str());
  }
  //Chi2Matched
  if(switchChi2Matched) {
    stereoandmatchedMEs.meChi2Matched = bookME1D(ibooker,"TH1Chi2Matched", hidmanager.createHistoLayer("Chi2_matched","layer",label,"").c_str() ,"RecHit Chi2 test"); 
    stereoandmatchedMEs.meChi2Matched->setAxisTitle(("Matched RecHit Chi2 test in " + label).c_str()); 
  }
  //NsimHitMatched
  if(switchNsimHitMatched) {
    stereoandmatchedMEs.meNsimHitMatched = bookME1D(ibooker,"TH1NsimHitMatched", hidmanager.createHistoLayer("NsimHit_matched","layer",label,"").c_str() ,"No. of assoc. simHits"); 
    stereoandmatchedMEs.meNsimHitMatched->setAxisTitle(("Number of assoc. simHits in " + label).c_str()); 
  }

  StereoAndMatchedMEsMap[label]=stereoandmatchedMEs;
 
}
//------------------------------------------------------------------------------------------
void SiStripRecHitsValid::createSubDetMEs(DQMStore::IBooker & ibooker,std::string label) {

  SubDetMEs subdetMEs;
  subdetMEs.meNumrphi = 0;
  subdetMEs.meBunchrphi = 0;
  subdetMEs.meEventrphi = 0;
  subdetMEs.meNumStereo = 0;
  subdetMEs.meBunchStereo = 0;
  subdetMEs.meEventStereo = 0;
  subdetMEs.meNumMatched = 0;
  subdetMEs.meBunchMatched = 0;
  subdetMEs.meEventMatched = 0;

  std::string HistoName;
  //Numrphi
  if (switchNumrphi){
    HistoName = "TH1Numrphi__" + label;
    subdetMEs.meNumrphi = bookME1D(ibooker,"TH1Numrphi",HistoName.c_str(),"Num of RecHits");
    subdetMEs.meNumrphi->setAxisTitle(("Total number of RecHits in "+ label).c_str());
  }  
  //Bunchrphi
  if(switchBunchrphi) {
    HistoName = "TH1Bunchrphi__" + label;
    subdetMEs.meBunchrphi = bookME1D(ibooker,"TH1Bunchrphi",HistoName.c_str(),"Bunch Crossing");
    subdetMEs.meBunchrphi->setAxisTitle(("Bunch crossing in " + label).c_str()); 
  }
  //Eventrphi
  if(switchEventrphi) {
    HistoName = "TH1Eventrphi__" + label;
    subdetMEs.meEventrphi = bookME1D(ibooker,"TH1Eventrphi",HistoName.c_str(),"Event (in-time bunch)");
    subdetMEs.meEventrphi->setAxisTitle(("Event (in-time bunch) in " + label).c_str()); 
  }
  //NumStereo
  if (switchNumStereo){
    HistoName = "TH1NumStereo__" + label;
    subdetMEs.meNumStereo = bookME1D(ibooker,"TH1NumStereo",HistoName.c_str(),"Num of RecHits in stereo modules");
    subdetMEs.meNumStereo->setAxisTitle(("Total number of RecHits, stereo modules in "+ label).c_str());
  }  
  //BunchStereo
  if(switchBunchStereo) {
    HistoName = "TH1BunchStereo__" + label;
    subdetMEs.meBunchStereo = bookME1D(ibooker,"TH1BunchStereo",HistoName.c_str(),"Bunch Crossing");
    subdetMEs.meBunchStereo->setAxisTitle(("Bunch crossing, stereo modules in " + label).c_str()); 
  }
  //EventStereo
  if(switchEventStereo) {
    HistoName = "TH1EventStereo__" + label;
    subdetMEs.meEventStereo = bookME1D(ibooker,"TH1EventStereo",HistoName.c_str(),"Event (in-time bunch)");
    subdetMEs.meEventStereo->setAxisTitle(("Event (in-time bunch), stereo modules in " + label).c_str()); 
  }
  //NumMatched
  if (switchNumMatched){
    HistoName = "TH1NumMatched__" + label;
    subdetMEs.meNumMatched = bookME1D(ibooker,"TH1NumMatched",HistoName.c_str(),"Num of matched RecHits" );
    subdetMEs.meNumMatched->setAxisTitle(("Total number of matched RecHits in "+ label).c_str());
  }  
  //BunchMatched
  if(switchBunchMatched) {
    HistoName = "TH1BunchMatched__" + label;
    subdetMEs.meBunchMatched = bookME1D(ibooker,"TH1BunchMatched",HistoName.c_str(),"Bunch Crossing");
    subdetMEs.meBunchMatched->setAxisTitle(("Bunch crossing, matched RecHits in " + label).c_str()); 
  }
  //EventMatched
  if(switchEventMatched) {
    HistoName = "TH1EventMatched__" + label;
    subdetMEs.meEventMatched = bookME1D(ibooker,"TH1EventMatched",HistoName.c_str(),"Event (in-time bunch)");
    subdetMEs.meEventMatched->setAxisTitle(("Event (in-time bunch), matched RecHits in " + label).c_str()); 
  }

  SubDetMEsMap[label]=subdetMEs;
}
//------------------------------------------------------------------------------------------
MonitorElement* SiStripRecHitsValid::bookME1D(DQMStore::IBooker & ibooker, const char* ParameterSetLabel, const char* HistoName, const char* HistoTitle)
{
  Parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
  return ibooker.book1D(HistoName,HistoTitle,
			   Parameters.getParameter<int32_t>("Nbinx"),
			   Parameters.getParameter<double>("xmin"),
			   Parameters.getParameter<double>("xmax")
			   );
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripRecHitsValid);

