#include "Validation/TrackerRecHits/interface/SiStripRecHitsValid.h"

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
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

//--- for RecHit
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h" 
#include "DataFormats/SiStripCluster/interface/SiStripClusterCollection.h" 
#include "DataFormats/Common/interface/OwnVector.h" 
#include "DQMServices/Core/interface/DQMStore.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

using namespace std;
using namespace edm;


//Constructor
SiStripRecHitsValid::SiStripRecHitsValid(const ParameterSet& ps) :
  conf_(ps),
  trackerHitAssociatorConfig_(ps, consumesCollector()),
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

void SiStripRecHitsValid::analyze(const edm::Event& e, const edm::EventSetup& es) {

  LogInfo("EventInfo") << " Run = " << e.id().run() << " Event = " << e.id().event();  
  
  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();
    
  // Step A: Get Inputs 
  edm::Handle<SiStripMatchedRecHit2DCollection> rechitsmatched;
  edm::Handle<SiStripRecHit2DCollection> rechitsrphi;
  edm::Handle<SiStripRecHit2DCollection> rechitsstereo;
  e.getByToken(matchedRecHitsToken_, rechitsmatched);
  e.getByToken(rphiRecHitsToken_, rechitsrphi);
  e.getByToken(stereoRecHitsToken_, rechitsstereo);

  //Variables in order to count total num of rechitrphi,rechitstereo, rechitmatched in subdetectors
  std::map<std::string, int > totnumrechitrphi;
  std::map<std::string, int > totnumrechitstereo;
  std::map<std::string, int > totnumrechitmatched;
  int totrechitrphi =0;
  int totrechitstereo =0;
  int totrechitmatched =0;
   
  TrackerHitAssociator associate(e, trackerHitAssociatorConfig_);
  
  edm::ESHandle<TrackerGeometry> pDD;
  es.get<TrackerDigiGeometryRecord> ().get (pDD);
  const TrackerGeometry &tracker(*pDD);
  
  SiStripHistoId hidmanager;
  SiStripFolderOrganizer fold_organ;
  for (auto const& theDetSet : *rechitsrphi) {
    DetId detid = theDetSet.detId();
    uint32_t myid = detid.rawId();       
    totrechitrphi += theDetSet.size();
  
    std::string label = hidmanager.getSubdetid(myid,tTopo,true);
    std::map<std::string, LayerMEs>::iterator iLayerME  = LayerMEsMap.find(label);
    std::pair<std::string,int32_t> det_lay_pair = fold_organ.GetSubDetAndLayer(myid,tTopo,true);

    totnumrechitrphi[det_lay_pair.first] = totnumrechitrphi[det_lay_pair.first] + theDetSet.size();
    //loop over rechits-rphi in the same subdetector
    if(iLayerME != LayerMEsMap.end()){
      for(auto const& rechit : theDetSet){	
        const GeomDetUnit *  det = tracker.idToDetUnit(detid);
        const StripTopology &topol=static_cast<const StripGeomDetUnit*>(det)->specificTopology();
        //analyze RecHits 
        rechitanalysis(rechit,topol,associate);
        // fill the result in a histogram
        fillME(iLayerME->second.meWclusrphi,rechitpro.clusiz);
        fillME(iLayerME->second.meAdcrphi,rechitpro.cluchg);
        fillME(iLayerME->second.mePosxrphi,rechitpro.x);
        fillME(iLayerME->second.meResolxrphi,rechitpro.resolxx);
        fillME(iLayerME->second.meNsimHitrphi,rechitpro.NsimHit);
        if (rechitpro.NsimHit > 0) {
          std::map<std::string, SubDetMEs>::iterator iSubDetME = SubDetMEsMap.find(det_lay_pair.first);
          fillME(iSubDetME->second.meBunchrphi, rechitpro.bunch);
          if (rechitpro.bunch == 0) fillME(iSubDetME->second.meEventrphi, rechitpro.event);
        }
        if ( rechitpro.resx != -999999. || rechitpro.pullMF != -999999. || rechitpro.chi2 != -999999. ){
          fillME(iLayerME->second.meResrphi,rechitpro.resx);
          fillME(iLayerME->second.mePullLFrphi,rechitpro.resx/sqrt(rechitpro.resolxx));
          fillME(iLayerME->second.mePullMFrphi,rechitpro.pullMF);
          fillME(iLayerME->second.meChi2rphi,rechitpro.chi2);
        }
      }
    }
  }

  // start loops over detectors with detected rechitsstereo
  for (auto const& theDetSet : *rechitsstereo) {
    DetId detid = theDetSet.detId();
    uint32_t myid= detid.rawId();       
    totrechitstereo += theDetSet.size();
 
    std::string label = hidmanager.getSubdetid(myid,tTopo,true);
    std::map<std::string, StereoAndMatchedMEs>::iterator iStereoAndMatchedME  = StereoAndMatchedMEsMap.find(label);
    std::pair<std::string,int32_t> det_lay_pair = fold_organ.GetSubDetAndLayer(myid,tTopo,true);
   
    totnumrechitstereo[det_lay_pair.first] = totnumrechitstereo[det_lay_pair.first] + theDetSet.size();
    //loop over rechits-stereo in the same subdetector
    if(iStereoAndMatchedME != StereoAndMatchedMEsMap.end()){
      for (auto const& rechit : theDetSet) {
        const GeomDetUnit *  det = tracker.idToDetUnit(detid);
        const StripTopology &topol=static_cast<const StripGeomDetUnit*>(det)->specificTopology();
        //analyze RecHits
        rechitanalysis(rechit,topol,associate);
        // fill the result in a histogram
        fillME(iStereoAndMatchedME->second.meWclusStereo,rechitpro.clusiz);
        fillME(iStereoAndMatchedME->second.meAdcStereo,rechitpro.cluchg);
        fillME(iStereoAndMatchedME->second.mePosxStereo,rechitpro.x);
        fillME(iStereoAndMatchedME->second.meResolxStereo,sqrt(rechitpro.resolxx));
        fillME(iStereoAndMatchedME->second.meNsimHitStereo,rechitpro.NsimHit);
        if (rechitpro.NsimHit > 0) {
          std::map<std::string, SubDetMEs>::iterator iSubDetME = SubDetMEsMap.find(det_lay_pair.first);
          fillME(iSubDetME->second.meBunchStereo, rechitpro.bunch);
          if (rechitpro.bunch == 0) fillME(iSubDetME->second.meEventStereo, rechitpro.event);
        }
        if ( rechitpro.resx != -999999. || rechitpro.pullMF != -999999. || rechitpro.chi2 != -999999. ){
          fillME(iStereoAndMatchedME->second.meResStereo,rechitpro.resx);
          fillME(iStereoAndMatchedME->second.mePullLFStereo,rechitpro.resx/sqrt(rechitpro.resolxx));
          fillME(iStereoAndMatchedME->second.mePullMFStereo,rechitpro.pullMF);
          fillME(iStereoAndMatchedME->second.meChi2Stereo,rechitpro.chi2);
        }
      }
    }
  }

  // start loops over detectors with detected rechitmatched
  for (auto const & theDetSet : *rechitsmatched) {
    DetId detid = theDetSet.detId();
    uint32_t myid = detid.rawId();       
    totrechitmatched += theDetSet.size();
 
    std::string label = hidmanager.getSubdetid(myid,tTopo,true);
    std::map<std::string, StereoAndMatchedMEs>::iterator iStereoAndMatchedME  = StereoAndMatchedMEsMap.find(label);
    std::pair<std::string,int32_t> det_lay_pair = fold_organ.GetSubDetAndLayer(myid,tTopo,true);
   
    totnumrechitmatched[det_lay_pair.first] = totnumrechitmatched[det_lay_pair.first] + theDetSet.size();
    //loop over rechits-matched in the same subdetector
    if(iStereoAndMatchedME != StereoAndMatchedMEsMap.end()){
      for (auto const& rechit : theDetSet) {
        const GluedGeomDet* gluedDet = static_cast<const GluedGeomDet*>(tracker.idToDet(rechit.geographicalId()));
        //analyze RecHits 
        rechitanalysis_matched(rechit, gluedDet, associate);
        // fill the result in a histogram
        fillME(iStereoAndMatchedME->second.mePosxMatched,rechitpro.x);
        fillME(iStereoAndMatchedME->second.mePosyMatched,rechitpro.y);
        fillME(iStereoAndMatchedME->second.meResolxMatched,sqrt(rechitpro.resolxx));
        fillME(iStereoAndMatchedME->second.meResolyMatched,sqrt(rechitpro.resolyy));
        fillME(iStereoAndMatchedME->second.meNsimHitMatched,rechitpro.NsimHit);
        if (rechitpro.NsimHit > 0) {
          std::map<std::string, SubDetMEs>::iterator iSubDetME = SubDetMEsMap.find(det_lay_pair.first);
          fillME(iSubDetME->second.meBunchMatched, rechitpro.bunch);
          if (rechitpro.bunch == 0) fillME(iSubDetME->second.meEventMatched, rechitpro.event);
        }
        if ( rechitpro.resx != -999999. || rechitpro.resy != -999999. || rechitpro.chi2 != -999999. ){
          fillME(iStereoAndMatchedME->second.meResxMatched,rechitpro.resx);
          fillME(iStereoAndMatchedME->second.meResyMatched,rechitpro.resy);
          fillME(iStereoAndMatchedME->second.meChi2Matched,rechitpro.chi2);
        }
      }
    }
  }//End of loops over detectors
  
  //now fill the cumulative histograms of the hits
  for (std::vector<std::string>::iterator iSubdet  = SubDetList_.begin(); iSubdet != SubDetList_.end(); ++iSubdet){
    std::map<std::string, SubDetMEs>::iterator iSubDetME  = SubDetMEsMap.find((*iSubdet));
    fillME(iSubDetME->second.meNumrphi,totnumrechitrphi[(*iSubdet)]);
    fillME(iSubDetME->second.meNumStereo,totnumrechitstereo[(*iSubdet)]);
    fillME(iSubDetME->second.meNumMatched,totnumrechitmatched[(*iSubdet)]);
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
  //if (stripDet == nullptr) throw MeasurementDetException("HitMatcher hit is not on StripGeomDetUnit");
  
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
  
  double selfAngle = topol.stripAngle( topol.strip( hit.localPosition()));
  
  LocalVector stripDir( sin(selfAngle), cos(selfAngle), 0); // vector along strip in hit frame
 
  LocalVector localStripDir( plane.toLocal(stripDet->surface().toGlobal( stripDir)));
  
  return std::pair<LocalPoint,LocalVector>( projectedPos, localStripDir);
}
//--------------------------------------------------------------------------------------------
void SiStripRecHitsValid::rechitanalysis(SiStripRecHit2D const rechit,const StripTopology &topol,TrackerHitAssociator& associate){
  
  rechitpro.resx = -999999.; rechitpro.resy = -999999.; rechitpro.pullMF = -999999.; 
  rechitpro.chi2 = -999999.; rechitpro.bunch = -999999.; rechitpro.event = -999999.;
  
  LocalPoint position=rechit.localPosition();
  LocalError error=rechit.localPositionError();
  MeasurementPoint Mposition = topol.measurementPosition(position);
  MeasurementError Merror = topol.measurementError(position,error);
  const auto & amplitudes=(rechit.cluster())->amplitudes();
  int totcharge=0;
  for(auto ia : amplitudes){
    totcharge += ia;
  }
  rechitpro.x = position.x();
  rechitpro.y = position.y();
  //rechitpro.z = position.z();
  rechitpro.resolxx = error.xx();
  //rechitpro.resolxy = error.xy();
  rechitpro.resolyy = error.yy();
  rechitpro.clusiz = amplitudes.size();
  rechitpro.cluchg = totcharge;

  auto const & matched = associate.associateHit(rechit);
  rechitpro.NsimHit = matched.size();

  if(!matched.empty()){

    float mindist = std::numeric_limits<float>::max();
    float dist = std::numeric_limits<float>::max();
    PSimHit const * closest = nullptr;
 
    for(auto const &m : matched){
      dist = fabs(rechitpro.x - m.localPosition().x());
      if(dist<mindist){
	mindist = dist;
	closest = &m;
      }
    }  
    rechitpro.bunch = closest->eventId().bunchCrossing();
    rechitpro.event = closest->eventId().event();
    rechitpro.resx = rechitpro.x - closest->localPosition().x();
    rechitpro.pullMF = (Mposition.x() - (topol.measurementPosition(closest->localPosition())).x())/sqrt(Merror.uu());
    
    //chi2test compare rechit errors with the simhit position ( using null matrix for the simhit). 
    //Can spot problems in the geometry better than a simple residual. (thanks to BorisM)
    AlgebraicVector  rhparameters(2);//= rechit.parameters();
    rhparameters[0] = position.x(); 
    rhparameters[1] = position.y();
    AlgebraicVector shparameters(2);
    shparameters[0] = closest->localPosition().x();
    shparameters[1] = closest->localPosition().y();
    AlgebraicVector r(rhparameters - shparameters);
    AlgebraicSymMatrix R(2);//  = rechit.parametersError();
    R[0][0] = error.xx();
    R[0][1] = error.xy();
    R[1][1] = error.yy();
    int ierr; 
    R.invert(ierr); // if (ierr != 0) throw exception;
    float est = R.similarity(r);
    rechitpro.chi2 = est;
  }

}


//--------------------------------------------------------------------------------------------
void SiStripRecHitsValid::rechitanalysis_matched(SiStripMatchedRecHit2D const rechit, const GluedGeomDet* gluedDet, TrackerHitAssociator& associate){
  
  rechitpro.resx = -999999.; rechitpro.resy = -999999.; rechitpro.pullMF = -999999.; 
  rechitpro.chi2 = -999999.; rechitpro.bunch = -999999.; rechitpro.event = -999999.;
  rechitpro.clusiz = -999999.; rechitpro.cluchg = -999999.;

  LocalPoint position=rechit.localPosition();
  LocalError error=rechit.localPositionError();

  rechitpro.x = position.x();
  rechitpro.y = position.y();
  //rechitpro.z = position.z();
  rechitpro.resolxx = error.xx();
  //rechitpro.resolxy = error.xy();
  rechitpro.resolyy = error.yy();

  auto const & matched = associate.associateHit(rechit);
  rechitpro.NsimHit = matched.size();

  if(matched.empty()) return;
    float mindist = std::numeric_limits<float>::max();
    float dist = std::numeric_limits<float>::max();
    float dist2 = std::numeric_limits<float>::max();
    float distx = std::numeric_limits<float>::max();
    float disty = std::numeric_limits<float>::max();
    PSimHit const * closest = nullptr;
    std::pair<LocalPoint,LocalVector> closestPair;

    const StripGeomDetUnit* partnerstripdet = static_cast<const StripGeomDetUnit*>(gluedDet->stereoDet());
    std::pair<LocalPoint,LocalVector> hitPair;

    for(auto const &m : matched){
      SiStripDetId hitDetId(m.detUnitId());
      if (hitDetId.stereo()) {  // project from the stereo sensor
      //project simhit;
	hitPair= projectHit(m,partnerstripdet,gluedDet->surface());
	distx = rechitpro.x - hitPair.first.x();
	disty = rechitpro.y - hitPair.first.y();
	dist2 = distx*distx+disty*disty;
	dist = sqrt(dist2);
	if(dist<mindist){
	  mindist = dist;
	  closestPair = hitPair;
	  closest = &m;
	}
      }
    }
    if (!closest) return;  
    rechitpro.bunch = closest->eventId().bunchCrossing();
    rechitpro.event = closest->eventId().event();
    rechitpro.resx = rechitpro.x - closestPair.first.x();
    rechitpro.resy = rechitpro.y - closestPair.first.y();
  
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
    float est = R.similarity(r);
    rechitpro.chi2 = est;
  


}

//--------------------------------------------------------------------------------------------
void SiStripRecHitsValid::createMEs(DQMStore::IBooker & ibooker,const edm::EventSetup& es){

  //Retrieve tracker topology from geometry
  edm::ESHandle<TrackerTopology> tTopoHandle;
  es.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();
  
  // take from eventSetup the SiStripDetCabling object - here will use SiStripDetControl later on
  es.get<SiStripDetCablingRcd>().get(SiStripDetCabling_);
    
  // get list of active detectors from SiStripDetCabling 
  std::vector<uint32_t> activeDets;
  SiStripDetCabling_->addActiveDetectorsRawIds(activeDets);

  SiStripFolderOrganizer folder_organizer;
  // folder_organizer.setSiStripFolderName(topFolderName_);
  std::string curfold = topFolderName_;
  folder_organizer.setSiStripFolderName(curfold);
  folder_organizer.setSiStripFolder();

  createTotalMEs(ibooker);
  // loop over detectors and book MEs
  edm::LogInfo("SiStripTkRecHits|SiStripRecHitsValid")<<"nr. of activeDets:  "<<activeDets.size();
  const std::string& tec = "TEC", tid = "TID", tob = "TOB", tib = "TIB";        
  for(auto detid_iterator=activeDets.begin(), detid_end=activeDets.end(); detid_iterator!=detid_end; ++detid_iterator){
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
      
    if(LayerMEsMap.find(label)==LayerMEsMap.end()) {
	
      // get detids for the layer
      // Keep in mind that when we are on the TID or TEC we deal with rings not wheel 
      int32_t lnumber = det_layer_pair.second;
      const std::string& lname = det_layer_pair.first; 
      std::vector<uint32_t> layerDetIds;
      if (lname == tec) {
        if (lnumber > 0) {
	  SiStripSubStructure::getTECDetectors(activeDets,layerDetIds,tTopo,2,0,0,0,abs(lnumber),0);
        } else if (lnumber < 0) {
	  SiStripSubStructure::getTECDetectors(activeDets,layerDetIds,tTopo,1,0,0,0,abs(lnumber),0);
        }
      } else if (lname == tid) {
         if (lnumber > 0) {
	   SiStripSubStructure::getTIDDetectors(activeDets,layerDetIds,tTopo,2,0,abs(lnumber),0);
        } else if (lnumber < 0) {
	  SiStripSubStructure::getTIDDetectors(activeDets,layerDetIds,tTopo,1,0,abs(lnumber),0);
        }
      } else if (lname == tob) {
	SiStripSubStructure::getTOBDetectors(activeDets,layerDetIds,tTopo,lnumber,0,0);
      } else if (lname == tib) {
	SiStripSubStructure::getTIBDetectors(activeDets,layerDetIds,tTopo,lnumber,0,0,0);
      }  
      LayerDetMap[label] = layerDetIds;

      // book Layer MEs 
      folder_organizer.setLayerFolder(detid,tTopo,det_layer_pair.second,true);
      createLayerMEs(ibooker,label);
    }
    // book sub-detector plots 
    if (SubDetMEsMap.find(det_layer_pair.first) == SubDetMEsMap.end()){
      auto sdet_pair = folder_organizer.getSubDetFolderAndTag(detid, tTopo);
      ibooker.setCurrentFolder(sdet_pair.first);
      createSubDetMEs(ibooker,det_layer_pair.first);        
    }
    //Create StereoAndMatchedMEs
    if(StereoAndMatchedMEsMap.find(label)==StereoAndMatchedMEsMap.end()) {
	
      // get detids for the stereo and matched layer. We are going to need a bool for these layers
      bool isStereo = false;
      // Keep in mind that when we are on the TID or TEC we deal with rings not wheel 
      std::vector<uint32_t> stereoandmatchedDetIds;        
      int32_t stereolnumber = det_layer_pair.second;
      const std::string& stereolname = det_layer_pair.first;
      if ( stereolname == tec && (tTopo->tecIsStereo(detid)) ) {
        if ( stereolnumber > 0 ) {
          SiStripSubStructure::getTECDetectors(activeDets,stereoandmatchedDetIds,tTopo,2,0,0,0,abs(stereolnumber),1);
	  isStereo = true;
        } else if ( stereolnumber < 0 ) {
	  SiStripSubStructure::getTECDetectors(activeDets,stereoandmatchedDetIds,tTopo,1,0,0,0,abs(stereolnumber),1);
	  isStereo = true;
        }
      } else if ( stereolname == tid && (tTopo->tidIsStereo(detid)) ) {
        if ( stereolnumber > 0 ) {
	  SiStripSubStructure::getTIDDetectors(activeDets,stereoandmatchedDetIds,tTopo,2,0,abs(stereolnumber),1);
	  isStereo = true;
        } else if ( stereolnumber < 0 ) {
	  SiStripSubStructure::getTIDDetectors(activeDets,stereoandmatchedDetIds,tTopo,1,0,abs(stereolnumber),1);
	  isStereo = true;
        }
      } else if ( stereolname == tob && (tTopo->tobIsStereo(detid)) ) {
	SiStripSubStructure::getTOBDetectors(activeDets,stereoandmatchedDetIds,tTopo,stereolnumber,0,0);
	isStereo = true;
      } else if ( stereolname == tib && (tTopo->tibIsStereo(detid)) ) {
	SiStripSubStructure::getTIBDetectors(activeDets,stereoandmatchedDetIds,tTopo,stereolnumber,0,0,0);
	isStereo = true;
      } 

      StereoAndMatchedDetMap[label] = stereoandmatchedDetIds;

      if(isStereo){
	//book StereoAndMatched MEs 
	folder_organizer.setLayerFolder(detid,tTopo,det_layer_pair.second,true);
	//Create the Monitor Elements only when we have a stereo module
	createStereoAndMatchedMEs(ibooker,label);
      }
    }
  }//end of loop over detectors
}
//------------------------------------------------------------------------------------------
void SiStripRecHitsValid::createTotalMEs(DQMStore::IBooker & ibooker) 
{
  totalMEs.meNumTotrphi = nullptr;
  totalMEs.meNumTotStereo = nullptr;
  totalMEs.meNumTotMatched = nullptr;

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

  layerMEs.meWclusrphi = nullptr;
  layerMEs.meAdcrphi = nullptr;
  layerMEs.mePosxrphi = nullptr;
  layerMEs.meResolxrphi = nullptr;
  layerMEs.meResrphi = nullptr;
  layerMEs.mePullLFrphi = nullptr;
  layerMEs.mePullMFrphi = nullptr;
  layerMEs.meChi2rphi = nullptr;
  layerMEs.meNsimHitrphi = nullptr;

  //Wclusrphi
  if(switchWclusrphi) {
    layerMEs.meWclusrphi = bookME1D(ibooker,"TH1Wclusrphi", hidmanager.createHistoLayer("Wclus_rphi","layer",label,"").c_str() ,"Cluster Width - Number of strips that belong to the RecHit cluster"); 
    layerMEs.meWclusrphi->setAxisTitle("Cluster Width [nr strips] in "+ label);
  }
  //Adcrphi
  if(switchAdcrphi) {
    layerMEs.meAdcrphi = bookME1D(ibooker,"TH1Adcrphi", hidmanager.createHistoLayer("Adc_rphi","layer",label,"").c_str() ,"RecHit Cluster Charge");
    layerMEs.meAdcrphi->setAxisTitle("cluster charge [ADC] in " + label);
  }
  //Posxrphi
  if(switchPosxrphi) {
    layerMEs.mePosxrphi = bookME1D(ibooker,"TH1Posxrphi", hidmanager.createHistoLayer("Posx_rphi","layer",label,"").c_str() ,"RecHit x coord."); 
    layerMEs.mePosxrphi->setAxisTitle("x RecHit coord. (local frame) in " + label);
  }
  //Resolxrphi
  if(switchResolxrphi) {
    layerMEs.meResolxrphi = bookME1D(ibooker,"TH1Resolxrphi", hidmanager.createHistoLayer("Resolx_rphi","layer",label,"").c_str() ,"RecHit resol(x) coord.");   //<resolor>~20micron  
    layerMEs.meResolxrphi->setAxisTitle("resol(x) RecHit coord. (local frame) in " + label);
  }
  //Resrphi
  if(switchResrphi) {
    layerMEs.meResrphi = bookME1D(ibooker,"TH1Resrphi", hidmanager.createHistoLayer("Res_rphi","layer",label,"").c_str() ,"Residuals of the hit x coordinate"); 
    layerMEs.meResrphi->setAxisTitle("RecHit Res(x) in " + label);
  }
  //PullLFrphi
  if(switchPullLFrphi) {
    layerMEs.mePullLFrphi = bookME1D(ibooker,"TH1PullLFrphi", hidmanager.createHistoLayer("Pull_LF_rphi","layer",label,"").c_str() ,"Pull distribution");  
    layerMEs.mePullLFrphi->setAxisTitle("Pull distribution (local frame) in " + label);
  }
  //PullMFrphi
  if(switchPullMFrphi) {
    layerMEs.mePullMFrphi = bookME1D(ibooker,"TH1PullMFrphi", hidmanager.createHistoLayer("Pull_MF_rphi","layer",label,"").c_str() ,"Pull distribution");  
    layerMEs.mePullMFrphi->setAxisTitle("Pull distribution (measurement frame) in " + label);
  }
  //Chi2rphi
  if(switchChi2rphi) {
    layerMEs.meChi2rphi = bookME1D(ibooker,"TH1Chi2rphi", hidmanager.createHistoLayer("Chi2_rphi","layer",label,"").c_str() ,"RecHit Chi2 test"); 
    layerMEs.meChi2rphi->setAxisTitle("RecHit Chi2 test in " + label); 
  }
  //NsimHitrphi
  if(switchNsimHitrphi) {
    layerMEs.meNsimHitrphi = bookME1D(ibooker,"TH1NsimHitrphi", hidmanager.createHistoLayer("NsimHit_rphi","layer",label,"").c_str() ,"No. of assoc. simHits"); 
    layerMEs.meNsimHitrphi->setAxisTitle("Number of assoc. simHits in " + label); 
  }

  LayerMEsMap[label]=layerMEs;
 
}
//------------------------------------------------------------------------------------------
void SiStripRecHitsValid::createStereoAndMatchedMEs(DQMStore::IBooker & ibooker,std::string label) 
{
  SiStripHistoId hidmanager;
  StereoAndMatchedMEs stereoandmatchedMEs; 

  stereoandmatchedMEs.meWclusStereo = nullptr;
  stereoandmatchedMEs.meAdcStereo = nullptr;
  stereoandmatchedMEs.mePosxStereo = nullptr;
  stereoandmatchedMEs.meResolxStereo = nullptr;
  stereoandmatchedMEs.meResStereo = nullptr;
  stereoandmatchedMEs.mePullLFStereo = nullptr;
  stereoandmatchedMEs.mePullMFStereo = nullptr;
  stereoandmatchedMEs.meChi2Stereo = nullptr;
  stereoandmatchedMEs.meNsimHitStereo = nullptr;
  stereoandmatchedMEs.mePosxMatched = nullptr;
  stereoandmatchedMEs.mePosyMatched = nullptr;
  stereoandmatchedMEs.meResolxMatched = nullptr;
  stereoandmatchedMEs.meResolyMatched = nullptr;
  stereoandmatchedMEs.meResxMatched = nullptr;
  stereoandmatchedMEs.meResyMatched = nullptr;
  stereoandmatchedMEs.meChi2Matched = nullptr;
  stereoandmatchedMEs.meNsimHitMatched = nullptr;

  //WclusStereo
  if(switchWclusStereo) {
    stereoandmatchedMEs.meWclusStereo = bookME1D(ibooker,"TH1WclusStereo", hidmanager.createHistoLayer("Wclus_stereo","layer",label,"").c_str() ,"Cluster Width - Number of strips that belong to the RecHit cluster");  
    stereoandmatchedMEs.meWclusStereo->setAxisTitle("Cluster Width [nr strips] in stereo modules in "+ label);
  }
  //AdcStereo
  if(switchAdcStereo) {
    stereoandmatchedMEs.meAdcStereo = bookME1D(ibooker,"TH1AdcStereo", hidmanager.createHistoLayer("Adc_stereo","layer",label,"").c_str() ,"RecHit Cluster Charge"); 
    stereoandmatchedMEs.meAdcStereo->setAxisTitle("cluster charge [ADC] in stereo modules in " + label);
  }
  //PosxStereo
  if(switchPosxStereo) {
    stereoandmatchedMEs.mePosxStereo = bookME1D(ibooker,"TH1PosxStereo", hidmanager.createHistoLayer("Posx_stereo","layer",label,"").c_str() ,"RecHit x coord."); 
    stereoandmatchedMEs.mePosxStereo->setAxisTitle("x RecHit coord. (local frame) in stereo modules in " + label);
  }
  //ResolxStereo
  if(switchResolxStereo) {
    stereoandmatchedMEs.meResolxStereo = bookME1D(ibooker,"TH1ResolxStereo", hidmanager.createHistoLayer("Resolx_stereo","layer",label,"").c_str() ,"RecHit resol(x) coord.");  
    stereoandmatchedMEs.meResolxStereo->setAxisTitle("resol(x) RecHit coord. (local frame) in stereo modules in " + label);
  }
  //ResStereo
  if(switchResStereo) {
    stereoandmatchedMEs.meResStereo = bookME1D(ibooker,"TH1ResStereo", hidmanager.createHistoLayer("Res_stereo","layer",label,"").c_str() ,"Residuals of the hit x coordinate"); 
    stereoandmatchedMEs.meResStereo->setAxisTitle("RecHit Res(x) in stereo modules in " + label);
  }
  //PullLFStereo
  if(switchPullLFStereo) {
    stereoandmatchedMEs.mePullLFStereo = bookME1D(ibooker,"TH1PullLFStereo", hidmanager.createHistoLayer("Pull_LF_stereo","layer",label,"").c_str() ,"Pull distribution");  
    stereoandmatchedMEs.mePullLFStereo->setAxisTitle("Pull distribution (local frame) in stereo modules in " + label);
  }
  //PullMFStereo
  if(switchPullMFStereo) {
    stereoandmatchedMEs.mePullMFStereo = bookME1D(ibooker,"TH1PullMFStereo", hidmanager.createHistoLayer("Pull_MF_stereo","layer",label,"").c_str() ,"Pull distribution");  
    stereoandmatchedMEs.mePullMFStereo->setAxisTitle("Pull distribution (measurement frame) in stereo modules in " + label);
  }
  //Chi2Stereo
  if(switchChi2Stereo) {
    stereoandmatchedMEs.meChi2Stereo = bookME1D(ibooker,"TH1Chi2Stereo", hidmanager.createHistoLayer("Chi2_stereo","layer",label,"").c_str() ,"RecHit Chi2 test");  
    stereoandmatchedMEs.meChi2Stereo->setAxisTitle("RecHit Chi2 test in stereo modules in " + label); 
  }
  //NsimHitStereo
  if(switchNsimHitStereo) {
    stereoandmatchedMEs.meNsimHitStereo = bookME1D(ibooker,"TH1NsimHitStereo", hidmanager.createHistoLayer("NsimHit_stereo","layer",label,"").c_str() ,"No. of assoc. simHits");  
    stereoandmatchedMEs.meNsimHitStereo->setAxisTitle("Number of assoc. simHits in stereo modules in " + label); 
  }
  //PosxMatched
  if(switchPosxMatched) {
    stereoandmatchedMEs.mePosxMatched = bookME1D(ibooker,"TH1PosxMatched", hidmanager.createHistoLayer("Posx_matched","layer",label,"").c_str() ,"RecHit x coord.");  
    stereoandmatchedMEs.mePosxMatched->setAxisTitle("x coord. matched RecHit (local frame) in " + label);
  }
  //PosyMatched
  if(switchPosyMatched) {
    stereoandmatchedMEs.mePosyMatched = bookME1D(ibooker,"TH1PosyMatched", hidmanager.createHistoLayer("Posy_matched","layer",label,"").c_str() ,"RecHit y coord."); 
    stereoandmatchedMEs.mePosyMatched->setAxisTitle("y coord. matched RecHit (local frame) in " + label);
  }
  //ResolxMatched
  if(switchResolxMatched) {
    stereoandmatchedMEs.meResolxMatched = bookME1D(ibooker,"TH1ResolxMatched", hidmanager.createHistoLayer("Resolx_matched","layer",label,"").c_str() ,"RecHit resol(x) coord.");  
    stereoandmatchedMEs.meResolxMatched->setAxisTitle("resol(x) coord. matched RecHit (local frame) in " + label);
  }
  //ResolyMatched
  if(switchResolyMatched) {
    stereoandmatchedMEs.meResolyMatched = bookME1D(ibooker,"TH1ResolyMatched", hidmanager.createHistoLayer("Resoly_matched","layer",label,"").c_str() ,"RecHit resol(y) coord."); 
    stereoandmatchedMEs.meResolyMatched->setAxisTitle("resol(y) coord. matched RecHit (local frame) in " + label);
  }
  //ResxMatched
  if(switchResxMatched) {
    stereoandmatchedMEs.meResxMatched = bookME1D(ibooker,"TH1ResxMatched", hidmanager.createHistoLayer("Resx_matched","layer",label,"").c_str() ,"Residuals of the hit x coord."); 
    stereoandmatchedMEs.meResxMatched->setAxisTitle("Res(x) in matched RecHit in " + label);
  }
  //ResyMatched
  if(switchResyMatched) {
    stereoandmatchedMEs.meResyMatched = bookME1D(ibooker,"TH1ResyMatched", hidmanager.createHistoLayer("Resy_matched","layer",label,"").c_str() ,"Residuals of the hit y coord."); 
    stereoandmatchedMEs.meResyMatched->setAxisTitle("Res(y) in matched RecHit in " + label);
  }
  //Chi2Matched
  if(switchChi2Matched) {
    stereoandmatchedMEs.meChi2Matched = bookME1D(ibooker,"TH1Chi2Matched", hidmanager.createHistoLayer("Chi2_matched","layer",label,"").c_str() ,"RecHit Chi2 test"); 
    stereoandmatchedMEs.meChi2Matched->setAxisTitle("Matched RecHit Chi2 test in " + label); 
  }
  //NsimHitMatched
  if(switchNsimHitMatched) {
    stereoandmatchedMEs.meNsimHitMatched = bookME1D(ibooker,"TH1NsimHitMatched", hidmanager.createHistoLayer("NsimHit_matched","layer",label,"").c_str() ,"No. of assoc. simHits"); 
    stereoandmatchedMEs.meNsimHitMatched->setAxisTitle("Number of assoc. simHits in " + label); 
  }

  StereoAndMatchedMEsMap[label]=stereoandmatchedMEs;
 
}
//------------------------------------------------------------------------------------------
void SiStripRecHitsValid::createSubDetMEs(DQMStore::IBooker & ibooker,std::string label) {

  SubDetMEs subdetMEs;
  subdetMEs.meNumrphi = nullptr;
  subdetMEs.meBunchrphi = nullptr;
  subdetMEs.meEventrphi = nullptr;
  subdetMEs.meNumStereo = nullptr;
  subdetMEs.meBunchStereo = nullptr;
  subdetMEs.meEventStereo = nullptr;
  subdetMEs.meNumMatched = nullptr;
  subdetMEs.meBunchMatched = nullptr;
  subdetMEs.meEventMatched = nullptr;

  std::string HistoName;
  //Numrphi
  if (switchNumrphi){
    HistoName = "TH1Numrphi__" + label;
    subdetMEs.meNumrphi = bookME1D(ibooker,"TH1Numrphi",HistoName.c_str(),"Num of RecHits");
    subdetMEs.meNumrphi->setAxisTitle("Total number of RecHits in "+ label);
  }  
  //Bunchrphi
  if(switchBunchrphi) {
    HistoName = "TH1Bunchrphi__" + label;
    subdetMEs.meBunchrphi = bookME1D(ibooker,"TH1Bunchrphi",HistoName.c_str(),"Bunch Crossing");
    subdetMEs.meBunchrphi->setAxisTitle("Bunch crossing in " + label); 
  }
  //Eventrphi
  if(switchEventrphi) {
    HistoName = "TH1Eventrphi__" + label;
    subdetMEs.meEventrphi = bookME1D(ibooker,"TH1Eventrphi",HistoName.c_str(),"Event (in-time bunch)");
    subdetMEs.meEventrphi->setAxisTitle("Event (in-time bunch) in " + label); 
  }
  //NumStereo
  if (switchNumStereo){
    HistoName = "TH1NumStereo__" + label;
    subdetMEs.meNumStereo = bookME1D(ibooker,"TH1NumStereo",HistoName.c_str(),"Num of RecHits in stereo modules");
    subdetMEs.meNumStereo->setAxisTitle("Total number of RecHits, stereo modules in "+ label);
  }  
  //BunchStereo
  if(switchBunchStereo) {
    HistoName = "TH1BunchStereo__" + label;
    subdetMEs.meBunchStereo = bookME1D(ibooker,"TH1BunchStereo",HistoName.c_str(),"Bunch Crossing");
    subdetMEs.meBunchStereo->setAxisTitle("Bunch crossing, stereo modules in " + label); 
  }
  //EventStereo
  if(switchEventStereo) {
    HistoName = "TH1EventStereo__" + label;
    subdetMEs.meEventStereo = bookME1D(ibooker,"TH1EventStereo",HistoName.c_str(),"Event (in-time bunch)");
    subdetMEs.meEventStereo->setAxisTitle("Event (in-time bunch), stereo modules in " + label); 
  }
  //NumMatched
  if (switchNumMatched){
    HistoName = "TH1NumMatched__" + label;
    subdetMEs.meNumMatched = bookME1D(ibooker,"TH1NumMatched",HistoName.c_str(),"Num of matched RecHits" );
    subdetMEs.meNumMatched->setAxisTitle("Total number of matched RecHits in "+ label);
  }  
  //BunchMatched
  if(switchBunchMatched) {
    HistoName = "TH1BunchMatched__" + label;
    subdetMEs.meBunchMatched = bookME1D(ibooker,"TH1BunchMatched",HistoName.c_str(),"Bunch Crossing");
    subdetMEs.meBunchMatched->setAxisTitle("Bunch crossing, matched RecHits in " + label); 
  }
  //EventMatched
  if(switchEventMatched) {
    HistoName = "TH1EventMatched__" + label;
    subdetMEs.meEventMatched = bookME1D(ibooker,"TH1EventMatched",HistoName.c_str(),"Event (in-time bunch)");
    subdetMEs.meEventMatched->setAxisTitle("Event (in-time bunch), matched RecHits in " + label); 
  }

  SubDetMEsMap[label]=subdetMEs;
}
//------------------------------------------------------------------------------------------
inline MonitorElement* SiStripRecHitsValid::bookME1D(DQMStore::IBooker & ibooker, const char* ParameterSetLabel, const char* HistoName, const char* HistoTitle)
{
  edm::ParameterSet parameters =  conf_.getParameter<edm::ParameterSet>(ParameterSetLabel);
  return ibooker.book1D(HistoName,HistoTitle,
			   parameters.getParameter<int32_t>("Nbinx"),
			   parameters.getParameter<double>("xmin"),
			   parameters.getParameter<double>("xmax")
			   );
}

//define this as a plug-in
DEFINE_FWK_MODULE(SiStripRecHitsValid);

