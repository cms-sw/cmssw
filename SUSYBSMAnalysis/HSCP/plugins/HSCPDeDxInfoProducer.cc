// -*- C++ -*-
//
// Package:    HSCPDeDxInfoProducer
// Class:      HSCPDeDxInfoProducer
// 
/**\class HSCPDeDxInfoProducer HSCPDeDxInfoProducer.cc RecoTracker/HSCPDeDxInfoProducer/src/HSCPDeDxInfoProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  andrea
//         Created:  Thu May 31 14:09:02 CEST 2007
//    Code Updates:  loic Quertenmont (querten)
//         Created:  Thu May 10 14:09:02 CEST 2008
//
//

#include "SUSYBSMAnalysis/HSCP/plugins/HSCPDeDxInfoProducer.h"

// system include files


using namespace reco;
using namespace std;
using namespace edm;

HSCPDeDxInfoProducer::HSCPDeDxInfoProducer(const edm::ParameterSet& iConfig)
{

   produces<ValueMap<susybsm::HSCPDeDxInfo> >();


   MaxNrStrips         = iConfig.getUntrackedParameter<unsigned>("maxNrStrips"        ,  255);
   MinTrackHits        = iConfig.getUntrackedParameter<unsigned>("MinTrackHits"       ,  4);
   MinTrackMomentum    = iConfig.getUntrackedParameter<double>  ("minTrackMomentum"   ,  0.0);
   MaxTrackMomentum    = iConfig.getUntrackedParameter<double>  ("maxTrackMomentum"   ,  99999.0); 
   MinTrackEta         = iConfig.getUntrackedParameter<double>  ("minTrackEta"        , -5.0);
   MaxTrackEta         = iConfig.getUntrackedParameter<double>  ("maxTrackEta"        ,  5.0);

   m_tracksTag = consumes<reco::TrackCollection>(iConfig.getParameter<edm::InputTag>("tracks"));
   m_trajTrackAssociationTag   = consumes<TrajTrackAssociationCollection>(iConfig.getParameter<edm::InputTag>("trajectoryTrackAssociation"));
   useTrajectory = iConfig.getUntrackedParameter<bool>("UseTrajectory", true);

   usePixel = iConfig.getParameter<bool>("UsePixel"); 
   useStrip = iConfig.getParameter<bool>("UseStrip");
   meVperADCPixel = iConfig.getParameter<double>("MeVperADCPixel"); 
   meVperADCStrip = iConfig.getParameter<double>("MeVperADCStrip"); 

   shapetest = iConfig.getParameter<bool>("ShapeTest");
   useCalibration = iConfig.getParameter<bool>("UseCalibration");
   m_calibrationPath = iConfig.getParameter<string>("calibrationPath");

//   Reccord             = iConfig.getUntrackedParameter<std::string>  ("Reccord"            , "SiStripDeDxMip_3D_Rcd");
//   ProbabilityMode     = iConfig.getUntrackedParameter<std::string>  ("ProbabilityMode"    , "Accumulation");
//   Prob_ChargePath     = NULL;

   if(!usePixel && !useStrip)
   edm::LogWarning("DeDxHitsProducer") << "Pixel Hits AND Strip Hits will not be used to estimate dEdx --> BUG, Please Update the config file";
}


HSCPDeDxInfoProducer::~HSCPDeDxInfoProducer(){}

// ------------ method called once each job just before starting event loop  ------------
void  HSCPDeDxInfoProducer::beginRun(edm::Run const& run, const edm::EventSetup& iSetup)
{
   if(useCalibration && calibGains.size()==0){
      edm::ESHandle<TrackerGeometry> tkGeom;
      iSetup.get<TrackerDigiGeometryRecord>().get( tkGeom );
      m_off = tkGeom->offsetDU(GeomDetEnumerators::PixelBarrel); //index start at the first pixel

      DeDxTools::makeCalibrationMap(m_calibrationPath, *tkGeom, calibGains, m_off);
   }

//   DeDxTools::buildDiscrimMap(run, iSetup, Reccord,  ProbabilityMode, Prob_ChargePath);
}



void HSCPDeDxInfoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  auto_ptr<ValueMap<susybsm::HSCPDeDxInfo> > trackDeDxAssociation(new ValueMap<susybsm::HSCPDeDxInfo> );
  ValueMap<susybsm::HSCPDeDxInfo>::Filler filler(*trackDeDxAssociation);

  edm::Handle<reco::TrackCollection> trackCollectionHandle;
  iEvent.getByToken(m_tracksTag,trackCollectionHandle);

  Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
  if(useTrajectory)iEvent.getByToken(m_trajTrackAssociationTag, trajTrackAssociationHandle);

  std::vector<susybsm::HSCPDeDxInfo> dEdxInfos( trackCollectionHandle->size() );

  TrajTrackAssociationCollection::const_iterator cit;
  if(useTrajectory)cit = trajTrackAssociationHandle->begin();
  for(unsigned int j=0;j<trackCollectionHandle->size();j++){            
     const reco::TrackRef track = reco::TrackRef( trackCollectionHandle.product(), j );

     susybsm::HSCPDeDxInfo hscpDeDxInfo;
 
     if(useTrajectory){  //trajectory allows to take into account the local direction of the particle on the module sensor --> muc much better 'dx' measurement
        const edm::Ref<std::vector<Trajectory> > traj = cit->key; cit++;
        const vector<TrajectoryMeasurement> & measurements = traj->measurements();
        for(vector<TrajectoryMeasurement>::const_iterator it = measurements.begin(); it!=measurements.end(); it++){
           TrajectoryStateOnSurface trajState=it->updatedState();
           if( !trajState.isValid()) continue;
     
           const TrackingRecHit * recHit=(*it->recHit()).hit();
           if(!recHit)continue;
           LocalVector trackDirection = trajState.localDirection();
           float cosine = trackDirection.z()/trackDirection.mag();

           processHit(recHit, trajState.localMomentum().mag(), cosine, hscpDeDxInfo, trajState.localPosition());
        }

     }else{ //assume that the particles trajectory is a straight line originating from the center of the detector  (can be improved)
        for(unsigned int h=0;h<track->recHitsSize();h++){
           const TrackingRecHit* recHit = &(*(track->recHit(h)));
           auto const & thit = static_cast<BaseTrackerRecHit const&>(*recHit);
           if(!thit.isValid())continue;//make sure it's a tracker hit

           const GlobalVector& ModuleNormal = recHit->detUnit()->surface().normalVector();         
           float cosine = (track->px()*ModuleNormal.x()+track->py()*ModuleNormal.y()+track->pz()*ModuleNormal.z())/track->p();

           processHit(recHit, track->p(), cosine, hscpDeDxInfo, LocalPoint(0.0,0.0));
        } 
     }

     dEdxInfos[j] = hscpDeDxInfo;
  }
  ///////////////////////////////////////
  
  filler.insert(trackCollectionHandle, dEdxInfos.begin(), dEdxInfos.end());
  filler.fill();
  iEvent.put(trackDeDxAssociation);   
}

void HSCPDeDxInfoProducer::processHit(const TrackingRecHit* recHit, float trackMomentum, float& cosine, susybsm::HSCPDeDxInfo& hscpDeDxInfo,  LocalPoint HitLocalPos){
      auto const & thit = static_cast<BaseTrackerRecHit const&>(*recHit);
      if(!thit.isValid())return;

      auto const & clus = thit.firstClusterRef();
      if(!clus.isValid())return;

      if(clus.isPixel()){
          if(!usePixel) return;

          auto& detUnit     = *(recHit->detUnit());
          float pathLen     = detUnit.surface().bounds().thickness()/fabs(cosine);
          float chargeAbs   = clus.pixelCluster().charge();
          hscpDeDxInfo.charges.push_back(chargeAbs);
          hscpDeDxInfo.pathlengths.push_back(pathLen);
          hscpDeDxInfo.detIds.push_back(thit.geographicalId());
          hscpDeDxInfo.localPosXs.push_back(HitLocalPos.x());
          hscpDeDxInfo.localPosYs.push_back(HitLocalPos.y());
          hscpDeDxInfo.clusterIndices.push_back(clus.key());
       }else if(clus.isStrip() && !thit.isMatched()){
          if(!useStrip) return;

          auto& detUnit     = *(recHit->detUnit());
          int   NSaturating = 0;
          float pathLen     = detUnit.surface().bounds().thickness()/fabs(cosine);
          float chargeAbs   = DeDxTools::getCharge(&(clus.stripCluster()),NSaturating, detUnit, calibGains, m_off);
          hscpDeDxInfo.charges.push_back(chargeAbs);
          hscpDeDxInfo.pathlengths.push_back(pathLen);
          hscpDeDxInfo.detIds.push_back(thit.geographicalId());
          hscpDeDxInfo.localPosXs.push_back(HitLocalPos.x());
          hscpDeDxInfo.localPosYs.push_back(HitLocalPos.y());
          hscpDeDxInfo.clusterIndices.push_back(clus.key());
       }else if(clus.isStrip() && thit.isMatched()){
          if(!useStrip) return;
          const SiStripMatchedRecHit2D* matchedHit=dynamic_cast<const SiStripMatchedRecHit2D*>(recHit);
          if(!matchedHit)return;

          auto& detUnitM     = *(matchedHit->monoHit().detUnit());
          int   NSaturating = 0;
          float pathLen     = detUnitM.surface().bounds().thickness()/fabs(cosine);
          float chargeAbs   = DeDxTools::getCharge(&(matchedHit->monoHit().stripCluster()),NSaturating, detUnitM, calibGains, m_off);
          hscpDeDxInfo.charges.push_back(chargeAbs);
          hscpDeDxInfo.pathlengths.push_back(pathLen);
          hscpDeDxInfo.detIds.push_back(thit.geographicalId());
          hscpDeDxInfo.localPosXs.push_back(HitLocalPos.x());
          hscpDeDxInfo.localPosYs.push_back(HitLocalPos.y());
          const OmniClusterRef monoClusterRef = matchedHit->monoClusterRef();
          hscpDeDxInfo.clusterIndices.push_back(monoClusterRef.key());

          auto& detUnitS     = *(matchedHit->stereoHit().detUnit());
          NSaturating = 0;
          pathLen     = detUnitS.surface().bounds().thickness()/fabs(cosine);
          chargeAbs   = DeDxTools::getCharge(&(matchedHit->stereoHit().stripCluster()),NSaturating, detUnitS, calibGains, m_off);
          hscpDeDxInfo.charges.push_back(chargeAbs);
          hscpDeDxInfo.pathlengths.push_back(pathLen);
          hscpDeDxInfo.detIds.push_back(thit.geographicalId());
          hscpDeDxInfo.localPosXs.push_back(HitLocalPos.x());
          hscpDeDxInfo.localPosYs.push_back(HitLocalPos.y());
          const OmniClusterRef stereoClusterRef = matchedHit->stereoClusterRef();
          hscpDeDxInfo.clusterIndices.push_back(stereoClusterRef.key());
       }
}



//define this as a plug-in
DEFINE_FWK_MODULE(HSCPDeDxInfoProducer);
