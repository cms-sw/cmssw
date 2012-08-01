// -*- C++ -*-
//
// Package:    HSCPDeDxInfoProducer
// Class:      HSCPDeDxInfoProducer
// 
/**\class HSCPDeDxInfoProducer HSCPDeDxInfoProducer.cc copy of RecoTracker/DeDx/src/DeDxDiscriminatorProducer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  loic Quertenmont (querten)
//         Created:  Thu Nov 17 14:09:02 CEST 2011
//

#include "SUSYBSMAnalysis/HSCP/interface/HSCPDeDxInfoProducer.h"

// system include files
#include <memory>
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/TrackReco/interface/DeDxData.h"
#include "AnalysisDataFormats/SUSYBSMObjects/interface/HSCPDeDxInfo.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "CondFormats/DataRecord/interface/SiStripDeDxMip_3D_Rcd.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxElectron_3D_Rcd.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxProton_3D_Rcd.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxPion_3D_Rcd.h"
#include "CondFormats/DataRecord/interface/SiStripDeDxKaon_3D_Rcd.h"


#include "TFile.h"

using namespace reco;
using namespace std;
using namespace edm;

HSCPDeDxInfoProducer::HSCPDeDxInfoProducer(const edm::ParameterSet& iConfig)
{

   produces<ValueMap<susybsm::HSCPDeDxInfo> >();

   m_tracksTag = iConfig.getParameter<edm::InputTag>("tracks");
   m_trajTrackAssociationTag   = iConfig.getParameter<edm::InputTag>("trajectoryTrackAssociation");

   usePixel = iConfig.getParameter<bool>("UsePixel"); 
   useStrip = iConfig.getParameter<bool>("UseStrip");
   if(!usePixel && !useStrip)
   edm::LogWarning("DeDxHitsProducer") << "Pixel Hits AND Strip Hits will not be used to estimate dEdx --> BUG, Please Update the config file";

   Formula             = iConfig.getUntrackedParameter<unsigned>("Formula"            ,  0);
   Reccord             = iConfig.getUntrackedParameter<string>  ("Reccord"            , "SiStripDeDxMip_3D_Rcd");
   ProbabilityMode     = iConfig.getUntrackedParameter<string>  ("ProbabilityMode"    , "Accumulation");


   MinTrackMomentum    = iConfig.getUntrackedParameter<double>  ("minTrackMomentum"   ,  0.0);
   MaxTrackMomentum    = iConfig.getUntrackedParameter<double>  ("maxTrackMomentum"   ,  99999.0); 
   MinTrackEta         = iConfig.getUntrackedParameter<double>  ("minTrackEta"        , -5.0);
   MaxTrackEta         = iConfig.getUntrackedParameter<double>  ("maxTrackEta"        ,  5.0);
   MaxNrStrips         = iConfig.getUntrackedParameter<unsigned>("maxNrStrips"        ,  255);
   MinTrackHits        = iConfig.getUntrackedParameter<unsigned>("MinTrackHits"       ,  3);

   shapetest           = iConfig.getParameter<bool>("ShapeTest");
   useCalibration      = iConfig.getParameter<bool>("UseCalibration");
   m_calibrationPath   = iConfig.getParameter<string>("calibrationPath");

   Prob_ChargePath = NULL;
}


HSCPDeDxInfoProducer::~HSCPDeDxInfoProducer(){}

// ------------ method called once each job just before starting event loop  ------------
void  HSCPDeDxInfoProducer::beginRun(edm::Run & run, const edm::EventSetup& iSetup)
{
   edm::ESHandle<PhysicsTools::Calibration::HistogramD3D> DeDxMapHandle_;    
   if(      strcmp(Reccord.c_str(),"SiStripDeDxMip_3D_Rcd")==0){
      iSetup.get<SiStripDeDxMip_3D_Rcd>() .get(DeDxMapHandle_);
   }else if(strcmp(Reccord.c_str(),"SiStripDeDxPion_3D_Rcd")==0){
      iSetup.get<SiStripDeDxPion_3D_Rcd>().get(DeDxMapHandle_);
   }else if(strcmp(Reccord.c_str(),"SiStripDeDxKaon_3D_Rcd")==0){
      iSetup.get<SiStripDeDxKaon_3D_Rcd>().get(DeDxMapHandle_);
   }else if(strcmp(Reccord.c_str(),"SiStripDeDxProton_3D_Rcd")==0){
      iSetup.get<SiStripDeDxProton_3D_Rcd>().get(DeDxMapHandle_);
   }else if(strcmp(Reccord.c_str(),"SiStripDeDxElectron_3D_Rcd")==0){
      iSetup.get<SiStripDeDxElectron_3D_Rcd>().get(DeDxMapHandle_);
   }else{
//      printf("The reccord %s is not known by the HSCPDeDxInfoProducer\n", Reccord.c_str());
//      printf("Program will exit now\n");
      exit(0);
   }
   DeDxMap_ = *DeDxMapHandle_.product();

   double xmin = DeDxMap_.rangeX().min;
   double xmax = DeDxMap_.rangeX().max;
   double ymin = DeDxMap_.rangeY().min;
   double ymax = DeDxMap_.rangeY().max;
   double zmin = DeDxMap_.rangeZ().min;
   double zmax = DeDxMap_.rangeZ().max;

   if(Prob_ChargePath)delete Prob_ChargePath;
   Prob_ChargePath  = new TH3D ("Prob_ChargePath"     , "Prob_ChargePath" , DeDxMap_.numberOfBinsX(), xmin, xmax, DeDxMap_.numberOfBinsY() , ymin, ymax, DeDxMap_.numberOfBinsZ(), zmin, zmax);

   

   if(strcmp(ProbabilityMode.c_str(),"Accumulation")==0){
//      printf("LOOOP ON P\n");
      for(int i=0;i<=Prob_ChargePath->GetXaxis()->GetNbins()+1;i++){
//         printf("LOOOP ON PATH\n");
         for(int j=0;j<=Prob_ChargePath->GetYaxis()->GetNbins()+1;j++){
//            printf("LOOOP ON CHARGE\n");

            double Ni = 0;
            for(int k=0;k<=Prob_ChargePath->GetZaxis()->GetNbins()+1;k++){ Ni+=DeDxMap_.binContent(i,j,k);} 

            for(int k=0;k<=Prob_ChargePath->GetZaxis()->GetNbins()+1;k++){
               double tmp = 0;
               for(int l=0;l<=k;l++){ tmp+=DeDxMap_.binContent(i,j,l);}

      	       if(Ni>0){
                  Prob_ChargePath->SetBinContent (i, j, k, tmp/Ni);
// 	          printf("P=%6.2f Path=%6.2f Charge%8.2f --> Prob=%8.3f\n",Prob_ChargePath->GetXaxis()->GetBinCenter(i), Prob_ChargePath->GetYaxis()->GetBinCenter(j), Prob_ChargePath->GetZaxis()->GetBinCenter(k), tmp/Ni);
  	       }else{
                  Prob_ChargePath->SetBinContent (i, j, k, 0);
	       }
            }
         }
      }
   }else if(strcmp(ProbabilityMode.c_str(),"Integral")==0){
//      printf("LOOOP ON P\n");
      for(int i=0;i<=Prob_ChargePath->GetXaxis()->GetNbins()+1;i++){
//         printf("LOOOP ON PATH\n");
         for(int j=0;j<=Prob_ChargePath->GetYaxis()->GetNbins()+1;j++){
//            printf("LOOOP ON CHARGE\n");

            double Ni = 0;
            for(int k=0;k<=Prob_ChargePath->GetZaxis()->GetNbins()+1;k++){ Ni+=DeDxMap_.binContent(i,j,k);}

            for(int k=0;k<=Prob_ChargePath->GetZaxis()->GetNbins()+1;k++){
               double tmp = DeDxMap_.binContent(i,j,k);

               if(Ni>0){
                  Prob_ChargePath->SetBinContent (i, j, k, tmp/Ni);
//                  printf("P=%6.2f Path=%6.2f Charge%8.2f --> Prob=%8.3f\n",Prob_ChargePath->GetXaxis()->GetBinCenter(i), Prob_ChargePath->GetYaxis()->GetBinCenter(j), Prob_ChargePath->GetZaxis()->GetBinCenter(k), tmp/Ni);
               }else{
                  Prob_ChargePath->SetBinContent (i, j, k, 0);
               }
            }
         }
      }   
   }else{
//      printf("The ProbabilityMode: %s is unknown\n",ProbabilityMode.c_str());
//      printf("The program will stop now\n");
      exit(0);
   }



/*
   for(int i=0;i<Prob_ChargePath->GetXaxis()->GetNbins();i++){
      for(int j=0;j<Prob_ChargePath->GetYaxis()->GetNbins();j++){
         double tmp = DeDxMap_.binContent(i,j);
         Prob_ChargePath->SetBinContent (i, j, tmp);
	 printf("%i %i --> %f\n",i,j,tmp);
      }
   }
*/



   if(MODsColl.size()==0){
      edm::ESHandle<TrackerGeometry> tkGeom;
      iSetup.get<TrackerDigiGeometryRecord>().get( tkGeom );
      m_tracker = tkGeom.product();

      vector<GeomDet*> Det = tkGeom->dets();
      for(unsigned int i=0;i<Det.size();i++){
         DetId  Detid  = Det[i]->geographicalId();
         int    SubDet = Detid.subdetId();

         if( SubDet == StripSubdetector::TIB ||  SubDet == StripSubdetector::TID ||
             SubDet == StripSubdetector::TOB ||  SubDet == StripSubdetector::TEC  ){

             StripGeomDetUnit* DetUnit     = dynamic_cast<StripGeomDetUnit*> (Det[i]);
             if(!DetUnit)continue;

             const StripTopology& Topo     = DetUnit->specificTopology();
             unsigned int         NAPV     = Topo.nstrips()/128;

             double Eta     = DetUnit->position().basicVector().eta();
             double R       = DetUnit->position().basicVector().transverse();
             double Thick   = DetUnit->surface().bounds().thickness();


             const TrapezoidalPlaneBounds* trapezoidalBounds( dynamic_cast<const TrapezoidalPlaneBounds*>(&(DetUnit->surface().bounds())));


             stModInfo* MOD = new stModInfo;
             MOD->DetId     = Detid.rawId();
             MOD->SubDet    = SubDet;
             MOD->Eta       = Eta;
             MOD->R         = R;
             MOD->Thickness = Thick;
             MOD->Width     = DetUnit->surface().bounds().width();
             MOD->Length    = DetUnit->surface().bounds().length();
             if(trapezoidalBounds!=NULL){
                std::vector<float> const & parameters = (*trapezoidalBounds).parameters();
                for(unsigned int p=0;p<parameters.size();p++)MOD->trapezoParams .push_back(parameters[p]);
             }
             MOD->NAPV      = NAPV;
             MODsColl[MOD->DetId] = MOD;
         }else{
            PixelGeomDetUnit* DetUnit = dynamic_cast<PixelGeomDetUnit*> (Det[i]);
             if(!DetUnit)continue;

             stModInfo* MOD = new stModInfo;
             MOD->DetId     = Detid.rawId();
             MOD->SubDet    = SubDet;
             MOD->Eta       = DetUnit->position().basicVector().eta();
             MOD->R         = DetUnit->position().basicVector().transverse();
             MOD->Thickness = DetUnit->surface().bounds().thickness();
             MOD->Width     = DetUnit->surface().bounds().width();
             MOD->Length    = DetUnit->surface().bounds().length();
             MOD->NAPV      = -1;
             MODsColl[MOD->DetId] = MOD;
         }
      }
 
      MakeCalibrationMap();
   }

}

void  HSCPDeDxInfoProducer::endJob()
{
   MODsColl.clear();
}



void HSCPDeDxInfoProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  auto_ptr<ValueMap<susybsm::HSCPDeDxInfo> > trackDeDxDiscrimAssociation(new ValueMap<susybsm::HSCPDeDxInfo> );  
  ValueMap<susybsm::HSCPDeDxInfo>::Filler filler(*trackDeDxDiscrimAssociation);

  Handle<TrajTrackAssociationCollection> trajTrackAssociationHandle;
  iEvent.getByLabel(m_trajTrackAssociationTag, trajTrackAssociationHandle);
  const TrajTrackAssociationCollection TrajToTrackMap = *trajTrackAssociationHandle.product();

  edm::Handle<reco::TrackCollection> trackCollectionHandle;
  iEvent.getByLabel(m_tracksTag,trackCollectionHandle);

   edm::ESHandle<TrackerGeometry> tkGeom;
   iSetup.get<TrackerDigiGeometryRecord>().get( tkGeom );
   m_tracker = tkGeom.product();
 
   std::vector<susybsm::HSCPDeDxInfo> dEdxInfos( TrajToTrackMap.size() );

   unsigned track_index = 0;
   for(TrajTrackAssociationCollection::const_iterator it = TrajToTrackMap.begin(); it!=TrajToTrackMap.end(); ++it, track_index++) {
      dEdxInfos[track_index] = susybsm::HSCPDeDxInfo();

      const Track      track = *it->val;
      const Trajectory traj  = *it->key;

      if(track.eta()  <MinTrackEta      || track.eta()>MaxTrackEta     ){continue;}
      if(track.p()    <MinTrackMomentum || track.p()  >MaxTrackMomentum){continue;}
      if(track.found()<MinTrackHits                                    ){continue;}

      std::vector<double> vect_probs;
      vector<TrajectoryMeasurement> measurements = traj.measurements();

      susybsm::HSCPDeDxInfo hscpDeDxInfo;
      for(vector<TrajectoryMeasurement>::const_iterator measurement_it = measurements.begin(); measurement_it!=measurements.end(); measurement_it++){

         TrajectoryStateOnSurface trajState = measurement_it->updatedState();
         if( !trajState.isValid() ) continue;

         const TrackingRecHit*         hit                 = (*measurement_it->recHit()).hit();
         const SiStripRecHit2D*        sistripsimplehit    = dynamic_cast<const SiStripRecHit2D*>(hit);
         const SiStripMatchedRecHit2D* sistripmatchedhit   = dynamic_cast<const SiStripMatchedRecHit2D*>(hit);
         const SiStripRecHit1D*        sistripsimple1dhit  = dynamic_cast<const SiStripRecHit1D*>(hit);
         const SiPixelRecHit*          pixelHit            = dynamic_cast<const SiPixelRecHit*>(hit);
	 
         if(sistripsimplehit){
                FillInfo(DeDxTools::GetCluster(sistripsimplehit), trajState,sistripsimplehit->geographicalId(), hscpDeDxInfo);       
                FillPosition(trajState,sistripsimplehit->geographicalId(),hscpDeDxInfo);
  	        hscpDeDxInfo.shapetest.push_back(DeDxTools::shapeSelection(DeDxTools::GetCluster(sistripsimplehit)->amplitudes()));
         }else if(sistripmatchedhit){
                FillInfo(DeDxTools::GetCluster(sistripmatchedhit->monoHit()), trajState, sistripmatchedhit->monoId(), hscpDeDxInfo);
                FillPosition(trajState,sistripmatchedhit->monoId(),hscpDeDxInfo);
	        hscpDeDxInfo.shapetest.push_back(DeDxTools::shapeSelection(DeDxTools::GetCluster(sistripmatchedhit->monoHit())->amplitudes()));
           
                FillInfo(DeDxTools::GetCluster(sistripmatchedhit->stereoHit()), trajState,sistripmatchedhit->stereoId(), hscpDeDxInfo);
                FillPosition(trajState,sistripmatchedhit->stereoId(),hscpDeDxInfo);
                hscpDeDxInfo.shapetest.push_back(true);
         }else if(sistripsimple1dhit){ 
	        FillInfo(DeDxTools::GetCluster(sistripsimple1dhit), trajState, sistripsimple1dhit->geographicalId(), hscpDeDxInfo);
                FillPosition(trajState,sistripsimple1dhit->geographicalId(),hscpDeDxInfo);
                hscpDeDxInfo.shapetest.push_back(DeDxTools::shapeSelection(DeDxTools::GetCluster(sistripsimple1dhit)->amplitudes()));
         }else if(pixelHit){
                double cosine = trajState.localDirection().z() / trajState.localDirection().mag();
                stModInfo* MOD = MODsColl[pixelHit->geographicalId()];
                hscpDeDxInfo.charge.push_back(pixelHit->cluster()->charge());
                hscpDeDxInfo.chargeUnSat.push_back(pixelHit->cluster()->charge());
                hscpDeDxInfo.probability.push_back(pixelHit->probabilityQ());
                hscpDeDxInfo.pathlength.push_back(MOD->Thickness/std::abs(cosine));
                hscpDeDxInfo.cosine.push_back(cosine);
                hscpDeDxInfo.detIds.push_back(pixelHit->geographicalId());
                hscpDeDxInfo.shapetest.push_back(false);
                FillPosition(trajState,pixelHit->geographicalId(),hscpDeDxInfo);
         }else{
         }
      }

      dEdxInfos[track_index] = hscpDeDxInfo;
   }

  filler.insert(trackCollectionHandle, dEdxInfos.begin(), dEdxInfos.end());
  filler.fill();
  iEvent.put(trackDeDxDiscrimAssociation);
}


void HSCPDeDxInfoProducer::FillPosition(TrajectoryStateOnSurface trajState,  const uint32_t &  detId, susybsm::HSCPDeDxInfo& hscpDeDxInfo)
{
   stModInfo* MOD          = MODsColl[detId];
   LocalPoint HitLocalPos  = trajState.localPosition();

   double Width      = MOD->Width;
   double Length     = MOD->Length;

   if(MOD->trapezoParams.size()>0){
      Length     = MOD->trapezoParams[3]*2;
      double t   = 0.5 + HitLocalPos.y()/Length ;
      Width      = 2* (MOD->trapezoParams[0] + (MOD->trapezoParams[1]-MOD->trapezoParams[0]) * t);
   }

   hscpDeDxInfo.modwidth.push_back(Width);
   hscpDeDxInfo.modlength.push_back(Length);
   hscpDeDxInfo.localx.push_back(HitLocalPos.x());
   hscpDeDxInfo.localy.push_back(HitLocalPos.y());
}


void HSCPDeDxInfoProducer::FillInfo(const SiStripCluster*   cluster, TrajectoryStateOnSurface trajState,  const uint32_t &  detId, susybsm::HSCPDeDxInfo& hscpDeDxInfo)
{
   // Get All needed variables
   LocalVector             trackDirection = trajState.localDirection();
   double                  cosine         = trackDirection.z()/trackDirection.mag();
   const vector<uint8_t>&  ampls          = cluster->amplitudes();
   stModInfo* MOD                         = MODsColl[detId];


   // Sanity Checks
   if( ampls.size()>MaxNrStrips)                                                                      {return;}

   // Find Probability for this given Charge and Path
   double charge = 0;
   if(useCalibration){
      for(unsigned int i=0;i<ampls.size();i++){
         int CalibratedCharge = ampls[i];
         CalibratedCharge = (int)(CalibratedCharge / MOD->Gain);
         if(CalibratedCharge>=1024){
            CalibratedCharge = 255;
         }else if(CalibratedCharge>254){
            CalibratedCharge = 254;
         }
         charge+=CalibratedCharge;
      }
   }else{
      charge = DeDxDiscriminatorTools::charge(ampls);
   }
   double path   = DeDxDiscriminatorTools::path(cosine,MOD->Thickness);

   int    BinX   = Prob_ChargePath->GetXaxis()->FindBin(trajState.localMomentum().mag());
   int    BinY   = Prob_ChargePath->GetYaxis()->FindBin(path);
   int    BinZ   = Prob_ChargePath->GetZaxis()->FindBin(charge/path);
   double Prob   = Prob_ChargePath->GetBinContent(BinX,BinY,BinZ);

   hscpDeDxInfo.charge.push_back(charge);
   hscpDeDxInfo.chargeUnSat.push_back(charge);
   hscpDeDxInfo.probability.push_back(Prob);
   hscpDeDxInfo.pathlength.push_back(path);
   hscpDeDxInfo.cosine.push_back(cosine);
   hscpDeDxInfo.detIds.push_back(detId);
}

void HSCPDeDxInfoProducer::MakeCalibrationMap(){
   if(!useCalibration)return;


   TChain* t1 = new TChain("SiStripCalib/APVGain");
   t1->Add(m_calibrationPath.c_str());

   unsigned int  tree_DetId;
   unsigned char tree_APVId;
   double        tree_Gain;

   t1->SetBranchAddress("DetId"             ,&tree_DetId      );
   t1->SetBranchAddress("APVId"             ,&tree_APVId      );
   t1->SetBranchAddress("Gain"              ,&tree_Gain       );

   for (unsigned int ientry = 0; ientry < t1->GetEntries(); ientry++) {
       t1->GetEntry(ientry);
       stModInfo* MOD  = MODsColl[tree_DetId];
       MOD->Gain = tree_Gain;
   }

   delete t1;

}

//define this as a plug-in
DEFINE_FWK_MODULE(HSCPDeDxInfoProducer);
