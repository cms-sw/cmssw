// -*- C++ -*-
//
// Package:    trackJet/JetCorePerfectSeedGenerator
// Class:      JetCorePerfectSeedGenerator
//
/**\class JetCorePerfectSeedGenerator JetCorePerfectSeedGenerator.cc trackJet/JetCorePerfectSeedGenerator/plugins/JetCorePerfectSeedGenerator.cc
 Description: [one line class summary]
 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Valerio Bertacchi
//         Created:  Mon, 18 Dec 2017 16:35:04 GMT
//
//


// system include files

#define jetDimX 30
#define jetDimY 30
#define Nlayer 4
#define Nover 3
#define Npar 5

#include "JetCorePerfectSeedGenerator.h"

#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSet.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/SiPixelDigi/interface/PixelDigi.h"
#include "DataFormats/GeometryVector/interface/VectorUtil.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"


#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"

#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"

#include "TrackingTools/GeomPropagators/interface/StraightLinePlaneCrossing.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"



#include <boost/range.hpp>
#include <boost/foreach.hpp>
#include "boost/multi_array.hpp"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

// #include "SimG4Core/Application/interface/G4SimTrack.h"
#include "SimDataFormats/Track/interface/SimTrack.h"
#include "SimDataFormats/Vertex/interface/SimVertex.h"


#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"


#include "TTree.h"


//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<> and also remove the line from
// constructor "usesResource("TFileService");"
// This will improve performance in multithreaded jobs.



JetCorePerfectSeedGenerator::JetCorePerfectSeedGenerator(const edm::ParameterSet& iConfig) :

      vertices_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      pixelClusters_(consumes<edmNew::DetSetVector<SiPixelCluster> >(iConfig.getParameter<edm::InputTag>("pixelClusters"))),
      cores_(consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("cores"))),
      simtracksToken(consumes<std::vector<SimTrack> >(iConfig.getParameter<edm::InputTag>("simTracks"))),
      simvertexToken(consumes<std::vector<SimVertex> >(iConfig.getParameter<edm::InputTag>("simVertex"))),
      PSimHitToken(consumes<std::vector<PSimHit> >(iConfig.getParameter<edm::InputTag>("simHit"))),
      ptMin_(iConfig.getParameter<double>("ptMin")),
      deltaR_(iConfig.getParameter<double>("deltaR")),
      chargeFracMin_(iConfig.getParameter<double>("chargeFractionMin")),
      centralMIPCharge_(iConfig.getParameter<double>("centralMIPCharge")),
      pixelCPE_(iConfig.getParameter<std::string>("pixelCPE"))

{
  produces<TrajectorySeedCollection>();
  produces<reco::TrackCollection>();



  //  edm::Service<TFileService> fileService;
  //  JetCorePerfectSeedGeneratorTree= fileService->make<TTree>("JetCorePerfectSeedGeneratorTree","JetCorePerfectSeedGeneratorTree");
  //  JetCorePerfectSeedGeneratorTree->Branch("cluster_measured",clusterMeas,"cluster_measured[30][30][4]/D");
  //  JetCorePerfectSeedGeneratorTree->Branch("jet_eta",&jet_eta);
  //  JetCorePerfectSeedGeneratorTree->Branch("jet_pt",&jet_pt);

    //  for(int i=0; i<Nlayer; i++){ //NOFLAG
    //    for(int j=0; j<jetDimX; j++){
    //      for(int k=0; k<jetDimY; k++){
    //        if(j<jetDimX && k<jetDimY && i< Nlayer) clusterMeas[j][k][i] = 0.0;
    //       }
    //      }
    //   }



}


JetCorePerfectSeedGenerator::~JetCorePerfectSeedGenerator()
{

}

#define foreach BOOST_FOREACH


void JetCorePerfectSeedGenerator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  evt_counter++;
  // std::cout << "NEW EVENT, event number" << evt_counter  <<std::endl;


  auto result = std::make_unique<TrajectorySeedCollection>();
  auto resultTracks = std::make_unique<reco::TrackCollection>();

  // evt_counter++;


  using namespace edm;
  using namespace reco;


  iSetup.get<IdealMagneticFieldRecord>().get( magfield_ );
  iSetup.get<GlobalTrackingGeometryRecord>().get(geometry_);
  iSetup.get<TrackingComponentsRecord>().get( "AnalyticalPropagator", propagator_ );

  iEvent.getByToken(pixelClusters_, inputPixelClusters);
  allSiPixelClusters.clear(); siPixelDetsWithClusters.clear();
  allSiPixelClusters.reserve(inputPixelClusters->dataSize()); // this is important, otherwise push_back invalidates the iterators

  edm::Handle<std::vector<SimTrack> > simtracks;
  iEvent.getByToken(simtracksToken, simtracks);
  edm::Handle<std::vector<SimVertex> > simvertex;
  iEvent.getByToken(simvertexToken, simvertex);

  iEvent.getByToken(PSimHitToken, simhits);

  Handle<std::vector<reco::Vertex> > vertices;
  iEvent.getByToken(vertices_, vertices);

  Handle<edm::View<reco::Candidate> > cores;
  iEvent.getByToken(cores_, cores);

  // iEvent.getByToken(pixeldigisimlinkToken, pixeldigisimlink);

  //--------------------------debuging lines ---------------------//
  edm::ESHandle<PixelClusterParameterEstimator> pe;
  const PixelClusterParameterEstimator* pp;
  iSetup.get<TkPixelCPERecord>().get(pixelCPE_, pe);
  pp = pe.product();
  //--------------------------end ---------------------//

  edm::ESHandle<TrackerTopology> tTopoHandle;
  iSetup.get<TrackerTopologyRcd>().get(tTopoHandle);
  const TrackerTopology* const tTopo = tTopoHandle.product();

  auto output = std::make_unique<edmNew::DetSetVector<SiPixelCluster>>();

print = false;
int jet_number = 0;
int seed_number = 0;


  for (unsigned int ji = 0; ji < cores->size(); ji++) { //loop jet
    jet_number++;

    if ((*cores)[ji].pt() > ptMin_) {
//       std::cout << "|____________________NEW JET_______________________________| jet number=" << jet_number  << " " << (*cores)[ji].pt() << " " << (*cores)[ji].eta() << " " << (*cores)[ji].phi() <<  std::endl;

      std::set<long long int> ids;
      const reco::Candidate& jet = (*cores)[ji];
      const reco::Vertex& jetVertex = (*vertices)[0];

      std::vector<GlobalVector> splitClustDirSet = splittedClusterDirections(jet, tTopo, pp, jetVertex, 1);
      //std::vector<GlobalVector> splitClustDirSet = splittedClusterDirections(jet, tTopo, pp, jetVertex);
      // bool l2off=(splitClustDirSet.size()==0);
      if(splitClustDirSet.size()==0) {//if layer 1 is broken find direcitons on layer 2
        splitClustDirSet = splittedClusterDirections(jet, tTopo, pp, jetVertex, 2);
        // std::cout << "split on lay2, in numero=" << splitClustDirSet.size() << "+jetDir" << std::endl;
      }
      if(inclusiveConeSeed) splitClustDirSet.clear();
      splitClustDirSet.push_back(GlobalVector(jet.px(),jet.py(),jet.pz()));
  //    std::cout << "splitted cluster number=" << splitClustDirSet.size() << std::endl;;

      for(int cc=0; cc<(int)splitClustDirSet.size(); cc++){


      GlobalVector bigClustDir = splitClustDirSet.at(cc);

      const auto & simtracksVector = simtracks.product();
      const auto & simvertexVector = simvertex.product();

      LocalPoint jetInter(0,0,0);

      jet_eta = jet.eta();
      jet_pt = jet.pt();

      auto jetVert = jetVertex; //trackInfo filling

      std::vector<PSimHit> goodSimHit;

      // edmNew::DetSetVector<SiPixelCluster>::const_iterator detIt = inputPixelClusters->begin();

      const GeomDet* globDet = DetectorSelector(2, jet, bigClustDir, jetVertex, tTopo); //select detector mostly hitten by the jet

      if(globDet == 0) continue;

      // const GeomDet* goodDet1 = DetectorSelector(1, jet, bigClustDir, jetVertex, tTopo);
      // const GeomDet* goodDet3 = DetectorSelector(3, jet, bigClustDir, jetVertex, tTopo);
      // const GeomDet* goodDet4 = DetectorSelector(4, jet, bigClustDir, jetVertex, tTopo);



      // for (; detIt != inputPixelClusters->end(); detIt++) { //loop deset //COMMENTATO DA QUIIIII
      //   const edmNew::DetSet<SiPixelCluster>& detset = *detIt;
      //   const GeomDet* det = geometry_->idToDet(detset.id()); //lui sa il layer con cast a  PXBDetId (vedi dentro il layer function)
      //
      //   for (auto cluster = detset.begin(); cluster != detset.end(); cluster++) { //loop cluster
      //
      //     const SiPixelCluster& aCluster = *cluster;
      //     det_id_type aClusterID= detset.id();
      //     if(DetId(aClusterID).subdetId()!=1) continue;
      //
      //     int lay = tTopo->layer(det->geographicalId());
      //
      //     std::pair<bool, Basic3DVector<float>> interPair = findIntersection(bigClustDir,(reco::Candidate::Point)jetVertex.position(), det);
      //     if(interPair.first==false) continue;
      //     Basic3DVector<float> inter = interPair.second;
      //     auto localInter = det->specificSurface().toLocal((GlobalPoint)inter);
      //
      //     GlobalPoint pointVertex(jetVertex.position().x(), jetVertex.position().y(), jetVertex.position().z());
      //
      //
      //     // GlobalPoint cPos = det->surface().toGlobal(pp->localParametersV(aCluster,(*geometry_->idToDetUnit(detIt->id())))[0].first);
      //     LocalPoint cPos_local = pp->localParametersV(aCluster,(*geometry_->idToDetUnit(detIt->id())))[0].first;
      //
      //     if(std::abs(cPos_local.x()-localInter.x())/pitchX<=jetDimX/2 && std::abs(cPos_local.y()-localInter.y())/pitchY<=jetDimY/2){ // per ora preso baricentro, da migliorare
      //
      //       if(det==goodDet1 || det==goodDet3 || det==goodDet4 || det==globDet) {
      //         // fillPixelMatrix(aCluster,lay,localInter, det, input_matrix_cluster);
      //         fillPixelMatrix(aCluster,lay,localInter, det, input_tensors);
      //         }
      //     } //cluster in ROI
      //   } //cluster
      // } //detset
      std::pair<std::vector<SimTrack>,std::vector<SimVertex>> goodSimTkVx;

      if(inclusiveConeSeed) {
        auto jetVert = jetVertex;
        goodSimTkVx = JetCorePerfectSeedGenerator::coreTracksFillingDeltaR(simtracksVector, simvertexVector,globDet,jet,jetVert );
       }
      else {
        std::vector<PSimHit> goodSimHit = JetCorePerfectSeedGenerator::coreHitsFilling(simhits,globDet,bigClustDir,jetVertex);
        goodSimTkVx = JetCorePerfectSeedGenerator::coreTracksFilling(goodSimHit,simtracksVector, simvertexVector);
      }
      seed_number = goodSimTkVx.first.size();
      std::cout << "seed number in deltaR cone =" << seed_number << std::endl;

      std::vector<std::array<double,5>> seedVector = JetCorePerfectSeedGenerator::seedParFilling(goodSimTkVx,globDet, jet);
      std::cout << "seedVector.size()=" << seedVector.size()<< std::endl;


      for(uint tk=0; tk<seedVector.size(); tk++ ){

        for(int pp=0; pp<5; pp++){
          std::cout << "seed " << tk<< ", int par " << pp << "=" << seedVector.at(tk).at(pp) << std::endl;
        }
        LocalPoint localSeedPoint = LocalPoint(seedVector.at(tk).at(0),seedVector.at(tk).at(1),0);
        double track_theta = 2*std::atan(std::exp(-seedVector.at(tk).at(2)));
        double track_phi =seedVector.at(tk).at(3);
        double pt =  1./ seedVector.at(tk).at(4);

        // std::cout << "OUT, pt=" <<pt << ", phi=" << track_phi << ", track_eta=" << seedVector.at(tk).at(2) << std::endl;

        double normdirR =  pt/sin(track_theta);
        const GlobalVector globSeedDir( GlobalVector::Polar(Geom::Theta<double>(track_theta), Geom::Phi<double> (track_phi), normdirR));
        LocalVector localSeedDir = globDet->surface().toLocal(globSeedDir);

	       int64_t  seedid=  (int64_t(localSeedPoint.x()*200.)<<0)+(int64_t(localSeedPoint.y()*200.)<<16)+(int64_t(seedVector.at(tk).at(2)*400.)<<32)+(int64_t(track_phi*400.)<<48);
	        if(ids.count(seedid)!=0) {
		          // continue;
              std::cout << "seed not removed with DirectSeed cleaner" << std::endl;
            }
	           ids.insert(seedid);

          //seed creation
        float em[15]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
        em[0]=0.15*0.15;
        em[2]=0.5e-5;
        em[5]=0.5e-5;
        em[9]=2e-5;
        em[14]=2e-5;
        long int detId=globDet->geographicalId();
        LocalTrajectoryParameters localParam(localSeedPoint, localSeedDir, TrackCharge(1));
        result->push_back(TrajectorySeed( PTrajectoryStateOnDet (localParam, pt, em, detId, /*surfaceSide*/ 0), edm::OwnVector< TrackingRecHit >() , PropagationDirection::alongMomentum));

        GlobalPoint globalSeedPoint = globDet->surface().toGlobal(localSeedPoint);
        reco::Track::CovarianceMatrix mm;
        resultTracks->push_back(reco::Track(1,1,reco::Track::Point(globalSeedPoint.x(),globalSeedPoint.y(),globalSeedPoint.z()),reco::Track::Vector(globSeedDir.x(),globSeedDir.y(),globSeedDir.z()),1,mm));
          std::cout << "seed " << tk<< ", out,  pt=" << pt << ", eta="<< globSeedDir.eta() << ", phi=" << globSeedDir.phi() <<std::endl;

      }


    } //bigcluster
  } //jet > pt
 } //jet
iEvent.put(std::move(result));
iEvent.put(std::move(resultTracks));
}













std::pair<bool, Basic3DVector<float>> JetCorePerfectSeedGenerator::findIntersection(const GlobalVector & dir,const  reco::Candidate::Point & vertex, const GeomDet* det){
     StraightLinePlaneCrossing vertexPlane(Basic3DVector<float>(vertex.x(),vertex.y(),vertex.z()), Basic3DVector<float>(dir.x(),dir.y(),dir.z()));

     std::pair<bool, Basic3DVector<float>> pos = vertexPlane.position(det->specificSurface());

     return pos;
}


std::pair<int,int> JetCorePerfectSeedGenerator::local2Pixel(double locX, double locY, const GeomDet* det){
    LocalPoint locXY(locX,locY);
    float pixX=(dynamic_cast<const PixelGeomDetUnit*>(det))->specificTopology().pixel(locXY).first;
    float pixY=(dynamic_cast<const PixelGeomDetUnit*>(det))->specificTopology().pixel(locXY).second;
    std::pair<int, int> out(pixX,pixY);
    return out;
}

LocalPoint JetCorePerfectSeedGenerator::pixel2Local(int pixX, int pixY, const GeomDet* det){
    float locX=(dynamic_cast<const PixelGeomDetUnit*>(det))->specificTopology().localX(pixX);
    float locY=(dynamic_cast<const PixelGeomDetUnit*>(det))->specificTopology().localY(pixY);
    LocalPoint locXY(locX,locY);
    return locXY;
}

  int JetCorePerfectSeedGenerator::pixelFlipper(const GeomDet* det){
    int out =1;
    LocalVector locZdir(0,0,1);
    GlobalVector globZdir  = det->specificSurface().toGlobal(locZdir);
    GlobalPoint globDetCenter = det->position();
    float direction = globZdir.x()*globDetCenter.x()+ globZdir.y()*globDetCenter.y()+ globZdir.z()*globDetCenter.z();
    //float direction = globZdir.dot(globDetCenter);
    if(direction<0) out =-1;
    // out=1;
    return out;
}



void JetCorePerfectSeedGenerator::fillPixelMatrix(const SiPixelCluster & cluster, int layer, auto inter, const GeomDet* det, tensorflow::NamedTensorList input_tensors ){//tensorflow::NamedTensorList input_tensors){

    int flip = pixelFlipper(det); // 1=not flip, -1=flip

    for(int i=0; i<cluster.size();i++){
      SiPixelCluster::Pixel pix = cluster.pixel(i);
      std::pair<int,int> pixInter = local2Pixel(inter.x(),inter.y(),det);
      int nx = pix.x-pixInter.first;
      int ny = pix.y-pixInter.second;
      nx=flip*nx;

      if(abs(nx)<jetDimX/2 && abs(ny)<jetDimY/2){
        nx = nx+jetDimX/2;
        ny = ny+jetDimY/2;
        // std::cout << "prefill" << std::endl;

        // input_tensors(0,nx,ny,layer-1) += (pix.adc)/(float)(14000);
//        std::cout << "filling (nx, ny,layer)" << nx<<","<<ny<<"," << layer-1 << ", pixel=" << (pix.adc)/(float)(14000) << std::endl;
        // input_tensors[1].second.matrix<float>()(0,0) = 2;

        // auto input_matrix_cluster = input_tensors[2].second.tensor<float,4>();
        input_tensors[2].second.tensor<float,4>()(0,nx,ny, layer-1) += (pix.adc)/(float)(14000);
        //  input_matrix_cluster(0,nx,ny,layer-1) + (pix.adc)/(float)(14000);
        // std::cout << "postfill" << std::endl;


        // if(nx<jetDimX && ny<jetDimY && layer-1< Nlayer && layer-1>=0 && nx>=0 && ny>=0) {
        //   clusterMeas[nx][ny][layer-1] += (pix.adc)/(float)(14000);//std::cout << "clusterMeas[nx][ny][layer-1] += (pix.adc)/(float)(14000) ="  << (pix.adc)/(float)(14000) << std::endl;;
        // }
      }
    }

}




const GeomDet* JetCorePerfectSeedGenerator::DetectorSelector(int llay, const reco::Candidate& jet, GlobalVector jetDir, const reco::Vertex& jetVertex, const TrackerTopology* const tTopo){

  struct trkNumCompare {
  bool operator()(std::pair<int,const GeomDet*> x, std::pair<int,const GeomDet*> y) const
  {return x.first > y.first;}
  };

  std::set<std::pair<int,const GeomDet*>, trkNumCompare> track4detSet;

  LocalPoint jetInter(0,0,0);

  edmNew::DetSetVector<SiPixelCluster>::const_iterator detIt = inputPixelClusters->begin();

  double minDist = 0.0;
  GeomDet* output = (GeomDet*)0;

  for (; detIt != inputPixelClusters->end(); detIt++) { //loop deset

    const edmNew::DetSet<SiPixelCluster>& detset = *detIt;
    const GeomDet* det = geometry_->idToDet(detset.id()); //lui sa il layer con cast a  PXBDetId (vedi dentro il layer function)
    for (auto cluster = detset.begin(); cluster != detset.end(); cluster++) { //loop cluster
      // const SiPixelCluster& aCluster = *cluster;
      auto aClusterID= detset.id();
      if(DetId(aClusterID).subdetId()!=1) continue;
      int lay = tTopo->layer(det->geographicalId());
      if(lay!=llay) continue;
      std::pair<bool, Basic3DVector<float>> interPair = findIntersection(jetDir,(reco::Candidate::Point)jetVertex.position(), det);
      if(interPair.first==false) continue;
      Basic3DVector<float> inter = interPair.second;
      auto localInter = det->specificSurface().toLocal((GlobalPoint)inter);
      if((minDist==0.0 || std::abs(localInter.x())<minDist) && std::abs(localInter.y())<3.35) {
          minDist = std::abs(localInter.x());
          output = (GeomDet*)det;
          // std::cout << "layer =" << llay << " selected det=" << det->gdetIndex() << " distX=" << localInter.x() << " distY=" << localInter.y() << ", center=" << det->position()<< std::endl;
      }
    } //cluster
  } //detset
  // std::cout << "OK DET= layer =" << llay << " selected det=" << output->gdetIndex() << std::endl;
  return output;
}



std::vector<GlobalVector> JetCorePerfectSeedGenerator::splittedClusterDirections(const reco::Candidate& jet, const TrackerTopology* const tTopo, auto pp, const reco::Vertex& jetVertex , int layer){
  std::vector<GlobalVector> clustDirs;

  edmNew::DetSetVector<SiPixelCluster>::const_iterator detIt_int = inputPixelClusters->begin();


  for (; detIt_int != inputPixelClusters->end(); detIt_int++) {

    const edmNew::DetSet<SiPixelCluster>& detset_int = *detIt_int;
    const GeomDet* det_int = geometry_->idToDet(detset_int.id());
    int lay = tTopo->layer(det_int->geographicalId());
    // std::cout<< "LAYYYYYYYYYYYY=" << lay <<std::endl;
    if(lay != layer) continue; //NB: saved bigclusetr on all the layers!!

    for (auto cluster = detset_int.begin(); cluster != detset_int.end(); cluster++) {
      const SiPixelCluster& aCluster = *cluster;
      // bool hasBeenSplit = false;
      // bool shouldBeSplit = false;
      GlobalPoint cPos = det_int->surface().toGlobal(pp->localParametersV(aCluster,(*geometry_->idToDetUnit(detIt_int->id())))[0].first);
      GlobalPoint ppv(jetVertex.position().x(), jetVertex.position().y(), jetVertex.position().z());
      GlobalVector clusterDir = cPos - ppv;
      GlobalVector jetDir(jet.px(), jet.py(), jet.pz());
      // std::cout <<"deltaR" << Geom::deltaR(jetDir, clusterDir)<<", jetDir="<< jetDir << ", clusterDir=" <<clusterDir << ", X=" << aCluster.sizeX()<< ", Y=" << aCluster.sizeY()<<std::endl;
      if (Geom::deltaR(jetDir, clusterDir) < deltaR_) {
            // check if the cluster has to be splitted
/*
            bool isEndCap =
                (std::abs(cPos.z()) > 30.f);  // FIXME: check detID instead!
            float jetZOverRho = jet.momentum().Z() / jet.momentum().Rho();
            if (isEndCap)
              jetZOverRho = jet.momentum().Rho() / jet.momentum().Z();
            float expSizeY =
                std::sqrt((1.3f*1.3f) + (1.9f*1.9f) * jetZOverRho*jetZOverRho);
            if (expSizeY < 1.f) expSizeY = 1.f;
            float expSizeX = 1.5f;
            if (isEndCap) {
              expSizeX = expSizeY;
              expSizeY = 1.5f;
            }  // in endcap col/rows are switched
            float expCharge =
                std::sqrt(1.08f + jetZOverRho * jetZOverRho) * centralMIPCharge_;
            // std::cout <<"jDir="<< jetDir << ", cDir=" <<clusterDir <<  ", carica=" << aCluster.charge() << ", expChar*cFracMin_=" << expCharge * chargeFracMin_ <<", X=" << aCluster.sizeX()<< ", expSizeX+1=" <<  expSizeX + 1<< ", Y="<<aCluster.sizeY() <<", expSizeY+1="<< expSizeY + 1<< std::endl;
           if (aCluster.charge() > expCharge * chargeFracMin_ && (aCluster.sizeX() > expSizeX + 1 ||  aCluster.sizeY() > expSizeY + 1)) {}
*/
	if(1){
              // shouldBeSplit = true;
              // std::cout << "trovato cluster con deltaR=" << Geom::deltaR(jetDir, clusterDir)<< ", on layer=" <<lay << std::endl;
              clustDirs.push_back(clusterDir);
            }
          }
        }
      }
      return clustDirs;

}


std::vector<PSimHit> JetCorePerfectSeedGenerator::coreHitsFilling(auto simhits,const GeomDet* globDet,GlobalVector bigClustDir,const reco::Vertex& jetVertex){
  std::vector<PSimHit> goodSimHit;
  std::vector<PSimHit>::const_iterator shIt = simhits->begin();
// std::set<const GeomDet*> simhitsDetSet;
  for (; shIt != simhits->end(); shIt++) { //loop deset
  // const edmNew::DetSet<PSimHit>& detset = *shIt;
    const GeomDet* det = geometry_->idToDet((*shIt).detUnitId());
    // if(det!=goodDet1 && det!=goodDet3 && det!=goodDet4 && det!=globDet) continue;
    if(det!=globDet) continue;
    std::pair<bool, Basic3DVector<float>> interPair = findIntersection(bigClustDir,(reco::Candidate::Point)jetVertex.position(), det);
    if(interPair.first==false) continue;
    Basic3DVector<float> inter = interPair.second;
    auto localInter = det->specificSurface().toLocal((GlobalPoint)inter);
    // if(jetInter.x()==0 && jetInter.y()==0 && jetInter.z()==0) jetInter = localInter; //filling infoTracks

    // int flip = pixelFlipper(det);
    if(std::abs(((*shIt).localPosition()).x()-localInter.x())/pitchX<=jetDimX/2 && std::abs(((*shIt).localPosition()).y()-localInter.y())/pitchY<=jetDimY/2){
      // std::cout << "good sim hit" << (*shIt).trackId()<< std::endl;
      goodSimHit.push_back((*shIt));
    }
  }
  return goodSimHit;
}

std::pair<std::vector<SimTrack>,std::vector<SimVertex>> JetCorePerfectSeedGenerator::coreTracksFilling(std::vector<PSimHit> goodSimHit,  const auto & simtracksVector, const auto & simvertexVector){
  std::vector<SimTrack> goodSimTrk;
  std::vector<SimVertex> goodSimVtx;



  for(uint j=0; j<simtracksVector->size(); j++){
      for(std::vector<PSimHit>::const_iterator it=goodSimHit.begin(); it!=goodSimHit.end(); ++it) {
        SimTrack st = simtracksVector->at(j);
        if(st.trackId()==(*it).trackId()) {
          // goodSimTrk.push_back(st);

            for(uint v =0; v<simvertexVector->size(); v++) {
              SimVertex sv = simvertexVector->at(v);
              if((int)sv.vertexId()==(int)st.vertIndex()){
                // if(st.vertIndex()==-1) goodSimVtx.push_back((SimVertex)jVert);
                //else
                goodSimTrk.push_back(st);
                goodSimVtx.push_back(sv);
              }
            }
      }
    }
  }
  std::pair<std::vector<SimTrack>,std::vector<SimVertex>> output(goodSimTrk,goodSimVtx);
  return output;
}

std::pair<std::vector<SimTrack>,std::vector<SimVertex>> JetCorePerfectSeedGenerator::coreTracksFillingDeltaR( const auto & simtracksVector,  const auto &  simvertexVector,const GeomDet* globDet, const reco::Candidate& jet, auto jetVertex){
  std::vector<SimTrack> goodSimTrk;
  std::vector<SimVertex> goodSimVtx;

  GlobalVector jetDir(jet.px(), jet.py(), jet.pz());

  for(uint j=0; j<simtracksVector->size(); j++){
    SimTrack st = simtracksVector->at(j);
    GlobalVector trkDir(st.momentum().Px(), st.momentum().Py(), st.momentum().Pz());
    if(Geom::deltaR(jetDir, trkDir) < deltaR_){
      if(st.charge()==0) continue;
      // if((int)st.vertIndex()==-1) {
      //   goodSimTrk.push_back(st);
      //   // goodSimVtx.push_back((SimVertex)jetVertex);
      //   std::cout << "problema" << std::endl;
      // }
      // else {
      for(uint v =0; v<simvertexVector->size(); v++) {
        SimVertex sv = simvertexVector->at(v);
        if((int)sv.vertexId()==(int)st.vertIndex()){
          // if(st.vertIndex()==-1) goodSimVtx.push_back((SimVertex)jVert);
          //else
          // std::cout << "goodsimtrack " << j<< ", filling good st, pt" << st.momentum().Pt() << ", eta="<< st.momentum().Eta() << ", phi=" << st.momentum().Phi() << std::endl;
          goodSimTrk.push_back(st);
          goodSimVtx.push_back(sv);
        }
      }
    // }

    }
    // else std::cout << "BAD sim track " << j<< ", pt" << st.momentum().Pt() << ", eta="<< st.momentum().Eta() << ", phi=" << st.momentum().Phi() << std::endl;
  }
  std::pair<std::vector<SimTrack>,std::vector<SimVertex>> output(goodSimTrk,goodSimVtx);
  return output;
}


std::vector<std::array<double,5>> JetCorePerfectSeedGenerator::seedParFilling(std::pair<std::vector<SimTrack>,std::vector<SimVertex>> goodSimTkVx,const GeomDet* globDet, const reco::Candidate& jet){
  std::vector<std::array<double,5>> output;
  std::vector<SimTrack> goodSimTrk=goodSimTkVx.first;
  std::vector<SimVertex> goodSimVtx=goodSimTkVx.second;

  std::cout << "goodSimTrk.size()" << goodSimTrk.size() << std::endl;
  for(uint j=0; j<goodSimTrk.size(); j++){
    SimTrack st = goodSimTrk.at(j);
    SimVertex sv = goodSimVtx.at(j);
    GlobalVector trkMom(st.momentum().x(),st.momentum().y(), st.momentum().z());
    GlobalPoint trkPos(sv.position().x(),sv.position().y(), sv.position().z());
    std::cout << "seed " << j<< ", very int pt" << st.momentum().Pt() << ", eta="<< st.momentum().Eta() << ", phi=" << st.momentum().Phi() << "------ internal point="<< trkMom.x() << "," << trkMom.y() << "," << trkMom.z() <<"," << trkPos.x() << "," << trkPos.y() << ","<< trkPos.z() << std::endl;

    bool old_approach = true; // if true use the directSeedGenerator like approahc to build the seed
    // if(old_approach){
    std::pair<bool, Basic3DVector<float>> trkInterPair;
    // std::cout << "sv " << (int)sv.vertexId() << "/// st " << (int)st.vertIndex()<< std::endl;
    trkInterPair = findIntersection(trkMom,(reco::Candidate::Point)trkPos, globDet);
    if(trkInterPair.first==false) {
      GlobalVector jetDir(jet.px(), jet.py(), jet.pz());
      double deltar = Geom::deltaR(jetDir, trkMom);
      // std::cout << "not intersection, deltaR=" << deltar << std::endl;

      continue;
    }
    Basic3DVector<float> trkInter = trkInterPair.second;

    auto localTrkInter = globDet->specificSurface().toLocal((GlobalPoint)trkInter);
    // std::cout << ", localtrackInter" << localTrkInter.x() << ", " << localTrkInter.y() << std::endl;



    // double tkpar[Npar];
    // // for(uint v =0; v<simvertexVector->size(); v++) {
    // //   SimVertex sv = simvertexVector->at(v);
    // //   if((int)sv.vertexId()==(int)st.vertIndex()){
    // //
    // //   }
    // // }
    //
    // tkpar[0] = localTrkInter.x();
    // tkpar[1] = localTrkInter.y();
    // tkpar[2] = st.momentum().Eta();
    // tkpar[3] = st.momentum().Phi();
    // tkpar[4] = 1/st.momentum().Pt();

    // std::cout << "IN, pt=" <<st.momentum().Pt() << ", phi=" << st.momentum().Phi() << ", track_eta=" << st.momentum().Eta() << std::endl;


    std::array<double,5> tkPar {{localTrkInter.x(), localTrkInter.y(), st.momentum().Eta(), st.momentum().Phi(), 1/st.momentum().Pt()}};
    output.push_back(tkPar);
  // }


  //vertex approach--------------------------------
  // auto localPos  = globDet->specificSurface().toLocal((GlobalPoint)trkPos);
  // std::array<double,5> tkPar {{localPos.x(), localPos.y(), st.momentum().Eta(), st.momentum().Phi(), 1/st.momentum().Pt()}};
  // output.push_back(tkPar);
  //end of vertex approach------------------------


  }
  return output;
}






// ------------ method called once each job just before starting event loop  ------------
void
JetCorePerfectSeedGenerator::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
JetCorePerfectSeedGenerator::endJob()
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
JetCorePerfectSeedGenerator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
// DEFINE_FWK_MODULE(JetCorePerfectSeedGenerator);
