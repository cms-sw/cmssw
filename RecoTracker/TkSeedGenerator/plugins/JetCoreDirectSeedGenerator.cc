// -*- C++ -*-
//
// Package:    trackJet/JetCoreDirectSeedGenerator
// Class:      JetCoreDirectSeedGenerator
//
/**\class JetCoreDirectSeedGenerator JetCoreDirectSeedGenerator.cc trackJet/JetCoreDirectSeedGenerator/plugins/JetCoreDirectSeedGenerator.cc

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

#include "JetCoreDirectSeedGenerator.h"

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
// #include "SimDataFormats/Track/interface/SimTrack.h"

#include "SimDataFormats/Vertex/interface/SimVertex.h"


#include "Geometry/TrackerGeometryBuilder/interface/PixelGeomDetUnit.h"

#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"



#include "TTree.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"


//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<> and also remove the line from
// constructor "usesResource("TFileService");"
// This will improve performance in multithreaded jobs.



JetCoreDirectSeedGenerator::JetCoreDirectSeedGenerator(const edm::ParameterSet& iConfig) :

      vertices_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertices"))),
      pixelClusters_(consumes<edmNew::DetSetVector<SiPixelCluster> >(iConfig.getParameter<edm::InputTag>("pixelClusters"))),
      cores_(consumes<edm::View<reco::Candidate> >(iConfig.getParameter<edm::InputTag>("cores"))),
      ptMin_(iConfig.getParameter<double>("ptMin")),
      deltaR_(iConfig.getParameter<double>("deltaR")),
      chargeFracMin_(iConfig.getParameter<double>("chargeFractionMin")),
      centralMIPCharge_(iConfig.getParameter<double>("centralMIPCharge")),
      pixelCPE_(iConfig.getParameter<std::string>("pixelCPE")),

      weightfilename_(iConfig.getParameter<edm::FileInPath>("weightFile").fullPath()),
      inputTensorName_(iConfig.getParameter<std::vector<std::string>>("inputTensorName")),
      outputTensorName_(iConfig.getParameter<std::vector<std::string>>("outputTensorName")),
      nThreads(iConfig.getParameter<unsigned int>("nThreads")),
      singleThreadPool(iConfig.getParameter<std::string>("singleThreadPool")),
      probThr(iConfig.getParameter<double>("probThr"))

{
  produces<TrajectorySeedCollection>();
  produces<reco::TrackCollection>();



  //  edm::Service<TFileService> fileService;
   //
  //  JetCoreDirectSeedGeneratorTree= fileService->make<TTree>("JetCoreDirectSeedGeneratorTree","JetCoreDirectSeedGeneratorTree");
  //  JetCoreDirectSeedGeneratorTree->Branch("cluster_measured",clusterMeas,"cluster_measured[30][30][4]/D");
  //  JetCoreDirectSeedGeneratorTree->Branch("jet_eta",&jet_eta);
  //  JetCoreDirectSeedGeneratorTree->Branch("jet_pt",&jet_pt);


    //  for(int i=0; i<Nlayer; i++){ //NOFLAG
    //    for(int j=0; j<jetDimX; j++){
    //      for(int k=0; k<jetDimY; k++){
    //        if(j<jetDimX && k<jetDimY && i< Nlayer) clusterMeas[j][k][i] = 0.0;
    //       }
    //      }
    //   }



}


JetCoreDirectSeedGenerator::~JetCoreDirectSeedGenerator()
{

}

#define foreach BOOST_FOREACH


void JetCoreDirectSeedGenerator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  evt_counter++;
  // std::cout << "NEW EVENT, event number" << evt_counter  <<std::endl;
  auto result = std::make_unique<TrajectorySeedCollection>();
  auto resultTracks = std::make_unique<reco::TrackCollection>();

  //-------------------TensorFlow setup - session (1/2)----------------------//

  tensorflow::setLogging("3");
  graph_=tensorflow::loadGraphDef(weightfilename_);
  // output_names_=iConfig.getParameter<std::vector<std::string>>("outputNames");
  // for(const auto & s : iConfig.getParameter<std::vector<std::string>>("outputFormulas")) { output_formulas_.push_back(StringObjectFunction<std::vector<float>>(s));}
  tensorflow::SessionOptions sessionOptions;
  tensorflow::setThreading(sessionOptions, nThreads, singleThreadPool);
  session_ = tensorflow::createSession(graph_, sessionOptions);
  tensorflow::TensorShape input_size_eta({1,1}) ;
  tensorflow::TensorShape input_size_pt({1,1}) ;
  tensorflow::TensorShape input_size_cluster({1,jetDimX,jetDimY,Nlayer});

  // std::cout << "input_size_cluster=" << input_size_cluster.num_elements() << "," << "," << input_size_cluster.dims() << "," <<  input_size_cluster.dim_size(0) << "," << input_size_cluster.dim_size(1) <<"," << input_size_cluster.dim_size(2) <<"," << input_size_cluster.dim_size(3) << std::endl;

  // input_size_cluster.set_dim(0,1);
  // input_size_cluster.set_dim(1,jetDimX);
  // input_size_cluster.set_dim(2,jetDimY);
  // input_size_cluster.set_dim(3,Nlayer);
  ;

    // tensorflow::TensorShape input_size_cluster   {1,1,1,1} ;
  //-----------------end of TF setup (1/2)----------------------//

  evt_counter++;
//  std::cout << "event number (iterative)=" << evt_counter<< ", event number (id)="<< iEvent.id().event() << std::endl;


  using namespace edm;
  using namespace reco;


  iSetup.get<IdealMagneticFieldRecord>().get( magfield_ );
  iSetup.get<GlobalTrackingGeometryRecord>().get(geometry_);
  iSetup.get<TrackingComponentsRecord>().get( "AnalyticalPropagator", propagator_ );

  iEvent.getByToken(pixelClusters_, inputPixelClusters);
  allSiPixelClusters.clear(); siPixelDetsWithClusters.clear();
  allSiPixelClusters.reserve(inputPixelClusters->dataSize()); // this is important, otherwise push_back invalidates the iterators

  // edm::Handle<std::vector<SimTrack> > simtracks;
  // iEvent.getByToken(simtracksToken, simtracks);
  // edm::Handle<std::vector<SimVertex> > simvertex;
  // iEvent.getByToken(simvertexToken, simvertex);

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
  for (unsigned int ji = 0; ji < cores->size(); ji++) { //loop jet
    jet_number++;

    if ((*cores)[ji].pt() > ptMin_) {
//       std::cout << "|____________________NEW JET_______________________________| jet number=" << jet_number  << " " << (*cores)[ji].pt() << " " << (*cores)[ji].eta() << " " << (*cores)[ji].phi() <<  std::endl;

      std::set<long long int> ids;
      const reco::Candidate& jet = (*cores)[ji];
      const reco::Vertex& jetVertex = (*vertices)[0];

      std::vector<GlobalVector> splitClustDirSet = splittedClusterDirections(jet, tTopo, pp, jetVertex, 1);
      //std::vector<GlobalVector> splitClustDirSet = splittedClusterDirections(jet, tTopo, pp, jetVertex);
      bool l2off=(splitClustDirSet.size()==0);
      if(splitClustDirSet.size()==0) {//if layer 1 is broken find direcitons on layer 2
        splitClustDirSet = splittedClusterDirections(jet, tTopo, pp, jetVertex, 2);
        // std::cout << "split on lay2, in numero=" << splitClustDirSet.size() << "+jetDir" << std::endl;
      }
      splitClustDirSet.push_back(GlobalVector(jet.px(),jet.py(),jet.pz()));
  //    std::cout << "splitted cluster number=" << splitClustDirSet.size() << std::endl;;
      for(int cc=0; cc<(int)splitClustDirSet.size(); cc++){

      //-------------------TensorFlow setup - tensor (2/2)----------------------//
      tensorflow::NamedTensorList input_tensors;
      input_tensors.resize(3);
      input_tensors[0] = tensorflow::NamedTensor(inputTensorName_[0], tensorflow::Tensor(tensorflow::DT_FLOAT, input_size_eta));
      input_tensors[1] = tensorflow::NamedTensor(inputTensorName_[1], tensorflow::Tensor(tensorflow::DT_FLOAT, input_size_pt));
      input_tensors[2] = tensorflow::NamedTensor(inputTensorName_[2], tensorflow::Tensor(tensorflow::DT_FLOAT, {input_size_cluster}));

      //put all the input tensor to 0
      input_tensors[0].second.matrix<float>()(0,0) =0.0;
      input_tensors[1].second.matrix<float>()(0,0) = 0.0;
      for(int x=0; x<jetDimX; x++){
        for(int y=0; y<jetDimY; y++){
          for(int l=0; l<4; l++){
              input_tensors[2].second.tensor<float,4>()(0,x,y,l) = 0.0;
            }
          }
        }
      // auto input_matrix_eta = input_tensors[0].second.tensor<float,2>();
      // auto input_matrix_pt = input_tensors[1].second.tensor<float,2>();
      // auto input_matrix_cluster = input_tensors[2].second.tensor<float,4>();

      //
      // std::vector<tensorflow::Tensor> inputs;
      // std::vector<std::string> input_names;
      //
      // ouput_names.push_back(inputTensorName_[0]);
      // ouput_names.push_back(inputTensorName_[1]);
      // ouput_names.push_back(inputTensorName_[2]);




      //-----------------end of TF setup (2/2)----------------------//

      GlobalVector bigClustDir = splitClustDirSet.at(cc);

      LocalPoint jetInter(0,0,0);

      jet_eta = jet.eta();
      jet_pt = jet.pt();
      // input_tensors(0).at(0) = jet.eta();
      // input_tensors[1](0) = jet.pt();
      // input_matrix_eta(0,0) = jet.eta();
      // input_matrix_pt(0,0) = jet.pt();
      input_tensors[0].second.matrix<float>()(0,0) = jet.eta();
      input_tensors[1].second.matrix<float>()(0,0) = jet.pt();

      auto jetVert = jetVertex; //trackInfo filling



      edmNew::DetSetVector<SiPixelCluster>::const_iterator detIt = inputPixelClusters->begin();

      const GeomDet* globDet = DetectorSelector(2, jet, bigClustDir, jetVertex, tTopo); //select detector mostly hitten by the jet

      if(globDet == 0) continue;

      const GeomDet* goodDet1 = DetectorSelector(1, jet, bigClustDir, jetVertex, tTopo);
      const GeomDet* goodDet3 = DetectorSelector(3, jet, bigClustDir, jetVertex, tTopo);
      const GeomDet* goodDet4 = DetectorSelector(4, jet, bigClustDir, jetVertex, tTopo);



      for (; detIt != inputPixelClusters->end(); detIt++) { //loop deset
        const edmNew::DetSet<SiPixelCluster>& detset = *detIt;
        const GeomDet* det = geometry_->idToDet(detset.id()); //lui sa il layer con cast a  PXBDetId (vedi dentro il layer function)

        for (auto cluster = detset.begin(); cluster != detset.end(); cluster++) { //loop cluster

          const SiPixelCluster& aCluster = *cluster;
          det_id_type aClusterID= detset.id();
          if(DetId(aClusterID).subdetId()!=1) continue;

          int lay = tTopo->layer(det->geographicalId());

          std::pair<bool, Basic3DVector<float>> interPair = findIntersection(bigClustDir,(reco::Candidate::Point)jetVertex.position(), det);
          if(interPair.first==false) continue;
          Basic3DVector<float> inter = interPair.second;
          auto localInter = det->specificSurface().toLocal((GlobalPoint)inter);

          GlobalPoint pointVertex(jetVertex.position().x(), jetVertex.position().y(), jetVertex.position().z());


          // GlobalPoint cPos = det->surface().toGlobal(pp->localParametersV(aCluster,(*geometry_->idToDetUnit(detIt->id())))[0].first);
          LocalPoint cPos_local = pp->localParametersV(aCluster,(*geometry_->idToDetUnit(detIt->id())))[0].first;

          if(std::abs(cPos_local.x()-localInter.x())/pitchX<=jetDimX/2 && std::abs(cPos_local.y()-localInter.y())/pitchY<=jetDimY/2){ // per ora preso baricentro, da migliorare

            if(det==goodDet1 || det==goodDet3 || det==goodDet4 || det==globDet) {
              // fillPixelMatrix(aCluster,lay,localInter, det, input_matrix_cluster);
              fillPixelMatrix(aCluster,lay,localInter, det, input_tensors);

              }
          } //cluster in ROI
        } //cluster
      } //detset

    // JetCoreDirectSeedGeneratorTree->Fill();

    //HERE SOMEHOW THE NN PRODUCE THE SEED FROM THE FILLED INPUT
// std::cout << "Filling complete" << std::endl;
    std::pair<double[jetDimX][jetDimY][Nover][Npar],double[jetDimX][jetDimY][Nover]> seedParamNN = JetCoreDirectSeedGenerator::SeedEvaluation(input_tensors);

    for(int i=0; i<jetDimX; i++){
      for(int j=0; j<jetDimY; j++){
        for(int o=0; o<Nover; o++){
          // if(seedParamNN.second[i][j][o]>(0.75-o*0.1-(l2off?0.25:0))){//0.99=probThr (doesn't work the variable, SOLVE THIS ISSUE!!)
          if(seedParamNN.second[i][j][o]>(0.85-o*0.1-(l2off?0.35:0))){//0.99=probThr (doesn't work the variable, SOLVE THIS ISSUE!!)

        //    std::cout << "prob success=" << seedParamNN.second[i][j][o]<< ", for (x,y)=" << i <<"," <<j << ", threshold="<< probThr << std::endl;
	    /*seedParamNN.first[i][j][o][0]=0;
	    seedParamNN.first[i][j][o][1]=0;
	    seedParamNN.first[i][j][o][2]=0;
	    seedParamNN.first[i][j][o][3]=0; */
            //NN pixel parametrization->local parametrization
	    std::pair<bool, Basic3DVector<float>> interPair =  findIntersection(bigClustDir,(reco::Candidate::Point)jetVertex.position(), globDet);
	    auto localInter = globDet->specificSurface().toLocal((GlobalPoint)interPair.second);

	    int flip = pixelFlipper(globDet); // 1=not flip, -1=flip
	    int nx=i-jetDimX/2;
	    int ny=j-jetDimY/2;
            nx=flip*nx;
      	    std::pair<int,int> pixInter = local2Pixel(localInter.x(),localInter.y(),globDet);
            nx = nx+pixInter.first;
            ny = ny+pixInter.second;
            LocalPoint xyLocal = pixel2Local(nx,ny,globDet);


            double xx = xyLocal.x()+seedParamNN.first[i][j][o][0]*0.01 ;
            double yy = xyLocal.y()+seedParamNN.first[i][j][o][1]*0.01 ;
            LocalPoint localSeedPoint = LocalPoint(xx,yy,0);

            // double jet_theta = 2*std::atan(std::exp(-jet_eta));
            double track_eta = seedParamNN.first[i][j][o][2]*0.01+bigClustDir.eta();//NOT SURE ABOUT THIS 0.01, only to debug
            double track_theta = 2*std::atan(std::exp(-track_eta));
            double track_phi = seedParamNN.first[i][j][o][3]*0.01+bigClustDir.phi();//NOT SURE ABOUT THIS 0.01, only to debug

            double pt =  1./ seedParamNN.first[i][j][o][4];
//	    double pt=10;
            double normdirR =  pt/sin(track_theta);

            const GlobalVector globSeedDir( GlobalVector::Polar(Geom::Theta<double>(track_theta), Geom::Phi<double> (track_phi), normdirR));
            LocalVector localSeedDir = globDet->surface().toLocal(globSeedDir);
//	    double pt2=2;
 //           double normdirR2 = pt2/sin(track_theta);
;
   //       const GlobalVector globSeedDir2( GlobalVector::Polar(Geom::Theta<double>(track_theta), Geom::Phi<double> (track_phi), normdirR2));
     //     LocalVector localSeedDir2 = globDet->surface().toLocal(globSeedDir2);
	    int64_t  seedid=  (int64_t(xx*200.)<<0)+(int64_t(yy*200.)<<16)+(int64_t(track_eta*400.)<<32)+(int64_t(track_phi*400.)<<48);
	    if(ids.count(seedid)!=0) {
//		std::cout << "Rejecting seed" << xx << " " << yy << " " << track_eta << " " << track_phi << " " << seedid << std::endl;
		continue;
            }
	    ids.insert(seedid);
//nn	    std::cout << "Creating seed" << xx << " " << yy << " " << track_eta << " " << track_phi << " " << seedid << std::endl;

	    //seed creation
            float em[15]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
            em[0]=0.15*0.15;
            em[2]=0.5e-5;
            em[5]=0.5e-5;
            em[9]=2e-5;
            em[14]=2e-5;
 /*[2]=1e-5;
            em[5]=1e-5;
            em[9]=2e-5;
            em[14]=2e-5;*/
            long int detId=globDet->geographicalId();
            LocalTrajectoryParameters localParam(localSeedPoint, localSeedDir, TrackCharge(1));
           result->push_back(TrajectorySeed( PTrajectoryStateOnDet (localParam, pt, em, detId, /*surfaceSide*/ 0), edm::OwnVector< TrackingRecHit >() , PropagationDirection::alongMomentum));
     //       LocalTrajectoryParameters localParam2(localSeedPoint, localSeedDir2, TrackCharge(1));
       //    result->push_back(TrajectorySeed( PTrajectoryStateOnDet (localParam2, pt2, em, detId, /*surfaceSide*/ 0), edm::OwnVector< TrackingRecHit >() , PropagationDirection::alongMomentum));

           GlobalPoint globalSeedPoint = globDet->surface().toGlobal(localSeedPoint);
	          reco::Track::CovarianceMatrix mm;
           resultTracks->push_back(reco::Track(1,1,reco::Track::Point(globalSeedPoint.x(),globalSeedPoint.y(),globalSeedPoint.z()),reco::Track::Vector(globSeedDir.x(),globSeedDir.y(),globSeedDir.z()),1,mm));

          }
        }
      }
    }



//    std::cout << "FILL!" << std::endl;

    // for(int i=0; i<Nlayer; i++){
    //   for(int j=0; j<jetDimX; j++){
    //     for(int k=0; k<jetDimY; k++){
    //       if(j<jetDimX && k<jetDimY && i< Nlayer) clusterMeas[j][k][i] = 0.0;
    //     }
    //   }
    // }
  } //bigcluster
  } //jet > pt
 } //jet
iEvent.put(std::move(result));
iEvent.put(std::move(resultTracks));
}













std::pair<bool, Basic3DVector<float>> JetCoreDirectSeedGenerator::findIntersection(const GlobalVector & dir,const  reco::Candidate::Point & vertex, const GeomDet* det){
     StraightLinePlaneCrossing vertexPlane(Basic3DVector<float>(vertex.x(),vertex.y(),vertex.z()), Basic3DVector<float>(dir.x(),dir.y(),dir.z()));

     std::pair<bool, Basic3DVector<float>> pos = vertexPlane.position(det->specificSurface());

     return pos;
}


std::pair<int,int> JetCoreDirectSeedGenerator::local2Pixel(double locX, double locY, const GeomDet* det){
    LocalPoint locXY(locX,locY);
    float pixX=(dynamic_cast<const PixelGeomDetUnit*>(det))->specificTopology().pixel(locXY).first;
    float pixY=(dynamic_cast<const PixelGeomDetUnit*>(det))->specificTopology().pixel(locXY).second;
    std::pair<int, int> out(pixX,pixY);
    return out;
}

LocalPoint JetCoreDirectSeedGenerator::pixel2Local(int pixX, int pixY, const GeomDet* det){
    float locX=(dynamic_cast<const PixelGeomDetUnit*>(det))->specificTopology().localX(pixX);
    float locY=(dynamic_cast<const PixelGeomDetUnit*>(det))->specificTopology().localY(pixY);
    LocalPoint locXY(locX,locY);
    return locXY;
}

  int JetCoreDirectSeedGenerator::pixelFlipper(const GeomDet* det){
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



void JetCoreDirectSeedGenerator::fillPixelMatrix(const SiPixelCluster & cluster, int layer, auto inter, const GeomDet* det, tensorflow::NamedTensorList input_tensors ){//tensorflow::NamedTensorList input_tensors){

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

std::pair<double[jetDimX][jetDimY][Nover][Npar],double[jetDimX][jetDimY][Nover]> JetCoreDirectSeedGenerator::SeedEvaluation(tensorflow::NamedTensorList input_tensors){

  // tensorflow::TensorShape input_size_cluster   {1,jetDimX,jetDimY,Nlayer} ;
  // tensorflow::TensorShape input_size_pt   {1} ;
  // tensorflow::TensorShape input_size_eta   {1} ;
  // tensorflow::NamedTensorList input_tensors;
  // input_tensors.resize(3);
  // input_tensors[0] = tensorflow::NamedTensor(inputTensorName_[0], tensorflow::Tensor(tensorflow::DT_FLOAT, input_size_eta));
  // input_tensors[1] = tensorflow::NamedTensor(inputTensorName_[1], tensorflow::Tensor(tensorflow::DT_FLOAT, input_size_pt));
  // input_tensors[2] = tensorflow::NamedTensor(inputTensorName_[2], tensorflow::Tensor(tensorflow::DT_FLOAT, input_size_cluster));

  // for(int lay=0; lay<Nlayer; lay++){
  //   for(int x=0;x<jetDimX/2; x++){
  //     for(int y=0;y<jetDimY/2; y++){
  //       input_tensors[2].second.matrix<float>()(0,nx,ny,layer-1) = clusterMeas[nx][ny][layer-1];
  //   }
  // }
  //
  // // for(size_t j =0; j < values_.size();j++) {
  // //     input_tensors[0].second.matrix<float>()(0,j) = values_[j];
  // // }

  // debug!!!
/*
  for(int x=0; x<jetDimX; x++){
    for(int y=0; y<jetDimY; y++){
      for(int l=0; l<4; l++){
        if(input_tensors[2].second.tensor<float,4>()(0,x,y,l)!=0){
          std::cout << "input, " << "x=" << x << ", y=" << y <<", lay=" << l << ", val =" << input_tensors[2].second.tensor<float,4>()(0,x,y,l) << std::endl;
        }
      }
    }
  } //end of debug
*/
  std::vector<tensorflow::Tensor> outputs;
  std::vector<std::string> output_names;
  output_names.push_back(outputTensorName_[0]);
  output_names.push_back(outputTensorName_[1]);
  tensorflow::run(session_, input_tensors, output_names, &outputs);
  auto matrix_output_par = outputs.at(0).tensor<float,5>();
  auto matrix_output_prob = outputs.at(1).tensor<float,5>();

  // double trackPar[jetDimX][jetDimY][Nover][Npar+1]; //NOFLAG
  // double trackProb[jetDimX][jetDimY][Nover];

  std::pair<double[jetDimX][jetDimY][Nover][Npar],double[jetDimX][jetDimY][Nover]> output_combined;


  for(int x=0; x<jetDimX; x++){
    for(int y=0; y<jetDimY; y++){
      for(int trk=0; trk<Nover; trk++){
        // trackProb[x][y][trk]=outputs.at(1).matrix<double>()(0,x,y,trk);
        output_combined.second[x][y][trk]=matrix_output_prob(0,x,y,trk,0);//outputs.at(1).matrix<double>()(0,x,y,trk);

        for(int p=0; p<Npar;p++){
          // trackPar[x][y][trk][p]=outputs.at(0).matrix<double>()(0,x,y,trk,p);
          output_combined.first[x][y][trk][p]=matrix_output_par(0,x,y,trk,p);//outputs.at(0).matrix<double>()(0,x,y,trk,p);
//          if(matrix_output_prob(0,x,y,trk,0)>0.9) std::cout << "internal output, prob= "<<matrix_output_prob(0,x,y,trk,0)<< ", x=" << x << ", y="<< y << ", trk=" <<trk << ", par=" << p << ",value="<< matrix_output_par(0,x,y,trk,p) << std::endl;
        }
      }
    }
  }

  // std::pair<double[jetDimX][jetDimY][Nover][Npar+1],double[jetDimX][jetDimY][Nover]> output_combined;
  // output_combined.first = trackPar;
  // output_combined.second = trackProb;
  return output_combined;




}


const GeomDet* JetCoreDirectSeedGenerator::DetectorSelector(int llay, const reco::Candidate& jet, GlobalVector jetDir, const reco::Vertex& jetVertex, const TrackerTopology* const tTopo){

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
std::vector<GlobalVector> JetCoreDirectSeedGenerator::splittedClusterDirections(const reco::Candidate& jet, const TrackerTopology* const tTopo, auto pp, const reco::Vertex& jetVertex , int layer){
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

           if (aCluster.charge() > expCharge * chargeFracMin_ && (aCluster.sizeX() > expSizeX + 1 ||  aCluster.sizeY() > expSizeY + 1)) {
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


std::vector<GlobalVector> JetCoreDirectSeedGenerator::splittedClusterDirectionsOld(const reco::Candidate& jet, const TrackerTopology* const tTopo, auto pp, const reco::Vertex& jetVertex ){
  std::vector<GlobalVector> clustDirs;

  edmNew::DetSetVector<SiPixelCluster>::const_iterator detIt_int = inputPixelClusters->begin();


  for (; detIt_int != inputPixelClusters->end(); detIt_int++) {

    const edmNew::DetSet<SiPixelCluster>& detset_int = *detIt_int;
    const GeomDet* det_int = geometry_->idToDet(detset_int.id());
    int lay = tTopo->layer(det_int->geographicalId());
//    if(lay != 1) continue;

    for (auto cluster = detset_int.begin(); cluster != detset_int.end(); cluster++) {
      const SiPixelCluster& aCluster = *cluster;

      GlobalPoint cPos = det_int->surface().toGlobal(pp->localParametersV(aCluster,(*geometry_->idToDetUnit(detIt_int->id())))[0].first);
      GlobalPoint ppv(jetVertex.position().x(), jetVertex.position().y(), jetVertex.position().z());
      GlobalVector clusterDir = cPos - ppv;
      GlobalVector jetDir(jet.px(), jet.py(), jet.pz());
      if (Geom::deltaR(jetDir, clusterDir) < deltaR_) {
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
          //  if (aCluster.charge() > expCharge * chargeFracMin_ && (aCluster.sizeX() > expSizeX + 1 ||  aCluster.sizeY() > expSizeY + 1)) {
 */           if (1) { //aCluster.charge() > expCharge * chargeFracMin_ && (aCluster.sizeX() > expSizeX + 1 ||  aCluster.sizeY() > expSizeY + 1)) {
//              std::cout << "trovato cluster con deltaR=" << Geom::deltaR(jetDir, clusterDir)<< std::endl;
              clustDirs.push_back(clusterDir);
            }
          }
        }
      }
      return clustDirs;

}


// ------------ method called once each job just before starting event loop  ------------
void
JetCoreDirectSeedGenerator::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void
JetCoreDirectSeedGenerator::endJob()
{
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
JetCoreDirectSeedGenerator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

//define this as a plug-in
// DEFINE_FWK_MODULE(JetCoreDirectSeedGenerator);
