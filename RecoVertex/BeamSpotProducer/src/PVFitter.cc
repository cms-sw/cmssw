/**_________________________________________________________________
   class:   PVFitter.cc
   package: RecoVertex/BeamSpotProducer
   


   author: Francisco Yumiceva, Fermilab (yumiceva@fnal.gov)
           Geng-Yuan Jeng, UC Riverside (Geng-Yuan.Jeng@cern.ch)
 
   version $Id: PVFitter.cc,v 1.2 2010/03/18 12:13:27 adamwo Exp $

________________________________________________________________**/

#include "RecoVertex/BeamSpotProducer/interface/PVFitter.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/View.h"

#include "DataFormats/TrackCandidate/interface/TrackCandidate.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/HitPattern.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "TFitterMinuit.h"
#include "Minuit2/FCNBase.h"
#include "RecoVertex/BeamSpotProducer/interface/FcnBeamSpotFitPV.h"

#include "TF1.h"

// ----------------------------------------------------------------------
// Useful function:
// ----------------------------------------------------------------------

// static char * formatTime(const std::time_t & t)  {
//   struct std::tm * ptm;
//   ptm = gmtime(&t);
//   static char ts[32];
//   strftime(ts,sizeof(ts),"%Y.%m.%d %H:%M:%S %Z",ptm);
//   return ts;
// }

PVFitter::PVFitter(const edm::ParameterSet& iConfig)
{

  debug_             = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<bool>("Debug");
  vertexLabel_     = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<edm::InputTag>("VertexCollection");
  do3DFit_           = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<bool>("Apply3DFit");
  //writeTxt_          = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<bool>("WriteAscii");
  //outputTxt_         = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<std::string>("AsciiFileName");

  minNrVertices_     = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<unsigned int>("minNrVerticesForFit");
  minVtxNdf_         = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<double>("minVertexNdf");
  maxVtxNormChi2_    = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<double>("maxVertexNormChi2");
  minVtxTracks_      = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<unsigned int>("minVertexNTracks");
  minVtxWgt_         = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<double>("minVertexMeanWeight");
  maxVtxR_           = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<double>("maxVertexR");
  maxVtxZ_           = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<double>("maxVertexZ");
  errorScale_        = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<double>("errorScale");
  sigmaCut_          = iConfig.getParameter<edm::ParameterSet>("PVFitter").getUntrackedParameter<double>("nSigmaCut");


  hPVx = new TH2F("hPVx","PVx vs PVz distribution",200,-maxVtxR_, maxVtxR_, 200, -maxVtxZ_, maxVtxZ_);
  hPVy = new TH2F("hPVy","PVy vs PVz distribution",200,-maxVtxR_, maxVtxR_, 200, -maxVtxZ_, maxVtxZ_);
  
}

PVFitter::~PVFitter() {

}


void PVFitter::readEvent(const edm::Event& iEvent)
{

//   frun = iEvent.id().run();
//   const edm::TimeValue_t ftimestamp = iEvent.time().value();
//   const std::time_t ftmptime = ftimestamp >> 32;

//   if (fbeginLumiOfFit == -1) freftime[0] = freftime[1] = ftmptime;
//   if (freftime[0] == 0 || ftmptime < freftime[0]) freftime[0] = ftmptime;
//   const char* fbeginTime = formatTime(freftime[0]);
//   sprintf(fbeginTimeOfFit,"%s",fbeginTime);

//   if (freftime[1] == 0 || ftmptime > freftime[1]) freftime[1] = ftmptime;
//   const char* fendTime = formatTime(freftime[1]);
//   sprintf(fendTimeOfFit,"%s",fendTime);

//   flumi = iEvent.luminosityBlock();
//   frunFit = frun;

//   if (fbeginLumiOfFit == -1 || fbeginLumiOfFit > flumi) fbeginLumiOfFit = flumi;
//   if (fendLumiOfFit == -1 || fendLumiOfFit < flumi) fendLumiOfFit = flumi;
//   std::cout << "flumi = " <<flumi<<"; fbeginLumiOfFit = " << fbeginLumiOfFit <<"; fendLumiOfFit = "<<fendLumiOfFit<<std::endl;

  //------ Primary Vertices
  edm::Handle< reco::VertexCollection > PVCollection;
  bool hasPVs = false;
  //edm::View<reco::Vertex> vertices;
  //const reco::VertexCollection & vertices = 0;

  if ( iEvent.getByLabel("offlinePrimaryVertices", PVCollection ) ) {
      //pv = *PVCollection;
      //vertices = *PVCollection;
      hasPVs = true;
  }
  //------

  //------ Beam Spot in current event
  edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
  const reco::BeamSpot *refBS =  0;
  if ( iEvent.getByLabel("offlineBeamSpot",recoBeamSpotHandle) )
      refBS = recoBeamSpotHandle.product();
  //-------
  
 
  if ( hasPVs ) {
      
      for (reco::VertexCollection::const_iterator pv = PVCollection->begin(); pv != PVCollection->end(); ++pv ) {

           
           //for ( size_t ipv=0; ipv != pv.size(); ++ipv ) {

          //--- vertex selection
          if ( pv->isFake() || pv->tracksSize()==0 )  return;
          if ( pv->ndof() < minVtxNdf_ || (pv->ndof()+3.)/pv->tracksSize()<2*minVtxWgt_ )  return;
          //---
          
          hPVx->Fill( pv->x(), pv->z() );
          hPVy->Fill( pv->y(), pv->z() );

          BeamSpotFitPVData pvData;
          pvData.position[0] = pv->x();
          pvData.position[1] = pv->y();
          pvData.position[2] = pv->z();
          pvData.posError[0] = pv->xError();
          pvData.posError[1] = pv->yError();
          pvData.posError[2] = pv->zError();
          pvData.posCorr[0] = pv->covariance(0,1)/pv->xError()/pv->yError();
          pvData.posCorr[1] = pv->covariance(0,2)/pv->xError()/pv->zError();
          pvData.posCorr[2] = pv->covariance(1,2)/pv->yError()/pv->zError();
          pvStore_.push_back(pvData);

      }
    
  }
    
    
  

}

bool PVFitter::runFitter() {

    bool fit_ok = false;
  
    if(debug_){
        std::cout << "Calculating beam spot with PVs ..." << std::endl;
    }

    TH1F *h1PVx = (TH1F*) hPVx->ProjectionX("h1PVx", 0, -1, "e");
    TH1F *h1PVy = (TH1F*) hPVy->ProjectionX("h1PVy", 0, -1, "e");
    TH1F *h1PVz = (TH1F*) hPVx->ProjectionY("h1PVz", 0, -1, "e");


    h1PVx->Fit("gaus","QLM0");
    h1PVy->Fit("gaus","QLM0");
    h1PVz->Fit("gaus","QLM0");

    
    if ( ! do3DFit_ ) {

      TF1 *gausx = h1PVx->GetFunction("gaus");
      TF1 *gausy = h1PVy->GetFunction("gaus");
      TF1 *gausz = h1PVz->GetFunction("gaus");
    
      fwidthX = gausx->GetParameter(2);
      fwidthY = gausy->GetParameter(2);
      fwidthZ = gausz->GetParameter(2);
      fwidthXerr = gausx->GetParError(2);
      fwidthYerr = gausy->GetParError(2);
      fwidthZerr = gausz->GetParError(2);
    
    }

    if ( pvStore_.size() <= minNrVertices_ ) return false;

    //
    // LL function and fitter
    //
    FcnBeamSpotFitPV* fcn = new FcnBeamSpotFitPV(pvStore_);
    TFitterMinuit minuitx;
    minuitx.SetMinuitFCN(fcn); 
    //
    // fit parameters: positions, widths, x-y correlations, tilts in xz and yz
    //
    minuitx.SetParameter(0,"x",0.,0.02,-10.,10.);
    minuitx.SetParameter(1,"y",0.,0.02,-10.,10.);
    minuitx.SetParameter(2,"z",0.,0.20,-30.,30.);
    minuitx.SetParameter(3,"ex",0.015,0.01,0.,10.);
    minuitx.SetParameter(4,"corrxy",0.,0.02,-1.,1.);
    minuitx.SetParameter(5,"ey",0.015,0.01,0.,10.);
    minuitx.SetParameter(6,"dxdz",0.,0.0002,-0.1,0.1);
    minuitx.SetParameter(7,"dydz",0.,0.0002,-0.1,0.1);
    minuitx.SetParameter(8,"ez",1.,0.1,0.,30.);
    minuitx.SetParameter(9,"scale",0.9,0.1,0.5,2.);
    //
    // first iteration without correlations
    //
    minuitx.FixParameter(4);
    minuitx.FixParameter(6);
    minuitx.FixParameter(7);
    minuitx.FixParameter(9);
    minuitx.SetMaxIterations(100);
    minuitx.SetPrintLevel(3);
    minuitx.CreateMinimizer();
    minuitx.Minimize();
    //
    // refit with harder selection on vertices
    //
    fcn->setLimits(minuitx.GetParameter(0)-sigmaCut_*minuitx.GetParameter(3),
                   minuitx.GetParameter(0)+sigmaCut_*minuitx.GetParameter(3),
                   minuitx.GetParameter(1)-sigmaCut_*minuitx.GetParameter(5),
                   minuitx.GetParameter(1)+sigmaCut_*minuitx.GetParameter(5),
                   minuitx.GetParameter(2)-sigmaCut_*minuitx.GetParameter(8),
                   minuitx.GetParameter(2)+sigmaCut_*minuitx.GetParameter(8));
    minuitx.Minimize();
    //
    // refit with correlations
    //
    minuitx.ReleaseParameter(4);
    minuitx.ReleaseParameter(6);
    minuitx.ReleaseParameter(7);
    minuitx.Minimize();
    // refit with floating scale factor
//   minuitx.ReleaseParameter(9);
//   minuitx.Minimize();

    if ( do3DFit_ ) {
      fwidthX = minuitx.GetParameter(3);
      fwidthY = minuitx.GetParameter(5);
      fwidthZ = minuitx.GetParameter(8);
      fwidthXerr = minuitx.GetParError(3);
      fwidthYerr = minuitx.GetParError(5);
      fwidthZerr = minuitx.GetParError(8);
    }
    
    pvStore_.clear();
    
    hPVx->Reset();
    hPVy->Reset();
    
    return true;
}

void PVFitter::dumpTxtFile(){
/*
  fasciiFile << "Runnumber " << frun << std::endl;
  fasciiFile << "BeginTimeOfFit " << fbeginTimeOfFit << std::endl;
  fasciiFile << "EndTimeOfFit " << fendTimeOfFit << std::endl;
  fasciiFile << "LumiRange " << fbeginLumiOfFit << " - " << fendLumiOfFit << std::endl;
  fasciiFile << "Type " << fbeamspot.type() << std::endl;
  fasciiFile << "X0 " << fbeamspot.x0() << std::endl;
  fasciiFile << "Y0 " << fbeamspot.y0() << std::endl;
  fasciiFile << "Z0 " << fbeamspot.z0() << std::endl;
  fasciiFile << "sigmaZ0 " << fbeamspot.sigmaZ() << std::endl;
  fasciiFile << "dxdz " << fbeamspot.dxdz() << std::endl;
  fasciiFile << "dydz " << fbeamspot.dydz() << std::endl;
  if (inputBeamWidth_ > 0 ) {
    fasciiFile << "BeamWidthX " << inputBeamWidth_ << std::endl;
    fasciiFile << "BeamWidthY " << inputBeamWidth_ << std::endl;
  } else {
    fasciiFile << "BeamWidthX " << fbeamspot.BeamWidthX() << std::endl;
    fasciiFile << "BeamWidthY " << fbeamspot.BeamWidthY() << std::endl;
  }
	
  for (int i = 0; i<6; ++i) {
    fasciiFile << "Cov("<<i<<",j) ";
    for (int j=0; j<7; ++j) {
      fasciiFile << fbeamspot.covariance(i,j) << " ";
    }
    fasciiFile << std::endl;
  }
  // beam width error
  if (inputBeamWidth_ > 0 ) {
    fasciiFile << "Cov(6,j) 0 0 0 0 0 0 " << "1e-4" << std::endl;
  } else {
    fasciiFile << "Cov(6,j) 0 0 0 0 0 0 " << fbeamspot.covariance(6,6) << std::endl;
  }
  fasciiFile << "EmittanceX " << fbeamspot.emittanceX() << std::endl;
  fasciiFile << "EmittanceY " << fbeamspot.emittanceY() << std::endl;
  fasciiFile << "BetaStar " << fbeamspot.betaStar() << std::endl;

*/
}

