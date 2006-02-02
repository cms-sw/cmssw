/** \file
 *
 *  $Date: 2006/02/02 18:24:02 $
 *  $Revision: 1.11 $
 *  \authors: G. Bevilacqua, N. Amapane, G. Cerminara, R. Bellan
 */

// system include files
#include <memory>

//C++ headers
#include <cmath>
//
#include <CLHEP/Random/RandGaussQ.h>
#include <CLHEP/Random/RandFlat.h>

// Framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Handle.h"

// Geometry
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTSimAlgo/interface/DTGeometry.h"
#include "Geometry/DTSimAlgo/interface/DTLayer.h"
#include "Geometry/CommonTopologies/interface/DTTopology.h"

#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/LocalVector.h"

// SimHits
#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

// Digis
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
//#include "DataFormats/MuonDetId/interface/DTDetId.h"
#include "DataFormats/MuonDetId/interface/DTWireId.h"
#include "DataFormats/MuonDetId/interface/DTLayerId.h"

// DTDigitizer
#include "SimMuon/DTDigitizer/interface/DTDigiSyncFactory.h"
#include "SimMuon/DTDigitizer/interface/DTDigiSyncBase.h"

#include "SimMuon/DTDigitizer/src/DTDriftTimeParametrization.h"
#include "SimMuon/DTDigitizer/src/DTDigitizer.h"

// namespaces
using namespace edm;
using namespace std;

// Constructor
DTDigitizer::DTDigitizer(const ParameterSet& conf_) {
  
  if (debug) cout<<"Creating a DTDigitizer"<<endl;
  
  //register the Producer with a label
  //  produces<DTDigiCollection>("MuonDTDigis"); // FIXME: Do I pass it by ParameterSet?
  produces<DTDigiCollection>(); // FIXME: Do I pass it by ParameterSet?  

  //Parameters:

  // build digis only for mu hits (for debug purposes) 
  onlyMuHits=conf_.getParameter<bool>("onlyMuHits");
  
  // interpolate parametrization function
  interpolate=conf_.getParameter<bool>("interpolate");
  
  // Set verbose output
  debug=conf_.getUntrackedParameter<bool>("debug"); 
  
  // Velocity of signal propagation along the wire (cm/ns)
  // For the default value
  // cfr. CMS-IN 2000-021:   (2.56+-0.17)x1e8 m/s
  //      CMS NOTE 2003-17:  (0.244)  m/ns
  vPropWire=conf_.getParameter<double>("vPropWire"); //24.4

  // Dead time for signals on the same wire (number from M. Pegoraro)  
  deadTime=conf_.getParameter<double>("deadTime"); //150
  
  // further configurable smearing
  smearing=conf_.getParameter<double>("Smearing"); // 3.

  // Sync Algo
  syncName = conf_.getParameter<string>("SyncName");
  theSync = DTDigiSyncFactory::get()->create(syncName,conf_.getParameter<ParameterSet>("pset"));

}

// Destructor
DTDigitizer::~DTDigitizer(){}

// method called to produce the data
void DTDigitizer::produce(Event& iEvent, const EventSetup& iSetup){

  cout << "--- Run: " << iEvent.id().run()
       << " Event: " << iEvent.id().event() << endl;
  
  //************ 1 ***************
  
  // create the container for the SimHits
  //  Handle<PSimHitContainer> simHits; 
  //  iEvent.getByLabel("r","MuonDTHits",simHits);
    
  // use MixCollection instead of the previous
  Handle<CrossingFrame> xFrame;
  iEvent.getByType(xFrame);
  
  auto_ptr<MixCollection<PSimHit> > 
    simHits( new MixCollection<PSimHit>(xFrame.product(),"MuonDTHits"));
    //    simHits( new MixCollection<PSimHit>(xFrame.product(),"MuonDTHits",pair<int,int>(-1,2)));
  //
  
  // create the pointer to the Digi container
  auto_ptr<DTDigiCollection> output(new DTDigiCollection());
  
  ESHandle<DTGeometry> muonGeom;
  iSetup.get<MuonGeometryRecord>().get(muonGeom);

  //************ 2 ***************

  // These are sorted by DetId, i.e. by layer and then by wire #
  //  map<DTDetId, vector<const PSimHit*> > wireMap;     
  DTWireIdMap wireMap;     
  
  for(MixCollection<PSimHit>::MixItr simHit = simHits->begin();
       simHit != simHits->end(); simHit++){
    
    // Create the id of the wire, the simHits in the DT known also the wireId
    //    DTDetId wireId(simHit->detUnitId());
    DTWireId wireId(simHit->detUnitId());
    // Fill the map
    wireMap[wireId].push_back(&(*simHit));
  }
  
  pair<float,bool> time(0.,false);

  //************ 3 ***************

  // Loop over the wires
  for(DTWireIdMapConstIter wire = wireMap.begin(); wire!=wireMap.end(); wire++){
    // SimHit Container associated to the wire
    const vector<const PSimHit*> & vhit = (*wire).second; 
    if(vhit.size()!=0) {
      TDContainer tdCont; // It is a vector<pair<const PSimHit*,float> >;
      
      //************ 4 ***************
      DTWireId wireId = (*wire).first;

      //FIXME
            const DTLayer* layer = dynamic_cast< const DTLayer* > (muonGeom->idToDet(wireId)); 
      // const DTLayer *layer = new DTLayer(); 

      // Loop on the hits of this wire    
      for (vector<const PSimHit*>::const_iterator hit=vhit.begin();
	   hit != vhit.end(); hit++){
	//************ 5 ***************
	
	time = computeTime(layer,wireId, *hit); 

	//************ 6 ***************
	if (time.second) {
	  tdCont.push_back(make_pair((*hit),time.first));
	} else {
	  if (debug) cout << "hit discarded" << endl;
	}
      }

      //************ 7 ***************

      // the loading must be done by layer but
      // the digitization must be done by wire (in order to take into account the dead time)

      storeDigis(wireId,tdCont,*output);
    }
    
  }

  //************ 8 ***************  
  // Load the Digi Container in the Event
  iEvent.put(output);
}

pair<float,bool> DTDigitizer::computeTime(const DTLayer* layer,
					  const DTWireId &wireId, const PSimHit *hit){
  LocalPoint entryP = hit->entryPoint();
  LocalPoint exitP = hit->exitPoint();
  int partType = hit->particleType();

  // Check if hits starts/ends on one of the cell's edges

  //FIXME
  // const DTTopology &topo = dynamic_cast<DTTopology>( layer->topology() );
  // float xwire = topo.wirePosition(wireId.wire()); 
  
  const DTTopology topo(0,0,0);
  float xwire = 0;
  
  float xEntry = entryP.x()-xwire;
  float xExit  = exitP.x()-xwire;

  DTTopology::Side entrySide = topo.onWhichBorder(xEntry,entryP.y(),entryP.z());
  DTTopology::Side exitSide  = topo.onWhichBorder(xExit,exitP.y(),exitP.z());

  //very temp
  //cout<<"###############"<<xEntry<<"\t\t"<<entryP.z()<<"\t\t"
  //   <<xExit<<"\t\t"<<exitP.z()<<"\t\t"<<partType<<endl; //"\t\t"<<(int)entrySide<<"\t\t"<<(int)exitSide<<endl;
  

  if (debug) dumpHit(hit, xEntry, xExit,topo);

  // The bolean is used to flag the drift time computation
  pair<float,bool> driftTime(0.,false);  

  // if delta in gas->ignore, since it is included in the parametrisation.
  // FIXME: should check that it is actually a delta ray produced by a nearby
  // muon hit. 
  if (partType == 11 && entrySide == DTTopology::none) {
    if (debug) cout << "    e- hit in gas; discarding " << endl;
    return driftTime;
  }
  
  //  LocalPoint locPt = hit->localPosition();

  // Local magnetic field  FIXME
  //  ESHandle<MagneticField> magnField;
  //  iSetup.get<IdealMagneticFieldRecord>().get(magnField);
  //  const LocalVector BLoc=layer->toLocal(magnField->inTesla(layer->toGlobal(locPt)));
  
  // FIXME
  LocalVector BLoc;


  float By = BLoc.y();
  float Bz = BLoc.z();

  // Radius and sagitta according to direction of momentum
  // (just for printing)
  // NOTE: in cmsim, d is always taken // pHat!
  LocalVector d = (exitP-entryP);
  LocalVector pHat = hit->localDirection().unit();
  LocalVector hHat = (d.cross(pHat.cross(d))).unit();
  float cosAlpha = hHat.dot(pHat);
  float sinAlpha = sqrt(1.-cosAlpha*cosAlpha);
  float radius_P = (d.mag())/(2.*cosAlpha);
  float sagitta_P = radius_P*(1.-sinAlpha);

  // Radius, sagitta according to field bending
  // (just for printing)
  float halfd = d.mag()/2.;
  float BMag = BLoc.mag();
  LocalVector pT = (pHat - (BLoc.unit()*pHat.dot(BLoc.unit())))*(hit->pabs());
  float radius_B = (pT.mag()/(0.3*BMag))*100.;
  float sagitta_B;
  if (radius_B > halfd) {
    sagitta_B = radius_B - sqrt(radius_B*radius_B - halfd*halfd);
  } else {
    sagitta_B = radius_B;
  }

  // cos(delta), delta= angle between direction at entry and hit segment
  // (just for printing)
  float delta = pHat.dot(d.unit());
  if (debug) cout << "   delta                 = " << delta  << endl
		  << "   cosAlpha              = " << cosAlpha << endl
		  << "   sinAlpha              = " << sinAlpha << endl
		  << "   pMag                  = " << pT.mag() << endl
		  << "   bMag                  = " << BMag << endl
		  << "   pT                    = " << pT << endl
		  << "   halfd                 = " << halfd << endl
		  << "   radius_P  (cm)        = " << radius_P << endl
		  << "   sagitta_P (um)        = " << sagitta_P*10000. << endl
		  << "   radius_B  (cm)        = " << radius_B << endl
		  << "   sagitta_B (um)        = " << sagitta_B*10000. << endl;

  // Select cases where parametrization can not be used.
  bool noParametrisation = 
    ( ( entrySide == DTTopology::none || exitSide == DTTopology::none ) // case # 2,3,8,9 or 11
      || (entrySide == exitSide)                   // case # 4 or 10
      || ((entrySide == DTTopology::xMin && exitSide == DTTopology::xMax) || 
	  (entrySide == DTTopology::xMax && exitSide == DTTopology::xMin)) // Hit is case # 7
      );

  // FIXME: now, debug warning only; consider treating those 
  // with TM algo. 
  if ( delta < 0.99996 // Track is not straight. FIXME: use sagitta?
       && (noParametrisation == false)) {
    if (debug) cout << "*** WARNING: hit is not straight, type = " << partType << endl;
  }

  //************ 5A ***************

  if (!noParametrisation) {
    
    LocalVector dir = hit->momentumAtEntry(); // ex Measurement3DVector dir = hit->measurementDirection(); //FIXME
    float theta = atan(dir.x()/-dir.z())*180/M_PI;

    // FIXME: use dir if M.S. is included as GARFIELD option...
    //        otherwise use hit segment dirction.
    //    LocalVector dir0 = (exitP-entryP).unit();
    //    float theta = atan(dir0.x()/-dir0.z())*180/M_PI;
    float x;

    // FIXME they aren't the same thing. It is not a problem: I can subtract the xWirein the following...
    Local3DPoint pt = hit->localPosition(); //ex Measurement3DPoint pt = hit->measurementPosition(); // FIXME
    
    if(fabs(pt.z()) < 0.002) { 
      // hit center within 20 um from z=0, no need to extrapolate.
      x = pt.x() - xwire;
    } else {
      x = xEntry - (entryP.z()*(xExit-xEntry))/(exitP.z()-entryP.z());
    }
    driftTime = driftTimeFromParametrization(x, theta, By, Bz);

    //very temp
    //    cout<<"###############"<<xEntry<<"\t\t"<<entryP.z()<<"\t\t"
    //<<xExit<<"\t\t"<<exitP.z()<<"\t\t"<<partType<<"\t"<<(int)entrySide<<"\t"<<(int)exitSide<<endl;


    //    if(abs(partType)==13 && (int)entrySide==0 && (int)exitSide==1) cout<<"############### ";
    //cout<<wireId<<endl;
  }

 
  if ((driftTime.second)==false) {
    // Parametrisation not applicable, or failed. Use time map.
    driftTime = driftTimeFromTimeMap();
  }
  
  //************ 5B ***************

  // Signal propagation, TOF etc.
  if (driftTime.second) {
    driftTime.first += externalDelays(topo,layer,wireId,hit);
  }
  return driftTime;
} 

//************ 5A ***************

pair<float,bool> DTDigitizer::driftTimeFromParametrization(float x, float theta, float By, float Bz) const {

  // Convert from CMSSW frame/units r.f. to parametrization ones.
  x *= 10.;  //cm -> mm 

  // FIXME: Current parametrisation can extrapolate above 21 mm,
  // however a detailed study is needed before using this.
  if (fabs(x) > 21.) { 
    if (debug) cout << "*** WARNING: parametrisation: x out of range = "
		    << x << ", skipping" << endl;
    return pair<float,bool>(0.f,false);
  }

  // Different r.f. of the parametrization:
  // X_par = X_ORCA; Y_par=Z_ORCA; Z_par = -Y_ORCA  

  float By_par = Bz;  // Bnorm
  float Bz_par = -By; // Bwire
  float theta_par = theta;

  // Parametrisation uses interpolation up to |theta|=45 deg,
  // |Bwire|=0.4, |Bnorm|=0.75; extrapolation above.
  if (fabs(theta_par)>45.) {
    if (debug) cout << "*** WARNING: extrapolating theta > 45: "
		    << theta << endl;
    // theta_par = min(fabs(theta_par),45.f)*((theta_par<0.)?-1.:1.);
  }
  if (fabs(By_par)>0.75) {
    if (debug) cout << "*** WARNING: extrapolating Bnorm > 0.75: "
		    << By_par << endl;
    // By_par = min(fabs(By_par),0.75f)*((By_par<0.)?-1.:1.);
  }
  if (fabs(Bz_par)>0.4) {
    if (debug) cout << "*** WARNING: extrapolating Bwire >0.4: "
		    << Bz_par << endl;
    // Bz_par = min(fabs(Bz_par),0.4)*((Bz_par<0.)?-1.:1.);
  }

  DTDriftTimeParametrization::drift_time DT;
  static DTDriftTimeParametrization par;
  unsigned short flag = par.MB_DT_drift_time (x, theta_par, By_par, Bz_par, 0, &DT, interpolate);

  if (debug) {
    cout << "    Parametrisation: x, theta, Bnorm, Bwire = "
	 << x << " " <<  theta_par << " " << By_par << " " << Bz_par << endl
	 << "  time=" << DT.t_drift
	 << "  sigma_m=" <<  DT.t_width_m
	 << "  sigma_p=" <<  DT.t_width_p << endl;
    if (flag!=1) {
      cout << "*** WARNING: call to parametrisation failed" << endl;
      return pair<float,bool>(0.f,false); 
    }
  }

  // Double half-gaussian smearing
  float time = asymGausSmear(DT.t_drift, DT.t_width_m, DT.t_width_p);

  if (debug) cout << "  drift time = " << time << endl;

  // Do not allow the smearing to lead to negative values
  return pair<float,bool>(max(time,0.f),true); 

}

float DTDigitizer::asymGausSmear(double mean, double sigmaLeft, double sigmaRight) const {

  double f = sigmaLeft/(sigmaLeft+sigmaRight);
  double t, u;
  u = RandGaussQ::shoot(0.,smearing);
  if (RandFlat::shoot() <= f) {
    t = RandGaussQ::shoot(mean,sigmaLeft);
    t = mean - fabs(t - mean) + u;
  } else {
    t = RandGaussQ::shoot(mean,sigmaRight);
    t = mean + fabs(t - mean) + u;
  }
  return static_cast<float>(t);
}

pair<float,bool> DTDigitizer::driftTimeFromTimeMap() const {
  // FIXME: not yet implemented.
  if (debug) cout << "   TimeMap " << endl;
  return pair<float,bool>(0.,false);
}

//************ 5B ***************

float DTDigitizer::externalDelays(const DTTopology &topo,
				  const DTLayer* layer,
				  const DTWireId &wireId, 
				  const PSimHit *hit) const {
  
  // Time of signal propagation along wire.
  
  float wireCoord = hit->localPosition().y();
  float halfL     = (topo.cellLenght())/2.;
  float propgL = halfL + wireCoord; // the FE is always located at the neg coord.

  float propDelay = propgL/vPropWire;

  // Real TOF.
  float tof = hit->tof();  

  // Delays and t0 according to theSync

  double sync= theSync->digitizerOffset(&wireId,layer);


  if (debug) {
    cout << "    propDelay =" << propDelay
	 << "; TOF=" << tof
	 << "; sync= " << sync
	 << endl;
  }
  
  return propDelay + tof + sync;
}


// accumulate digis by layer

void DTDigitizer::storeDigis(DTWireId &wireId, 
			     TDContainer &hits,
			     DTDigiCollection &output){

  //************ 7A ***************

  // sort signal times
  sort(hits.begin(),hits.end(),hitLessT());

  //************ 7B ***************

  float wakeTime = -999999.0;
  int digiN = 0; // Digi identifier within the cell (for multiple digis)

  // loop over signal times and drop signals inside dead time
  for ( TDContainer::const_iterator hit = hits.begin() ; hit != hits.end() ; 
	hit++ ) {

    if (onlyMuHits && abs((*hit).first->particleType())!=13) continue;

    //************ 7C ***************
	
    float time = (*hit).second;
    if ( time > wakeTime ) {
      // Note that digi is constructed with a float value (in ns)
      DTDigi digi(wireId.wire(), time, digiN);
      
      if(debug){
	cout<<"--------------"<<endl;
	cout << "Digi time " << digi.time() << endl;
	cout<<"id: "<<wireId<<endl;
      }

      // FIXME- not yet ported in CMSSW

      // Add association between THIS digi and the corresponding SimTrack
      // FIXME: still, several hits in this cell may have the same
      // SimTrack ID (eg. those coming from secondaries), so the association 
      // is not univoque.
      // stat->det().simDet()->addLink(Digi.channel(),
      // (*hit).first->packedTrackId(),1.);
      // int localId = wireId.wire();
      // theAssociationMap.createLinks(localId, stat->det().simDet());

      //************ 7D ***************

      DTLayerId layerID = wireId.layerId();  //taking the layer in which reside the wire
      output.insertDigi(layerID, digi); // ordering Digis by layer
      digiN++;
      wakeTime = time + deadTime;
    }
  }
  
}

void DTDigitizer::dumpHit(const PSimHit * hit,
                          float xEntry, float xExit,
                          const DTTopology &topo) {
  
  LocalPoint entryP = hit->entryPoint();
  LocalPoint exitP = hit->exitPoint();
  
  DTTopology::Side entrySide = topo.onWhichBorder(xEntry,entryP.y(),entryP.z());
  DTTopology::Side exitSide  = topo.onWhichBorder(xExit,exitP.y(),exitP.z());
  //  ProcessTypeEnumerator pTypes;
  
  cout << endl
       << "------- SimHit: " << endl
       << "   Particle type         = " << hit->particleType() << endl
       << "   process type          = " << hit->processType() << endl
       << "   process type          = " << hit->processType() << endl
    // << "   packedTrackId         = " << hit->packedTrackId() << endl
       << "   trackId               = " << hit->trackId() << endl // new,is the same as the
                                                                  // previous?? FIXME-Check
       << "   |p|                   = " << hit->pabs() << endl
       << "   Energy loss           = " << hit->energyLoss() << endl
    // << "   timeOffset            = " << hit->timeOffset() << endl
    // << "   measurementPosition   = " << hit->measurementPosition() << endl
    // << "   measurementDirection  = " << hit->measurementDirection() << endl      //FIXME
       << "   localDirection        = " << hit->momentumAtEntry().unit() << endl    //FIXME is it a versor?
       << "   Entry point           = " << entryP << " cell x = " << xEntry << endl
       << "   Exit point            = " << exitP << " cell x = " << xExit << endl
       << "   DR =                  = " << (exitP-entryP).mag() << endl
       << "   Dx =                  = " << (exitP-entryP).x() << endl
       << "   Cell w,h,l            = (" << topo.cellWidth()
       << " , " << topo.cellHeight() 
       << " , " << topo.cellLenght() << ") cm" << endl
       << "   DY entry from edge    = " << topo.cellLenght()/2.-fabs(entryP.y())
       << "   DY exit  from edge    = " << topo.cellLenght()/2.-fabs(exitP.y())
       << "   entrySide = "  << (int)entrySide
       << " ; exitSide = " << (int)exitSide << endl;

}

