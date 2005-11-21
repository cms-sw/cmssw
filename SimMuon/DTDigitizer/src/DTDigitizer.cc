// -*- C++ -*-
//
// Package:    DTDigitizer
// Class:      DTDigitizer
// 
/**\class DTDigitizer DTDigitizer.cc src/DTDigitizer/src/DTDigitizer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Riccardo Bellan
//         Created:  Fri Nov  4 18:56:35 CET 2005
// $Id$
//
//
// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
//#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

//
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/DTSimAlgo/interface/DTGeometry.h"


#include "SimDataFormats/TrackingHit/interface/PSimHitContainer.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "FWCore/Framework/interface/Handle.h"
#include "DataFormats/MuonDetId/interface/DTDetId.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "Geometry/Vector/interface/LocalPoint.h"
#include "Geometry/Vector/interface/LocalVector.h"

//
#include "SimMuon/DTDigitizer/interface/DTDriftTimeParametrization.h"

//FIXME mancano altri include

//FIXME la natura di hit non l'ho ancora determinata, percio' il codice non e' consistente.

DTDigitizer::DTDigitizer(const edm::ParameterSet& iConfig):conf_(iConfig){
  
  cout<<"Creating a DTDigitizer"<<endl;
  
  //register the Producer
  produces<PSimHitContainer>();
  //if do put with a label
  produces<PSimHitContainer>("DTDigitization");
  
  //Parameters:

  // build digis only for mu hits (for debug purposes) 
  onlyMuHits=conf_.getParameter<bool>("onlyMuHits");
  
  // interpolate parametrization function
  interpolate=conf_.getParameter<bool>("interpolate");
  
  // Set verbose output
  debug=conf_.getUntrackedParameter<bool>("debug"); // il default come lo metto? non esiste piu'!!
  
  // Velocity of signal propagation along the wire (cm/ns)
  // For the default value
  // cfr. CMS-IN 2000-021:   (2.56+-0.17)x1e8 m/s
  //      CMS NOTE 2003-17:  (0.244)  m/ns
  vPropWire=conf_.getParameter<double>("vPropWire"); //24.4

  // Dead time for signals on the same wire (number from M. Pegoraro)  
  deadTime=conf_.getParameter<float>("deadTime"); //150

  // further configurable smearing
  smearing=conf_.getParameter<float>("Smearing"); // 3.
}


DTDigitizer::~DTDigitizer(){}


// method called to produce the data
void DTDigitizer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup){
  using namespace edm;
  
  //************ 1 ***************
  
  // create the container for the SimHits
  Handle<PSimHitContainer> simHits;         //edm::Handle has the same implementation as a pointer
  // fill the container
  iSetup.getByLabel("MuonDTHits",simHits); // FIXME-check

  // create the pointer to the Digi container
  std::auto_ptr<DTDigiCollection> output(new DTDigiCollection());
  
  ESHandle<DTGeometry> muonGeom;
  iSetup.get<MuonGeometryRecord>().get(muonGeom);


  //************ 2 ***************

  // These are sorted by DetId, i.e. by layer and then by wire #
  map<DTDetId, vector<PSimHit*> > wireMap;     
  //  map<DTDetId, vector<PSimHit> > detUnitMap;
 
  // loop over the SimHits -  FIXME check the iterator...
  // maybe can be: PSimHitContainer::const_iterator...
  for(PSimHitContainer::const_iterator simHit = simHits->begin();
       simHit != simHits->end(); simHit++){
    
    // Create the id of the wire, the simHits in the DT known also the wireId
    DTDetId wireId(simHit->detUnitId());
    // Fill the map
    WireMap[wireId].push_back(*simHit); 
  }
  
  pair<float,bool> time(0.,false);

  //************ 3 ***************

  // Loop over the wires
  for(map<DTDetId, vector<PSimHit*> >::const_iterator wire =
	WireMap.begin(); wire!=WireMap.end(); wire++){
    // SimHit Container associated to the wire
    const vector<PSimHit*> & vhit = (*wire).second; // perche' by reference??
    if(vhit.size()!=0) {
      TDContainer tdCont; // Is a vector<pair<const PSimHit*,float> >;

      //************ 4 ***************
      DTDetId wireId = (*wire).first;
      DTGeomDetUnit* layer = muonGeom->idToDet(wireId); // FIXME - check,altern ->  wireId.layer() ?
      // Loop on the hits of this wire    
      for (vector<PSimHit>::const_iterator hit=vhit.begin();
	   hit != vhit.end(); hit++){

	//************ 5 ***************
	
	time = computeTime(layer,wireId, *hit); // FIXME
	
	//************ 6 ***************

	if (time.second) {
	  tdCont.push_back(make_pair(*hit,time.first));
	} else {
	  if (debug) cout << "hit discarded" << endl;
	}
      }

      //************ 7 ***************

      // the loading must be done by layer but
      // the digitization must be done by wire (in order to take into account the dead time)

      storeDigis(wireId,wire,WireMap.end(),tdCont,output);
    }
    
  }

  //************ 8 ***************  
  // Load the Digi Container in the Event
  iEvent.put(output);
}

//FIXME PSimHit (pointer-obj) inconsistency:
pair<float,bool> DTDigitizer::computeTime(DTGeomDetUnit* layer,const DTDetId &wireId, const PSimHit *hit) 
{
  LocalPoint entryP = hit->entryPoint();
  LocalPoint exitP = hit->exitPoint();
  int partType = hit->particleType();

  // Check if hits starts/ends on one of the cell's edges

  // FIXME
  DTTopology & topo = layer->specificTopology();
  LocalPoint  wirePos = topo.wirePosition(wireId.wire()); // mettere nell'impl della topolo che accetta il numero del filo nel layer
  
  float xwire = wirePos.x(); 
  float xEntry = entryP.x()-xwire;
  float xExit  = exitP.x()-xwire;

  sides entrySide = onWhichBorder(xEntry,entryP.y(),entryP.z(),topo);
  sides exitSide  = onWhichBorder(xExit,exitP.y(),exitP.z(),topo);

  if (debug) dumpHit(hit, xEntry, xExit,topo);

  // The bolean is used to flag the drift time computation
  pair<float,bool> driftTime(0.,false);  

  // if delta in gas->ignore, since it is included in the parametrisation.
  // FIXME: should check that it is actually a delta ray produced by a nearby
  // muon hit. NON mi e' molto chiara il perche' e' messo in modo esplicito
  if (partType == 11 && entrySide == none) {
    if (debug) cout << "    e- hit in gas; discarding " << endl;
    return driftTime;
  }

  // Local magnetic field  FIXME
  LocalPoint locPt = hit->localPosition();
  //event setu -> Record -> Magnetc Field
  LocalVector BLoc = layer->magneticField(locPt); // FIXME

  /* // per accedere al CM e' facile, cio' che vorrei fare e' pero' mettere questo accesso in DTGeomDetUnit per poter fare
     // layer->magneticField(locPt); 
     
  ESHandle<MagneticField> magnField;
  iSetup.get<IdealMagneticFieldRecord>().get(magnField);
  
  //devo convertire localPt in un GlobalPoint usando la posizione del layer
  
  const GlobalPoint globalPt(0.,0.,0.);
  const GlobalVector BLoc=magnField->inTesla(globalPt);
  */

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
    ( ( entrySide == none || exitSide == none ) // case # 2,3,8,9 or 11
      || (entrySide == exitSide)                   // case # 4 or 10
      || ((entrySide == xMin && exitSide == xMax) || 
	  (entrySide == xMax && exitSide == xMin)) // Hit is case # 7
      );

  // FIXME: now, debug warning only; consider treating those 
  // with TM algo. 
  if ( delta < 0.99996 // Track is not straight. FIXME: use sagitta?
       && (noParametrisation == false)) {
    if (debug) cout << "*** WARNING: hit is not straight, type = " << partType << endl;
  }

  //************ 5A ***************

  if (!noParametrisation) {
    
    LocalVector dir = hit->momentumAtEntry(); // ex  Measurement3DVector dir = hit->measurementDirection();
    float theta = atan(dir.x()/-dir.z())*180/M_PI;

    // FIXME: use dir if M.S. is included as GARFIELD option...
    //        otherwise use hit segment dirction.
    //    LocalVector dir0 = (exitP-entryP).unit();
    //    float theta = atan(dir0.x()/-dir0.z())*180/M_PI;
    float x;
    Local3DPoint pt = hit->localPosition(); //ex     Measurement3DPoint pt = hit->measurementPosition();
    if(fabs(pt.z()) < 0.002) { 
      // hit center within 20 um from z=0, no need to extrapolate.
      x = pt.x();
    } else {
      x = xEntry - (entryP.z()*(xExit-xEntry))/(exitP.z()-entryP.z());
    }
    driftTime = driftTimeFromParametrization(x, theta, By, Bz);
  }

 
  if ((driftTime.second)==false) {
    // Parametrisation not applicable, or failed. Use time map.
    driftTime = driftTimeFromTimeMap();
  }
  
  //************ 5B ***************

  // Signal propagation, TOF etc.
  if (driftTime.second) {
    driftTime.first += externalDelays(stat, wireId, hit);
  }
  return driftTime;
} 


DTDigitizer::sides DTDigitizer::onWhichBorder(float x, float y, float z,
						  DTTopology& topo){

  // epsilon = Tolerance to determine if a hit starts/ends on the cell border.
  // Current value comes from CMSIM, where hit position is
  // always ~10um far from surface. For OSCAR the discrepancy is < 1um.
  const float epsilon = 0.0015; // 15 um
  const float plateThickness = 0.15; // aluminium plate: 1.5 mm
  const float IBeamThickness = 0.1;  // I-beam : 1 mm

  sides side = none;

  if ( fabs(z) > ((topo.cellHeight()-plateThickness)/2.-epsilon)) {
    if (z > 0.) { 
      side = zMax; // This is currently the INNER surface.
    } else {
      side = zMin;
    }
  } else if ( fabs(x) > ((topo.cellWidth()-IBeamThickness)/2.-epsilon)) {
    if (x > 0.) {
      side = xMax; 
    } else {
      side = xMin;
    }
  }   // FIXME: else if ymax, ymin...
  
  return side;
}


//************ 5A ***************

//OK!
pair<float,bool> DTDigitizer::driftTimeFromParametrization(float x, float theta, float By, float Bz) const {

  // Convert from ORCA frame/units r.f. to parametrization ones.
  x *= 10.;  //cm -> mm  FIXME??


  // FIXME: Current parametrisation can extrapolate above 21 mm,
  // however a detailed study is needed before using this.
  if (fabs(x) > 21.) { 
    if (debug) cout << "*** WARNING: parametrisation: x out of range = "
		    << x << ", skipping" << endl;
    return pair<float,bool>(0.f,false);
  }

  // Different r.f. of the parametrization:
  // X_par = X_ORCA; Y_par=Z_ORCA; Z_par = -Y_ORCA  
  //FIXME??
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

//OK!
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

//FIXME PSimHit (pointer-obj) inconsistency:
float DTDigitizer::externalDelays(DTGeomDetUnit* layer, 
				  const DTDetId &wireId, 
				  const PSimHit *hit) const {

  // Time of signal propagation along wire.
  
  // FIXME
  //DTWire* wire = layer->getWire(wireId);  //FIXME
  //DTWireType* wire_type = wire->wireType(); // FIXME
  
  float wireCoord = hit->localPosition().y();
  float halfL     = (wire_type->length())/2.;
  float propgL  ;/*  FIXME = (stat->layer()->getFEPosition() == 
		     MuBarEnum::ZNeg ? halfL + wireCoord : 
		     halfL - wireCoord ); */ 
  float propDelay = propgL/vPropWire;

  // Real TOF.
  float tof = hit->tof();  

  // Delays and t0 according to theSync

  //FIXME Che cosa c'e' adesso??
  double sync = theSync->digitizerOffset(&wireId);

  if (debug) {
    cout << "    propDelay =" << propDelay
	 << "; TOF=" << tof
	 << "; sync= " << sync
	 << endl;
  }
  
  return propDelay + tof + sync;
}


// accumulate digis by layer

void DTDigitizer::storeDigis(DTDetId wireId, 
			     map<DTDetId, vector<PSimHit> >::const_iterator wire,
			     map<DTDetId, vector<PSimHit> >::const_iterator end,
			     TDContainer hits,std::auto_ptr<DTDigiCollection> &output){
  //Just for check poi magari lo tolgo
  static DTDetId lastWireId;
  if(debug)
    if(lastWireId==wireId){
      cout<<"Error in accumulateDigis"<<endl;
      return digis;
    }
    else lastWireId=wireId;


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
      DTDigi digi(wireId, time, digiN);
      
      /* Devo e se si come, tradurre questo?

      // Add association between THIS digi and the corresponding SimTrack
      // FIXME: still, several hits in this cell may have the same
      // SimTrack ID (eg. those coming from secondaries), so the association 
      // is not univoque.
      stat->det().simDet()->addLink(Digi.channel(),
				    (*hit).first->packedTrackId(),1.);
      // int localId = wireId.wire();
      // theAssociationMap.createLinks(localId, stat->det().simDet());
      */

      //************ 7D ***************
      
      static vector<DTDigi> digis;
      static DTDetId lastID;
      
      DTDetId layerID = wireId.layerID();  //taking the layer in which reside the wire
      
      if(lastID==layerID) digis.push_back(digi);
      else{
	if(digis.size()) output.put(digis,layerID); // -> [AA]
	digis.clear();
	digis.push_back(digi);
	lastID=layerID;
      }
      if(wire==(end-1) && digis.size()!=0) output.put(digis,layerID); // -> [AA]
              
      digiN++;
      wakeTime = time + deadTime;
    }
  }


  // _____ [AA] _____ oppure fare ? Molto probabilmente si.
  /*
   DTDigiCollection::Range outputRange;
   outputRange.first = digis.begin();
   outputRange.second = digis.end();
   output.put(outputRange,layerID);
   digis.clear();
  */
}

void DTDigitizer::dumpHit(const PSimHit * hit,
                          float xEntry, float xExit,
                          DTTopology &topo) {
  
  LocalPoint entryP = hit->entryPoint();
  LocalPoint exitP = hit->exitPoint();
  
  sides entrySide = onWhichBorder(xEntry,entryP.y(),entryP.z(),topo);
  sides exitSide  = onWhichBorder(xExit,exitP.y(),exitP.z(),topo);
  //  ProcessTypeEnumerator pTypes;
  
  cout << endl
       << "------- SimHit: " << endl
       << "   Particle type         = " << hit->particleType() << endl
       << "   process type          = " << hit->processType() << endl
       << "   process type          = " << hit->processType() << endl
       << "   packedTrackId         = " << hit->packedTrackId() << endl
       << "   |p|                   = " << hit->pabs() << endl
       << "   Energy loss           = " << hit->energyLoss() << endl
       << "   timeOffset            = " << hit->timeOffset() << endl
       << "   measurementPosition   = " << hit->measurementPosition() << endl
       << "   measurementDirection  = " << hit->measurementDirection() << endl      //FIXME
       << "   localDirection        = " << hit->momentumAtEntry().unit() << endl    //FIXME   non sono sicuro debba essere un versore
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


