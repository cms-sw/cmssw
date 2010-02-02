//#include "SLHCUpgradeSimulations/L1Trigger/interface/DTStubMatchesCollection.h"
#include "SimDataFormats/SLHC/interface/DTStubMatchesCollection.h"


using namespace std;

// 090320 SV get closest stub 
TrackerStub* DTStubMatchesCollection::getClosestStub(int phi, int theta, int lay) const
{
  int distMin = 1000;
  TrackerStub* stubMin = new TrackerStub(); //_stubs[0]; (Ignazio)

  for(unsigned int s = 0; s<_stubs.size(); s++)
    {
      /* 
	 int phitk = _stubs[s]->phi()*4096;
	 int thetatk = _stubs[s]->theta()*4096;	
	 cout 	<< "stub : phi " << phitk << " theta " << thetatk
	 << " deltaPhi " << (phi-phitk) << " deltaTheta " << (theta - thetatk)
	 << " layer " << _stubs[s]->layer() << endl; 
      */		 
      // Ignazio 091116 adopting our_to_tracker_lay_Id converter
      if( _stubs[s]->layer() == our_to_tracker_lay_Id(lay) )
	{ 
	  int phitk = static_cast<int>(_stubs[s]->phi()*4096.);
	  /*
	    int thetatk = static_cast<int>(_stubs[s]->theta()*4096.);	
	    float dist = 
	    sqrt( (phi-phitk)*(phi-phitk) + (theta - thetatk)*(theta - thetatk) );
	  */
	  int dist = abs(static_cast<int>(phi)-phitk);
	  /*
	    cout << "stub : phi " << phitk << " theta " << thetatk
	    << " deltaPhi " << (phi-phitk) << " deltaTheta " << (theta - thetatk)
	    << " layer " << _stubs[s]->layer() << " dist " << dist << endl;  
	  */		
	  if(dist < distMin)
	    {
	      distMin = dist;
	      stubMin = _stubs[s];
	    }	
	  /*  
	      cout << " dist " << dist << " distMin " << distMin << endl; 
	  */
	}
    }
  return stubMin;
}



// 090320 SV get closest stub - mod PZ 091002
TrackerStub* DTStubMatchesCollection::getClosestPhiStub(int phi, int lay) const
{
  int distMin = 1000;
  TrackerStub* stubMin = new TrackerStub(); //_stubs[0]; (Ignazio)

  for(unsigned int s = 0; s<_stubs.size(); s++)
    {		 
      // Ignazio 091116 adopting our_to_tracker_lay_Id converter
      if( _stubs[s]->layer() ==  our_to_tracker_lay_Id(lay) ) 
	{ 
	  int phitk = static_cast<int>(_stubs[s]->phi()*4096.);
	  int dist = abs(static_cast<int>(phi)-phitk);
/*	  
	    cout << "stub : phi " << phitk // << " theta " << thetatk
	    << " deltaPhi " << (phi-phitk) //<< " deltaTheta " << (theta - thetatk)
	    << " layer " << _stubs[s]->layer() << " dist " << dist << endl;  
*/	  		
	  if(dist < distMin)
	    {
	      distMin = dist;
	      stubMin = _stubs[s];
	    }	
	  /*  
	      cout << " dist " << dist << " distMin " << distMin << endl; 
	  */
	}
    }
  return stubMin;
}
// 090320 SV get closest stub - mod PZ 091002
TrackerStub* DTStubMatchesCollection::getClosestThetaStub(int theta, int lay) const
{
  int distMin = 1000;
  TrackerStub* stubMin = new TrackerStub(); //_stubs[0]; (Ignazio)

  for(unsigned int s = 0; s<_stubs.size(); s++)
    {	 
      // Ignazio 091116 adopting our_to_tracker_lay_Id converter
      if( _stubs[s]->layer() == our_to_tracker_lay_Id(lay) )
	{ 
	  int thetatk = static_cast<int>(_stubs[s]->theta()*4096.);

	  int dist = abs(static_cast<int>(theta)-thetatk);
	  /*
	    cout << "stub : phi " << phitk << " theta " << thetatk
	    << " deltaPhi " << (phi-phitk) << " deltaTheta " << (theta - thetatk)
	    << " layer " << _stubs[s]->layer() << " dist " << dist << endl;  
	  */		
	  if(dist < distMin)
	    {
	      distMin = dist;
	      stubMin = _stubs[s];
	    }	
	  /*  
	      cout << " dist " << dist << " distMin " << distMin << endl; 
	  */
	}
    }
  return stubMin;
}

// 090320 SV get closest stub - mod PZ 091002
TrackerStub* DTStubMatchesCollection::getStub(int lay) const
{
  TrackerStub* stub = new TrackerStub(); //_stubs[0]; (Ignazio)

  for(unsigned int s = 0; s<_stubs.size(); s++)
    {
      // Ignazio 091116 adopting our_to_tracker_lay_Id converter	 
      if( _stubs[s]->layer() == our_to_tracker_lay_Id(lay) )
	stub = _stubs[s];
    }
    
  return stub;
}


// 090320 SV get closest stub - mod PZ 091002
int DTStubMatchesCollection::countStubs(int lay) const
{
  int nstubs =0;
  for(unsigned int s = 0; s<_stubs.size(); s++)
    {
      // Ignazio 091116 adopting our_to_tracker_lay_Id converter	 
      if( _stubs[s]->layer() == our_to_tracker_lay_Id(lay) ) 
	nstubs++;
    }
    
  return nstubs;
}



// 090206 PZ method for DT matches ordering
void DTStubMatchesCollection::orderDTTriggers() 
{
  // cout << "Ordering DT matches" << endl;
  // utility variables
  int DTStubMatch_sort[24][5][12][10];
  int ind[24][5][12];
  
  // order DT phi-eta matches: first by higher code, then by lower phib   
  // initialize and store matches per bx,wh,sect 
  int maxmatch = 10;  
  for(int ibx = 8; ibx < 25; ibx++) {
    for(int iwh = -2; iwh < 3; iwh++) {
      for(int isec = 1; isec < 13; isec++) {
	ind[ibx-8][iwh+2][isec-1] = 0;
      }
    }
  }
  int ndtmatch = numDt();
  for(int ibx = 8; ibx < 25; ibx++) {
    for(int iwh = -2; iwh < 3; iwh++) {
      for(int isec = 1; isec < 13; isec++) {
	for(int dm = 0; dm < ndtmatch ; dm++) {	 
	  if( dtmatch(dm)->bx() == ibx && 
	      dtmatch(dm)->wheel()== iwh && 
	      dtmatch(dm)->sector() == isec) {
	    if(ind[ibx-8][iwh+2][isec-1]<maxmatch)
	      DTStubMatch_sort[ibx-8][iwh+2][isec-1][ind[ibx-8][iwh+2][isec-1]] = dm;
	    ind[ibx-8][iwh+2][isec-1]++;	     
	  }	   
	} 
      }
    }     
  }
  
  // order matches 
  // loop over matches for every bx,wh,sect  
  for(int ibx = 8; ibx < 25; ibx++) {
    for(int iwh = -2; iwh < 3; iwh++) {
      for(int isec = 1; isec < 13; isec++) {
	int im[10];
	for(int i = 0; i<maxmatch; i++) 
	  im[i] = 0;
	int ntrig = ind[ibx-8][iwh+2][isec-1];
	
	// now order all matches
	
	for(int itrig = 0; itrig < ntrig; itrig++) {	 
	  im[itrig] = DTStubMatch_sort[ibx-8][iwh+2][isec-1][itrig];  
	  if(itrig == 0) {
	    // this is the first match; set as first:	  
	    int trig_order = 1;
	    _dtmatches[im[itrig]]->setTrigOrder(trig_order);
	  }	  
	  else if(itrig > 0) {
	    // these are the remaining matches: check against already ordered ones:	  
	    int cod_curr = _dtmatches[im[itrig]]->code();
	    int phib_curr = _dtmatches[im[itrig]]->phib_ts();	    
	    // stop loop when the current trigger is higher grade than already 
	    // ordered ones 
	    int istop = -1;		
	    for(int i = 0; i<itrig; i++){
	      int cod_i = _dtmatches[im[i]]->code();
	      int phib_i = _dtmatches[im[i]]->phib_ts();	
	      
	      //first case: current code is higher than stored code	
	      
	      if( cod_i < cod_curr) {
		istop = i; 
		break;
	      }
	      
	      // second case: code is the same: order by lower bending angle

	      else if(cod_curr == cod_i) 		  
		if(phib_i > phib_curr ) {
		  istop = i; 
		  break;
		}

	    } // end loop over already ordered matches
	    
	    // now set ordinal number
	    
	    // first case: current match is lowest rank
	    if(istop < 0)
	      _dtmatches[im[itrig]]->setTrigOrder(itrig+1);		
	    
	    // second case: current match goes somewhere in the existing list 
	    else {
	      int trig_order = _dtmatches[im[istop]]->trig_order();
	      _dtmatches[im[itrig]]->setTrigOrder(trig_order);
	      for(int i = 0; i<itrig; i++) {
		if(_dtmatches[im[i]]->trig_order() >= trig_order)
		  _dtmatches[im[i]]->setTrigOrder(_dtmatches[im[i]]->trig_order()+1);
	      }
	    }
	  } // end loop over matches following the first at fixed sect, wh and bx 
	} // end loop over all of the matches at fixed sect, wh and bx
      } // end loop over sect
    } // end loop over wh
  } // end loop over bx  
}



void DTStubMatchesCollection::extrapolateDTToTracker()
{
  // cout << "Extrapolating to tracker layers: numDt() = " << numDt() << endl;
  if(numDt()) {
    // loop on tracker layers, loop on dtmatches    
    for(int l=0; l<StackedLayersInUseTotal; l++) {
      for (int dm = 0; dm < numDt(); dm++)
	dtmatch(dm)->extrapolateToTrackerLayer(l);
    }//end loop on layers 
  }//end if _dtmatch  
  return;
}



void DTStubMatchesCollection::eraseDTStubMatch(int dm) 
{  
  //calls _dtmatches[dm] destructor and removes entry from vector
  _dtmatches.erase(_dtmatches.begin()+dm);
  
  return;
}


  
void DTStubMatchesCollection::removeRedundantDTStubMatch() 
{
  cout << " \n\n*** DTStubMatchesCollection::removeRedundantDTStubMatch " << endl;
  // SV 090428 Redundant DTStubMatch cancellation 
  // choose one layer for extrapolation
  int lay = 2; 
  /*
    cout << "BEFORE CANCELLATION: Num DTStubMatch " << numDt() << endl;
    for(int dm = 0; dm < numDt(); dm++){
    cout << "N. " << dm << "  ";
    dtmatch(dm)->print();
    } 
  */  
  // find II tracks in SAME station SAME sector SAME bx and remove single L in anycase
  for (int dmI = 0; dmI < numDt(); dmI++) {
    if( dtmatch(dmI)->flagReject()==false ){
      // record mb I track station, sector and bx
      int stationI = dtmatch(dmI)->station();
      int bxI = dtmatch(dmI)->bx();
      int sectorI = dtmatch(dmI)->sector();
      
      for (int dmII = 0; dmII < numDt(); dmII++) {
	if( dtmatch(dmII)->station() == stationI
	    && dtmatch(dmII)->bx() == bxI
	    && dtmatch(dmII)->sector() == sectorI
	    && dmI != dmII
	    && dtmatch(dmII)->flagReject()==false 
	    && dtmatch(dmII)->code()<=7 )
	  dtmatch(dmII)->setRejection(true);
      }
    }
  }// end L II track rejection 
  
  // collect mb1 and mb2 DTStubMatch at same bx and sector and compare phi, phib 
  for (int dm1 = 0; dm1 < numDt(); dm1++) {
    if( dtmatch(dm1)->station()==1 
	&& dtmatch(dm1)->flagReject()==false ){
      // record mb1 track sector and bx
      int bx1 = dtmatch(dm1)->bx();
      int sector1 = dtmatch(dm1)->sector();      
      // find tracks in mb2 SAME sector SAME bx
      for (int dm2 = 0; dm2 < numDt(); dm2++) {
	if( dtmatch(dm2)->station()==2
	    && dtmatch(dm2)->flagReject()==false 
	    && dtmatch(dm2)->bx() == bx1
	    && dtmatch(dm2)->sector() == sector1 ) {
	  float phi1 = static_cast<float>(dtmatch(dm1)->predPhi(lay)/4096.);
	  float phi2 = static_cast<float>(dtmatch(dm2)->predPhi(lay)/4096.);
	  
	  float phib1 = static_cast<float>(dtmatch(dm1)->phib_ts()/512.);
	  float phib2 = static_cast<float>(dtmatch(dm2)->phib_ts()/512.);	  
	  /*
	    cout << "COMPARING ..." << endl;
	    dtmatch(dm1)->print();
	    cout << "WITH... " << endl;
	    dtmatch(dm2)->print();
	    cout << "DeltaPhi " << fabs(phi1 - phi2) 
	    << " DeltaPhiB " << fabs(phib1 - phib2) << endl;
	  */
	  // remove redundant DTStubMatch: for the moment keep the one with higher quality
	  // remove if inside tolerance and if L in anycase
	  float mean_sigma_phi = 0.05;	// chosen with layer 2
	  float mean_sigma_phib = 0.05;	
	  if((fabs(phi1 - phi2) < (3.*mean_sigma_phi)  &&  
	      fabs(phib1 - phib2) < (3.*mean_sigma_phib))
	     ||
	     (dtmatch(dm2)->code()<=7 || dtmatch(dm1)->code()<=7 )  ) {
	    int dmc = -1;
	    if( dtmatch(dm2)->code() <= dtmatch(dm1)->code() ) {
	      //eraseDTStubMatch(dm2);
	      dtmatch(dm2)->setRejection(true);
	      dmc = dm2;
	    }
	    else{
	      //eraseDTStubMatch(dm1);
	      dtmatch(dm1)->setRejection(true);
	      dmc = dm1;
	    }	    
	    cout << "DTStubMatch " << dmc << " set Rejected ! " << endl;
	  }		
	} //end mb2 selection 
      } //end mb2 loop
    } //end mb1 selection 
  } //end mb1 loop
  
  /* 
     cout << "AFTER CANCELLATION FLAG: Num DTStubMatch " << numDt() << endl;
     for(int dm = 0; dm < numDt(); dm++){
     cout << "N. " << dm << "  ";
     dtmatch(dm)->print();
     }
  */
  /* 
     ATTENTION: erase change pointers!! FIX
     for(int dm = 0; dm < numDt(); dm++)
     if(dtmatch(dm)->flagReject()==true){
     eraseDTStubMatch(dm);	
     break;
    }
    cout << "AFTER CANCELLATION: Num DTStubMatch " << numDt() << endl;
    for(int dm = 0; dm < numDt(); dm++){
    cout << "N. " << dm << "  ";
    dtmatch(dm)->print();
    }
    
  */
  return;
}

//end



// Ignazio ***********************************************************************
void DTStubMatchesCollection::addDT(DTBtiTrigger const bti, 
				    DTChambPhSegm const tsphi,
				    bool debug_dttrackmatch)
{
  int wh   = tsphi.wheel();
  int st   = tsphi.station();
  int se   = tsphi.sector();
  int bx   = tsphi.step();
  int code = tsphi.code()*4;
  if(tsphi.station() == 1) code = code + 2;
  if(bti.code() == 8) code = code + 1; 
  int phi  = tsphi.phi(); 
  int phib = tsphi.phiB();
  float theta = 
    atan( sqrt(bti.cmsPosition().x()*bti.cmsPosition().x() + 
	       bti.cmsPosition().y()*bti.cmsPosition().y())/bti.cmsPosition().z() );
  if(bti.cmsPosition().z() < 0) theta += TMath::Pi();
  bool flagBxOK = false;
  if(bx == 16) flagBxOK = true;
  DTStubMatch* aDTStubMatch = 
    new DTStubMatch(wh, st, se, bx, code, phi, phib, theta, 
		    bti.cmsPosition(), bti.cmsDirection(), flagBxOK);
  _dtmatches.push_back(aDTStubMatch); 
  if(aDTStubMatch->station()==1) _dtmatches_st1++; 
  if(aDTStubMatch->station()==2) _dtmatches_st2++;
  return; 
}



void DTStubMatchesCollection::addDT(const DTBtiTrigger& bti, 
				    const DTTSPhiTrigger& tsphi,
				    bool debug_dttrackmatch)
{
  int wh   = tsphi.wheel();
  int st   = tsphi.station();
  int se   = tsphi.sector();
  int bx   = tsphi.step();
  int code = tsphi.code()*4;
  if(tsphi.station() == 1) code = code + 2;
  if(bti.code() == 8) code = code + 1; 
  int phi  = tsphi.phi(); 
  int phib = tsphi.phiB();
  float theta = 
    atan( sqrt(tsphi.cmsPosition().x()*tsphi.cmsPosition().x() + 
	       tsphi.cmsPosition().y()*tsphi.cmsPosition().y())/tsphi.cmsPosition().z() );
  if(tsphi.cmsPosition().z() < 0) theta += TMath::Pi();
  bool flagBxOK = false;
  if(bx == 16) flagBxOK = true;
  DTStubMatch* aDTStubMatch = 
    new DTStubMatch(wh, st, se, bx, code, phi, phib, theta, 
		    tsphi.cmsPosition(), tsphi.cmsDirection(), flagBxOK);
  _dtmatches.push_back(aDTStubMatch); 
  if(aDTStubMatch->station()==1) _dtmatches_st1++; 
  if(aDTStubMatch->station()==2) _dtmatches_st2++;

  /*
  // OK, this gives 1!
    cout << ( tsphi.cmsDirection().x()*tsphi.cmsDirection().x() + 
	    tsphi.cmsDirection().y()*tsphi.cmsDirection().y() ) << endl;
  */

  return; 
}


/*
void DTStubMatchesCollection::addDT(const DTChambThSegm& tstheta, 
				    const DTTSPhiTrigger& tsphi,
				    bool debug_dttrackmatch)
{
  for(int i=0; i<7; i++) {
    if(tstheta.code(i) > 0) {

      int thwh = tstheta.ChamberId().wheel();
      int thstat = tstheta.ChamberId().station();
      int thsect = tstheta.ChamberId().sector();
      int thqual= tstheta.quality(i);
      int bti_id = (i+1)*8 - 3;
      DTBtiId id = DTBtiId(thwh, thstat, thsect, 2, bti_id);
      DTChamberId chaid = DTChamberId(thwh, thstat, thsect);
      const DTChamber* chamb = muonGeom->chamber(chaid);
      continue;
      DTTrigGeom* _geom = new DTTrigGeom(const_cast<DTChamber*>(chamb), false);
      GlobalPoint  gpbti = _geom->CMSPosition(DTBtiId(chaid, 2, bti_id)); 
      GlobalVector gdbti = GlobalVector(); // ?????????????????????
      continue;
      float thposx = gpbti.x();
      float thposy = gpbti.y();
      float thposz = gpbti.z();  
      int wh   = tsphi.wheel();
      int st   = tsphi.station();
      int se   = tsphi.sector();
      int bx   = tsphi.step();
      int code = tsphi.code()*4 + thqual;	// code 0,1=L, 2,3=H, 4=LL, 5=HL, 6=HH 
      if(tsphi.station() == 1) code = code + 2;
      int phi  = tsphi.phi(); 
      int phib = tsphi.phiB();
      float theta = atan(sqrt( thposx*thposx + thposy*thposy)/thposz );
      if(thposz<0) theta += TMath::Pi();
      bool flagBxOK = false;
      if(bx == 16) flagBxOK = true;
      DTStubMatch* aDTStubMatch = 
	new DTStubMatch(wh, st, se, bx, code, phi, phib, theta, 
			gpbti, gdbti, flagBxOK);
      _dtmatches.push_back(aDTStubMatch); 
      if(aDTStubMatch->station()==1) _dtmatches_st1++; 
      if(aDTStubMatch->station()==2) _dtmatches_st2++;
      delete _geom;
      delete chamb;
    } // end if tstheta.code(i) > 0
  } // end loop over i to get bti_id = (i+1)*8 - 3
}
*/
