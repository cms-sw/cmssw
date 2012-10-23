#include "SimDataFormats/SLHC/interface/DTMatchesCollection.h"
#include "SimDataFormats/SLHC/src/DTUtils.h"

using namespace std;


//090112 PLZ count stubs in matching window
int DTMatchesCollection::nstubsInWindow (int phi, int theta, 
					     int sdtphi, int sdttheta, 
					     int lay) const
{
  int nstubs = 0;
  float nsigmas = 3 ;
  int phimax = static_cast<int>(2*TMath::Pi()*4096.);
  for(unsigned int s = 0; s<_stubs.size(); s++)
    {
      // adopting our_to_tracker_lay_Id converter
      if( _stubs[s]->layer() == our_to_tracker_lay_Id(lay) && _stubs[s]-> PTflag() )
	{ 
	  int phitk = static_cast<int>(_stubs[s]->phi()*4096.);
	  int thetatk = static_cast<int>(_stubs[s]->theta()*4096.);
	  
	  int dist_phi = deltaPhi(phi, phitk);
	  /*
	  int dist_phi = abs(phi-phitk);
	  // Reminder: 2pi round window !!!!!
	  int dist_phi_max = abs(phi+phimax-phitk);
          int dist_phi_min = abs(phi-phimax-phitk);
	  if(dist_phi_max < dist_phi) dist_phi = dist_phi_max;
	  if(dist_phi_min < dist_phi) dist_phi = dist_phi_min;
	  */
	  int dist_theta = abs(theta-thetatk);
	  // check if stub is in window	  
	  float nsphi = 
	    static_cast<float>(dist_phi)/static_cast<float>(sdtphi);
	  float nstheta = 
	    static_cast<float>(dist_theta)/static_cast<float>(sdttheta);
	  if( nsphi <= nsigmas && nstheta <= nsigmas){		      
	    nstubs++;
	    //	    cout << " stub # " << nstubs 
	    //	    << " layer " << _stubs[s]->layer() <<" MC id " << _stubs[s]->MCid() 
	    //	    << " nsphi " << nsphi << " nstheta " << nstheta 
	    //	    << endl;
	  }
	}
    }    
  return nstubs;
}





//100513 PLZ : store stubs in matching window
void DTMatchesCollection::getAllStubsInWindow(int phi, int theta, 
						  int sdtphi,int sdttheta, 
						  int lay) const
{  
  int nstubs = 0;
  TrackerStub* Stubs_in_window[20];
  int phimax = static_cast<int>(2*TMath::Pi()*4096.);
  int nsigmas = 3;
  
  for(unsigned int s = 0; s<_stubs.size(); s++)
    {
      // Ignazio 091116 adopting our_to_tracker_lay_Id converter
      if( _stubs[s]->layer() == our_to_tracker_lay_Id(lay) && _stubs[s]-> PTflag())
	{ 
	  int phitk = static_cast<int>(_stubs[s]->phi()*4096.);
	  int dist_phi = deltaPhi(phi, phitk);
	  /*
	  int dist_phi = abs(phi-phitk);
	  // Reminder: 2pi round window !!!!!
	  int dist_phi_max = abs(phi+phimax-phitk);
          int dist_phi_min = abs(phi-phimax-phitk);
	  if(dist_phi_max < dist_phi) dist_phi = dist_phi_max;
	  if(dist_phi_min < dist_phi) dist_phi = dist_phi_min;
	  */	  
	  int thetatk = static_cast<int>(_stubs[s]->theta()*4096.);
	  int dist_theta = abs(static_cast<int>(theta)-thetatk);	  
	  if(dist_theta < nsigmas*sdttheta && dist_phi < nsigmas*sdtphi) {          
	    if (nstubs < 21) Stubs_in_window[nstubs] = _stubs[s]; 
	    nstubs++;
	  }
	}
    }
  if(nstubs >20) 
    cout << " Warning: too many stubs in matching window: " << nstubs << endl;
}






// 090320 SV get closest stub 
TrackerStub* DTMatchesCollection::getClosestStub(int phi, int theta, 
						     int sdtphi, int sdttheta,
						     int lay) const
{
  int distMin = 10000;
  int phimax = static_cast<int>(2*TMath::Pi()*4096.);
  int nsigmas =3;
  TrackerStub* stubMin = new TrackerStub(); //_stubs[0]; (Ignazio)
  //   cout << "DT   : phi " << phi << " theta " << theta << endl;
  for(unsigned int s = 0; s<_stubs.size(); s++)
    {
      if( _stubs[s]->layer() == our_to_tracker_lay_Id(lay))
	{
	  /* 
	     int phitk = _stubs[s]->phi()*4096;
	     int thetatk = _stubs[s]->theta()*4096;	
	     cout 	<< "stub : phi " << phitk << " theta " << thetatk
	     << " deltaPhi " << (phi-phitk) << " deltaTheta " << (theta - thetatk)
	     << " layer " << _stubs[s]->layer()
	     << " PTflag " << _stubs[s]->PTflag() << endl; 
	  */
	  // Ignazio 091116 adopting our_to_tracker_lay_Id converter
	  if(_stubs[s]-> PTflag())
	    { 
	      int phitk = static_cast<int>(_stubs[s]->phi()*4096.);
	      int dist_phi = deltaPhi(phi, phitk);
	      /*
	      int dist_phi = abs(phi-phitk);
	      // Reminder: 2pi round window !!!!!
	      int dist_phi_max = abs(phi+phimax-phitk);
	      int dist_phi_min = abs(phi-phimax-phitk);
	      if(dist_phi_max < dist_phi) dist_phi = dist_phi_max;
	      if(dist_phi_min < dist_phi) dist_phi = dist_phi_min;
	      */
	      int thetatk = static_cast<int>(_stubs[s]->theta()*4096.);
	      int dist_theta = abs(static_cast<int>(theta)-thetatk);
	      
	      float nsphi =  nsigmas*sdtphi;
	      float nstheta = nsigmas*sdttheta;
	      /*	    
		cout << " phitk " << phitk << " theta " << thetatk 
		<< " phi cut " << nsphi  << " theta cut "  << nstheta << endl;
	      */
	      if(dist_theta < nsigmas*sdttheta && dist_phi < nsigmas*sdtphi) {  
		if(dist_phi < distMin) {
		  distMin = dist_phi;
		  //	      distMin = sqrt(dist_phi*dist_phi+dist_theta*dist_theta);
		  stubMin = _stubs[s];
		}
	      }		      
	      //  cout << " dist " << dist_phi << " distMin " << distMin << endl; 
	    }
	}
    }
  return stubMin;
}






// 090320 SV get closest stub - mod PZ 091002
TrackerStub* DTMatchesCollection::getClosestPhiStub(int phi, int lay) const
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
	  if(dist < distMin) {
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
TrackerStub* DTMatchesCollection::getClosestThetaStub(int theta, int lay) const
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
	  if(dist < distMin) {
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
TrackerStub* DTMatchesCollection::getStub(int lay) const
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





//110112 PLZ count tracklets in matching window
int DTMatchesCollection::ntrackletsInWindow (int phi, int theta, 
						 int sdtphi, int sdttheta, 
						 int superlay) const
{
  int ntracklets = 0;
  float nsigmas = 3 ;
  int phimax = static_cast<int>(2*TMath::Pi()*4096.);
  for(unsigned int s = 0; s<_tracklets.size(); s++)
    {
      // adopting our_to_tracker_superlay_Id converter
      if( _tracklets[s]->sl() == superlay && _tracklets[s]->PTFlag() )
	{ 
	  int phitk = static_cast<int>(_tracklets[s]->phi()*4096.);
	  int thetatk = static_cast<int>(_tracklets[s]->theta()*4096.);
	  
	  int dist_phi = deltaPhi(phi, phitk);
	  /*
	  int dist_phi = abs(phi-phitk);
	  // Reminder: 2pi round window !!!!!
	  int dist_phi_max = abs(phi+phimax-phitk);
          int dist_phi_min = abs(phi-phimax-phitk);
	  if(dist_phi_max < dist_phi) dist_phi = dist_phi_max;
	  if(dist_phi_min < dist_phi) dist_phi = dist_phi_min;
	  */
	  int dist_theta = abs(theta-thetatk);
	  // check if stub is in window	  
	  float nsphi = 
	    static_cast<float>(dist_phi)/static_cast<float>(sdtphi);
	  float nstheta = 
	    static_cast<float>(dist_theta)/static_cast<float>(sdttheta);
	  if( nsphi <= nsigmas && nstheta <= nsigmas){		      
	    ntracklets++;
	    /*	
	      cout << " tracklet # " << ntracklets 
	      << " superlayer " << _tracklets[s]->sl() 
	      << " nsphi " << nsphi << " nstheta " << nstheta 
	      << endl;
	    */
	  }
	}
    }    
  return ntracklets;
}





// 110608 PLZ get closest tracklet 
TrackerTracklet* 
DTMatchesCollection::getClosestTracklet(int phi, int theta, 
					    int sdtphi, int sdttheta,
					    int superlay) const
{
  int distMin = 10000;
  int phimax = static_cast<int>(2*TMath::Pi()*4096.);
  int nsigmas =3;
  TrackerTracklet* TrackletMin = new TrackerTracklet(); //_Tracklets[0]; (Ignazio)
  //   cout << "DT   : phi " << phi << " theta " << theta << endl;
  //   cout << _tracklets.size() << endl;
  for(unsigned int s = 0; s<_tracklets.size(); s++)
    {
      if( _tracklets[s]->sl() == superlay)
      {
	/*	
	  int phitk = _tracklets[s]->phi()*4096;
	  int thetatk = _tracklets[s]->theta()*4096;	
	  cout 	<< "Tracklet : phi " << phitk << " theta " << thetatk
	  << " deltaPhi " << (phi-phitk) << " deltaTheta " << (theta - thetatk)
	  << " layer " << _tracklets[s]->layer()
	  << " PTFlag " << _tracklets[s]->PTFlag()
	  << endl; 
	*/
      if(_tracklets[s]-> PTFlag())
	{ 
	  int phitk = static_cast<int>(_tracklets[s]->phi()*4096.);
	  int dist_phi = deltaPhi(phi, phitk); 
	  /*
	  int dist_phi = abs(phi-phitk);
	  // Reminder: 2pi round window !!!!!
	  int dist_phi_max = abs(phi+phimax-phitk);
          int dist_phi_min = abs(phi-phimax-phitk);
	  if(dist_phi_max < dist_phi) dist_phi = dist_phi_max;
	  if(dist_phi_min < dist_phi) dist_phi = dist_phi_min;
	  */
	  int thetatk = static_cast<int>(_tracklets[s]->theta()*4096.);
	  int dist_theta = abs(static_cast<int>(theta)-thetatk);
	  
	  float nsphi =  nsigmas*sdtphi;
	  float nstheta = nsigmas*sdttheta;
	  // cout << " phitk " << phitk << " theta " << thetatk 
	  // << " phi cut " << nsphi  << " theta cut "  << nstheta << endl;
	  if(dist_theta < nsigmas*sdttheta && dist_phi < nsigmas*sdtphi) {  
	    if(dist_phi < distMin) {
	      distMin = dist_phi;
	      // distMin = sqrt(dist_phi*dist_phi+dist_theta*dist_theta);
	      TrackletMin = _tracklets[s];
	    }
	  }	
	  // cout << " dist " << dist_phi << " distMin " << distMin << endl;  
	}
      }
    }
  return TrackletMin;
}


//110112 PLZ count tracks in matching window
int DTMatchesCollection::ntracksInWindow (int phi, int theta, 
						 int sdtphi, int sdttheta) const
{
  int ntracks = 0;
  float nsigmas = 3 ;
  int phimax = static_cast<int>(2*TMath::Pi()*4096.);
  for(unsigned int s = 0; s<_tracks.size(); s++)
    {
      // adopting our_to_tracker_superlay_Id converter
      if( _tracks[s]->PTFlag() )
	{ 
	  int phitk = static_cast<int>(_tracks[s]->phi()*4096.);
	  int thetatk = static_cast<int>(_tracks[s]->theta()*4096.);
	  
	  int dist_phi = deltaPhi(phi, phitk);
	  /*
	  int dist_phi = abs(phi-phitk);
	  // Reminder: 2pi round window !!!!!
	  int dist_phi_max = abs(phi+phimax-phitk);
          int dist_phi_min = abs(phi-phimax-phitk);
	  if(dist_phi_max < dist_phi) dist_phi = dist_phi_max;
	  if(dist_phi_min < dist_phi) dist_phi = dist_phi_min;
	  */
	  int dist_theta = abs(theta-thetatk);
	  // check if stub is in window	  
	  float nsphi = 
	    static_cast<float>(dist_phi)/static_cast<float>(sdtphi);
	  float nstheta = 
	    static_cast<float>(dist_theta)/static_cast<float>(sdttheta);
	  if( nsphi <= nsigmas && nstheta <= nsigmas){		      
	    ntracks++;
	    /*	
	      cout << " tracklet # " << ntracklets 
	      << " superlayer " << _tracklets[s]->sl() 
	      << " nsphi " << nsphi << " nstheta " << nstheta 
	      << endl;
	    */
	  }
	}
    }    
  return ntracks;
}



//100513 PLZ : store tracks in matching window
void DTMatchesCollection::getAllTracksInWindow(int phi, int theta, 
					       int sdtphi,int sdttheta, vector<TrackerTrack*>& Tracks_in_window,int ntracks) const
{  
//  int ntracks = 0;
// TrackerTrack* Tracks_in_window[20];
  int phimax = static_cast<int>(2*TMath::Pi()*4096.);
  int nsigmas = 3;
  
  for(unsigned int s = 0; s<_tracks.size(); s++)
    {
      if(_tracks[s]-> PTFlag())
	{ 
	  int phitk = static_cast<int>(_tracks[s]->phi()*4096.);
	  int dist_phi = deltaPhi(phi, phitk);
	  	  
	  int thetatk = static_cast<int>(_tracks[s]->theta()*4096.);
	  int dist_theta = abs(static_cast<int>(theta)-thetatk);	  
	  if(dist_theta < nsigmas*sdttheta && dist_phi < nsigmas*sdtphi) {          
	    if (ntracks < 21) Tracks_in_window.push_back(_tracks[s]); 
	    ntracks++;
	  }
	}
    }
  if(ntracks >20) 
    cout << " Warning: too many tracks in matching window: " << ntracks << endl;
}




// 090320 SV get closest stub - mod PZ 091002
int DTMatchesCollection::countStubs(int lay) const
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
void DTMatchesCollection::orderDTTriggers() 
{
  // cout << "Ordering DT matches" << endl;
  // utility variables
  int DTMatch_sort[24][5][12][10];
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
	      DTMatch_sort[ibx-8][iwh+2][isec-1][ind[ibx-8][iwh+2][isec-1]] = dm;
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
	/*
	  ------ now order all matches ------------------------------------------
	  First compare quality code; when these are equal then compare by 
	  bending angle: better grade for higher bending.
	  -----------------------------------------------------------------------
	 */	
	for(int itrig = 0; itrig < ntrig; itrig++) {	 
	  im[itrig] = DTMatch_sort[ibx-8][iwh+2][isec-1][itrig];  
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
	    for(int i = 0; i<itrig; i++)
	      {
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
	  } // end of setting of order position of the current match
	} // end loop for ordering all matches ----------------------------------
      } // end loop over sect
    } // end loop over wh
  } // end loop over bx  
}






void DTMatchesCollection::extrapolateDTToTracker()
{
  // cout << "Extrapolating to tracker layers: numDt() = " << numDt() << endl;
  if(numDt()) {
  
//      extrapolate each dtmatch to vertex
      for (int dm = 0; dm < numDt(); dm++)
	dtmatch(dm)->extrapolateToVertex();
	
//      extrapolate each DTmatch to each tracker layer
    // loop on tracker layers, loop on dtmatches    
    for(int l=0; l<StackedLayersInUseTotal; l++) {
      for (int dm = 0; dm < numDt(); dm++)
	dtmatch(dm)->extrapolateToTrackerLayer(l);
    }//end loop on layers 
  }//end if _dtmatch  
  return;
}






void DTMatchesCollection::eraseDTMatch(int dm) 
{  
  //calls _dtmatches[dm] destructor and removes entry from vector
  _dtmatches.erase(_dtmatches.begin()+dm);
  
  return;
}





  
void DTMatchesCollection::removeRedundantDTMatch() 
{
  // cout << " \n\n*** DTMatchesCollection::removeRedundantDTMatch " << endl;
  // SV 090428 Redundant DTMatch cancellation 
  // choose one layer for extrapolation (central layer for the time being)
  int lay = 2; 
  int nsigma_cut = 3;
  
  // find II tracks in SAME station SAME sector SAME bx and remove single L in anycase
  for (int dmI = 0; dmI < numDt(); dmI++) {
    if( dtmatch(dmI)->flagReject() == false ) {
      // record mb I track station, sector and bx
      int stationI = dtmatch(dmI)->station();
      int bxI = dtmatch(dmI)->bx();
      int sectorI = dtmatch(dmI)->sector();      
      // for (int dmII = 0; dmII < numDt(); dmII++) {
      for (int dmII=(dmI+1); dmII < numDt(); dmII++) {  // Ignazio
	if( dtmatch(dmII)->station() == stationI
	    && dtmatch(dmII)->bx() == bxI
	    && dtmatch(dmII)->sector() == sectorI
	    //&& dmI != dmII                            // Ignazio
	    && dtmatch(dmII)->flagReject() == false 
	    && dtmatch(dmII)->code()<=7 )
	  dtmatch(dmII)->setRejection(true);
      }
    }
  }// end L II track rejection 
  
  // collect mb1 and mb2 DTMatch at same bx in same sector and compare phi, phib 
  for (int dm1 = 0; dm1 < numDt(); dm1++) {
    if( dtmatch(dm1)->station() == 1                 // dm1 --> station1
	&& dtmatch(dm1)->flagReject() == false ) {
      // record mb1 track sector and bx
      int bx1 = dtmatch(dm1)->bx();
      int sector1 = dtmatch(dm1)->sector();      
      // find tracks in mb2: SAME sector SAME bx
      for (int dm2 = 0; dm2 < numDt(); dm2++) {
	if( dtmatch(dm2)->station() == 2             // dm2 --> station2
	    && dtmatch(dm2)->flagReject() == false 
	    && dtmatch(dm2)->bx() == bx1
	    && dtmatch(dm2)->sector() == sector1 ) {
	  // get quantities to compare					
	  int phi1 = dtmatch(dm1)->predPhi(lay);
	  int phi2 = dtmatch(dm2)->predPhi(lay);
	  
	  int theta1 = dtmatch(dm1)->predTheta();
	  int theta2 = dtmatch(dm2)->predTheta();
	  
	  float phib1 = static_cast<float>(dtmatch(dm1)->phib_ts());
	  float phib2 = static_cast<float>(dtmatch(dm2)->phib_ts());
	  
	  // needing a small correction in phi predicted (extrapolation precision?)
	  // correction in phib due to field between ST1 and ST2
	  int dphicor = static_cast<int>(-0.0097*phib1*phib1+1.0769*phib1+4.2324);
	  int dphibcor = static_cast<int>(0.3442*phib1);
 	  
	  int dphi = abs(phi1-phi2)-dphicor;	
	  int dtheta = abs(theta1-theta2);	
	  int dphib = static_cast<int>(fabs(phib1-phib2))-dphibcor;	  
	  // tolerances parameterization
	  int sigma_phi =static_cast<int>(0.006*phib1*phib1+0.4821*phib1+37.64);
	  int sigma_phib =static_cast<int>(0.0005*phib1*phib1+0.01211*phib1+3.4125);
	  int sigma_theta = 100;
	  
	  // remove redundant DTMatch: 
	  // for the moment keep the one with higher quality
	  // remove if inside all tolerances
	  if((dphi < (nsigma_cut*sigma_phi)  &&  
	      dphib < (nsigma_cut*sigma_phib) && 
	      dtheta < (nsigma_cut*sigma_theta))  ) {
	    if( dtmatch(dm2)->code() <= dtmatch(dm1)->code() ) {
	      //eraseDTMatch(dm2);
	      dtmatch(dm2)->setRejection(true);
	      //	    cout << "DTMatch " << dm2 << " set Rejected ! " << endl;
	    }
	    else {
	      //eraseDTMatch(dm1);
	      dtmatch(dm1)->setRejection(true);
	      //	    cout << "DTMatch " << dm1 << " set Rejected ! " << endl;
	    }	    
	  }		
	} //end mb2 selection 
      } //end mb2 loop
    } //end mb1 selection 
  } //end mb1 loop
  
  /*
    cout << "AFTER CANCELLATION FLAG: Num DTMatch " << numDt() << endl;
    for(int dm = 0; dm < numDt(); dm++){
    dtmatch(dm)->print();
    }*/
  
  /* 
     ATTENTION: erase change pointers!! FIX
     for(int dm = 0; dm < numDt(); dm++)
     if(dtmatch(dm)->flagReject()==true){
     eraseDTMatch(dm);	
     break;
     }
     cout << "AFTER CANCELLATION: Num DTMatch " << numDt() << endl;
     for(int dm = 0; dm < numDt(); dm++){
     cout << "N. " << dm << "  ";
     dtmatch(dm)->print();
     }
  */
  return;
}

//end





// Ignazio ***********************************************************************
void DTMatchesCollection::addDT(DTBtiTrigger const bti, 
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
  DTMatch* aDTMatch = 
    new DTMatch(wh, st, se, bx, code, phi, phib, theta, 
		    bti.cmsPosition(), bti.cmsDirection(), flagBxOK);
  _dtmatches.push_back(aDTMatch); 
  if(aDTMatch->station()==1) _dtmatches_st1++; 
  if(aDTMatch->station()==2) _dtmatches_st2++;
  return; 
}





void DTMatchesCollection::addDT(const DTBtiTrigger& bti, 
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
    atan(sqrt(tsphi.cmsPosition().x()*tsphi.cmsPosition().x() + 
	      tsphi.cmsPosition().y()*tsphi.cmsPosition().y())/tsphi.cmsPosition().z() );
  if(tsphi.cmsPosition().z() < 0) theta += TMath::Pi();
  bool flagBxOK = false;
  if(bx == 16) flagBxOK = true;
  DTMatch* aDTMatch = 
    new DTMatch(wh, st, se, bx, code, phi, phib, theta, 
		    tsphi.cmsPosition(), tsphi.cmsDirection(), flagBxOK);
  _dtmatches.push_back(aDTMatch); 
  if(aDTMatch->station()==1) _dtmatches_st1++; 
  if(aDTMatch->station()==2) _dtmatches_st2++;

  /*
  // OK, this gives 1!
    cout << ( tsphi.cmsDirection().x()*tsphi.cmsDirection().x() + 
	    tsphi.cmsDirection().y()*tsphi.cmsDirection().y() ) << endl;
  */

  return; 
}




/*
void DTMatchesCollection::addDT(const DTChambThSegm& tstheta, 
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
      DTMatch* aDTMatch = 
	new DTMatch(wh, st, se, bx, code, phi, phib, theta, 
			gpbti, gdbti, flagBxOK);
      _dtmatches.push_back(aDTMatch); 
      if(aDTMatch->station()==1) _dtmatches_st1++; 
      if(aDTMatch->station()==2) _dtmatches_st2++;
      delete _geom;
      delete chamb;
    } // end if tstheta.code(i) > 0
  } // end loop over i to get bti_id = (i+1)*8 - 3
}
*/

