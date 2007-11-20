
// -*- C++ -*-
//
// Package:    IpsTOA
// Class:      IpsTOA
// 
/**\class IpsTOA IpsTOA.cc temp/TrackOriginAnalyzer/src/IpsTOA.cc

 Description: This analyzer calculates the relative composition from a list of
 particle categories.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Victor Bazterra
//         Created:  Tue Mar 13 14:15:40 CDT 2007
// $Id: IpsTOA.cc,v 1.2 2007/10/10 19:02:53 bazterra Exp $
//
//

// system include files
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "TFile.h"
#include "TH1D.h"

#include "SimTracker/TrackHistory/interface/TrackOrigin.h"

// user include files
#include "DataFormats/BTauReco/interface/JetTracksAssociation.h"
#include "DataFormats/BTauReco/interface/TrackCountingTagInfo.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BTauReco/interface/TrackIPTagInfo.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include <DataFormats/Common/interface/View.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "RecoBTag/Analysis/interface/Tools.h"
#include "RecoBTag/BTagTools/interface/SignedDecayLength3D.h"
#include "RecoBTag/BTagTools/interface/SignedImpactParameter3D.h"

#include "RecoVertex/PrimaryVertexProducer/interface/PrimaryVertexSorter.h"

#include "SimTracker/Records/interface/TrackAssociatorRecord.h"

#include "SimDataFormats/TrackingAnalysis/interface/TrackingParticle.h"
#include "SimDataFormats/TrackingAnalysis/interface/TrackingVertexContainer.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"

#include "SimTracker/TrackAssociation/interface/TrackAssociatorBase.h"

#include "TrackingTools/Records/interface/TransientTrackRecord.h"

#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"

//
// class decleration
//

class IpsTOA : public edm::EDAnalyzer 
{

public:

  explicit IpsTOA(const edm::ParameterSet&);
  ~IpsTOA();

private:

  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob();

  // VEB
  double momentum_;

  // Member data
  
  typedef std::set<int> sint;
  typedef std::vector<double> vdouble;
  typedef std::vector<std::string> vstring;
  typedef std::vector<edm::ParameterSet> vParameterSet;

  std::string trackCollection_;
  std::string tagInfoCollection_;
  
  std::string rootFile_;
  bool antiparticles_;
  bool status_;
  
  struct range_t
  {
    bool ips2d;
    double minIps, maxIps;
    vstring vetoList, typeList;
    
    range_t(
      bool ips2d_,
      double minIps_,
      double maxIps_,
      vstring vetoList_,
	  vstring typeList_
    )
    {
      ips2d = ips2d_;
      minIps = minIps_;
      maxIps = maxIps_;
      vetoList = vetoList_;
      typeList = typeList_;
    }
    
    range_t(const range_t & orig)
    {
      ips2d = orig.ips2d;
      minIps = orig.minIps;
      maxIps = orig.maxIps;
      vetoList = orig.vetoList;
      typeList = orig.typeList;
    }
  };
  
  typedef std::vector<range_t> vrange_t;  
  
  vrange_t ranges_;

  bool associationByHits_;
  TrackAssociatorBase * associator_;
    
  // Loop in different track collections
  
  void LoopOverTrackCountingInfo(
    std::size_t,
    reco::RecoToSimCollection &,
    reco::TrackIPTagInfoCollection::const_iterator,
    reco::TrackIPTagInfoCollection::const_iterator
  );
  
  // Track origin

  struct counter_info_t
  {
    int pdgId;
    int tracks;
     						          
    counter_info_t(int pdgId_, int tracks_)
    {
      pdgId = pdgId_;
      tracks = tracks_;
    }

    bool operator< (const counter_info_t & i) const
    {
      if (pdgId < i.pdgId)
        return true;
      else if (pdgId > i.pdgId)
        return false;
 
      if (tracks < i.tracks)
        return true;
	
      return false;
    }          
  }; 

  typedef std::map<int,int>  counter_index_t;
  typedef std::pair<int,int> counter_index_pair_t;

  typedef std::map<int,std::size_t>  counter_buffer_t;
  typedef std::pair<int,std::size_t> counter_buffer_pair_t;

  typedef std::map<counter_info_t,std::size_t>  counter_t;
  typedef std::pair<counter_info_t,std::size_t> counter_pair_t;
 
  bool insideJet_;
 
  std::vector<counter_t> counters_;
   
  counter_index_t counter_index_; 
  counter_buffer_t counter_buffer_;
   
  edm::ESHandle<ParticleDataTable> pdt_;
  
  void InitCounter();
  void UpdateCounter(std::size_t);
  void Count(int barcode=0,int pdgId=0);

};

//
// constructors and destructor
//

IpsTOA::IpsTOA(const edm::ParameterSet& iConfig)
{
  trackCollection_   = iConfig.getParameter<std::string>( "trackCollection" );
  tagInfoCollection_ = iConfig.getParameter<std::string> ( "tagInfoCollection" ); 

  rootFile_ = iConfig.getParameter<std::string> ( "rootFile" );
  
  insideJet_         = iConfig.getParameter<bool> ( "insideJet" );
  antiparticles_     = iConfig.getParameter<bool> ( "antiparticles" );
  associationByHits_ = iConfig.getParameter<bool> ( "associationByHits" );
  
  status_ = iConfig.getParameter<bool> ( "status2" );
   
  vstring rangeNames;
  std::size_t rangeNumber = iConfig.getParameterSetNames ( rangeNames );
  
  ranges_.reserve(rangeNumber);

  edm::ParameterSet pset;
  
  counters_.resize(rangeNumber);

  for(std::size_t i=0; i<rangeNumber; i++)
  {
    pset = iConfig.getParameter<edm::ParameterSet> ( rangeNames[i] );
    ranges_.push_back(
      range_t(
        pset.getParameter<bool> ("ips2d"),
        pset.getParameter<double> ("minIps"),
        pset.getParameter<double> ("maxIps"),
        pset.getParameter<vstring> ("veto"),
        vstring()
      )
    );
  }
}

IpsTOA::~IpsTOA() { }

//
// member functions
//

// ------------ method called to for each event  ------------
void
IpsTOA::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  // Tracking particle information
  edm::Handle<TrackingParticleCollection> TPCollection;  
  iEvent.getByType(TPCollection);

  // Track collection
  edm::Handle<reco::TrackCollection> trackCollection;
  iEvent.getByLabel(trackCollection_,trackCollection);

  // Track impact parameters tag info
  edm::Handle<reco::TrackIPTagInfoCollection> tagInfoCollection;
  iEvent.getByLabel(tagInfoCollection_, tagInfoCollection);	
 
  reco::RecoToSimCollection association = associator_->associateRecoToSim ( trackCollection, TPCollection, &iEvent ); 

  std::cout << std::endl;
  std::cout << "New event" << std::endl;

  for(std::size_t cid=0; cid<ranges_.size(); cid++) 
  {
    // loop over all TrackCountingInfo (there is one per jet).
    LoopOverTrackCountingInfo(
      cid,
      association,
      tagInfoCollection->begin(),
      tagInfoCollection->end()
    );
  }
}    


// ------------ method called once each job just before starting event loop  ------------
void 
IpsTOA::beginJob(const edm::EventSetup& iSetup) 
{
  // Get the associator by hits
  edm::ESHandle<TrackAssociatorBase> associator;

  if ( !trackCollection_.empty() )
  {
    if(associationByHits_)
    {  
      iSetup.get<TrackAssociatorRecord>().get("TrackAssociatorByHits",associator);
      associator_ = (TrackAssociatorBase *) associator.product();
    }
    else  
    {
      iSetup.get<TrackAssociatorRecord>().get("TrackAssociatorByChi2",associator);
      associator_ = (TrackAssociatorBase *) associator.product();  
    }
  }
  
  // Get the particles table.
  iSetup.getData( pdt_ );
}


// ------------ method called once each job just after ending the event loop  ------------
void 
IpsTOA::endJob() 
{
  TFile file(rootFile_.c_str(), "RECREATE"); 
  file.cd();

  double vetoedVals;
  double totalVals;

  std::multimap<double, std::string> particleTypes;

  for(std::size_t cid = 0; cid < ranges_.size(); cid++)
  {  
    particleTypes.clear();
    vetoedVals = 0; totalVals = 0;
    
    std::cout << "List of all long lived particle found" << std::endl;
    for ( counter_t::iterator it = counters_[cid].begin(); it != counters_[cid].end(); it++)
    {
      std::ostringstream particle;                    
      if ( !it->first.pdgId )
      {
        particle << "Fake" << std::endl;
        std::cout << " fake tracks -> " << it->second << std::endl;    
      }
      else
      { 
        particle << pdt_->particle(HepPDT::ParticleID(it->first.pdgId))->name(); 
        std::cout << " particle " << particle.str() << " associated to " << it->first.tracks << " tracks -> " << it->second << std::endl;
      }

      if (
        std::find (
          ranges_[cid].vetoList.begin(),
          ranges_[cid].vetoList.end(),
          particle.str()
      ) == ranges_[cid].vetoList.end()
      )
      {
        particle << "#" << it->first.tracks;
        particleTypes.insert(std::pair<double,std::string>(it->second, particle.str()));
        totalVals += it->second;
      }
      else 
      {
        vetoedVals += it->second;
        totalVals += it->second;
      }	      
    }

    std::cout << " number of vetoed tracks " << vetoedVals << std::endl;
    std::cout << " total number of tracks " << totalVals << std::endl;
    std::cout << " % of vetoed tracks " << ((vetoedVals/totalVals) * 100) << std::endl;
        
    std::ostringstream hName, hTitle;

    if ( trackCollection_.empty() ) 
    {
      hTitle << "Track origins for the whole TrackingParticle collection";
      hName  << "TrackingParticleCollection";
    }
    else if( tagInfoCollection_.empty() )
    {
      hTitle << "Track origins for the whole track collection";
      hName  << "recoTrackCollection";
    } 
    else if( ranges_[cid].ips2d )
    {
      hTitle << "reco::track origins with 2D IPS [" << ranges_[cid].minIps << "," << ranges_[cid].maxIps << "]";
      hName  << "ips2d_" << ranges_[cid].minIps << "_" << ranges_[cid].maxIps;
    }
    else
    {
      hTitle << "reco::track origins with 3D IPS [" << ranges_[cid].minIps << "," << ranges_[cid].maxIps << "]";
      hName  << "ips3d_" << ranges_[cid].minIps << "_" << ranges_[cid].maxIps;
    }
    
    for(std::size_t v=0; v<ranges_[cid].vetoList.size(); v++)
    {
      hTitle << "_" << ranges_[cid].vetoList[v];
      hName << "_" << ranges_[cid].vetoList[v];
    }

  	// creating and saving the pie
    TH1D histogram(
      hName.str().c_str(),
      hTitle.str().c_str(),
      particleTypes.size(), 
      0., 
      Double_t(particleTypes.size()) 
    );
	
	  // creating and saving the histogram
    int i = 1;    
    std::cout << "Particle size " <<  particleTypes.size() << std::endl;
    
    std::map<double, std::string>::const_iterator it;

    for(it = particleTypes.begin(); it != particleTypes.end(); it++, i++)
    {
      histogram.GetXaxis()->SetBinLabel(i, it->second.c_str());
      histogram.SetBinContent(i, Double_t(it->first));
    }
    
    histogram.Write();
    file.Flush();
  }
}


void 
IpsTOA::LoopOverTrackCountingInfo(
  std::size_t cid,
  reco::RecoToSimCollection & association,
  reco::TrackIPTagInfoCollection::const_iterator tagInfo,
  reco::TrackIPTagInfoCollection::const_iterator tagInfoEnd
)
{
  bool init;
  float ips;
  std::size_t i, j; 

  // Initialive the TrackOrigin object.
  int status;
  if (status_)
    status = -2;
  else
    status = -1;
    
  TrackOrigin tracer(status);

  if(!insideJet_) InitCounter();

  for(; tagInfo != tagInfoEnd; tagInfo++)
  {
    j = 0; init = true; 

    reco::TrackRefVector tracks = tagInfo->selectedTracks();
    const std::vector<reco::TrackIPTagInfo::TrackIPData> & ipData = tagInfo->impactParameterData();

    std::vector<std::size_t> indexes;
    
    if(ranges_[cid].ips2d)
      indexes = tagInfo->sortedIndexes(reco::TrackIPTagInfo::IP2DSig);
    else
      indexes = tagInfo->sortedIndexes();

    std::cout << std::endl;
    std::cout << "New Jet" << std::endl;

    // counter initialization
    if (insideJet_) InitCounter();

    for(i=0; i < tracks.size(); i++)
    {
      if(ranges_[cid].ips2d)
        ips = ipData[indexes[i]].ip2d.value();
      else
	ips = ipData[indexes[i]].ip3d.value();

      std::cout << "TMP IPS " << ips << std::endl;
      if( init && ips < ranges_[cid].maxIps )
      {
        j = i;
        init = false;
      }      
      if ( ips == -100.0 || ips < ranges_[cid].minIps )
        break;
    }

    std::cout << "TMP INDEX " << j << " " << i << std::endl;

    // there is at least one track with IPS in the range
    if(i > j)
      for(std::size_t k=j; k<i; k++)
      {        
        // If the track is not fake then get the orginal particles
        if (tracer.evaluate(tracks[indexes[k]], association, associationByHits_))
        {
          const HepMC::GenParticle * particle = tracer.particle();
          // If the origin can be determined then take the first particle as the original
          if (particle)
            Count(particle->barcode(), particle->pdg_id());
        }
        else
          Count(0,0);
      }
    if (insideJet_) UpdateCounter(cid);
  }
  if (!insideJet_) UpdateCounter(cid);
}   


void
IpsTOA::InitCounter()
{
  counter_buffer_.clear();
  counter_index_.clear();
}


void
IpsTOA::Count(int barcode, int pdgId)
{
  counter_buffer_t::iterator it = counter_buffer_.find(barcode);
  
  if ( it != counter_buffer_.end() )
  {
    it->second++;
  }
  else 
  { 
    counter_buffer_.insert( counter_buffer_pair_t(barcode, 1) );
    counter_index_.insert( counter_index_pair_t(barcode, pdgId) );
  }
}


void
IpsTOA::UpdateCounter(std::size_t cid)
{
  counter_buffer_t::const_iterator csi = counter_buffer_.begin();

  std::size_t particleType;

  for (; csi != counter_buffer_.end(); csi++)
  {
    if (antiparticles_)    
      particleType = counter_index_[csi->first];
    else 	
      particleType = abs(counter_index_[csi->first]);

	if ( !particleType )
	{ 
      counter_info_t info (particleType, 1); 
	  counter_t::iterator ci = counters_[cid].find( info );
      if ( ci != counters_[cid].end() )
        ci->second += csi->second;
      else 	
        counters_[cid].insert( counter_pair_t (info, csi->second) );
	}
	else
	{   
      counter_info_t info = counter_info_t (particleType, csi->second);      
	  counter_t::iterator ci = counters_[cid].find( info );
      if ( ci != counters_[cid].end() )
        ci->second++;
      else 	
        counters_[cid].insert( counter_pair_t (info, 1) );
    }
  }
}


DEFINE_ANOTHER_FWK_MODULE(IpsTOA);
