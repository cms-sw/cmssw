// system include files
#include <memory>
#include <utility>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"
#include "SimDataFormats/CrossingFrame/interface/MixCollection.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"

#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"
#include "SimDataFormats/Track/interface/SimTrackContainer.h"

#include "Validation/Mixing/interface/MixCollectionValidation.h"
#include "TFile.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

using namespace edm;

MixCollectionValidation::MixCollectionValidation(const edm::ParameterSet& iConfig): outputFile_(iConfig.getParameter<std::string>("outputFile")), 
                                                                                    minbunch_(iConfig.getParameter<int>("minBunch")),
                                                                                    maxbunch_(iConfig.getParameter<int>("maxBunch")),
                                                                                    verbose_(iConfig.getUntrackedParameter<bool>("verbose",false)),
                                                                                    dbe_(0),nbin_(maxbunch_-minbunch_+1)
{

  if ( outputFile_.size() != 0 ) {
    edm::LogInfo("OutputInfo") << " Ecal SimHits Task histograms will be saved to " << outputFile_.c_str();
  } else {
    edm::LogInfo("OutputInfo") << " Ecal SimHits Task histograms will NOT be saved";
  }
  
  // get hold of back-end interface
  dbe_ = edm::Service<DQMStore>().operator->();           
  if ( dbe_ ) {
    if ( verbose_ ) { dbe_->setVerbose(1); } 
    else            { dbe_->setVerbose(0); }
  }
                                                                                                            
  if ( dbe_ ) {
    if ( verbose_ ) dbe_->showDirStructure();
  }

  // get hold of back-end interface
  dbe_ = Service<DQMStore>().operator->(); 
//  dbe_->showDirStructure();
  dbe_->setCurrentFolder("MixingV/Mixing");

  // define the histograms according to the configuration 

  ParameterSet ps=iConfig.getParameter<ParameterSet>("mixObjects");
  names_ = ps.getParameterNames();

  for (std::vector<std::string>::iterator it=names_.begin();it!= names_.end();++it)
    {
      ParameterSet pset=ps.getParameter<ParameterSet>((*it));
      if (!pset.exists("type"))  continue; //to allow replacement by empty pset
      std::string object = pset.getParameter<std::string>("type");
      std::vector<InputTag>  tags=pset.getParameter<std::vector<InputTag> >("input");

if ( object == "HepMCProduct" ) {

        std::string title = "Log10 Number of GenParticle in " + object;
	std::string name = "NumberOf" + object;
        nrHepMCProductH_ = dbe_->bookProfile(name,title,nbin_,minbunch_,maxbunch_+1,40,0.,40.);
        
        HepMCProductTags_ = tags;
    
      }
      else if ( object == "SimTrack" ) {
        
        std::string title = "Log10 Number of " + object;
	std::string name = "NumberOf" + object;
        nrSimTrackH_ = dbe_->bookProfile(name,title,nbin_,minbunch_,maxbunch_+1,40,0.,40.);
        
        SimTrackTags_ = tags;
    
      }
      else if ( object == "SimVertex" )  {
        
        std::string title = "Log10 Number of " + object;
	std::string name = "NumberOf" + object;
        nrSimVertexH_ = dbe_->bookProfile(name,title,nbin_,minbunch_,maxbunch_+1,40,0.,40.);
    
        SimVertexTags_ = tags;

      }
      else if ( object == "PSimHit" ) {
        std::vector<std::string> subdets=pset.getParameter<std::vector<std::string> >("subdets");
        for (unsigned int ii=0;ii<subdets.size();ii++) {

          std::string title = "Log10 Number of " + subdets[ii];
	  std::string name = "NumberOf" + subdets[ii];
          SimHitNrmap_[subdets[ii]] = dbe_->bookProfile(name,title,nbin_,minbunch_,maxbunch_+1,40,0.,40.);

          title = "Time of " + subdets[ii];
	  name = "TimeOf" + subdets[ii];
          SimHitTimemap_[subdets[ii]] = dbe_->bookProfile(name,title,nbin_,minbunch_,maxbunch_+1,40,-125.,375.);

        }

        PSimHitTags_ = tags;
        
      }
      else if ( object == "PCaloHit" ) {
        std::vector<std::string> subdets=pset.getParameter<std::vector<std::string> >("subdets");
        for (unsigned int ii=0;ii<subdets.size();ii++) {

          std::string title = "Log10 Number of " + subdets[ii];
	  std::string name = "NumberOf" + subdets[ii];
          CaloHitNrmap_[subdets[ii]] = dbe_->bookProfile(name,title,nbin_,minbunch_,maxbunch_+1,40,0.,40.);
        
          title = "Time of " + subdets[ii];
	  name = "TimeOf" + subdets[ii];
          CaloHitTimemap_[subdets[ii]] = dbe_->bookProfile(name,title,nbin_,minbunch_,maxbunch_+1,40,-125.,375.);

        }

        PCaloHitTags_ = tags;
      }
    }
                
}

MixCollectionValidation::~MixCollectionValidation()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}

void MixCollectionValidation::beginJob(edm::EventSetup const&iSetup) {
} 


void MixCollectionValidation::endJob() {
 if (outputFile_.size() != 0 && dbe_ ) dbe_->save(outputFile_);
}

//
// member functions
//

// ------------ method called to analyze the data  ------------
void
MixCollectionValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iConfig)
{

   using namespace edm;

   if ( HepMCProductTags_.size() > 0 ) {
     bool gotHepMCProduct; 	
     edm::Handle<CrossingFrame<HepMCProduct> > crossingFrame;
     std::string HepMCProductLabel = HepMCProductTags_[0].label();
     gotHepMCProduct = iEvent.getByLabel("mix",HepMCProductLabel,crossingFrame);

     if (gotHepMCProduct){
       std::auto_ptr<MixCollection<HepMCProduct> > 
         hepMCProduct (new MixCollection<HepMCProduct>(crossingFrame.product ()));
       MixCollection<HepMCProduct>::MixItr hitItr;
     
       fillGenParticleMulti(hitItr, hepMCProduct, nrHepMCProductH_);
     }	
   }

   if ( SimTrackTags_.size() > 0 ) {
     bool gotSimTrack;
     edm::Handle<CrossingFrame<SimTrack> > crossingFrame;
     std::string SimTrackLabel = SimTrackTags_[0].label();
     gotSimTrack = iEvent.getByLabel("mix",SimTrackLabel,crossingFrame);

     if (gotSimTrack){
       std::auto_ptr<MixCollection<SimTrack> > 
         simTracks (new MixCollection<SimTrack>(crossingFrame.product ()));
       MixCollection<SimTrack>::MixItr hitItr;
     
       fillMultiplicity(hitItr, simTracks, nrSimTrackH_);
     }
   }

   if ( SimVertexTags_.size() > 0 ) {
     bool gotSimVertex;
     edm::Handle<CrossingFrame<SimVertex> > crossingFrame;
     std::string SimVertexLabel = SimVertexTags_[0].label();
     gotSimVertex = iEvent.getByLabel("mix",SimVertexLabel,crossingFrame);

     if (gotSimVertex){
       std::auto_ptr<MixCollection<SimVertex> > 
         simVerteces (new MixCollection<SimVertex>(crossingFrame.product ()));
       MixCollection<SimVertex>::MixItr hitItr;
     
       fillMultiplicity(hitItr, simVerteces, nrSimVertexH_);
     }
   }

   if ( PSimHitTags_.size() > 0 ) {

     edm::Handle<CrossingFrame<PSimHit> > crossingFrame;

     for ( int i = 0; i < (int)PSimHitTags_.size(); i++ ) {
       bool gotPSimHit;
       std::string PSimHitLabel = PSimHitTags_[i].label()+PSimHitTags_[i].instance();
       gotPSimHit = iEvent.getByLabel("mix",PSimHitLabel,crossingFrame);

       if (gotPSimHit){
         std::auto_ptr<MixCollection<PSimHit> > 
           simHits (new MixCollection<PSimHit>(crossingFrame.product ()));

         MixCollection<PSimHit>::MixItr hitItr;

         fillMultiplicity(hitItr, simHits, SimHitNrmap_[PSimHitTags_[i].instance()]);

         fillSimHitTime(hitItr, simHits, SimHitTimemap_[PSimHitTags_[i].instance()]);
       }
     }
   }

   if ( PCaloHitTags_.size() > 0 ) {

     edm::Handle<CrossingFrame<PCaloHit> > crossingFrame;

     for ( int i = 0; i < (int)PCaloHitTags_.size(); i++ ) {
       bool gotPCaloHit;
       std::string PCaloHitLabel = PCaloHitTags_[i].label()+PCaloHitTags_[i].instance();
       gotPCaloHit = iEvent.getByLabel("mix",PCaloHitLabel,crossingFrame);

       if (gotPCaloHit){
         std::auto_ptr<MixCollection<PCaloHit> > 
           caloHits (new MixCollection<PCaloHit>(crossingFrame.product ()));

         MixCollection<PCaloHit>::MixItr hitItr;

         fillMultiplicity(hitItr, caloHits, CaloHitNrmap_[PCaloHitTags_[i].instance()]);

         fillCaloHitTime(hitItr, caloHits, CaloHitTimemap_[PCaloHitTags_[i].instance()]);
       }
     }
   }

}

template<class T1, class T2> void MixCollectionValidation::fillMultiplicity(T1 & theItr_, T2 & theColl_, MonitorElement * theProfile_) {

  std::vector<int> theMult(nbin_);

  for ( theItr_ = theColl_->begin() ; theItr_ != theColl_->end() ; ++theItr_) {

    int bunch = (*theItr_).eventId().bunchCrossing();
    int index = bunch - minbunch_;
    if ( index >= 0 && index < nbin_ ) { theMult[index] += 1; }
    else { edm::LogWarning("MixCollectionValidation") << "fillMultiplicity: bunch number " << bunch << " out of range"; }

  }
  
  for ( int i = 0; i < nbin_; i++ ) {
    theProfile_->Fill(float(i+minbunch_+0.5),std::log10(std::max(float(0.1),float(theMult[i]))));
  }
}


template<class T1, class T2> void MixCollectionValidation::fillGenParticleMulti(T1 & theItr_, T2 & theColl_, MonitorElement * theProfile_) {

  std::vector<int> theMult(nbin_);

  for ( theItr_ = theColl_->begin() ; theItr_ != theColl_->end() ; ++theItr_) {

    int bunch = theItr_.bunch();
    int index = bunch - minbunch_;
    if ( index >= 0 && index < nbin_ ) { theMult[index] += (*theItr_).GetEvent()->particles_size(); }
    else { edm::LogWarning("MixCollectionValidation") << "fillMultiplicity: bunch number " << bunch << " out of range"; }

  }
  
  for ( int i = 0; i < nbin_; i++ ) {
    theProfile_->Fill(float(i+minbunch_+0.5),std::log10(std::max(float(0.1),float(theMult[i]))));
  }
}

template<class T1, class T2> void MixCollectionValidation::fillSimHitTime(T1 & theItr_, T2 & theColl_, MonitorElement * theProfile_) {

  for ( theItr_ = theColl_->begin() ; theItr_ != theColl_->end() ; ++theItr_) {

    int bunch = (*theItr_).eventId().bunchCrossing();
    float time = (*theItr_).timeOfFlight();
    int index = bunch - minbunch_;
    if ( index >= 0 && index < nbin_ ) { theProfile_->Fill(float(bunch+0.5),time); }
    else { edm::LogWarning("MixCollectionValidation") << "fillSimHitTime: bunch number " << bunch << " out of range"; }

  }
  
}

template<class T1, class T2> void MixCollectionValidation::fillCaloHitTime(T1 & theItr_, T2 & theColl_, MonitorElement * theProfile_) {

  for ( theItr_ = theColl_->begin() ; theItr_ != theColl_->end() ; ++theItr_) {

    int bunch = (*theItr_).eventId().bunchCrossing();
    float time = (*theItr_).time();
    int index = bunch - minbunch_;
    if ( index >= 0 && index < nbin_ ) { theProfile_->Fill(float(bunch+0.5),time); }
    else { edm::LogWarning("MixCollectionValidation") << "fillCaloHitTime: bunch number " << bunch << " out of range"; }

  }
  
}
