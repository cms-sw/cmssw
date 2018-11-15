// system include files
#include "Validation/Mixing/interface/MixCollectionValidation.h"

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "TFile.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <memory>
#include <utility>

using namespace edm;

MixCollectionValidation::MixCollectionValidation(const edm::ParameterSet& iConfig):
  minbunch_(iConfig.getParameter<int>("minBunch")),
  maxbunch_(iConfig.getParameter<int>("maxBunch")),
  verbose_(iConfig.getUntrackedParameter<bool>("verbose",false)),
  nbin_(maxbunch_-minbunch_+1)
{
  // Histograms will be defined according to the configuration
  ParameterSet mixObjextsSet_ = iConfig.getParameter<ParameterSet>("mixObjects");
}

MixCollectionValidation::~MixCollectionValidation()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

void MixCollectionValidation::bookHistograms(DQMStore::IBooker & iBooker, edm::Run const & iRun, edm::EventSetup const & /* iSetup */)
{
  iBooker.setCurrentFolder("MixingV/Mixing");

  std::vector<std::string> names = mixObjextsSet_.getParameterNames();

  for (std::vector<std::string>::iterator it = names.begin();it!= names.end();++it)
  {
    ParameterSet pset = mixObjextsSet_.getParameter<ParameterSet>((*it));
    if (!pset.exists("type"))  continue; //to allow replacement by empty pset
    std::string object = pset.getParameter<std::string>("type");
    std::vector<InputTag> tags = pset.getParameter<std::vector<InputTag> >("input");

    if ( object == "HepMCProduct" ) {

      std::string title = "Log10 Number of GenParticle in " + object;
      std::string name = "NumberOf" + object;
      nrHepMCProductH_ = iBooker.bookProfile(name,title,nbin_,minbunch_,maxbunch_+1,40,0.,40.);

      HepMCProductTags_ = tags;
      if (!HepMCProductTags_.empty()) {
        crossingFrame_Hep_Token_ = consumes<CrossingFrame<HepMCProduct> >(
            edm::InputTag("mix", HepMCProductTags_[0].label()));
      }
    }
    else if ( object == "SimTrack" ) {

      std::string title = "Log10 Number of " + object;
      std::string name = "NumberOf" + object;
      nrSimTrackH_ = iBooker.bookProfile(name,title,nbin_,minbunch_,maxbunch_+1,40,0.,40.);

      SimTrackTags_ = tags;
      if (!SimTrackTags_.empty()) {
        crossingFrame_SimTr_Token_ = consumes<CrossingFrame<SimTrack> >(
            edm::InputTag("mix", SimTrackTags_[0].label()));
      }
    }
    else if ( object == "SimVertex" )  {

      std::string title = "Log10 Number of " + object;
      std::string name = "NumberOf" + object;
      nrSimVertexH_ = iBooker.bookProfile(name,title,nbin_,minbunch_,maxbunch_+1,40,0.,40.);

      SimVertexTags_ = tags;
      if (!SimVertexTags_.empty()) {
        crossingFrame_SimVtx_Token_ = consumes<CrossingFrame<SimVertex> >(
            edm::InputTag("mix", SimVertexTags_[0].label()));
      }
    }
    else if ( object == "PSimHit" ) {
      std::vector<std::string> subdets=pset.getParameter<std::vector<std::string> >("subdets");
      for (unsigned int ii=0;ii<subdets.size();ii++) {

        std::string title = "Log10 Number of " + subdets[ii];
        std::string name = "NumberOf" + subdets[ii];
        SimHitNrmap_[subdets[ii]] = iBooker.bookProfile(name,title,nbin_,minbunch_,maxbunch_+1,40,0.,40.);

        title = "Time of " + subdets[ii];
        name = "TimeOf" + subdets[ii];
        SimHitTimemap_[subdets[ii]] = iBooker.bookProfile(name,title,nbin_,minbunch_,maxbunch_+1,40,-125.,375.);

      }

      PSimHitTags_ = tags;
      for (auto const & it : PSimHitTags_)
        crossingFrame_PSimHit_Tokens_.push_back(consumes<CrossingFrame<PSimHit> >(
            edm::InputTag("mix", it.label() + it.instance())));
    }
    else if ( object == "PCaloHit" ) {
      std::vector<std::string> subdets=pset.getParameter<std::vector<std::string> >("subdets");
      for (unsigned int ii=0;ii<subdets.size();ii++) {

        std::string title = "Log10 Number of " + subdets[ii];
        std::string name = "NumberOf" + subdets[ii];
        CaloHitNrmap_[subdets[ii]] = iBooker.bookProfile(name,title,nbin_,minbunch_,maxbunch_+1,40,0.,40.);

        title = "Time of " + subdets[ii];
        name = "TimeOf" + subdets[ii];
        CaloHitTimemap_[subdets[ii]] = iBooker.bookProfile(name,title,nbin_,minbunch_,maxbunch_+1,40,-125.,375.);

      }

      PCaloHitTags_ = tags;
      for (auto const & it : PCaloHitTags_)
        crossingFrame_PCaloHit_Tokens_.push_back(consumes<CrossingFrame<PCaloHit> >(
            edm::InputTag("mix", it.label() + it.instance())));
    }
  }
}

void MixCollectionValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iConfig)
{
  using namespace edm;

  if ( !HepMCProductTags_.empty() ) {
    bool gotHepMCProduct;
    edm::Handle<CrossingFrame<HepMCProduct> > crossingFrame;
    gotHepMCProduct = iEvent.getByToken(crossingFrame_Hep_Token_, crossingFrame);

    if (gotHepMCProduct){
      std::unique_ptr<MixCollection<HepMCProduct> >
          hepMCProduct (new MixCollection<HepMCProduct>(crossingFrame.product ()));
      MixCollection<HepMCProduct>::MixItr hitItr;

      fillGenParticleMulti(hitItr, hepMCProduct, nrHepMCProductH_);
    }
  }

  if ( !SimTrackTags_.empty() ) {
    bool gotSimTrack;
    edm::Handle<CrossingFrame<SimTrack> > crossingFrame;
    gotSimTrack = iEvent.getByToken(crossingFrame_SimTr_Token_,crossingFrame);

    if (gotSimTrack){
      std::unique_ptr<MixCollection<SimTrack> >
          simTracks (new MixCollection<SimTrack>(crossingFrame.product ()));
      MixCollection<SimTrack>::MixItr hitItr;

      fillMultiplicity(hitItr, simTracks, nrSimTrackH_);
    }
  }

  if ( !SimVertexTags_.empty() ) {
    bool gotSimVertex;
    edm::Handle<CrossingFrame<SimVertex> > crossingFrame;
    std::string SimVertexLabel = SimVertexTags_[0].label();
    gotSimVertex = iEvent.getByToken(crossingFrame_SimVtx_Token_, crossingFrame);

    if (gotSimVertex){
      std::unique_ptr<MixCollection<SimVertex> >
          simVerteces (new MixCollection<SimVertex>(crossingFrame.product ()));
      MixCollection<SimVertex>::MixItr hitItr;

      fillMultiplicity(hitItr, simVerteces, nrSimVertexH_);
    }
  }

  if ( !PSimHitTags_.empty() ) {

    edm::Handle<CrossingFrame<PSimHit> > crossingFrame;

    for ( int i = 0; i < (int)PSimHitTags_.size(); i++ ) {
      bool gotPSimHit;
      gotPSimHit = iEvent.getByToken(crossingFrame_PSimHit_Tokens_[i], crossingFrame);

      if (gotPSimHit){
        std::unique_ptr<MixCollection<PSimHit> >
            simHits (new MixCollection<PSimHit>(crossingFrame.product ()));

        MixCollection<PSimHit>::MixItr hitItr;

        fillMultiplicity(hitItr, simHits, SimHitNrmap_[PSimHitTags_[i].instance()]);

        fillSimHitTime(hitItr, simHits, SimHitTimemap_[PSimHitTags_[i].instance()]);
      }
    }
  }

  if ( !PCaloHitTags_.empty() ) {

    edm::Handle<CrossingFrame<PCaloHit> > crossingFrame;

    for ( int i = 0; i < (int)PCaloHitTags_.size(); i++ ) {
      bool gotPCaloHit;
      std::string PCaloHitLabel = PCaloHitTags_[i].label()+PCaloHitTags_[i].instance();
      gotPCaloHit = iEvent.getByToken(crossingFrame_PCaloHit_Tokens_[i], crossingFrame);

      if (gotPCaloHit){
        std::unique_ptr<MixCollection<PCaloHit> >
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
