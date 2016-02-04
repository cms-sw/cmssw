# include "SimDataFormats/CrossingFrame/interface/CrossingFramePlaybackInfo.h"

CrossingFramePlaybackInfo::CrossingFramePlaybackInfo(int minBunch, int maxBunch, unsigned int maxNbSources):maxNbSources_(maxNbSources),minBunch_(minBunch)
{
  //initialise data structures
  nBcrossings_=maxBunch-minBunch+1;
  pileupFileNr_.resize(maxNbSources_);
  idFirstPileup_.resize(maxNbSources_);
  nrEvents_.resize(maxNbSources_);
  for (unsigned int i=0;i<maxNbSources_;++i) {
    pileupFileNr_[i].resize(nBcrossings_);
    idFirstPileup_[i].resize(nBcrossings_);
    nrEvents_[i].resize(nBcrossings_);
    for (int j=0;j<nBcrossings_;++j) {
      (pileupFileNr_[i])[j]=-1;
      (idFirstPileup_[i])[j]=edm::EventID(0,0,0);
      (nrEvents_[i])[j]=0;
    }
  }
}

void CrossingFramePlaybackInfo::setEventStartInfo(std::vector<edm::EventID> &id, std::vector<int>& fileNr, std::vector<unsigned int>& nrEvents, const unsigned int s)
{
  idFirstPileup_[s]=id;
  pileupFileNr_[s]=fileNr;
  nrEvents_[s]=nrEvents;
}
