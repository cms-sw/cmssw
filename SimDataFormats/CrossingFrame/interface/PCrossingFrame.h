#ifndef PCROSSING_FRAME_H
#define PCROSSING_FRAME_H

/** \class PCrossingFrame
 *
 * PCrossingFrame allow the write the transient CrossingFrame
 *
 * \author EmiliaBecheva, Claude Charlot,  LLR Palaiseau
 *
 * \version   1st Version April 2009
 *
 ************************************************************/

#include "SimDataFormats/CrossingFrame/interface/CrossingFrame.h"

template <class T> 
class PCrossingFrame
{

  public:	

  PCrossingFrame(){}
  PCrossingFrame(const CrossingFrame<T>& cf);
  PCrossingFrame(const PCrossingFrame<T> &pcf){};

  ~PCrossingFrame() {;}
  
  // getters for data members of PCrossingFrame
  edm::EventID getEventID() const {return Pid_;}
  std::vector<const T *> getPileups() const {return PCFpileups_;}
  int getBunchSpace() const {return PbunchSpace_;}
  unsigned int getMaxNbSources() const {return PmaxNbSources_; }
  std::string getSubDet() const { return PCFsubdet_;}
  unsigned int getPileupFileNr() const {return PCFpileupFileNr_;}
  edm::EventID getIdFirstPileup() const {return PCFidFirstPileup_;}
  std::vector<unsigned int> getPileupOffsetsBcr() const {return PCFpileupOffsetsBcr_;}   
  std::vector< std::vector<unsigned int> > getPileupOffsetsSource() const {return PCFpileupOffsetsSource_;} //one per source
  std::pair<int,int> getBunchRange() const {return std::pair<int,int>(firstPCrossing_,lastPCrossing_);}
	

  private:
  unsigned int PmaxNbSources_;
  int PbunchSpace_; 
  edm::EventID Pid_;
  int firstPCrossing_;
  int lastPCrossing_;
  std::vector<const T * > PCFpileups_;
  std::vector<const T * > PCFsignals_;
  std::string PCFsubdet_;
  unsigned int PCFpileupFileNr_;
  edm::EventID PCFidFirstPileup_;
  std::vector<unsigned int> PCFpileupOffsetsBcr_;
  std::vector< std::vector<unsigned int> > PCFpileupOffsetsSource_;
};


template <class T>
PCrossingFrame<T>::PCrossingFrame(const CrossingFrame<T>& cf)
{
 //get data members from CrossingFrame
 PmaxNbSources_= cf.getMaxNbSources();
 PbunchSpace_ = cf.getBunchSpace();
 Pid_ = cf.getEventID();
 firstPCrossing_ = cf.getBunchRange().first;
 lastPCrossing_ = cf.getBunchRange().second;
 PCFpileups_ = cf.getPileups();
 PCFsignals_ = cf.getSignal();
 PCFsubdet_ = cf.getSubDet();
 PCFpileupFileNr_ = cf.getPileupFileNr();
 PCFidFirstPileup_ = cf.getIdFirstPileup();
 PCFpileupOffsetsBcr_ = cf.getPileupOffsetsBcr();
 PCFpileupOffsetsSource_ = cf.getPileupOffsetsSource();
}

#endif 
