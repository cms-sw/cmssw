#ifndef SimMuL1_LCT_h
#define SimMuL1_LCT_h

/*
 * Class LCT 
 */

class LCT
{
 public:
  /// constructor
  LCT();
  /// copy constructor
  LCT(const LCT&);
  /// destructor
  ~LCT();

  /// get the underlying trigger digi
  const CSCCorrelatedDigi* getTriggerDigi() const {return triggerDigi_;}
  /// get the alct
  const ALCT* getALCT() const {return alct_;}
  /// get the clct
  const CLCT* getCLCT() const {return clct_;}
  /// get the bunch crossing
  const int getBX() const {return triggerDigi_->getBX();}
  /// get the detector Id
  const int getDetId() const {return detId_;}
  /// is the LCT a ghost?
  const bool isGhost() const {return isGhost_;}
  /// does the LCT match?
  const bool deltaOk() const {return deltaOk_;}
  /// is the LCT in the readout? 
  const book inReadout() const {return inReadout_;}

  /// set the underlying trigger digi
  void setTriggerDigi(const CSCCorrelatedDigi* d) {triggerDigi_ = d;}
  /// set the alct
  void setALCT(const ALCT* alct) {return alct_;}
  /// set the clct
  void setCLCT(const CLCT* clct) {return clct_;}
  /// set the bunch crossing
  void setBX(const int bx) {return ;}
  /// set the detector Id
  void setDetId(const int detId) {return detId_;}
  /// is the LCT a ghost?
  void SetGhost(const bool ghost) {isGhost_ = ghost;}
  /// does the LCT match?
  void SetDeltaOk(const book ok) {deltaOk_ = ok;}
  /// is the LCT in the readout? 
  void SetReadout(const bool inReadout) {inReadout_ = inReadout;}

 private:
  const CSCCorrelatedDigi triggerDigi_;
  const ALCT* alct_;
  const CLCT* clct_;
  int bx_;
  int detId_;
  bool isGhost_;
  bool deltaOk_;
  bool inReadout_;
};

#endif
