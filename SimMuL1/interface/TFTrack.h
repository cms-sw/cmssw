class TFTrack
{
 public:
  /// constructor
  TFTrack();
  /// copy constructor
  TFTrack(const TFTrack&);
  /// destructor
  ~TFTrack();  

  /// L1 track
  const csc::L1Track* getL1Track() const {return l1track_;}
  /// collection of trigger digis
  std::vector<const CSCCorrelatedLCTDigi* >& 
    getTriggerDigis() const {return return triggerDigis_;} 
  /// collection of MPC LCTs
  
  /// track sign
  bool sign() const {return l1track_->sign();}
  /// bunch crossing 
  int getBX() const {return l1track_->bx();}
  /// has stub in muon barrel?
  bool hasStubBarrel();  
  /// has stub in muon endcap?
  bool hasStubEndcap();
  /// has stub in particular endcap station
  bool hasStubEndcapStaion(int station);
  /// 
  
 private:
  const csc::L1Track* l1track_;
    std::vector < const CSCCorrelatedLCTDigi * > trgdigis;
    std::vector < CSCDetId > trgids;
    std::vector < std::pair<float, float> > trgetaphis;
    std::vector < csctf::TrackStub > trgstubs;
    std::vector < MPLCT* > mplcts;
    std::vector < CSCDetId > ids; // chamber ids
    unsigned phi_packed;
    unsigned eta_packed;
    unsigned pt_packed;
    unsigned q_packed;
    double phi;
    double eta;
    double pt;
    double dr;
    bool deltaOk1;
    bool deltaOk2;
    bool deltaOkME1;
    bool debug;
  
  
};
