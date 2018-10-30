#ifndef SimDataFormats_GeneratorProducts_GenLumiInfoProduct_h
#define SimDataFormats_GeneratorProducts_GenLumiInfoProduct_h

#include <vector>

/** \class GenLumiInfoProduct
 *
 */

class GenLumiInfoProduct {
 public:
  // a few forward declarations
  struct XSec;
  struct ProcessInfo;

  // constructors, destructors
  GenLumiInfoProduct();
  GenLumiInfoProduct(const int id);
  GenLumiInfoProduct(const GenLumiInfoProduct &other);
  virtual ~GenLumiInfoProduct();

  // getters

  const int  getHEPIDWTUP() const {return hepidwtup_;}
  const std::vector<ProcessInfo>& getProcessInfos() const {return internalProcesses_;}

  // setters

  void setHEPIDWTUP(const int id) { hepidwtup_ = id;}
  void setProcessInfo(const std::vector<ProcessInfo> & processes) {internalProcesses_ = processes;}

  // Struct- definitions
  struct XSec {
  public:
  XSec() : value_(-1.), error_(-1.) {}
  XSec(double v, double e = -1.) :
    value_(v), error_(e) {}
  XSec(const XSec &other) :
    value_(other.value_), error_(other.error_) {}

    double value() const { return value_; }
    double error() const { return error_; }

    bool isSet() const { return value_ >= 0.; }
    bool hasError() const { return error_ >= 0.; }

    operator double() const { return value_; }
    operator bool() const { return isSet(); }

    bool operator == (const XSec &other) const
    { return value_ == other.value_ && error_ == other.error_; }
    bool operator != (const XSec &other) const { return !(*this == other); }

  private:
    double value_, error_;
  };


  struct FinalStat {
  public:
  FinalStat() : n_(0), sum_(0.0), sum2_(0.0) {}
  FinalStat(unsigned int n1, double sum1, double sum21) :
    n_(n1), sum_(sum1), sum2_(sum21) {}
  FinalStat(const FinalStat &other) :
    n_(other.n_), sum_(other.sum_), sum2_(other.sum2_) {}

    unsigned int n() const { return n_; }
    double sum() const { return sum_; }
    double sum2() const{ return sum2_; }

    void add(const FinalStat& other)
    {
      n_ += other.n();
      sum_ += other.sum();
      sum2_ += other.sum2();
      }

    bool operator == (const FinalStat &other) const
    { return n_ == other.n_ && sum_ == other.sum_ && sum2_ == other.sum2_; }
    bool operator != (const FinalStat &other) const { return !(*this == other); }
  private:
    unsigned int	n_;
    double		sum_;
    double		sum2_;
  };


  struct ProcessInfo {
  public:
  ProcessInfo():process_(-1),nPassPos_(0),nPassNeg_(0),nTotalPos_(0),nTotalNeg_(0){}
  ProcessInfo(int id):process_(id),nPassPos_(0),nPassNeg_(0),nTotalPos_(0),nTotalNeg_(0){}

    // accessors
    int process() const {return process_;}
    XSec lheXSec() const {return lheXSec_;}

    unsigned int nPassPos() const {return nPassPos_;}
    unsigned int nPassNeg() const {return nPassNeg_;}
    unsigned int nTotalPos() const {return nTotalPos_;}
    unsigned int nTotalNeg() const {return nTotalNeg_;}

    FinalStat tried() const {return tried_;}
    FinalStat selected() const {return selected_;}
    FinalStat killed() const {return killed_;}
    FinalStat accepted() const {return accepted_;}
    FinalStat acceptedBr() const {return acceptedBr_;}

    // setters
    void addOthers(const ProcessInfo& other){
      nPassPos_ += other.nPassPos();
      nPassNeg_ += other.nPassNeg();
      nTotalPos_ += other.nTotalPos();
      nTotalNeg_ += other.nTotalNeg();
      tried_.add(other.tried());
      selected_.add(other.selected());
      killed_.add(other.killed());
      accepted_.add(other.accepted());
      acceptedBr_.add(other.acceptedBr());
    }
    void setProcess(int id) { process_ = id; }
    void setLheXSec(double value, double err) { lheXSec_ = XSec(value,err); }
    void setNPassPos(unsigned int n) { nPassPos_ = n; }
    void setNPassNeg(unsigned int n) { nPassNeg_ = n; }
    void setNTotalPos(unsigned int n) { nTotalPos_ = n; }
    void setNTotalNeg(unsigned int n) { nTotalNeg_ = n; }
    void setTried(unsigned int n, double sum, double sum2) { tried_ = FinalStat(n,sum,sum2); }
    void setSelected(unsigned int n, double sum, double sum2) { selected_ = FinalStat(n,sum,sum2); }
    void setKilled(unsigned int n, double sum, double sum2) { killed_ = FinalStat(n,sum,sum2); }
    void setAccepted(unsigned int n, double sum, double sum2) { accepted_ = FinalStat(n,sum,sum2); }
    void setAcceptedBr(unsigned int n, double sum, double sum2) { acceptedBr_ = FinalStat(n,sum,sum2); }
	  
  private:
    int             process_;
    XSec            lheXSec_;
    unsigned int    nPassPos_;
    unsigned int    nPassNeg_;
    unsigned int    nTotalPos_;
    unsigned int    nTotalNeg_;
    FinalStat       tried_;
    FinalStat       selected_;
    FinalStat       killed_;
    FinalStat       accepted_;
    FinalStat       acceptedBr_;

  };



  // methods used by EDM
  virtual bool mergeProduct(const GenLumiInfoProduct &other);
  void swap(GenLumiInfoProduct& other);
  virtual bool isProductEqual(const GenLumiInfoProduct &other) const;
 private:
  // cross sections
  int     hepidwtup_;
  std::vector<ProcessInfo> internalProcesses_; 


};

#endif // SimDataFormats_GeneratorProducts_GenLumiInfoProduct_h
