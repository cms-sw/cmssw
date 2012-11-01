#ifndef SimCalorimetry_EcalElectronicsEmulationEcalSimpleProducer_h
#define SimCalorimetry_EcalElectronicsEmulationEcalSimpleProducer_h

#include "FWCore/Framework/interface/EDProducer.h"
#include <memory>
#include <TFormula.h>
#include <string>

/** This edm producer generates Ecal Digis (data frames and TPGs)
 * according to a given pattern. The pattern is defined
 * as function of event id, crystal/TT, and time sample. Only barrel
 * is currently supported for the crystal channel data.
 * <P>Module parameters  (in addition to standard source parameters):
 * <UL><LI>string formula: formula of crystal channel time sample encoded ADC
 * counts.</LI>
 *     <LI>string tpFormula: formula of trigger primitives.</LI>
 *     <LI>untracked bool verbose: verbosity switch</LI>
 * </UL>
 * The crystal ADC formula is parametrized with the following variables:
 * <UL><LI>ieta0: crystal eta index starting from 0 at eta- end of barrel</LI>
 *     <LI>iphi0: crystal phi index starting at Phi=0deg. in std CMS coordinates</LI>
 *     <LI>ievt0 event sequence number within the job run starting from 0</LI> 
 *     <LI>isample0 sample time position starting from 0</LI>
 * </UL>
 * The trigger primitive formula is parametrized with the following variables:
 * <UL><LI>ieta0: trigger tower eta index starting from 0 at eta- end of barrel</LI>
 *     <LI>iphi0: trigger tower index starting at Phi=0deg. in std CMS coordinates</LI>
 *     <LI>ievt0 event sequence number within the job run starting from 0</LI> 
 *     <LI>isample0 sample time position starting from 0</LI>
 * </UL>
 * In both formulae 'itt0' shortcut can be used for the trigger tower index
 * within the SM starting at 0 from lowest relative eta and lowest phi and
 * increasing first with phi then with eta. The syntax for the formula is the
 * syntax defined in ROOT <A href=http://root.cern.ch/root/html/TFormula.html>
 * TFormula</A>
 *
 */
class EcalSimpleProducer: public edm::EDProducer {

  //constructor(s) and destructor(s)
public:
  /** Constructs an EcalSimpleProducer
   * @param pset CMSSW configuration
   * @param sdesc description of this input source
   */
  EcalSimpleProducer(const edm::ParameterSet& pset);

  /**Destructor
   */
  virtual ~EcalSimpleProducer(){};

  /** Called at start of job.
   * @param es the event setup
   */
  void beginJob(){};

  /** The main method. It produces the event.
   * @param evt [out] produced event.
   */
  virtual void produce(edm::Event& evt, const edm::EventSetup&);
  
  //method(s)
public:
private:
  /** Help function to replace a pattern within a string. Every occurance
   * of the pattern is replaced. An exact match is performed: no wild card.
   * @param s string to operate on
   * @param pattern to replace.
   * @param string to substitute to the pattern
   */
  void replaceAll(std::string& s, const std::string& from,
		  const std::string& to) const;

  /** Converts c-array index (contiguous integer starting from 0) to
   * std CMSSW ECAL crystal eta index.
   * @param iEta0 c-array index. '0' postfix reminds the index starts from 0
   * @return std CMSSW ECAL crystal index.
   */
  int cIndex2iEta(int iEta0) const{
    return (iEta0<85)?iEta0-85:iEta0-84;
  }
  
  /** Converts c-array index (contiguous integer starting from 0) to
   * std CMSSW ECAL crystal phi index.
   * @param iPhi0 c-array index. '0' postfix reminds the index starts from 0
   * @return std CMSSW ECAL crystal index.
   */
  int cIndex2iPhi(int iPhi0) const{
    return (iPhi0+10)%360+1;
  }
  
  /** Converts c-array index (contiguous integer starting from 0) to
   * std CMSSW ECAL trigger tower eta index.
   * @param iEta0 c-array index. '0' postfix reminds the index starts from 0
   * @return std CMSSW ECAL trigger tower index.
   */
  int cIndex2iTtEta(int iEta0) const{
    return (iEta0<28)?iEta0-28:iEta0-27;
  }
  
  /** Converts c-array index (contiguous integer starting from 0) to
   * std CMSSW ECAL trigger tower phi index.
   * @param iPhi0 c-array index. '0' postfix reminds the index starts from 0
   * @return std CMSSW ECAL trigger tower index.
   */
  int cIndex2iTtPhi(int iPhi0) const{
    return iPhi0+1;
  }
  
  //attribute(s)
protected:
private:
  /** Formula defining the data frame samples
   */
  std::auto_ptr<TFormula> formula_;

  /** Formula defining the trigger primitives
   */
  std::auto_ptr<TFormula> tpFormula_;

  /** Formula defining the sim hits
   */
  std::auto_ptr<TFormula> simHitFormula_;
  
  /** Verbosity switch
   */
  bool verbose_;
};

#endif
