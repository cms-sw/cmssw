#ifndef MBUEandQCDVALIDATION_H
#define MBUEandQCDVALIDATION_H

/*class MBUEandQCDValidation
 *  
 *  Class to fill Event Generator dqm monitor elements; works on HepMCProduct
 *
 *  $Date: 2011/12/29 10:53:10 $
 *  $Revision: 1.3 $
 *
 */

// framework & common header files
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Run.h"

#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

//DQM services
#include "DQMServices/Core/interface/DQMStore.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"

#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
#include "Validation/EventGenerator/interface/CaloCellManager.h"
#include "Validation/EventGenerator/interface/WeightManager.h"

#include <vector>

class MBUEandQCDValidation : public edm::EDAnalyzer
{
    public:
	explicit MBUEandQCDValidation(const edm::ParameterSet&);
	virtual ~MBUEandQCDValidation();
	virtual void beginJob();
	virtual void endJob();  
	virtual void analyze(const edm::Event&, const edm::EventSetup&);
	virtual void beginRun(const edm::Run&, const edm::EventSetup&);
	virtual void endRun(const edm::Run&, const edm::EventSetup&);

    private:

    WeightManager _wmanager;

    edm::InputTag hepmcCollection_;
    edm::InputTag genchjetCollection_;
    edm::InputTag genjetCollection_;

    unsigned int verbosity_;

	/// PDT table
	edm::ESHandle<HepPDT::ParticleDataTable> fPDGTable ;

    ///  status 1 GenParticle collection
    std::vector<const HepMC::GenParticle*> hepmcGPCollection;
    std::vector<double> hepmcCharge;

    /// manager of calorimetric cell structure
    CaloCellManager* theCalo;

    unsigned int getHFbin(double eta);

    bool isCharged(unsigned int i);
    bool isNeutral(unsigned int i);
    bool isNeutrino(unsigned int i);

    std::vector<double> eneInCell;

	///ME's "container"
	DQMStore *dbe;

    MonitorElement* nEvt;

    MonitorElement* nNoFwdTrig;
    MonitorElement* nSaFwdTrig;

    MonitorElement* nbquark;
    MonitorElement* ncandbquark;
    MonitorElement* ncnobquark;

    ///QCD-09-010 analysis
    MonitorElement* nEvt1;
    MonitorElement* dNchdpt1;
    MonitorElement* dNchdeta1;

    //QCD-10-001 analysis
    MonitorElement* nEvt2;
    MonitorElement* leadTrackpt;
    MonitorElement* leadTracketa;
    MonitorElement* dNchdeta2;
    MonitorElement* dNchdpt2;
    MonitorElement* nCha;
    MonitorElement* dNchdSpt;
    MonitorElement* dNchdphi;
    MonitorElement* dSptdphi;
    MonitorElement* nChaDenLpt;
    MonitorElement* sptDenLpt;

    //Charged jets
    MonitorElement* nChj;
    MonitorElement* dNchjdeta;
    MonitorElement* dNchjdpt;
    MonitorElement* leadChjpt;
    MonitorElement* leadChjeta;
    MonitorElement* pt1pt2optotch;

    //Identified particles multiplicities
    MonitorElement* nPPbar;
    MonitorElement* nKpm;
    MonitorElement* nK0s;
    MonitorElement* nL0;
    MonitorElement* nNNbar;
    MonitorElement* nGamma;
    MonitorElement* nXim;
    MonitorElement* nOmega;

    //Identified particles momentum specturm
    MonitorElement* pPPbar;
    MonitorElement* pKpm;
    MonitorElement* pK0s;
    MonitorElement* pL0;
    MonitorElement* pNNbar;
    MonitorElement* pGamma;
    MonitorElement* pXim;
    MonitorElement* pOmega;

    MonitorElement* elePt;
    MonitorElement* muoPt;

    //Jets no neutrino
    MonitorElement* nDijet;
    MonitorElement* nj;
    MonitorElement* dNjdeta;
    MonitorElement* dNjdpt;
    MonitorElement* pt1pt2optot;
    MonitorElement* pt1pt2balance;
    MonitorElement* pt1pt2Dphi;
    MonitorElement* pt1pt2InvM;
    MonitorElement* pt3Frac;
    MonitorElement* sumJEt;
    MonitorElement* missEtosumJEt;
    MonitorElement* sumPt;
    MonitorElement* sumChPt;

    //Forward energy flow
    MonitorElement* nHFflow;
    MonitorElement* dEdetaHFmb;
    MonitorElement* dEdetaHFdj;

    MonitorElement* nHFSD;
    MonitorElement* EmpzHFm;
    MonitorElement* ntHFm;
    MonitorElement* eneHFmSel;

    // Jet Multiplicity Analysis
    MonitorElement*    _JM25njets ;
    MonitorElement*    _JM25ht    ;
    MonitorElement*    _JM25pt1   ;
    MonitorElement*    _JM25pt2   ;
    MonitorElement*    _JM25pt3   ;
    MonitorElement*    _JM25pt4   ;
    MonitorElement*    _JM80njets ;
    MonitorElement*    _JM80ht    ;
    MonitorElement*    _JM80pt1   ;
    MonitorElement*    _JM80pt2   ;
    MonitorElement*    _JM80pt3   ;
    MonitorElement*    _JM80pt4   ;

    //differential jet rates
    MonitorElement *djr10, *djr21, *djr32, *djr43;

    // SumET hiostograms
    MonitorElement *_sumEt ;
    MonitorElement *_sumEt1;
    MonitorElement *_sumEt2;
    MonitorElement *_sumEt3;
    MonitorElement *_sumEt4;
    MonitorElement *_sumEt5;


    static const unsigned int nphiBin = 36;

    static const unsigned int initSize = 1000; 

};

#endif
