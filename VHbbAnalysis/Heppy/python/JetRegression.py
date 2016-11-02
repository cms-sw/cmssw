from VHbbAnalysis.Heppy.vhbbobj import ptRel
from PhysicsTools.HeppyCore.utils.deltar import deltaR,deltaPhi

from math import *
import ROOT
import array
class JetRegression :
    def __init__(self,weightfile,name) :
#      weights = { "../weights/Zll_weights_phys14.xml" , "../weights/Wln_weights_phys14.xml", "../weights/Znn_weights_phys14.xml"}		
#      reader_name = {"jet0Regression_zll", "jet0Regression_wln","jet0Regression_znn"}	
#      for i in range(0,3):
        reader = ROOT.TMVA.Reader()
        self.Jet_pt =array.array('f',[0])
#        self.Jet_corr = array.array('f',[0])
        #self.rho = array.array('f',[0])
        self.nPVs = array.array('f',[0])
        self.Jet_eta = array.array('f',[0])
        self.Jet_mt = array.array('f',[0])
        self.Jet_leadTrackPt = array.array('f',[0])
        self.Jet_leptonPtRel = array.array('f',[0])
        self.Jet_leptonPt =  array.array('f',[0])
        self.Jet_leptonDeltaR = array.array('f',[0])
        #self.Jet_chmEF = array.array('f',[0])
        #self.Jet_chHEF = array.array('f',[0])
        self.Jet_neHEF = array.array('f',[0])
        #self.Jet_totHEF  = array.array('f',[0])
        self.Jet_neEmEF = array.array('f',[0])
        #self.Jet_chMult = array.array('f',[0])
        self.Jet_vtxPt = array.array('f',[0])
        self.Jet_vtxMass = array.array('f',[0])
        self.Jet_vtx3dL = array.array('f',[0])
        self.Jet_vtxNtrk = array.array('f',[0])
        self.Jet_vtx3deL = array.array('f',[0])
        reader.AddVariable("Jet_pt",self.Jet_pt)
        #reader.AddVariable("Jet_corr",self.Jet_corr)
        #reader.AddVariable("rho",self.rho)
        reader.AddVariable("nPVs",self.nPVs) 
        reader.AddVariable("Jet_eta",self.Jet_eta)
        reader.AddVariable("Jet_mt",self.Jet_mt)
        reader.AddVariable("Jet_leadTrackPt",self.Jet_leadTrackPt)
        reader.AddVariable("Jet_leptonPtRel",self.Jet_leptonPtRel)
        reader.AddVariable("Jet_leptonPt",self.Jet_leptonPt)
        reader.AddVariable("Jet_leptonDeltaR",self.Jet_leptonDeltaR)
        #reader.AddVariable("Jet_chEmEF",self.Jet_chEmEF)
        #reader.AddVariable("Jet_chHEF",self.Jet_chHEF)
        #reader.AddVariable("Jet_neHEF",self.Jet_neHEF)
        reader.AddVariable("Jet_neHEF",self.Jet_neHEF)
        reader.AddVariable("Jet_neEmEF",self.Jet_neEmEF)
        #reader.AddVariable("Jet_chMult",self.Jet_chMult)
        reader.AddVariable("Jet_vtxPt",self.Jet_vtxPt)
        reader.AddVariable("Jet_vtxMass",self.Jet_vtxMass)
        reader.AddVariable("Jet_vtx3dL",self.Jet_vtx3dL)
        reader.AddVariable("Jet_vtxNtrk",self.Jet_vtxNtrk)
        reader.AddVariable("Jet_vtx3deL",self.Jet_vtx3deL)
        reader.BookMVA(name,weightfile)
        self.reader=reader
        self.name=name

    def evaluateRegression(self, event, attrName="pt_reg", analysis=""):
#self.readCollections( event.input )
# uniform with https://github.com/degrutto/HbbRegression 
	#self.rho[0] = event.rho
        self.nPVs[0] = len(event.goodVertices) 
	for j in event.jetsForHiggs :
            pt_corr = 1.0
            if ("JEC" in analysis and hasattr(j, analysis)):
                pt_corr = getattr(j, analysis)/getattr(j, "corr") if hasattr(j, "corr") and getattr(j, "corr") !=0 else 1.0
                #print "JEC"+analysis, pt_corr
            elif ("JER" in analysis and hasattr(j, analysis)):
                pt_corr = getattr(j, analysis)/getattr(j, "corrJER") if hasattr(j, "corrJER") and getattr(j, "corrJER")!=0 else 1.0
                #print "JER"+analysis, pt_corr
            if pt_corr<=0.:
                pt_corr = 1.0
            self.Jet_pt[0] = j.pt()*pt_corr
            #self.Jet_corr[0] = j.rawFactor()/(pt_corr if "JER" not in analysis else 1.0)
           # self.Jet_corr[0] =  (1/ j.rawFactor() ) / j.corrJER * ( pt_corr if "JER" not in analysis else 1.0 )
            #self.Jet_corr[0] =  getattr(j, "corr") if hasattr(j, "corr") * (pt_corr if "JER" not in analysis else 1.0 ) else 0.0
            self.Jet_mt[0] = j.mt()*pt_corr
            self.Jet_eta[0] = j.eta()
            self.Jet_leadTrackPt[0] = j.leadTrackPt()
            if len(j.leptons) > 0       :
                self.Jet_leptonPtRel[0] = ptRel(j.leptons[0].p4(),j.p4()*pt_corr)
                self.Jet_leptonPt[0] =  j.leptons[0].pt()
                self.Jet_leptonDeltaR[0] = deltaR(j.leptons[0].p4().eta(),j.leptons[0].p4().phi(),j.p4().eta(),j.p4().phi())
            else:
                self.Jet_leptonPtRel[0] = 0
                self.Jet_leptonPt[0] =  0
                self.Jet_leptonDeltaR[0] =0
            #self.Jet_chEmEF[0] = min(1,0, j.chargedEmEnergyFraction())
            #self.Jet_chHEF[0] = min(1.-, j.chargedHadronEnergyFraction())
            self.Jet_neHEF[0] = min(1.0,j.neutralHadronEnergyFraction())
           # self.Jet_totHEF[0] = min(1.0, j.chargedHadronEnergyFraction() + j.neutralHadronEnergyFraction())    
            self.Jet_neEmEF[0] = min(1.0,j.neutralEmEnergyFraction())
            #self.Jet_chMult[0] = j.chargedMultiplicity()
            self.Jet_vtxPt[0] = sqrt(j.userFloat("vtxPx")**2 + j.userFloat("vtxPy")**2)
            self.Jet_vtxMass[0] = j.userFloat("vtxMass")
            self.Jet_vtx3dL[0] = max(0., j.userFloat("vtx3DVal"))
            self.Jet_vtxNtrk[0] = j.userFloat("vtxNtracks")
            self.Jet_vtx3deL[0] = self.Jet_vtx3dL[0]/j.userFloat("vtx3DSig") if j.userFloat("vtx3DSig") > 0 else 0
            setattr(j,attrName+analysis,self.reader.EvaluateRegression(self.name)[0])# * self.Jet_pt[0])

                #j.pt_reg = self.reader.EvaluateRegression(self.name)[0]
		



