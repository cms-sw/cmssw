from PhysicsTools.Heppy.analyzers.core.Analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoHandle import AutoHandle
from PhysicsTools.HeppyCore.utils.deltar import deltaR
from copy import deepcopy
from math import *
import itertools
import ROOT
import array, os
class AdditionalBTag( Analyzer ):

    def declareHandles(self):
        super(AdditionalBTag, self).declareHandles()
        #self.handles['btagnew'] = AutoHandle( ("combinedInclusiveSecondaryVertexV2BJetTags","","EX"), "edm::AssociationVector<edm::RefToBaseProd<reco::Jet>,vector<float>,edm::RefToBase<reco::Jet>,unsigned int,edm::helper::AssociationIdenticalKeyReference>")
        #self.handles['btagcsv'] = AutoHandle( ("combinedSecondaryVertexBJetTags","","EX"), "edm::AssociationVector<edm::RefToBaseProd<reco::Jet>,vector<float>,edm::RefToBase<reco::Jet>,unsigned int,edm::helper::AssociationIdenticalKeyReference>")
        self.handles['btagSoftEl'] = AutoHandle( ("softPFElectronBJetTags","","EX"), "edm::AssociationVector<edm::RefToBaseProd<reco::Jet>,vector<float>,edm::RefToBase<reco::Jet>,unsigned int,edm::helper::AssociationIdenticalKeyReference>")
        self.handles['btagSoftMu'] = AutoHandle( ("softPFMuonBJetTags","","EX"), "edm::AssociationVector<edm::RefToBaseProd<reco::Jet>,vector<float>,edm::RefToBase<reco::Jet>,unsigned int,edm::helper::AssociationIdenticalKeyReference>")
        
        self.bdtVars = ["Jet_CSV", "Jet_CSVIVF", "Jet_JP", "Jet_JBP", "Jet_SoftMu", "Jet_SoftEl"] 
        reader = ROOT.TMVA.Reader()
        self.Jet_CSV = array.array('f',[0]) 
        self.Jet_CSVIVF = array.array('f',[0]) 
        self.Jet_JP = array.array('f',[0]) 
        self.Jet_JBP = array.array('f',[0]) 
        self.Jet_SoftMu = array.array('f',[0]) 
        self.Jet_SoftEl = array.array('f',[0]) 
        
        for var in self.bdtVars: 
            reader.AddVariable(var, getattr(self, var))
        
        #https://github.com/cms-data/RecoBTag-Combined/blob/master/CombinedMVAV2_13_07_2015.weights.xml.gz
        self.weightfile = os.environ['CMSSW_BASE']+"/src/VHbbAnalysis/Heppy/data/btag/CombinedMVAV2_13_07_2015.weights.xml"
        print "booking BDT from {0}".format(self.weightfile)  
        reader.BookMVA("bdt", self.weightfile)
        self.reader=reader

    def readTag(self, event, name):
        newtags =  self.handles[name].product()
        for i in xrange(0,len(newtags)) :
             for j in event.cleanJets :
                if j.physObj == newtags.key(i).get() :
                    setattr(j, name, newtags.value(i))

    def addNewBTag(self,event):
        #self.readTag(event, "btagnew")
        #self.readTag(event, "btagcsv")
        self.readTag(event, "btagSoftEl")
        self.readTag(event, "btagSoftMu")

    def normalize(self, varname):
        v = getattr(self, varname)
        if v[0] <=0:
            v[0] = 0
        setattr(self, varname, v)
    def process(self, event):

        self.readCollections( event.input )
        self.addNewBTag(event)
        for j in event.cleanJets:

            self.Jet_CSV[0]=j.btag("pfCombinedSecondaryVertexV2BJetTags") 
            self.Jet_CSVIVF[0]=j.btag("pfCombinedInclusiveSecondaryVertexV2BJetTags") 
            self.Jet_JP[0]=j.btag("pfJetProbabilityBJetTags") 
            self.Jet_JBP[0]=j.btag("pfJetBProbabilityBJetTags") 
            self.Jet_SoftMu[0]=j.btagSoftMu
            self.Jet_SoftEl[0]=j.btagSoftEl
            
            for var in self.bdtVars:
                self.normalize(var)
            j.btagBDT = self.reader.EvaluateMVA("bdt")
        return True


