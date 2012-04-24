import ROOT
from DataFormats.FWLite import Events, Handle
import math
import array
from tdrstyle import setTDRStyle


def deltaPhi(phi1, phi2):
     PHI = abs(phi1-phi2)
     if (PHI<=3.14159265):
         return PHI
     else:
         return 2*3.14159265-PHI

def deltaR(eta1, phi1, eta2, phi2) :
    deta = eta1-eta2
    dphi = deltaPhi(phi1,phi2)
    return math.sqrt(deta*deta + dphi*dphi)


def histograms(fileName, label1, label2):
    events = Events(fileName)
    #vertex
    VertexHandle = Handle('std::vector<reco::Vertex>')
    vertexLabel = "offlinePrimaryVertices"

    #taus
    TauHandle  = Handle('std::vector<reco::PFTau>')
    offlineTauLabel = label1
    hltTauLabel  = label2

    # Histograms
    resolutionPt = ROOT.TH1F("resolutionPt","",200,-1.,1.)
    deltaZTauTau = ROOT.TH1F("deltaZTauTau","",200,-1.,1.)
    #xbin = [18,20,22,26,30,40,50]
    xbin = [15,16,18,19,20,22,26,30,40,50]
    xbins = array.array('d',xbin)
    numerator = ROOT.TH1F("numerator","",len(xbin)-1,xbins)
    denominator = ROOT.TH1F("denominator","",len(xbin)-1,xbins)
    xbinEta = [0.,0.5,1.,1.5,2.5]
    xbinsEta = array.array('d',xbinEta)
    numeratorEta = ROOT.TH1F("numeratorEta","",len(xbinEta)-1,xbinsEta)
    denominatorEta = ROOT.TH1F("denominatorEta","",len(xbinEta)-1,xbinsEta)
    numeratorVertex = ROOT.TH1F("numeratorVertex","",10,0,40)
    denominatorVertex = ROOT.TH1F("denominatorVertex","",10,0,40)

    #cuts
    hltPtCut = 20.
    for event in events:
        # getting the handle from the event
        event.getByLabel(offlineTauLabel, TauHandle)
        offlineTaus = TauHandle.product()
        
        event.getByLabel(hltTauLabel,TauHandle)
        hltTaus = TauHandle.product()

        event.getByLabel(vertexLabel,VertexHandle)
        offlinevertex = VertexHandle.product().at(0).position()
        vertexSize = VertexHandle.product().size()

        #loop on denominator
        for tau in offlineTaus:
            if tau.pt() < 15: continue
            #if abs(tau.eta())<2.1: continue
            if not(tau.leadPFChargedHadrCand().isNonnull()): continue
            #if (tau.signalPFChargedHadrCands().size() < 2): continue
            denominator.Fill(tau.pt())
            if tau.pt() > hltPtCut:
                 denominatorEta.Fill(abs(tau.eta()))
                 denominatorVertex.Fill(vertexSize*1.)
            #loop on numerator
            for hltTau in hltTaus:
                #pT cut at HLT
                if hltTau.pt() < hltPtCut: continue
                if deltaR(tau.eta(),tau.phi(),hltTau.eta(),hltTau.phi()) < 0.2:
                    resolutionPt.Fill( (hltTau.pt()-tau.pt())/tau.pt())
                    numerator.Fill(tau.pt())                    
                    if tau.pt() > hltPtCut:
                         numeratorEta.Fill(abs(tau.eta()))
                         numeratorVertex.Fill(vertexSize*1.)
                    if hltTau.leadPFChargedHadrCand().isNonnull():
                        deltaZTauTau.Fill ( tau.leadPFChargedHadrCand().trackRef().dz(offlinevertex) - hltTau.leadPFChargedHadrCand().trackRef().dz(offlinevertex) )
    numerator.Sumw2()
    denominator.Sumw2()            
    numeratorEta.Sumw2()
    denominatorEta.Sumw2()
    numeratorVertex.Sumw2()
    denominatorVertex.Sumw2()
    return deltaZTauTau,resolutionPt, numeratorEta, denominatorEta, numerator, denominator, numeratorVertex, denominatorVertex



def plot(file, var='pt', rel=True):

     #just to avoid opening windows
     ROOT.gROOT.SetBatch()

     setTDRStyle()
     frame = ROOT.TH1F("frame","",100,15,50)
     frameEta = ROOT.TH1F("frameEta","",10,0.,2.5)
     frameVertex = ROOT.TH1F("frameVertex","",100,0.,50.)
     frame.GetXaxis().SetTitle("Offline Tau pT [GeV]")
     frameEta.GetXaxis().SetTitle("Offline Tau Eta ")
     frameVertex.GetXaxis().SetTitle("Offline # Vertices")
     frame.GetYaxis().SetTitle("Efficiency")
     frameEta.GetYaxis().SetTitle("Efficiency")
     frameVertex.GetYaxis().SetTitle("Efficiency")
     frame.SetMinimum(0.01)
     frame.SetMaximum(1.01)
     frameEta.SetMinimum(0.01)
     frameEta.SetMaximum(1.01)
     frameVertex.SetMinimum(0.01)
     frameVertex.SetMaximum(1.01)
     
     nHisto = 2
     if var == 'eta':
          nHisto = 2
          frame = frameEta
     if var == 'pt':
          nHisto = 4
     if var == "vertex":
          nHisto = 6
          frame = frameVertex
     
     #Eff plotting
     c1 = ROOT.TCanvas()
     c1.Divide(2,2)
     c1.cd(1)
     myhistos = histograms(file,"offlineSelectedTaus","hltPFTaus")
     myhistos[nHisto].Divide(myhistos[nHisto],myhistos[nHisto+1],1,1,"B")
     frame.DrawCopy()
     myhistos[nHisto].Draw("pE1same")
     
     c1.cd(2)
     frame.DrawCopy()
     myhistosLeadTrk = histograms(file,"offlineSelectedTaus","hltSelectedPFTausTrackFinding")
     if rel:
          myhistosLeadTrkRel = histograms(file,"offlineSelectedTaus","hltPFTaus")
          myhistosLeadTrk[nHisto].Divide(myhistosLeadTrk[nHisto],myhistosLeadTrkRel[nHisto],1,1,"B")
     else:
          myhistosLeadTrk[nHisto].Divide(myhistosLeadTrk[nHisto],myhistosLeadTrk[nHisto+1],1,1,"B")
          
     myhistosLeadTrk[nHisto].SetMarkerColor(2)
     myhistosLeadTrk[nHisto].Draw("pE1same")
          
     c1.cd(3)
     frame.DrawCopy()
     myhistosLeadTrkIso = histograms(file,"offlineSelectedTaus","hltSelectedPFTausTrackFindingLooseIsolation")
     if rel:
          myhistosLeadTrkIsoRel = histograms(file,"offlineSelectedTaus","hltSelectedPFTausTrackFinding")
          myhistosLeadTrkIso[nHisto].Divide(myhistosLeadTrkIso[nHisto],myhistosLeadTrkIsoRel[nHisto],1,1,"B")
     else:
          myhistosLeadTrkIso[nHisto].Divide(myhistosLeadTrkIso[nHisto],myhistosLeadTrkIso[nHisto+1],1,1,"B")
     myhistosLeadTrkIso[nHisto].SetMarkerColor(3)
     myhistosLeadTrkIso[nHisto].Draw("pe1same")

     c1.cd(4)
     # myhistosLeadTrkVertex = histograms(file,"offlineSelectedTaus","hltIsoMuPFTauVertexFinderRecoMuon")
     myhistosLeadTrkVertex = histograms(file,"offlineSelectedTaus","hltSelectedPFTausTrackFindingLooseIsolation")
     myhistosLeadTrkVertex[nHisto].Divide(myhistosLeadTrkVertex[nHisto],myhistosLeadTrkVertex[nHisto+1],1,1,"B")
     myhistosLeadTrkVertex[nHisto].SetMarkerColor(4)
     frame.DrawCopy()
     myhistosLeadTrkVertex[nHisto].Draw("pe1same")

     c1.SaveAs("eff_"+var+".png")
