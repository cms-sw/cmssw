
####### 

#  automatized plots generator for b-tagging performances
#  Adrien Caudron, 2013, UCL

#######

class plotInfo :
    def __init__ (self, name, title, #mandatory
                  legend="", Xlabel="", Ylabel="", logY=False, grid=False,
                  binning=None, Rebin=None,
                  doNormalization=False,
                  listTagger=None,
                  doPerformance=False, tagFlavor="B", mistagFlavor=["C","DUSG"]):
        self.name = name #name of the histos without postfix as PT/ETA bin or flavor
        self.title = title #title of the histograms : better if specific for the histogram
        self.legend = legend #legend name, if contain 'KEY', it will be replace by the list of keys you provide (as flavor, tagger ...)
        self.Xlabel = Xlabel #label of the X axis
        self.Ylabel = Ylabel #label of the Y axis
        self.logY = logY #if True : Y axis will be in log scale
        self.grid = grid #if True : a grid will be drawn
        self.binning = binning #if you want to change the binning put a list with [nBins,xmin,xmax]
        self.Rebin = Rebin #if you want to rebin the histos
        self.doNormalization = doNormalization #if you want to normalize to 1 all the histos 
        self.doPerformance = doPerformance #if you want to draw the performance as TGraph
        if self.doPerformance : 
            #replace TAG by the tag flavor choosen (B, C, UDSG ...)
            self.title = name.replace("TAG",tagFlavor)
            self.Xlabel = Xlabel.replace("TAG",tagFlavor)
            self.Ylabel = Ylabel.replace("TAG",tagFlavor)
            self.legend = legend.replace("TAG",tagFlavor)
            self.tagFlavor = tagFlavor
            self.mistagFlavor = mistagFlavor
        if listTagger is None :
            self.listTagger=None #you will take the list of tagger defined centrally
        else :
            self.listTagger=listTagger #you take the list passed as argument
#define here the histograms you interested by
#by jets
jetPt = plotInfo(name="jetPt", title="Pt of all jets", legend="isVAL KEY-jets", Xlabel="Pt (GeV/c)", Ylabel="abitrary units",
                 logY=False, grid=False,
                 binning=[300,10.,310.], Rebin=20, doNormalization=True,
                 listTagger=["CSV"]
                 )
jetEta = plotInfo(name="jetEta", title="Eta of all jets", legend="isVAL KEY-jets", Xlabel="#eta", Ylabel="abitrary units",
                  logY=False, grid=False,
                  binning=[11,90], Rebin=4, doNormalization=True,
                  listTagger=["CSV"]
                  )
discr = plotInfo(name="discr", title="Discriminant of all jets", legend="isVAL KEY-jets", Xlabel="Discriminant", Ylabel="abitrary units",
                 logY=False, grid=False,
                 binning=None, Rebin=None, doNormalization=True
                 )
effVsDiscrCut_discr = plotInfo(name="effVsDiscrCut_discr", title="Efficiency versus discriminant cut for all jets", legend="isVAL KEY-jets", Xlabel="Discriminant", Ylabel="efficiency",
                               logY=True, grid=True
                               )
#MC only
FlavEffVsBEff_discr = plotInfo(name="FlavEffVsBEff_B_discr", title="b-tag efficiency versus non b-tag efficiency", 
                               legend="KEY FLAV-jets versus b-jets", Xlabel="b-tag efficiency", Ylabel="non b-tag efficiency",
                               logY=True, grid=True
                               )
#MC only
performance = plotInfo(name="effVsDiscrCut_discr", title="TAG-tag efficiency versus non TAG-tag efficiency", 
                       legend="isVAL KEY-jets versus TAG-jets", Xlabel="TAG-tag efficiency", Ylabel="non TAG-tag efficiency",
                       logY=True, grid=True, 
                       doPerformance=True, tagFlavor="B", mistagFlavor=["C","DUSG"]
                       )
#MC only, to do C vs B and C vs light
performanceC = plotInfo(name="effVsDiscrCut_discr", title="TAG-tag efficiency versus non TAG-tag efficiency", 
                       legend="isVAL KEY-jets versus TAG-jets", Xlabel="TAG-tag efficiency", Ylabel="non TAG-tag efficiency",
                       logY=True, grid=True, 
                       doPerformance=True, tagFlavor="C", mistagFlavor=["B","DUSG"]
                       )
#by tracks
IP = plotInfo(name="ip_3D", title="Impact parameter", legend="isVAL KEY-jets", Xlabel="IP [cm]", Ylabel="abitrary units",
              logY=False, grid=False,
              binning=None,Rebin=None, doNormalization=True,
              listTagger=["IPTag"]
              )
IPe = plotInfo(name="ipe_3D", title="Impact parameter error", legend="isVAL KEY-jets", Xlabel="IPE [cm]", Ylabel="abitrary units",
               logY=False, grid=False, 
               binning=None, Rebin=None, doNormalization=True,
               listTagger=["IPTag"]
               )
IPs = plotInfo(name="ips_3D", title="Impact parameter significance", legend="isVAL KEY-jets", Xlabel="IPS", Ylabel="abitrary units", 
               logY=False, grid=False, 
               binning=None, Rebin=None, doNormalization=True,
               listTagger=["IPTag"]
               )
NTracks = plotInfo(name="selTrksNbr_3D", title="number of selected tracks", legend="isVAL KEY-jets", Xlabel="number of selected tracks", Ylabel="abitrary units",
                   logY=False, grid=False,
                   binning=None, Rebin=None, doNormalization=True,
                   listTagger=["IPTag"]
                   )
distToJetAxis = plotInfo(name="jetDist_3D", title="track distance to the jet axis", legend="isVAL KEY-jets", Xlabel="distance to the jet axis [cm]", Ylabel="abitrary units",
                         logY=False, grid=False,
                         binning=None, Rebin=None, doNormalization=True, 
                         listTagger=["IPTag"]
                         )
decayLength = plotInfo(name="decLen_3D", title="track decay length", legend="isVAL KEY-jets", Xlabel="decay length [cm]", Ylabel="abitrary units",
                       logY=False, grid=False,
                       binning=None, Rebin=None, doNormalization=True, listTagger=["IPTag"]
                       )
NHits = plotInfo(name="tkNHits_3D", title="Number of Hits / selected tracks", legend="isVAL KEY-jets", Xlabel="Number of Hits", Ylabel="abitrary units",
                 logY=False, grid=False,
                 binning=None, Rebin=None, doNormalization=True,
                 listTagger=["IPTag"]
                 )
NPixelHits = plotInfo(name="tkNPixelHits_3D", title="Number of Pixel Hits / selected tracks", legend="isVAL KEY-jets", Xlabel="Number of Pixel Hits", Ylabel="abitrary units",
                      logY=False, grid=False, 
                      binning=None, Rebin=None, doNormalization=True,
                      listTagger=["IPTag"]
                      )
NormChi2 = plotInfo(name="tkNChiSqr_3D", title="Normalized Chi2", legend="isVAL KEY-jets", Xlabel="Normilized Chi2", Ylabel="abitrary units",
                    logY=False, grid=False,
                    binning=None, Rebin=None, doNormalization=True,
                    listTagger=["IPTag"]
                    )
trackPt = plotInfo(name="tkPt_3D", title="track Pt", legend="isVAL KEY-jets", Xlabel="track Pt", Ylabel="abitrary units",
                   logY=False, grid=False,
                   binning=None, Rebin=None, doNormalization=True,
                   listTagger=["IPTag"]
                   )
#by SV and for CSV information
flightDist3Dval = plotInfo(name="flightDistance3dVal", title="3D flight distance value", legend="isVAL KEY-jets", Xlabel="3D flight distance value [cm]", Ylabel="abitrary units",
                           logY=False, grid=False,
                           binning=None, Rebin=None, doNormalization=True,
                           listTagger=["CSVTag"]
                           )
flightDist3Dsig = plotInfo(name="flightDistance3dSig", title="3D flight distance significance", legend="isVAL KEY-jets", Xlabel="3D flight distance significance", Ylabel="abitrary units",
                           logY=False, grid=False,
                           binning=None, Rebin=None, doNormalization=True,
                           listTagger=["CSVTag"]
                           )
jetNSecondaryVertices = plotInfo(name="jetNSecondaryVertices", title="Number of SV / jet", legend="isVAL KEY-jets", Xlabel="Number of SV / jet", Ylabel="abitrary units",
                           logY=False, grid=False,
                           binning=None, Rebin=None, doNormalization=True,
                           listTagger=["CSVTag"]
                           )
#Reco and pseudo vertex information
vertexMass = plotInfo(name="vertexMass", title="vertex mass", legend="isVAL KEY-jets", Xlabel="vertex mass GeV/c^2", Ylabel="abitrary units",
                      logY=False, grid=False,
                      binning=None, Rebin=None, doNormalization=True,
                      listTagger=["CSVTag"]
                      )
vertexNTracks = plotInfo(name="vertexNTracks", title="number of tracks at SV", legend="isVAL KEY-jets", Xlabel="number of tracks at SV", Ylabel="abitrary units",
                      logY=False, grid=False,
                      binning=None, Rebin=None, doNormalization=True,
                      listTagger=["CSVTag"]
                      )
vertexJetDeltaR = plotInfo(name="vertexJetDeltaR", title="Delta R between the SV and the jet axis", legend="isVAL KEY-jets", Xlabel="Delta R between the SV and the jet axis", Ylabel="abitrary units",
                      logY=False, grid=False,
                      binning=None, Rebin=None, doNormalization=True,
                      listTagger=["CSVTag"]
                      )
vertexEnergyRatio = plotInfo(name="vertexEnergyRatio", title="Energy Ratio between SV and the jet", legend="isVAL KEY-jets", Xlabel="Energy Ratio between SV and the jet", Ylabel="abitrary units",
                      logY=False, grid=False,
                      binning=None, Rebin=None, doNormalization=True,
                      listTagger=["CSVTag"]
                      )
#Reco, pseudo and no vertex information
vertexCategory = plotInfo(name="vertexCategory", title="Reco, Pseudo, No vertex", legend="isVAL KEY-jets", Xlabel="Reco, Pseudo, No vertex", Ylabel="abitrary units",
                          logY=False, grid=False,
                          binning=None, Rebin=None, doNormalization=True,
                          listTagger=["CSVTag"]
                          )
trackSip3dVal = plotInfo(name="trackSip3dVal", title="track IP 3D", legend="isVAL KEY-jets", Xlabel="track IP 3D [cm]", Ylabel="abitrary units",
                         logY=False, grid=False,
                         binning=None, Rebin=None, doNormalization=True,
                         listTagger=["CSVTag"]
                         )
trackSip3dSig = plotInfo(name="trackSip3dSig", title="track IPS 3D", legend="isVAL KEY-jets", Xlabel="track IPS 3D", Ylabel="abitrary units",
                         logY=False, grid=False,
                         binning=None, Rebin=None, doNormalization=True,
                         listTagger=["CSVTag"]
                         )
trackSip3dSigAboveCharm = plotInfo(name="trackSip3dSigAboveCharm", title="first track IPS 3D lifting SV mass above charm", legend="isVAL KEY-jets", Xlabel="first track IPS 3D lifting SV mass above charm", Ylabel="abitrary units",
                                   logY=False, grid=False,
                                   binning=None, Rebin=None, doNormalization=True,
                                   listTagger=["CSVTag"]
                                   )
trackDeltaR = plotInfo(name="trackDeltaR", title="Delta R between the track and the jet axis", legend="isVAL KEY-jets", Xlabel="DeltaR(track,jet axis)", Ylabel="abitrary units",
                       logY=False, grid=False,
                       binning=None, Rebin=None, doNormalization=True,
                       listTagger=["CSVTag"]
                       )
trackEtaRel = plotInfo(name="trackEtaRel", title="track eta relative to the jet axis", legend="isVAL KEY-jets", Xlabel="track eta relative to the jet axis", Ylabel="abitrary units",
                       logY=False, grid=False,
                       binning=None, Rebin=None, doNormalization=True,
                       listTagger=["CSVTag"]
                       )
trackDecayLenVal = plotInfo(name="trackDecayLenVal", title="track decay length", legend="isVAL KEY-jets", Xlabel="track decay length", Ylabel="abitrary units",
                            logY=False, grid=False,
                            binning=None, Rebin=None, doNormalization=True,
                            listTagger=["CSVTag"]
                            )
trackSumJetDeltaR = plotInfo(name="trackSumJetDeltaR", title="Delta R between track 4-vector sum and jet axis", legend="isVAL KEY-jets", Xlabel="Delta R between track 4-vector sum and jet axis", Ylabel="abitrary units",
                             logY=False, grid=False,
                             binning=None, Rebin=None, doNormalization=True,
                             listTagger=["CSVTag"]
                             )
trackJetDist = plotInfo(name="trackJetDist", title="track distance to jet axis", legend="isVAL KEY-jets", Xlabel="track distance to jet axis", Ylabel="abitrary units",
                        logY=False, grid=False,
                        binning=None, Rebin=None, doNormalization=True,
                        listTagger=["CSVTag"]
                        )
trackSumJetEtRatio = plotInfo(name="trackSumJetEtRatio", title="track sum Et / jet energy", legend="isVAL KEY-jets", Xlabel="track sum Et / jet energy", Ylabel="abitrary units",
                              logY=False, grid=False,
                              binning=None, Rebin=None, doNormalization=True,
                              listTagger=["CSVTag"]
                              )
trackPtRel = plotInfo(name="trackPtRel", title="track Pt relative to jet axis", legend="isVAL KEY-jets", Xlabel="track Pt relative to jet axis", Ylabel="abitrary units",
                      logY=False, grid=False,
                      binning=None, Rebin=None, doNormalization=True,
                      listTagger=["CSVTag"]
                      )
trackPtRatio = plotInfo(name="trackPtRatio", title="track Pt relative to jet axis, normalized to its energy", legend="isVAL KEY-jets", Xlabel="track Pt relative to jet axis, normalized to its energy", Ylabel="abitrary units",
                        logY=False, grid=False,
                        binning=None, Rebin=None, doNormalization=True,
                        listTagger=["CSVTag"]
                        )
trackMomentum = plotInfo(name="trackMomentum", title="track momentum", legend="isVAL KEY-jets", Xlabel="track momentum [GeV/c]", Ylabel="abitrary units",
                         logY=False, grid=False,
                         binning=None, Rebin=None, doNormalization=True,
                         listTagger=["CSVTag"]
                         )
trackPPar = plotInfo(name="trackPPar", title="track parallel momentum along the jet axis", legend="isVAL KEY-jets", Xlabel="track parallel momentum along the jet axis", Ylabel="abitrary units",
                     logY=False, grid=False,
                     binning=None, Rebin=None, doNormalization=True,
                     listTagger=["CSVTag"]
                     )
trackPParRatio = plotInfo(name="trackPParRatio", title="track parallel momentum along the jet axis, normalized to its energy", legend="isVAL KEY-jets", Xlabel="track parallel momentum along the jet axis, normalized to its energy", Ylabel="abitrary units",
                          logY=False, grid=False,
                          binning=None, Rebin=None, doNormalization=True,
                          listTagger=["CSVTag"]
                          )
