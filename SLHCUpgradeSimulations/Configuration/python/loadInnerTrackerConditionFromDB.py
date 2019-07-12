import socket
from CondCore.CondDB.CondDB_cfi import *
'''Helper procedure that loads phase-2 Inner Trackers Conditions from database'''

CondDBPhase2ITConnection = CondDB.clone( connect = cms.string( 'frontier://FrontierPrep/CMS_CONDITIONS' ) )
loadPhase2InneTrackerConditions = cms.ESSource( "PoolDBESSource",
                                                CondDBPhase2ITConnection,
                                                globaltag        = cms.string( '' ),
                                                snapshotTime     = cms.string( '' ),
                                                toGet            = cms.VPSet(),   # hook to override or add single payloads
                                                DumpStat         = cms.untracked.bool( False ),
                                                ReconnectEachRun = cms.untracked.bool( False ),
                                                RefreshAlways    = cms.untracked.bool( False ),
                                                RefreshEachRun   = cms.untracked.bool( False ),
                                                RefreshOpenIOVs  = cms.untracked.bool( False ),
                                                pfnPostfix       = cms.untracked.string( '' ),
                                                pfnPrefix        = cms.untracked.string( '' ),
                                              )

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class DBConfiguration():
    '''Helper class to store the  configuration object DB'''
    def __init__(self,vGeometry=None,vLA=None,vLAwidth=None,vSimLA=None,vGenError=None,vTemplate1D=None,vTemplate2Dnum=None,vTemplate2Dden=None):
        self._vGeometry      = vGeometry
        self._vLA            = vLA 
        self._vLAwidth       = vLAwidth
        self._vSimLA         = vSimLA
        self._vGenError      = vGenError
        self._vTemplate1D    = vTemplate1D 
        self._vTemplate2Dnum = vTemplate2Dnum 
        self._vTemplate2Dden = vTemplate2Dden 

    def retAssociations(self):

        records = {
            "SiPixelLorentzAngleSimRcd:"               : "SiPixelSimLorentzAngle_phase2_T%s_v%s_mc"       % (self.vGeometry,self.vSimLA),
            "SiPixelLorentzAngleRcd:forWidth"          : "SiPixelLorentzAngle_phase2_forWidth_T%s_v%s_mc" % (self.vGeometry,self.vLAwidth),
            "SiPixelLorentzAngleRcd:"                  : "SiPixelLorentzAngle_phase2_T%s_v%s_mc"          % (self.vGeometry,self.vLA),
            #########################################
            # Not yet implemented
            #########################################
            #"SiPixelGenErrorDBObjectRcd:"              : "SiPixelGenErrorDBObject_phase2_T%s_v%s_mc"      % (self.vGeometry,self.vGenError),
            #"SiPixelTemplateDBObjectRcd:"              : "SiPixelTemplateDBObject_phase2_T%s_v%s_mc"      % (self.vGeometry,self.vTemplate1D),
            #"SiPixel2DTemplateDBObjectRcd:denominator" : "SiPixel2DTemplateDBObject_phase2_T%s_v%s_den"   % (self.vGeometry,self.vTemplate2Dden),
            #"SiPixel2DTemplateDBObjectRcd:numerator"   : "SiPixel2DTemplateDBObject_phase2_T%s_v%s_num"   % (self.vGeometry,self.vTemplate2Dnum)
        }
        
        return records

    ###################################
    # Print version used
    ###################################
    def printConfig(self):
    
        associations = dict(
            vGeometry      = None,
            vLA            = 'SiPixelLorentzAngleRcd',
            vLAwidth       = 'SiPixelLorentzAngleRcd (forWidth)',
            vSimLA         = 'SiPixelLorentzAngleSimRcd', 
            vGenError      = 'SiPixelGenErrorDBObjectRcd',  
            vTemplate1D    = 'SiPixelTemplatedDBOjectRcd',
            vTemplate2Dnum = 'SiPixel2DTemplateDBObjectRcd (numerator)',
            vTemplate2Dden = 'SiPixel2DTemplateDBObjectRcd (denominator)',
        )    

        attrs = self.__dict__
        print(" ===>>> Customization of Inner Tracker conditions for geometry T%s" % self.vGeometry)
        for key,value in attrs.items():
            if value is not None:
                if associations[key] is not None:
                    print (" Customizing %s with Tag version n. %s" % (associations[key],value))
            
    ###################################
    # Geometry version
    ###################################
    @property
    def vGeometry(self):
        """version of the Tracker Geometry."""
        return self._vGeometry

    @vGeometry.setter
    def vGeometry(self, value):
        self._vGeometry = value       

    ###################################
    # LorentzAngle version
    ###################################
    @property
    def vLA(self):
        """version of the Lorentz Angle payload."""
        return self._vLA

    @vLA.setter
    def vLA(self, value):
        self._vLA = value  

    ###################################
    # LorentzAngle width version
    ###################################
    @property
    def vLAwidth(self):
        """version of the Lorentz Angle width payload."""
        return self._vLAwidth

    @vLAwidth.setter
    def vLAwidth(self, value):
        self._vLAwidth = value  
    
    ###################################
    # Sim LA version
    ###################################
    @property
    def vSimLA(self):
        """version of the Simulation Lorentz Angle payload."""
        return self._vSimLA

    @vSimLA.setter
    def vSimLA(self, value):
        self._vSimLA = value  

    ###################################
    # Generic Errors version
    ###################################
    @property
    def vGenError(self):
        """version of the Generic Error payload."""
        return self._vGenError

    @vGenError.setter
    def vGenError(self, value):
        self._vGenError = value  

    ###################################
    # 1D Template version
    ###################################
    @property
    def vTemplate1D(self):
        """version of the Template 1D payload."""
        return self._vTemplate1D

    @vTemplate1D.setter
    def vTemplate1D(self, value):
        self._vTemplate1D = value  

    ###################################
    # 2D template (numerator) version
    ###################################
    @property
    def vTemplate2Dnum(self):
        """version of the Template 2D (numerator) payload."""
        return self._vTemplate2Dnum

    @vTemplate2Dnum.setter
    def vTemplate2Dnum(self, value):
        self._vTemplate2Dnum = value  

    ###################################
    # 2D template (denominator) version
    ###################################
    @property
    def vTemplate2Dden(self):
        """version of the Template 2D (denominator) payload."""
        return self._vTemplate2Dden

    @vTemplate2Dden.setter
    def vTemplate2Dden(self, value):
        print("setter of vTemplate2Dden called")
        self._vTemplate2Dden = value  

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def appendConditions(DBConfig):
    '''function to load all the addtional Inner Tracker payload (geometry dependent)'''

    if not hasattr(loadPhase2InneTrackerConditions ,'toGet'):
        loadPhase2InneTrackerConditions.toGet=cms.VPSet()
    
    toExtend=[]
    for record,tag in DBConfig.retAssociations().iteritems():
        rcd, label = tuple(record.split(':'))
        toExtend.append(
            cms.PSet(
                record = cms.string(rcd),
                tag    = cms.string(tag),
                label = cms.untracked.string(label)
                )
        )
    

    print toExtend
    loadPhase2InneTrackerConditions.toGet.extend(cms.VPSet(*toExtend))

    
                                                 
    
