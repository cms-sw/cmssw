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
class DBConfiguration():
    '''Helper class to store the  configuration object DB'''
    def __init__(self,vGeometry=None,vLA=None,vLAwidth=None,vSimLA=None,vGenError=None,vTemplate1D=None,vTemplate2Dn=None,vTemplate2Dd=None):
        self._vGeometry    = vGeometry
        self._vLA          = vLA 
        self._vLAwidth     = vLAwidth
        self._vSimLA       = vSimLA
        self._vGenError    = vGenError
        self._vTemplate1D  = vTemplate1D 
        self._vTemplate2Dn = vTemplate2Dn 
        self._vTemplate2Dd = vTemplate2Dd 

    ###################################
    # Print version used
    ###################################
    def printConfig(self):
    
        associations = dict(
            vGeometry    = None,
            vLA          = 'SiPixelLorentzAngleRcd',
            vLAwidth     = 'SiPixelLorentzAngleRcd (forWidth)',
            vSimLA       = 'SiPixelLorentzAngleSimRcd', 
            vGenError    = 'SiPixelGenErrorDBObjectRcd',  
            vTemplate1D  = 'SiPixelTemplatedDBOjectRcd',
            vTemplate2Dn = 'SiPixel2DTemplateDBObjectRcd (numerator)',
            vTemplate2Dd = 'SiPixel2DTemplateDBObjectRcd (denominator)',
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
    def vTemplate2Dn(self):
        """version of the Template 2D (numerator) payload."""
        return self._vTemplate2Dn

    @vTemplate2Dn.setter
    def vTemplate2Dn(self, value):
        self._vTemplate2Dn = value  

    ###################################
    # 2D template (denominator) version
    ###################################

    @property
    def vTemplate2Dd(self):
        """version of the Template 2D (denominator) payload."""
        return self._vTemplate2Dd

    @vTemplate2Dd.setter
    def vTemplate2Dd(self, value):
        print("setter of vTemplate2Dd called")
        self._vTemplate2Dd = value  

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def appendConditions(DBConfig):
    '''function to load all the addtional Inner Tracker payload (geometry dependent)'''

    if not hasattr(loadPhase2InneTrackerConditions ,'toGet'):
        loadPhase2InneTrackerConditions.toGet=cms.VPSet()
    
    loadPhase2InneTrackerConditions.toGet.extend(cms.VPSet(cms.PSet(record = cms.string('SiPixelLorentzAngleRcd'),
                                                                    tag = cms.string("SiPixelLorentzAngle_phase2_T%s_v%s_mc" % (DBConfig.vGeometry,DBConfig.vLA))
                                                                   ),
                                                           cms.PSet(record = cms.string('SiPixelLorentzAngleSimRcd'),
                                                                    tag = cms.string("SiPixelSimLorentzAngle_phase2_T%s_v%s_mc" % (DBConfig.vGeometry,DBConfig.vSimLA))
                                                                   ),                                            
                                                           cms.PSet(record = cms.string('SiPixelLorentzAngleRcd'),
                                                                    tag = cms.string("SiPixelLorentzAngle_phase2_forWidth_T%s_v%s_mc" % (DBConfig.vGeometry,DBConfig.vLAwidth)),
                                                                    label = cms.untracked.string("forWidth")
                                                                   ),
                                                           #
                                                           # NOT YET IMPLEMENTED, BUT SOON TO BE INTRODUCED
                                                           #
                                                           # cms.PSet(record = cms.string('SiPixelGenErrorDBObjectRcd'),
                                                           #          tag = cms.string("SiPixelGenErrorDBObject_phase2_T%s_v%s_mc" % (DBConfig.vGeometry,DBConfig.vGenError))
                                                           #         ),
                                                           # cms.PSet(record = cms.string('SiPixelTemplateDBObjectRcd'),
                                                           #          tag = cms.string("SiPixelTemplateDBObject_phase2_T%s_v%s_mc" % (DBConfig.vGeometry,DBConfig.vTemplate1D))
                                                           #         ),
                                                           # cms.PSet(record = cms.string('SiPixel2DTemplateDBObjectRcd'),
                                                           #          tag = cms.string("SiPixel2DTemplateDBObject_phase2_T%s_v%s_den" % (DBConfig.vGeometry,DBConfig.vTemplate2Dd) ),
                                                           #          label = cms.untracked.string("denominator")
                                                           #         ),                                            
                                                           # cms.PSet(record = cms.string('SiPixel2DTemplateDBObjectRcd'),
                                                           #          tag = cms.string("SiPixel2DTemplateDBObject_phase2_T%s_v%s_num" % (DBConfig.vGeometry,DBConfig.vTemplate2Dn) ),
                                                           #          label = cms.untracked.string("numerator")
                                                           #         )
                                                           )
                                                 )
                                                 
                                                 
    
