'''Customization functions for cmsDriver to change the phase-2 tracker reco material for geometry D49'''
import FWCore.ParameterSet.Config as cms
def customizeRecoMaterialD49(process):
    
    ''' will replace one tracker reco material file with another one for geometry D49
    syntax: --customise SLHCUpgradeSimulations/Configuration/customise_TkRecoMaterial.customizeRecoMaterialD49
    '''
    
    if hasattr(process,'XMLIdealGeometryESSource') and hasattr(process.XMLIdealGeometryESSource,'geomXMLFiles'):
        
        try:
            process.XMLIdealGeometryESSource.geomXMLFiles.remove(
                'Geometry/TrackerRecoData/data/PhaseII/TiltedTracker613_MB_2019_04/trackerRecoMaterial.xml'
            )
            process.XMLIdealGeometryESSource.geomXMLFiles.append(
                'Geometry/TrackerRecoData/data/PhaseII/TiltedTracker613_MB_2019_04/v2_ITonly/trackerRecoMaterial.xml'
            )
        except ValueError:
            raise SystemExit("\n\n ERROR! Could not replace trackerRecoMaterial.xml file, please check if D49 geometry is being used \n\n")
            
    return process
