import re, os
from FileNamesHelper import *


test_edm_file = re.compile("_EdmSize$", re.IGNORECASE)



#to match float we could instead use: [-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?

edmSize_line_parsing_reg = re.compile( \
	r"""
	# <C++ type>_<module_name>_[opt:_<module label>]_<process name which produced>.(dot)	
	^([^_]+)_([^_]+)_([^_]*)_([^.]+[.])
	# <plain_size> <compressed_size>
	\s([^\s]+)\s(.+)$
""", re.VERBOSE)


def parseEdmSize(lines):
	"""
	Returns a list of dictionaries

	Example of data:
	>>> parseEdmSize(lines = ( 'File MINBIAS__RAW2DIGI,RECO.root Events 8000', 'TrackingRecHitsOwned_generalTracks__RECO. 407639 18448.4', 'recoPreshowerClusterShapes_multi5x5PreshowerClusterShape_multi5x5PreshowerXClustersShape_RECO. 289.787 41.3311', 'recoPreshowerClusterShapes_multi5x5PreshowerClusterShape_multi5x5PreshowerYClustersShape_RECO. 289.767 47.2686', 'recoCaloClustersToOnerecoClusterShapesAssociation_hybridSuperClusters_hybridShapeAssoc_RECO. 272.111 65.4852'))
	[{'module_name': 'generalTracks', 'module_label': '', 'size_compressed': '18448.4', 'cpp_type': 'TrackingRecHitsOwned', 'size_uncompressed': '407639'}, {'module_name': 'multi5x5PreshowerClusterShape', 'module_label': 'multi5x5PreshowerXClustersShape', 'size_compressed': '41.3311', 'cpp_type': 'recoPreshowerClusterShapes', 'size_uncompressed': '289.787'}, {'module_name': 'multi5x5PreshowerClusterShape', 'module_label': 'multi5x5PreshowerYClustersShape', 'size_compressed': '47.2686', 'cpp_type': 'recoPreshowerClusterShapes', 'size_uncompressed': '289.767'}, {'module_name': 'hybridSuperClusters', 'module_label': 'hybridShapeAssoc', 'size_compressed': '65.4852', 'cpp_type': 'recoCaloClustersToOnerecoClusterShapesAssociation', 'size_uncompressed': '272.111'}]

	"""
	#reg returns (cpp_type, mod_name, mod_label, proc_name, size_uncomp, size_comp)

	#TODO: I could change this into shorter ---...

	return [ {"cpp_type": cpp_type, "module_name": mod_name, "module_label": mod_label,
			"size_uncompressed": size_uncomp, "size_compressed": size_comp} # we filter out the proc_name, AND CONVERT TO DICTIONARY
		for (cpp_type, mod_name, mod_label, proc_name, size_uncomp, size_comp) in [
			reg.groups() for reg in [
				edmSize_line_parsing_reg.search(line) for line in lines] 
			if reg ] # we filter out not matched lines
	]

""" Get EdmSize file size for the candle, step in current dir """
def getEdmReport(path, candle, step):
	files = os.listdir(path)
	edm_files = [os.path.join(path, f) for f in files
				 if test_edm_file.search(f) 
					and os.path.isfile(os.path.join(path, f)) ]

	""" get the size of file if it is the root file for current candle and step """
	# TODO: a function candle, step --> file name

	try:
		edm_fn = [f for f in edm_files
			 if f_candle_and_step_inJobID(candle, step, getJobID_fromEdmSizeFileName(f))][0] #that's in the same dir so candle and step is more than enough
	except IndexError, e: #this would happen if there's no Edmsize report existing !!!
		return False

	# open the file and read into lines
	edm_file = open(edm_fn)
	lines = edm_file.readlines()
	edm_file.close()

	#return the parsed data
	products = parseEdmSize(lines)
	
	return products

