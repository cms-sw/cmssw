#!/usr/bin/env python
import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.usedOutput import *

import re


process = cms.Process("SIZE");

# ==========  DEFINITION OF EVENT CONTENTS =======

# !!! to add more event contents load the file  !!
import Configuration.EventContent.EventContent_cff as eventContent
import Configuration.EventContent.EventContentCosmics_cff as eventContent_cosmics

# !!! and add the event contents to the list here !!
EventContents_def = {
	"RECO": 		eventContent.RECOEventContent, 
	"AOD": 			eventContent.AODEventContent,
	"RECO_COSMICS": 	eventContent_cosmics.RECOEventContent, 
	"AOD_COSMICS": 		eventContent_cosmics.AODEventContent,
	#new output commands
	"RECOSIM": 		eventContent.RECOSIMEventContent,
	"AODSIM": 		eventContent.AODSIMEventContent,
	"RAW": 			eventContent.RAWEventContent,
	"RAWSIM": 		eventContent.RAWSIMEventContent,
	"FEVTDEBUG": 		eventContent.FEVTDEBUGEventContent,
	"FEVTDEBUGHLT": 	eventContent.FEVTDEBUGHLTEventContent
}



"""
process = cms.Process("SIZE");
process.load("Configuration.EventContent.EventContent_cff")
## define event conent to be used
if options.event_content == "RECO" :
    local_outputCommands = process.RECOEventContent.outputCommands;  +
elif options.event_content == "RECOSIM" :
    local_outputCommands = process.RECOSIMEventContent.outputCommands; +
elif options.event_content == "AOD" :
    local_outputCommands = process.AODEventContent.outputCommands; +
elif options.event_content == "AODSIM" :
    local_outputCommands = process.AODSIMEventContent.outputCommands; +
elif options.event_content == "RAW" :
    local_outputCommands = process.RAWEventContent.outputCommands; +
elif options.event_content == "RAWSIM" :
    local_outputCommands = process.RAWSIMEventContent.outputCommands;
elif options.event_content == "FEVTDEBUG" :
    local_outputCommands = process.FEVTDEBUGEventContent.outputCommands;
elif options.event_content == "FEVTDEBUGHLT" :
    local_outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands;
elif options.event_content == "ALL" :
    local_outputCommands = cms.untracked.vstring('keep *_*_*_*');
    xchecksize = True;


"""
# ==========  END OF DEFINITION OF EVENT CONTENTS =======

def rule_passes(rule_regexp, product_repr):
	""" 
		rule: rule(keep, drop)_A_B_C_D
		product: A_B_C
	"""	
	#we have altered the rules to have "*" as the 4th parameter (if split by _) 
	# so all the D's will pass
	return rule_regexp.match(product_repr+"_")

def rule_to_regexp(rule):
	""" returns a tuple (rule_name, rule_regexp) e.g. ("keep", <regexp for matching product names>  """
	#this part might be moved out and regular expression cached
	(rule_name, rule_test) = rule.split()
	
	# we create a regexp out of rule
	
	#we ignore the 4th rule:
	rule_parts =rule_test.split("_")
	if len(rule_parts) == 4 :
		# we replace the last one to asterix
		rule_parts[3] = "*" 
	rule_test = "_".join(rule_parts) 
	
	# make a regexp
	rule_test = rule_test.replace("*", ".*")

	rule_regexp = re.compile("^"+rule_test+"$")
	
	return (rule_name, rule_regexp)

def product_in_EventContent(rules, product):
	"""
	products are in format {"cpp_type": cpp_type, "module_name": mod_name, "module_label": mod_label,
		"size_uncompressed": size_uncomp, "size_compressed": size_comp}  
	--- Some simple doctests ---

	>>> product_in_EventContent(rules = rule_to_regexp(['drop *', 'keep *_logErrorHarvester_*_*', 'keep *_hybridSuperClusters_*_*', 'keep recoSuperClusters_correctedHybridSuperClusters_*_*']), product = {'module_name': 'hybridSuperClusters', 'module_label': 'hybridShapeAssoc', 'size_compressed': '65.4852', 'cpp_type': 'recoCaloClustersToOnerecoClusterShapesAssociation', 'size_uncompressed': '272.111'})
	True
	
	>>> product_in_EventContent(rules = rule_to_regexp(['drop *', 'keep *_logErrorHarvester_*_*', 'keep DetIdedmEDCollection_siStripDigis_*_*', 'keep *_siPixelClusters_*_*', 'keep *_siStripClusters_*_*']), product = {'module_name': 'hybridSuperClusters', 'module_label': 'hybridShapeAssoc', 'size_compressed': '65.4852', 'cpp_type': 'recoCaloClustersToOnerecoClusterShapesAssociation', 'size_uncompressed': '272.111'})
	False
	
	"""


	product_repr = "%(cpp_type)s_%(module_name)s_%(module_label)s" % product
	result = "keep"
	
	""" rule is in format: 
		> keep  *_nuclearInteractionMaker_*_*
		> keep *_horeco_*_*
		> drop *
	"""
	for (rule_name, rule_regexp) in rules:

		if rule_passes(rule_regexp, product_repr):
			result = rule_name
	return result == "keep"


#initialize
EventContents = {}
for (ec_name, obj) in EventContents_def.items():
	# I'm not sure if this is needed, but I feel more comfident dealing with Python objects that CMSSWs PSet
	rules_txt = [a for a in obj.outputCommands]

	# we create a list of precompiled regexps for our rules
	rules_regexp = map(rule_to_regexp, rules_txt)

	EventContents[ec_name] = {"text_rules": rules_txt, "rules_regexp": rules_regexp}

#this will be used by importing script
def List_ECs_forProduct(product):
	""" returns a list of EC titles the product belongs to """
	EC_list = []
	for (ec_name, ec) in EventContents.items():
		if product_in_EventContent(ec["rules_regexp"], product):
			EC_list.append(ec_name)
	return EC_list

def getTxtEventContentRules():
	#TODO: We should where to assign products to Event-Content, on harvesting or on importing part
	""" returns a dictionary of lists with rules """
	txt_rules = {}
	for (ec_name, ec) in EventContents.items():
		txt_rules[ec_name] = ec["text_rules"]
	return txt_rules
	
#a test
if __name__ == "__main__":
	print "==== The  event contents data is: === "
	print EventContents
	prod = {'module_name': 'hybridSuperClusters', 'module_label': 'hybridShapeAssoc', 'size_compressed': '65.4852', 'cpp_type': 'recoCaloClustersToOnerecoClusterShapesAssociation', 'size_uncompressed': '272.111'}
	print List_ECs_forProduct(prod)
	
