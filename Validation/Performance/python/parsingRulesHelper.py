import re

""" The module includes rule based regexp parser to automatize the parsing of information from simple text based files """


""" a function used to compile the regexps, to be called with map """
rulesRegexpCompileFunction = lambda x: ( len(x)==2 and (x[0], re.compile(x[1])) or (x[0], re.compile(x[1]), x[2]) )

def rulesParser(parsing_rules, lines, compileRules = True):
		""" 
			Applies the (provided) regular expression rules (=rule[1] for rule in parsing_rules)
			to each line and if it matches the line,
			puts the mached information to the dictionary as the specified keys (=rule[0]) which is later returned
			Rule[3] contains whether the field is required to be found. If so and it isn't found the exception would be raised.
			rules = [
			  ( (field_name_1_to_match, field_name_2), regular expression, /optionaly: is the field required? if so "req"/ )
			]
		 """
		info = {}
		#we compile the parsing rules
		if compileRules:
			parsing_rules = map(rulesRegexpCompileFunction, parsing_rules)
		""" we dynamicaly check if line passes any of the rules and in this way put the information to the info dict. """
		for line in lines:
			for rule in parsing_rules:
				if rule[1].match(line):
					g = rule[1].match(line).groups()
					#print g
					#print "rule fields:"  + str(rule[0])
					i = 0
					for field_name in rule[0]:
						"we use empty field name to mark unneeded parts of regular expression"
						if field_name != "":
							#print str(i) + ":" + field_name
							# we do want to store None values as empty strings ""
							#TODO: we might want to change it if we multiple introduced rules having same result targets
							if g[i] == None:
								info[field_name] = ""
							else:
								info[field_name] = g[i]
						i += 1
		#For the values which do not exist we put "" and check for REQUIRED values
		missing_fields = []
		for rule in parsing_rules:
			for field_name in rule[0]:
				if field_name:
					if not info.has_key(field_name):
						info[field_name] = ""
					""" check for required fields"""
					if len(rule) == 3 and rule[2] =="req":
						if not info[field_name]:
							missing_fields.append(field_name)
		return (info, missing_fields)
