import json


def sanitize_record(record, use_case):  # @smarchesin TODO: define sanitize use-case functions that read replacements from file
	"""
	Sanitize record replacing translation errors

	Params:
		record (str): target record

	Returns: the sanitized record
	"""
	if record:
		if use_case == 'colon':
			record = record.replace('octopus', 'polyp')
			record = record.replace('hairy', 'villous')
			record = record.replace('villous adenoma-tubule', 'tubulo-villous adenoma')
			record = record.replace('villous adenomas-tubule', 'tubulo-villous adenoma')
			record = record.replace('villous adenomas tubule', 'tubulo-villous adenoma')
			record = record.replace('tubule adenoma-villous', 'tubulo-villous adenoma')
			record = record.replace('tubular adenoma-villous', 'tubulo-villous adenoma')
			record = record.replace('villous adenoma tubule-', 'tubulo-villous adenoma ')
			record = record.replace('villous adenoma tubule', 'tubulo-villous adenoma')
			record = record.replace('tubulovilloso adenoma', 'tubulo-villous adenoma')
			record = record.replace('blind', 'caecum')
			record = record.replace('cecal', 'caecum')
			record = record.replace('rectal', 'rectum')
			record = record.replace('sigma', 'sigmoid')
			record = record.replace('proximal colon', 'right colon')
		if use_case == 'cervix':
			record = record.replace('octopus', 'polyp')
			record = record.replace('his cassock', 'lamina propria')
			record = record.replace('tunica propria', 'lamina propria')
			record = record.replace('l-sil', 'lsil')
			record = record.replace('h-sil', 'hsil')
			record = record.replace('cin ii / iii', 'cin23')
			record = record.replace('cin iii', 'cin3')
			record = record.replace('cin ii', 'cin2')
			record = record.replace('cin i', 'cin1')
			record = record.replace('cin-iii', 'cin3')
			record = record.replace('cin-ii', 'cin2')
			record = record.replace('cin-i', 'cin1')
			record = record.replace('cin1-2', 'cin1 cin2')
			record = record.replace('cin2-3', 'cin2 cin3')
			record = record.replace('cin-1', 'cin1')
			record = record.replace('cin-2', 'cin2')
			record = record.replace('cin-3', 'cin3')
			record = record.replace('cin 2 / 3', 'cin23')
			record = record.replace('cin 2/3', 'cin23')
			record = record.replace('cin 1-2', 'cin1 cin2')
			record = record.replace('cin 2-3', 'cin2 cin3')
			record = record.replace('cin 1', 'cin1')
			record = record.replace('cin 2', 'cin2')
			record = record.replace('cin 3', 'cin3')
			record = record.replace('ii-iii cin', 'cin2 cin3')
			record = record.replace('i-ii cin', 'cin1 cin2')
			record = record.replace('iii cin', 'cin3')
			record = record.replace('ii cin', 'cin2')
			record = record.replace('i cin', 'cin1')
	return record


def sanitize_code(code):
	"""
	Sanitize code removing unnecessary characters

	Params:
		code (str): target code

	Returns: the sanitized code
	"""

	if code:
		code = code.replace('-', '')
		code = code.ljust(7, '0')
	return code


def sanitize_codes(codes):
	"""
	Sanitize codes by splitting and removing unnecessarsy characters

	Params:
		codes (list): target codes

	Returns: the sanitized codes
	"""

	codes = codes.split(';')
	codes = [sanitize_code(code) for code in codes]
	return codes


def read_rules(rules):
	"""
	Read rules stored within file

	Params: 
		rules (str): path to rules file

	Returns: a dict of trigger: [candidates] representing rules for each use-case
	"""

	with open(rules, 'r') as file:
		lines = file.readlines()

	rules = {'colon': {}, 'cervix': {}, 'celiac': {}, 'lung': {}}
	for line in lines:
		trigger, candidates, position, mode, use_cases = line.strip().split('\t')
		use_cases = use_cases.split(',')
		for use_case in use_cases:
			rules[use_case][trigger] = (candidates.split(','), position, mode)
	return rules


def read_dysplasia_mappings(mappings):
	"""
	Read dysplasia mappings stored within file

	Params:
		mappings (str): path to dysplasia mappings file

	Returns: a dict of {trigger: grade} representing mappings for each use-case
	"""

	with open(mappings, 'r') as file:
		lines = file.readlines()

	mappings = {'colon': {}, 'cervix': {}, 'celiac': {}, 'lung': {}}
	for line in lines:
		trigger, grade, use_cases = line.strip().split('\t')
		use_cases = use_cases.split(',')
		for use_case in use_cases:
			mappings[use_case][trigger] = grade.split(',')
	return mappings


def read_cin_mappings(mappings):
	"""
	Read cin mappings stored within file

	Params:
		mappings (str): path to cin mappings file

	Returns: a dict of {trigger: grade} representing mappings for cervical intraephitelial neoplasia
	"""

	with open(mappings, 'r') as file:
		lines = file.readlines()

	mappings = {}
	for line in lines:
		trigger, grade = line.strip().split('\t')
		mappings[trigger] = grade
	return mappings


def read_hierarchies(hrels):
	"""
	Read hierarchy relations stored within file
	
	Params:
		hrels (str): hierarchy relations file path
		
	Returns: the list of hierarchical relations
	"""
	
	with open(hrels, 'r') as f:
		rels = f.readlines()
	return [rel.strip() for rel in rels]


def store_concepts(concepts, out_path, indent=4, sort_keys=True):
	"""
	Store report concepts 

	Params:
		concepts (dict): report concepts
		out_path (str): output file-path w/o extension
		indent (int): indentation level
		sort_keys (bool): sort keys

	Returns: True
	"""

	with open(out_path + '.json', 'w') as out:
		json.dump(concepts, out, indent=indent, sort_keys=sort_keys)
	return True


def load_concepts(concept_fpath):
	"""
	Load stored concepts

	Params:
		concept_fpath (str): file-path to stored concepts

	Returns: the dict containing the report (stored) concepts
	"""

	with open(concept_fpath, 'r') as f:
		concepts = json.load(f)
	return concepts


def store_labels(labels, out_path, indent=4, sort_keys=True):
	"""
	Store report labels 

	Params:
		labels (dict): report labels
		out_path (str): output file-path w/o extension
		indent (int): indentation level
		sort_keys (bool): sort keys

	Returns: True
	"""

	with open(out_path + '.json', 'w') as out:
		json.dump(labels, out, indent=indent, sort_keys=sort_keys)
	return True


def load_labels(label_fpath):
	"""
	Load stored labels

	Params:
		label_fpath (str): file-path to stored labels

	Returns: the dict containing the report (stored) labels
	"""

	with open(label_fpath, 'r') as f:
		labels = json.load(f)
	return labels


##### AOEC RELATED FUNCTIONS #####

def aoec_colon_concepts2labels(report_concepts):
	"""
	Convert the concepts extracted from colon reports to the set of pre-defined labels used for classification
	
	Params:
		report_concepts (dict(list)): the dict containing for each colon report the extracted concepts
		
	Returns: a dict containing for each colon report the set of pre-defined labels where 0 = abscence and 1 = presence
	"""
	
	report_labels = dict()
	# loop over reports
	for rid, rconcepts in report_concepts.items():
		# assign pre-defined set of labels to current report
		report_labels[rid] = {'cancer': 0, 'hgd': 0, 'lgd': 0,'hyperplastic': 0, 'ni': 0}
		# textify diagnosis section
		diagnosis = ' '.join([concept[1].lower() for concept in rconcepts['Diagnosis']])
		# update pre-defined labels w/ 1 in case of label presence
		if ('colon adenocarcinoma') in diagnosis:  # update 'cancer'
			report_labels[rid]['cancer'] = 1
		if ('dysplasia') in diagnosis:  # diagnosis contains dysplasia
			if ('mild') in diagnosis:  # update lgd
				report_labels[rid]['lgd'] = 1
			if ('moderate') in diagnosis:  # update lgd
				report_labels[rid]['lgd'] = 1
			if ('severe') in diagnosis:  # update hgd
				report_labels[rid]['hgd'] = 1
		if ('hyperplastic polyp') in diagnosis:  # update hyperplastic
			report_labels[rid]['hyperplastic'] = 1
		if sum(report_labels[rid].values()) == 0:  # update ni
			report_labels[rid]['ni'] = 1   
	return report_labels


def aoec_colon_labels2binary(report_labels):
	"""
	Convert the pre-defined labels extracted from colon reports to binary labels used for classification
	
	Params:
		report_labels (dict(list)): the dict containing for each colon report the pre-defined labels
		
	Returns: a dict containing for each colon report the set of binary labels where 0 = abscence and 1 = presence
	"""
	
	binary_labels = dict()
	# loop over reports
	for rid, rlabels in report_labels.items():
		# assign binary labels to current report
		binary_labels[rid] = {'cancer_or_dysplasia': 0, 'other': 0}
		# update binary labels w/ 1 in case of label presence
		if rlabels['cancer'] == 1 or rlabels['lgd'] == 1 or rlabels['hgd'] == 1:  # update 'cancer_or_dysplasia' label
			binary_labels[rid]['cancer_or_dysplasia'] = 1
		else:  # update 'other' label
			binary_labels[rid]['other'] = 1  
	return binary_labels


##### RADBOUD RELATED FUNCTIONS #####

def radboud_colon_concepts2labels(report_concepts):
	"""
	Convert the concepts extracted from reports to the set of pre-defined labels used for classification
	
	Params:
		report_concepts (dict(list)): the dict containing for each report the extracted concepts
		
	Returns: a dict containing for each report the set of pre-defined labels where 0 = abscence and 1 = presence
	"""
	
	report_labels = dict()
	# loop over reports
	for rid, rconcepts in report_concepts.items():
		report_labels[rid] = dict()
		# assign pre-defined set of labels to current report
		report_labels[rid]['labels'] = {'cancer': 0, 'hgd': 0, 'lgd': 0,'hyperplastic': 0, 'ni': 0}
		# textify diagnosis section
		diagnosis = ' '.join([concept[1].lower() for concept in rconcepts['concepts']['Diagnosis']])
		# update pre-defined labels w/ 1 in case of label presence
		if ('colon adenocarcinoma') in diagnosis:  # update 'cancer'
			report_labels[rid]['labels']['cancer'] = 1
		if ('dysplasia') in diagnosis:  # diagnosis contains dysplasia
			if ('mild') in diagnosis:  # update lgd
				report_labels[rid]['labels']['lgd'] = 1
			if ('moderate') in diagnosis:  # update lgd
				report_labels[rid]['labels']['lgd'] = 1
			if ('severe') in diagnosis:  # update hgd
				report_labels[rid]['labels']['hgd'] = 1
		if ('hyperplastic polyp') in diagnosis:  # update hyperplastic
			report_labels[rid]['labels']['hyperplastic'] = 1
		if sum(report_labels[rid]['labels'].values()) == 0:  # update ni
			report_labels[rid]['labels']['ni'] = 1   
		report_labels[rid]['slide_ids'] = rconcepts['slide_ids']
	return report_labels


def radboud_colon_labels2binary(report_labels):
	"""
	Convert the pre-defined labels extracted from reports to binary labels used for classification
	
	Params:
		report_labels (dict(list)): the dict containing for each report the pre-defined labels
		
	Returns: a dict containing for each report the set of binary labels where 0 = abscence and 1 = presence
	"""
	
	binary_labels = dict()
	# loop over reports
	for rid, rlabels in report_labels.items():
		binary_labels[rid] = dict()
		# assign binary labels to current report
		binary_labels[rid]['labels'] = {'cancer_or_dysplasia': 0, 'other': 0}
		# update binary labels w/ 1 in case of label presence
		if rlabels['labels']['cancer'] == 1 or rlabels['labels']['lgd'] == 1 or rlabels['labels']['hgd'] == 1:  # update 'cancer_or_adenoma' label
			binary_labels[rid]['labels']['cancer_or_dysplasia'] = 1
		else:  # update 'other' label
			binary_labels[rid]['labels']['other'] = 1  
		binary_labels[rid]['slide_ids'] = rlabels['slide_ids']
	return binary_labels