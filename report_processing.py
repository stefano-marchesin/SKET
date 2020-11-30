import math
import string
import re
import roman
import pandas as pd

import utils

from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from googletrans import Translator


class ReportProc(object):

	def __init__(self):
		"""
		Set translator and build regular expression to split text based on bullets

		Params:
			None

		Returns: None
		"""

		self.translator = Translator()
		# build regex for bullet patterns
		self.bullets_regex = re.compile('((?<=(^i-ii(\s|:|\.)))|(?<=(^i-iii(\s|:|\.)))|(?<=(^ii-iii(\s|:|\.)))|(?<=(^i-iv(\s|:|\.)))|(?<=(^ii-iv(\s|:|\.)))|(?<=(^iii-iv(\s|:|\.)))|(?<=(^i and ii(\s|:|\.)))|(?<=(^i and iii(\s|:|\.)))|(?<=(^ii and iii(\s|:|\.)))|(?<=(^i and iv(\s|:|\.)))|(?<=(^ii and iv(\s|:|\.)))|(?<=(^iii and iv(\s|:|\.)))|(?<=(^i(\s|:|\.)))|(?<=(^ii(\s|:|\.)))|(?<=(^iii(\s|:|\.)))|(?<=(^iv(\s|:|\.)))|(?<=(\si-ii(\s|:|\.)))|(?<=(\si-iii(\s|:|\.)))|(?<=(\sii-iii(\s|:|\.)))|(?<=(\si-iv(\s|:|\.)))|(?<=(\sii-iv(\s|:|\.)))|(?<=(\siii-iv(\s|:|\.)))|(?<=(\si and ii(\s|:|\.)))|(?<=(\si and iii(\s|:|\.)))|(?<=(\sii and iii(\s|:|\.)))|(?<=(\si and iv(\s|:|\.)))|(?<=(\sii and iv(\s|:|\.)))|(?<=(\siii and iv(\s|:|\.)))|(?<=(\si(\s|:|\.)))|(?<=(\sii(\s|:|\.)))|(?<=(\siii(\s|:|\.)))|(?<=(\siv(\s|:|\.))))(.*?)((?=(\si+(\s|:|\.|-)))|(?=(\siv(\s|:|\.|-)))|(?=($)))')
		self.ranges_regex = re.compile('\d-\d')

	### COMMON FUNCTIONS ###


	def load_dataset(self, reports_path, sheet, header): 
		"""
		Load reports dataset

		Params:
			reports_path (str): reports.xlsx fpath
			sheet (str): name of the excel sheet to use 
			header (int): row index used as header
		
		Returns: the loaded datasrt
		"""

		dataset = pd.read_excel(io=reports_path, sheet_name=sheet, header=header)
		return dataset

	def translate_text(self, text, src_lang, dest_lang='en'):
		"""
		Translate a given text from source language to destination language

		Params:
			report (str): target text 
			src_lang (str): source language 
			dest_lang (str): destination language

		Returns: the translated text (lowercase)
		"""

		if type(text) == str:
			trans_text = self.translator.translate(text=text.lower(), src=src_lang, dest=dest_lang).text.lower()
		else:
			trans_text = ''
		return trans_text


	### AOEC SPECIFIC FUNCTIONS ###


	def translate_aoec_reports(self, dataset):
		"""
		Read AOEC reports and extract the required fields 

		Params:
			dataset (pandas DataFrame): target dataset

		Returns: a dictionary containing the required reports fields
		"""

		reports = dict()
		for report in tqdm(dataset.itertuples()):
			reports[report._1] = {'diagnosis_nlp': self.translate_text(report.Diagnosi, src_lang='it', dest_lang='en'), 
								  'materials': self.translate_text(report.Materiali, src_lang='it', dest_lang='en'), 
								  'procedure': report.Procedura if type(report.Procedura) == str else '', 
								  'topography': report.Topografia if type(report.Topografia) == str else '', 
								  'diagnosis_struct': report._5 if type(report._5) == str else '', 
								  'age': int(report.Età) if not math.isnan(report.Età) else 0, 
								  'gender': report.Sesso if type(report.Sesso) == str else ''}
		return reports

	def translate_aoec_reports_v2(self, dataset):
		"""
		Read AOEC reports and extract the required fields for the new data provided

		Params:
			dataset (pandas DataFrame): target dataset

		Returns: a dictionary containing the required reports fields
		"""

		reports = dict()
		for report in tqdm(dataset.itertuples()):
			rid = report.FILENAME + '_' + report.CODICEINT + '_' + str(report.IDINTERNO)
			reports[rid] = {'diagnoses': translate_text(report.TESTODIAGNOSI, src_lang='it', dest_lang='en'), 
							'materials': translate_text(report.MATERIALE, src_lang='it', dest_lang='en'), 
							'procedure': report.SNOMEDPROCEDURA if type(report.SNOMEDPROCEDURA) == str else '', 
							'topography': report.SNOMEDTOPOGRAFIA if type(report.SNOMEDTOPOGRAFIA) == str else '', 
							'diagnosis_struct': report.SNOMEDDIAGNOSI if type(report.SNOMEDDIAGNOSI) == str else '', 
							'birth_date': report.NATOIL if report.NATOIL else '', 
							'visit_date': report.DATAORAFINEVALIDAZIONE if report.DATAORAFINEVALIDAZIONE else '',
							'gender': report.SESSO if type(report.SESSO) == str else '', 
							'image': report.FILENAME, 
							'codeint': report.CODICEINT, 
							'internalid': report.IDINTERNO}
		return reports


	def split_aoec_diagnosis(self, diagnoses, internalid):
		"""
		Split the section 'diagnoses' within AOEC reports relying on bullets (i.e. '1', '2', etc.)

		Params:
			diagnoses (str): the 'diagnoses' section of AOEC reports
			internalid (int): the internal id specifying the current diagnosis

		Returns: the part of the 'diagnoses' section related to the current internalid
		"""
		
		diagnosis = ''
		# split diagnosis on new lines
		dlist = diagnoses.split('\n')
		if internalid <= len(dlist):  # the considered diagnosis can be found within diagnoses
			# associate the requested diagnosis
			diagnosis = dlist[internalid-1]
		else:  # the considered diagnosis cannot be found within diagnoses (i.e., diagnosis merged with others or not present)
			for d in dlist:
				# identify wether current diagnosis presents a range of bullets (i.e., 1-5, etc.)
				ranges = self.ranges_regex.findall(d)
				if ranges:  # range found
					# sanity check on number of ranges identified - there should be only one range otherwise stop process and verify
					assert len(ranges) == 1
					# get min/max range values
					edges = [int(ranges[0].split('-')[0]), int(ranges[0].split('-')[1])]
					if edges[0] <= internalid <= edges[1]:  # diagnosis found within previous diagnoses section
						diagnosis = d
						break
				if str(internalid) in d:  # diagnosis found within previous diagnoses section
					diagnosis = d
					break
				
			if not diagnosis:  # diagnosis not found within previous diagnoses section - keep the entire 'diagnoses' section @smarchesin TODO: remove prints after testing
				diagnosis = diagnoses
				print('no diagnosis found within previous sections')
				print(internalid)
				print(diagnoses)
				print()
		return diagnosis

	def process_aoec_reports(self, reports):
		"""
		Process AOEC report sections and prepare them for linking
		
		Params:
			reports (dict(dict(str))): the dict of AOEC reports containing diagnoses 
			
		Returns: a dict containing for each report the diagnosis associated to the given internal id
		"""

		for rid, report in tqdm(reports.items()):
			reports[rid]['diagnosis_nlp'] = self.split_aoec_diagnosis(report['diagnoses'], report['internalid']) 
		return reports

	### RADBOUD SPECIFIC FUNCTIONS


	def translate_radboud_reports(self, dataset):
		"""
		Read Radboud reports and extract the required fields 

		Params:
			dataset (pandas DataFrame): target dataset

		Returns: a dictionary containing the required reports fields
		"""

		reports = dict()
		for report in tqdm(dataset.itertuples()):
			reports[report.Studynumber] = {'conclusions': self.translate_text(report._4, src_lang='nl', dest_lang='en'), 
										   'diagnosis_1': self.translate_text(report._5, src_lang='nl', dest_lang='en'),
										   'diagnosis_2': self.translate_text(report._6, src_lang='nl', dest_lang='en'),
										   'diagnosis_3': self.translate_text(report._7, src_lang='nl', dest_lang='en')}
		return reports

	def translate_radboud_reports_v2(self, dataset):
		"""
		Read Radboud reports and extract the required fields 

		Params:
			dataset (pandas DataFrame): target dataset

		Returns: a dictionary containing the required reports fields
		"""

		reports = dict()
		for report in tqdm(dataset.itertuples()):
			reports[report._5] = {'conclusions': self.translate_text(report._1, src_lang='nl', dest_lang='en'), 
										   'diagnosis_1': self.translate_text(report._2, src_lang='nl', dest_lang='en'),
										   'diagnosis_2': self.translate_text(report._3, src_lang='nl', dest_lang='en'),
										   'diagnosis_3': self.translate_text(report._4, src_lang='nl', dest_lang='en'),
										   'snomed_1': report._6 if type(report._6) == str else '',
										   'snomed_2': report._7 if type(report._7) == str else '',
										   'snomed_3': report._8 if type(report._8) == str else '',
										  }
		return reports

	def split_radboud_conclusions(self, conclusions):
		"""
		Split the section 'conclusions' within reports relying on bullets (i.e. 'i', 'ii', etc.)
		
		Params:
			conclusions (str): the 'conclusions' section of radboud reports
			
		Returns: a dict containing the 'conclusions' section divided as a bullet list 
		"""
		
		sections = defaultdict(str)
		# use regex to identify bullet-divided sections within 'conclusions'
		for groups in self.bullets_regex.findall(conclusions):
			# identify the target bullet for the given section
			bullet = [group for group in groups[:65] if group and any(char.isalpha() or char.isdigit() for char in group)][0].strip()
			if 'and' in bullet:  # composite bullet
				bullets = bullet.split(' and ')
			elif '-' in bullet:  # composite bullet
				bullets = bullet.split('-')
			else:  # single bullet 
				bullets = [bullet]
			# loop over bullets and concatenate corresponding sections
			for bullet in bullets:
				if groups[65] != 'and':  # the section is not a conjunction between two bullets (e.g., 'i and ii')
					sections[bullet.translate(str.maketrans('', '', string.punctuation)).upper()] += ' ' + groups[65]  # store them using uppercased roman numbers as keys - required to make Python 'roman' library working
		if bool(sections):  # 'sections' contains split sections
			return sections
		else:  # 'sections' is empty - assign the whole 'conclusions' to 'sections'
			sections['whole'] = conclusions
			return sections

	def process_radboud_reports(self, reports):  # @smarchesin TODO: once testing is finished, remove unsplitted_reports and misplitted_reports - they refer to erroneous or overly complicated cases provided by Radboud
		"""
		Process Radboud report sections and prepare them for linking

		Params: 
			reports (dict(dict(str))): the dict of radboud reports containing conclusions/diagnoses

		Returns: a dict containing for each report and each block the corresponding conclusion
		"""

		proc_reports = dict()
		unsplitted_reports = dict()
		misplitted_reports = dict()
		for rid, rdata in tqdm(reports.items()):
			if rdata['conclusions']:  # split conclusions and associate to each block the corresponding conclusion
				# deepcopy rdata to avoid removing elements from input reports
				sections = deepcopy(rdata)
				# split conclusions into sections
				conclusions = self.split_radboud_conclusions(sections.pop('conclusions'))
				pid = '_'.join(rid.split('_')[:-1])  # remove block and slide ids from report id - keep patient id
				related_ids = [rel_id for rel_id in reports.keys() if pid in rel_id]  # get all the ids related to the current patient
				# get block ids from related_ids
				block_ids = set([rel_id[:-2] if 'V' not in rel_id.split('_')[-1] else '_'.join(rel_id.split('_')[:-1])[:-2] for rel_id in related_ids])
				if 'whole' in conclusions:  # unable to split conclusions - either single conclusion or not appropriately specified
					if len(block_ids) == 1:  # single conclusion
						# create dict to store block diagnosis and slide ids
						proc_reports[max(block_ids)] = dict()
						# store conclusion - i.e., the final diagnosis
						proc_reports[max(block_ids)]['diagnosis'] = conclusions['whole']
						# store slide ids associated to the current block diagnosis
						proc_reports[max(block_ids)]['slide_ids'] = [sid[-2:] if ((max(block_ids) in sid) and ('V' not in sid.split('_')[-1])) else '_'.join(sid.split('_')[-2:])[2:] for sid in reports.keys() 
																			if (((max(block_ids) in sid) and ('V' not in sid.split('_')[-1])) or ((max(block_ids) in sid) and ('V' in sid.split('_')[-1])))]
					else:  # not appropriately specified
						unsplitted_reports[rid] = rdata['conclusions']
				else:
					block_ix2id = {int(block_id[-1]): block_id for block_id in block_ids}
					if len(conclusions) < len(block_ids):  # fewer conclusions have been identified than the actual number of blocks - store and fix later
						misplitted_reports[rid] = rdata['conclusions']
					else:  # associate the given conclusions to the corresponding blocks
						# loop over conclusions and fill proc_reports
						for cid, cdata in conclusions.items():
							block_ix = roman.fromRoman(cid)  # convert conclusion id (roman number) into corresponding arabic number (i.e., block index)
							if block_ix in block_ix2id:  # block with bloc_ix present within dataset
								# create dict to store block diagnosis and slide ids
								proc_reports[block_ix2id[block_ix]] = dict()
								# store conclusion - i.e., the final diagnosis
								proc_reports[block_ix2id[block_ix]]['diagnosis'] = cdata
								# store slide ids associated to the current block diagnosis
								proc_reports[block_ix2id[block_ix]]['slide_ids'] = [sid[-2:] if ((block_ix2id[block_ix] in sid) and ('V' not in sid.split('_')[-1])) else '_'.join(sid.split('_')[-2:])[2:] for sid in reports.keys() 
																					if (((block_ix2id[block_ix] in sid) and ('V' not in sid.split('_')[-1])) or ((block_ix2id[block_ix] in sid) and ('V' in sid.split('_')[-1])))]
		return proc_reports, unsplitted_reports, misplitted_reports


	############ DEPRECATED ############

	# The part below refers to a previous understanding of radboud reports which is not the correct one -- therefore, the proposed functions have been deprecated and should not be used to process reports

	'''

	def process_radboud_reports(self, reports):  # @smarchesin TODO: modify to integrate structured part too (i.e. SNOMED codes)
		"""
		Process Radboud report sections and prepare them for linking
		
		Params:
			reports (dict(dict(str))): the dict of radboud reports containing conclusions/diagnoses 
			
		Returns: a dict containing for each report the list of sections associated to each diagnosis
		"""

		proc_reports = dict()
		for rid, rdata in reports.items():
			# deepcopy rdata to avoid removing elements from input reports
			sections = deepcopy(rdata)
			proc_reports[rid] = dict()
			# split conclusions into sections
			conclusions = self.split_radboud_conclusions(sections.pop('conclusions'))
			# pre process each diagnosis and concatenate its corresponding 'conclusions' section
			if sections['diagnosis_1']:  # diagnosis 1 present
				# split diagnosis into separate sections by * delimiter
				diagnosis = sections['diagnosis_1'].split('*')
				# remove unnecessary empty elements
				diagnosis = [field.strip() for field in diagnosis if field.strip()]  
				# concatenate the corresponding 'conclusions' section to diagnosis
				if 'whole' in conclusions:  # 'conclusions' cannot be divided in multiple sections
					diagnosis_and_conclusions = [diagnosis, conclusions['whole']]
				elif 'i' in conclusions:  # found section 'i' within conclusions
					diagnosis_and_conclusions = [diagnosis, conclusions['i']]
				# assign current diagnosis to proc_reports
				proc_reports[rid]['diagnosis_1'] = diagnosis_and_conclusions
			if sections['diagnosis_2']:  # diagnosis 2 present
				# split diagnosis into separate sections by * delimiter
				diagnosis = sections['diagnosis_2'].split('*')
				# remove unnecessary empty elements
				diagnosis = [field.strip() for field in diagnosis if field.strip()]  
				# concatenate the corresponding 'conclusions' section to diagnosis
				if 'whole' in conclusions:  # 'conclusions' cannot be divided in multiple sections
					diagnosis_and_conclusions = [diagnosis, conclusions['whole']]
				elif 'ii' in conclusions:  # found section 'ii' within conclusions
					diagnosis_and_conclusions = [diagnosis, conclusions['ii']]
				# assign current diagnosis to proc_reports
				proc_reports[rid]['diagnosis_2'] = diagnosis_and_conclusions
			if sections['diagnosis_3']:  # diagnosis 3 present
				# split diagnosis into separate sections by * delimiter
				diagnosis = sections['diagnosis_3'].split('*')
				# remove unnecessary empty elements
				diagnosis = [field.strip() for field in diagnosis if field.strip()]  
				# concatenate the corresponding 'conclusions' section to diagnosis
				if 'whole' in conclusions:  # 'conclusions' cannot be divided in multiple sections
					diagnosis_and_conclusions = [diagnosis, conclusions['whole']]
				elif 'iii' in conclusions:  # found section 'iii' within conclusions
					diagnosis_and_conclusions = [diagnosis, conclusions['iii']]
				# assign current diagnosis to proc_reports
				proc_reports[rid]['diagnosis_3'] = diagnosis_and_conclusions

			# sanity check - proc_reports[rid] must be not empty
			assert bool(proc_reports[rid]) == True
		return proc_reports

	def process_radboud_reports_v2(self, reports):  
		"""
		Process Radboud report sections and prepare them for linking
		
		Params:
			reports (dict(dict(str))): the dict of radboud reports containing conclusions/diagnoses 
			
		Returns: a dict containing for each report the list of sections associated to each diagnosis
		"""

		proc_reports = dict()
		for rid, rdata in reports.items():
			# deepcopy rdata to avoid removing elements from input reports
			sections = deepcopy(rdata)
			proc_reports[rid] = dict()
			# split conclusions into sections
			conclusions = self.split_radboud_conclusions(sections.pop('conclusions'))
			# pre process each diagnosis and concatenate its corresponding 'conclusions' section
			if sections['diagnosis_1'] or sections['snomed_1']:  # diagnosis 1 present
				proc_reports[rid]['diagnosis_1'] = {'nlp': [], 'struct': []}
				if sections['diagnosis_1']:  # nlp part present
					# split diagnosis into separate sections by * delimiter
					diagnosis = sections['diagnosis_1'].split('*')
					# remove unnecessary empty elements
					diagnosis = [field.strip() for field in diagnosis if field.strip()]  
					# concatenate the corresponding 'conclusions' section to diagnosis
					if 'whole' in conclusions:  # 'conclusions' cannot be divided in multiple sections
						diagnosis_and_conclusions = [diagnosis, conclusions['whole']]
					elif 'i' in conclusions:  # found section 'i' within conclusions
						diagnosis_and_conclusions = [diagnosis, conclusions['i']]
					# assign current diagnosis to proc_reports
					proc_reports[rid]['diagnosis_1']['nlp'] = diagnosis_and_conclusions
				if sections['snomed_1']:  # struct part present
					codes = [code.strip() for code in sections['snomed_1'].split('*')]
					proc_reports[rid]['diagnosis_1']['struct'] = codes
			
			if sections['diagnosis_2'] or sections['snomed_2']:  # diagnosis 2 present
				proc_reports[rid]['diagnosis_2'] = {'nlp': [], 'struct': []}
				if sections['diagnosis_2']:  # nlp part present
					# split diagnosis into separate sections by * delimiter
					diagnosis = sections['diagnosis_2'].split('*')
					# remove unnecessary empty elements
					diagnosis = [field.strip() for field in diagnosis if field.strip()]  
					# concatenate the corresponding 'conclusions' section to diagnosis
					if 'whole' in conclusions:  # 'conclusions' cannot be divided in multiple sections
						diagnosis_and_conclusions = [diagnosis, conclusions['whole']]
					elif 'ii' in conclusions:  # found section 'ii' within conclusions
						diagnosis_and_conclusions = [diagnosis, conclusions['ii']]
					# assign current diagnosis to proc_reports
					proc_reports[rid]['diagnosis_2']['nlp'] = diagnosis_and_conclusions
				if sections['snomed_2']:  # struct part present
					codes = [code.strip() for code in sections['snomed_2'].split('*')]
					proc_reports[rid]['diagnosis_2']['struct'] = codes

			if sections['diagnosis_3'] or sections['snomed_3']:  # diagnosis 3 present
				proc_reports[rid]['diagnosis_3'] = {'nlp': [], 'struct': []}
				if sections['diagnosis_3']:  # nlp part present
					# split diagnosis into separate sections by * delimiter
					diagnosis = sections['diagnosis_3'].split('*')
					# remove unnecessary empty elements
					diagnosis = [field.strip() for field in diagnosis if field.strip()]  
					# concatenate the corresponding 'conclusions' section to diagnosis
					if 'whole' in conclusions:  # 'conclusions' cannot be divided in multiple sections
						diagnosis_and_conclusions = [diagnosis, conclusions['whole']]
					elif 'iii' in conclusions:  # found section 'iii' within conclusions
						diagnosis_and_conclusions = [diagnosis, conclusions['iii']]
					# assign current diagnosis to proc_reports
					proc_reports[rid]['diagnosis_3']['nlp'] = diagnosis_and_conclusions
				if sections['snomed_3']:  # struct part present
					codes = [code.strip() for code in sections['snomed_3'].split('*')]
					proc_reports[rid]['diagnosis_3']['struct'] = codes

			# sanity check - proc_reports[rid] must be not empty
			assert bool(proc_reports[rid]) == True
		return proc_reports

	'''