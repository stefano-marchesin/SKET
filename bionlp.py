import spacy
import fasttext
import textdistance
import statistics
import itertools
import operator
import re

import utils

from tqdm import tqdm
from collections import Counter
from spacy.tokens import Span
from spacy.matcher import PhraseMatcher
from negex.negation import Negex
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel


class BioNLP(object):

	def __init__(self, biospacy, rules, dysplasia_mappings, cin_mappings, biofast=None, bert=None):
		"""
		Load models and rules

		Params:
			biospacy (str): full spaCy pipeline for biomedical data
			rules (str): hand-crafted rules file path
			dysplasia_mappings (str): dysplasia mappings file path
			cin_mappings (str): cin mappings file path
			biofast (str): biomedical fasttext model
			bert (str): biomedical bert model

		Returns: None
		"""

		# prepare spaCy model
		self.nlp = spacy.load(biospacy)
		# prepare PhraseMatcher model
		self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
		# prepare Negex model
		self.negex = Negex(self.nlp, language ="en_clinical", chunk_prefix = ["free of", "free from"])  # chunk_prefix allows to match also negations chunked together w/ entity mentions
		self.negex.add_patterns(preceding_negations=["free from"])  # @smarchesin TODO: read negations from file if the number of patterns rises
		self.negex.remove_patterns(following_negations=["free"])  # 'free' pattern clashes w/ 'free of' and 'free from' -- @smarchesin TODO: is there a way to fix this without removing 'free'?

		# load hand-crafted rules
		self.rules = utils.read_rules(rules)
		# set patterns for PhraseMatcher 
		self.patterns = {use_case: {trigger: [self.nlp(candidate) for candidate in candidates[0]] for trigger, candidates in rules.items()} for use_case, rules in self.rules.items()}

		# add expand_entity_mentions to spaCy processing pipeline
		self.nlp.add_pipe(self.expand_entity_mentions, name='expand_entities', after='ner')  
		# add negation detector to spaCy pipeline
		self.nlp.add_pipe(self.negex, last=True)

		if biofast and bert:
			print('Please provide either FastText model or BERT model, not both.')
			return False
		if biofast == None and bert == None:
			print('No model provided. Please provide either FastText model and BERT model')
			return False

		if biofast:  # prepare fasttext model
			self.biofast_model = fasttext.load_model(biofast)
		else:
			self.biofast_model = None
		if bert:  # prepare bert model
			self.bert_tokenizer = AutoTokenizer.from_pretrained(bert)
			self.bert_model = AutoModel.from_pretrained(bert)
		else:
			self.bert_model = None

		# load dysplasia mappings
		self.dysplasia = utils.read_dysplasia_mappings(dysplasia_mappings)
		# load cin mappings
		self.cin = utils.read_cin_mappings(cin_mappings)
		# define set of ad hoc linking functions
		self.ad_hoc_linking = {'colon': self.ad_hoc_colon_linking, 'cervix': self.ad_hoc_cervix_linking}

		# set parameter to store the hand-crated rules restricted to a specific use-case (updated w/ self.set_rules() func)
		self.use_case_rules = dict()
		# set parameter to store dysplasia  mappings restricted to a specific use-case
		self.use_case_dysplasia = dict()


	### COMMON FUNCTIONS ###


	def restrict2use_case(self, use_case):  # @smarchesin TODO: remove all the triggers within PhraseMacher by looping over use cases - then consider only the use case ones
		"""
		Restrict hand crafted rules to the considered use-case

		Params: 
			use_case (str): the considered use case

		Returns: the updated rules, candidates, and mappings
		"""

		# restrict hand crafted rules
		self.use_case_rules = self.rules[use_case]
		self.use_case_dysplasia = self.dysplasia[use_case]
		self.use_case_ad_hoc_linking = self.ad_hoc_linking[use_case]
		for trigger, candidates in self.patterns[use_case].items(): 
			if trigger in self.matcher:  # trigger already present within PhraseMatcher - remove it and then add new one (for a different use case)
				self.matcher.remove(trigger)
			self.matcher.add(trigger, None, *candidates)
		return True

	def expand_entity_mentions(self, doc):
		"""
		Expand entity mentions relying on hand-crafted rules

		Params:
			doc (spacy.tokens.doc.Doc): text processed w/ spaCy models

		Returns: a new set of entities for doc
		"""

		spans = list()
		# loop over restricted entities and expand entity mentions based on hand-crafted rules
		for ent in doc.ents:
			# identify triggers for current entity mention
			triggers = [trigger for trigger in self.use_case_rules.keys() if (trigger in ent.text)]
			if triggers:  # current entity presents a trigger
				# keep longest trigger as candidate trigger - e.g., adenocarcinoma instead of carcinoma
				trigger = max(triggers, key=len)
				candidates, location, mode = self.use_case_rules[trigger]
				# check whether the entity mention contains any rule's candidate and exclude those candidates already contained within the entity mention
				target_candidates = [candidate for candidate in candidates if (candidate not in ent.text)]
				# search target candidates within preceding, subsequent or both tokens
				if location == 'PRE':  # candidates are matched on preceding tokens 
					if mode == 'EXACT':  # candidates are matched by exact matching immediately preceding tokens  
						spans = self.pre_exact_match(doc, ent, target_candidates, spans)
					elif mode == 'LOOSE':  # candidates are matched by finding matches within preceding tokens
						spans = self.pre_loose_match(doc, ent, trigger, target_candidates, spans)
					else:  # wrong or mispelled mode - return exception
						print("The mode is wrong or mispelled in the rules.txt file")
						return False
				elif location == 'POST':  # candidates are matched on subsequent tokens 
					if mode == 'EXACT':  # candidates are matched by exact matching immediately subsequent tokens
						spans = self.post_exact_match(doc, ent, target_candidates, spans)
					elif mode == 'LOOSE':  # candidates are matched by finding matches within subsequent tokens
						spans = self.post_loose_match(doc, ent, trigger, target_candidates, spans)
					else:  # wrong or mispelled mode - return exception
						print("The mode is wrong or mispelled in the rules.txt file")
						return False
				elif location == 'BOTH':  # candidates are matched on preceding and subsequent tokens
					if mode == 'EXACT':  # candidates are matched by exact matching immediately preceding and subsequent tokens
						spans = self.pre_exact_match(doc, ent, target_candidates, spans)
						spans = self.post_exact_match(doc, ent, target_candidates, spans)	
					elif mode == 'LOOSE':  # candidates are matched by finding matches within preceding and subsequent tokens
						spans = self.pre_loose_match(doc, ent, trigger, target_candidates, spans)
						spans = self.post_loose_match(doc, ent, trigger, target_candidates, spans)
					else:  # wrong or mispelled mode - return exception
						print("The mode is wrong or mispelled in the rules.txt file")
						return False
				else:  # error in the rules.txt file
					print("The positional information is wrong or mispelled in the rules.txt file")
					return False
			else:  # current entity does not present a trigger
				spans.append([ent.start, ent.end])
		if spans:  # doc contains valid entity mentions
			# merge entities w/ overlapping spans
			merged_spans = self.merge_spans(spans)
			doc.ents = [Span(doc, span[0], span[1], label='ENTITY') for span in merged_spans]
		return doc

	def pre_exact_match(self, doc, ent, candidates, spans):
		"""
		Perform exact matching between entity mention and preceding candidates and return the extended span (i.e., entity mention + candidate)

		Params: 
			doc (spacy.tokens.doc.Doc): text processed w/ spaCy models
			ent (spacy.tokens.doc.Doc.ents): entity mention found by NER
			candidates (list(string)): list of candidates associated to the trigger
			spans (list(list)): list of span ranges [start, end]

		Returns: the list of expanded spans given the entity mentions 
		""" 

		matched_candidate_ix = None
		ix = self.skip_pre_punct(doc, ent.start-1)  # returns previous token index if token.is_punct != True, otherwise None
		if type(ix) == int:   
			for candidate in candidates:  # loop over candidates 
				num_tokens = len(candidate.split())  # number of tokens to inspect
				pre_tokens = doc[ix-num_tokens+1:ix+1]
				if candidate == pre_tokens.text:  # exact match between candidate and tokens
					matched_candidate_ix = pre_tokens.start
		if matched_candidate_ix:
			# expand entity mention
			spans.append([matched_candidate_ix, ent.end])
		else:
			# keep current entity mention as is
			spans.append([ent.start, ent.end])
		return spans

	def skip_pre_punct(self, doc, ix):
		"""
		Get (recursively) the index of the precedent token where token.is_alpha == True (closing punctuation not allowed)

		Params:
			doc (spacy.tokens.doc.Doc): the processed document
			ix (int): the current index

		Returns: the correct token index or None if skip_punct meets EOS
		"""

		if ix == -1 or doc[ix].text == '.':  # BOS or closing punctuation
			return None
		elif not doc[ix].is_punct:  # base case
			return ix
		else:  # recursive case
			return self.skip_pre_punct(doc, ix-1) 

	def post_exact_match(self, doc, ent, candidates, spans):
		"""
		Perform exact matching between entity mention and subsequent candidates and return the extended span (i.e., entity mention + candidate)

		Params: 
			doc (spacy.tokens.doc.Doc): text processed w/ spaCy models
			ent (spacy.tokens.doc.Doc.ents): entity mention found by NER
			candidates (list(string)): list of candidates associated to the trigger
			spans (list(list)): list of span ranges [start, end]

		Returns: the list of expanded spans given the entity mentions 
		"""

		matched_candidate_ix = None
		ix = self.skip_post_punct(doc, ent.end)  # returns next token index if token.is_punct != True, otherwise None
		if type(ix) == int:  
			for candidate in candidates:  # loop over candidates
				num_tokens = len(candidate.split())  # number of tokens to inspect
				post_tokens = doc[ix:ix+num_tokens]
				if candidate == post_tokens.text:  # exact match between candidate and tokens
					matched_candidate_ix = post_tokens.end
		if matched_candidate_ix:
			# expand entity mention
			spans.append([ent.start, matched_candidate_ix])
		else:
			# keep current entity mention as is
			spans.append([ent.start, ent.end])
		return spans

	def skip_post_punct(self, doc, ix):
		"""
		Get (recursively) the index of the posterior token where token.is_alpha == True (closing punctuation not allowed)
		
		Params:
			doc (spacy.tokens.doc.Doc): the processed document
			ix (int): the current index
			
		Returns: the correct token index or None if skip_punct meets EOS
		"""
		
		if ix == len(doc) or doc[ix].text == '.':  # EOS or closing punctuation
			return None
		elif not doc[ix].is_punct:  # base case
			return ix
		else:  # recursive case
			return self.skip_post_punct(doc, ix+1)

	def pre_loose_match(self, doc, ent, trigger, candidates, spans):
		"""
		Perform loose matching between entity mention and preceding candidates and return the extended span (i.e., entity mention + candidate)

		Params: 
			doc (spacy.tokens.doc.Doc): text processed w/ spaCy models
			ent (spacy.tokens.doc.Doc.ents): entity mention found by NER
			trigger (string): token triggered for the entity mention
			candidates (list(string)): list of candidates associated to the trigger
			spans (list(list)): list of span ranges [start, end]

		Returns: the list of expanded preceding spans given the entity mentions 
		"""

		matched_candidates = list()
		ix = self.get_pre_tokens(doc, ent) # returns previous token index if not token.is_punct == True, otherwise None
		if type(ix) == int:  
			# perform matching over doc and return matches
			matches = self.matcher(doc)
			for m_id, m_start, m_end in matches:
				if self.matcher.vocab.strings[m_id] != trigger:  # match w/ different trigger
					continue
				if (m_start < ix) or (m_end > ent.start):  # match out of bounds
					continue
				if doc[m_start:m_end].text not in candidates:  # match out of candidates
					continue
				matched_candidates.append(m_start)  # match found - store starting index
		if matched_candidates:
			# keep earliest candidate index for entity mention's expansion
			fix = min(matched_candidates)
			# expand entity mention
			spans.append([fix, ent.end])
		else:
			# keep current entity mention as is
			spans.append([ent.start, ent.end])
		return spans

	def get_pre_tokens(self, doc, ent):
		"""
		Get index of the first token 

		Params:
			doc (spacy.tokens.doc.Doc): the processed document
			ent (spacy.tokens.span.Span): the current entity

		Returns: the correct token index or None if skip_punct meets BOS
		"""

		sent_ix = ent.sent.start
		if ent.start == sent_ix:  # entity mention is the first token in the sentence - skip it
				return None
		else:  # return index of the first token in sentence
			return sent_ix

	def post_loose_match(self, doc, ent, trigger, candidates, spans):
		"""
		Perform loose matching between entity mention and subsequent candidates and return the extended span (i.e., entity mention + candidate)

		Params: 
			doc (spacy.tokens.doc.Doc): text processed w/ spaCy models
			ent (spacy.tokens.doc.Doc.ents): entity mention found by NER
			trigger (string): token triggered for the entity mention
			candidates (list(string)): list of candidates associated to the trigger
			spans (list(list)): list of span ranges [start, end]

		Returns: the list of expanded subsequent spans given the entity mentions 
		"""

		matched_candidates = list()
		ix = self.get_post_tokens(doc, ent)
		if type(ix) == int:  # returns next token index if not token.is_punct == True, otherwise None
			# perform matching over doc and return matches
			matches = self.matcher(doc)
			for m_id, m_start, m_end in matches:
				if self.matcher.vocab.strings[m_id] != trigger:  # match w/ different trigger
					continue
				if (m_start < ent.end) or (m_end > ix):  # match out of bounds
					continue
				if doc[m_start:m_end].text not in candidates:  # match out of candidates
					continue
				matched_candidates.append(m_end)  # match found - store closing index
		if matched_candidates:
			# keep latest candidate index for entity mention's expansion
			lix = max(matched_candidates)
			# expand entity mention
			spans.append([ent.start, lix])
		else:
			# keep current entity mention as is
			spans.append([ent.start, ent.end])
		return spans

	def get_post_tokens(self, doc, ent):
		"""
		Get (recursively) the index of the subsequent token where token.is_alpha == True
		
		Params:
			doc (spacy.tokens.doc.Doc): the processed document
			ent (spacy.tokens.span.Span): the current entity
			
		Returns: the correct token index or None if skip_punct meets EOS
		"""
		
		sent_ix = ent.sent.end
		if ent.end == sent_ix:  # entity mention is the last token in the sentence - skip it
			return None
		else:  # return index of the last token in sentence
			return sent_ix

	def merge_spans(self, spans):
		"""
		Merge spans w/ overlapping ranges
		
		Params:
			spans (list(list)): list of span ranges [start, end]
			
		Returns: a list of merged span ranges [start, end]
		"""
		
		spans.sort(key=lambda span: span[0]) 
		merged_spans = [[spans[0][0], spans[0][1]]]  # avoid copying by reference
		for current in spans:
			previous = merged_spans[-1]
			if current[0] < previous[1]:
				previous[1] = max(previous[1], current[1])
			else:
				merged_spans.append(current)
		return merged_spans

	def extract_entity_mentions(self, text, keep_negated=False):
		"""
		Extract entity mentions identified within text.

		Params:
			text (str): text to be processed.
			keep_negated (bool): keep negated entity mentions

		Returns: a list of named/unnamed detected entity mentions
		"""

		doc = self.nlp(text)
		if keep_negated:  # keep negated mentions
			return [mention for mention in doc.ents]
		else: 
			return [mention for mention in doc.ents if mention._.negex == False]

	def text_similarity(self, mention, label):
		"""
		Compute different similarity measures between entity mention and concept label

		Params:
			mention (spacy.tokens.span.Span): entity mention extracted from text
			label (spacy.tokens.doc.Doc): concept label from ontology

		Returns: dict of similarity scores per metric plus aggregated score 
		"""

		sim_scores = dict()
		# compute similarity metrics
		if self.bert_model:  # compute bert-based similarity 
			mention_tokens = self.bert_tokenizer(mention.text, return_tensors="pt")
			mention_embs = self.bert_model(**mention_tokens)[1].detach().numpy()
			label_tokens = self.bert_tokenizer(label.text, return_tensors="pt")
			label_embs =self.bert_model(**label_tokens)[1].detach().numpy()
			sim_scores['bert'] = cosine_similarity(mention_embs, label_embs)[0]
		else:  # compute semantic + lexical similarity
			sim_scores['ratcliff_obershelp'] = textdistance.ratcliff_obershelp.normalized_similarity(mention.text, label.text)
			sim_scores['word2vec'] = mention.similarity(label)
			sim_scores['fasttext'] = cosine_similarity([self.biofast_model.get_sentence_vector(mention.text)], [self.biofast_model.get_sentence_vector(label.text)])[0]
		return sim_scores

	def associate_mention2candidate(self, mention, labels, bert_thr=0.99, w2v_thr=0.7, ft_thr=0.7, ro_thr=0.5):  # @smarchesin TODO: needs to be improved -- if not_scores_word2vec occurs also when OOV, fix this
		"""
		Associate entity mention to candidate concept label

		Params:
			mention (spacy.token.span.Span): entity mention extracted from text
			labels (list(spacy.token.span.Span)): list of concept labels from reference ontology
			bert_thr (float): threshold to keep candidate labels matched using bert embeddings similarity
			w2v_thr (float): threshold to keep candidate labels matched using word2vec embeddings similarity
			ft_thr (float): threshold to keep candidate labels matched using fasttext embeddings similarity
			ro_thr (float): threshold to keep candidate labels matched using ratcliff obershelp sub-string matching similarity

		Returns: candidate ontology concept (or None)
		"""

		# perform similarity scores between the entity mention and the list of ontology labels
		scores_and_labels = [(self.text_similarity(mention, label), label.text) for label in labels]
		if self.bert_model:  # bert-based matching
			scores_bert = [score_and_label for score_and_label in scores_and_labels if score_and_label[0]['bert'] >= bert_thr]
			if not scores_bert:  # no match found
				return [[mention.text, None]]
			else:  # match(es) found
				best_bert = max(scores_bert, key=lambda score:score[0]['bert'])[1]
				return [[mention.text, best_bert]]
		else:  # semantic + lexical matching
			# keep labels w/ score greater or equal to word2vec threshold 
			scores_word2vec = [score_and_label for score_and_label in scores_and_labels if score_and_label[0]['word2vec'] >= w2v_thr]
			if not scores_word2vec:  # no match found
				return [[mention.text, None]]
			elif len(scores_and_labels) > 1:  # matched mutiple candidate labels with word2vec
				# get word2vec candidate label w/ highest score
				best_word2vec = max(scores_word2vec, key=lambda score:score[0]['word2vec'])[1]
				# keep labels w/ score greater or equal to fasttext threshold 
				scores_fasttext = [score_and_label for score_and_label in scores_and_labels if score_and_label[0]['fasttext'] >= ft_thr]
				if not scores_fasttext:  # no match found w/ fasttext
					return [[mention.text, best_word2vec]]
				else:  # compare word2vec candidates w/ fasttext ones
					# get fasttext candidate label w/ highest score
					best_fasttext = max(scores_fasttext, key=lambda score:score[0]['fasttext'])[1]
					if best_word2vec == best_fasttext:  # cross-check: word2vec and fasttext have the same best match
						return [[mention.text, best_word2vec]]
					else:  # cross-check: word2vec and fasttext have different best matches
						# keep labels w/ score greater or equal to string matching threshold 
						scores_string_match = [score_and_label for score_and_label in scores_and_labels if score_and_label[0]['ratcliff_obershelp'] >= ro_thr]
						if not scores_string_match:  # no match found w/ string matching
							# keep word2vec candidate label with highest score
							return [[mention.text, best_word2vec]]
						else:  # perform majority vote between word2vec, fasttext, and string matching 
							# get ratcliff obershelp candidate label w/ highest score
							best_string_match = max(scores_string_match, key=lambda score:score[0]['ratcliff_obershelp'])[1]
							# perform majority vote and keep most_common candidate label
							majority = Counter([best_word2vec, best_fasttext, best_string_match]).most_common()[0]
							if  majority[1] > 1:  # return best candidate from majority vote
								return [[mention.text, majority[0]]]
							else:  # majority vote provided no clear answer
								return [[mention.text, best_word2vec]]
			else:  # link entity mention to the only candidate label matched
				return [[mention.text, scores_word2vec[0][1]]]

	def link_mentions_to_concepts(self, mentions, labels, use_case_ontology, raw=False, bert_thr=0.99, w2v_thr=0.7, ft_thr=0.7, ro_thr=0.5):
		"""
		Link identified entity mentions to ontology concepts 

		Params:
			mentions (list(spacy.token.span.Span)): list of entity mentions extracted from text
			labels (list(spacy.token.span.Span)): list of concept labels from reference ontology
			use_case_ontology (pandas DataFrame): reference ontology restricted to the use case considered
			raw (bool): boolean to keep raw or cleaned version of linked concepts
			bert_thr (float): threshold to keep candidate labels matched using bert embeddings similarity 
			w2v_thr (float): threshold to keep candidate labels matched using word2vec embeddings similarity
			ft_thr (float): threshold to keep candidate labels matched using fasttext embeddings similarity
			ro_thr (float): threshold to keep candidate labels matched using ratcliff obershelp sub-string matching similarity

		Returns: a dict of identified ontology concepts {semantic_area: [iri, mention, label], ...}
		"""

		# link mentions to concepts
		mentions_and_concepts = [self.use_case_ad_hoc_linking(mention, labels, bert_thr=bert_thr, w2v_thr=w2v_thr, ft_thr=ft_thr, ro_thr=ro_thr) for mention in mentions]
		mentions_and_concepts = list(itertools.chain.from_iterable(mentions_and_concepts))
		# extract linked data from ontology
		linked_data = [(mention_and_concept[0], use_case_ontology.loc[use_case_ontology['label'].str.lower() == mention_and_concept[1]][['iri', 'label', 'semantic_area_label']].values[0].tolist()) 
						for mention_and_concept in mentions_and_concepts if mention_and_concept[1] is not None]
		# filter out linked data 'semantic_area_label' == None
		linked_data = [linked_datum for linked_datum in linked_data if linked_datum[1][2] is not None]
		if raw:
			return linked_data
		else:
			# return linked concepts divided into semantic areas
			linked_concepts = {area: [] for area in set(use_case_ontology['semantic_area_label'].tolist()) if area is not None}
			for linked_datum in linked_data:
				linked_concepts[str(linked_datum[1][2])].append([linked_datum[1][0], linked_datum[1][1]])
			return linked_concepts


	### COLON SPECIFIC LINKING FUNCTIONS ###


	def ad_hoc_colon_linking(self, mention, labels, bert_thr=0.99, w2v_thr=0.7, ft_thr=0.7, ro_thr=0.5): 
		"""
		Perform set of colon ad hoc linking functions 

		Params: 
			mention (spacy.tokens.span.Span): entity mention extracted from text
			labels (list(spacy.token.span.Span)): list of concept labels from reference ontology
			bert_thr (float): threshold to keep candidate labels matched using bert embeddings similarity
			w2v_thr (float): threshold to keep candidate labels matched using word2vec embeddings similarity
			ft_thr (float): threshold to keep candidate labels matched using fasttext embeddings similarity
			ro_thr (float): threshold to keep candidate labels matched using ratcliff obershelp sub-string matching similarity

		Returns: matched ontology concept label(s)
		"""

		if 'dysplasia' in mention.text:  # mention contains 'dysplasia'
			return self.link_colon_dysplasia(mention)
		elif 'carcinoma' in mention.text:  # mention contains 'carcinoma'
			return self.link_colon_adenocarcinoma(mention)
		elif 'hyperplastic' in mention.text:  # mention contains 'hyperplastic'
			return self.link_colon_hyperplastic_polyp(mention)
		elif 'biopsy' in mention.text:  # mention contains 'biopsy'
			return self.link_colon_biopsy(mention, labels, bert_thr, w2v_thr, ft_thr, ro_thr)
		elif 'colon' == mention.text:  # mention matches 'colon' -- @smarchesin TODO: once we get a better similarity function, self.link_colon_nos should be deprecated
			return self.link_colon_nos(mention)
		elif 'polyp' == mention.text:  # mention matches 'polyp' -- @smarchesin TODO: once we get a better similarity function, self.link_colon_polyp should be deprecated
			return self.link_colon_polyp(mention)
		else:  # none of the ad hoc functions was required -- perform similarity-based linking
			return self.associate_mention2candidate(mention, labels, bert_thr, w2v_thr, ft_thr, ro_thr)

	def link_colon_dysplasia(self, mention):
		"""
		Identify (when possible) the colon dysplasia grade and link the dysplasia mention to the correct concept 
		
		Params:
			mention (spacy.tokens.span.Span): (dysplasia) entity mention extracted from text
		
		Returns: matched ontology concept label(s)
		"""
		
		dysplasia_mention = mention.text
		# identify dysplasia grades within mention
		grades = [self.use_case_dysplasia[trigger] for trigger in self.use_case_dysplasia.keys() if trigger in dysplasia_mention]
		grades = set(itertools.chain.from_iterable(grades))
		if grades:  # at least one dysplasia grade identified
			return [[dysplasia_mention, grade] for grade in grades]
		else:  # no dysplasia grades identified - map to simple Colon Dysplasia
			return [[dysplasia_mention, 'colon dysplasia']]

	def link_colon_adenocarcinoma(self, mention):  # @smarchesin TODO: needs to be improved to handle a larger pool of 'metastatic' cases -- and possibly other unpredicted situations
		"""
		Link (adeno)carcinoma mentions to the correct concepts

		Params:
			mention (spacy.tokens.span.Span): (hyperplastic polyp) entity mention extracted from text

		Returns: matched ontology concept label
		"""

		carcinoma_mention = mention.text
		if 'metasta' in carcinoma_mention:  # metastatic adenocarcinoma found -- 'metasta' handles both metasta-tic and metasta-sis/ses
			return [[carcinoma_mention, 'metastatic adenocarcinoma']]
		else:  # colon adenocarcinoma found
			return [[carcinoma_mention, 'colon adenocarcinoma']]

	def link_colon_hyperplastic_polyp(self, mention):
		"""
		Link hyperplastic polyp mentions to the correct concepts

		Params:
			mention (spacy.tokens.span.Span): (hyperplastic polyp) entity mention extracted from text

		Returns: matched ontology concept label(s)
		"""

		hyperplastic_mention = mention.text
		# idenfity presence of hyperplastic adenomatous polyps
		if 'adenomatous' in hyperplastic_mention:  # hyperplastic adenomatous polyp found
			return [[hyperplastic_mention, 'colon hyperplastic polyp'], [hyperplastic_mention, 'adenoma']]
		elif 'inflammatory' in hyperplastic_mention:  # hyperplastic inflammatory polyp found
			return [[hyperplastic_mention, 'colon hyperplastic polyp'], [hyperplastic_mention, 'colon inflammatory polyp']]
		elif 'glands' in hyperplastic_mention:  # hyperplastic glands found - skip it
			return [[hyperplastic_mention, None]]
		else:  # hyperplastic polyp found
			return [[hyperplastic_mention, 'colon hyperplastic polyp']]

	def link_colon_nos(self, mention): 
		"""
		Link colon mentions to the colon concept

		Params:
			mention (spacy.tokens.span.Span): (colon) entity mention extracted from text

		Returns: matched ontology concept label
		"""

		colon_mention = mention.text
		assert colon_mention == 'colon'
		return [[colon_mention, 'colon, nos']]

	def link_colon_polyp(self, mention):  # @smarchesin TODO: what about plural nouns?
		"""
		Link polyp mentions to the polyp concept

		Params:
			mention (spacy.tokens.span.Span): (polyp) entity mention extracted from text

		Returns: matched ontology concept label
		"""

		polyp_mention = mention.text
		assert polyp_mention == 'polyp'
		return [[polyp_mention, 'polyp of colon']]

	def link_colon_biopsy(self, mention, labels, bert_thr=0.99, w2v_thr=0.7, ft_thr=0.7, ro_thr=0.5):  # @smarchesin TODO: too hardcoded? too simplistic? what about plural nouns?
		"""
		Link colon biopsy mentions and the associated anatomical locations (if any)

		Params:
			mention (spacy.tokens.span.Span): entity mention extracted from text
			labels (list(spacy.token.span.Span)): list of concept labels from reference ontology
			bert_thr (float): threshold to keep candidate labels matched using bert embeddings similarity
			w2v_thr (float): threshold to keep candidate labels matched using word2vec embeddings similarity
			ft_thr (float): threshold to keep candidate labels matched using fasttext embeddings similarity
			ro_thr (float): threshold to keep candidate labels matched using ratcliff obershelp sub-string matching similarity

		Returns: colon biopsy concept and matched anatomical locations (if any)
		"""

		if mention.text == 'biopsy':  # mention contains 'biopsy' only
			return [[mention.text, 'biopsy of colon']]
		elif mention.text == 'colon biopsy':  # mention contains 'colon biopsy'
			return [[mention.text, 'biopsy of colon'], [mention.text, 'colon, nos']]
		elif mention[:2].text == 'colon biopsy':  # 'colon biopsy' as first term - match rest of mention w/ similarity-based linking
			anatomical_location = self.associate_mention2candidate(mention[2:], labels, bert_thr, w2v_thr, ft_thr, ro_thr)
			if anatomical_location[0][1]:  # anatomical location found within mention
				return [[mention.text, 'biopsy of colon']] + anatomical_location
			else:  # return 'colon, nos' because of 'colon' within biopsy mention
				return [[mention.text, 'biopsy of colon'], [mention.text, 'colon, nos']]
		elif mention[-2:].text == 'colon biopsy':  # 'colon biopsy' as last term - match rest of mention w/ similarity-based linking
			anatomical_location = self.associate_mention2candidate(mention[:-2], labels, bert_thr, w2v_thr, ft_thr, ro_thr)
			if anatomical_location[0][1]:  # anatomical location found within mention
				return [[mention.text, 'biopsy of colon']] + anatomical_location
			else:  # return 'colon, nos' because of 'colon' within biopsy mention
				return [[mention.text, 'biopsy of colon'], [mention.text, 'colon, nos']]
		elif mention[0].text == 'biopsy':  # 'biopsy' as first term - match rest of mention w/ similarity-based linking
			if 'colon' not in mention.text:  # 'colon' not in mention
				return [[mention.text, 'biopsy of colon']] + self.associate_mention2candidate(mention[1:], labels, bert_thr, w2v_thr, ft_thr, ro_thr)
			else:  # 'colon' in mention -- hard to handle appropriately, keep biopsy and colon as concepts
				return [[mention.text, 'biopsy of colon'], [mention.text, 'colon, nos']]
		elif mention[-1].text == 'biopsy':  # 'biopsy' as last term - match rest of mention w/ similarity-based linking
			if 'colon' not in mention.text:  # 'colon' not in mention
				return [[mention.text, 'biopsy of colon']] + self.associate_mention2candidate(mention[:-1], labels, bert_thr, w2v_thr, ft_thr, ro_thr)
			else:  # 'colon' in mention -- hard to handle appropriately, keep biopsy and colon as concepts
				return [[mention.text, 'biopsy of colon'], [mention.text, 'colon, nos']]
		else:  # biopsy not BOS or EOS
			if 'colon' not in mention.text:  # 'colon' not in mention
				biopsy_idx = [idx for idx, term in enumerate(mention) if 'biopsy' in term.text][0]  # get 'biopsy' mention index
				pre_anatomical_location = [['', '']]
				post_anatomical_location = [['', '']]
				if mention[:biopsy_idx]:  # link mention before 'biopsy'
					pre_anatomical_location = self.associate_mention2candidate(mention[:biopsy_idx], labels, bert_thr, w2v_thr, ft_thr, ro_thr)  
				if mention[biopsy_idx+1:]:  # link mention after 'biopsy'
					post_anatomical_location = self.associate_mention2candidate(mention[biopsy_idx+1:], labels, bert_thr, w2v_thr, ft_thr, ro_thr) 
				if pre_anatomical_location[0][1] and post_anatomical_location[0][1]:  # both mentions matched
					return [[mention.text, 'biopsy of colon']] + pre_anatomical_location + post_anatomical_location
				elif pre_anatomical_location[0][1]:  # only pre mention matched
					return [[mention.text, 'biopsy of colon']] + pre_anatomical_location
				elif post_anatomical_location[0][1]:  # only post mention matched
					return [[mention.text, 'biopsy of colon']] + post_anatomical_location
				else:  # no mention matched - return only 'biopsy of colon' concept
					return [[mention.text, 'biopsy of colon']]
			else:  # 'colon' in mention -- hard to handle appropriately, keep biopsy and colon as concepts
				return [[mention.text, 'biopsy of colon'], [mention.text, 'colon, nos']]


	### CERVIX SPECIFIC LINKING FUNCTIONS ###


	def ad_hoc_cervix_linking(self, mention, labels, bert_thr=0.99, w2v_thr=0.7, ft_thr=0.7, ro_thr=0.5): 
		"""
		Perform set of cervix ad hoc linking functions 

		Params: 
			mention (spacy.tokens.span.Span): entity mention extracted from text
			labels (list(spacy.token.span.Span)): list of concept labels from reference ontology
			bert_thr (float): threshold to keep candidate labels matched using bert embeddings similarity
			w2v_thr (float): threshold to keep candidate labels matched using word2vec embeddings similarity
			ft_thr (float): threshold to keep candidate labels matched using fasttext embeddings similarity
			ro_thr (float): threshold to keep candidate labels matched using ratcliff obershelp sub-string matching similarity

		Returns: matched ontology concept label(s)
		"""

		if 'dysplasia' in mention.text or 'squamous intraepithelial lesion' in mention.text:  # mention contains 'dysplasia' or 'squamous intraepithelial lesion'
			return self.link_cervix_dysplasia(mention)
		elif re.search(r'\bcin\d*', mention.text) or re.search(r'sil\b', mention.text):  # mention contains 'cin' or 'sil'
			return self.link_cervix_cin(mention)
		elif 'hpv' in mention.text:  # mention contains 'hpv'
			return self.link_cervix_hpv(mention, labels, bert_thr, w2v_thr, ft_thr, ro_thr)
		elif 'infection' in mention.text:  # mention contains 'infection'
			return self.skip_cervix_infection(mention, labels)
		elif 'epithelium' in mention.text or 'junction' in mention.text:  # mention contains 'epithelium' or 'junction'
			return self.link_cervix_epithelium(mention)
		elif 'leep' in mention.text:  # mention contains 'leep'
			return self.link_cervix_leep(mention)
		elif 'biopsy' in mention.text and 'portio' in mention.text:  # mention contains 'biopsy portio'
			return self.link_cervix_conization(mention)
		else:  # none of the ad hoc functions was required -- perform similarity-based linking
			return self.associate_mention2candidate(mention, labels, bert_thr, w2v_thr, ft_thr, ro_thr)

	def link_cervix_dysplasia(self, mention):
		"""
		Identify (when possible) the cervix dysplasia grade and link the dysplasia mention to the correct concept 
		
		Params:
			mention (spacy.tokens.span.Span): (dysplasia) entity mention extracted from text

		Returns: matched ontology concept label(s)
		"""
		
		dysplasia_mention = mention.text
		# identify dysplasia grades within mention
		grades = [self.use_case_dysplasia[trigger] for trigger in self.use_case_dysplasia.keys() if trigger in dysplasia_mention]
		grades = set(itertools.chain.from_iterable(grades))
		if grades:  # at least one dysplasia grade identified
			return [[dysplasia_mention, grade] for grade in grades]
		else:  # no dysplasia grades identified - map to simple CIN
			return [[dysplasia_mention, 'cervical intraepithelial neoplasia']]

	def link_cervix_cin(self, mention):
		"""
		Identify (when possible) the cervix cin/sil grade and link the cin/sil mention to the correct concept 
		
		Params:
			mention (spacy.tokens.span.Span): (cin) entity mention extracted from text
		
		Returns: matched ontology concept label(s)
		"""
		
		cin_mention = mention.text
		# identify cin/sil grades within mention
		grades = [self.cin[trigger] for trigger in self.cin.keys() if trigger in cin_mention]
		if grades:  # at least one cin/sil grade identified
			return [[cin_mention, grade] for grade in grades]
		else:  # no cin/sil grades identified - map to simple cin/sil
			return [[cin_mention, 'cervical intraepithelial neoplasia']]

	def link_cervix_hpv(self, mention, labels, bert_thr=0.99, w2v_thr=0.7, ft_thr=0.7, ro_thr=0.5):  # @smarchesin TODO: too hardcoded? too simplistic?
		"""
		Link cervix hpv mentions and the associated anatomical locations (if any)

		Params:
			mention (spacy.tokens.span.Span): entity mention extracted from text
			labels (list(spacy.token.span.Span)): list of concept labels from reference ontology
			bert_thr (float): threshold to keep candidate labels matched using bert embeddings similarity
			w2v_thr (float): threshold to keep candidate labels matched using word2vec embeddings similarity
			ft_thr (float): threshold to keep candidate labels matched using fasttext embeddings similarity
			ro_thr (float): threshold to keep candidate labels matched using ratcliff obershelp sub-string matching similarity

		Returns: cervix hpv concept and matched anatomical locations (if any)
		"""

		if mention.text == 'hpv':  # mention contains 'hpv' only
			return [[mention.text, 'human papilloma virus infection']]
		elif mention.text == 'hpv infection':  # mention contains 'hpv infection'
			return [[mention.text, 'human papilloma virus infection']]
		elif mention[:2].text == 'hpv infection':  # 'hpv infection' as first term - match rest of mention w/ similarity-based linking
				return [[mention[:2].text, 'human papilloma virus infection']] + self.associate_mention2candidate(mention[2:], labels, bert_thr, w2v_thr, ft_thr, ro_thr)
		elif mention[-2:].text == 'hpv infection':  # 'hpv infection' as last term - match rest of mention w/ similarity-based linking
			return [[mention[-2:].text, 'human papilloma virus infection']] + self.associate_mention2candidate(mention[:-2], labels, bert_thr, w2v_thr, ft_thr, ro_thr)
		elif mention[0].text == 'hpv':  # 'hpv' as first term - match rest of mention w/ similarity-based linking
			return [[mention[0].text, 'human papilloma virus infection']] + self.associate_mention2candidate(mention[1:], labels, bert_thr, w2v_thr, ft_thr, ro_thr)
		elif mention[-1].text == 'hpv':  # 'hpv' as last term - match rest of mention w/ similarity-based linking
			return [[mention[-1].text, 'human papilloma virus infection']] + self.associate_mention2candidate(mention[:-1], labels, bert_thr, w2v_thr, ft_thr, ro_thr)
		else:  # biopsy not BOS or EOS
			hpv_idx = [idx for idx, term in enumerate(mention) if 'hpv' in term.text][0]  # get 'hpv' mention index 
			pre_anatomical_location = [['', '']]
			post_anatomical_location = [['', '']]
			if mention[:hpv_idx]:  # link mention before 'hpv'
				pre_anatomical_location = self.associate_mention2candidate(mention[:hpv_idx], labels, bert_thr, w2v_thr, ft_thr, ro_thr)  
			if mention[hpv_idx+1:]:  # link mention after 'hpv'
				post_anatomical_location = self.associate_mention2candidate(mention[hpv_idx+1:], labels, bert_thr, w2v_thr, ft_thr, ro_thr)  
			if pre_anatomical_location[0][1] and post_anatomical_location[0][1]:  # both mentions matched
				return [[mention[hpv_idx].text, 'human papilloma virus infection']] + pre_anatomical_location + post_anatomical_location
			elif pre_anatomical_location[0][1]:  # only pre mention matched
				return [[mention[hpv_idx].text, 'human papilloma virus infection']] + pre_anatomical_location
			elif post_anatomical_location[0][1]:  # only post mention matched
				return [[mention[hpv_idx].text, 'human papilloma virus infection']] + post_anatomical_location
			else:  # no mention matched - return only 'human papilloma virus infection' concept
				return [[mention[hpv_idx].text, 'human papilloma virus infection']]

	def link_cervix_epithelium(self, mention): 
		"""
		Identify (when possible) the cervix epithelium type and link it to the correct concept 

		Params:
			mention (spacy.tokens.span.Span): entity mention extracted from text
			labels (list(spacy.token.span.Span)): list of concept labels from reference ontology

		Returns: cervix epithelium concept
		"""

		epithelium_mention = mention.text
		# identify epithelium types within mention
		if 'simple' in epithelium_mention:
			return [[epithelium_mention, 'simple epithelium']]
		elif 'pavement' in epithelium_mention:
			return [[epithelium_mention, 'pavement epithelium']]
		elif 'junction' in epithelium_mention:
			return [[epithelium_mention, 'cervical squamo-columnar junction']]
		elif 'ectocervical' in epithelium_mention:
			return [[epithelium_mention, 'exocervical epithelium']]
		elif 'exocervical' in epithelium_mention:
			return [[epithelium_mention, 'exocervical epithelium']]
		elif 'glandular' in epithelium_mention:
			return [[epithelium_mention, 'cervix glandular epithelium']]
		elif 'squamous' in epithelium_mention:
			return [[epithelium_mention, 'cervix squamous epithelium']]
		else:  # epithelium type not found
			return [[epithelium_mention, 'cervix epithelium']]

	def link_cervix_leep(self, mention): 
		"""
		Link cervical leep mention to the correct concept 

		Params:
			mention (spacy.tokens.span.Span): entity mention extracted from text

		Returns: cervical leep concept
		"""

		leep_mention = mention.text
		assert 'leep' in leep_mention
		return [[leep_mention, 'loop electrosurgical excision']]

	def link_cervix_conization(self, mention): 
		"""
		Link biopsy portio mention to conization concept

		Params:
			mention (spacy.tokens.span.Span): entity mention extracted from text

		Returns: conization concept
		"""

		conization_mention = mention.text
		assert 'biopsy' in conization_mention and 'portio' in conization_mention
		return [[conization_mention, 'conization']]

	def skip_cervix_infection(self, mention, labels):  # @smarchesin TODO: remove the third case (else condition) after testing
		"""
		Skip 'infection' mentions that are associated to 'hpv'

		Params:
			mention (spacy.tokens.span.Span): entity mention extracted from text
			labels (list(spacy.token.span.Span)): list of concept labels from reference ontology

		Returns: cervix concept if 'infection' is not associated to 'hpv' or None otherwise
		"""

		if mention.text == 'infection':  # mention contains 'infection' only -- skip it
			return [[mention.text, None]]
		elif mention.text == 'viral infection':  # mention contains 'viral infection' only -- skip it
			return [[mention.text, None]]
		else:  # mention contains other terms other than 'infection' -- unhandled
			print('mention contains unhandled "infection" mention -- set temp to None')
			print(mention.text)
			return [[mention.text, None]]


	### ONTOLOGY-RELATED FUNCTIONS ###


	def process_ontology_concepts(self, labels):
		"""
		Process ontology labels using scispaCy

		Params:
			labels (list): list of concept labels

		Returns: a list of processed concept labels
		"""

		return [self.nlp(label) for label in labels]

	def lookup_snomed_codes(self, snomed_codes, use_case_ontology):
		"""
		Lookup for ontology concepts associated to target SNOMED codes

		Params:
			snomed_codes (list(str)/str): target SNOMED codes
			use_case_ontology (pandas DataFrame): reference ontology restricted to the use case considered

		Returns: a dict of identified ontology concepts {semantic_area: [iri, label], ...}
		"""
		
		lookups = {area: [] for area in set(use_case_ontology['semantic_area_label'].tolist()) if area is not None}
		if type(snomed_codes) == list:  # search for list of snomed codes
			snomed_codes = [code for code in snomed_codes if code]
			if snomed_codes:
				linked_data = use_case_ontology.loc[use_case_ontology['SNOMED'].isin(snomed_codes)][['iri', 'label', 'semantic_area_label']]
				if not linked_data.empty:  # matches found within ontology
					for linked_datum in linked_data.values.tolist():
						lookups[str(linked_datum[2])].append([linked_datum[0], linked_datum[1]])
			return lookups
		else:  # search for single snomed code
			if snomed_codes:
				linked_data = use_case_ontology.loc[use_case_ontology['SNOMED'] == snomed_codes][['iri', 'label', 'semantic_area_label']]
				if not linked_data.empty:  # match found within ontology
					linked_datum = linked_data.values[0].tolist()
					lookups[str(linked_datum[2])].append([linked_datum[0], linked_datum[1]])
			return lookups


	### AOEC SPECIFIC FUNCTIONS ###


	def aoec_entity_linking(self, reports, onto_proc, labels, use_case, use_case_ontology, raw=False, bert_thr=0.99, w2v_thr=0.7, ft_thr=0.7, ro_thr=0.5):  # @smarchesin TODO: is there a better way to handle 'polyps' within 'materials' section?
		"""
		Perform entity linking over translated AOEC reports
		
		Params:
			reports (dict): target reports
			onto_proc (OntologyProc): instance of OntologyProc class
			labels (list(spacy.tokens.doc.Doc)): list of processed ontology concepts
			use_case (str): the use_case considered - i.e. colon, lung, cervix, or celiac
			use_case_ontology (pandas.core.frame.DataFrame): ontology data restricted to given use case
			raw (bool): boolean to keep raw or cleaned version of linked concepts
			bert_thr (float): threshold to keep candidate labels matched using bert embeddings similarity 
			w2v_thr (float): threshold to keep candidate labels matched using word2vec embeddings similarity
			ft_thr (float): threshold to keep candidate labels matched using fasttext embeddings similarity
			ro_thr (float): threshold to keep candidate labels matched using ratcliff obershelp sub-string matching similarity
			
		Returns: a dict containing the linked concepts for each report w/ distinction between 'nlp' and 'struct' concepts
		"""
		
		concepts = dict()
		# loop over AOEC reports and perform linking
		for rid, rdata in tqdm(reports.items()):
			concepts[rid] = dict()
			# extract entity mentions from text sections
			mentions = self.extract_entity_mentions(utils.sanitize_record(rdata['diagnosis_nlp'], use_case))
			# consider 'polyp' as a stopwords in 'materials' section
			mentions += self.extract_entity_mentions(re.sub('polyp[s]?(\s|$)+', ' ', utils.sanitize_record(rdata['materials'], use_case)))  # @smarchesin TODO: should be restricted to use case
			# link and store 'nlp' concepts
			nlp_concepts = self.link_mentions_to_concepts(mentions, labels, use_case_ontology, raw, bert_thr, w2v_thr, ft_thr, ro_thr)
			# link and store 'struct' concepts
			struct_concepts = self.lookup_snomed_codes(utils.sanitize_codes(rdata['diagnosis_struct']) + 
													   utils.sanitize_codes(rdata['procedure']) + 
													   utils.sanitize_codes(rdata['topography']), 
													   use_case_ontology)
			# merge 'nlp' and 'struct' concepts 
			concepts[rid] = onto_proc.merge_nlp_and_struct(nlp_concepts, struct_concepts)
		# return concepts divided into 'nlp' and 'struct' sections (used for debugging, evaluation, and other applications)
		return concepts


	### RADBOUD SPECIFIC FUNCTIONS ###

	def radboud_entity_linking(self, reports, onto_proc, labels, use_case, use_case_ontology, raw=False, bert_thr=0.99, w2v_thr=0.7, ft_thr=0.7, ro_thr=0.5):
		"""
		Perform entity linking over translated and processed Radboud reports

		Params:
			reports (dict): target reports 
			onto_proc (OntologyProc): instance of OntologyProc class
			labels (list(spacy.tokens.doc.Doc)): list of processed ontology concepts
			use_case (str): the use_case considered - i.e. colon, lung, cervix, or celiac
			use_case_ontology (pandas.core.frame.DataFrame): ontology data restricted to given use case
			raw (bool): boolean to keep raw or cleaned version of linked concepts
			bert_thr (float): threshold to keep candidate labels matched using bert embeddings similarity 
			w2v_thr (float): threshold to keep candidate labels matched using word2vec embeddings similarity
			ft_thr (float): threshold to keep candidate labels matched using fasttext embeddings similarity
			ro_thr (float): threshold to keep candidate labels matched using ratcliff obershelp sub-string matching similarity

		Returns: a dict containing the linked concepts for each report w/ list of associated slides
		"""
		
		concepts = dict()
		# loop over Radboud processed reports and perform linking
		for rid, rdata in tqdm(reports.items()):
			concepts[rid] = dict()
			# extract entity mentions from conclusions
			nlp_mentions = self.extract_entity_mentions(utils.sanitize_record(rdata['diagnosis'], use_case))
			# link and store concepts from conclusions
			nlp_concepts = self.link_mentions_to_concepts(nlp_mentions, labels, use_case_ontology, raw, bert_thr, w2v_thr, ft_thr, ro_thr)
			# assign conclusion concepts to concepts dict
			concepts[rid]['concepts'] = nlp_concepts
			# assign slide ids to concepts dict
			concepts[rid]['slide_ids'] = rdata['slide_ids']
		# return linked concepts divided per diagnosis
		return concepts