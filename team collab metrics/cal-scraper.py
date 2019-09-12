from bs4 import BeautifulSoup
import requests
import webbrowser
import time
import sys
from html.parser import HTMLParser
import lxml
from lxml.html.clean import Cleaner
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re

# import nltk
# nltk.download('punkt')

# Credits to Calgary iGEM for scraper classes
##
# MLStripper:
# 			Removes HTML tags
##
class MLStripper(HTMLParser):
	def __init__(self):
		self.reset()
		self.strict = False
		self.convert_charrefs= True
		self.fed = []
	def handle_data(self, d):
		self.fed.append(d)
	def get_data(self):
		return ''.join(self.fed)
	

##
# Parser:
# Scrapes iGEM teams' wiki pages that contain software information
##

class Parser():	

	def stripTags(self, text):
		s = MLStripper()
		s.feed(text)
		return s.get_data()
	
	##
	# getData
	# Returns text from wiki containing software info
	# Accepts year as a string
	##
	
	def getData(self,year):
	
		# All wiki software data [0]: teamName [1]: scraped text
		softData = []			
		
		#  Retreive links to wiki pages containing software info
		linksWithSoftware = self.getSoftLinks(year)

		# For each team with software, get text from pages containing software info
		for i in range(0, len(linksWithSoftware), 1):
			for entry in range(0, len(linksWithSoftware[i]), 1):
				softwareWikiSource = requests.get(linksWithSoftware[i][entry]).text
				soup = BeautifulSoup(softwareWikiSource, 'lxml')
				
				# Removes javascript and css styling
				
				for tag in soup():
					for attribute in ['class', 'id', 'name', 'style']:
						del tag[attribute]			
						
				[tag.decompose() for tag in soup("script")]
				[tag.decompose() for tag in soup("style")]
				
				content = soup.find('body')
				content = self.stripTags(str(content))
				
				while '\t' in content:
					content = content.replace('\t', ' ')
					
				# Saves text from pages - note - only grabs info if page has more than 250 characters
				
				if len(content) > 250:
				
					content = self.cleanText(content)
					
					# First entry requires extension of array
					if  entry == 0:
						softData.append([])
						
						# Extract team name from links and save to softData[0]
						linkParts = linksWithSoftware[i][entry].split(':')
						teamName = linkParts[2].split('/')
						softData[len(softData)-1].append(teamName[0])
						
						# Save wiki text to softData[1]
						softData[len(softData)-1].append(content[1:])
						
						
					else:
						# Add page text to existing text for team
						softData[len(softData)-1][1] + ' ' + content
			
		return softData
	
	
	##
	#	getSoftLinks:
	#		Returns all populated software wikis links for the year as an array
	# 		Takes year as a string
	##
	
	def getSoftLinks(self, year):
		year = str(year)	
		links = []							# All possible software pages
		linksWithSoftware = []			# All populated software pages
		
		# Retrieves webpage containing list of all teams in a given year
		try:
			# Page containing team names and respective tracks for a given year
			trackSource = requests.get('http://igem.org/Team_List?year=' + year + '&name=Championship&division=igem').text
		
		except requests.ConnectionError as e:
			print(e)
			raise e
		except Exception as e:
			print(e.__class__.__name__)
			raise e

		# Retrieves page contents
		trackSoup = BeautifulSoup(trackSource, 'lxml')

		# Gets links to pages that potentially contain information on the software tool
		for tr in trackSoup.find_all('tr')[1:]:
			tds = tr.find_all('td')			
			
			# If team is software track - look at pages where software/project info may be
			if tds[5].text == "Software" or tds[5].text == "Software Tools":
				name = tr.find('a', class_ = 'team_name').text
				link1 = tds[2].find('a')['href'] + "/Software"
				link2 = tds[2].find('a')['href'] + '/Project'
				link3 = tds[2].find('a')['href'] + '/Description'
				links.append([link1, link2, link3])
				
			# If team is not software track - only look at /software page
			else:
				links.append([tds[2].find('a')["href"] + "/Software"])
				
		# Identifies pages that are not created and does not include them in linksWithSoftware
		for i in range(0, len(links), 1):
			for x in range (0, len(links[i]), 1):
				wikiSource = requests.get(links[i][x]).text
				if "There is currently no text in this page." in wikiSource or\
					"In order to be considered for the" in wikiSource or\
					"you must fill this page." in wikiSource or\
					"This page is used by the judges to evaluate your team for the" in wikiSource or\
					"Regardless of the topic, iGEM projects often create or adapt computational tools to move the project forward." in wikiSource or\
					"You can write a background of your team here. Give us a background of your team, the members, etc. Or tell us more about something of your choosing." in wikiSource or\
					"get started with your first molecular biology experiments" in wikiSource or\
					"Be descriptive but concise (1-2 paragraphs)" in wikiSource:
					pass
				else:
					linksWithSoftware.append(links[i][x])		
					
				
		# Combine links under same team into sub array 
		links.clear()
		links.append([linksWithSoftware[0]])
		for i in range(1, len(linksWithSoftware), 1):
			teamName = (linksWithSoftware[i].split(':'))[2].split('/')
			parentTeamName =(linksWithSoftware[i-1].split(':'))[2].split('/')
			
			if teamName == parentTeamName:
				links[len(links)-1].append(linksWithSoftware[i])
			else:
				links.append([linksWithSoftware[i]])
		
		return links
		
	##
	#	cleanText:
	#		Cleans wiki text by removing new lines and providing capitalization for first word in sentence
	##
	
	def cleanText(self, text):
		sentences = [sentence for sentence in nltk.tokenize.sent_tokenize(text)]
		
		normalizedSentences = self.stripNewLines(sentences)
		sentences = ''		

		# print(normalizedSentences)
		
		for sentence in normalizedSentences:
			sentences += str(' ') + str(sentence.capitalize())

		return sentences
		
	##
	#	stripNewLines:
	#		Takes an array of sentences and removes new line symbols
	# 			also looks for patterns suggesting that the sentence is part of a menu bar
	##
	
	def stripNewLines(self, sentences):
		normalizedSentences = []		
		sentences = [s.lower() for s in sentences]
		
		for s in sentences:
			s = re.sub(r'[^\x00-\x7f]',r'', s)
			if '\n' in s:
				s = s.replace('\n',' ')
				while '  ' in s:
					s = s.replace('  ', ' ')
			
			if not 'page discussion view source history teams log in' in s and "recent changes what links here related changes special pages my preferences printable version permanent link privacy policy disclaimers" not in s and ('loading menubar' not in s and (not re.search(r'^team:\w+', s))):
				normalizedSentences.append(s)
		
		return normalizedSentences

p = Parser()
for year in range(2010, 2019):
	print(p.getData(year))
