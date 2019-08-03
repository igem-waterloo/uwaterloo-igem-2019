import requests
import urllib.request
import time
from bs4 import BeautifulSoup
print("Starting script.")
visited = set()

def get_soup(team_page_link):
	global visited
	print(team_page_link)
	# print(visited)
	if team_page_link is not None and "igem.org/Team" in team_page_link and team_page_link not in visited:
		visited.add(team_page_link)
		# print(team_page_link)
		team_page_soup = BeautifulSoup(requests.get(team_page_link).text, "html.parser")
		# print(team_page_soup)
		page_links = team_page_soup.findAll('a')
		# print("pppp" + team_page_link, page_links)
		for i in page_links:
			link = i.get("href")
			# print("link", link)
			get_soup(link)

for year in range(2004, 2019):
	url = "https://igem.org/Team_Wikis?year=" + str(year)
	response = requests.get(url)
	soup = BeautifulSoup(response.text, "html.parser")
	cur_year_links = soup.findAll('a')
	for i in cur_year_links:
		team_page_link = i.get("href")
		get_soup(team_page_link)
print("Finished script.")
