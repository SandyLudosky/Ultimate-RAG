import httplib2
from bs4 import BeautifulSoup, SoupStrainer

http = httplib2.Http()

def get_links(url):
    status, response = http.request(url)
    links = []
    for link in BeautifulSoup(response, parse_only=SoupStrainer('a')):
        if link.has_attr('href'):
            links.append(link.attrs['href'])
    return links