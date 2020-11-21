from scrapy import Spider
from itemloaders import ItemLoader
from AtlasISScrapy import items
import re
from itemloaders.processors import TakeFirst


class AtlasScraper(Spider):
    regex_name = re.compile(r'name.*?(?=,\"key\")')
    regex_key = re.compile(r'(?:\"key\":)(.*?)\}')
    regex_demographics = re.compile(r'\{.*\}')

    name = "AtlasSpider"
    start_urls = [
        "https://atlas.immobilienscout24.de/orte/deutschland/brandenburg/potsdam?searchQuery=Potsdam&marketingFocus=APARTMENT_BUY&userIntent=SELL#/"
    ]

    def parse(self, response):
        #from scrapy.shell import inspect_response
        #inspect_response(response, self)
        for script in response.css("script[type='text/javascript']").getall():
            if "subHierarchyInfo" in script:
                stadtteile = [row.split(":")[1].replace('"', '') for row in self.regex_name.findall(script)]
                stadtteile_links = [row.replace('"', '') for row in self.regex_key.findall(script)]

                print(stadtteile)
                print(stadtteile_links)

        for stadtteil, link in zip(stadtteile, stadtteile_links):        
            yield response.follow("https://atlas.immobilienscout24.de" + link, callback=self.parse_stadtteil_page, meta={'stadtteil': stadtteil})


    def parse_stadtteil_page(self, response):
        for script in response.css("script[type='text/javascript']").getall():
            if "subHierarchyInfo" in script:
                street_names = [row.split(":")[1].replace('"', '') for row in self.regex_name.findall(script)]
                street_links = [row.replace('"', '') for row in self.regex_key.findall(script)]

                print(street_names)
                print(street_links)
                break

        for street, link in zip(street_names, street_links):        
            yield response.follow("https://atlas.immobilienscout24.de" + link, callback=self.parse_street_page, meta={'stadtteil': response.meta['stadtteil'], 'streetname': street, 'link': link})


    def parse_street_page(self, response):
        meta = response.meta
        loader = ItemLoader(items.Street(), response)
        loader.default_output_processor = TakeFirst()
        loader.add_value('stadtteil', meta['stadtteil'])
        loader.add_value('name', meta['streetname'])
        loader.add_value('link', meta['link'])

        for script in response.css("script[type='text/javascript']").getall():
            if "demographicInfo" in script:
                loader.add_value('demographics', self.regex_demographics.search(script).group())
                break




        yield loader.load_item()
