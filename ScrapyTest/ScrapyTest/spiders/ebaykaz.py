from json import load
from typing import List
import scrapy
from scrapy.loader import ItemLoader
from itemloaders.processors import TakeFirst
from ScrapyTest.items import Listing

proxy_server = "http://80.241.222.138:80"

class Ebaykaz(scrapy.Spider):
    name = "ebaykaz"
    start_urls =  [
        "https://www.ebay-kleinanzeigen.de/s-wohnung-kaufen/wohnung/k0c196"
    ]

    
    def parse(self, response):
        print(f"PARSING.... {response.request}")
        for div in response.css("li.ad-listitem").css("article"):
            loader = ItemLoader(Listing(), selector=div)
            loader.default_output_processor = TakeFirst()
            loader.add_css('heading', "div.aditem-main h2 a::text")
            loader.add_css('link', "div.aditem-main h2 a::attr(href)")
            loader.add_css('descr', "div.aditem-main p::text")
            loader.add_css('size', "div.aditem-main p span::text")
            loader.add_css('rooms', "div.aditem-main p span::text")
            loader.add_css('preis', "div.aditem-details strong::text")
            # top_level_item = loader.load_item()

            item_link = response.css("div.aditem-main h2 a::attr(href)").get()
            yield response.follow("https://www.ebay-kleinanzeigen.de" + item_link, callback=self.parse_detail_page, meta={'loader': loader, 'proxy': proxy_server})

        """yield Listing(
            {
            'heading': div.css("div.aditem-main h2 a::text").get(),
            'link': div.css("div.aditem-main h2 a::attr(href)").get(),
            'descr': div.css("div.aditem-main p::text").get(),
            'endtxt': div.css("div.aditem-main p.text-module-end span::text").getall(),
            'preis': div.css("div.aditem-details strong::text").get()
        }
        )"""
        

        next_page = response.css("a.pagination-next::attr(href)").get()
        if next_page:
            yield response.follow("https://www.ebay-kleinanzeigen.de" + next_page, callback=self.parse, meta={'proxy': proxy_server})
        else:
            #from scrapy.shell import inspect_response
            #inspect_response(response, self)
            print("NO NEXTPAGE")


    def parse_detail_page(self, response):
        # item = response.meta['item']
        # loader = ItemLoader(item, response=response)
        loader = response.meta['loader']
        # loader.default_output_processor = TakeFirst()

        loader.add_css("ausstattung", "li.checktag::text")
        yield loader.load_item()
