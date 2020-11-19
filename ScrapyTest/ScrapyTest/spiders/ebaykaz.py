from typing import List
import scrapy
from ScrapyTest.items import Listing

class Ebaykaz(scrapy.Spider):
    name = "ebaykaz"
    start_urls =  [
        "https://www.ebay-kleinanzeigen.de/s-wohnung-kaufen/wohnung/k0c196"
    ]

    
    def parse(self, response):
        for div in response.css("li.ad-listitem").css("article"):
            yield Listing(
                {
                'heading': div.css("div.aditem-main").css("h2").css("a::text").get(),
                'link': div.css("div.aditem-main").css("h2").css("a::attr(href)").get(),
                'descr': div.css("div.aditem-main").css("p::text").get(),
                'endtxt': div.css("div.aditem-main").css("p.text-module-end").css("span::text").getall(),
                'preis': div.css("div.aditem-details").css("strong::text").get()
            }
            )
        

        next_page = response.css("a.pagination-next::attr(href)").get()
        if next_page:
            yield scrapy.Request("https://www.ebay-kleinanzeigen.de/" + next_page)
