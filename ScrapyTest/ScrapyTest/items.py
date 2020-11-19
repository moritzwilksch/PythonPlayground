# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class Listing(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    heading = scrapy.Field()
    link = scrapy.Field()
    descr = scrapy.Field()
    endtxt = scrapy.Field()
    preis = scrapy.Field()
