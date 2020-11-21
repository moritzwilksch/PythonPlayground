# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from itemloaders.processors import Compose, TakeFirst
import scrapy
import json

def jsonify(text: str) -> str:
    return json.loads(text)



class Street(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    name = scrapy.Field()
    stadtteil = scrapy.Field()
    alter = scrapy.Field()
    link = scrapy.Field()
    demographics = scrapy.Field(input_processor=Compose(TakeFirst(), jsonify))
