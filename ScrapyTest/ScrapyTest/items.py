# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy
from itemloaders.processors import Compose, Identity, Join, MapCompose, TakeFirst
import re

m2_pattern = re.compile("\d+,?\d* m\\u00b2")
rooms_pattern = re.compile("(\d+)(?= Zimmer)")

def extract_size(endtext):
    endtext = " ".join(endtext)
    match = m2_pattern.match(endtext)
    return match.group() if match else endtext

def extract_rooms(endtext):
    endtext = " ".join(endtext)
    match = rooms_pattern.findall(endtext)
    return match[0] if match else endtext

class Listing(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    heading = scrapy.Field(input_processor=MapCompose(lambda s: s.strip()))
    link = scrapy.Field(output_processor=TakeFirst())
    descr = scrapy.Field(input_processor=MapCompose(lambda s: s.strip()))
    size = scrapy.Field(input_processor=Compose(extract_size))
    rooms = scrapy.Field(input_processor=Compose(extract_rooms))
    preis = scrapy.Field(input_processor=MapCompose(lambda s: s.replace("€", "").replace(".", "").replace("VB", "").strip().replace(" ", "")))
    ausstattung = scrapy.Field(input_processor=Join("###"))
