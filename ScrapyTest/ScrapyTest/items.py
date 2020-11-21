# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy
from itemloaders.processors import Compose, Identity, Join, MapCompose, TakeFirst
import re
from urllib.parse import unquote

m2_pattern = re.compile("\d+,?\d* m\\u00b2")
rooms_pattern = re.compile("(\d+)(?= Zimmer)")

def decode_special_chars(text: str) -> str:
    return unquote(text)

def extract_size(endtext):
    endtext = " ".join(endtext)
    match = m2_pattern.match(endtext)
    return match.group().replace(" m\u00b2", "").replace(",", ".") if match else None

def extract_rooms(endtext):
    endtext = " ".join(endtext)
    match = rooms_pattern.findall(endtext)
    return match[0] if match else None

class Listing(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    heading = scrapy.Field(input_processor=MapCompose(lambda s: s.strip(), decode_special_chars))
    link = scrapy.Field(output_processor=TakeFirst())
    descr = scrapy.Field(input_processor=MapCompose(lambda s: s.strip(), decode_special_chars))
    size = scrapy.Field(input_processor=Compose(extract_size, decode_special_chars))
    rooms = scrapy.Field(input_processor=Compose(extract_rooms, decode_special_chars))
    preis = scrapy.Field(input_processor=MapCompose(lambda s: s.replace("€", "").replace(".", "").replace("VB", "").strip().replace(" ", ""), decode_special_chars))
    ausstattung = scrapy.Field(input_processor=Compose(Join("; "), decode_special_chars))
