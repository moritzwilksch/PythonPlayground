# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from scrapy.exporters import JsonItemExporter
import re


class ScrapytestPipeline:

    def open_spider(self, spider):
        self.file = open("exportdata.json", 'w+b')
        self.exporter = JsonItemExporter(self.file)
        self.exporter.start_exporting()

    def close_spider(self, spider):
        self.exporter.finish_exporting()
        self.file.close()

    def process_item(self, item, spider):
        # dirty_preis = item['preis']
        # match = re.match(r'\d+\.\d+', dirty_preis)
        # item['preis'] = match.group().replace(".", "") if match else "0"


        self.exporter.export_item(item)
        return item